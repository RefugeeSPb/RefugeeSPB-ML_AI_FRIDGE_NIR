"""Прогноз для одного домохозяйства: личная модель и подмешивание похожих.

Строки и признаки — из forecast.py (ver01). Здесь три способа обучить модель
под конкретное домохозяйство:

- личная — только его строки;
- с похожими — его строки и строки k самых похожих домохозяйств с весом w;
- поправка к общей — общая модель даёт исходный логит (baseline), личная
  модель на строках домохозяйства учится его поправлять.

Похожесть домохозяйств — косинусная близость профилей корзины: доля походов,
в которых куплен каждый базовый продукт, только по истории до заданной даты.
"""

import sqlite3

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

import forecast

LABEL = "in_next_basket"
# Внутри одного домохозяйства эти признаки постоянны
HOUSEHOLD_CONSTANT = ["hh_size", "segment"]
CATEGORICAL_FEATURES = [c for c in forecast.CATEGORICAL_FEATURES if c not in HOUSEHOLD_CONSTANT]
NUMERIC_FEATURES = [c for c in forecast.NUMERIC_FEATURES if c not in HOUSEHOLD_CONSTANT]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def load(db_path):
    """forecast.load плюс id партии у событий — нужен для обрезки истории."""
    data = forecast.load(db_path)
    with sqlite3.connect(db_path) as connection:
        events = pd.read_sql_query("""
            select id_lot, id_household as hh, id_base_product as base_product, event_type,
                   quantity_base_delta as delta, recorded_at
            from lot_event
        """, connection)
    events["recorded_at"] = pd.to_datetime(events["recorded_at"])
    data["events"] = events
    return data


def truncate_history(data, households, since):
    """Домохозяйства households будто поставили приложение в момент since.

    Удаляются их походы и покупки до since и все события партий, заведённых
    до since: остаток в приложении начинается с нуля.
    """
    households = set(households)
    trips, purchases, events = data["trips"], data["purchases"], data["events"]
    in_scope = lambda frame: frame["hh"].isin(households)
    added = events[events["event_type"] == "added"].set_index("id_lot")["recorded_at"]
    old_lot = events["id_lot"].map(added) < since
    result = dict(data)
    result["trips"] = trips[~(in_scope(trips) & (trips["actual_at"] < since))].reset_index(drop=True)
    result["purchases"] = purchases[~(in_scope(purchases) & (purchases["actual_at"] < since))].reset_index(drop=True)
    result["events"] = events[~(in_scope(events) & old_lot)].reset_index(drop=True)
    return result


def basket_profiles(data, before):
    """Доля походов домохозяйства с каждым базовым продуктом до даты before."""
    purchases = data["purchases"][data["purchases"]["actual_at"] < before]
    n_trips = data["trips"][data["trips"]["actual_at"] < before].groupby("hh").size()
    counts = purchases.groupby(["hh", "base_product"]).size().unstack(fill_value=0)
    return counts.div(n_trips.reindex(counts.index), axis=0).fillna(0.0)


def nearest_households(profiles, k):
    """Для каждого домохозяйства — k самых похожих (без него самого) и их близость."""
    values = profiles.to_numpy(dtype="float64")
    values = values / np.clip(np.linalg.norm(values, axis=1, keepdims=True), 1e-12, None)
    similarity = values @ values.T
    np.fill_diagonal(similarity, -np.inf)
    order = np.argsort(-similarity, axis=1)[:, :k]
    index = profiles.index.to_numpy()
    return {
        hh: (index[order[i]], similarity[i, order[i]])
        for i, hh in enumerate(index)
    }


def _classifier(iterations, learning_rate, depth, seed, boost_from_average=True):
    return CatBoostClassifier(
        iterations=iterations, learning_rate=learning_rate, depth=depth, l2_leaf_reg=10,
        loss_function="Logloss", random_seed=seed, thread_count=-1, verbose=0,
        boost_from_average=boost_from_average, allow_writing_files=False,
    )


def _pool(frame, features, categorical, label=True, weight=None, baseline=None):
    return Pool(frame[features], label=frame[LABEL] if label else None, cat_features=categorical,
                weight=weight, baseline=baseline)


def fit_global(train, iterations, seed, learning_rate=0.08, depth=6):
    model = _classifier(iterations, learning_rate, depth, seed)
    model.fit(_pool(train, forecast.FEATURES, forecast.CATEGORICAL_FEATURES))
    return model


def global_logit(model, frame):
    return model.predict(_pool(frame, forecast.FEATURES, forecast.CATEGORICAL_FEATURES, label=False),
                         prediction_type="RawFormulaVal")


def fit_personal(train, hh, params, seed, neighbours=(), neighbour_weight=0.0, by_hh=None):
    """Личная модель домохозяйства hh; neighbours подмешиваются с весом neighbour_weight.

    Без своих строк (новое домохозяйство) модель учится только на похожих.
    """
    rows = by_hh if by_hh is not None else train.groupby("hh").indices
    own = rows.get(hh, np.array([], dtype=np.int64))
    parts, weights = [own], [np.ones(len(own))]
    if neighbour_weight > 0:
        for other in neighbours:
            if other in rows:
                parts.append(rows[other])
                weights.append(np.full(len(rows[other]), neighbour_weight))
    frame = train.iloc[np.concatenate(parts)]
    if frame[LABEL].nunique() < 2:
        return None
    model = _classifier(params["iterations"], params["learning_rate"], params["depth"], seed)
    model.fit(_pool(frame, FEATURES, CATEGORICAL_FEATURES, weight=np.concatenate(weights)))
    return model


def fit_correction(train, hh, base_logit, params, seed, by_hh=None):
    """Личная поправка к общей модели: бустинг от её логита на строках hh."""
    rows = by_hh if by_hh is not None else train.groupby("hh").indices
    own = rows.get(hh, np.array([], dtype=np.int64))
    frame = train.iloc[own]
    if frame[LABEL].nunique() < 2:
        return None
    model = _classifier(params["iterations"], params["learning_rate"], params["depth"], seed,
                        boost_from_average=False)
    model.fit(_pool(frame, FEATURES, CATEGORICAL_FEATURES, baseline=base_logit[own]))
    return model


def personal_logit(model, frame):
    return model.predict(_pool(frame, FEATURES, CATEGORICAL_FEATURES, label=False),
                         prediction_type="RawFormulaVal")


def recall_at_k(frame, scores, k=10):
    """Recall@k следующего похода по походам frame (как pt.next_basket_metrics)."""
    ranked = frame[["visit_id", LABEL]].copy()
    ranked["score"] = np.asarray(scores, dtype="float64")
    ranked = ranked[ranked.groupby("visit_id")[LABEL].transform("max")]
    if ranked.empty:
        return np.nan
    ranked = ranked.sort_values(["visit_id", "score"], ascending=[True, False], kind="stable")
    top = ranked[ranked.groupby("visit_id").cumcount() < k]
    hits = top.groupby("visit_id")[LABEL].sum()
    return float((hits / ranked.groupby("visit_id")[LABEL].sum()).mean())


def household_metrics(frame, scores, k=10):
    """Recall@k и PR-AUC отдельно по каждому домохозяйству."""
    from sklearn.metrics import average_precision_score

    scores = np.asarray(scores, dtype="float64")
    table = {}
    for hh, index in frame.groupby("hh").indices.items():
        part = frame.iloc[index]
        labels = part[LABEL].to_numpy()
        table[hh] = {
            f"Recall@{k}": recall_at_k(part, scores[index], k),
            "PR-AUC": average_precision_score(labels, scores[index]) if labels.any() else np.nan,
        }
    return pd.DataFrame(table).T.rename_axis("hh")
