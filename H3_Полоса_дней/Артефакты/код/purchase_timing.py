"""Что из «холодильника» покупатель купит снова и когда.

Момент прогноза — каждый визит покупателя. Кандидаты — товары, купленные им
за последние WINDOW_DAYS дней, включая этот визит. Для каждой пары известно,
через сколько дней была следующая покупка (если была до конца периода).

Признаки подобраны так, чтобы у каждого был аналог в базе AI-Fridge:
покупки и их количество, категория товара, частота походов в магазин.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

PAIR = ["CUSTOMER_ID", "PRODUCT_ID"]
WINDOW_DAYS = 60
_DAY_KEY = 100_000

CATEGORICAL_FEATURES = ["PRODUCT_SUBCLASS"]
NUMERIC_FEATURES = [
    "pair_n",
    "pair_days_since_last",
    "pair_bought_now",
    "pair_last_gap",
    "pair_mean_gap",
    "pair_recency_ratio",
    "pair_last_amount",
    "pair_mean_amount",
    "pair_days_per_unit",
    "pair_left_ratio",
    "pair_visit_share",
    "cust_n_visits",
    "cust_mean_visit_gap",
    "cust_days_since_first",
    "cust_basket",
    "cust_mean_basket",
    "cust_fridge_size",
    "prod_n_buyers",
    "prod_repeat_share",
    "prod_median_gap",
    "prod_mean_amount",
    "prod_unit_price",
    "prod_gap_ratio",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def _day_number(days):
    return ((days - pd.Timestamp("2000-01-01")) // pd.Timedelta(days=1)).to_numpy()


def product_stats(events, before):
    """Статистика товара только по покупкам до даты before."""
    hist = events[events["day"] < pd.Timestamp(before)]
    gaps = (hist["day"] - hist["prev_day"]).dt.days
    pair_counts = hist.groupby(PAIR).size()
    by_product = pair_counts.groupby(level="PRODUCT_ID")
    return pd.DataFrame({
        "prod_n_buyers": by_product.size(),
        "prod_repeat_share": (pair_counts > 1).groupby(level="PRODUCT_ID").mean(),
        "prod_median_gap": gaps.groupby(hist["PRODUCT_ID"]).median(),
        "prod_mean_amount": hist.groupby("PRODUCT_ID")["amount"].mean(),
        "prod_unit_price": hist.groupby("PRODUCT_ID")["unit_price"].median(),
    })


def monthly_product_stats(events):
    """product_stats на начало каждого месяца: строка видит только прошлые месяцы."""
    months = pd.date_range(
        events["day"].min().normalize() + pd.offsets.MonthBegin(1),
        events["day"].max() + pd.offsets.MonthBegin(1),
        freq="MS",
    )
    return pd.concat(
        {month: product_stats(events, month) for month in months},
        names=["month", "PRODUCT_ID"],
    )


def visit_table(visits):
    """Визиты с признаками покупателя на момент визита (визит включён)."""
    table = visits.sort_values(["CUSTOMER_ID", "day"]).reset_index(drop=True)
    by_customer = table.groupby("CUSTOMER_ID")
    table["cust_n_visits"] = by_customer.cumcount() + 1
    first = by_customer["day"].transform("min")
    table["cust_days_since_first"] = (table["day"] - first).dt.days
    table["cust_mean_visit_gap"] = (
        table["cust_days_since_first"] / (table["cust_n_visits"] - 1)
    ).where(table["cust_n_visits"] > 1)
    table["cust_basket"] = table["basket"]
    table["cust_mean_basket"] = by_customer["basket"].cumsum() / table["cust_n_visits"]
    table["next_visit"] = by_customer["day"].shift(-1)
    table["next_basket"] = by_customer["basket"].shift(-1)
    return table


def build_rows(events, visit_frame, products, start, end, horizon_days):
    """Строки «визит × товар из холодильника» для визитов в [start, end).

    Остаются визиты, у которых окно метки (визит, визит + horizon_days]
    целиком лежит до end: метка и дата следующей покупки не заглядывают
    за границу периода. products — результат monthly_product_stats.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    last_visit = end - pd.Timedelta(days=horizon_days + 1)
    snap = visit_frame[(visit_frame["day"] >= start) & (visit_frame["day"] <= last_visit)]
    snap = snap.reset_index(drop=True)

    ev = events.loc[events["day"] < end, PAIR + [
        "PRODUCT_SUBCLASS", "day", "prev_day", "amount", "pair_n",
        "pair_amount_cum", "pair_first_day",
    ]]
    ev_day = _day_number(ev["day"])
    # events отсортированы по паре и дате: следующая строка той же пары — следующая покупка
    same_pair = ev[PAIR].eq(ev[PAIR].shift(-1)).all(axis=1)
    next_day = _day_number(ev["day"].shift(-1).where(same_pair)).astype("float64")
    next_day = np.nan_to_num(next_day, nan=np.inf)
    # Состояние пары после покупки действует до следующей покупки,
    # но не дольше окна холодильника.
    cover_end = np.minimum(next_day, ev_day + WINDOW_DAYS)

    customer = ev["CUSTOMER_ID"].to_numpy(dtype="int64")
    visit_key = snap["CUSTOMER_ID"].to_numpy(dtype="int64") * _DAY_KEY + _day_number(snap["day"])
    lo = np.searchsorted(visit_key, customer * _DAY_KEY + ev_day, side="left")
    hi = np.searchsorted(visit_key, customer * _DAY_KEY + cover_end.astype("int64"), side="left")
    counts = hi - lo

    ev_idx = np.repeat(np.arange(len(ev)), counts)
    offsets = np.repeat(np.cumsum(counts) - counts, counts)
    v_idx = np.repeat(lo, counts) + (np.arange(len(ev_idx)) - offsets)

    e = ev.iloc[ev_idx].reset_index(drop=True)
    v = snap.iloc[v_idx].reset_index(drop=True)
    visit_day = _day_number(v["day"])
    last_day = ev_day[ev_idx]
    first_day = _day_number(e["pair_first_day"])
    pair_n = e["pair_n"].to_numpy(dtype="float32")
    amount_cum = e["pair_amount_cum"].to_numpy(dtype="float32")
    last_amount = e["amount"].to_numpy(dtype="float32")

    frame = pd.DataFrame({
        "visit_id": v_idx,
        "day": v["day"],
        "CUSTOMER_ID": e["CUSTOMER_ID"],
        "PRODUCT_ID": e["PRODUCT_ID"],
        "PRODUCT_SUBCLASS": e["PRODUCT_SUBCLASS"],
    })
    frame["pair_n"] = pair_n
    frame["pair_days_since_last"] = visit_day - last_day
    frame["pair_bought_now"] = (visit_day == last_day).astype("float32")
    frame["pair_last_gap"] = (e["day"] - e["prev_day"]).dt.days
    span = (last_day - first_day).astype("float32")
    with np.errstate(divide="ignore", invalid="ignore"):
        frame["pair_mean_gap"] = np.where(pair_n > 1, span / (pair_n - 1), np.nan)
        frame["pair_recency_ratio"] = frame["pair_days_since_last"] / frame["pair_mean_gap"]
        # Сколько дней уходит на единицу товара: прошлые покупки, растянутые до последней
        frame["pair_days_per_unit"] = np.where(
            pair_n > 1, span / (amount_cum - last_amount), np.nan
        )
        frame["pair_left_ratio"] = (1.0 - frame["pair_days_since_last"] / (
            last_amount * frame["pair_days_per_unit"]
        )).clip(lower=-5.0)
    frame["pair_last_amount"] = last_amount
    frame["pair_mean_amount"] = amount_cum / pair_n

    for column in ["cust_n_visits", "cust_mean_visit_gap", "cust_days_since_first",
                   "cust_basket", "cust_mean_basket"]:
        frame[column] = v[column].to_numpy()
    frame["pair_visit_share"] = frame["pair_n"] / frame["cust_n_visits"]
    frame["cust_fridge_size"] = np.bincount(v_idx, minlength=len(snap))[v_idx]

    frame["month"] = frame["day"].dt.to_period("M").dt.to_timestamp()
    frame = frame.join(products, on=["month", "PRODUCT_ID"])
    frame["prod_gap_ratio"] = frame["pair_days_since_last"] / frame["prod_median_gap"]

    days_to_next = next_day[ev_idx] - visit_day
    frame["label"] = (days_to_next <= horizon_days).astype("int8")
    frame["days_to_purchase"] = np.where(frame["label"] == 1, days_to_next, np.nan)
    next_visit = _day_number(v["next_visit"])
    frame["has_next_visit"] = v["next_visit"].lt(end).to_numpy()
    frame["in_next_basket"] = frame["has_next_visit"] & (next_day[ev_idx] == next_visit)
    # Аналог plan_date в AI-Fridge: дата следующего похода известна заранее
    frame["days_to_next_visit"] = np.where(
        frame["has_next_visit"], next_visit - visit_day, np.nan
    )

    frame[NUMERIC_FEATURES] = frame[NUMERIC_FEATURES].replace([np.inf, -np.inf], np.nan)
    frame[NUMERIC_FEATURES] = frame[NUMERIC_FEATURES].astype("float32")

    future = snap["next_visit"].lt(end)
    stats = {
        "Визитов": len(snap),
        "Строк": len(frame),
        "Товаров в холодильнике на визит": len(frame) / max(len(snap), 1),
        "Доля строк с покупкой в окне": float(frame["label"].mean()),
        "Покрытие следующей корзины": float(
            frame["in_next_basket"].sum() / snap.loc[future, "next_basket"].sum()
        ),
    }
    return frame, stats


def next_basket_metrics(frame, scores, k=10):
    """Попадание в следующую корзину: топ-k товаров холодильника по score.

    Учитываются визиты, у которых следующий визит есть и в нём куплен
    хотя бы один товар из холодильника.
    """
    mask = frame["has_next_visit"].to_numpy()
    ranked = frame.loc[mask, ["visit_id", "in_next_basket"]].copy()
    ranked["score"] = np.asarray(scores, dtype="float64")[mask]
    ranked = ranked[ranked.groupby("visit_id")["in_next_basket"].transform("max")]
    ranked = ranked.sort_values(["visit_id", "score"], ascending=[True, False], kind="stable")
    top = ranked[ranked.groupby("visit_id").cumcount() < k]
    hits = top.groupby("visit_id")["in_next_basket"].sum()
    positives = ranked.groupby("visit_id")["in_next_basket"].sum()
    return {
        f"Recall@{k}": float((hits / positives).mean()),
        f"HitRate@{k}": float((hits > 0).mean()),
    }


def evaluate(frame, scores, k=10):
    """Качество «купит в окне» и попадание в следующую корзину."""
    scores = np.asarray(scores, dtype="float64")
    labels = frame["label"].to_numpy()
    return {
        "PR-AUC": average_precision_score(labels, scores),
        "ROC-AUC": roc_auc_score(labels, scores),
        **next_basket_metrics(frame, scores, k),
    }


def date_metrics(actual, predicted):
    """Ошибка дня покупки среди пар, купленных в окне."""
    error = np.abs(np.asarray(predicted, dtype="float64") - np.asarray(actual, dtype="float64"))
    return {
        "MAE, дней": float(error.mean()),
        "±1 день": float((error <= 1).mean()),
        "±3 дня": float((error <= 3).mean()),
    }


def remaining_days(frame, fallback_gap):
    """Сколько дней осталось до обычного интервала: личного, а без него — товара."""
    since = frame["pair_days_since_last"]
    product_gap = frame["prod_median_gap"].fillna(fallback_gap)
    by_product = (product_gap - since).clip(lower=1)
    personal = (frame["pair_mean_gap"] - since).clip(lower=1)
    return {
        "Интервал товара": by_product.to_numpy(),
        "Личный интервал": personal.where(frame["pair_n"] > 1, by_product).to_numpy(),
    }


def hazard_loss(logits, days, weight=None):
    """Правдоподобие дискретного времени до покупки внутри окна.

    logits[:, d] — шанс купить в день d + 1 при условии, что до этого не купил.
    days — день покупки (1..H) или 0, если в окне покупки не было.
    """
    horizon = logits.shape[1]
    log_buy = nn.functional.logsigmoid(logits)
    log_wait = nn.functional.logsigmoid(-logits)
    bought = days > 0
    waited_days = torch.where(bought, days - 1, torch.full_like(days, horizon))
    positions = torch.arange(horizon, device=logits.device).unsqueeze(0)
    loglik = ((positions < waited_days.unsqueeze(1)).float() * log_wait).sum(dim=1)
    day_index = (days - 1).clamp(min=0).unsqueeze(1)
    loglik = loglik + bought.float() * log_buy.gather(1, day_index).squeeze(1)
    if weight is None:
        return -loglik.mean()
    return -(weight * loglik).sum() / weight.sum()


def hazard_summary(logits):
    """Вероятность купить по каждый день, вероятность в окне и ожидаемый день покупки."""
    wait = torch.cumprod(torch.sigmoid(-logits), dim=1)
    bought_by = 1.0 - wait
    previous = torch.cat([torch.zeros_like(bought_by[:, :1]), bought_by[:, :-1]], dim=1)
    per_day = bought_by - previous
    days = torch.arange(1, logits.shape[1] + 1, device=logits.device, dtype=logits.dtype)
    in_window = bought_by[:, -1]
    expected_day = (per_day * days).sum(dim=1) / in_window.clamp(min=1e-6)
    return bought_by, in_window, expected_day
