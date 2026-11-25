"""Гипотеза Д и гипотеза 8 на сохранённой модели следующего похода.

Пороги Д выбираются только на validation. Сетка 0,01…0,99 с шагом 0,01.
При равном L берётся больший порог. Тест эту сетку не двигает.
Гипотеза 8 модель не переучивает и порог не подбирает.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

import forecast

HERE = Path(__file__).resolve().parent
DB = HERE / "fridge_lab.sqlite"
MODEL = HERE / "models" / "catboost_next_trip_v1.cbm"
PARTS = (
    ("val", "2026-06-01", "2026-08-01"),
    ("test", "2026-08-01", "2026-10-01"),
)
THRESHOLDS = np.round(np.arange(0.01, 1.0, 0.01), 2)
RATIOS = (
    ("1:1", 1.0, 1.0, False),
    ("2:1", 2.0, 1.0, False),
    ("3:1", 3.0, 1.0, False),
    ("1:2 скоропортящиеся", 1.0, 2.0, True),
)
ROUNDS = 2000
SEED = 0


def load_part(data, start, end):
    """Вернуть строки прогноза с известным следующим походом.

    Горизонт метки в ``forecast.build_rows`` фиксирован: 14 суток.
    Индекс сбрасывается.

    Args:
        data (dict): Результат ``forecast.load``.
        start: Начало окна походов.
        end: Конец окна походов.

    Returns:
        DataFrame строк, где ``has_next_visit`` истинно.
    """
    frame, _stats = forecast.build_rows(data, start, end, 14)
    frame = frame[frame["has_next_visit"]].reset_index(drop=True)
    return frame


def probabilities(model, frame):
    """Вернуть вероятность класса 1 по сохранённой модели.

    Модель не обучается. Признаки — ``forecast.FEATURES``,
    категориальные — ``forecast.CATEGORICAL_FEATURES``.

    Args:
        model (CatBoostClassifier): Загруженная модель.
        frame (pandas.DataFrame): Строки со столбцами
            ``forecast.FEATURES``.

    Returns:
        ndarray вероятностей ``predict_proba`` для класса 1.
    """
    pool = Pool(frame[forecast.FEATURES], cat_features=forecast.CATEGORICAL_FEATURES)
    return model.predict_proba(pool)[:, 1]


def top_mask(frame, score):
    """Вернуть маску десятки с наибольшим score в каждом походе.

    Сортировка внутри похода: ``score`` убывая, при равенстве
    ``base_product`` возрастая. В десятку входят строки с рангом
    строго меньше 10. Если в походе меньше 10 строк, все они
    True.

    Args:
        frame (pandas.DataFrame): Строки с ``visit_id`` и
            ``base_product``. Индекс сохраняется в результате.
        score: Баллы той же длины, что ``frame``, в том же порядке.

    Returns:
        ndarray bool длины ``len(frame)`` в порядке индекса
        ``frame``.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> frame = pd.DataFrame({
        ...     "visit_id": [1, 1],
        ...     "base_product": [2, 1],
        ... })
        >>> top_mask(frame, np.array([0.1, 0.9]))
        array([ True,  True])
    """
    order = frame.assign(_score=score).sort_values(
        ["visit_id", "_score", "base_product"], ascending=[True, False, True]
    )
    rank = order.groupby("visit_id", sort=False).cumcount()
    chosen = pd.Series(False, index=frame.index)
    chosen.loc[order.index[rank.to_numpy() < 10]] = True
    return chosen.to_numpy()


def counts(frame, selected):
    """Посчитать пропуски, лишние и среднюю длину списка.

    Пропуск — истинная метка и не выбран. Лишний — ложная метка
    и выбран. Длина списка — сумма выбранных по ``visit_id``.

    Args:
        frame (pandas.DataFrame): Строки с ``in_next_basket`` и
            ``visit_id``.
        selected: Булева маска той же длины, что ``frame``.

    Returns:
        Кортеж ``(missed, extra, total, mean_listed)``:
        число пропусков, число лишних, ``len(frame)``, среднее
        число выбранных на поход.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> frame = pd.DataFrame({
        ...     "visit_id": [1, 1],
        ...     "in_next_basket": [True, False],
        ... })
        >>> counts(frame, np.array([False, True]))
        (1, 1, 2, 1.0)
    """
    label = frame["in_next_basket"].to_numpy().astype(bool)
    selected = np.asarray(selected, dtype=bool)
    missed = int((label & ~selected).sum())
    extra = int((~label & selected).sum())
    listed = frame.assign(_selected=selected).groupby("visit_id")["_selected"].sum()
    return missed, extra, len(frame), float(listed.mean())


def loss(missed, extra, total, cost_miss, cost_extra):
    """Вернуть среднюю цену ошибки на строку.

    Считает ``(cost_miss * missed + cost_extra * extra) / total``.

    Args:
        missed: Число пропусков.
        extra: Число лишних.
        total: Делитель, обычно число строк.
        cost_miss: Цена пропуска.
        cost_extra: Цена лишнего.

    Returns:
        Средняя цена на одну единицу ``total``.

    Raises:
        ZeroDivisionError: Если ``total`` равен 0.

    Examples:
        >>> loss(1, 2, 10, 3.0, 1.0)
        0.5
    """
    return (cost_miss * missed + cost_extra * extra) / total


def choose_threshold(frame, score, cost_miss, cost_extra):
    """Выбрать порог с наименьшим L на переданном кадре.

    Перебирает ``THRESHOLDS``. При равном L берётся больший
    порог. Тестовый кадр сюда не передаётся.

    Args:
        frame (pandas.DataFrame): Строки validation для
            ``counts``.
        score: Баллы той же длины, что ``frame``. Порог режет
            как ``score >= threshold``.
        cost_miss: Цена пропуска для ``loss``.
        cost_extra: Цена лишнего для ``loss``.

    Returns:
        Кортеж ``(threshold, value, size)``: выбранный порог,
        его L и средняя длина списка на этом пороге.
    """
    best = None
    for threshold in THRESHOLDS:
        missed, extra, total, size = counts(frame, score >= threshold)
        value = loss(missed, extra, total, cost_miss, cost_extra)
        key = (value, -float(threshold))
        if best is None or key < best[0]:
            best = (key, float(threshold), value, size)
    return best[1], best[2], best[3]


def recall_at_10(frame, score):
    """Вернуть средний Recall@10 по походам с хотя бы одной покупкой.

    Внутри похода ранг 0 — наибольший ``score``, при равенстве
    меньший ``base_product``. Recall похода — доля купленных
    позиций с рангом строго меньше 10. Походы без истинной
    метки в среднее не входят.

    Args:
        frame (pandas.DataFrame): Строки с ``visit_id``,
            ``base_product``, ``in_next_basket``.
        score: Баллы той же длины, что ``frame``, в том же порядке.

    Returns:
        Кортеж ``(recall, n)``: среднее Recall@10 по таким
        походам и их число. Если купленных строк нет —
        ``(nan, 0)``.
    """
    order = frame.assign(_score=score, _label=frame["in_next_basket"].astype(bool)).sort_values(
        ["visit_id", "_score", "base_product"], ascending=[True, False, True]
    )
    order["_rank"] = order.groupby("visit_id", sort=False).cumcount()
    bought = order[order["_label"]]
    per_trip = bought.groupby("visit_id")["_rank"].apply(lambda ranks: float(np.mean(ranks.to_numpy() < 10)))
    return float(per_trip.mean()), int(len(per_trip))


def household_changes(frame, score, threshold, cost_miss, cost_extra):
    """Оценить 95% интервал относительного снижения L порога.

    База раунда — минимум L десятки и L порога 0,5. Изменение —
    ``(base - cost) / base``. Семьи тянутся с возвращением
    ``ROUNDS`` раз, генератор ``SEED``. Границы — элементы
    отсортированного массива с индексами
    ``int(0.025 * (ROUNDS - 1))`` и ``int(0.975 * (ROUNDS - 1))``.

    Args:
        frame (pandas.DataFrame): Строки с ``hh``,
            ``in_next_basket``, ``visit_id``, ``base_product``.
        score: Баллы той же длины, что ``frame``.
        threshold: Порог варианта «cost».
        cost_miss: Цена пропуска.
        cost_extra: Цена лишнего.

    Returns:
        Кортеж ``(low, high)`` — границы интервала как float.

    Notes:
        В каждом раунде результат делится на ``base``.
    """
    label = frame["in_next_basket"].to_numpy().astype(bool)
    masks = {
        "top": top_mask(frame, score),
        "half": score >= 0.5,
        "cost": score >= threshold,
    }
    table = pd.DataFrame({"hh": frame["hh"].to_numpy()})
    for name, selected in masks.items():
        table[f"{name}_miss"] = label & ~selected
        table[f"{name}_extra"] = ~label & selected
    table["n"] = 1
    grouped = table.groupby("hh").sum(numeric_only=True)
    households = grouped.index.to_numpy()
    rng = np.random.default_rng(SEED)
    changes = []
    for _round in range(ROUNDS):
        draw = rng.choice(len(households), size=len(households), replace=True)
        part = grouped.iloc[draw]
        base = min(
            loss(part["top_miss"].sum(), part["top_extra"].sum(), part["n"].sum(), cost_miss, cost_extra),
            loss(part["half_miss"].sum(), part["half_extra"].sum(), part["n"].sum(), cost_miss, cost_extra),
        )
        cost = loss(part["cost_miss"].sum(), part["cost_extra"].sum(), part["n"].sum(), cost_miss, cost_extra)
        changes.append((base - cost) / base)
    changes = np.sort(np.asarray(changes))
    low = changes[int(0.025 * (ROUNDS - 1))]
    high = changes[int(0.975 * (ROUNDS - 1))]
    return float(low), float(high)


def main():
    """Прогнать гипотезу Д и гипотезу 8 на сохранённой модели.

    Пороги Д выбираются только на validation по сетке
    ``THRESHOLDS``. Тест сетку не двигает. Гипотеза 8 модель не
    переучивает и порог не подбирает. База — ``DB``, модель —
    ``MODEL``.

    Notes:
        Читает SQLite и файл ``.cbm``, пишет вердикты в stdout.
    """
    data = forecast.load(DB)
    model = CatBoostClassifier()
    model.load_model(MODEL)
    parts = {}
    for name, start, end in PARTS:
        frame = load_part(data, start, end)
        frame["p"] = probabilities(model, frame)
        parts[name] = frame
        print(
            f"{name}: походов {frame['visit_id'].nunique()}, строк {len(frame)}",
            flush=True,
        )
    test = parts["test"]
    recall, trips = recall_at_10(test, test["p"].to_numpy())
    print(f"проверка модели, Recall@10 на тесте: {recall:.3f}, походов с покупкой {trips}", flush=True)

    print("гипотеза Д")
    passed = []
    worsened = []
    for name, cost_miss, cost_extra, perishable in RATIOS:
        val = parts["val"]
        test_part = parts["test"]
        if perishable:
            val = val[val["is_perishable"] >= 0.5]
            test_part = test_part[test_part["is_perishable"] >= 0.5]
        threshold, _val_loss, _val_size = choose_threshold(
            val, val["p"].to_numpy(), cost_miss, cost_extra
        )
        score = test_part["p"].to_numpy()
        top_miss, top_extra, total, top_size = counts(test_part, top_mask(test_part, score))
        half_miss, half_extra, _total, half_size = counts(test_part, score >= 0.5)
        cost_missed, cost_extra_n, _total, cost_size = counts(test_part, score >= threshold)
        top_loss = loss(top_miss, top_extra, total, cost_miss, cost_extra)
        half_loss = loss(half_miss, half_extra, total, cost_miss, cost_extra)
        cost_loss = loss(cost_missed, cost_extra_n, total, cost_miss, cost_extra)
        base = min(top_loss, half_loss)
        change = (base - cost_loss) / base
        size_ok = cost_size <= 1.5 * top_size
        ok = change >= 0.10 and size_ok
        worse = change < -0.05
        low, high = household_changes(test_part, score, threshold, cost_miss, cost_extra)
        passed.append(ok)
        worsened.append(worse)
        print(
            f"{name}: порог {threshold:.2f}, L десятка {top_loss:.4f}, L 0,5 {half_loss:.4f}, "
            f"L порога {cost_loss:.4f}, снижение {change:.3f}, "
            f"список {cost_size:.2f} против десятки {top_size:.2f}, "
            f"95% снижения [{low:.3f}; {high:.3f}]"
        )
    confirmed = any(passed) and not any(worsened)
    print("гипотеза Д подтверждена" if confirmed else "гипотеза Д не подтверждена")

    print("гипотеза 8")
    perishable = test[test["is_perishable"] >= 0.5]
    to_plan = -perishable["app_supply_vs_plan"].fillna(365).to_numpy()
    rule = np.where(perishable["pair_n"].to_numpy() >= 3, to_plan, to_plan - 10_000)
    rule_recall, rule_trips = recall_at_10(perishable, rule)
    model_recall, model_trips = recall_at_10(perishable, perishable["p"].to_numpy())
    gain = model_recall - rule_recall
    low, high = recall_change_interval(perishable, rule, perishable["p"].to_numpy())
    print(
        f"скоропортящиеся, походов {model_trips}: правило {rule_recall:.3f}, "
        f"CatBoost {model_recall:.3f}, прирост {gain:.3f}, 95% [{low:.3f}; {high:.3f}]"
    )
    print("гипотеза 8 подтверждена" if gain >= 0.10 else "гипотеза 8 не подтверждена")


def _trip_recall(frame, score):
    """Вернуть Recall@10 каждого похода с хотя бы одной покупкой.

    Ранг внутри похода считается как в ``recall_at_10``.

    Args:
        frame (pandas.DataFrame): Строки с ``hh``, ``visit_id``,
            ``base_product``, ``in_next_basket``.
        score: Баллы той же длины, что ``frame``.

    Returns:
        Series float с MultiIndex ``(hh, visit_id)``. Походы без
        истинной метки отсутствуют.
    """
    order = frame.assign(_score=score, _label=frame["in_next_basket"].astype(bool)).sort_values(
        ["visit_id", "_score", "base_product"], ascending=[True, False, True]
    )
    order["_rank"] = order.groupby("visit_id", sort=False).cumcount()
    bought = order[order["_label"]]
    return bought.groupby(["hh", "visit_id"], sort=False)["_rank"].apply(
        lambda ranks: float(np.mean(ranks.to_numpy() < 10))
    )


def recall_change_interval(frame, rule, model):
    """Оценить 95% интервал прироста Recall@10 модели над правилом.

    Единица бутстрепа — семья. ``ROUNDS`` выборок с возвращением,
    генератор ``SEED``. В раунде прирост — среднее ``model`` минус
    среднее ``rule`` по повторно взятым походам. Границы — как в
    ``household_changes``.

    Args:
        frame (pandas.DataFrame): Строки с ``hh``, ``visit_id``,
            ``base_product``, ``in_next_basket``.
        rule: Баллы правила той же длины, что ``frame``.
        model: Баллы модели той же длины, что ``frame``.

    Returns:
        Кортеж ``(low, high)`` — границы интервала как float.
    """
    table = pd.concat(
        [_trip_recall(frame, rule).rename("rule"), _trip_recall(frame, model).rename("model")],
        axis=1,
    ).reset_index()
    grouped = list(table.groupby("hh"))
    grouped = list(table.groupby("hh"))
    rng = np.random.default_rng(SEED)
    changes = []
    for _round in range(ROUNDS):
        draw = rng.choice(len(grouped), size=len(grouped), replace=True)
        parts = [grouped[index][1] for index in draw]
        sample = pd.concat(parts, ignore_index=True)
        changes.append(float(sample["model"].mean() - sample["rule"].mean()))
    changes = np.sort(np.asarray(changes))
    return float(changes[int(0.025 * (ROUNDS - 1))]), float(changes[int(0.975 * (ROUNDS - 1))])


if __name__ == "__main__":
    main()
