"""Основные гипотезы списка на fridge_lab_v2.sqlite.

Старая база и models/catboost_next_trip_v1.cbm не читаются и не пишутся.
Пороги взяты до прогона и здесь не двигаются.

Гипотеза 0. CatBoost против доли походов. Прирост Recall@10 не меньше 0,10.
Гипотеза 8. Та же модель, только скоропортящиеся, против «привычный продукт
кончается». Прирост Recall@10 не меньше 0,10. Модель для этого не переучивается.
Гипотеза 2. Две модели с одним числом деревьев. Долгосрочная: средний интервал,
доля походов, дней на единицу по всей истории. С недавней историей: то же плюс
последний интервал, среднее и разброс последних трёх, расход за 30 дней и его
отношение к расходу за всю историю. Прирост Recall@10 в группе не меньше 0,05,
на остальных семьях падение не больше 0,02.

Группа — не флаг генератора. На момент похода берутся только более ранние
покупки. По продукту последние 3 интервала — недавние, остальные — история.
Продукт без шести интервалов не входит. Интервалы складываются в две кучи.
DriftScore = |медиана недавних − медиана истории| / max(медиана истории, 1).
Нет таких продуктов — балл 0. Порог выбирается только на validation: сетка
0,25…1,50 с шагом 0,25, в группе не меньше 30 семей, берётся наибольший прирост,
при равенстве — больший порог. В группу теста входят походы, где балл уже
не ниже порога. Остальные семьи — те, у кого ни один тестовый поход порога
не достиг.
"""

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score

import cost_hd
import forecast

HERE = Path(__file__).resolve().parent
DB = HERE.parents[2] / "Общие_артефакты" / "данные" / "fridge_lab_v2.sqlite"
HORIZON = 14
SEED = 42
GAIN = 0.10
GROUP_GAIN = 0.05
OTHER_DROP = 0.02
MIN_HOUSEHOLDS = 30
DRIFT_GRID = (0.25, 0.50, 0.75, 1.00, 1.25, 1.50)
PERIODS = (
    ("train", "2025-10-01", "2026-06-01"),
    ("val", "2026-06-01", "2026-08-01"),
    ("test", "2026-08-01", "2026-10-01"),
)
LONG = ["base_product", "pair_mean_gap", "pair_visit_share", "pair_days_per_unit"]
RECENT = LONG + ["pair_last_gap", "recent_mean_3", "recent_std_3", "app_rate_30d", "app_rate_ratio"]


def fit(frame, features, eval_frame=None, iterations=2000):
    """Обучить CatBoost на строках frame по списку features.

    Метка — ``in_next_basket``. Категориальный признак в Pool —
    только ``base_product``. Семя ``SEED``. Файлы CatBoost не
    пишутся. Если задан ``eval_frame``, включаются ``od_type=Iter``,
    ``od_wait=100`` и ``use_best_model``.

    Args:
        frame (pandas.DataFrame): Обучающие строки со столбцами
            ``features`` и ``in_next_basket``.
        features (list): Имена столбцов признаков.
        eval_frame (pandas.DataFrame | None): Строки для ранней
            остановки. По умолчанию None — учится ``iterations``
            деревьев без eval_set.
        iterations: Число деревьев, если ранней остановки нет.
            По умолчанию 2000.

    Returns:
        Обученный ``CatBoostClassifier``.

    Notes:
        Печатает число деревьев и время обучения в секундах.
    """
    params = dict(
        iterations=iterations,
        learning_rate=0.08,
        depth=6,
        l2_leaf_reg=10,
        loss_function="Logloss",
        random_seed=SEED,
        thread_count=-1,
        verbose=0,
        allow_writing_files=False,
    )
    if eval_frame is not None:
        params["od_type"] = "Iter"
        params["od_wait"] = 100
    model = CatBoostClassifier(**params)
    train_pool = Pool(
        frame[features],
        label=frame["in_next_basket"].astype("int8"),
        cat_features=["base_product"],
    )
    eval_pool = None
    if eval_frame is not None:
        eval_pool = Pool(
            eval_frame[features],
            label=eval_frame["in_next_basket"].astype("int8"),
            cat_features=["base_product"],
        )
    started = perf_counter()
    model.fit(train_pool, eval_set=eval_pool, use_best_model=eval_frame is not None)
    print(f"деревьев {model.tree_count_}, {perf_counter() - started:.0f} с", flush=True)
    return model


def scores(model, frame, features):
    """Вернуть вероятность класса 1 по строкам frame.

    Модель не обучается. В Pool категориальный признак —
    только ``base_product``.

    Args:
        model (CatBoostClassifier): Обученная модель.
        frame (pandas.DataFrame): Строки со столбцами ``features``.
        features (list): Имена столбцов признаков.

    Returns:
        ndarray вероятностей ``predict_proba`` для класса 1,
        по одной на строку ``frame``.
    """
    pool = Pool(frame[features], cat_features=["base_product"])
    return model.predict_proba(pool)[:, 1]


def hit_rate(frame, score):
    """Вернуть долю походов с хотя бы одной купленной позицией в десятке.

    Внутри похода строки сортируются по ``score`` убывая, при
    равенстве — по ``base_product`` возрастая. В десятку входят
    первые 10 строк похода.

    Args:
        frame (pandas.DataFrame): Строки с ``visit_id``,
            ``base_product``, ``in_next_basket``.
        score: Баллы той же длины, что ``frame``, в том же порядке.

    Returns:
        Среднее по походам индикатора, что в десятке есть хотя бы
        одна строка с истинным ``in_next_basket``.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> frame = pd.DataFrame({
        ...     "visit_id": [1, 1, 2, 2],
        ...     "base_product": [1, 2, 1, 2],
        ...     "in_next_basket": [True, False, False, True],
        ... })
        >>> hit_rate(frame, np.array([0.9, 0.1, 0.9, 0.1]))
        1.0
    """
    order = frame.assign(_score=score, _label=frame["in_next_basket"].astype(bool)).sort_values(
        ["visit_id", "_score", "base_product"], ascending=[True, False, True]
    )
    top = order[order.groupby("visit_id", sort=False).cumcount() < 10]
    return float(top.groupby("visit_id")["_label"].any().mean())


def gap_table(purchases):
    """Вернуть интервалы между соседними покупками пары.

    Пара — ``hh`` и ``base_product``. Интервал — разность
    ``actual_at`` в сутках (секунды / 86400). Первая покупка пары
    не входит: у неё разность NA.

    Args:
        purchases (pandas.DataFrame): Покупки с ``hh``,
            ``base_product``, ``actual_at``.

    Returns:
        DataFrame столбцов ``hh``, ``base_product``, ``at``,
        ``gap``, отсортированный по этим полям кроме ``gap``.
    """
    ordered = purchases.sort_values(["hh", "base_product", "actual_at"])
    gap = ordered.groupby(["hh", "base_product"], sort=False)["actual_at"].diff().dt.total_seconds() / 86400.0
    keep = gap.notna()
    table = pd.DataFrame(
        {
            "hh": ordered.loc[keep, "hh"].to_numpy(),
            "base_product": ordered.loc[keep, "base_product"].to_numpy(),
            "at": ordered.loc[keep, "actual_at"].to_numpy(),
            "gap": gap.loc[keep].to_numpy(dtype="float64"),
        }
    )
    return table.sort_values(["hh", "base_product", "at"])


def attach_recent(frame, purchases):
    """Добавить среднее и разброс последних трёх интервалов пары.

    Интервалы считаются по покупкам не позже ``as_of``
    (``merge_asof`` назад, точные совпадения разрешены).
    Стандартное отклонение — с ``ddof=0``. Исходный ``frame`` не
    меняется: возвращается копия.

    Args:
        frame (pandas.DataFrame): Строки с ``hh``,
            ``base_product``, ``as_of``.
        purchases (pandas.DataFrame): Покупки с ``hh``,
            ``base_product``, ``actual_at``.

    Returns:
        Копия ``frame`` со столбцами ``recent_mean_3`` и
        ``recent_std_3`` типа float32.
    """
    ordered = purchases.sort_values(["hh", "base_product", "actual_at"]).copy()
    ordered["gap"] = ordered.groupby(["hh", "base_product"], sort=False)["actual_at"].diff().dt.total_seconds() / 86400.0
    grouped = ordered.groupby(["hh", "base_product"], sort=False)["gap"]
    ordered["gap_2"] = grouped.shift(1)
    ordered["gap_3"] = grouped.shift(2)
    three = ordered[["gap", "gap_2", "gap_3"]]
    ordered["recent_mean_3"] = three.mean(axis=1, skipna=True)
    ordered["recent_std_3"] = three.std(axis=1, skipna=True, ddof=0)
    left = frame[["hh", "base_product", "as_of"]].copy()
    left["_row"] = np.arange(len(frame))
    joined = pd.merge_asof(
        left.sort_values("as_of"),
        ordered.sort_values("actual_at"),
        left_on="as_of",
        right_on="actual_at",
        by=["hh", "base_product"],
        direction="backward",
        allow_exact_matches=True,
    ).sort_values("_row")
    frame = frame.copy()
    frame["recent_mean_3"] = joined["recent_mean_3"].to_numpy(dtype="float32")
    frame["recent_std_3"] = joined["recent_std_3"].to_numpy(dtype="float32")
    return frame


def drift_by_visit(frame, gaps):
    """Вернуть DriftScore каждого похода.

    Для продукта семьи на момент ``as_of`` нужны не меньше шести
    интервалов строго до этого момента. Последние три идут в
    «недавние», остальные — в «историю». Балл похода:

    ``|медиана недавних − медиана истории| / max(медиана истории, 1)``.

    Интервалы всех подходящих продуктов семьи складываются в две
    общие кучи, затем берутся медианы куч. Нет ни одного
    подходящего продукта — балл 0.0.

    Args:
        frame (pandas.DataFrame): Строки с ``visit_id``, ``hh``,
            ``as_of``. Дубликаты ``visit_id`` схлопываются.
        gaps (pandas.DataFrame): Интервалы ``gap_table`` с
            ``hh``, ``base_product``, ``at``, ``gap``.

    Returns:
        Словарь ``visit_id`` → float.
    """
    buckets = {}
    for (hh, _product), part in gaps.groupby(["hh", "base_product"], sort=False):
        buckets.setdefault(hh, []).append((part["at"].to_numpy(), part["gap"].to_numpy()))
    visits = frame.drop_duplicates("visit_id")
    scores = {}
    for visit, hh, as_of in zip(visits["visit_id"], visits["hh"], visits["as_of"]):
        recent_parts = []
        history_parts = []
        moment = np.datetime64(as_of)
        for times, values in buckets.get(hh, ()):
            count = int(np.searchsorted(times, moment, side="right"))
            if count < 6:
                continue
            history_parts.append(values[: count - 3])
            recent_parts.append(values[count - 3 : count])
        if not recent_parts:
            scores[visit] = 0.0
            continue
        history_median = float(np.median(np.concatenate(history_parts)))
        recent_median = float(np.median(np.concatenate(recent_parts)))
        scores[visit] = abs(recent_median - history_median) / max(history_median, 1.0)
    return scores


def recall_of(frame, score, visits):
    """Вернуть Recall@10 на подмножестве походов visits.

    Баллы для среза берутся как ``score`` в порядке индекса
    ``frame``: ``np.asarray(score)[frame.index.get_indexer(part.index)]``.

    Args:
        frame (pandas.DataFrame): Строки с ``visit_id`` и меткой
            для ``cost_hd.recall_at_10``.
        score: Баллы той же длины, что ``frame``, в том же порядке.
        visits: Набор ``visit_id`` для среза.

    Returns:
        Кортеж ``(recall, n)``. Если срез пуст —
        ``(nan, 0)``. Иначе результат ``cost_hd.recall_at_10``.
    """
    part = frame[frame["visit_id"].isin(visits)]
    if part.empty:
        return float("nan"), 0
    return cost_hd.recall_at_10(part, np.asarray(score)[frame.index.get_indexer(part.index)])


def main():
    """Прогнать гипотезы 0, 8 и 2 на ``fridge_lab_v2.sqlite``.

    Пороги ``GAIN``, ``GROUP_GAIN``, ``OTHER_DROP`` не меняются.
    Учёба, валидация и тест — интервалы ``PERIODS``. Гипотеза 2
    выбирает порог DriftScore только на validation; если семей
    меньше ``MIN_HOUSEHOLDS`` на всей сетке или в тесте нет
    походов группы, печатает отказ и выходит.

    Notes:
        Читает SQLite по ``DB``, обучает CatBoost, пишет вердикты
        в stdout. Модельные файлы не сохраняет.
    """
    print("пороги до цифр: список и скоропортящиеся +0,10; смена +0,05 и падение остальных не больше 0,02", flush=True)
    data = forecast.load(DB)
    parts = {}
    for name, start, end in PERIODS:
        started = perf_counter()
        frame, _stats = forecast.build_rows(data, start, end, HORIZON)
        frame = attach_recent(frame, data["purchases"])
        parts[name] = frame
        print(
            f"{name}: походов {frame['visit_id'].nunique()}, строк {len(frame)}, {perf_counter() - started:.0f} с",
            flush=True,
        )
    gaps = gap_table(data["purchases"])
    train, val, test = parts["train"], parts["val"], parts["test"]
    scored = {name: frame[frame["has_next_visit"]].reset_index(drop=True) for name, frame in (("val", val), ("test", test))}

    print("гипотеза 0, учу CatBoost на всех признаках", flush=True)
    full = forecast.FEATURES
    model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.08,
        depth=6,
        l2_leaf_reg=10,
        loss_function="Logloss",
        od_type="Iter",
        od_wait=100,
        random_seed=SEED,
        thread_count=-1,
        verbose=0,
        allow_writing_files=False,
    )
    started = perf_counter()
    model.fit(
        Pool(train[full], label=train["in_next_basket"].astype("int8"), cat_features=forecast.CATEGORICAL_FEATURES),
        eval_set=Pool(val[full], label=val["in_next_basket"].astype("int8"), cat_features=forecast.CATEGORICAL_FEATURES),
        use_best_model=True,
    )
    print(f"деревьев {model.tree_count_}, {perf_counter() - started:.0f} с", flush=True)
    held = scored["test"]
    probability = model.predict_proba(Pool(held[full], cat_features=forecast.CATEGORICAL_FEATURES))[:, 1]
    frequency = held["pair_visit_share"].to_numpy()
    model_recall, trips = cost_hd.recall_at_10(held, probability)
    rule_recall, _trips = cost_hd.recall_at_10(held, frequency)
    gain = model_recall - rule_recall
    print(
        f"тест, походов {trips}: частота {rule_recall:.3f}, CatBoost {model_recall:.3f}, "
        f"прирост {gain:.3f}, HitRate {hit_rate(held, probability):.3f}, "
        f"PR-AUC {average_precision_score(held['in_next_basket'], probability):.3f}",
        flush=True,
    )
    print("гипотеза 0 подтверждена" if gain >= GAIN else "гипотеза 0 не подтверждена", flush=True)

    print("гипотеза 8, та же модель, только скоропортящиеся", flush=True)
    perishable = held[held["is_perishable"] >= 0.5]
    perishable_score = probability[held.index.get_indexer(perishable.index)]
    to_plan = -perishable["app_supply_vs_plan"].fillna(365).to_numpy()
    habit = np.where(perishable["pair_n"].to_numpy() >= 3, to_plan, to_plan - 10_000)
    habit_recall, habit_trips = cost_hd.recall_at_10(perishable, habit)
    fresh_recall, fresh_trips = cost_hd.recall_at_10(perishable, perishable_score)
    fresh_gain = fresh_recall - habit_recall
    low, high = cost_hd.recall_change_interval(perishable, habit, perishable_score)
    print(
        f"скоропортящиеся, походов {fresh_trips}: правило {habit_recall:.3f}, "
        f"CatBoost {fresh_recall:.3f}, прирост {fresh_gain:.3f}, 95% [{low:.3f}; {high:.3f}]",
        flush=True,
    )
    print("гипотеза 8 подтверждена" if fresh_gain >= GAIN else "гипотеза 8 не подтверждена", flush=True)

    print("гипотеза 2, учу долгосрочную модель", flush=True)
    long_model = fit(train, LONG, val)
    trees = int(long_model.tree_count_)
    print("гипотеза 2, учу модель с недавней историей на том же числе деревьев", flush=True)
    recent_model = fit(train, RECENT, iterations=trees)
    drift = {name: drift_by_visit(frame, gaps) for name, frame in scored.items()}
    long_score = {name: scores(long_model, frame, LONG) for name, frame in scored.items()}
    recent_score = {name: scores(recent_model, frame, RECENT) for name, frame in scored.items()}

    print("порог DriftScore по validation", flush=True)
    chosen = None
    for threshold in DRIFT_GRID:
        households, visits = split_group(scored["val"], drift["val"], threshold)
        if len(households) < MIN_HOUSEHOLDS:
            print(f"порог {threshold:.2f}: семей {len(households)}, меньше 30, мимо", flush=True)
            continue
        base, _n = recall_of(scored["val"], long_score["val"], visits)
        trial, n_visits = recall_of(scored["val"], recent_score["val"], visits)
        trial_gain = trial - base
        print(
            f"порог {threshold:.2f}: семей {len(households)}, походов {n_visits}, "
            f"прирост {trial_gain:.3f}",
            flush=True,
        )
        key = (trial_gain, threshold)
        if chosen is None or key > chosen[0]:
            chosen = (key, threshold)
    if chosen is None:
        print("гипотеза 2 не подтверждена: на validation нет порога с 30 семьями", flush=True)
        return
    threshold = chosen[1]
    print(f"порог зафиксирован: {threshold:.2f}", flush=True)

    households, visits = split_group(scored["test"], drift["test"], threshold)
    if not visits:
        print("гипотеза 2 не подтверждена: в тесте нет походов с этим порогом", flush=True)
        return
    others = set(scored["test"]["hh"]) - households
    other_visits = set(scored["test"].loc[scored["test"]["hh"].isin(others), "visit_id"])
    long_group, group_trips = recall_of(scored["test"], long_score["test"], visits)
    recent_group, _trips = recall_of(scored["test"], recent_score["test"], visits)
    long_other, other_trips = recall_of(scored["test"], long_score["test"], other_visits)
    recent_other, _other = recall_of(scored["test"], recent_score["test"], other_visits)
    group_gain = recent_group - long_group
    other_change = recent_other - long_other
    group_frame = scored["test"][scored["test"]["visit_id"].isin(visits)]
    group_low, group_high = cost_hd.recall_change_interval(
        group_frame,
        _aligned(scored["test"], long_score["test"], visits),
        _aligned(scored["test"], recent_score["test"], visits),
    )
    print(
        f"тест, порог {threshold:.2f}, семей в группе {len(households)}, походов {group_trips}: "
        f"долгосрочная {long_group:.3f}, недавняя {recent_group:.3f}, прирост {group_gain:.3f}, "
        f"95% [{group_low:.3f}; {group_high:.3f}]",
        flush=True,
    )
    print(
        f"остальные семьи {len(others)}, походов {other_trips}: "
        f"долгосрочная {long_other:.3f}, недавняя {recent_other:.3f}, разница {other_change:.3f}",
        flush=True,
    )
    confirmed = group_gain >= GROUP_GAIN and other_change >= -OTHER_DROP
    print("гипотеза 2 подтверждена" if confirmed else "гипотеза 2 не подтверждена", flush=True)


def _aligned(frame, score, visits):
    """Вернуть баллы строк, чей ``visit_id`` входит в visits.

    Порядок совпадает с порядком таких строк в ``frame``.

    Args:
        frame (pandas.DataFrame): Строки с ``visit_id``.
        score: Баллы той же длины, что ``frame``.
        visits: Набор ``visit_id``.

    Returns:
        ndarray баллов на отобранных строках.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> frame = pd.DataFrame({"visit_id": [1, 2, 1]})
        >>> _aligned(frame, np.array([0.1, 0.2, 0.3]), {1})
        array([0.1, 0.3])
    """
    part = frame["visit_id"].isin(visits)
    return np.asarray(score)[part.to_numpy()]


def split_group(frame, visit_score, threshold):
    """Вернуть семьи и походы не ниже порога DriftScore.

    Семья входит, если максимум балла её походов в ``frame``
    не ниже порога. Поход входит, если его балл не ниже порога.

    Args:
        frame (pandas.DataFrame): Строки с ``visit_id`` и ``hh``.
        visit_score (dict): Отображение ``visit_id`` → балл.
        threshold: Порог сравнения.

    Returns:
        Кортеж ``(households, visits)`` — множества семей и
        походов.

    Examples:
        >>> import pandas as pd
        >>> frame = pd.DataFrame({
        ...     "visit_id": [10, 10, 20],
        ...     "hh": [1, 1, 2],
        ... })
        >>> split_group(frame, {10: 1.0, 20: 0.2}, 0.5)
        ({1}, {10})
    """
    trip_score = frame["visit_id"].map(visit_score)
    household_max = trip_score.groupby(frame["hh"]).transform("max")
    households = set(frame.loc[household_max >= threshold, "hh"])
    visits = set(frame.loc[trip_score >= threshold, "visit_id"])
    return households, visits


if __name__ == "__main__":
    main()
