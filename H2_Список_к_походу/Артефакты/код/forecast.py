"""Строки «поход × продукт из холодильника» для прогноза покупок по fridge_lab.

Момент прогноза — через AS_OF_HOURS часов после похода: покупки уже внесены
в кладовую, дата следующего планового похода известна. Кандидаты — базовые
продукты, купленные домохозяйством за последние WINDOW_DAYS дней.

Признаки — только то, что видит приложение к моменту прогноза: чеки, события
кладовой с recorded_at не позже момента (забытые и запоздавшие отметки так и
остаются шумом), профиль домохозяйства. Таблицы sim_* — правда генератора,
в признаки они не попадают.
"""

import sqlite3

import numpy as np
import pandas as pd

WINDOW_DAYS = 60
AS_OF_HOURS = 3
RATE_DAYS = 30

CATEGORICAL_FEATURES = ["base_product", "category", "segment"]
NUMERIC_FEATURES = [
    "pair_n",
    "pair_days_since_last",
    "pair_bought_now",
    "pair_last_gap",
    "pair_mean_gap",
    "pair_recency_ratio",
    "pair_last_amount_ratio",
    "pair_days_per_unit",
    "pair_left_ratio",
    "pair_visit_share",
    "app_stock_ratio",
    "app_rate_30d",
    "app_rate_ratio",
    "app_days_of_supply",
    "app_supply_vs_plan",
    "app_waste_share",
    "app_days_since_event",
    "last_lot_days_to_expiry",
    "hh_size",
    "hh_n_trips",
    "hh_mean_trip_gap",
    "hh_days_since_first",
    "hh_basket",
    "hh_mean_basket",
    "hh_fridge_size",
    "trip_is_planned",
    "days_to_plan",
    "shelf_days",
    "is_perishable",
    "prod_median_gap",
    "prod_n_households",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

_DAY = pd.Timedelta(days=1)


def _days(delta):
    """Вернуть длительность в сутках.

    Делит значение на ``pd.Timedelta(days=1)``. Дробная часть
    сохраняется.

    Args:
        delta: Интервал Timedelta или Series таких интервалов.

    Returns:
        Число суток той же формы, что даёт деление на один день.

    Examples:
        >>> import pandas as pd
        >>> _days(pd.Timedelta(hours=12))
        0.5
    """
    return delta / _DAY


def load(db_path):
    """Загрузить таблицы приложения и генератора из SQLite.

    Даты переводятся в datetime. Походы сортируются по ``hh`` и
    ``actual_at``. К покупкам присоединяются ``hh`` и ``actual_at``
    похода. Строки ``shopping_trip`` без ``actual_at`` не читаются.

    Args:
        db_path: Путь к файлу SQLite.

    Returns:
        Словарь DataFrame с ключами ``trips``, ``purchases``,
        ``events``, ``bases``, ``households``, ``changes``.

    Notes:
        Открывает ``db_path`` через sqlite3. Таблица
        ``sim_household_change`` читается; в признаки
        ``build_rows`` она не входит.
    """
    with sqlite3.connect(db_path) as connection:
        read = lambda sql: pd.read_sql_query(sql, connection)
        trips = read("""
            select id as trip, id_household as hh, actual_at, planned_date, created_at
            from shopping_trip where actual_at is not null
        """)
        purchases = read("""
            select pl.id_shopping_trip as trip, p.id_base_product as base_product,
                   sum(pl.quantity_base) as amount, sum(pl.price) as spend, count(*) as lines,
                   min(l.best_before) as best_before
            from purchase_line pl
            join product p on p.id = pl.id_product
            left join lot l on l.id_purchase_line = pl.id
            group by pl.id_shopping_trip, p.id_base_product
        """)
        events = read("""
            select id_household as hh, id_base_product as base_product, event_type,
                   quantity_base_delta as delta, recorded_at
            from lot_event
        """)
        bases = read("""
            select b.id as base_product, b.name, c.name as category,
                   b.typical_shelf_days as shelf_days, b.is_perishable
            from base_product b join category c on c.id = b.id_category
        """)
        households = read("select id as hh, created_at, size as hh_size, segment from household")
        changes = read("select id_household as hh, changed_at, kind from sim_household_change")

    for frame, columns in [(trips, ["actual_at", "planned_date", "created_at"]),
                           (purchases, ["best_before"]), (events, ["recorded_at"]),
                           (households, ["created_at"]), (changes, ["changed_at"])]:
        for column in columns:
            frame[column] = pd.to_datetime(frame[column])
    trips = trips.sort_values(["hh", "actual_at"]).reset_index(drop=True)
    purchases = purchases.merge(trips[["trip", "hh", "actual_at"]], on="trip")
    return {"trips": trips, "purchases": purchases, "events": events, "bases": bases,
            "households": households, "changes": changes}


def visit_table(trips, purchases):
    """Вернуть походы с признаками домохозяйства на момент прогноза.

    Момент прогноза ``as_of`` — ``actual_at`` плюс ``AS_OF_HOURS``
    часов. ``days_to_plan`` заполняется, только если
    ``plan_created`` не позже ``as_of``. Исходный ``trips`` не
    меняется.

    Args:
        trips (pandas.DataFrame): Походы с ``trip``, ``hh``,
            ``actual_at``, ``planned_date``, ``created_at``.
        purchases (pandas.DataFrame): Покупки с ``trip`` и
            ``lines``.

    Returns:
        Копия походов со столбцами ``as_of``, ``day``,
        ``hh_basket``, ``hh_n_trips``, ``hh_days_since_first``,
        ``hh_mean_trip_gap``, ``hh_mean_basket``,
        ``trip_is_planned``, ``next_trip``, ``next_day``,
        ``plan_date``, ``plan_created``, ``days_to_plan``.
        При одном походе семьи ``hh_mean_trip_gap`` — NA.
        У последнего похода семьи ``next_trip`` и ``next_day`` —
        NA.
    """
    table = trips.copy()
    table["as_of"] = table["actual_at"] + pd.Timedelta(hours=AS_OF_HOURS)
    table["day"] = table["actual_at"].dt.normalize()
    table["hh_basket"] = table["trip"].map(purchases.groupby("trip")["lines"].sum()).fillna(0)
    by_hh = table.groupby("hh")
    table["hh_n_trips"] = by_hh.cumcount() + 1
    table["hh_days_since_first"] = _days(table["day"] - by_hh["day"].transform("min"))
    table["hh_mean_trip_gap"] = (
        table["hh_days_since_first"] / (table["hh_n_trips"] - 1)
    ).where(table["hh_n_trips"] > 1)
    table["hh_mean_basket"] = by_hh["hh_basket"].cumsum() / table["hh_n_trips"]
    table["trip_is_planned"] = table["planned_date"].notna().astype("float32")
    table["next_trip"] = by_hh["trip"].shift(-1)
    table["next_day"] = by_hh["day"].shift(-1)

    # Дата ближайшего планового похода после этого: план составлен на прошлом
    # плановом походе и известен, если создан не позже момента прогноза.
    planned = table["planned_date"].notna()
    plan_date = table["planned_date"].where(planned)
    plan_created = table["created_at"].where(planned)
    table["plan_date"] = plan_date.groupby(table["hh"]).shift(-1)
    table["plan_created"] = plan_created.groupby(table["hh"]).shift(-1)
    table[["plan_date", "plan_created"]] = table.groupby("hh")[["plan_date", "plan_created"]].bfill()
    known = table["plan_created"] <= table["as_of"]
    table["days_to_plan"] = _days(table["plan_date"] - table["day"]).where(known)
    return table


def pair_purchases(purchases):
    """Вернуть покупки пары с накопленной историей.

    Пара — ``hh`` и ``base_product``. Строки сортируются по паре
    и ``actual_at``, индекс сбрасывается.

    Args:
        purchases (pandas.DataFrame): Покупки с ``hh``,
            ``base_product``, ``actual_at``, ``amount``.

    Returns:
        DataFrame тех же покупок со столбцами ``pair_n``,
        ``pair_amount_cum``, ``pair_first``, ``pair_prev``.
        У первой покупки пары ``pair_prev`` — NA.
    """
    frame = purchases.sort_values(["hh", "base_product", "actual_at"]).reset_index(drop=True)
    by_pair = frame.groupby(["hh", "base_product"])
    frame["pair_n"] = by_pair.cumcount() + 1
    frame["pair_amount_cum"] = by_pair["amount"].cumsum()
    frame["pair_first"] = by_pair["actual_at"].transform("min")
    frame["pair_prev"] = by_pair["actual_at"].shift(1)
    return frame


def monthly_product_stats(pairs, households):
    """Вернуть медианный интервал и число семей на начало месяца.

    Месяцы — с первого начала месяца после минимума ``actual_at``
    до начала месяца после максимума включительно. Для месяца
    берутся только покупки строго раньше этой даты.

    Args:
        pairs (pandas.DataFrame): Покупки пары с ``actual_at``,
            ``pair_prev``, ``base_product``, ``hh``.
        households: Аргумент не используется.

    Returns:
        DataFrame с MultiIndex ``(month, base_product)`` и
        столбцами ``prod_median_gap``, ``prod_n_households``.
    """
    gaps = _days(pairs["actual_at"].dt.normalize() - pairs["pair_prev"].dt.normalize())
    months = pd.date_range(pairs["actual_at"].min().normalize() + pd.offsets.MonthBegin(1),
                           pairs["actual_at"].max() + pd.offsets.MonthBegin(1), freq="MS")
    stats = {}
    for month in months:
        before = pairs["actual_at"] < month
        stats[month] = pd.DataFrame({
            "prod_median_gap": gaps[before].groupby(pairs.loc[before, "base_product"]).median(),
            "prod_n_households": pairs[before].groupby("base_product")["hh"].nunique(),
        })
    return pd.concat(stats, names=["month", "base_product"])


def _cumulative_events(events):
    """Вернуть накопленные суммы событий пары по recorded_at.

    ``consumed`` и ``wasted`` — минус ``delta`` на строках
    соответствующего ``event_type``, иначе 0. ``added`` — ``delta``
    на ``added``, иначе 0. ``cum_stock`` — cumsum исходного
    ``delta``. Исходный ``events`` не меняется.

    Args:
        events (pandas.DataFrame): События с ``hh``,
            ``base_product``, ``event_type``, ``delta``,
            ``recorded_at``.

    Returns:
        DataFrame столбцов ``hh``, ``base_product``,
        ``recorded_at``, ``last_event``, ``cum_stock``,
        ``cum_consumed``, ``cum_wasted``, ``cum_added``.
        ``last_event`` совпадает с ``recorded_at``.
    """
    frame = events.sort_values("recorded_at").copy()
    frame["consumed"] = np.where(frame["event_type"] == "consumed", -frame["delta"], 0.0)
    frame["wasted"] = np.where(frame["event_type"] == "wasted", -frame["delta"], 0.0)
    frame["added"] = np.where(frame["event_type"] == "added", frame["delta"], 0.0)
    by_pair = frame.groupby(["hh", "base_product"])
    for column, source in [("cum_stock", "delta"), ("cum_consumed", "consumed"),
                           ("cum_wasted", "wasted"), ("cum_added", "added")]:
        frame[column] = by_pair[source].cumsum()
    frame["last_event"] = frame["recorded_at"]
    return frame[["hh", "base_product", "recorded_at", "last_event",
                  "cum_stock", "cum_consumed", "cum_wasted", "cum_added"]]


def _asof(left, right, left_on, right_on, columns, direction="backward", exact=True):
    """Вернуть столбцы правой таблицы, сопоставленные as-of по паре.

    Пара — ``hh`` и ``base_product``. Порядок строк ``left``
    сохраняется. ``left`` не меняется.

    Args:
        left (pandas.DataFrame): Левая таблица с ``hh``,
            ``base_product`` и столбцом ``left_on``.
        right (pandas.DataFrame): Правая таблица с ``hh``,
            ``base_product``, ``right_on`` и ``columns``.
        left_on: Имя столбца времени в ``left``.
        right_on: Имя столбца времени в ``right``.
        columns (list): Имена столбцов ``right``, которые вернуть.
        direction: Направление ``merge_asof``. По умолчанию
            ``backward``.
        exact: Передаётся в ``allow_exact_matches``. По умолчанию
            True.

    Returns:
        DataFrame столбцов ``columns`` с индексом ``left``.
    """
    order = left[left_on].to_numpy().argsort(kind="stable")
    joined = pd.merge_asof(
        left.iloc[order][["hh", "base_product", left_on]].reset_index(),
        right.sort_values(right_on)[list(dict.fromkeys(["hh", "base_product", right_on] + columns))],
        left_on=left_on, right_on=right_on, by=["hh", "base_product"],
        direction=direction, allow_exact_matches=exact,
    )
    return joined.set_index("index")[columns].reindex(left.index)


def build_rows(data, start, end, horizon_days):
    """Собрать строки «поход × продукт» для походов в [start, end).

    Остаются походы, у которых окно метки (день похода,
    + ``horizon_days``] целиком лежит до ``end``. Кандидаты —
    базовые продукты семьи; после as-of оставляются пары с
    покупкой строго меньше ``WINDOW_DAYS`` суток назад.
    Признаки — состояние на ``as_of``. Покупки на ``end`` и позже
    в метку не входят.

    Args:
        data (dict): Результат ``load`` с ключами ``trips``,
            ``purchases``, ``events``, ``bases``, ``households``.
        start: Начало окна походов, для ``pd.Timestamp``.
        end: Конец окна походов, не включая эту дату в метку.
        horizon_days: Длина окна метки ``label`` в сутках.

    Returns:
        Кортеж ``(frame, stats)``. ``frame`` — строки с
        идентификаторами, столбцами ``FEATURES``, метками
        ``label``, ``days_to_purchase``, ``has_next_visit``,
        ``in_next_basket``, ``days_to_next_visit``. ``stats`` —
        словарь ключей «Походов», «Строк», «Продуктов в
        холодильнике на поход», «Доля строк с покупкой в окне»,
        «Покрытие следующей корзины».

    Notes:
        Читает только переданные таблицы; файлы не пишет.
        ``pair_visit_share`` равен ``pair_n / hh_n_trips``.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    last_day = end - pd.Timedelta(days=horizon_days + 1)
    visits = visit_table(data["trips"], data["purchases"])
    pairs = pair_purchases(data["purchases"])
    products = monthly_product_stats(pairs, data["households"])

    snap = visits[(visits["day"] >= start) & (visits["day"] <= last_day)]
    candidates = pairs[["hh", "base_product"]].drop_duplicates()
    rows = snap.merge(candidates, on="hh").reset_index(drop=True)

    last = _asof(rows, pairs, "as_of", "actual_at",
                 ["actual_at", "pair_n", "amount", "pair_amount_cum", "pair_first", "pair_prev",
                  "spend", "best_before"])
    rows = pd.concat([rows, last.add_prefix("last_")], axis=1)
    rows = rows[rows["last_actual_at"].notna()]
    since = _days(rows["day"] - rows["last_actual_at"].dt.normalize())
    rows = rows[since < WINDOW_DAYS].reset_index(drop=True)

    future = pairs[pairs["actual_at"] < end].rename(columns={"actual_at": "next_at", "trip": "next_purchase_trip"})
    upcoming = _asof(rows, future, "as_of", "next_at", ["next_at", "next_purchase_trip"],
                     direction="forward", exact=False)
    rows = pd.concat([rows, upcoming], axis=1)

    cum = _cumulative_events(data["events"])
    now = _asof(rows, cum, "as_of", "recorded_at",
                ["cum_stock", "cum_consumed", "cum_wasted", "cum_added", "last_event"])
    rows["rate_from"] = rows["as_of"] - pd.Timedelta(days=RATE_DAYS)
    earlier = _asof(rows, cum, "rate_from", "recorded_at", ["cum_consumed"])

    frame = rows[["trip", "hh", "base_product", "day", "as_of"]].copy()
    frame = frame.merge(data["households"][["hh", "hh_size", "segment"]], on="hh", how="left")
    frame = frame.merge(data["bases"][["base_product", "category", "shelf_days", "is_perishable"]],
                        on="base_product", how="left")
    frame["visit_id"] = rows["trip"].to_numpy()

    pair_n = rows["last_pair_n"].to_numpy(dtype="float64")
    amount_cum = rows["last_pair_amount_cum"].to_numpy(dtype="float64")
    last_amount = rows["last_amount"].to_numpy(dtype="float64")
    mean_amount = amount_cum / pair_n
    span = _days(rows["last_actual_at"] - rows["last_pair_first"]).to_numpy()
    since = _days(rows["day"] - rows["last_actual_at"].dt.normalize()).to_numpy()

    frame["pair_n"] = pair_n
    frame["pair_days_since_last"] = since
    frame["pair_bought_now"] = (rows["last_actual_at"].to_numpy() == rows["actual_at"].to_numpy()).astype("float32")
    frame["pair_last_gap"] = _days(rows["last_actual_at"] - rows["last_pair_prev"]).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        frame["pair_mean_gap"] = np.where(pair_n > 1, span / (pair_n - 1), np.nan)
        frame["pair_recency_ratio"] = since / frame["pair_mean_gap"]
        frame["pair_last_amount_ratio"] = last_amount / mean_amount
        days_per_unit = np.where(pair_n > 1, span / (amount_cum - last_amount), np.nan)
        frame["pair_days_per_unit"] = days_per_unit * mean_amount
        frame["pair_left_ratio"] = np.clip(1.0 - since / (last_amount * days_per_unit), -5.0, None)
        frame["pair_visit_share"] = pair_n / rows["hh_n_trips"].to_numpy()

        stock = now["cum_stock"].clip(lower=0).to_numpy()
        rate_30d = (now["cum_consumed"] - earlier["cum_consumed"].fillna(0)).to_numpy() / RATE_DAYS
        known_days = np.clip(_days(rows["as_of"] - rows["last_pair_first"]).to_numpy(), 1, None)
        rate_all = now["cum_consumed"].to_numpy() / known_days
        frame["app_stock_ratio"] = stock / mean_amount
        frame["app_rate_30d"] = rate_30d / mean_amount
        frame["app_rate_ratio"] = rate_30d / rate_all
        rate = np.where(rate_30d > 0, rate_30d, rate_all)
        frame["app_days_of_supply"] = np.clip(np.where(rate > 0, stock / rate, np.nan), None, 365)
        frame["app_waste_share"] = now["cum_wasted"].to_numpy() / now["cum_added"].to_numpy()
    frame["app_days_since_event"] = _days(rows["as_of"] - now["last_event"]).to_numpy()
    frame["last_lot_days_to_expiry"] = _days(rows["last_best_before"] - rows["day"]).to_numpy()

    for column in ["hh_n_trips", "hh_mean_trip_gap", "hh_days_since_first", "hh_basket",
                   "hh_mean_basket", "trip_is_planned", "days_to_plan"]:
        frame[column] = rows[column].to_numpy()
    frame["app_supply_vs_plan"] = frame["app_days_of_supply"] - frame["days_to_plan"]
    frame["hh_fridge_size"] = frame.groupby("visit_id")["base_product"].transform("size")

    frame["month"] = frame["day"].dt.to_period("M").dt.to_timestamp()
    frame = frame.join(products, on=["month", "base_product"])

    days_to_next = _days(rows["next_at"].dt.normalize() - rows["day"]).to_numpy()
    days_to_next = np.where(np.isnan(days_to_next), np.inf, np.maximum(days_to_next, 1))
    frame["label"] = (days_to_next <= horizon_days).astype("int8")
    frame["days_to_purchase"] = np.where(frame["label"] == 1, days_to_next, np.nan)
    next_day = rows["next_day"]
    frame["has_next_visit"] = (next_day < end).to_numpy()
    frame["in_next_basket"] = frame["has_next_visit"] & (
        rows["next_purchase_trip"].to_numpy() == rows["next_trip"].to_numpy())
    frame["days_to_next_visit"] = np.where(frame["has_next_visit"], _days(next_day - rows["day"]), np.nan)

    frame["segment"] = frame["segment"].fillna("не указан")
    frame[NUMERIC_FEATURES] = frame[NUMERIC_FEATURES].replace([np.inf, -np.inf], np.nan).astype("float32")

    future_visits = snap["next_day"] < end
    next_basket = data["purchases"].groupby("trip").size()
    stats = {
        "Походов": len(snap),
        "Строк": len(frame),
        "Продуктов в холодильнике на поход": len(frame) / max(len(snap), 1),
        "Доля строк с покупкой в окне": float(frame["label"].mean()),
        "Покрытие следующей корзины": float(
            frame["in_next_basket"].sum()
            / snap.loc[future_visits, "next_trip"].map(next_basket).sum()
        ),
    }
    return frame, stats
