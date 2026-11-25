-- Учебная база AI-Fridge для анализа потребления и прогноза покупок.
--
-- Соглашения:
--   * Аккаунт — домохозяйство (household): прогноз строится на него, а не на человека.
--   * Количество хранится ещё и в базовых единицах (*_base): граммы, миллилитры или штуки.
--   * Время — местное время домохозяйства, текст 'YYYY-MM-DD HH:MM:SS'; дата — 'YYYY-MM-DD'.
--     Смещение часового пояса не пишется: date() в SQLite перевела бы его в UTC.
--   * Партия (lot) не меняется после создания. Всё, что с ней происходит, — строки lot_event.
--     Остаток партии = сумма quantity_base_delta её событий.

pragma foreign_keys = on;

-- Справочники ---------------------------------------------------------------

create table unit (
    id         integer primary key,
    name       text not null unique,
    dimension  text not null check (dimension in ('mass', 'volume', 'count')),
    to_base    real not null check (to_base > 0)          -- кг → 1000 г
);

create table category (
    id    integer primary key,
    name  text not null unique
);

create table storage_place (
    id    integer primary key,
    name  text not null unique
);

-- То, что прогнозируем: «молоко», а не конкретная марка и фасовка.
create table base_product (
    id                  integer primary key,
    name                text not null unique,
    id_category         integer not null references category (id),
    dimension           text not null check (dimension in ('mass', 'volume', 'count')),
    typical_shelf_days  integer not null check (typical_shelf_days > 0),
    is_perishable       integer not null check (is_perishable in (0, 1)),
    id_storage_place    integer not null references storage_place (id)
);

-- То, что покупаем: конкретная фасовка базового продукта.
create table product (
    id               integer primary key,
    id_base_product  integer not null references base_product (id),
    name             text not null unique,
    pack_size        real not null check (pack_size > 0),
    id_unit          integer not null references unit (id),
    pack_size_base   real not null check (pack_size_base > 0),
    price            real check (price > 0)                    -- ориентир цены за упаковку, руб.
);

-- Аккаунт -------------------------------------------------------------------

create table household (
    id          integer primary key,
    created_at  text not null check (created_at glob '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9] [0-9][0-9]:[0-9][0-9]:[0-9][0-9]'),
    size        integer check (size > 0),
    segment     text
);

create table member (
    id            integer primary key,
    id_household  integer not null references household (id),
    joined_at     text not null check (joined_at glob '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9] [0-9][0-9]:[0-9][0-9]:[0-9][0-9]')
);

-- Покупки -------------------------------------------------------------------

create table shopping_trip (
    id            integer primary key,
    id_household  integer not null references household (id),
    planned_date  text check (planned_date glob '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]'),
    actual_at     text check (actual_at glob '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9] [0-9][0-9]:[0-9][0-9]:[0-9][0-9]'),
    created_at    text not null check (created_at glob '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9] [0-9][0-9]:[0-9][0-9]:[0-9][0-9]')
);

create table purchase_line (
    id                 integer primary key,
    id_shopping_trip   integer not null references shopping_trip (id),
    id_product         integer not null references product (id),
    packs              real not null check (packs > 0),
    quantity_base      real not null check (quantity_base > 0),
    price              real check (price >= 0),                -- за всю строку
    source             text not null check (source in ('manual', 'receipt', 'voice', 'recommendation')),
    id_recommendation  integer references recommendation (id),
    check ((source = 'recommendation') = (id_recommendation is not null))
);

-- Кладовая ------------------------------------------------------------------

create table lot (
    id                integer primary key,
    id_household      integer not null references household (id),
    id_product        integer not null references product (id),
    id_purchase_line  integer unique references purchase_line (id),   -- null — добавлено без покупки
    id_storage_place  integer not null references storage_place (id),
    added_at          text not null check (added_at glob '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9] [0-9][0-9]:[0-9][0-9]:[0-9][0-9]'),
    quantity_base     real not null check (quantity_base > 0),
    best_before       text check (best_before glob '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]'),
    source            text not null check (source in ('checkout', 'manual', 'receipt', 'voice', 'gift')),
    check ((source = 'checkout') = (id_purchase_line is not null))
);

-- added     — приход, +quantity_base партии (ровно одно на партию);
-- consumed  — съели, отрицательная дельта;
-- wasted    — выбросили, отрицательная дельта и причина;
-- corrected — пользователь поправил остаток, дельта любого знака;
-- mistake   — партия заведена по ошибке, дельта снимает весь остаток.
create table lot_event (
    id                   integer primary key,
    id_lot               integer not null references lot (id),
    id_household         integer not null references household (id),
    id_base_product      integer not null references base_product (id),
    event_type           text not null check (event_type in ('added', 'consumed', 'wasted', 'corrected', 'mistake')),
    quantity_base_delta  real not null,
    waste_reason         text check (waste_reason in ('spoiled', 'expired', 'other')),
    occurred_at          text not null check (occurred_at glob '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9] [0-9][0-9]:[0-9][0-9]:[0-9][0-9]'),
    recorded_at          text not null check (recorded_at glob '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9] [0-9][0-9]:[0-9][0-9]:[0-9][0-9]'),
    id_member            integer references member (id),
    check (recorded_at >= occurred_at),
    check (
        (event_type = 'added' and quantity_base_delta > 0)
        or (event_type in ('consumed', 'wasted', 'mistake') and quantity_base_delta < 0)
        or event_type = 'corrected'
    ),
    check ((event_type = 'wasted') = (waste_reason is not null))
);

-- ML: прогнозы и реакция пользователя ---------------------------------------

create table model_run (
    id             integer primary key,
    model_name     text not null,
    model_version  text not null,
    trained_until  text not null,                  -- граница train: прогнозы раньше неё — утечка
    created_at     text not null,
    params         text                            -- JSON
);

create table forecast (
    id               integer primary key,
    id_model_run     integer not null references model_run (id),
    id_household     integer not null references household (id),
    id_base_product  integer not null references base_product (id),
    as_of            text not null,                -- признаки посчитаны по событиям до этого момента
    p_buy_7d         real check (p_buy_7d between 0 and 1),
    p_buy_14d        real check (p_buy_14d between 0 and 1),
    expected_days    real check (expected_days >= 0),
    lower_days       real check (lower_days >= 0),
    upper_days       real check (upper_days >= lower_days),
    p_spoil          real check (p_spoil between 0 and 1),
    unique (id_model_run, id_household, id_base_product, as_of)
);

create table recommendation (
    id               integer primary key,
    id_forecast      integer not null references forecast (id),
    id_household     integer not null references household (id),
    id_base_product  integer not null references base_product (id),
    shown_at         text not null,
    position         integer check (position > 0),
    threshold        real,
    action           text check (action in ('added', 'dismissed', 'ignored')),
    action_at        text
);

-- Только для синтетики: что генератор заложил в данные --------------------
-- В приложении этих таблиц нет. Нужны, чтобы сверять с ними выводы моделей.

create table sim_household_product (
    id_household       integer not null references household (id),
    id_base_product    integer not null references base_product (id),
    id_product         integer not null references product (id),   -- привычная фасовка
    rate_base_per_day  real not null check (rate_base_per_day > 0), -- до изменений поведения
    primary key (id_household, id_base_product)
);

create table sim_household_change (
    id            integer primary key,
    id_household  integer not null references household (id),
    changed_at    text not null,
    kind          text not null,
    detail        text not null                     -- JSON: множители, добавленные и убранные продукты
);

-- Индексы -------------------------------------------------------------------

create index product_base_idx on product (id_base_product);
create index member_household_idx on member (id_household);
create index trip_household_idx on shopping_trip (id_household, actual_at);
create index purchase_line_trip_idx on purchase_line (id_shopping_trip);
create index purchase_line_product_idx on purchase_line (id_product);
create index lot_household_idx on lot (id_household, added_at);
create index lot_event_lot_idx on lot_event (id_lot);
create index lot_event_history_idx on lot_event (id_household, id_base_product, occurred_at);
create index forecast_lookup_idx on forecast (id_household, id_base_product, as_of);
create index recommendation_household_idx on recommendation (id_household, shown_at);

-- Представления -------------------------------------------------------------

create view v_product as
select
    p.id              as id_product,
    p.name            as product,
    b.id              as id_base_product,
    b.name            as base_product,
    c.name            as category,
    b.dimension,
    p.pack_size,
    u.name            as unit,
    p.pack_size_base,
    p.price,
    p.price / p.pack_size_base as price_per_base,
    b.typical_shelf_days,
    b.is_perishable,
    s.name            as storage_place
from product p
join base_product b on b.id = p.id_base_product
join category c on c.id = b.id_category
join unit u on u.id = p.id_unit
join storage_place s on s.id = b.id_storage_place;

-- Покупки базового продукта по дням и интервал до следующей покупки (цикл повторной покупки).
-- days_to_next = null: следующей покупки в данных ещё нет.
create view v_purchase_cycle as
with daily as (
    select
        t.id_household,
        p.id_base_product,
        date(t.actual_at)     as purchase_date,
        sum(pl.quantity_base) as quantity_base,
        sum(pl.price)         as spent,
        max(pl.source = 'recommendation') as from_recommendation
    from purchase_line pl
    join shopping_trip t on t.id = pl.id_shopping_trip
    join product p on p.id = pl.id_product
    where t.actual_at is not null
    group by t.id_household, p.id_base_product, date(t.actual_at)
)
select
    daily.*,
    row_number() over w                    as purchase_n,
    lag(purchase_date) over w              as prev_purchase_date,
    lead(purchase_date) over w             as next_purchase_date,
    julianday(purchase_date) - julianday(lag(purchase_date) over w)  as days_since_prev,
    julianday(lead(purchase_date) over w) - julianday(purchase_date) as days_to_next
from daily
window w as (partition by id_household, id_base_product order by purchase_date);

-- Жизнь партии: сколько съели и выбросили, когда закончилась и чем.
create view v_lot_lifecycle as
with totals as (
    select
        e.id_lot,
        sum(e.quantity_base_delta) as remaining_base,
        sum(case when e.event_type = 'consumed' then -e.quantity_base_delta else 0 end) as consumed_base,
        sum(case when e.event_type = 'wasted' then -e.quantity_base_delta else 0 end)   as wasted_base,
        max(e.event_type = 'mistake') as is_mistake,
        max(case when e.event_type in ('consumed', 'wasted', 'mistake') then e.occurred_at end) as last_out_at,
        min(case when e.event_type = 'consumed' then e.occurred_at end) as first_consumed_at,
        max(julianday(e.recorded_at) - julianday(e.occurred_at)) as max_record_delay_days
    from lot_event e
    group by e.id_lot
)
select
    l.id as id_lot,
    l.id_household,
    p.id_base_product,
    l.id_product,
    l.source,
    l.added_at,
    l.best_before,
    l.quantity_base,
    t.consumed_base,
    t.wasted_base,
    t.remaining_base,
    case when t.remaining_base <= 1e-9 then t.last_out_at end as finished_at,
    case
        when t.is_mistake then 'mistake'
        when t.remaining_base > 1e-9 then 'open'
        when t.wasted_base = 0 then 'consumed'
        when t.consumed_base = 0 then 'wasted'
        else 'mixed'
    end as outcome,
    case when t.remaining_base <= 1e-9
         then julianday(t.last_out_at) - julianday(l.added_at) end as days_to_finish,
    case when t.remaining_base <= 1e-9 and julianday(t.last_out_at) > julianday(l.added_at)
         then t.consumed_base / (julianday(t.last_out_at) - julianday(l.added_at)) end as consumed_per_day,
    t.first_consumed_at,
    t.max_record_delay_days
from lot l
join product p on p.id = l.id_product
left join totals t on t.id_lot = l.id;

-- Остаток базового продукта после каждого дня, в который он менялся.
-- Остаток на произвольную дату — последняя строка с day <= этой даты.
create view v_stock_by_day as
with changes as (
    select id_household, id_base_product, date(occurred_at) as day, sum(quantity_base_delta) as delta_base
    from lot_event
    group by id_household, id_base_product, date(occurred_at)
)
select
    changes.*,
    sum(delta_base) over (
        partition by id_household, id_base_product order by day rows unbounded preceding
    ) as stock_base
from changes;

-- Потребление и списания аккаунта по дням и категориям, в рублях по каталожной цене.
create view v_household_daily as
select
    e.id_household,
    date(e.occurred_at) as day,
    c.name              as category,
    sum(case when e.event_type = 'consumed' then 1 else 0 end) as consumed_events,
    sum(case when e.event_type = 'wasted' then 1 else 0 end)   as wasted_events,
    sum(case when e.event_type = 'consumed' then -e.quantity_base_delta * p.price / p.pack_size_base else 0 end) as consumed_rub,
    sum(case when e.event_type = 'wasted' then -e.quantity_base_delta * p.price / p.pack_size_base else 0 end)   as wasted_rub
from lot_event e
join lot l on l.id = e.id_lot
join product p on p.id = l.id_product
join base_product b on b.id = p.id_base_product
join category c on c.id = b.id_category
where e.event_type in ('consumed', 'wasted')
group by e.id_household, date(e.occurred_at), c.name;

-- Поход в магазин: план против факта, размер корзины, сумма.
create view v_trip as
select
    t.id as id_shopping_trip,
    t.id_household,
    t.planned_date,
    t.actual_at,
    julianday(date(t.actual_at)) - julianday(t.planned_date) as days_late,
    julianday(date(t.actual_at)) - julianday(date(lag(t.actual_at) over (
        partition by t.id_household order by t.actual_at
    ))) as days_since_prev_trip,
    count(pl.id)    as lines,
    sum(pl.price)   as spent,
    sum(pl.source = 'recommendation') as recommended_lines
from shopping_trip t
left join purchase_line pl on pl.id_shopping_trip = t.id
where t.actual_at is not null
group by t.id;
