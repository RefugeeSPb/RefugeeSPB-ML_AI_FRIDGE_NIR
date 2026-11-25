"""Синтетические домохозяйства для fridge_lab.sqlite.

Каждое домохозяйство живёт по дням:
  * у него есть привычная корзина: базовые продукты и скорость их расхода,
    зависящая от состава семьи;
  * в магазин ходят примерно с привычным интервалом, дату следующего похода
    планируют заранее; если кончилось молоко или хлеб — бывает внеплановый поход;
  * в походе докупают то, чего не хватит до следующего похода, иногда что-то
    забывают и покупают лишнее под настроение;
  * едят из партии с самым ранним сроком; просроченное чаще всего выбрасывают;
  * в приложении отмечают не всё и не сразу — у каждого своя аккуратность;
  * у части домохозяйств в какой-то момент меняется поведение.

В базу попадает то, что увидело бы приложение. Что генератор заложил на самом
деле, записано в sim_household_product и sim_household_change.

Использование:
    python build_db.py && python generate.py --households 200 --days 365
"""

import argparse
import json
import math
import sqlite3
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# Доля продуктов категории в привычной корзине и сколько дней уходит
# на одну упаковку у пары взрослых.
CATEGORY_PROFILE = {
    "Молочные": (0.45, 5),
    "Мясо и птица": (0.35, 6),
    "Рыба и морепродукты": (0.20, 8),
    "Яйца": (0.60, 9),
    "Хлеб и выпечка": (0.45, 3),
    "Овощи": (0.45, 7),
    "Зелень": (0.35, 5),
    "Фрукты и ягоды": (0.35, 6),
    "Бакалея": (0.35, 45),
    "Консервы и соусы": (0.25, 40),
    "Напитки": (0.30, 7),
    "Заморозка": (0.25, 20),
    "Сладкое и снеки": (0.30, 10),
    "Детское питание": (0.70, 3),
    "Готовая еда": (0.15, 2),
}
DAYS_PER_PACK = {
    "соль": 150, "сода": 200, "перец чёрный молотый": 150, "дрожжи сухие": 20,
    "сахар": 25, "мука": 40, "масло подсолнечное": 25, "масло оливковое": 40,
    "чай чёрный": 40, "кофе молотый": 20, "кофе растворимый": 30, "мёд": 40,
}
ESSENTIALS = {
    "молоко", "яйца куриные", "хлеб белый", "хлеб чёрный", "батон", "картофель",
    "лук репчатый", "морковь", "сахар", "соль", "масло подсолнечное", "чай чёрный",
    "масло сливочное", "сыр твёрдый", "макароны", "гречка", "кефир", "сметана",
    "яблоки", "бананы",
}
# Без этих категорий не дотягивают до следующего похода: кончились — идут в магазин
TOP_UP_CATEGORIES = {"Молочные", "Хлеб и выпечка", "Яйца", "Детское питание"}
BABY_CATEGORY = "Детское питание"

SEGMENTS = {
    "один": dict(share=0.30, size=(1, 1), adults=1, trip_days=6.0,
                 categories={"Готовая еда": 2.5, "Бакалея": 0.8}),
    "пара": dict(share=0.30, size=(2, 2), adults=2, trip_days=4.5, categories={}),
    "семья с детьми": dict(share=0.25, size=(3, 5), adults=2, trip_days=3.5,
                           categories={"Сладкое и снеки": 1.4, "Молочные": 1.2}),
    "семья с малышом": dict(share=0.08, size=(3, 4), adults=2, trip_days=3.0, baby=True,
                            categories={"Молочные": 1.2, "Готовая еда": 0.5}),
    "пенсионеры": dict(share=0.07, size=(1, 2), adults=1, trip_days=3.0,
                       categories={"Сладкое и снеки": 0.6, "Готовая еда": 0.3, "Бакалея": 1.3,
                                   "Напитки": 0.6, "Заморозка": 0.6}),
}

# Масштаб привычной корзины: доли CATEGORY_PROFILE умножаются на него
BASKET_SCALE = 0.65
CHANGES = ["меньше дома", "диета", "больше едоков", "сменили рацион", "реже ходят в магазин"]
CHANGE_SHARE = 0.3
FREEZE_CATEGORIES = {"Мясо и птица", "Рыба и морепродукты"}
FROZEN_DAYS = 90
FORGET_SHARE = 0.06
IMPULSE_PER_TRIP = 0.8
EPS = 1e-6


class Catalog:
    def __init__(self, connection):
        self.bases = {}
        for id_, name, category, dimension, shelf, perishable, place in connection.execute(
            """select b.id, b.name, c.name, b.dimension, b.typical_shelf_days, b.is_perishable,
                      b.id_storage_place
               from base_product b join category c on c.id = b.id_category"""
        ):
            self.bases[id_] = dict(name=name, category=category, dimension=dimension,
                                   shelf=shelf, perishable=bool(perishable), place=place)
        self.products = {}
        self.packs = {id_: [] for id_ in self.bases}
        for id_, base, pack_base, price in connection.execute(
            "select id, id_base_product, pack_size_base, price from product order by id"
        ):
            self.products[id_] = dict(base=base, pack=pack_base, price=price)
            self.packs[base].append(id_)
        self.places = {name: id_ for id_, name in connection.execute("select id, name from storage_place")}


class Lot:
    __slots__ = ("id", "base", "true_left", "app_left", "pending", "best_before", "added", "log_every")

    def __init__(self, id_, base, quantity, best_before, added, log_every):
        self.id, self.base = id_, base
        self.true_left = quantity      # физически осталось
        self.app_left = 0.0            # осталось по данным приложения, растёт с событием added
        self.pending = 0.0             # съели, но ещё не отметили
        self.best_before, self.added = best_before, added
        self.log_every = log_every     # отмечают, когда съели столько с прошлой отметки


class Generator:
    def __init__(self, catalog, start, days, seed, test_span=None, test_changes=0):
        self.catalog, self.start, self.days = catalog, start, days
        self.test_span = test_span
        self.test_left = test_changes
        self.rng = np.random.default_rng(seed)
        self.rows = {table: [] for table in INSERTS}
        self.counter = Counter()
        # Популярность одна на всех: молоко берут почти все, булгур — немногие
        self.popularity = {
            id_: float(self.rng.lognormal(0, 0.5)) * (2.2 if base["name"] in ESSENTIALS else 1.0)
            for id_, base in catalog.bases.items()
        }

    def new_id(self, table):
        self.counter[table] += 1
        return self.counter[table]

    def day(self, index):
        return self.start + timedelta(days=int(index))

    def moment(self, index, after=None, earliest=7, latest=23):
        """Случайное время дня index, но позже after."""
        stamp = datetime.combine(self.day(index), datetime.min.time()) + timedelta(
            hours=float(self.rng.uniform(earliest, latest)))
        if after is not None and stamp <= after:
            stamp = after + timedelta(minutes=float(self.rng.uniform(5, 60)))
        return stamp.replace(microsecond=0)

    @staticmethod
    def text(stamp):
        return stamp.strftime("%Y-%m-%d %H:%M:%S")

    def daily_rate(self, base_id, product_id, size, in_category=1):
        """Расход в базовых единицах в день.

        Похожие продукты делят спрос: кто берёт и кефир, и ряженку, и йогурт,
        пьёт каждого меньше. Отсюда деление на корень из числа продуктов категории.
        """
        base = self.catalog.bases[base_id]
        days = DAYS_PER_PACK.get(base["name"], CATEGORY_PROFILE[base["category"]][1])
        pack = self.catalog.products[product_id]["pack"]
        crowd = 1 / math.sqrt(max(1.0, in_category / 2))
        return pack / days * (size / 2) ** 0.75 * crowd * float(self.rng.lognormal(0, 0.35))

    def price(self, product_id, index):
        """Цена упаковки в день index: инфляция ~8% в год, акции, разброс магазинов."""
        price = self.catalog.products[product_id]["price"] * (1 + 0.08 * index / 365)
        if self.rng.random() < 0.2:
            price *= self.rng.uniform(0.7, 0.9)
        return price * self.rng.uniform(0.97, 1.05)

    def run(self, households):
        names = list(SEGMENTS)
        shares = np.array([SEGMENTS[name]["share"] for name in names])
        for _ in range(households):
            Household(self, names[self.rng.choice(len(names), p=shares / shares.sum())]).simulate()
        return self.rows


class Household:
    def __init__(self, gen, segment):
        self.gen, self.rng, self.catalog = gen, gen.rng, gen.catalog
        profile = SEGMENTS[segment]
        self.id = gen.new_id("household")
        self.size = int(self.rng.integers(profile["size"][0], profile["size"][1] + 1))
        self.has_baby = profile.get("baby", False)
        self.trip_days = profile["trip_days"] * self.rng.uniform(0.7, 1.3)
        self.diligence = float(self.rng.beta(4, 2))          # доля отметок, сделанных вовремя
        self.log_share = float(self.rng.choice([1.0, 0.5, 0.34]))  # отмечают целиком, половинами, третями
        self.receipt_share = float(self.rng.uniform(0, 0.8))
        self.top_up_chance = float(self.rng.uniform(0.1, 0.5))
        self.today = self.first_day = int(self.rng.integers(0, 90))
        self.multiplier = {}           # категория, «*» или «trip» → множитель после изменения
        self.lots = {}                 # базовый продукт → открытые партии по сроку годности
        self.occasional = {}           # куплено под настроение → скорость расхода
        self.cleanups = []             # (день, партия): тогда закроют забытую партию
        self.top_up = None             # (день, продукты) внепланового похода

        created = gen.moment(self.first_day)
        gen.rows["household"].append((self.id, gen.text(created), self.size, segment))
        self.members = []
        for _ in range(min(self.size, profile["adults"] + int(self.rng.random() < 0.15))):
            self.members.append(gen.new_id("member"))
            gen.rows["member"].append((self.members[-1], self.id, gen.text(created)))

        chosen = []
        for base_id, base in self.catalog.bases.items():
            if not self.allowed(base_id):
                continue
            chance = BASKET_SCALE * CATEGORY_PROFILE[base["category"]][0] \
                * profile["categories"].get(base["category"], 1.0) * gen.popularity[base_id]
            if self.rng.random() < min(0.95, chance):
                chosen.append(base_id)
        self.regular = {}
        for base_id in chosen:
            self.add_regular(base_id, record=True, basket=chosen)

        latest = gen.days - 30
        self.change_day = None
        span = gen.test_span
        if span and gen.test_left > 0:
            start = max(self.first_day + 90, span[0])
            end = min(span[1], latest)
            if start < end:
                self.change_day = int(self.rng.integers(start, end))
                gen.test_left -= 1
        elif self.first_day + 90 < latest and self.rng.random() < CHANGE_SHARE:
            self.change_day = int(self.rng.integers(self.first_day + 90, latest))

    def allowed(self, base_id):
        return self.has_baby or self.catalog.bases[base_id]["category"] != BABY_CATEGORY

    def add_regular(self, base_id, record=False, basket=None):
        packs = self.catalog.packs[base_id]
        # Большие семьи чаще берут крупную фасовку, одинокие — мелкую
        weights = np.array([self.catalog.products[p]["pack"] for p in packs]) ** ((self.size - 2) / 4)
        product = packs[self.rng.choice(len(packs), p=weights / weights.sum())]
        category = self.catalog.bases[base_id]["category"]
        basket = list(self.regular) if basket is None else basket
        in_category = sum(self.catalog.bases[b]["category"] == category for b in basket)
        rate = self.gen.daily_rate(base_id, product, self.size, in_category)
        self.regular[base_id] = dict(product=product, rate=rate)
        if record:
            self.gen.rows["sim_household_product"].append((self.id, base_id, product, rate))

    def rate(self, base_id):
        category = self.catalog.bases[base_id]["category"]
        return self.regular[base_id]["rate"] * self.multiplier.get(category, 1.0) * self.multiplier.get("*", 1.0)

    # Отметки в приложении -----------------------------------------------------

    def event(self, lot, kind, delta, occurred, reason=None, delay_days=0.0):
        # Остаток в приложении — сумма записанных дельт, поэтому копим уже округлённую
        delta = round(delta, 3)
        if delta == 0:
            return
        recorded = occurred + timedelta(days=delay_days, minutes=float(self.rng.uniform(0, 30)))
        self.gen.rows["lot_event"].append((
            self.gen.new_id("lot_event"), lot.id, self.id, lot.base, kind, delta, reason,
            self.gen.text(occurred), self.gen.text(recorded.replace(microsecond=0)),
            self.members[int(self.rng.integers(len(self.members)))],
        ))
        lot.app_left += delta

    def write_down(self, lot, kind, amount, occurred, reason=None, final=False):
        """Отметить в приложении: вовремя, с опозданием или забыть.

        Забытое потребление попадёт в следующую отметку. Если забыли закрыть
        партию, её закроют при уборке кладовой через 2–8 недель.
        """
        roll = self.rng.random()
        if roll >= self.diligence + (1 - self.diligence) * 0.6:
            if final:
                self.cleanups.append((self.today + int(self.rng.integers(14, 61)), lot))
            return False
        delay = 0.0 if roll < self.diligence else float(self.rng.uniform(1, 5))
        self.event(lot, kind, -amount, occurred, reason, delay)
        return True

    # Партии -------------------------------------------------------------------

    def new_lot(self, base_id, product, quantity, added, purchase_line=None, source="checkout",
                shelf_share=(0.8, 1.1)):
        base = self.catalog.bases[base_id]
        quantity = round(quantity, 3)
        place = base["place"]
        shelf = base["shelf"] * self.rng.uniform(*shelf_share)
        if source == "checkout" and base["category"] in FREEZE_CATEGORIES \
                and base["perishable"] and self.rng.random() < 0.25:
            place, shelf = self.catalog.places["Морозилка"], FROZEN_DAYS
        best_before = self.today + max(1, round(shelf))
        lot = Lot(self.gen.new_id("lot"), base_id, quantity, best_before, added, quantity * self.log_share)
        self.gen.rows["lot"].append((
            lot.id, self.id, product, purchase_line, place, self.gen.text(added), quantity,
            self.gen.day(best_before).isoformat(), source,
        ))
        self.event(lot, "added", quantity, added)
        self.lots.setdefault(base_id, []).append(lot)
        self.lots[base_id].sort(key=lambda item: item.best_before)

    def close(self, lot):
        self.lots[lot.base].remove(lot)

    def stock(self, base_id):
        return sum(lot.true_left for lot in self.lots.get(base_id, []))

    # Поход в магазин ----------------------------------------------------------

    def line_source(self):
        if self.rng.random() < self.receipt_share:
            return "receipt"
        return "voice" if self.rng.random() < 0.05 else "manual"

    def buy(self, trip_id, at, base_id, product, packs):
        line = self.gen.new_id("purchase_line")
        quantity = packs * self.catalog.products[product]["pack"]
        self.gen.rows["purchase_line"].append((
            line, trip_id, product, packs, quantity,
            round(packs * self.gen.price(product, self.today), 2), self.line_source(), None,
        ))
        added = at + timedelta(minutes=float(self.rng.uniform(20, 90)))
        self.new_lot(base_id, product, quantity, added.replace(microsecond=0), purchase_line=line)

    def trip(self, planned_day, created_at, only=None):
        """Докупить то, чего не хватит до следующего похода (или только only)."""
        at = self.gen.moment(self.today, earliest=9, latest=21)
        trip_id = self.gen.new_id("shopping_trip")
        self.gen.rows["shopping_trip"].append((
            trip_id, self.id, None if planned_day is None else self.gen.day(planned_day).isoformat(),
            self.gen.text(at), self.gen.text(min(created_at, at)),
        ))
        planned = only is None
        interval = self.next_interval() if planned else max(1, self.next_trip - self.today)

        for base_id, info in list(self.regular.items()):
            if not planned and base_id not in only:
                continue
            # Скоропортящееся берут впритык, долгое хранение — с запасом
            perishable = self.catalog.bases[base_id]["perishable"]
            need = self.rate(base_id) * (interval + (0.5 if perishable else 1.5))
            have = self.stock(base_id)
            if have >= need or self.rng.random() < FORGET_SHARE:
                continue
            pack = self.catalog.products[info["product"]]["pack"]
            extra = self.rng.uniform(1.0, 1.15) if perishable else self.rng.uniform(1.0, 1.6)
            packs = max(1, math.ceil((need * extra - have) / pack))
            if self.catalog.bases[base_id]["shelf"] >= 90:
                packs = min(packs, 3)
            self.buy(trip_id, at, base_id, info["product"], packs)

        if not planned:
            return
        for _ in range(self.rng.poisson(IMPULSE_PER_TRIP)):
            candidates = [b for b in self.catalog.bases if b not in self.regular and self.allowed(b)]
            weights = np.array([self.gen.popularity[b] for b in candidates])
            base_id = candidates[self.rng.choice(len(candidates), p=weights / weights.sum())]
            packs = self.catalog.packs[base_id]
            product = packs[int(self.rng.integers(len(packs)))]
            self.occasional[base_id] = self.gen.daily_rate(base_id, product, self.size) * self.rng.uniform(0.5, 1.5)
            self.buy(trip_id, at, base_id, product, 1)

        # План следующего похода; фактически пойдут на день-два раньше или позже
        self.planned = self.today + interval
        self.planned_at = at + timedelta(hours=float(self.rng.uniform(0.5, 3)))
        self.next_trip = max(self.today + 1, self.planned + int(self.rng.choice([-1, 0, 0, 0, 1, 1, 2])))

    def next_interval(self):
        return max(1, round(self.trip_days * self.multiplier.get("trip", 1.0) * self.rng.uniform(0.6, 1.4)))

    # Потребление и порча ------------------------------------------------------

    def daily_amount(self, base_id, rate):
        if self.catalog.bases[base_id]["dimension"] == "count":
            return float(self.rng.poisson(rate))
        return rate * float(self.rng.gamma(2.0, 0.5))

    def consume(self, base_id, amount):
        for lot in list(self.lots.get(base_id, [])):
            if amount <= EPS:
                break
            take = min(amount, lot.true_left)
            lot.true_left -= take
            lot.pending += take
            amount -= take
            finished = lot.true_left <= EPS
            if finished or lot.pending >= lot.log_every - EPS:
                occurred = self.gen.moment(self.today, after=lot.added)
                # «Закончилось» в приложении списывает весь остаток партии
                recorded = lot.app_left if finished else lot.pending
                if self.write_down(lot, "consumed", recorded, occurred, final=finished):
                    lot.pending = 0.0
            if finished:
                self.close(lot)
        if base_id in self.occasional and not self.lots.get(base_id):
            del self.occasional[base_id]

    def spoil(self):
        for base_id, lots in list(self.lots.items()):
            perishable = self.catalog.bases[base_id]["perishable"]
            for lot in list(lots):
                if self.today > lot.best_before:
                    chance, reason = (0.6 if perishable else 0.3), "expired"
                elif perishable:
                    chance, reason = 0.004, "spoiled"
                else:
                    continue
                if self.rng.random() >= chance:
                    continue
                occurred = self.gen.moment(self.today, after=lot.added)
                if lot.pending > EPS and self.write_down(lot, "consumed", lot.pending, occurred):
                    lot.pending = 0.0
                lot.true_left = 0.0
                # При списании приложение подставляет весь остаток (BL-03)
                if lot.app_left > EPS:
                    self.write_down(lot, "wasted", lot.app_left, occurred, reason=reason, final=True)
                self.close(lot)

    def tidy_up(self):
        """Закрыть партии, которые давно кончились, но остались открытыми в приложении."""
        due = [lot for day, lot in self.cleanups if day <= self.today]
        self.cleanups = [(day, lot) for day, lot in self.cleanups if day > self.today]
        for lot in due:
            if lot.app_left > EPS:
                self.event(lot, "corrected", -lot.app_left, self.gen.moment(self.today, after=lot.added))

    # Изменение поведения ------------------------------------------------------

    def change(self):
        kind = CHANGES[int(self.rng.integers(len(CHANGES)))]
        if kind == "меньше дома":
            self.multiplier["*"] = round(float(self.rng.uniform(0.4, 0.7)), 2)
            detail = {"расход": self.multiplier["*"]}
        elif kind == "больше едоков":
            self.multiplier["*"] = round(float(self.rng.uniform(1.3, 1.7)), 2)
            detail = {"расход": self.multiplier["*"]}
        elif kind == "диета":
            detail = {"Сладкое и снеки": 0.3, "Хлеб и выпечка": 0.4, "Овощи": 1.5, "Фрукты и ягоды": 1.5}
            self.multiplier.update(detail)
        elif kind == "реже ходят в магазин":
            self.multiplier["trip"] = round(float(self.rng.uniform(1.5, 2.0)), 2)
            detail = {"интервал походов": self.multiplier["trip"]}
        else:
            dropped = [int(b) for b in self.rng.permutation(list(self.regular))[:3]]
            for base_id in dropped:
                del self.regular[base_id]
            fresh = [int(b) for b in self.rng.permutation(list(self.catalog.bases))
                     if int(b) not in self.regular and self.allowed(int(b))][:4]
            for base_id in fresh:
                self.add_regular(base_id)
            detail = {"убрали": [self.catalog.bases[b]["name"] for b in dropped],
                      "добавили": [self.catalog.bases[b]["name"] for b in fresh]}
        self.gen.rows["sim_household_change"].append((
            self.gen.new_id("sim_household_change"), self.id,
            self.gen.text(self.gen.moment(self.today)), kind, json.dumps(detail, ensure_ascii=False),
        ))

    # День за днём -------------------------------------------------------------

    def onboarding(self):
        """При регистрации вносят то, что уже лежит дома: початое, со сроком на исходе."""
        added = self.gen.moment(self.today)
        for base_id, info in self.regular.items():
            if self.rng.random() >= 0.6:
                continue
            quantity = self.catalog.products[info["product"]]["pack"] * self.rng.uniform(0.2, 1.0)
            if self.catalog.bases[base_id]["dimension"] == "count":
                quantity = max(1.0, round(quantity))
            self.new_lot(base_id, info["product"], quantity, added, source="manual", shelf_share=(0.2, 1.0))
        self.planned = self.today + int(self.rng.integers(0, max(1, round(self.trip_days))))
        self.planned_at = added
        self.next_trip = self.planned

    def simulate(self):
        self.onboarding()
        for self.today in range(self.first_day, self.gen.days):
            if self.today == self.change_day:
                self.change()
            if self.today >= self.next_trip:
                self.trip(self.planned, self.planned_at)
            elif self.top_up and self.top_up[0] == self.today:
                self.trip(None, self.gen.moment(self.today - 1), only=self.top_up[1])
            self.top_up = None

            was_empty = self.empty_staples()
            for base_id in list(self.regular):
                self.consume(base_id, self.daily_amount(base_id, self.rate(base_id)))
            for base_id, rate in list(self.occasional.items()):
                self.consume(base_id, self.daily_amount(base_id, rate))
            self.spoil()
            self.tidy_up()

            # Решают «сбегать в магазин» один раз — в день, когда продукт кончился
            ran_out = self.empty_staples() - was_empty
            if ran_out and self.next_trip - self.today >= 2 and self.rng.random() < self.top_up_chance:
                self.top_up = (self.today + 1, self.empty_staples())

    def empty_staples(self):
        return {b for b in self.regular
                if self.catalog.bases[b]["category"] in TOP_UP_CATEGORIES and self.stock(b) <= EPS}


INSERTS = {
    "household": "insert into household (id, created_at, size, segment) values (?, ?, ?, ?)",
    "member": "insert into member (id, id_household, joined_at) values (?, ?, ?)",
    "shopping_trip": "insert into shopping_trip (id, id_household, planned_date, actual_at, created_at) "
                     "values (?, ?, ?, ?, ?)",
    "purchase_line": "insert into purchase_line (id, id_shopping_trip, id_product, packs, quantity_base, "
                     "price, source, id_recommendation) values (?, ?, ?, ?, ?, ?, ?, ?)",
    "lot": "insert into lot (id, id_household, id_product, id_purchase_line, id_storage_place, added_at, "
           "quantity_base, best_before, source) values (?, ?, ?, ?, ?, ?, ?, ?, ?)",
    "lot_event": "insert into lot_event (id, id_lot, id_household, id_base_product, event_type, "
                 "quantity_base_delta, waste_reason, occurred_at, recorded_at, id_member) "
                 "values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
    "sim_household_product": "insert into sim_household_product values (?, ?, ?, ?)",
    "sim_household_change": "insert into sim_household_change values (?, ?, ?, ?, ?)",
}
CLEAR_ORDER = ["recommendation", "forecast", "model_run", "sim_household_change", "sim_household_product",
               "lot_event", "lot", "purchase_line", "shopping_trip", "member", "household"]


def summary(connection):
    one = lambda sql: connection.execute(sql).fetchone()[0]
    rows = lambda sql: connection.execute(sql).fetchall()
    print("Строк:", {table: one(f"select count(*) from {table}") for table in INSERTS})
    print("Сегменты:", dict(rows("select segment, count(*) from household group by 1")))
    print("Походов в неделю:", round(one(
        """select avg(n / weeks) from (
               select count(*) as n,
                      max(1, (julianday(max(actual_at)) - julianday(min(actual_at))) / 7.0) as weeks
               from shopping_trip group by id_household)"""), 2))
    print("Доля внеплановых походов:", round(one("select avg(planned_date is null) from shopping_trip"), 3))
    print("Строк в чеке, в среднем:", round(one("select avg(lines) from v_trip"), 1))
    print("Итог партий:", dict(rows("select outcome, count(*) from v_lot_lifecycle group by 1")))
    consumed, wasted = rows("select sum(consumed_rub), sum(wasted_rub) from v_household_daily")[0]
    print("Доля выброшенного, в рублях:", round(wasted / (consumed + wasted), 3))
    print("Отметок позже чем через сутки:", round(one(
        """select avg(julianday(recorded_at) - julianday(occurred_at) > 1)
           from lot_event where event_type <> 'added'"""), 3))
    print("Расход в месяц на домохозяйство, руб.:", round(one(
        """select sum(spent) / (count(distinct id_household) * 12.0) from v_trip"""), 0))
    print("Медиана дней до повторной покупки:")
    cycles = rows("""select c.name, pc.days_to_next from v_purchase_cycle pc
                     join base_product b on b.id = pc.id_base_product
                     join category c on c.id = b.id_category
                     where pc.days_to_next is not null""")
    by_category = {}
    for category, days in cycles:
        by_category.setdefault(category, []).append(days)
    for category, values in sorted(by_category.items(), key=lambda item: np.median(item[1])):
        print(f"  {category:<22} {np.median(values):5.1f}   циклов: {len(values)}")
    print("Изменения поведения:", dict(rows("select kind, count(*) from sim_household_change group by 1")))


def main():
    parser = argparse.ArgumentParser(description="Синтетические домохозяйства для fridge_lab.sqlite")
    parser.add_argument("--db", type=Path, default=HERE / "fridge_lab.sqlite")
    parser.add_argument("--households", type=int, default=200)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2025, 10, 1))
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-from", type=date.fromisoformat, default=None)
    parser.add_argument("--test-until", type=date.fromisoformat, default=None)
    parser.add_argument("--test-changes", type=int, default=0)
    args = parser.parse_args()
    test_span = None
    if args.test_changes:
        if args.test_from is None or args.test_until is None:
            raise SystemExit("Для --test-changes нужны --test-from и --test-until")
        test_span = (
            (args.test_from - args.start).days,
            (args.test_until - args.start).days,
        )
    if not args.db.exists():
        raise SystemExit(f"Нет базы {args.db}: сначала python build_db.py")

    connection = sqlite3.connect(args.db)
    connection.execute("pragma foreign_keys = on")
    generator = Generator(
        Catalog(connection), args.start, args.days, args.seed, test_span, args.test_changes,
    )
    rows = generator.run(args.households)
    if args.test_changes and generator.test_left:
        raise SystemExit(f"В тест попало смен меньше, чем просили: не хватило {generator.test_left}")
    with connection:
        for table in CLEAR_ORDER:
            connection.execute(f"delete from {table}")
        for table, sql in INSERTS.items():
            connection.executemany(sql, rows[table])
    assert not connection.execute("pragma foreign_key_check").fetchall()
    summary(connection)
    connection.close()


if __name__ == "__main__":
    main()
