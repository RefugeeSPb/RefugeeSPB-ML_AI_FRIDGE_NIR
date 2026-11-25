"""1) Код для определения пар с не меньше чем 3 покупками.

Единица счёта — семья × продукт, не семья целиком. В полосу
входит пара, у которой уже есть не меньше трёх покупок
(``pair_n >= 3``) и известен день следующей покупки в пределах
14 суток. База — ``fridge_lab_v3.sqlite``, 500 семей, семя 44.

Повтор целиком: ``python interval_h3b.py``.
"""

from pathlib import Path

import forecast

HERE = Path(__file__).resolve().parent
DB = HERE.parents[2] / "Общие_артефакты" / "данные" / "fridge_lab_v3.sqlite"
HORIZON = 14
MIN_PURCHASES = 3
TRAIN = ("2025-10-01", "2026-06-01")
VAL = ("2026-06-01", "2026-08-01")
TEST = ("2026-08-01", "2026-10-01")


def with_min_purchases(data, start, end, min_n=MIN_PURCHASES):
    """Оставить пары семья × продукт с не меньше ``min_n`` покупок.

    Args:
        data (dict): Результат ``forecast.load``.
        start: Начало отрезка походов, включая эту дату.
        end: Конец отрезка. Походы на эту дату в метку не входят.
        min_n: Минимум покупок пары. По умолчанию 3.

    Returns:
        pandas.DataFrame: Строки с известным ``days_to_purchase``.

    Examples:
        test = with_min_purchases(data, *TEST)
    """
    frame, _stats = forecast.build_rows(data, start, end, HORIZON)
    return frame[(frame["pair_n"] >= min_n) & frame["days_to_purchase"].notna()].copy()


def main():
    """Напечатать путь базы, порог покупок и границы отрезков.

    Базу не открывает и строки не собирает.

    Examples:
        python 1_семьи_от_трех_покупок.py
    """
    print(f"500 семей, семя 44: {DB}")
    print(f"пара семья × продукт, покупок >= {MIN_PURCHASES}, горизонт {HORIZON} суток")
    print(f"учёба {TRAIN[0]} — {TRAIN[1]}")
    print(f"валидация {VAL[0]} — {VAL[1]}")
    print(f"тест {TEST[0]} — {TEST[1]}")


if __name__ == "__main__":
    main()
