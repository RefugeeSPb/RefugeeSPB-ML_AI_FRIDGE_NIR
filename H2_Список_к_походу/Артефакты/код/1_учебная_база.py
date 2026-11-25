"""1) Учебная база H2.

Та же ``fridge_lab.sqlite``, что у H1: 200 семей, семя генератора 42.
Кандидаты — продукты семьи за последние 60 дней. Прогноз через 3 часа
после похода. Метка — продукт в следующей корзине.

Повтор на 500 семьях читает ``fridge_lab_v2.sqlite``, семя 43.
Сборка строк — ``forecast.build_rows``.
"""

from pathlib import Path

import forecast

HERE = Path(__file__).resolve().parent
DB_200 = HERE.parents[2] / "Общие_артефакты" / "данные" / "fridge_lab.sqlite"
DB_500 = HERE.parents[2] / "Общие_артефакты" / "данные" / "fridge_lab_v2.sqlite"
HORIZON = 14
TRAIN = ("2025-10-01", "2026-06-01")
VAL = ("2026-06-01", "2026-08-01")
TEST = ("2026-08-01", "2026-10-01")


def build(data, start, end):
    """Собрать строки «поход × продукт» на одном отрезке.

    Окно метки — ``HORIZON`` суток. В счёт входят походы, у которых
    это окно целиком лежит до ``end``. Поэтому при тесте до
    1 октября 2026 последний поход в счёте — 16 сентября.

    Args:
        data (dict): Результат ``forecast.load``.
        start: Начало отрезка походов, включая эту дату.
        end: Конец отрезка. Походы на эту дату в метку не входят.

    Returns:
        pandas.DataFrame: Строки с ``pair_visit_share``,
        ``in_next_basket`` и столбцами ``forecast.FEATURES``.

    Examples:
        frame, _stats = build(data, *TEST)
    """
    frame, _stats = forecast.build_rows(data, start, end, HORIZON)
    return frame


def main():
    """Напечатать пути баз и границы трёх отрезков.

    Базу не открывает и строки не собирает.

    Examples:
        python 1_учебная_база.py
    """
    print(f"200 семей: {DB_200}")
    print(f"500 семей, семя 43: {DB_500}")
    print(f"кандидаты: {forecast.WINDOW_DAYS} дней, прогноз +{forecast.AS_OF_HOURS} ч")
    print(f"учёба {TRAIN[0]} — {TRAIN[1]}")
    print(f"валидация {VAL[0]} — {VAL[1]}")
    print(f"тест {TEST[0]} — {TEST[1]}, горизонт метки {HORIZON} суток")


if __name__ == "__main__":
    main()
