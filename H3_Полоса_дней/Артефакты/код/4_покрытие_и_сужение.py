"""4) Расчёт покрытия и сужения.

Сравнение окна по F_d с полосой ошибок. Оба порога записаны
9 ноября до сборки базы с семенем 44. Модели этот файл не учит:
читает ``данные/метрики/полоса_h3b.csv``, пишет
``результаты/полоса.csv``.

500 семей: ошибки 0,794 / 5,59 дня, F_d 0,827 / 4,38 дня,
короче на 21,6%.
"""

from pathlib import Path

import pandas as pd

import eval_h3

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "данные" / "метрики" / "полоса_h3b.csv"
OUT = HERE.parent / "результаты" / "полоса.csv"
COVER_MIN = 0.78
NARROWER_MIN = 0.10


def narrower(base_width, model_width):
    """Вернуть относительное сужение окна против базы.

    Args:
        base_width: Средняя ширина полосы ошибок.
        model_width: Средняя ширина окна по F_d.

    Returns:
        float: ``(base_width - model_width) / base_width``.

    Examples:
        >>> round(narrower(5.59, 4.38), 3)
        0.216
    """
    return (base_width - model_width) / base_width


def compare(cover, base_width, model_width):
    """Вернуть сужение и вердикт двух порогов.

    Подтверждена, если покрытие не ниже ``COVER_MIN`` и сужение
    не меньше ``NARROWER_MIN``.

    Args:
        cover: Покрытие окна по F_d.
        base_width: Ширина полосы ошибок.
        model_width: Ширина окна по F_d.

    Returns:
        tuple: ``(shrink, verdict)``. ``verdict`` — строка
        «подтверждена» или «не подтверждена».

    Examples:
        >>> shrink, verdict = compare(0.827, 5.59, 4.38)
        >>> round(shrink, 3), verdict
        (0.216, 'подтверждена')
    """
    shrink = narrower(base_width, model_width)
    passed = cover >= COVER_MIN and shrink >= NARROWER_MIN
    verdict = "подтверждена" if passed else "не подтверждена"
    return shrink, verdict


def main():
    """Записать вердикт H3 по сохранённой таблице. Сеть не учит.

    Notes:
        Каталог ``результаты`` создаётся при отсутствии.
        Файл ``полоса.csv`` перезаписывается.

    Examples:
        python 4_покрытие_и_сужение.py
    """
    row = eval_h3.load_official(DATA)
    result = eval_h3.verdict(row)
    out = pd.DataFrame(
        [
            {
                "база": row["база"],
                "строк": int(row["строк"]),
                "ошибки_покрытие": round(result["ошибки_покрытие"], 3),
                "ошибки_ширина": round(result["ошибки_ширина"], 2),
                "Fd_покрытие": round(result["покрытие"], 3),
                "Fd_ширина": round(result["Fd_ширина"], 2),
                "сужение": round(result["сужение"], 3),
                "порог_покрытия": COVER_MIN,
                "порог_сужения": NARROWER_MIN,
                "вердикт": result["вердикт"],
            }
        ]
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(
        f"{row['база']}, строк {int(row['строк'])}: "
        f"ошибки {result['ошибки_покрытие']:.3f} / "
        f"{result['ошибки_ширина']:.2f} дн, "
        f"F_d {result['покрытие']:.3f} / {result['Fd_ширина']:.2f} дн, "
        f"короче на {result['сужение']:.1%}"
    )
    print(f"гипотеза 3 {result['вердикт']}")


if __name__ == "__main__":
    main()
