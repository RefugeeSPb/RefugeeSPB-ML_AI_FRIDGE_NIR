"""Вердикт H3 по сохранённой таблице покрытия и ширины. Модель не учит.

Покрытие — доля строк, где настоящий день покупки внутри окна.
Ширина — среднее число дней окна. Общая полоса лучше полосы ошибок,
если покрытие не ниже 0,78 и окно короче не меньше чем на 10%.
"""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "данные" / "метрики" / "полоса_h3b.csv"
OUT = HERE.parent / "результаты" / "полоса.csv"
COVER_MIN = 0.78
NARROWER_MIN = 0.10


def load_official(path):
    """Прочитать одну официальную строку теста H3.

    Args:
        path: Путь к ``полоса_h3b.csv``.

    Returns:
        Series с покрытием, шириной и порогами.
    """
    frame = pd.read_csv(path)
    if len(frame) != 1:
        raise ValueError(f"ожидалась одна строка, пришло {len(frame)}")
    return frame.iloc[0]


def verdict(row):
    """Собрать вердикт порогов покрытия и сужения.

    Args:
        row: Строка с ``Fd_покрытие``, ``ошибки_ширина``, ``Fd_ширина``.

    Returns:
        Словарь средних, сужения и строки вердикта.

    Examples:
        >>> verdict(pd.Series({
        ...     "Fd_покрытие": 0.827,
        ...     "ошибки_ширина": 5.59,
        ...     "Fd_ширина": 4.38,
        ... }))["вердикт"]
        'подтверждена'
    """
    cover = float(row["Fd_покрытие"])
    base_width = float(row["ошибки_ширина"])
    model_width = float(row["Fd_ширина"])
    narrower = (base_width - model_width) / base_width
    confirmed = cover >= COVER_MIN and narrower >= NARROWER_MIN
    return {
        "покрытие": cover,
        "ошибки_покрытие": float(row["ошибки_покрытие"]),
        "ошибки_ширина": base_width,
        "Fd_ширина": model_width,
        "сужение": narrower,
        "порог_покрытия": COVER_MIN,
        "порог_сужения": NARROWER_MIN,
        "вердикт": "подтверждена" if confirmed else "не подтверждена",
    }


def main():
    """Прочитать csv теста, записать вердикт, напечатать итог."""
    row = load_official(DATA)
    result = verdict(row)
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
        f"ошибки {result['ошибки_покрытие']:.3f} / {result['ошибки_ширина']:.2f} дн, "
        f"F_d {result['покрытие']:.3f} / {result['Fd_ширина']:.2f} дн, "
        f"короче на {result['сужение']:.1%}"
    )
    print(f"гипотеза 3 {result['вердикт']}")


if __name__ == "__main__":
    main()
