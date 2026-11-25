"""Вердикт H1 по сохранённым Recall@10 семей. Модели не учит.

Сравнение — общая модель минус модель только на этой семье.
Семя бутстрепа 42, 2000 повторов. Общая лучше, если низ
95% интервала строго выше нуля.
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "данные" / "метрики" / "forecast_v2_households.csv"
OUT = HERE.parent / "результаты" / "recall10.csv"
SEED = 42
ROUNDS = 2000


def load_households(path):
    """Прочитать Recall@10 по семьям из двухстрочной шапки csv.

    Args:
        path: Путь к ``forecast_v2_households.csv``.

    Returns:
        DataFrame со столбцами ``hh``, ``личная``, ``общая``.
    """
    raw = pd.read_csv(path, header=None)
    body = raw.iloc[3:].copy()
    body.columns = [
        "hh",
        "freq_r",
        "freq_p",
        "glob_r",
        "glob_p",
        "pers_r",
        "pers_p",
        "sim_r",
        "sim_p",
        "corr_r",
        "corr_p",
    ]
    frame = pd.DataFrame(
        {
            "hh": pd.to_numeric(body["hh"]),
            "личная": pd.to_numeric(body["pers_r"]),
            "общая": pd.to_numeric(body["glob_r"]),
        }
    )
    return frame.reset_index(drop=True)


def interval(diff, seed=SEED, rounds=ROUNDS):
    """Оценить 95% интервал среднего прироста.

    Args:
        diff (numpy.ndarray): Прирост по семьям.
        seed: Семя генератора. По умолчанию 42.
        rounds: Число выборок с возвращением. По умолчанию 2000.

    Returns:
        Кортеж ``(low, high)``.
    """
    rng = np.random.default_rng(seed)
    draws = np.sort(
        np.array(
            [rng.choice(diff, len(diff), replace=True).mean() for _ in range(rounds)]
        )
    )
    return float(draws[int(0.025 * (rounds - 1))]), float(
        draws[int(0.975 * (rounds - 1))]
    )


def verdict(frame):
    """Посчитать прирост, интервал и вердикт «лучше».

    Общая лучше личной, если низ интервала строго выше нуля.

    Args:
        frame (pandas.DataFrame): Столбцы ``личная`` и ``общая``.

    Returns:
        Словарь с числом семей, средними, приростом, границами
        интервала и строкой вердикта.
    """
    personal = frame["личная"].to_numpy(dtype="float64")
    general = frame["общая"].to_numpy(dtype="float64")
    diff = general - personal
    gain = float(diff.mean())
    low, high = interval(diff)
    confirmed = low > 0
    return {
        "семей": int(len(frame)),
        "только_семья": float(personal.mean()),
        "общая": float(general.mean()),
        "прирост": gain,
        "низ_CI": low,
        "верх_CI": high,
        "вердикт": "подтверждена" if confirmed else "не подтверждена",
        "доля_общая_выше": float((diff > 0).mean()),
    }


def main():
    """Прочитать csv семей, записать вердикт, напечатать итог. Семя 42."""
    frame = load_households(DATA)
    result = verdict(frame)
    row = pd.DataFrame(
        [
            {
                "база": "200 семей семя 42",
                "семей": result["семей"],
                "только_семья": round(result["только_семья"], 3),
                "общая": round(result["общая"], 3),
                "прирост": round(result["прирост"], 3),
                "низ_CI": round(result["низ_CI"], 3),
                "верх_CI": round(result["верх_CI"], 3),
                "вердикт": result["вердикт"],
            }
        ]
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    row.to_csv(OUT, index=False)
    print(
        f"семей {result['семей']}, семя {SEED}: "
        f"личная {result['только_семья']:.3f}, "
        f"общая {result['общая']:.3f}, "
        f"прирост {result['прирост']:.3f}, "
        f"95% [{result['низ_CI']:.3f}; {result['верх_CI']:.3f}], "
        f"лучше если низ > 0"
    )
    print(f"гипотеза 1 {result['вердикт']}")


if __name__ == "__main__":
    main()
