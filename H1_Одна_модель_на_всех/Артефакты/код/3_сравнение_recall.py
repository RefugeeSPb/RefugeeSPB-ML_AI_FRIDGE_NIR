"""3) Сравнение Recall@10: общая минус личная.

Два куска прогона.

После обучения — ``personal_h1.household_recall`` и ``interval``
(``np.percentile``, 2000 повторов, семя 42).

Таблица на слайде — ``eval_h1.py``: читает
``данные/метрики/forecast_v2_households.csv``, пишет
``результаты/recall10.csv``. Запуск этого файла повторяет eval_h1.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import forecast_v2 as f2
import eval_h1

HERE = Path(__file__).resolve().parent
SEED = 42
ROUNDS = 2000
DATA = HERE.parent / "данные" / "метрики" / "forecast_v2_households.csv"
OUT = HERE.parent / "результаты" / "recall10.csv"


def household_recall(frame, scores):
    """Посчитать Recall@10 отдельно по каждой семье.

    Для семьи вызывается ``forecast_v2.recall_at_k`` по её
    строкам и соответствующему срезу ``scores``. Если у семьи
    нет похода с положительной меткой, значение может быть
    NaN.

    Args:
        frame (pandas.DataFrame): Строки теста. Нужны ``hh``,
            ``visit_id`` и метка ``in_next_basket``.
        scores: Баллы той же длины, что ``frame``. Приводятся
            к ``float64``.

    Returns:
        pandas.Series: Recall@10, имя ``recall``, индекс —
        идентификатор семьи.

    Examples:
        general = household_recall(test, test_logit)
        personal = household_recall(test, personal_scores)
    """
    values = {}
    scores = np.asarray(scores, dtype="float64")
    for hh, index in frame.groupby("hh").indices.items():
        values[hh] = f2.recall_at_k(frame.iloc[index], scores[index])
    return pd.Series(values, name="recall")


def interval(diff):
    """Оценить 95% интервал среднего прироста бутстрепом по семьям.

    ``ROUNDS`` выборок с возвращением, генератор ``SEED``.
    Границы — процентили 2.5 и 97.5 по средним выборок.

    Args:
        diff (numpy.ndarray): Прирост общей модели над личной
            по семьям, одно число на семью.

    Returns:
        tuple: ``(low, high)`` — нижняя и верхняя границы
        интервала как float.

    Examples:
        low, high = interval((general - personal).to_numpy())
    """
    rng = np.random.default_rng(SEED)
    draws = np.array([rng.choice(diff, len(diff)).mean() for _round in range(ROUNDS)])
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(low), float(high)


def main():
    """Записать вердикт H1 по сохранённым Recall@10 семей.

    Модели не учит. Читает ``DATA``, считает прирост и интервал
    через ``eval_h1``, пишет одну строку в ``OUT``, печатает
    итог в stdout. Семя бутстрепа 42.

    Notes:
        Создаёт каталог ``результаты``, если его нет. Файл
        ``recall10.csv`` перезаписывается.

    Examples:
        python 3_сравнение_recall.py
    """
    frame = eval_h1.load_households(DATA)
    result = eval_h1.verdict(frame)
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
