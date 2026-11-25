"""H1. Общая модель лучше модели только на этой семье.

База — CatBoost, обученный только на строках семьи, с нуля.
Вариант — одна общая модель на всех семьях. Метрика — Recall@10,
среднее по семьям. Общая лучше, если низ 95% интервала
разницы «общая − личная» строго выше нуля.

Семя модели и бутстрепа — 42. Настройки личной модели выбраны на
40 семьях первой базы и здесь те же: 150 деревьев, шаг 0,05,
глубина 3. Поправка к общей модели в вердикт не входит.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

import forecast
import forecast_v2 as f2

HERE = Path(__file__).resolve().parent
DB = HERE.parents[2] / "Общие_артефакты" / "данные" / "fridge_lab.sqlite"
SEED = 42
ROUNDS = 2000
PERSONAL = {"iterations": 150, "learning_rate": 0.05, "depth": 3}


def build(data):
    """Собрать train, validation и test тем же окном, что список H2.

    Args:
        data (dict): Результат ``forecast_v2.load``.

    Returns:
        Кортеж трёх DataFrame: учёба до 1 июня 2026, валидация
        июнь–июль, тест с 1 августа.
    """
    train, _stats = forecast.build_rows(data, "2025-10-01", "2026-06-01", 14)
    validation, _stats = forecast.build_rows(data, "2026-06-01", "2026-08-01", 14)
    test, _stats = forecast.build_rows(data, "2026-08-01", "2026-10-01", 14)
    return train, validation, test


def stop_trees(train, validation):
    """Выбрать число деревьев общей модели ранней остановкой.

    Семя ``SEED``. На тесте число деревьев не двигается.

    Args:
        train (pandas.DataFrame): Строки учёбы.
        validation (pandas.DataFrame): Строки валидации.

    Returns:
        Число деревьев лучшей итерации.
    """
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
    pool = lambda frame: Pool(frame[forecast.FEATURES], frame[f2.LABEL], cat_features=forecast.CATEGORICAL_FEATURES)
    model.fit(pool(train), eval_set=pool(validation), use_best_model=True)
    return int(model.tree_count_)


def household_recall(frame, scores):
    """Вернуть Recall@10 отдельно по каждой семье.

    Args:
        frame (pandas.DataFrame): Строки теста с ``hh``.
        scores: Баллы той же длины, что ``frame``.

    Returns:
        Series Recall@10, индекс — идентификатор семьи.
    """
    values = {}
    scores = np.asarray(scores, dtype="float64")
    for hh, index in frame.groupby("hh").indices.items():
        values[hh] = f2.recall_at_k(frame.iloc[index], scores[index])
    return pd.Series(values, name="recall")


def interval(diff):
    """Оценить 95% интервал среднего прироста бутстрепом по семьям.

    ``ROUNDS`` выборок с возвращением, генератор ``SEED``.

    Args:
        diff (numpy.ndarray): Прирост общей модели над личной
            по семьям.

    Returns:
        Кортеж ``(low, high)``.
    """
    rng = np.random.default_rng(SEED)
    draws = np.array([rng.choice(diff, len(diff)).mean() for _round in range(ROUNDS)])
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(low), float(high)


def main():
    """Прогнать H1: общая модель лучше модели только на этой семье.

    Notes:
        Читает SQLite, обучает одну общую модель и по одной личной
        модели на семью, пишет вердикт в stdout. Общая лучше,
        если низ интервала строго выше нуля.
    """
    db = Path(sys.argv[1]) if len(sys.argv) > 1 else DB
    print(f"база {db.name}", flush=True)
    print(f"личные настройки до теста: {PERSONAL}", flush=True)
    data = f2.load(db)
    train, validation, test = build(data)
    trees = stop_trees(train, validation)
    print(f"деревьев общей модели {trees}", flush=True)
    train_full = pd.concat([train, validation], ignore_index=True)
    global_model = f2.fit_global(train_full, trees, SEED)
    by_hh = train_full.groupby("hh").indices
    test_logit = f2.global_logit(global_model, test)
    personal_scores = np.empty(len(test))
    for done, (hh, index) in enumerate(test.groupby("hh").indices.items(), start=1):
        part = test.iloc[index]
        personal = f2.fit_personal(train_full, hh, PERSONAL, SEED, by_hh=by_hh)
        personal_scores[index] = (
            test_logit[index] if personal is None else f2.personal_logit(personal, part)
        )
        if done % 50 == 0:
            print(f"семей {done}", flush=True)
    general = household_recall(test, test_logit)
    personal = household_recall(test, personal_scores)
    common = general.dropna().index.intersection(personal.dropna().index)
    general = general.loc[common]
    personal = personal.loc[common]
    gain = float((general - personal).mean())
    low, high = interval((general - personal).to_numpy())
    confirmed = low > 0
    print(
        f"семей {len(common)}, семя {SEED}: "
        f"личная {personal.mean():.3f}, общая {general.mean():.3f}"
    )
    print(
        f"общая минус личная {gain:.3f}, "
        f"интервал [{low:.3f}; {high:.3f}], "
        f"лучше если низ > 0"
    )
    print("гипотеза 1 подтверждена" if confirmed else "гипотеза 1 не подтверждена")


if __name__ == "__main__":
    main()
