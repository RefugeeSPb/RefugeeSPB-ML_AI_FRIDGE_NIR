"""1) Обучение и валидация общей модели.

Текст как в прогоне H1: ``personal_h1.stop_trees`` и ``forecast_v2.fit_global``.
Число деревьев — ранняя остановка на июне–июле (od_wait=100, потолок 2000).
Потом общая модель учится на учёбе + валидации с этим числом деревьев.
Семя 42. Повтор целиком: ``python personal_h1.py``.
"""

from catboost import CatBoostClassifier, Pool

import forecast
import forecast_v2 as f2

SEED = 42


def stop_trees(train, validation):
    """Выбрать число деревьев общей модели ранней остановкой.

    Учит CatBoost на ``train`` с eval_set ``validation``. Остановка по
    ``od_type="Iter"``, ``od_wait=100``, потолок 2000 итераций.
    ``use_best_model=True``. Семя модели — ``SEED``. Число деревьев
    после этого вызова на тесте не двигается.

    Args:
        train (pandas.DataFrame): Строки учёбы. Нужны столбцы
            ``forecast.FEATURES`` и метка ``forecast_v2.LABEL``.
        validation (pandas.DataFrame): Строки валидации той же схемы.

    Returns:
        int: ``tree_count_`` лучшей итерации.

    Examples:
        trees = stop_trees(train, validation)
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


def _classifier(iterations, learning_rate, depth, seed, boost_from_average=True):
    """Собрать CatBoostClassifier с настройками прогона H1.

    Args:
        iterations: Число деревьев.
        learning_rate: Шаг градиентного бустинга.
        depth: Глубина дерева.
        seed: Семя ``random_seed``.
        boost_from_average (bool): Стартовать от среднего метки.
            По умолчанию True. Для поправки к общей модели в
            ``forecast_v2.fit_correction`` передают False.

    Returns:
        CatBoostClassifier: Модель с Logloss, l2_leaf_reg=10,
        verbose=0, без записи служебных файлов.

    Examples:
        model = _classifier(150, 0.05, 3, SEED)
    """
    return CatBoostClassifier(
        iterations=iterations, learning_rate=learning_rate, depth=depth, l2_leaf_reg=10,
        loss_function="Logloss", random_seed=seed, thread_count=-1, verbose=0,
        boost_from_average=boost_from_average, allow_writing_files=False,
    )


def _pool(frame, features, categorical, label=True, weight=None, baseline=None):
    """Собрать Pool CatBoost по таблице строк.

    Args:
        frame (pandas.DataFrame): Строки с признаками и, при
            ``label=True``, столбцом ``forecast_v2.LABEL``.
        features: Список имён признаков.
        categorical: Категориальные признаки из ``features``.
        label (bool): Если True, метка берётся из ``frame``.
            Если False, метка не передаётся (предсказание).
            По умолчанию True.
        weight: Веса строк для ``Pool``. None — без весов.
        baseline: Исходный логит для ``Pool``. None — без
            baseline.

    Returns:
        catboost.Pool: Таблица для ``fit`` или ``predict``.

    Examples:
        pool = _pool(train, forecast.FEATURES, forecast.CATEGORICAL_FEATURES)
    """
    return Pool(frame[features], label=frame[f2.LABEL] if label else None, cat_features=categorical,
                weight=weight, baseline=baseline)


def fit_global(train, iterations, seed, learning_rate=0.08, depth=6):
    """Обучить одну общую модель на переданных строках.

    Без eval_set и ранней остановки: число деревьев уже выбрано
    ``stop_trees``. Признаки — ``forecast.FEATURES``.

    Args:
        train (pandas.DataFrame): Строки учёбы. В прогоне H1 это
            учёт плюс валидация.
        iterations: Число деревьев.
        seed: Семя CatBoost.
        learning_rate: Шаг. По умолчанию 0.08.
        depth: Глубина. По умолчанию 6.

    Returns:
        CatBoostClassifier: Обученная общая модель.

    Examples:
        trees = stop_trees(train, validation)
        model = fit_global(train_full, trees, SEED)
    """
    model = _classifier(iterations, learning_rate, depth, seed)
    model.fit(_pool(train, forecast.FEATURES, forecast.CATEGORICAL_FEATURES))
    return model
