"""2) Обучение и валидация личных моделей.

Текст как в прогоне H1: ``forecast_v2.fit_personal`` и цикл из
``personal_h1.main``. Настройки заморожены до теста: 150 деревьев,
шаг 0,05, глубина 3, семя 42. Похожие семьи и поправка в вердикт H1
не входят: ``neighbours`` пустой, ``neighbour_weight=0``.
Повтор целиком: ``python personal_h1.py``.
"""

import numpy as np
from catboost import CatBoostClassifier, Pool

import forecast_v2 as f2

SEED = 42
PERSONAL = {"iterations": 150, "learning_rate": 0.05, "depth": 3}


def _classifier(iterations, learning_rate, depth, seed, boost_from_average=True):
    """Собрать CatBoostClassifier с настройками прогона H1.

    Args:
        iterations: Число деревьев.
        learning_rate: Шаг градиентного бустинга.
        depth: Глубина дерева.
        seed: Семя ``random_seed``.
        boost_from_average (bool): Стартовать от среднего метки.
            По умолчанию True.

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
            Если False, метка не передаётся. По умолчанию True.
        weight: Веса строк для ``Pool``. None — без весов.
        baseline: Исходный логит для ``Pool``. None — без
            baseline.

    Returns:
        catboost.Pool: Таблица для ``fit`` или ``predict``.

    Examples:
        pool = _pool(own, f2.FEATURES, f2.CATEGORICAL_FEATURES)
    """
    return Pool(frame[features], label=frame[f2.LABEL] if label else None, cat_features=categorical,
                weight=weight, baseline=baseline)


def fit_personal(train, hh, params, seed, neighbours=(), neighbour_weight=0.0, by_hh=None):
    """Обучить модель только на строках семьи ``hh``.

    Если ``neighbour_weight > 0``, к своим строкам добавляются
    строки ``neighbours`` с этим весом. В вердикте H1 соседей
    нет: ``neighbours=()``, ``neighbour_weight=0``. Если у
    собранных строк меньше двух значений метки, модель не
    учится.

    Args:
        train (pandas.DataFrame): Учёба (в прогоне — учёба плюс
            валидация). Нужны ``hh``, ``f2.FEATURES`` и метка.
        hh: Идентификатор семьи.
        params (dict): Ключи ``iterations``, ``learning_rate``,
            ``depth``. В прогоне — ``PERSONAL``.
        seed: Семя CatBoost.
        neighbours: Идентификаторы похожих семей. По умолчанию
            пустой кортеж.
        neighbour_weight (float): Вес строк соседей. По
            умолчанию 0.0 — соседи не берутся.
        by_hh (dict | None): Готовый ``groupby("hh").indices``.
            None — индекс считается внутри функции.

    Returns:
        CatBoostClassifier | None: Обученная личная модель либо
        None, если уникальных значений метки меньше двух.

    Examples:
        model = fit_personal(train_full, hh, PERSONAL, SEED, by_hh=by_hh)
    """
    rows = by_hh if by_hh is not None else train.groupby("hh").indices
    own = rows.get(hh, np.array([], dtype=np.int64))
    parts, weights = [own], [np.ones(len(own))]
    if neighbour_weight > 0:
        for other in neighbours:
            if other in rows:
                parts.append(rows[other])
                weights.append(np.full(len(rows[other]), neighbour_weight))
    frame = train.iloc[np.concatenate(parts)]
    if frame[f2.LABEL].nunique() < 2:
        return None
    model = _classifier(params["iterations"], params["learning_rate"], params["depth"], seed)
    model.fit(_pool(frame, f2.FEATURES, f2.CATEGORICAL_FEATURES, weight=np.concatenate(weights)))
    return model


def personal_logit(model, frame):
    """Вернуть сырой логит личной модели по строкам ``frame``.

    Признаки — ``f2.FEATURES`` (без постоянных по семье
    ``hh_size`` и ``segment``).

    Args:
        model (CatBoostClassifier): Результат ``fit_personal``.
        frame (pandas.DataFrame): Строки той же семьи.

    Returns:
        numpy.ndarray: ``RawFormulaVal`` той же длины, что
        ``frame``.

    Examples:
        scores = personal_logit(model, test.iloc[index])
    """
    return model.predict(_pool(frame, f2.FEATURES, f2.CATEGORICAL_FEATURES, label=False),
                         prediction_type="RawFormulaVal")
