"""2) Расчёт окна вокруг названного дня покупки.

Считается не вероятность, а база сравнения для H3.

1. CatBoost RMSE называет один день («молоко на 6-й день»).
2. На июне–июле считаются ошибки этого дня: факт минус прогноз.
3. Квантили ошибок 0,1 и 0,9 дают одну длину окна на все продукты.
4. Эту длину переносят на тест и рисуют вокруг названного дня.

Результат шага — левый и правый край окна ошибок. Покрытие и
сужение этого окна против F_d считает файл 4.

Два квантильных CatBoost в вердикт не входят.
Повтор целиком: ``python interval_h3b.py``.
"""

import numpy as np
from catboost import CatBoostRegressor, Pool

import forecast

SEED = 0


def pool(frame):
    """Собрать Pool признака и дня покупки.

    Args:
        frame (pandas.DataFrame): Строки с ``forecast.FEATURES`` и
            ``days_to_purchase``.

    Returns:
        catboost.Pool: Таблица для регрессии дня.

    Examples:
        train_pool = pool(train)
    """
    return Pool(
        frame[forecast.FEATURES],
        frame["days_to_purchase"],
        cat_features=forecast.CATEGORICAL_FEATURES,
    )


def fit_point(train, val):
    """Обучить модель, которая называет один день покупки.

    Потеря RMSE. Потолок 400 деревьев, шаг 0,08, глубина 6.
    Остановка — 30 раундов без улучшения на ``val``.

    Args:
        train (pandas.DataFrame): Строки учёбы.
        val (pandas.DataFrame): Строки июня–июля.

    Returns:
        CatBoostRegressor: Модель одного дня.

    Examples:
        point = fit_point(train, val)
    """
    model = CatBoostRegressor(
        loss_function="RMSE",
        iterations=400,
        learning_rate=0.08,
        depth=6,
        random_seed=SEED,
        early_stopping_rounds=30,
        verbose=False,
    )
    model.fit(pool(train), eval_set=pool(val))
    return model


def residual_band(point, val):
    """Посчитать длину окна по ошибкам validation.

    Args:
        point (CatBoostRegressor): Результат ``fit_point``.
        val (pandas.DataFrame): Строки июня–июля с известным днём.

    Returns:
        tuple: ``(low_shift, high_shift)`` — квантили 0,1 и 0,9
        остатка «факт − прогноз».

    Examples:
        low_shift, high_shift = residual_band(point, val)
    """
    residual = val["days_to_purchase"].to_numpy() - point.predict(pool(val))
    low_shift, high_shift = np.quantile(residual, [0.1, 0.9])
    return float(low_shift), float(high_shift)


def apply_band(point, frame, low_shift, high_shift):
    """Нарисовать окно ошибок вокруг названного дня.

    Правый край не ставится левее левого.

    Args:
        point (CatBoostRegressor): Точечная модель.
        frame (pandas.DataFrame): Строки, обычно тест.
        low_shift: Квантиль 0,1 остатка validation.
        high_shift: Квантиль 0,9 остатка validation.

    Returns:
        tuple: ``(low, high)`` — массивы краёв той же длины,
        что ``frame``.

    Examples:
        low, high = apply_band(point, test, low_shift, high_shift)
    """
    day = point.predict(pool(frame))
    low = day + low_shift
    high = np.maximum(day + high_shift, low)
    return low, high
