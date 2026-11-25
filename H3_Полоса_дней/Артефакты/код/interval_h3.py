"""Гипотеза 3: 80-процентный интервал дней до покупки на fridge_lab.

Квантильная модель против интервала по ошибкам точечного прогноза на validation.
Тест не участвует в выборе границ. Порог из текста гипотезы не меняется.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

import forecast

HERE = Path(__file__).resolve().parent
DB = HERE / "fridge_lab.sqlite"
HORIZON = 14
SEED = 0
PARTS = (
    ("train", "2025-10-03", "2026-06-01"),
    ("val", "2026-06-01", "2026-08-01"),
    ("test", "2026-08-01", "2026-10-01"),
)


def rows(data, start, end):
    """Собрать строки полосы: продукт брали ≥ 3 раз, день покупки известен.

    Args:
        data (dict): Результат ``forecast.load``.
        start: Начало окна, дата.
        end: Конец окна, дата.

    Returns:
        DataFrame строк для интервала дней.
    """
    frame, _stats = forecast.build_rows(data, start, end, HORIZON)
    frame = frame[(frame["pair_n"] >= 3) & frame["days_to_purchase"].notna()].copy()
    return frame


def pool(frame):
    """Собрать CatBoost Pool признака и дня покупки.

    Args:
        frame (pandas.DataFrame): Строки с ``FEATURES`` и ``days_to_purchase``.

    Returns:
        Pool для регрессии дня.
    """
    return Pool(
        frame[forecast.FEATURES],
        frame["days_to_purchase"],
        cat_features=forecast.CATEGORICAL_FEATURES,
    )


def fit_point(train, val):
    """Обучить точечный CatBoost дня покупки.

    Args:
        train (pandas.DataFrame): Строки учёбы.
        val (pandas.DataFrame): Строки валидации, ранняя остановка.

    Returns:
        Обученный ``CatBoostRegressor`` с RMSE.
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


def fit_quantile(train, val, alpha):
    """Обучить квантильный CatBoost края полосы.

    Args:
        train (pandas.DataFrame): Строки учёбы.
        val (pandas.DataFrame): Строки валидации.
        alpha: Уровень квантиля, 0,1 или 0,9.

    Returns:
        Обученный ``CatBoostRegressor`` с Quantile.
    """
    model = CatBoostRegressor(
        loss_function=f"Quantile:alpha={alpha}",
        iterations=400,
        learning_rate=0.08,
        depth=6,
        random_seed=SEED,
        early_stopping_rounds=30,
        verbose=False,
    )
    model.fit(pool(train), eval_set=pool(val))
    return model


def coverage(actual, low, high):
    """Доля строк, где день покупки внутри окна.

    Args:
        actual: Настоящие дни.
        low: Левый край.
        high: Правый край.

    Returns:
        Число от 0 до 1.
    """
    return float(np.mean((actual >= low) & (actual <= high)))


def mean_width(low, high):
    """Средняя длина окна в днях.

    Args:
        low: Левый край.
        high: Правый край.

    Returns:
        Среднее ``high - low``.
    """
    return float(np.mean(high - low))


def calibrate(actual, low, high):
    """Сдвинуть края на validation ближе всего к покрытию 0,80.

    Args:
        actual: Настоящие дни.
        low: Левый край до сдвига.
        high: Правый край до сдвига.

    Returns:
        Сдвиг ``delta`` в днях.
    """
    best = None
    for delta in np.linspace(-5, 10, 61):
        shifted_low = low - delta
        shifted_high = np.maximum(high + delta, shifted_low)
        covered = coverage(actual, shifted_low, shifted_high)
        width = mean_width(shifted_low, shifted_high)
        key = (abs(covered - 0.80), 0 if 0.78 <= covered <= 0.82 else 1, width)
        if best is None or key < best[0]:
            best = (key, float(delta))
    return best[1]


def main():
    db = Path(sys.argv[1]) if len(sys.argv) > 1 else DB
    print(f"база {db.name}", flush=True)
    data = forecast.load(db)
    parts = {name: rows(data, start, end) for name, start, end in PARTS}
    train, val, test = parts["train"], parts["val"], parts["test"]
    print(f"строк train/val/test: {len(train)} {len(val)} {len(test)}", flush=True)
    point = fit_point(train, val)
    low_model = fit_quantile(train, val, 0.1)
    high_model = fit_quantile(train, val, 0.9)

    val_actual = val["days_to_purchase"].to_numpy()
    val_point = point.predict(pool(val))
    residual = val_actual - val_point
    residual_low, residual_high = np.quantile(residual, [0.1, 0.9])

    val_low = low_model.predict(pool(val))
    val_high = high_model.predict(pool(val))
    delta = calibrate(val_actual, val_low, val_high)

    test_actual = test["days_to_purchase"].to_numpy()
    test_point = point.predict(pool(test))
    base_low = test_point + residual_low
    base_high = np.maximum(test_point + residual_high, base_low)
    model_low = low_model.predict(pool(test)) - delta
    model_high = np.maximum(high_model.predict(pool(test)) + delta, model_low)

    base_cover = coverage(test_actual, base_low, base_high)
    model_cover = coverage(test_actual, model_low, model_high)
    base_width = mean_width(base_low, base_high)
    model_width = mean_width(model_low, model_high)
    narrower = (base_width - model_width) / base_width
    same_band = 0.78 <= base_cover <= 0.82 and 0.78 <= model_cover <= 0.82
    confirmed = 0.78 <= model_cover <= 0.82 and narrower >= 0.10 and same_band
    print(f"сдвиг квантилей на validation: {delta:.2f} дня")
    print(
        f"тест, интервал по ошибкам: покрытие {base_cover:.3f}, ширина {base_width:.3f}"
    )
    print(
        f"тест, квантильная модель: покрытие {model_cover:.3f}, ширина {model_width:.3f}, "
        f"короче на {narrower:.3f}"
    )
    print("гипотеза 3 подтверждена" if confirmed else "гипотеза 3 не подтверждена")


if __name__ == "__main__":
    main()
