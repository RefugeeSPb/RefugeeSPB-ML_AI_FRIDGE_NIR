"""2) Общая модель H2.

Те же настройки, что у общей модели H1: Logloss, шаг 0,08, глубина 6,
``l2_leaf_reg`` 10, семя 42. Категории — продукт, категория, сегмент.

На 200 семьях это модель H1. Ранняя остановка на июне–июле дала
104 дерева, затем модель учат на учёбе вместе с июнем–июлем.
На 500 семьях, семя 43, модель учат заново. Остановка дала 265 деревьев.
В ``check_main_v2.py`` на тест идёт модель после остановки, второго
``fit`` в этом файле нет.
"""

from catboost import CatBoostClassifier, Pool

import forecast

SEED = 42
TREES_200 = 104
TREES_500 = 265


def stop_trees(train, validation):
    """Выбрать число деревьев ранней остановкой на валидации.

    Учит на ``train``, смотрит Logloss ``validation``. Остановка:
    ``od_wait=100``, потолок 2000, ``use_best_model=True``.
    На 200 семьях записанный результат — ``TREES_200``.
    На 500 семьях — ``TREES_500``.

    Args:
        train (pandas.DataFrame): Строки учёбы до 1 июня.
            Нужны ``forecast.FEATURES`` и ``in_next_basket``.
        validation (pandas.DataFrame): Строки июня–июля
            той же схемы.

    Returns:
        int: Число деревьев лучшей итерации.

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
    pool = lambda frame: Pool(
        frame[forecast.FEATURES],
        label=frame["in_next_basket"].astype("int8"),
        cat_features=forecast.CATEGORICAL_FEATURES,
    )
    model.fit(pool(train), eval_set=pool(validation), use_best_model=True)
    return int(model.tree_count_)


def fit_global(train, iterations):
    """Доучить общую модель на переданных строках.

    Без eval_set: число деревьев уже выбрано ``stop_trees``.
    Для 200 семей в ``train`` лежат учёба и июнь–июль,
    ``iterations`` равен 104.

    Args:
        train (pandas.DataFrame): Строки для обучения.
        iterations: Число деревьев.

    Returns:
        CatBoostClassifier: Обученная общая модель.

    Examples:
        model = fit_global(train_full, TREES_200)
    """
    model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=0.08,
        depth=6,
        l2_leaf_reg=10,
        loss_function="Logloss",
        random_seed=SEED,
        thread_count=-1,
        verbose=0,
        allow_writing_files=False,
    )
    model.fit(
        Pool(
            train[forecast.FEATURES],
            label=train["in_next_basket"].astype("int8"),
            cat_features=forecast.CATEGORICAL_FEATURES,
        )
    )
    return model
