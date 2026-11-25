"""3) Расчёт вероятности покупки в день.

Сеть HazardMLP на каждый из 14 дней даёт шанс купить в этот день,
если раньше не купили. Это выход ``forward``. Дальше
``purchase_timing.hazard_summary`` складывает дневные шансы в
накопленную долю F_d: «покупка уже случилась к этому дню».

Окно гипотезы — не один день, а края F_d: первый день, где доля
доходит до 0,1, и первый, где доходит до 0,9.

Дообучение: AdamW, шаг 1e-4, 3 эпохи, семя 0. На июне–июле из весов
0,25, 1 и 4 остаётся 0,25, сдвиг краёв 0. Старт —
``Веса_моделей/H3/mlp_hazard_старт.pt``. Повтор целиком: ``python interval_h3b.py``.
"""

import numpy as np
from torch import nn

import purchase_timing as pt

HORIZON = 14
SEED = 0
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 3
BATCH_SIZE = 4096
PENALTY_WEIGHTS = (0.25, 1.0, 4.0)


class HazardMLP(nn.Module):
    """Сеть дневного шанса покупки.

    Категории идут через векторы. Дальше слои 256 и 128: линейный,
    нормализация, ReLU, dropout 0,2. Выходов ``horizon``, по одному
    на день. Сигмоида выхода — вероятность купить в этот день,
    если до него ещё не купили.

    Args:
        n_numeric: Число числовых входов вместе с флагами пропусков.
        cardinalities: Число уровней каждой категории.
        horizon: Длина окна в днях, в прогоне 14.
        hidden: Ширины скрытых слоёв. По умолчанию (256, 128).
        dropout: Доля dropout. По умолчанию 0,2.
    """

    def __init__(self, n_numeric, cardinalities, horizon, hidden=(256, 128), dropout=0.2):
        super().__init__()
        dims = [min(16, (n + 1) // 2 + 1) for n in cardinalities]
        self.embeddings = nn.ModuleList(
            nn.Embedding(n + 1, d, padding_idx=0) for n, d in zip(cardinalities, dims)
        )
        layers, width = [], n_numeric + sum(dims)
        for size in hidden:
            layers += [nn.Linear(width, size), nn.BatchNorm1d(size), nn.ReLU(), nn.Dropout(dropout)]
            width = size
        layers.append(nn.Linear(width, horizon))
        self.body = nn.Sequential(*layers)

    def forward(self, numeric, codes):
        """Вернуть логиты дневного шанса покупки.

        Args:
            numeric: Тензор числовых признаков.
            codes: Тензор кодов категорий, столбцы как в
                ``forecast.CATEGORICAL_FEATURES``.

        Returns:
            torch.Tensor: Логиты формы ``(n, horizon)``.
            Вероятность дня — сигмоида этого логита.
        """
        parts = [numeric] + [embedding(codes[:, i]) for i, embedding in enumerate(self.embeddings)]
        return self.body(torch.cat(parts, dim=1))


def day_probability(logits):
    """Посчитать вероятность купить в каждый день, если ещё не купили.

    Args:
        logits: Тензор ``(n, horizon)``, сырой выход сети.

    Returns:
        torch.Tensor: ``h_d = sigmoid(logits)``, та же форма.

    Examples:
        hazard = day_probability(logits)
    """
    return logits.sigmoid()


def cumulative_F(logits):
    """Сложить дневные шансы в долю «уже купили к дню d».

    Args:
        logits: Тензор ``(n, horizon)``, сырой выход сети.

    Returns:
        torch.Tensor: F_d той же формы.

    Examples:
        bought_by, in_window, expected = pt.hazard_summary(logits)
    """
    bought_by, _window, _expected = pt.hazard_summary(logits)
    return bought_by


def band_edges(bought_by, low_level=0.1, high_level=0.9):
    """Считать края окна по накопленной F_d.

    Если уровень не достигнут, берётся 14-й день. Правый край
    не меньше левого.

    Args:
        bought_by: Массив F_d, форма ``(n, 14)``.
        low_level: Порог левого края. По умолчанию 0,1.
        high_level: Порог правого края. По умолчанию 0,9.

    Returns:
        tuple: ``(low, high)`` — дни 1..14 как float.

    Examples:
        low, high = band_edges(curve)
    """

    def first_day(level):
        hit = bought_by >= level
        index = np.where(hit.any(axis=1), hit.argmax(axis=1), HORIZON - 1)
        return (index + 1).astype(np.float64)

    low = first_day(low_level)
    high = np.maximum(first_day(high_level), low)
    return low, high
