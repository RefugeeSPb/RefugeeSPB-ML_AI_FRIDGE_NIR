"""Гипотеза 3б. Короткая полоса на новой базе.

Порог записан до этой базы: покрытие не ниже 0,78 и полоса короче
интервала ошибок точечного прогноза не меньше чем на 10%. Оба условия вместе.
Семя генератора 44, 500 семей. Старые базы не читаются.

Вес штрафа и сдвиг краёв выбираются только на июне–июле.
Для каждого веса берётся сдвиг с самой узкой полосой среди тех,
где покрытие на validation уже не ниже 0,78. Если таких нет — сдвиг
с самым высоким покрытием. Среди весов без просадки F_14 больше 0,01
берётся самая узкая полоса, которая на validation держит 0,78.
Тест открывается один раз после этой записи. Порог после теста не двигается.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

import forecast
import interval_h3
import interval_h3a as h3a

HERE = Path(__file__).resolve().parent
DB = HERE.parents[2] / "Общие_артефакты" / "данные" / "fridge_lab_v3.sqlite"
COVER_MIN = 0.78
NARROWER_MIN = 0.10


def choose_delta(actual, low, high):
    """Подобрать сдвиг краёв на validation.

    Среди сдвигов с покрытием не ниже 0,78 берётся самая узкая полоса.
    Иначе — сдвиг с самым высоким покрытием.

    Args:
        actual: Настоящие дни на validation.
        low: Левый край до сдвига.
        high: Правый край до сдвига.

    Returns:
        Кортеж ``(delta, cover, width)``.
    """
    best = None
    for delta in np.linspace(-5, 10, 61):
        shifted_low = low - delta
        shifted_high = np.maximum(high + delta, shifted_low)
        cover = interval_h3.coverage(actual, shifted_low, shifted_high)
        width = interval_h3.mean_width(shifted_low, shifted_high)
        passed = cover >= COVER_MIN
        key = (0 if passed else 1, width if passed else -cover, abs(float(delta)))
        if best is None or key < best[0]:
            best = (key, float(delta), cover, width)
    return best[1], best[2], best[3]


def choose_weight(candidates):
    """Выбрать вес дообучения по validation.

    Сначала веса без просадки F_14. Среди них — самая узкая полоса
    с покрытием уже не ниже 0,78.

    Args:
        candidates: Список словарей с ``weight``, ``cover``, ``width``,
            ``pr_ok``.

    Returns:
        Выбранный словарь кандидата.
    """
    guarded = [item for item in candidates if item["pr_ok"]]
    pool = guarded or candidates
    passed = [item for item in pool if item["cover"] >= COVER_MIN]
    if passed:
        return min(passed, key=lambda item: (item["width"], item["weight"]))
    return min(pool, key=lambda item: (-item["cover"], item["width"], item["weight"]))


def main():
    """Прогнать H3б: окно по F_d против полосы ошибок, семя 44."""
    db = Path(sys.argv[1]) if len(sys.argv) > 1 else DB
    print(f"база {db.name}", flush=True)
    print(
        f"порог до теста: покрытие не ниже {COVER_MIN:.2f}, уже не меньше чем на {NARROWER_MIN:.0%}",
        flush=True,
    )
    torch.manual_seed(h3a.SEED)
    torch.set_num_threads(16)
    bundle = h3a.load_bundle()
    frozen = bundle["model"]
    data = forecast.load(db)

    network_parts = {
        "train": forecast.build_rows(data, "2025-10-01", "2026-06-01", h3a.HORIZON)[0],
        "val": forecast.build_rows(data, "2026-06-01", "2026-08-01", h3a.HORIZON)[0],
        "test": forecast.build_rows(data, "2026-08-01", "2026-10-01", h3a.HORIZON)[0],
    }
    scored = {
        name: frame[(frame["pair_n"] >= 3) & frame["days_to_purchase"].notna()].copy()
        for name, frame in network_parts.items()
    }
    print(
        f"строк полосы train/val/test: {len(scored['train'])} {len(scored['val'])} {len(scored['test'])}",
        flush=True,
    )

    point = interval_h3.fit_point(scored["train"], scored["val"])
    val_actual_base = scored["val"]["days_to_purchase"].to_numpy()
    residual_low, residual_high = np.quantile(
        val_actual_base - point.predict(interval_h3.pool(scored["val"])),
        [0.1, 0.9],
    )

    train_numeric, train_codes, train_days, _ = h3a.encode(network_parts["train"], bundle)
    val_numeric, val_codes, _, val_bought = h3a.encode(network_parts["val"], bundle)
    val_score_numeric, val_score_codes, val_days, _ = h3a.encode(scored["val"], bundle)

    frozen_curve, frozen_window = h3a.forward_summary(frozen, val_numeric, val_codes)
    frozen_ap = average_precision_score(val_bought.numpy(), frozen_window)
    print(f"validation, F_14 до дообучения, PR-AUC {frozen_ap:.4f}", flush=True)

    candidates = []
    tuned_models = {}
    val_actual = val_days.numpy()
    for weight in h3a.PENALTY_WEIGHTS:
        tuned = h3a.fine_tune(frozen, train_numeric, train_codes, train_days, weight)
        curve, window = h3a.forward_summary(tuned, val_numeric, val_codes)
        pr_auc = average_precision_score(val_bought.numpy(), window)
        score_curve, _window = h3a.forward_summary(tuned, val_score_numeric, val_score_codes)
        low, high = h3a.band_edges(score_curve)
        delta, cover, width = choose_delta(val_actual, low, high)
        item = {
            "weight": weight,
            "delta": delta,
            "cover": cover,
            "width": width,
            "pr_auc": pr_auc,
            "pr_ok": pr_auc >= frozen_ap - h3a.PR_AUC_DROP,
        }
        candidates.append(item)
        tuned_models[weight] = tuned
        print(
            f"validation, вес {weight:g}: покрытие {cover:.3f}, ширина {width:.3f}, "
            f"сдвиг {delta:.2f}, PR-AUC {pr_auc:.4f}",
            flush=True,
        )

    chosen = choose_weight(candidates)
    print(
        f"зафиксировано до теста: вес {chosen['weight']:g}, сдвиг {chosen['delta']:.2f}",
        flush=True,
    )

    test_actual = scored["test"]["days_to_purchase"].to_numpy()
    test_point = point.predict(interval_h3.pool(scored["test"]))
    base_low = test_point + residual_low
    base_high = np.maximum(test_point + residual_high, base_low)
    base_cover, base_width = h3a.interval_metrics(test_actual, base_low, base_high)

    test_score_numeric, test_score_codes, _, _ = h3a.encode(scored["test"], bundle)
    test_curve, _window = h3a.forward_summary(tuned_models[chosen["weight"]], test_score_numeric, test_score_codes)
    low, high = h3a.band_edges(test_curve)
    model_low = low - chosen["delta"]
    model_high = np.maximum(high + chosen["delta"], model_low)
    model_cover, model_width = h3a.interval_metrics(test_actual, model_low, model_high)
    narrower = (base_width - model_width) / base_width
    confirmed = model_cover >= COVER_MIN and narrower >= NARROWER_MIN
    print(f"тест, интервал по ошибкам: покрытие {base_cover:.3f}, ширина {base_width:.3f}")
    print(
        f"тест, накопленная полоса: покрытие {model_cover:.3f}, ширина {model_width:.3f}, "
        f"короче на {narrower:.3f}"
    )
    print("гипотеза 3б подтверждена" if confirmed else "гипотеза 3б не подтверждена")


if __name__ == "__main__":
    main()
