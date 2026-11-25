"""Гипотеза 3а: полоса по накопленной F_d и короткое дообучение hazard-сети.

Вес штрафа и сдвиг порогов выбираются только на validation.
Тест считается один раз, после того как вес и сдвиг зафиксированы.
Пороги подтверждения те же, что у гипотезы 3: покрытие 78–82% и полоса
уже интервала ошибок точечного прогноза не меньше чем на 10%.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import forecast
import interval_h3
import purchase_timing as pt

HERE = Path(__file__).resolve().parent
DB = HERE.parents[2] / "Общие_артефакты" / "данные" / "fridge_lab.sqlite"
CHECKPOINT = HERE.parents[2] / "Веса_моделей" / "H3" / "mlp_hazard_старт.pt"
HORIZON = 14
SEED = 0
# Зафиксировано до теста. Шаг в десять раз меньше, чем при первом обучении.
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 3
BATCH_SIZE = 4096
# Сетка веса штрафа за промах квантиля. Ноль не входит: гипотеза — про дообучение.
PENALTY_WEIGHTS = (0.25, 1.0, 4.0)
# Допуск на validation: вероятность «купят за 14 дней» не должна просесть сильнее.
PR_AUC_DROP = 0.01


def signed_log(values):
    return np.sign(values) * np.log1p(np.abs(values))


class HazardMLP(nn.Module):
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
        parts = [numeric] + [embedding(codes[:, i]) for i, embedding in enumerate(self.embeddings)]
        return self.body(torch.cat(parts, dim=1))


def encode(frame, bundle):
    values = (signed_log(frame[forecast.NUMERIC_FEATURES].to_numpy(np.float32)) - bundle["numeric_mean"]) / bundle["numeric_std"]
    flags = frame[bundle["missing_columns"]].isna().to_numpy(np.float32)
    numeric = np.concatenate([np.nan_to_num(values, nan=0.0), flags], axis=1).astype(np.float32)
    codes = np.stack(
        [frame[column].map(bundle["category_maps"][column]).fillna(0).to_numpy(np.int64) for column in forecast.CATEGORICAL_FEATURES],
        axis=1,
    )
    days = frame["days_to_purchase"].fillna(0).to_numpy(np.int64)
    bought = frame["label"].to_numpy(np.int64)
    return (
        torch.from_numpy(numeric),
        torch.from_numpy(codes),
        torch.from_numpy(days),
        torch.from_numpy(bought),
    )


def load_bundle():
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    maps = saved["category_maps"]
    model = HazardMLP(
        len(forecast.NUMERIC_FEATURES) + len(saved["missing_columns"]),
        [len(maps[column]) for column in forecast.CATEGORICAL_FEATURES],
        int(saved["horizon_days"]),
    )
    model.load_state_dict(saved["state_dict"])
    bundle = {
        "model": model,
        "category_maps": maps,
        "numeric_mean": np.asarray(saved["numeric_mean"], dtype=np.float32),
        "numeric_std": np.asarray(saved["numeric_std"], dtype=np.float32),
        "missing_columns": list(saved["missing_columns"]),
    }
    return bundle


def forward_summary(model, numeric, codes):
    model.eval()
    bought_by = []
    in_window = []
    with torch.no_grad():
        for start in range(0, len(numeric), BATCH_SIZE):
            stop = start + BATCH_SIZE
            logits = model(numeric[start:stop], codes[start:stop])
            bought, window, _expected = pt.hazard_summary(logits)
            bought_by.append(bought.numpy())
            in_window.append(window.numpy())
    return np.concatenate(bought_by), np.concatenate(in_window)


def band_edges(bought_by, low_level=0.1, high_level=0.9):
    def first_day(level):
        hit = bought_by >= level
        index = np.where(hit.any(axis=1), hit.argmax(axis=1), HORIZON - 1)
        return (index + 1).astype(np.float64)

    low = first_day(low_level)
    high = np.maximum(first_day(high_level), low)
    return low, high


def pinball_penalty(logits, days):
    """Ожидаемый пинболл дней 0,1 и 0,9. Только строки, где покупка попала в 14 дней."""
    bought = days > 0
    if not torch.any(bought):
        return logits.sum() * 0.0
    bought_by, in_window, _expected = pt.hazard_summary(logits[bought])
    previous = torch.cat([torch.zeros_like(bought_by[:, :1]), bought_by[:, :-1]], dim=1)
    mass = (bought_by - previous) / in_window.clamp(min=1e-6).unsqueeze(1)
    day = torch.arange(1, HORIZON + 1, device=logits.device, dtype=logits.dtype)
    target = days[bought].to(logits.dtype).unsqueeze(1)
    diff = target - day.unsqueeze(0)
    loss = logits.new_zeros(())
    for alpha in (0.1, 0.9):
        pin = torch.where(diff >= 0, alpha * diff, (alpha - 1.0) * diff)
        loss = loss + (mass * pin).sum(dim=1).mean()
    return loss


def fine_tune(model, numeric, codes, days, weight):
    tuned = HazardMLP(
        numeric.shape[1],
        [embedding.num_embeddings - 1 for embedding in model.embeddings],
        HORIZON,
    )
    tuned.load_state_dict(model.state_dict())
    tuned.train()
    optimizer = torch.optim.AdamW(tuned.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(
        TensorDataset(numeric, codes, days),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )
    for epoch in range(1, EPOCHS + 1):
        total = 0.0
        seen = 0
        for batch_numeric, batch_codes, batch_days in loader:
            optimizer.zero_grad()
            logits = tuned(batch_numeric, batch_codes)
            loss = pt.hazard_loss(logits, batch_days) + weight * pinball_penalty(logits, batch_days)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(batch_days)
            seen += len(batch_days)
        print(f"  вес {weight:g}, эпоха {epoch}: потеря {total / seen:.4f}", flush=True)
    tuned.eval()
    return tuned


def interval_metrics(actual, low, high):
    return interval_h3.coverage(actual, low, high), interval_h3.mean_width(low, high)


def choose(candidates):
    """Сначала полоса внутри 78–82% и без просадки F_14, затем ближайшая к 80%."""
    guarded = [item for item in candidates if item["pr_ok"]]
    pool = guarded or candidates
    inside = [item for item in pool if 0.78 <= item["cover"] <= 0.82]
    if inside:
        return min(inside, key=lambda item: (item["width"], item["weight"]))
    return min(pool, key=lambda item: (abs(item["cover"] - 0.80), item["width"], item["weight"]))


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(16)
    db = Path(sys.argv[1]) if len(sys.argv) > 1 else DB
    print(f"база {db.name}", flush=True)
    bundle = load_bundle()
    frozen = bundle["model"]
    data = forecast.load(db)

    # Сеть училась на срезе с 1 октября. Полоса ошибок — тем же кодом, что гипотеза 3.
    network_parts = {
        "train": forecast.build_rows(data, "2025-10-01", "2026-06-01", HORIZON)[0],
        "val": forecast.build_rows(data, "2026-06-01", "2026-08-01", HORIZON)[0],
        "test": forecast.build_rows(data, "2026-08-01", "2026-10-01", HORIZON)[0],
    }
    scored = {name: frame[(frame["pair_n"] >= 3) & frame["days_to_purchase"].notna()].copy() for name, frame in network_parts.items()}
    baseline_rows = {name: interval_h3.rows(data, start, end) for name, start, end in interval_h3.PARTS}
    print(
        f"строк полосы train/val/test: {len(scored['train'])} {len(scored['val'])} {len(scored['test'])}",
        flush=True,
    )
    if len(scored["test"]) != len(baseline_rows["test"]):
        raise SystemExit("тестовые строки сети и гипотезы 3 не совпали")

    point = interval_h3.fit_point(baseline_rows["train"], baseline_rows["val"])
    val_actual_base = baseline_rows["val"]["days_to_purchase"].to_numpy()
    residual_low, residual_high = np.quantile(
        val_actual_base - point.predict(interval_h3.pool(baseline_rows["val"])),
        [0.1, 0.9],
    )
    test_actual = scored["test"]["days_to_purchase"].to_numpy()
    test_point = point.predict(interval_h3.pool(baseline_rows["test"]))
    base_low = test_point + residual_low
    base_high = np.maximum(test_point + residual_high, base_low)
    base_cover, base_width = interval_metrics(test_actual, base_low, base_high)

    train_numeric, train_codes, train_days, _ = encode(network_parts["train"], bundle)
    val_numeric, val_codes, _, val_bought = encode(network_parts["val"], bundle)
    val_score_numeric, val_score_codes, val_days, _ = encode(scored["val"], bundle)
    test_score_numeric, test_score_codes, _, _ = encode(scored["test"], bundle)

    frozen_curve, frozen_window = forward_summary(frozen, val_numeric, val_codes)
    frozen_ap = average_precision_score(val_bought.numpy(), frozen_window)
    print(f"validation, F_14 до дообучения, PR-AUC {frozen_ap:.4f}", flush=True)

    candidates = []
    tuned_models = {}
    val_actual = val_days.numpy()
    for weight in PENALTY_WEIGHTS:
        tuned = fine_tune(frozen, train_numeric, train_codes, train_days, weight)
        curve, window = forward_summary(tuned, val_numeric, val_codes)
        pr_auc = average_precision_score(val_bought.numpy(), window)
        score_curve, _ = forward_summary(tuned, val_score_numeric, val_score_codes)
        low, high = band_edges(score_curve)
        delta = interval_h3.calibrate(val_actual, low, high)
        shifted_low = low - delta
        shifted_high = np.maximum(high + delta, shifted_low)
        cover, width = interval_metrics(val_actual, shifted_low, shifted_high)
        item = {
            "weight": weight,
            "delta": delta,
            "cover": cover,
            "width": width,
            "pr_auc": pr_auc,
            "pr_ok": pr_auc >= frozen_ap - PR_AUC_DROP,
        }
        candidates.append(item)
        tuned_models[weight] = tuned
        print(
            f"validation, вес {weight:g}: покрытие {cover:.3f}, ширина {width:.3f}, "
            f"сдвиг {delta:.2f}, PR-AUC {pr_auc:.4f}",
            flush=True,
        )

    chosen = choose(candidates)
    print(
        f"зафиксировано до теста: вес {chosen['weight']:g}, сдвиг {chosen['delta']:.2f}",
        flush=True,
    )
    model = tuned_models[chosen["weight"]]
    test_curve, _ = forward_summary(model, test_score_numeric, test_score_codes)
    low, high = band_edges(test_curve)
    model_low = low - chosen["delta"]
    model_high = np.maximum(high + chosen["delta"], model_low)
    model_cover, model_width = interval_metrics(test_actual, model_low, model_high)
    narrower = (base_width - model_width) / base_width
    same_band = 0.78 <= base_cover <= 0.82 and 0.78 <= model_cover <= 0.82
    confirmed = 0.78 <= model_cover <= 0.82 and narrower >= 0.10 and same_band and chosen["pr_ok"]
    print(f"тест, интервал по ошибкам: покрытие {base_cover:.3f}, ширина {base_width:.3f}")
    print(
        f"тест, накопленная полоса: покрытие {model_cover:.3f}, ширина {model_width:.3f}, "
        f"короче на {narrower:.3f}"
    )
    print(
        f"validation PR-AUC F_14: было {frozen_ap:.4f}, у выбранного веса {chosen['pr_auc']:.4f}"
    )
    print("гипотеза 3а подтверждена" if confirmed else "гипотеза 3а не подтверждена")


if __name__ == "__main__":
    main()
