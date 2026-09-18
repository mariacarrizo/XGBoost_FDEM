"""Per-target RMSE/R2 breakdown for the lc600k architecture-search models.

Reuses train_and_test() from train_test_xgboost_synthetic.py with each n_layers
count's own tuned architecture (Results/architecture_search/architectures/
arch_{n}layers_lc600k.json), and additionally records per-layer R2 (the existing
train_and_test only returns per-layer RMSE).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from train_test_xgboost_synthetic import build_dataset, scale_and_split

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "Results"
ARCH_DIR = RESULTS_DIR / "architecture_search" / "architectures"

TAG = "lc600k"
LAYER_COUNTS = list(range(3, 13))
SEED = 42
EARLY_STOPPING_ROUNDS = 50


def train_and_test_per_layer(n_layers, architecture):
    from xgboost import XGBRegressor

    X, y = build_dataset(n_layers, TAG, SEED)
    X_train, X_val, X_test, y_train, y_val, y_test, _, _ = scale_and_split(X, y, SEED)

    per_layer_rmse = []
    per_layer_r2 = []
    for i in range(n_layers):
        model = XGBRegressor(
            **architecture,
            objective="reg:squarederror",
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbosity=0,
        )
        model.fit(
            X_train, y_train[:, i],
            eval_set=[(X_val, y_val[:, i])],
            verbose=False,
        )
        y_pred = model.predict(X_test)
        per_layer_rmse.append(float(np.sqrt(np.mean((y_test[:, i] - y_pred) ** 2))))
        per_layer_r2.append(float(r2_score(y_test[:, i], y_pred)))

    return per_layer_rmse, per_layer_r2


def main():
    rows = []
    for n_layers in LAYER_COUNTS:
        arch_path = ARCH_DIR / f"arch_{n_layers}layers_{TAG}.json"
        with open(arch_path) as f:
            architecture = json.load(f)
        print(f"=== {n_layers} layers: training {n_layers} per-target models ===")
        per_layer_rmse, per_layer_r2 = train_and_test_per_layer(n_layers, architecture)
        for i, (rmse, r2) in enumerate(zip(per_layer_rmse, per_layer_r2)):
            rows.append({"n_layers": n_layers, "layer_index": i + 1, "rmse": rmse, "r2": r2})
        print(f"[{n_layers} layers] per-layer RMSE={[round(v, 4) for v in per_layer_rmse]}")
        print(f"[{n_layers} layers] per-layer R2  ={[round(v, 4) for v in per_layer_r2]}")

    df = pd.DataFrame(rows)
    csv_path = RESULTS_DIR / f"per_layer_breakdown_{TAG}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[per-layer] csv -> {csv_path}")

    layer_counts = sorted(df["n_layers"].unique())
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(layer_counts)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for color, n_layers in zip(colors, layer_counts):
        sub = df[df["n_layers"] == n_layers]
        rel_depth = sub["layer_index"] / n_layers
        ax1.plot(rel_depth, sub["rmse"], "o-", color=color, label=f"{n_layers} layers")
        ax2.plot(rel_depth, sub["r2"], "o-", color=color, label=f"{n_layers} layers")
    ax1.set_xlabel("Relative depth position (layer / n_layers)")
    ax1.set_ylabel("Per-layer RMSE (log10 S/m, normalized)")
    ax1.set_title("Per-layer RMSE by model (lc600k)")
    ax1.grid(alpha=0.3)
    ax2.set_xlabel("Relative depth position (layer / n_layers)")
    ax2.set_ylabel("Per-layer R2")
    ax2.set_title("Per-layer R2 by model (lc600k)")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plot_path = RESULTS_DIR / f"per_layer_breakdown_{TAG}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"[per-layer] plot -> {plot_path}")


if __name__ == "__main__":
    main()
