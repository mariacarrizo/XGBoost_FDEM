"""Model-complexity (bias/variance) sweep at a fixed dataset size.

Step 2 of the architecture workflow, after src/learning_curve_synthetic.py (step 1,
data size) settled on a working dataset size. Fixes that dataset size and sweeps
XGBoost max_depth -- the primary capacity knob for tree ensembles -- while holding
every other hyperparameter at fixed, reasonable (untuned) defaults. Reports train
vs. validation RMSE per layer count so the under/over-fitting regime and the
achievable ceiling are visible *before* spending time on a full hyperparameter
search (src/architecture_search_synthetic.py, step 3): a joint random search run on
the wrong side of the bias/variance curve wastes most of its iterations.

Runs once per requested layer count, since a max_depth that overfits for a 3-layer
problem may still be underfitting a 12-layer one (mirrors the reasoning in
learning_curve_synthetic.py for dataset size).

Expects a database at least as large as --size to already exist under Models/ and
Data/ for every requested layer count, as produced by src/create_database_synthetic.py.
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from learning_curve_synthetic import build_full_pool
from train_test_xgboost_synthetic import scale_and_split

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "Results"

DEFAULT_DEPTHS = [2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 25]

# Everything except max_depth held fixed at reasonable, untuned defaults so the
# sweep isolates the effect of tree depth rather than re-optimizing other knobs.
BASE_ARCHITECTURE = {
    "n_estimators": 600,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "gamma": 0,
    "reg_alpha": 0,
    "reg_lambda": 1,
}


def evaluate_depth(X_train, X_val, X_test, y_train, y_val, y_test, depth,
                    base_architecture, early_stopping_rounds, n_layers):
    t0 = time.time()
    models = []
    best_iterations = []
    for i in range(n_layers):
        model = XGBRegressor(
            **base_architecture,
            max_depth=depth,
            objective="reg:squarederror",
            early_stopping_rounds=early_stopping_rounds,
            verbosity=0,
        )
        model.fit(
            X_train, y_train[:, i],
            eval_set=[(X_val, y_val[:, i])],
            verbose=False,
        )
        models.append(model)
        best_iterations.append(model.best_iteration)
    train_seconds = time.time() - t0

    y_train_pred = np.column_stack([m.predict(X_train) for m in models])
    y_val_pred = np.column_stack([m.predict(X_val) for m in models])
    y_test_pred = np.column_stack([m.predict(X_test) for m in models])

    return {
        "max_depth": depth,
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
        "train_r2": float(r2_score(y_train, y_train_pred)),
        "val_rmse": float(np.sqrt(mean_squared_error(y_val, y_val_pred))),
        "val_mae": float(mean_absolute_error(y_val, y_val_pred)),
        "val_r2": float(r2_score(y_val, y_val_pred)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
        "test_r2": float(r2_score(y_test, y_test_pred)),
        "gap_rmse": float(np.sqrt(mean_squared_error(y_val, y_val_pred))
                           - np.sqrt(mean_squared_error(y_train, y_train_pred))),
        "mean_best_iteration": float(np.mean(best_iterations)),
        "train_seconds": train_seconds,
    }


def save_results(rows, tag, size, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values(["n_layers", "max_depth"]).reset_index(drop=True)

    csv_path = output_dir / f"model_complexity_{tag}_{size}.csv"
    df.to_csv(csv_path, index=False)

    best_idx = df.groupby("n_layers")["val_rmse"].idxmin()
    best_per_layer = df.loc[best_idx].sort_values("n_layers")
    best_path = output_dir / f"model_complexity_{tag}_{size}_best_depth.csv"
    best_per_layer.to_csv(best_path, index=False)

    layer_counts = sorted(df["n_layers"].unique())
    n_cols = 5
    n_rows = int(np.ceil(len(layer_counts) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, n_layers in zip(axes, layer_counts):
        sub = df[df["n_layers"] == n_layers].sort_values("max_depth")
        ax.plot(sub["max_depth"], sub["train_rmse"], "o--", color="tab:blue", label="Train")
        ax.plot(sub["max_depth"], sub["val_rmse"], "o-", color="tab:red", label="Validation")
        best_depth = best_per_layer.loc[best_per_layer["n_layers"] == n_layers, "max_depth"].values[0]
        ax.axvline(best_depth, color="gray", linestyle=":", linewidth=1)
        ax.set_title(f"{n_layers} layers (best depth={best_depth})", fontsize=9)
        ax.grid(alpha=0.3)
    for ax in axes[len(layer_counts):]:
        ax.axis("off")
    axes[0].legend(fontsize=8)
    for i in range(0, len(layer_counts), n_cols):
        axes[i].set_ylabel("RMSE (log10 S/m, normalized)")
    for i in range(max(0, len(layer_counts) - n_cols), len(layer_counts)):
        axes[i].set_xlabel("max_depth")

    fig.suptitle(f"XGBoost bias/variance sweep vs. max_depth (dataset size={size:,}, tag={tag})")
    plt.tight_layout()
    plot_path = output_dir / f"model_complexity_{tag}_{size}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)

    # Overlay plot: generalization gap vs depth, all layer counts together.
    fig2, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(layer_counts)))
    for color, n_layers in zip(colors, layer_counts):
        sub = df[df["n_layers"] == n_layers].sort_values("max_depth")
        ax.plot(sub["max_depth"], sub["gap_rmse"], "o-", color=color, label=f"{n_layers} layers")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("max_depth")
    ax.set_ylabel("Validation - Train RMSE (overfitting gap)")
    ax.set_title(f"Generalization gap vs. max_depth (dataset size={size:,})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    gap_plot_path = output_dir / f"model_complexity_{tag}_{size}_gap.png"
    plt.savefig(gap_plot_path, dpi=150)
    plt.close(fig2)

    print(f"[complexity] summary    -> {csv_path}")
    print(f"[complexity] best depth -> {best_path}")
    print(f"[complexity] plot       -> {plot_path}")
    print(f"[complexity] gap plot   -> {gap_plot_path}")
    print(best_per_layer[["n_layers", "max_depth", "train_rmse", "val_rmse", "gap_rmse", "val_r2"]].to_string(index=False))
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, nargs="+", default=list(range(3, 13)),
                         help="Numbers of layers to run the sweep for (database must exist for each)")
    parser.add_argument("--tag", type=str, required=True,
                         help="Suffix tag identifying the database, matching create_database_synthetic.py")
    parser.add_argument("--size", type=int, default=600_000,
                         help="Fixed dataset size to hold constant while sweeping complexity")
    parser.add_argument("--depths", type=int, nargs="+", default=DEFAULT_DEPTHS,
                         help="max_depth values to sweep, ascending")
    parser.add_argument("--base-architecture-json", type=str, default=None,
                         help="Path to a JSON file overriding BASE_ARCHITECTURE defaults "
                              "(every key except max_depth)")
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    depths = sorted(args.depths)

    base_architecture = dict(BASE_ARCHITECTURE)
    if args.base_architecture_json:
        with open(args.base_architecture_json) as f:
            base_architecture.update(json.load(f))
    print(f"[complexity] base architecture (max_depth swept separately): {base_architecture}")

    rows = []
    for n_layers in args.layers:
        print(f"\n[pool] loading {n_layers}-layer database (tag='{args.tag}')...")
        X_full, y_full = build_full_pool(n_layers, args.tag, args.seed)
        if X_full.shape[0] < args.size:
            raise ValueError(
                f"{n_layers}-layer database only has {X_full.shape[0]} samples, but --size is "
                f"{args.size}."
            )
        X_sub = X_full.iloc[:args.size]
        y_sub = y_full.iloc[:args.size]
        X_train, X_val, X_test, y_train, y_val, y_test, _, _ = scale_and_split(X_sub, y_sub, args.seed)

        for depth in depths:
            t0 = time.time()
            metrics = evaluate_depth(
                X_train, X_val, X_test, y_train, y_val, y_test, depth,
                base_architecture, args.early_stopping_rounds, n_layers,
            )
            metrics["n_layers"] = n_layers
            print(
                f"[{n_layers} layers, depth={depth:>2}] train_RMSE={metrics['train_rmse']:.5f} "
                f"val_RMSE={metrics['val_rmse']:.5f} gap={metrics['gap_rmse']:+.5f} "
                f"val_R2={metrics['val_r2']:.5f} mean_best_iter={metrics['mean_best_iteration']:.0f} "
                f"({time.time() - t0:.1f}s)"
            )
            rows.append(metrics)

    save_results(rows, args.tag, args.size, output_dir)


if __name__ == "__main__":
    main()
