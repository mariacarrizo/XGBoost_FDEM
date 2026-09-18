"""Grid-search XGBoost architecture + dataset size across synthetic FDEM databases.

Unlike train_test_xgboost_synthetic.py (which tunes ONE architecture and shares it
across every layer count), this script runs an independent random search for every
(n_layers, tag) combination, then trains/evaluates that combo's own tuned
architecture on its full train/val/test split. This answers: for each number of
EC layers, which dataset size needs its own (possibly different) best architecture,
and which (layers, size, architecture) combination performs best overall.

Each candidate architecture is scored by fitting once (per output) on a training
subset with early stopping against the REAL held-out validation split -- not via
k-fold CV on a small subset (that proved unreliable for the harder, low-SNR,
high-layer-count problems: a tiny CV fold can't reliably tell a good deep
architecture from an overfit one, and silently favors underfit "safe" candidates).

Reuses build_dataset / scale_and_split / train_and_test / DEFAULT_PARAM_DIST from
train_test_xgboost_synthetic.py so the feature engineering and train/val/test split
are identical to that script.

Expects databases for every requested (layers, tag) pair to already exist under
Models/ and Data/, as produced by src/create_database_synthetic.py.
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
from joblib import Parallel, delayed
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterSampler, train_test_split
from xgboost import XGBRegressor

from model_complexity_synthetic import BASE_ARCHITECTURE
from train_test_xgboost_synthetic import (
    DEFAULT_PARAM_DIST,
    build_dataset,
    scale_and_split,
    train_and_test,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "Results"

DEFAULT_TAGS = ["lc200k", "new400", "lc600k", "lc800k"]
DEFAULT_TAG_SIZES = [200_000, 400_000, 600_000, 800_000]

# A pure random search with a modest budget over a ~9-dimensional space can miss
# good regions entirely: it may sample a productive max_depth, but pair it with
# incompatible regularization on that particular draw, and never test the
# combination that actually works. Seed the search with the light-regularization
# architecture from model_complexity_synthetic.py's depth sweep (already shown, in
# step 2, to beat the joint search's own candidates at several depths) so these
# known-good joint combinations are always evaluated alongside the random ones.
SEED_DEPTHS = [8, 10, 13, 16]


def _to_native(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _evaluate_candidate(params, X_train_search, y_train_search, X_val, y_val, n_layers, early_stopping_rounds):
    """Fit one candidate architecture (one XGBRegressor per output, early-stopped
    against the REAL validation split) and return its validation RMSE.

    n_jobs=1 per model: candidates are evaluated in parallel across processes
    instead, so each individual fit stays single-threaded to avoid oversubscription.
    """
    val_preds = np.empty_like(y_val)
    for i in range(n_layers):
        model = XGBRegressor(
            **params,
            objective="reg:squarederror",
            early_stopping_rounds=early_stopping_rounds,
            n_jobs=1,
            verbosity=0,
        )
        model.fit(
            X_train_search, y_train_search[:, i],
            eval_set=[(X_val, y_val[:, i])],
            verbose=False,
        )
        val_preds[:, i] = model.predict(X_val)
    return float(np.sqrt(mean_squared_error(y_val, val_preds)))


def tune_one(n_layers, tag, seed, n_iter, search_fraction, n_jobs, early_stopping_rounds, noise_level=0.0):
    """Random search for a single (n_layers, tag) combination.

    Unlike sklearn's RandomizedSearchCV (which scores candidates via k-fold CV on
    a small subset -- unreliable for the harder/high-layer-count problems, since a
    tiny CV fold can't discriminate deep from shallow architectures on a low-SNR
    target), each candidate here is scored by fitting once on a (larger) training
    subset with early stopping against the REAL held-out validation split, matching
    the evaluation method already validated in learning_curve_synthetic.py and
    model_complexity_synthetic.py.
    """
    X, y = build_dataset(n_layers, tag, seed, noise_level=noise_level)
    X_train, X_val, _, y_train, y_val, _, _, _ = scale_and_split(X, y, seed)

    if search_fraction < 1.0:
        X_train_search, _, y_train_search, _ = train_test_split(
            X_train, y_train, train_size=search_fraction, random_state=seed
        )
    else:
        X_train_search, y_train_search = X_train, y_train

    random_candidates = list(ParameterSampler(DEFAULT_PARAM_DIST, n_iter=n_iter, random_state=seed))
    seed_candidates = [{**BASE_ARCHITECTURE, "max_depth": d} for d in SEED_DEPTHS]
    candidates = seed_candidates + random_candidates

    t0 = time.time()
    val_rmses = Parallel(n_jobs=n_jobs)(
        delayed(_evaluate_candidate)(
            params, X_train_search, y_train_search, X_val, y_val, n_layers, early_stopping_rounds
        )
        for params in candidates
    )
    search_seconds = time.time() - t0

    best_idx = int(np.argmin(val_rmses))
    best_params = {k: _to_native(v) for k, v in candidates[best_idx].items()}
    search_rmse = float(val_rmses[best_idx])
    return best_params, search_rmse, X_train_search.shape[0], search_seconds


def run_grid(layers, tags, seed, n_iter, search_fraction, n_jobs,
             early_stopping_rounds, noise_level, output_dir, save_models):
    arch_dir = output_dir / "architectures"
    arch_dir.mkdir(parents=True, exist_ok=True)
    if save_models:
        models_dir = output_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for n_layers in layers:
        for tag in tags:
            print(f"\n=== [{n_layers} layers, tag={tag}] tuning architecture ===")
            best_params, search_rmse, n_search, search_seconds = tune_one(
                n_layers, tag, seed, n_iter, search_fraction, n_jobs, early_stopping_rounds, noise_level
            )
            print(f"[{n_layers} layers, tag={tag}] search val RMSE (early-stopped, scaled)={search_rmse:.5f} "
                  f"on {n_search} training samples ({search_seconds:.1f}s)")

            arch_path = arch_dir / f"arch_{n_layers}layers_{tag}.json"
            with open(arch_path, "w") as f:
                json.dump(best_params, f, indent=2)

            print(f"=== [{n_layers} layers, tag={tag}] training/testing tuned architecture ===")
            t0 = time.time()
            metrics, artifact = train_and_test(
                n_layers=n_layers, tag=tag, seed=seed, architecture=best_params,
                early_stopping_rounds=early_stopping_rounds, noise_level=noise_level,
            )
            fit_seconds = time.time() - t0
            print(f"[{n_layers} layers, tag={tag}] test RMSE={metrics['rmse']:.5f} "
                  f"MAE={metrics['mae']:.5f} R2={metrics['r2']:.5f} ({fit_seconds:.1f}s)")

            if save_models:
                import joblib
                model_path = models_dir / f"xgb_{n_layers}layers_{tag}.joblib"
                joblib.dump(artifact, model_path)

            rows.append({
                "n_layers": n_layers,
                "tag": tag,
                "n_train": metrics["n_train"],
                "n_val": metrics["n_val"],
                "n_test": metrics["n_test"],
                "search_rmse": search_rmse,
                "search_seconds": search_seconds,
                "test_mae": metrics["mae"],
                "test_mse": metrics["mse"],
                "test_rmse": metrics["rmse"],
                "test_r2": metrics["r2"],
                "fit_seconds": fit_seconds,
                **best_params,
            })

    return pd.DataFrame(rows)


def summarize(df, tags, tag_sizes, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    size_map = dict(zip(tags, tag_sizes)) if len(tags) == len(tag_sizes) else {}
    df["dataset_size"] = df["tag"].map(size_map)

    csv_path = output_dir / "architecture_search_summary.csv"
    df.sort_values(["n_layers", "dataset_size"]).to_csv(csv_path, index=False)
    print(f"\n[search] summary -> {csv_path}")

    best_overall = df.loc[df["test_rmse"].idxmin()]
    best_per_layer = df.loc[df.groupby("n_layers")["test_rmse"].idxmin()].sort_values("n_layers")

    best_overall_path = output_dir / "best_overall.json"
    with open(best_overall_path, "w") as f:
        json.dump(best_overall.to_dict(), f, indent=2, default=str)
    best_per_layer_path = output_dir / "best_per_layer.csv"
    best_per_layer.to_csv(best_per_layer_path, index=False)

    print(f"[search] best overall combo -> {best_overall_path}")
    print(f"  n_layers={best_overall['n_layers']}, tag={best_overall['tag']} "
          f"(size={best_overall['dataset_size']}), test_rmse={best_overall['test_rmse']:.5f}")
    print(f"[search] best size/architecture per layer count -> {best_per_layer_path}")
    print(best_per_layer[["n_layers", "tag", "dataset_size", "test_rmse", "test_r2"]].to_string(index=False))

    # Heatmap: n_layers x dataset size -> test RMSE
    layer_counts = sorted(df["n_layers"].unique())
    if size_map:
        ordered_tags = [t for t in tags if t in df["tag"].unique()]
    else:
        ordered_tags = sorted(df["tag"].unique())
    pivot = df.pivot(index="n_layers", columns="tag", values="test_rmse").reindex(
        index=layer_counts, columns=ordered_tags
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    im = ax1.imshow(pivot.values, aspect="auto", cmap="viridis_r")
    ax1.set_xticks(range(len(ordered_tags)))
    ax1.set_xticklabels(
        [f"{size_map.get(t, t):,}" if size_map else t for t in ordered_tags], rotation=45, ha="right"
    )
    ax1.set_yticks(range(len(layer_counts)))
    ax1.set_yticklabels(layer_counts)
    ax1.set_xlabel("Dataset size")
    ax1.set_ylabel("Number of layers")
    ax1.set_title("Test RMSE by layers x dataset size")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax1.text(j, i, f"{val:.3f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax1, label="Test RMSE")

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(layer_counts)))
    for color, n_layers in zip(colors, layer_counts):
        sub = df[df["n_layers"] == n_layers].sort_values("dataset_size")
        ax2.plot(sub["dataset_size"], sub["test_rmse"], "o-", color=color, label=f"{n_layers} layers")
    ax2.set_xlabel("Dataset size")
    ax2.set_ylabel("Test RMSE")
    ax2.set_title("Test RMSE vs dataset size (own tuned architecture per point)")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plot_path = output_dir / "architecture_search.png"
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"[search] plot -> {plot_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, nargs="+", default=list(range(3, 13)),
                         help="Numbers of layers to search over")
    parser.add_argument("--tags", type=str, nargs="+", default=DEFAULT_TAGS,
                         help="Dataset-size tags to search over, matching create_database_synthetic.py "
                              "(ascending size order, used for plot ordering)")
    parser.add_argument("--tag-sizes", type=int, nargs="+", default=DEFAULT_TAG_SIZES,
                         help="Numeric dataset size for each --tags entry, same order, for axis labels/sorting")
    parser.add_argument("--n-iter", type=int, default=30, help="Number of random candidate architectures to try")
    parser.add_argument("--search-fraction", type=float, default=0.02,
                         help="Fraction of each combo's training data used to fit search candidates "
                              "(each is still early-stopped against the REAL full validation split)")
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--noise-level", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1,
                         help="Candidates evaluated in parallel across this many workers")
    parser.add_argument("--save-models", action="store_true",
                         help="Also save each combo's trained model bundle (.joblib). Off by default: "
                              "with many layers x sizes this is a lot of files.")
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_DIR / "architecture_search"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = run_grid(
        layers=args.layers, tags=args.tags, seed=args.seed,
        n_iter=args.n_iter, search_fraction=args.search_fraction,
        n_jobs=args.n_jobs, early_stopping_rounds=args.early_stopping_rounds,
        noise_level=args.noise_level, output_dir=output_dir, save_models=args.save_models,
    )
    summarize(df, args.tags, args.tag_sizes, output_dir)


if __name__ == "__main__":
    main()
