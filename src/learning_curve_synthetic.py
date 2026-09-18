"""Learning-curve analysis for the synthetic FDEM databases.

Answers the reviewer question: is 400,000 training samples "sufficiently large"?
Trains/evaluates the same XGBoost architecture on increasing dataset sizes drawn
from one shared shuffled pool (so every smaller size is a strict prefix of every
larger size, i.e. we are only ever adding data, never resampling it) and reports
validation/test RMSE vs. dataset size, to show explicitly where performance
plateaus.

Runs this sweep once per requested layer count (--layers accepts several values),
since the LHS parameter space grows with the number of layers -- a dataset size
that plateaus for a 3-layer database may not for a 9-layer one. All layer counts
share one architecture (tuned once, or supplied via --architecture-json), so
differences between layer-count curves reflect data volume and problem
dimensionality, not re-tuned model complexity.

Generates the required database itself (under Models/ and Data/, via
src/create_database_synthetic.py) if it doesn't already exist or is too small
for the largest requested --sizes value.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from create_database_synthetic import generate_database  # noqa: E402
from train_test_xgboost_synthetic import (  # noqa: E402
    DEFAULT_PARAM_DIST,
    build_features,
    load_database,
    scale_and_split,
)

RESULTS_DIR = ROOT_DIR / "Results"
MODELS_DIR = ROOT_DIR / "Models"
DATA_DIR = ROOT_DIR / "Data"


def ensure_database(n_layers, tag, n_extract):
    models_path = MODELS_DIR / f"models_log_{n_layers}layers_{tag}.npy"
    data_path = DATA_DIR / f"data_{n_layers}layers_{tag}.npy"
    if models_path.exists() and data_path.exists():
        return
    print(f"[data] {n_layers}-layer '{tag}': generating {n_extract:,} samples...")
    t0 = time.time()
    generate_database(n_layers=n_layers, n_models=max(n_extract, 100_000), n_extract=n_extract,
                       sigma_min=1e-3, sigma_max=1.0, height=0.1, random_seed=42, tag=tag,
                       n_jobs=-1)
    print(f"[data] {n_layers}-layer '{tag}': done in {time.time() - t0:.0f}s")

DEFAULT_SIZES = [50_000, 100_000, 200_000, 300_000, 400_000, 500_000, 600_000, 700_000, 800_000]


def build_full_pool(n_layers, tag, seed):
    """Load a database and return one fixed shuffled (X, y) pool.

    Prefixes of this pool (X.iloc[:n]) are used as the "dataset of size n" for
    every n, so growing the dataset size never changes which rows are already
    included -- it only adds new ones.
    """
    models_log, data_raw = load_database(n_layers, tag)
    target_cols = [f"logs{i + 1}" for i in range(n_layers)]
    y_full = pd.DataFrame(models_log, columns=target_cols)
    X_full = build_features(data_raw)

    combined = pd.concat([y_full, X_full], axis=1).sample(frac=1, random_state=seed).reset_index(drop=True)
    X_full = combined[X_full.columns]
    y_full = combined[target_cols]
    return X_full, y_full


def tune_architecture(X_sub, y_sub, seed, n_iter, cv, search_fraction, n_jobs):
    """One RandomizedSearchCV run to pick a single architecture shared by every
    dataset size, so the learning curve isolates the effect of data volume
    rather than re-optimizing model complexity at each size."""
    X_train, _, _, y_train, _, _, _, _ = scale_and_split(X_sub, y_sub, seed)
    _, X_search, _, y_search = train_test_split(
        X_train, y_train, test_size=search_fraction, random_state=seed
    )

    estimator = MultiOutputRegressor(XGBRegressor(objective="reg:squarederror"))
    param_dist = {f"estimator__{k}": v for k, v in DEFAULT_PARAM_DIST.items()}

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="neg_mean_squared_error",
        cv=cv,
        random_state=seed,
        n_jobs=n_jobs,
        verbose=1,
    )
    search.fit(X_search, y_search)
    best_params = {k.replace("estimator__", ""): v for k, v in search.best_params_.items()}
    print(f"[tune] best architecture: {best_params}")
    return best_params


def evaluate_size(X_full, y_full, size, seed, architecture, early_stopping_rounds, n_layers):
    X_sub = X_full.iloc[:size]
    y_sub = y_full.iloc[:size]
    X_train, X_val, X_test, y_train, y_val, y_test, _, _ = scale_and_split(X_sub, y_sub, seed)

    t0 = time.time()
    models = []
    for i in range(n_layers):
        model = XGBRegressor(
            **architecture,
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
    train_seconds = time.time() - t0

    y_val_pred = np.column_stack([m.predict(X_val) for m in models])
    y_test_pred = np.column_stack([m.predict(X_test) for m in models])

    return {
        "dataset_size": size,
        "n_train": X_train.shape[0],
        "n_val": X_val.shape[0],
        "n_test": X_test.shape[0],
        "val_rmse": float(np.sqrt(mean_squared_error(y_val, y_val_pred))),
        "val_mae": float(mean_absolute_error(y_val, y_val_pred)),
        "val_r2": float(r2_score(y_val, y_val_pred)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
        "test_mae": float(mean_absolute_error(y_test, y_test_pred)),
        "test_r2": float(r2_score(y_test, y_test_pred)),
        "train_seconds": train_seconds,
    }


def save_results(rows, tag, chosen_size, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values(["n_layers", "dataset_size"]).reset_index(drop=True)
    df["val_rmse_pct_change"] = df.groupby("n_layers")["val_rmse"].pct_change() * 100

    csv_path = output_dir / f"learning_curve_{tag}.csv"
    df.to_csv(csv_path, index=False)

    layer_counts = sorted(df["n_layers"].unique())
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(layer_counts)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for color, n_layers in zip(colors, layer_counts):
        sub = df[df["n_layers"] == n_layers]
        ax1.plot(sub["dataset_size"], sub["val_rmse"], "o-", color=color, label=f"{n_layers} layers")
        ax2.plot(sub["dataset_size"], sub["test_rmse"], "s-", color=color, label=f"{n_layers} layers")
    if chosen_size in df["dataset_size"].values:
        ax1.axvline(chosen_size, color="gray", linestyle=":")
        ax2.axvline(chosen_size, color="gray", linestyle=":", label=f"Chosen size ({chosen_size:,})")
    ax1.set_xlabel("Dataset size (samples)")
    ax1.set_ylabel("RMSE (log10 S/m, normalized)")
    ax1.set_title("Validation RMSE")
    ax1.grid(alpha=0.3)
    ax2.set_xlabel("Dataset size (samples)")
    ax2.set_title("Test RMSE")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)
    fig.suptitle("XGBoost learning curve: synthetic database")
    plt.tight_layout()
    plot_path = output_dir / f"learning_curve_{tag}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)

    print(f"[learning-curve] summary -> {csv_path}")
    print(f"[learning-curve] plot    -> {plot_path}")
    print(df.to_string(index=False))
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, nargs="+", default=[9],
                         help="Numbers of layers to run the learning curve for (database must exist for each)")
    parser.add_argument("--tune-layers", type=int, default=None,
                         help="Layer count whose database is used to tune the shared architecture "
                              "(default: the largest value in --layers)")
    parser.add_argument("--tag", type=str, required=True,
                         help="Suffix tag identifying the database, matching create_database_synthetic.py")
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES,
                         help="Dataset sizes to evaluate, ascending")
    parser.add_argument("--chosen-size", type=int, default=400_000,
                         help="Dataset size to mark on the plot as the one used in the paper")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--architecture-json", type=str, default=None,
                         help="Path to a JSON file with a pre-tuned architecture; skips RandomizedSearchCV if given "
                              "(e.g. the xgb_architecture_{tag}.json produced by train_test_xgboost_synthetic.py, "
                              "to keep both experiments on the exact same architecture)")
    parser.add_argument("--tune-size", type=int, default=None,
                         help="Dataset size whose subset is used to tune the shared architecture "
                              "(default: the largest value in --sizes)")
    parser.add_argument("--tune-n-iter", type=int, default=30, help="RandomizedSearchCV n_iter")
    parser.add_argument("--tune-cv", type=int, default=3, help="RandomizedSearchCV cv folds")
    parser.add_argument("--search-fraction", type=float, default=0.02,
                         help="Fraction of the tuning size's training data used for the hyperparameter search")
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--n-jobs", type=int, default=-1, help="n_jobs for RandomizedSearchCV")
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    sizes = sorted(args.sizes)
    tune_size = args.tune_size or sizes[-1]
    tune_layers = args.tune_layers or max(args.layers)

    pools = {}
    for n_layers in args.layers:
        ensure_database(n_layers, args.tag, sizes[-1])
        print(f"[pool] loading {n_layers}-layer database (tag='{args.tag}')...")
        X_full, y_full = build_full_pool(n_layers, args.tag, args.seed)
        if X_full.shape[0] < sizes[-1]:
            raise ValueError(
                f"{n_layers}-layer database only has {X_full.shape[0]} samples, but the largest "
                f"requested size is {sizes[-1]}. Regenerate with a larger --n-extract in "
                f"create_database_synthetic.py."
            )
        pools[n_layers] = (X_full, y_full)

    if args.architecture_json:
        with open(args.architecture_json) as f:
            architecture = json.load(f)
        print(f"[tune] loaded architecture from {args.architecture_json}: {architecture}")
    else:
        print(f"[tune] tuning shared architecture on a {tune_size}-sample subset of the "
              f"{tune_layers}-layer database...")
        X_tune, y_tune = pools[tune_layers]
        architecture = tune_architecture(
            X_tune.iloc[:tune_size], y_tune.iloc[:tune_size],
            seed=args.seed, n_iter=args.tune_n_iter, cv=args.tune_cv,
            search_fraction=args.search_fraction, n_jobs=args.n_jobs,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        arch_path = output_dir / f"learning_curve_architecture_{args.tag}.json"
        with open(arch_path, "w") as f:
            json.dump(architecture, f, indent=2)
        print(f"[tune] saved architecture -> {arch_path}")

    rows = []
    for n_layers in args.layers:
        X_full, y_full = pools[n_layers]
        for size in sizes:
            print(f"\n=== Evaluating {n_layers}-layer model, dataset size {size:,} ===")
            t0 = time.time()
            metrics = evaluate_size(
                X_full, y_full, size, args.seed, architecture,
                args.early_stopping_rounds, n_layers,
            )
            metrics["n_layers"] = n_layers
            print(
                f"[{n_layers} layers, {size:,}] val_RMSE={metrics['val_rmse']:.5f} "
                f"test_RMSE={metrics['test_rmse']:.5f} val_R2={metrics['val_r2']:.5f} "
                f"({time.time() - t0:.1f}s)"
            )
            rows.append(metrics)

    save_results(rows, args.tag, args.chosen_size, output_dir)


if __name__ == "__main__":
    main()
