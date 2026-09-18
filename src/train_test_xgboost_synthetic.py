"""Train and test XGBoost models across synthetic FDEM databases with different layer counts.

Generalizes notebooks/TrainAndTestXGBoostSynthetic.ipynb: builds the same log-amplitude/phase
feature set and per-output XGBRegressor architecture used there, tunes the XGBoost
hyperparameters once (RandomizedSearchCV) on a single reference layer count, then reuses that
identical architecture to train and evaluate one model per requested layer count (3-9 by
default) so the results are directly comparable across databases.

Expects the synthetic databases to already exist under Models/ and Data/, as produced by
src/create_database_synthetic.py (models_log_{n}layers_{tag}.npy / data_{n}layers_{tag}.npy).
"""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "Models"
DATA_DIR = ROOT_DIR / "Data"
RESULTS_DIR = ROOT_DIR / "Results"

# Raw FDEM response columns: fixed regardless of the number of conductivity layers,
# since they only depend on coil spacings/orientations (H, V, P at 2/4/8 m).
RAW_COLUMNS = [
    "h2op", "h4op", "h8op", "h2ip", "h4ip", "h8ip",
    "v2op", "v4op", "v8op", "v2ip", "v4ip", "v8ip",
    "p2op", "p4op", "p8op", "p2ip", "p4ip", "p8ip",
]
CONFIGS = ["h2", "h4", "h8", "v2", "v4", "v8", "p2", "p4", "p8"]

DEFAULT_PARAM_DIST = {
    "n_estimators": [300, 350, 400, 450, 500, 550, 600],
    "max_depth": [4, 6, 8, 10, 12, 14, 16, 18, 20],
    "min_child_weight": [1, 3, 5, 7, 9],
    "subsample": np.linspace(0.5, 1, 5).tolist(),
    "colsample_bytree": np.linspace(0.5, 1, 5).tolist(),
    "gamma": [0, 0.1, 0.3, 0.5, 1],
    "reg_alpha": [0, 0.01, 0.1, 1, 10],
    "reg_lambda": [0.01, 0.1, 1, 10, 100],
    "learning_rate": np.linspace(0.01, 0.2, 10).tolist(),
}


def load_database(n_layers, tag):
    models_path = MODELS_DIR / f"models_log_{n_layers}layers_{tag}.npy"
    data_path = DATA_DIR / f"data_{n_layers}layers_{tag}.npy"
    if not models_path.exists() or not data_path.exists():
        raise FileNotFoundError(
            f"Missing database for {n_layers} layers (tag='{tag}'). Expected "
            f"{models_path} and {data_path}. Generate it first with, e.g.:\n"
            f"  python src/create_database_synthetic.py --layers {n_layers} --tag {tag}"
        )
    models_log = np.load(models_path)
    data = np.load(data_path)
    return models_log, data


def add_noise(data_raw, noise_level, seed):
    """Add relative Gaussian noise to raw OP/IP responses.

    Each value gets independent noise ~ N(0, (noise_level * |value|)^2), so the
    noise scales with signal magnitude like typical FDEM instrument noise.
    """
    if noise_level <= 0:
        return data_raw
    rng = np.random.default_rng(seed)
    return data_raw + rng.normal(0.0, noise_level * np.abs(data_raw))


def add_noise_range(data_raw, max_noise_level, seed):
    """Add relative Gaussian noise like `add_noise`, but with each sample getting
    its own noise level drawn uniformly from [0, max_noise_level], instead of one
    fixed level for the whole dataset.

    Used to train a noise-robust model: exposing the model to a spread of noise
    magnitudes during training (rather than only clean data) so it learns to
    degrade gracefully instead of extrapolating wildly on noisy inputs.
    """
    if max_noise_level <= 0:
        return data_raw
    rng = np.random.default_rng(seed)
    sample_noise_level = rng.uniform(0.0, max_noise_level, size=(data_raw.shape[0], 1))
    return data_raw + rng.normal(0.0, 1.0, size=data_raw.shape) * sample_noise_level * np.abs(data_raw)


def build_features(data_raw):
    """Convert raw OP/IP responses into log-amplitude/phase features."""
    raw = pd.DataFrame(data_raw, columns=RAW_COLUMNS)
    feats = {}
    for cfg in CONFIGS:
        complex_resp = raw[f"{cfg}op"] + 1j * raw[f"{cfg}ip"]
        feats[f"{cfg}amp"] = np.log10(np.abs(complex_resp))
        feats[f"{cfg}pha"] = np.angle(complex_resp)
    amp_cols = [f"{c}amp" for c in CONFIGS]
    pha_cols = [f"{c}pha" for c in CONFIGS]
    return pd.DataFrame(feats, columns=amp_cols + pha_cols)


def build_dataset(n_layers, tag, seed, noise_level=0.0, noise_augment_max=None):
    """Load a database, engineer features, and shuffle (X, y) in step with each other.

    noise_augment_max, if given, overrides noise_level and instead gives every
    sample its own random noise level in [0, noise_augment_max] (see add_noise_range).
    """
    models_log, data_raw = load_database(n_layers, tag)
    if noise_augment_max is not None:
        data_raw = add_noise_range(data_raw, noise_augment_max, seed)
    else:
        data_raw = add_noise(data_raw, noise_level, seed)
    target_cols = [f"logs{i + 1}" for i in range(n_layers)]
    y = pd.DataFrame(models_log, columns=target_cols)
    X = build_features(data_raw)

    combined = pd.concat([y, X], axis=1).sample(frac=1, random_state=seed)
    X = combined[X.columns]
    y = combined[target_cols]
    return X, y


def scale_and_split(X, y, seed, test_size=0.2, val_size=0.25):
    scale_X = MinMaxScaler()
    scale_y = MinMaxScaler()
    X_norm = scale_X.fit_transform(X)
    y_norm = scale_y.fit_transform(y)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X_norm, y_norm, test_size=test_size, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=seed
    )
    return X_train, X_val, X_test, y_train, y_val, y_test, scale_X, scale_y


def tune_architecture(n_layers, tag, seed, n_iter, cv, search_fraction, n_jobs, noise_level=0.0,
                       noise_augment_max=None):
    """Run one RandomizedSearchCV to pick a single hyperparameter set shared by every
    output and every layer-count database (same architecture for all)."""
    print(f"[tune] searching XGBoost architecture on the {n_layers}-layer database...")
    X, y = build_dataset(n_layers, tag, seed, noise_level=noise_level,
                          noise_augment_max=noise_augment_max)
    X_train, _, _, y_train, _, _, _, _ = scale_and_split(X, y, seed)

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


def train_and_test(n_layers, tag, seed, architecture, early_stopping_rounds, noise_level=0.0,
                    noise_augment_max=None):
    X, y = build_dataset(n_layers, tag, seed, noise_level=noise_level,
                          noise_augment_max=noise_augment_max)
    X_train, X_val, X_test, y_train, y_val, y_test, scale_X, scale_y = scale_and_split(
        X, y, seed
    )

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

    y_pred = np.column_stack([m.predict(X_test) for m in models])

    per_layer_rmse = [
        float(np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i])))
        for i in range(n_layers)
    ]
    metrics = {
        "n_layers": n_layers,
        "n_train": X_train.shape[0],
        "n_val": X_val.shape[0],
        "n_test": X_test.shape[0],
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "mse": float(mean_squared_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": float(r2_score(y_test, y_pred)),
        "per_layer_rmse": per_layer_rmse,
    }

    artifact = {
        "models": models,
        "scale_X": scale_X,
        "scale_y": scale_y,
        "feature_names": list(X.columns),
        "architecture": architecture,
    }
    return metrics, artifact


def save_comparison(all_metrics, tag, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = [
        {k: v for k, v in m.items() if k != "per_layer_rmse"} for m in all_metrics
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / f"xgb_synthetic_comparison_{tag}_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    detail_rows = [
        {"n_layers": m["n_layers"], "layer_index": i + 1, "rmse": rmse}
        for m in all_metrics
        for i, rmse in enumerate(m["per_layer_rmse"])
    ]
    detail_df = pd.DataFrame(detail_rows)
    detail_path = output_dir / f"xgb_synthetic_comparison_{tag}_per_layer.csv"
    detail_df.to_csv(detail_path, index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(summary_df["n_layers"], summary_df["rmse"], "o-", color="tab:blue", label="RMSE")
    ax1.set_xlabel("Number of layers")
    ax1.set_ylabel("Overall RMSE (log10 S/m, normalized)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax1b = ax1.twinx()
    ax1b.plot(summary_df["n_layers"], summary_df["r2"], "s--", color="tab:red", label="R2")
    ax1b.set_ylabel("R2", color="tab:red")
    ax1b.tick_params(axis="y", labelcolor="tab:red")
    ax1.set_title("Overall test performance vs. layer count")

    for m in all_metrics:
        layer_idx = np.arange(1, m["n_layers"] + 1) / m["n_layers"]
        ax2.plot(layer_idx, m["per_layer_rmse"], "o-", label=f"{m['n_layers']} layers")
    ax2.set_xlabel("Relative depth position (layer / n_layers)")
    ax2.set_ylabel("Per-layer RMSE")
    ax2.set_title("Per-layer RMSE by database")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plot_path = output_dir / f"xgb_synthetic_comparison_{tag}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)

    print(f"[compare] summary  -> {summary_path}")
    print(f"[compare] per-layer -> {detail_path}")
    print(f"[compare] plot     -> {plot_path}")
    return summary_df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, nargs="+", default=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                         help="Numbers of layers to train/test models for")
    parser.add_argument("--tag", type=str, default="new400",
                         help="Suffix tag identifying the databases, matching create_database_synthetic.py")
    parser.add_argument("--tune-layers", type=int, default=None,
                         help="Layer count whose database is used to tune the shared architecture "
                              "(default: the largest value in --layers)")
    parser.add_argument("--architecture-json", type=str, default=None,
                         help="Path to a JSON file with a pre-tuned architecture; skips RandomizedSearchCV if given")
    parser.add_argument("--tune-n-iter", type=int, default=30, help="RandomizedSearchCV n_iter")
    parser.add_argument("--tune-cv", type=int, default=3, help="RandomizedSearchCV cv folds")
    parser.add_argument("--search-fraction", type=float, default=0.02,
                         help="Fraction of the tuning layer's training data used for the hyperparameter search")
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--noise-level", type=float, default=0.0,
                         help="Relative Gaussian noise std as a fraction of each raw OP/IP "
                              "response's magnitude, applied before feature engineering "
                              "(0 disables noise; e.g. 0.02 for ~2%% noise)")
    parser.add_argument("--noise-augment-max", type=float, default=None,
                         help="If set, overrides --noise-level: every training sample gets its "
                              "own random noise level drawn uniformly from [0, this value], so "
                              "the model is trained on a spread of noise magnitudes instead of "
                              "one fixed level (e.g. 0.10 for up to 10%% noise). Use this to "
                              "train a noise-robust model.")
    parser.add_argument("--model-suffix", type=str, default="",
                         help="Extra suffix appended to the saved model bundle filename, e.g. "
                              "'_noiseaug', to avoid overwriting an existing bundle for the same tag")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1, help="n_jobs for RandomizedSearchCV")
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_DIR),
                         help="Directory for comparison CSVs/plot")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    tune_layers = args.tune_layers or max(args.layers)

    if args.architecture_json:
        with open(args.architecture_json) as f:
            architecture = json.load(f)
        print(f"[tune] loaded architecture from {args.architecture_json}: {architecture}")
    else:
        architecture = tune_architecture(
            n_layers=tune_layers,
            tag=args.tag,
            seed=args.seed,
            n_iter=args.tune_n_iter,
            cv=args.tune_cv,
            search_fraction=args.search_fraction,
            n_jobs=args.n_jobs,
            noise_level=args.noise_level,
            noise_augment_max=args.noise_augment_max,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        arch_path = output_dir / f"xgb_architecture_{args.tag}{args.model_suffix}.json"
        with open(arch_path, "w") as f:
            json.dump(architecture, f, indent=2)
        print(f"[tune] saved architecture -> {arch_path}")

    MODELS_DIR.mkdir(exist_ok=True)

    all_metrics = []
    for n_layers in args.layers:
        print(f"\n=== Training/testing {n_layers}-layer model ===")
        metrics, artifact = train_and_test(
            n_layers=n_layers,
            tag=args.tag,
            seed=args.seed,
            architecture=architecture,
            early_stopping_rounds=args.early_stopping_rounds,
            noise_level=args.noise_level,
            noise_augment_max=args.noise_augment_max,
        )
        print(
            f"[{n_layers} layers] MAE={metrics['mae']:.5f} MSE={metrics['mse']:.5f} "
            f"RMSE={metrics['rmse']:.5f} R2={metrics['r2']:.5f}"
        )

        model_path = MODELS_DIR / f"xgb_{n_layers}layers_{args.tag}{args.model_suffix}.joblib"
        joblib.dump(artifact, model_path)
        print(f"[{n_layers} layers] saved model bundle -> {model_path}")

        all_metrics.append(metrics)

    save_comparison(all_metrics, args.tag, output_dir)


if __name__ == "__main__":
    main()
