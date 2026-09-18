"""Train and test an XGBoost model on the field-configuration synthetic database.

Field equivalent of train_test_xgboost_synthetic.py, generalizing
notebooks/TrainAndTestXGBoostField.ipynb: same per-output XGBRegressor approach, but
built for the DUALEM field instrument's actual channel set rather than the full
synthetic one.

The field instrument only measures HCP/VCP at 4 m and 8 m spacing (no 2 m, no PRP),
and only measures in-phase at 4 m/8 m (no in-phase at 2 m) -- see src/FDEM.py's
FDEM_field(). So Data/data_9layers_new400_field.npy stores 10 raw columns
[h2op, h4op, h8op, h4ip, h8ip, v2op, v4op, v8op, v4ip, v8ip] (h2op/v2op are recorded
but unusable as features since they have no matching in-phase to form an
amplitude/phase pair), and the feature set built here is the resulting 8 columns:
h4amp, h8amp, v4amp, v8amp, h4pha, h8pha, v4pha, v8pha.

NOTE: notebooks/TrainAndTestXGBoostField.ipynb's own dataframe-construction cells
(cells 7-14) still assume the full 18-column synthetic layout and raise a shape
mismatch against the current field .npy files -- they were copy-pasted from the
synthetic notebook and never updated after the field database was regenerated with
FDEM_field's reduced channel set. This script instead builds features directly from
the 10-column raw layout that the field .npy files actually contain, verified against
FDEM_field()'s column order and against how the notebook's later (still-consistent)
cells decode Data/Field_data.npy's real HCP/VCP measurements into the same 8 features.

Expects Models/models_log_9layers_new400_field.npy and Data/data_9layers_new400_field.npy
to already exist.
"""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
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

N_LAYERS = 9
TAG = "new400_field"

# Raw column order in Data/data_9layers_new400_field.npy, per FDEM_field() in src/FDEM.py:
# HOP = [h2op, h4op, h8op], HIP = [h4ip, h8ip] (2m dropped -- no in-phase channel),
# VOP = [v2op, v4op, v8op], VIP = [v4ip, v8ip] (2m dropped), concatenated H then V.
RAW_COLUMNS = ["h2op", "h4op", "h8op", "h4ip", "h8ip", "v2op", "v4op", "v8op", "v4ip", "v8ip"]
# Configs usable as amplitude/phase features: only ones with both OP and IP recorded.
CONFIGS = ["h4", "h8", "v4", "v8"]

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


def load_database(tag=TAG):
    models_path = MODELS_DIR / f"models_log_{N_LAYERS}layers_{tag}.npy"
    data_path = DATA_DIR / f"data_{N_LAYERS}layers_{tag}.npy"
    if not models_path.exists() or not data_path.exists():
        raise FileNotFoundError(f"Missing field database: expected {models_path} and {data_path}")
    return np.load(models_path), np.load(data_path)


def build_features(data_raw):
    """Convert the 10-column raw field response into 8 log-amplitude/phase features."""
    raw = pd.DataFrame(data_raw, columns=RAW_COLUMNS)
    feats = {}
    for cfg in CONFIGS:
        complex_resp = raw[f"{cfg}op"] + 1j * raw[f"{cfg}ip"]
        feats[f"{cfg}amp"] = np.log10(np.abs(complex_resp))
        feats[f"{cfg}pha"] = np.angle(complex_resp)
    amp_cols = [f"{c}amp" for c in CONFIGS]
    pha_cols = [f"{c}pha" for c in CONFIGS]
    return pd.DataFrame(feats, columns=amp_cols + pha_cols)


def build_dataset(tag, seed):
    models_log, data_raw = load_database(tag)
    target_cols = [f"logs{i + 1}" for i in range(N_LAYERS)]
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

    X_temp, X_test, y_temp, y_test = train_test_split(X_norm, y_norm, test_size=test_size, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size, random_state=seed)
    return X_train, X_val, X_test, y_train, y_val, y_test, scale_X, scale_y


def tune_architecture(tag, seed, n_iter, cv, search_fraction, n_jobs):
    print(f"[tune] searching XGBoost architecture on the field ({tag}) database...")
    X, y = build_dataset(tag, seed)
    X_train, _, _, y_train, _, _, _, _ = scale_and_split(X, y, seed)

    _, X_search, _, y_search = train_test_split(X_train, y_train, test_size=search_fraction, random_state=seed)

    estimator = MultiOutputRegressor(XGBRegressor(objective="reg:squarederror"))
    param_dist = {f"estimator__{k}": v for k, v in DEFAULT_PARAM_DIST.items()}

    search = RandomizedSearchCV(
        estimator=estimator, param_distributions=param_dist, n_iter=n_iter,
        scoring="neg_mean_squared_error", cv=cv, random_state=seed, n_jobs=n_jobs, verbose=1,
    )
    search.fit(X_search, y_search)
    best_params = {k.replace("estimator__", ""): v for k, v in search.best_params_.items()}
    print(f"[tune] best architecture: {best_params}")
    return best_params


def train_and_test(tag, seed, architecture, early_stopping_rounds):
    X, y = build_dataset(tag, seed)
    X_train, X_val, X_test, y_train, y_val, y_test, scale_X, scale_y = scale_and_split(X, y, seed)

    models = []
    for i in range(N_LAYERS):
        model = XGBRegressor(**architecture, objective="reg:squarederror",
                              early_stopping_rounds=early_stopping_rounds, verbosity=0)
        model.fit(X_train, y_train[:, i], eval_set=[(X_val, y_val[:, i])], verbose=False)
        models.append(model)

    y_pred = np.column_stack([m.predict(X_test) for m in models])

    per_layer_rmse = [float(np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))) for i in range(N_LAYERS)]
    metrics = {
        "n_layers": N_LAYERS,
        "n_train": X_train.shape[0], "n_val": X_val.shape[0], "n_test": X_test.shape[0],
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "mse": float(mean_squared_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": float(r2_score(y_test, y_pred)),
        "per_layer_rmse": per_layer_rmse,
    }
    artifact = {
        "models": models, "scale_X": scale_X, "scale_y": scale_y,
        "feature_names": list(X.columns), "architecture": architecture,
    }
    return metrics, artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", type=str, default=TAG)
    parser.add_argument("--architecture-json", type=str, default=None,
                         help="Path to a JSON file with a pre-tuned architecture; skips RandomizedSearchCV if given")
    parser.add_argument("--tune-n-iter", type=int, default=30)
    parser.add_argument("--tune-cv", type=int, default=3)
    parser.add_argument("--search-fraction", type=float, default=0.02)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.architecture_json:
        with open(args.architecture_json) as f:
            architecture = json.load(f)
        print(f"[tune] loaded architecture from {args.architecture_json}: {architecture}")
    else:
        architecture = tune_architecture(args.tag, args.seed, args.tune_n_iter, args.tune_cv,
                                          args.search_fraction, args.n_jobs)
        arch_path = output_dir / f"xgb_architecture_{args.tag}.json"
        with open(arch_path, "w") as f:
            json.dump(architecture, f, indent=2)
        print(f"[tune] saved architecture -> {arch_path}")

    metrics, artifact = train_and_test(args.tag, args.seed, architecture, args.early_stopping_rounds)
    print(f"[{args.tag}] MAE={metrics['mae']:.5f} MSE={metrics['mse']:.5f} "
          f"RMSE={metrics['rmse']:.5f} R2={metrics['r2']:.5f}")

    MODELS_DIR.mkdir(exist_ok=True)
    model_path = MODELS_DIR / f"xgb_{N_LAYERS}layers_{args.tag}.joblib"
    joblib.dump(artifact, model_path)
    print(f"saved model bundle -> {model_path}")

    metrics_path = output_dir / f"xgb_field_metrics_{args.tag}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"saved metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
