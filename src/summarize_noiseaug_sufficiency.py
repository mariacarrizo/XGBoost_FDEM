"""Build the figures/tables that defend "400k samples is sufficient" for the
noise-augmented 9-layer XGBoost model, from the results already produced by:
  - src/architecture_search_noiseaug.py  (held-out RMSE + found architectures per size)
  - src/gn_baseline_synthetic.py, run per size with the *_noiseaug_tuned bundles
    (MC/MR transect model error vs. GN-homogeneous baseline)

Produces, under Results/noiseaug_sufficiency/:
  - architecture_convergence.csv/.tex : the 5 found architectures, side by side
  - holdout_rmse_vs_size.png          : held-out RMSE vs. dataset size, by noise level
  - mc_mr_model_error_vs_size.png     : MC/MR transect model error vs. size, by noise
                                        level, with the GN-homogeneous baseline as a
                                        reference band
  - diminishing_returns.csv/.tex      : % improvement per metric going from 400k to
                                        each larger size, to quantify "sufficiency"

Run inside the `xgb-fdem` conda env:
    conda run -n xgb-fdem python src/summarize_noiseaug_sufficiency.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "Results"
OUT_DIR = RESULTS_DIR / "noiseaug_sufficiency"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Validated colorblind-safe categorical palette (dataviz skill), slots 1-3.
COLOR_0 = "#2a78d6"  # blue
COLOR_5 = "#eb6834"  # orange
COLOR_10 = "#1baf7a"  # aqua
NOISE_COLORS = {0.0: COLOR_0, 0.05: COLOR_5, 0.10: COLOR_10}
NOISE_MARKERS = {0.0: "o", 0.05: "s", 0.10: "^"}

SIZES = [
    ("new400", 400_000, "400k"),
    ("d10m800k", 800_000, "800k"),
    ("d10m1600k", 1_600_000, "1.6M"),
    ("d10m3200k", 3_200_000, "3.2M"),
    ("d10m4000k", 4_000_000, "4M"),
]


def to_latex_table(df, float_format="%.4g"):
    """Minimal DataFrame -> LaTeX tabular writer (avoids the jinja2 dependency
    pandas' own DataFrame.to_latex/Styler path requires)."""
    def fmt(v):
        if isinstance(v, float):
            return float_format % v
        return str(v)

    lines = [
        "\\begin{tabular}{" + "l" * len(df.columns) + "}",
        "\\toprule",
        " & ".join(str(c) for c in df.columns) + " \\\\",
        "\\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join(fmt(v) for v in row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def load_architecture_table():
    df = pd.read_csv(RESULTS_DIR / "architecture_search_noiseaug" / "architectures_found_by_size.csv")
    label_map = {tag: label for tag, _, label in SIZES}
    df["size_label"] = df["tag"].map(label_map)
    cols = ["size_label", "max_depth", "n_estimators", "learning_rate", "subsample",
            "colsample_bytree", "min_child_weight", "gamma", "reg_alpha", "reg_lambda"]
    df = df[cols].rename(columns={"size_label": "Training samples"})
    df.to_csv(OUT_DIR / "architecture_convergence.csv", index=False)
    with open(OUT_DIR / "architecture_convergence.tex", "w") as f:
        f.write(to_latex_table(df, float_format="%.4g"))
    print(f"[table] architecture convergence -> {OUT_DIR / 'architecture_convergence.csv'}")
    return df


def load_holdout_rmse():
    return pd.read_csv(RESULTS_DIR / "architecture_search_noiseaug" / "architecture_search_noiseaug_summary.csv")


def load_mc_mr():
    rows = []
    for tag, n_extract, label in SIZES:
        csv_path = RESULTS_DIR / f"gn_baseline_synthetic_{tag}_noiseaug_tuned" / f"gn_baseline_summary_{tag}_noiseaug_tuned.csv"
        df = pd.read_csv(csv_path)
        df["tag"] = tag
        df["n_train_samples"] = n_extract
        df["size_label"] = label
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def plot_holdout_rmse(df):
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for noise_level, sub in df.groupby("noise_level"):
        sub = sub.sort_values("n_train_samples")
        ax.plot(sub["n_train_samples"], sub["rmse_log10_sigma"],
                color=NOISE_COLORS[noise_level], marker=NOISE_MARKERS[noise_level],
                markersize=7, linewidth=2, label=f"{noise_level:.0%} noise")
    ax.axvline(400_000, color="#898781", linestyle=":", linewidth=1.2, zorder=0)
    ax.text(400_000 * 1.08, ax.get_ylim()[1], "400k", color="#52514e", fontsize=9,
            va="top", ha="left")
    ax.set_xscale("log")
    ax.set_xlabel("Training samples")
    ax.set_ylabel("Held-out RMSE (log$_{10}$ $\\sigma$)")
    ax.set_title("Accuracy plateaus beyond 400k training samples", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path = OUT_DIR / "holdout_rmse_vs_size.png"
    fig.savefig(out_path, dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[figure] -> {out_path}")


def plot_mc_mr(df):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    scenario_labels = {"MC": "Middle conductive body", "MR": "Middle resistive body"}
    for ax, scenario in zip(axes, ["MC", "MR"]):
        sub_all = df[df["scenario"] == scenario]
        for noise_level, sub in sub_all.groupby("noise_level"):
            sub = sub.sort_values("n_train_samples")
            ax.plot(sub["n_train_samples"], sub["model_error_xgboost_pct"],
                    color=NOISE_COLORS[noise_level], marker=NOISE_MARKERS[noise_level],
                    markersize=7, linewidth=2, label=f"XGBoost, {noise_level:.0%} noise")
            gn = sub["model_error_gn_homog_pct"].iloc[0]  # ~constant across size, by construction
            ax.axhline(sub["model_error_gn_homog_pct"].mean(), color=NOISE_COLORS[noise_level],
                       linewidth=1.2, linestyle="--", alpha=0.55)
        ax.axvline(400_000, color="#898781", linestyle=":", linewidth=1.2, zorder=0)
        ax.set_xscale("log")
        ax.set_xlabel("Training samples")
        ax.set_title(scenario_labels[scenario])
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e1e0d9", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Model error (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("XGBoost (solid) vs. GN-homogeneous baseline (dashed) -- model error vs. dataset size",
                 y=1.18, fontsize=11)
    fig.tight_layout()
    out_path = OUT_DIR / "mc_mr_model_error_vs_size.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] -> {out_path}")


def diminishing_returns_table(holdout_df, mc_mr_df):
    rows = []
    base_size = 400_000
    for noise_level, sub in holdout_df.groupby("noise_level"):
        sub = sub.sort_values("n_train_samples")
        base = sub.loc[sub["n_train_samples"] == base_size, "rmse_log10_sigma"].iloc[0]
        for _, r in sub.iterrows():
            pct_change = 100 * (base - r["rmse_log10_sigma"]) / base
            rows.append({
                "metric": "held-out RMSE (log10 sigma)",
                "noise_level": noise_level,
                "n_train_samples": r["n_train_samples"],
                "value": r["rmse_log10_sigma"],
                "pct_improvement_vs_400k": pct_change,
            })
    for scenario in ["MC", "MR"]:
        for noise_level, sub in mc_mr_df[mc_mr_df["scenario"] == scenario].groupby("noise_level"):
            sub = sub.sort_values("n_train_samples")
            base = sub.loc[sub["n_train_samples"] == base_size, "model_error_xgboost_pct"].iloc[0]
            for _, r in sub.iterrows():
                pct_change = 100 * (base - r["model_error_xgboost_pct"]) / base
                rows.append({
                    "metric": f"{scenario} model error (%)",
                    "noise_level": noise_level,
                    "n_train_samples": r["n_train_samples"],
                    "value": r["model_error_xgboost_pct"],
                    "pct_improvement_vs_400k": pct_change,
                })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "diminishing_returns.csv", index=False)
    with open(OUT_DIR / "diminishing_returns.tex", "w") as f:
        f.write(to_latex_table(out, float_format="%.3g"))
    print(f"[table] diminishing returns -> {OUT_DIR / 'diminishing_returns.csv'}")
    return out


def main():
    arch_df = load_architecture_table()
    print(arch_df.to_string(index=False))

    holdout_df = load_holdout_rmse()
    plot_holdout_rmse(holdout_df)

    mc_mr_df = load_mc_mr()
    plot_mc_mr(mc_mr_df)

    dr = diminishing_returns_table(holdout_df, mc_mr_df)
    print(dr.to_string(index=False))


if __name__ == "__main__":
    main()
