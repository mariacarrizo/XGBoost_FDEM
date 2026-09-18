"""Shared plotting helpers for the XGBoost train/test manuscript figures.

Factors out the figure-generation code that was duplicated between
notebooks/TrainAndTestXGBoostSynthetic.ipynb and notebooks/TrainAndTestXGBoostField.ipynb
(feature importance, density plot, error histogram, train/test profile fits) so both the
synthetic and field driver scripts (make_manuscript_figures_synthetic.py /
make_manuscript_figures_field.py) can produce the same figures from a single
implementation. Each function only draws a figure and saves it; loading data, models,
and running the forward model stays in the driver scripts since that differs between
the synthetic and field configurations.
"""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split


def reconstruct_split(bundle, X, y, seed, test_size=0.2, val_size=0.25):
    """Re-derive the same train/val/test split a train_and_test() run produced, using
    the bundle's already-fitted scalers (not refit) and the same split seed -- so
    figures always reflect the exact test set the saved model was evaluated on.
    """
    X_norm = bundle["scale_X"].transform(X)
    y_norm = bundle["scale_y"].transform(y)
    X_temp, X_test, y_temp, y_test = train_test_split(X_norm, y_norm, test_size=test_size, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size, random_state=seed)
    return X_train, X_val, X_test, y_train, y_val, y_test


def pick_random_examples(X, y, n=10, seed=10):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=n, replace=False)
    return X[idx], y[idx]


def op_ip_to_amp_pha(data_op, data_ip):
    data_complex = data_op + 1j * data_ip
    return np.log10(np.abs(data_complex)), np.angle(data_complex)


def amp_pha_to_op_ip(data_log_amp, data_pha):
    data_op = (10 ** data_log_amp) * np.cos(data_pha)
    data_ip = (10 ** data_log_amp) * np.sin(data_pha)
    return data_op, data_ip


def plot_correlation_heatmap(X_df, out_path, title="Correlation Matrix"):
    plt.figure(figsize=(14, 9))
    sns.heatmap(X_df.corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".1f")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_scatter_matrix(X_df, out_path, sample_n=1000, seed=42):
    n = min(sample_n, len(X_df))
    datasamp = X_df.sample(n, random_state=seed)
    axes = pd.plotting.scatter_matrix(datasamp, alpha=0.2, figsize=(12, 12))
    for ax in axes.flatten():
        ax.xaxis.label.set_rotation(90)
        ax.yaxis.label.set_rotation(0)
        ax.yaxis.label.set_ha("right")
    plt.gcf().subplots_adjust(wspace=0, hspace=0)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_feature_importance_pie(models, keys, colors, hatch, out_path, ncols=3):
    """One pie chart per output layer, sliced by gain-based feature importance.

    `keys`/`colors`/`hatch` are parallel arrays over the model's feature columns
    (amplitude features first, then phase features, matching build_features()'s
    column order), one entry per feature.
    """
    n_layers = len(models)
    nrows = int(np.ceil(n_layers / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2 * nrows))
    ax = np.atleast_1d(axes).ravel()

    for i, model in enumerate(models):
        booster = model.get_booster()
        importance = booster.get_score(importance_type="gain")
        # get_score only returns features that were actually split on; keep the
        # feature order fixed and default missing ones to 0 gain.
        gains = [importance.get(f"f{j}", 0.0) for j in range(len(keys))]
        ax[i].pie(gains, colors=colors, hatch=hatch, textprops={"size": "smaller"})
        ax[i].set_title(f"$\\sigma_{{{i + 1}}}$")
    for j in range(n_layers, len(ax)):
        ax[j].axis("off")

    patches = [mpatches.Patch(facecolor=c, hatch=h, label=k) for c, h, k in zip(colors, hatch, keys)]
    fig.legend(handles=patches, loc="lower center", ncol=min(6, len(keys)), bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_feature_importance_bars(models, keys, colors, hatch, out_path, bar_width=0.35, gap=0.12):
    """Two stacked bars per output layer -- one for amplitude features, one for
    phase features -- sliced by gain-based feature importance.

    Gains are normalized by that layer's total gain across all features (amplitude
    and phase combined), so the two bars for a layer always sum to 100% -- this
    keeps every layer on the same, comparable scale (raw XGBoost gain values are
    not comparable across the separately-trained per-layer regressors) and makes
    the amplitude/phase balance directly visible, since a bar's height is that
    group's share of the layer's total gain.

    `keys`/`colors`/`hatch` are parallel arrays over the model's feature columns,
    amplitude features first then phase features (matching build_features()'s
    column order and plot_feature_importance_pie()'s convention); split evenly in
    half here (first half = amplitude, second half = phase).
    """
    n_layers = len(models)
    n_features = len(keys)
    half = n_features // 2
    x = np.arange(n_layers)

    fig, ax = plt.subplots(figsize=(2 * n_layers, 6), constrained_layout=True)

    for i, model in enumerate(models):
        booster = model.get_booster()
        importance = booster.get_score(importance_type="gain")
        gains = np.array([importance.get(f"f{j}", 0.0) for j in range(n_features)])
        total = gains.sum()
        gains_pct = 100 * gains / total if total > 0 else gains

        for offset, group_gains, group_colors, group_hatch in [
            (-(bar_width / 2 + gap / 2), gains_pct[:half], colors[:half], hatch[:half]),
            (bar_width / 2 + gap / 2, gains_pct[half:], colors[half:], hatch[half:]),
        ]:
            bottom = 0.0
            for g, c, h in zip(group_gains, group_colors, group_hatch):
                ax.bar(x[i] + offset, g, width=bar_width, bottom=bottom, color=c,
                       hatch=h, edgecolor="black", linewidth=0.3)
                bottom += g

    ax.set_xticks(x)
    ax.set_xticklabels([f"$\\sigma_{{{i + 1}}}$" for i in range(n_layers)], fontsize=16)
    ax.set_ylabel("Feature Importance [%]", fontsize=15)
    ax.tick_params(axis="y", labelsize=13)
    ax.spines[["top", "right"]].set_visible(False)

    patches = [mpatches.Patch(facecolor=c, hatch=h, edgecolor="black", label=k)
               for c, h, k in zip(colors, hatch, keys)]
    ax.legend(handles=patches, loc="lower center", ncol=half, fontsize=16,
              bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_density_grid(y_test_sigmas, y_pred_sigmas, per_layer_rmse, out_path, ncols=3,
                       max_points=5000, seed=0):
    """KDE of predicted vs. true conductivity per layer (mS/m, log-log axes).

    seaborn's bivariate kdeplot costs roughly O(n) per subplot (~1ms/point here) --
    at the full 80k-row test set that's over a minute per layer, ~12 minutes total.
    Since this is a visual density estimate (the RMSE annotation is computed
    separately from the full test set), it's subsampled to `max_points` per layer.
    """
    n_layers = y_test_sigmas.shape[1]
    nrows = int(np.ceil(n_layers / ncols))
    fig, ax = plt.subplots(nrows, ncols, figsize=(10 / 3 * ncols, 7 / 3 * nrows),
                            sharex=True, sharey=True, constrained_layout=True)
    axes = np.atleast_1d(ax).ravel()

    rng = np.random.default_rng(seed)
    n_rows = y_test_sigmas.shape[0]
    idx = rng.choice(n_rows, size=min(max_points, n_rows), replace=False)

    lo, hi = 1, 1000
    for i in range(n_layers):
        sns.kdeplot(x=y_test_sigmas[idx, i] * 1000, y=y_pred_sigmas[idx, i] * 1000,
                    fill=True, cmap="viridis", thresh=0, levels=np.linspace(0, 1, 10), ax=axes[i])
        axes[i].plot([lo, hi * 1.5], [lo, hi * 1.5], ":r")
        axes[i].set_xlim([lo, hi])
        axes[i].set_ylim([lo, hi])
        axes[i].set_xscale("log")
        axes[i].set_yscale("log")
        axes[i].set_title(f"$\\sigma_{{{i + 1}}}$ - RMSE: {per_layer_rmse[i]:.4f}", fontsize=9)
    for j in range(n_layers, len(axes)):
        axes[j].axis("off")

    for i in range(0, n_layers, ncols):
        axes[i].set_ylabel("Predicted $\\sigma$ [mS/m]", fontsize=9)
    for i in range(max(0, n_layers - ncols), n_layers):
        axes[i].set_xlabel("True $\\sigma$ [mS/m]", fontsize=9)

    from matplotlib.cm import ScalarMappable
    sm = ScalarMappable(cmap="viridis")
    fig.colorbar(sm, orientation="horizontal", ax=ax, pad=0.02, location="bottom",
                 shrink=0.5, label="Density of samples")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_scatter_nrmse(y_test_sigmas, y_pred_sigmas, out_path, n_samples=1000, seed=0, ncols=3):
    """Per-sample relative error ("NRMSE", %) vs. true conductivity, one subplot per
    layer, for a random subset of `n_samples` test points. Points are colored by the
    local density of true conductivity values (in linear mS/m space) to show how much
    more densely the low end of the (log-uniformly sampled) conductivity range is
    populated than the high end.
    """
    from scipy.stats import gaussian_kde

    n_layers = y_test_sigmas.shape[1]
    nrows = int(np.ceil(n_layers / ncols))
    fig, ax = plt.subplots(nrows, ncols, figsize=(10 / 3 * ncols, 7 / 3 * nrows),
                            sharex=True, constrained_layout=True)
    axes = np.atleast_1d(ax).ravel()

    rng = np.random.default_rng(seed)
    n_rows = y_test_sigmas.shape[0]
    idx = rng.choice(n_rows, size=min(n_samples, n_rows), replace=False)

    sm = None
    for i in range(n_layers):
        true_mSm = y_test_sigmas[idx, i] * 1000
        pred_mSm = y_pred_sigmas[idx, i] * 1000
        nrmse_pct = np.abs(true_mSm - pred_mSm) / true_mSm * 100

        density = gaussian_kde(true_mSm)(true_mSm)
        order = np.argsort(density)
        sm = axes[i].scatter(true_mSm[order], nrmse_pct[order], c=density[order],
                              cmap="viridis", s=10)
        axes[i].set_xscale("log")
        axes[i].set_yscale("log")
        axes[i].set_title(f"$\\sigma_{{{i + 1}}}$", fontsize=10)
    for j in range(n_layers, len(axes)):
        axes[j].axis("off")

    for i in range(0, n_layers, ncols):
        axes[i].set_ylabel("NRMSE [%]", fontsize=9)
    for i in range(max(0, n_layers - ncols), n_layers):
        axes[i].set_xlabel("True $\\sigma$ [mS/m]", fontsize=9)

    fig.colorbar(sm, ax=ax, location="right", shrink=0.6, label="Density of samples")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_error_histogram(y_test_sigmas, y_pred_sigmas, out_path, threshold_mSm=10, ncols=3):
    """Histogram of prediction error (mS/m), split by whether the true value is
    above or below `threshold_mSm`, since XGBoost error scales with signal magnitude."""
    n_layers = y_test_sigmas.shape[1]
    nrows = int(np.ceil(n_layers / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(10 / 3 * ncols, 6 / 3 * nrows), sharey=True)
    ax = np.atleast_1d(axs).ravel()

    for s in range(n_layers):
        true_mSm = y_test_sigmas[:, s] * 1000
        error_mSm = (y_test_sigmas[:, s] - y_pred_sigmas[:, s]) * 1000
        above = error_mSm[true_mSm > threshold_mSm]
        below = error_mSm[true_mSm <= threshold_mSm]
        above = above[above > -1000]
        below = below[below > -1000]
        ax[s].hist(above, label=f"> {threshold_mSm} mS/m")
        ax[s].hist(below, label=f"$\\leq$ {threshold_mSm} mS/m")
        ax[s].set_title(f"$\\sigma_{{{s + 1}}}$", fontsize=12)
        ax[s].set_ylabel("Frequency", fontsize=10)
        ax[s].set_xlabel("Error [mS/m]", fontsize=10)
    for j in range(n_layers, len(ax)):
        ax[j].axis("off")

    fig.legend([f"> {threshold_mSm} mS/m", f"$\\leq$ {threshold_mSm} mS/m"], ncol=2,
               bbox_to_anchor=(0.68, 0.06))
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_profile_fit_grid(models_true_sigma, models_pred_sigma, models_true_norm, models_pred_norm,
                           data_true_norm, data_pred_norm, thk, title, out_path, ncols=5):
    """Step-profile fit for a handful of examples: true vs. predicted conductivity-
    with-depth, with each panel's title reporting % RMSE in normalized model space
    and in normalized feature ("data") space -- same layout as the notebooks'
    "Train Models"/"Test Models" figures.
    """
    n = models_true_sigma.shape[0]
    nrows = int(np.ceil(n / ncols))
    fig, ax = plt.subplots(nrows, ncols, figsize=(2 * ncols, 3 * nrows), sharey=True, sharex=True)
    axes = ax.ravel()
    depth_edges = np.hstack(([0], np.cumsum(thk), [10]))

    for i in range(n):
        axes[i].step(np.hstack((1000 * models_true_sigma[i], [1000 * models_true_sigma[i, -1]])),
                     depth_edges, "r")
        axes[i].step(np.hstack((1000 * models_pred_sigma[i], [1000 * models_pred_sigma[i, -1]])),
                     depth_edges, ":b")
        model_rmse = 100 * np.sqrt(np.mean((models_pred_norm[i] - models_true_norm[i]) ** 2))
        data_rmse = 100 * np.sqrt(np.mean((data_pred_norm[i] - data_true_norm[i]) ** 2))
        axes[i].set_title(f"Model {i + 1} Error: {model_rmse:.2f} %\nData Error: {data_rmse:.2f} %",
                           fontsize=8)
        if i >= n - ncols:
            axes[i].set_xlabel("$\\sigma$ [mS/m]")
    for j in range(n, len(axes)):
        axes[j].axis("off")

    axes[0].legend(["True", "Pred"])
    axes[-1].invert_yaxis()
    axes[-1].set_xscale("log")
    for i in range(0, n, ncols):
        axes[i].set_ylabel("Depth [m]")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def per_layer_rmse_normalized(y_test_norm, y_pred_norm):
    return [float(root_mean_squared_error(y_test_norm[:, i], y_pred_norm[:, i]))
            for i in range(y_test_norm.shape[1])]
