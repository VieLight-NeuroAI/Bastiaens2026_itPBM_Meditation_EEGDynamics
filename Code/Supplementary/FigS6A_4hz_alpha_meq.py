# ============================================================
# FigS6A_4hz_alpha_meq.py
#
# Supplementary Figure S6A (Supplementary S.6):
#   Source-alpha dynamics during 4 Hz PBM were not reliably associated
#   with MEQ total score.
#
# Panels:
#   A. 4 Hz relative alpha trajectory colored by MEQ
#   B. Descriptive median-split (high/low MEQ) trajectory
#   C. Post-Dur20 and mean(Dur10_15_20) vs MEQ
#   D. 4 Hz alpha feature-MEQ correlation summary
#
# The 4 Hz rows of Table S3 are exported by
#   Code/Main_figures/Fig07_alpha_meq_specificity_vs_sham.py
# ============================================================

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, t

from pbm_paper_common import (
    OUT_TABLES, OUT_STATS, OUT_FIGS,
    BAND_SCHEME,
    build_source_trajectory_table, load_score_table, fdr_bh, sem, savefig,
)


BAND = "Alpha"
METRIC = "relative"
NETWORK = "all_rois"
CONDITION = "Cond1"
CONDITION_LABEL = "4 Hz"
SCORE_COL = "MEQ_Total"
PRIMARY_FEATURE = "RecoveryFromD4"
SCATTER_FEATURES = ["RecoveryFromD4", "LateDuringMean_change"]
CONDITION_COLOR = "#4C78A8"
GROUP_COLORS = {"High score": "#1f77b4", "Low score": "#ff7f0e"}

KEY_FEATURES = [
    "D1_change",
    "LateDuringMean_change",
    "D4_change",
    "POST_change",
    "RecoveryFromD4",
    "ReboundIndex",
]

FEATURE_LABELS = {
    "D1_change": "Dur5",
    "D2_change": "Dur10",
    "D3_change": "Dur15",
    "D4_change": "Dur20",
    "POST_change": "Post",
    "LateDuringMean_change": "mean(Dur10_15_20)",
    "RecoveryFromD4": "Post-Dur20",
    "ReboundIndex": "Post-mean(Dur10_15_20)",
}

FEATURE_LABELS_PANEL_D = {
    **FEATURE_LABELS,
    "LateDuringMean_change": "mean\n(Dur10_15_20)",
    "ReboundIndex": "Post-mean\n(Dur10_15_20)",
}

FEATURE_NOTE = (
    "Dur5, Dur10, Dur15, Dur20, and Post are changes from PRE; "
    "mean(Dur10_15_20)=mean(Dur10,Dur15,Dur20); Post-Dur20=Post minus Dur20."
)

OUT_DIR = OUT_FIGS / "figureS6A_4hz_alpha_mechanism"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def p_text(value):
    if pd.isna(value):
        return "NA"
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def add_regression_with_ci(ax, x, y, color="black"):
    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(valid) < 3:
        return np.nan, np.nan, len(valid)

    xv = valid["x"].to_numpy(dtype=float)
    yv = valid["y"].to_numpy(dtype=float)
    r, p = pearsonr(xv, yv)

    if xv.min() == xv.max():
        return r, p, len(valid)

    coef = np.polyfit(xv, yv, deg=1)
    xx = np.linspace(xv.min(), xv.max(), 200)
    yy = coef[0] * xx + coef[1]

    n = len(xv)
    x_mean = xv.mean()
    sxx = np.sum((xv - x_mean) ** 2)
    residuals = yv - (coef[0] * xv + coef[1])
    dof = n - 2

    if dof > 0 and sxx > 0:
        mse = np.sum(residuals ** 2) / dof
        tcrit = t.ppf(0.975, dof)
        se_fit = np.sqrt(mse * (1 / n + ((xx - x_mean) ** 2 / sxx)))
        ci = tcrit * se_fit
        ax.fill_between(xx, yy - ci, yy + ci, color="gray", alpha=0.22, linewidth=0)

    ax.plot(xx, yy, color=color, linewidth=2.2)

    return r, p, len(valid)


def feature_correlations(df):
    rows = []
    for feature in KEY_FEATURES:
        sub = df[[feature, "MeditationScore"]].dropna()
        row = {
            "Feature": feature,
            "N": len(sub),
            "Pearson_r": np.nan,
            "Pearson_p": np.nan,
        }
        if len(sub) >= 3:
            row["Pearson_r"], row["Pearson_p"] = pearsonr(sub[feature], sub["MeditationScore"])
        rows.append(row)

    stats = pd.DataFrame(rows)
    adj, reject = fdr_bh(stats["Pearson_p"].values, alpha=0.05)
    stats["Pearson_p_FDR"] = adj
    stats["Significant_FDR"] = reject

    return stats


features = build_source_trajectory_table(
    band=BAND,
    metric=METRIC,
    network=NETWORK,
    band_scheme=BAND_SCHEME,
)
features = features[features["Condition"] == CONDITION].copy()
features["FeatureSource"] = "StandardSourcePipeline"

scores = load_score_table(score_col=SCORE_COL)
merged = features.merge(scores, on=["Subject", "Condition"], how="inner")

features.to_csv(OUT_TABLES / "figureS6A_4hz_alpha_features.csv", index=False)
merged.to_csv(OUT_TABLES / "figureS6A_4hz_alpha_features_with_MEQ.csv", index=False)

stats = feature_correlations(merged)
stats.to_csv(OUT_STATS / "figureS6A_4hz_alpha_MEQ_feature_correlations.csv", index=False)

print("Figure S6A 4 Hz N:", merged["Subject"].nunique())
print(stats.sort_values("Pearson_p"))


time_cols = ["PRE_change", "D1_change", "D2_change", "D3_change", "D4_change", "POST_change"]
time_labels = ["PRE", "Dur5", "Dur10", "Dur15", "Dur20", "Post"]
time_cols = [c for c in time_cols if c in merged.columns]
time_labels = [lab for lab, col in zip(time_labels, ["PRE_change", "D1_change", "D2_change", "D3_change", "D4_change", "POST_change"]) if col in time_cols]
x = np.arange(len(time_cols))

score_median = merged["MeditationScore"].median()
merged["ScoreGroup"] = np.where(merged["MeditationScore"] >= score_median, "High score", "Low score")

fig, axes = plt.subplots(
    2,
    2,
    figsize=(17.2, 9.4),
    gridspec_kw={"wspace": 0.30, "hspace": 0.44},
)


# Panel A
ax = axes[0, 0]
norm = plt.Normalize(merged["MeditationScore"].min(), merged["MeditationScore"].max())
cmap = plt.cm.viridis

for _, row in merged.iterrows():
    ax.plot(
        x,
        [row[c] for c in time_cols],
        marker="o",
        linewidth=1.35,
        alpha=0.78,
        color=cmap(norm(row["MeditationScore"])),
    )

ax.errorbar(
    x,
    [merged[c].mean() for c in time_cols],
    yerr=[sem(merged[c]) for c in time_cols],
    marker="o",
    linewidth=3,
    capsize=5,
    color="black",
    label="Group mean +/- SEM",
)
ax.axhline(0, color="black", linestyle="--", linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(time_labels)
ax.set_ylabel("Source alpha relative power change from PRE")
ax.set_title("A. 4 Hz alpha trajectory colored by MEQ", pad=8)
ax.grid(True, axis="y", alpha=0.25)
ax.legend(frameon=False)
cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.046, pad=0.03)
cbar.set_label(SCORE_COL)


# Panel B
ax = axes[0, 1]
for group, g in merged.groupby("ScoreGroup"):
    color = GROUP_COLORS.get(group, "gray")
    for _, row in g.iterrows():
        ax.plot(
            x,
            [row[c] for c in time_cols],
            color=color,
            alpha=0.13,
            linewidth=1.0,
            zorder=1,
        )
    ax.errorbar(
        x,
        [g[c].mean() for c in time_cols],
        yerr=[sem(g[c]) for c in time_cols],
        marker="o",
        linewidth=3,
        capsize=5,
        color=color,
        label=f"{group}, n={g['Subject'].nunique()}",
        zorder=3,
    )
ax.axhline(0, color="black", linestyle="--", linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(time_labels)
ax.set_ylabel("Change from PRE")
ax.set_title("B. Descriptive high/low MEQ trajectory", pad=8)
ax.grid(True, axis="y", alpha=0.25)
ax.legend(frameon=False)


# Panel C
panel_c_spec = axes[1, 0].get_subplotspec()
axes[1, 0].remove()
c_axes = panel_c_spec.subgridspec(1, 2, wspace=0.14).subplots(sharey=True)
score_values = merged["MeditationScore"].dropna().to_numpy(dtype=float)
ypad = 0.05 * (score_values.max() - score_values.min()) if score_values.size else 1.0
for idx, (ax, feature) in enumerate(zip(c_axes, SCATTER_FEATURES)):
    feature_values = merged[feature].dropna().to_numpy(dtype=float)
    xpad = 0.05 * (feature_values.max() - feature_values.min()) if feature_values.size else 0.01
    if xpad == 0:
        xpad = 0.01
    ax.scatter(
        merged[feature],
        merged["MeditationScore"],
        s=58,
        alpha=0.86,
        color=CONDITION_COLOR,
        edgecolor="white",
        linewidth=0.7,
    )
    r, p, n = add_regression_with_ci(ax, merged[feature], merged["MeditationScore"])
    feature_row = stats[stats["Feature"] == feature].iloc[0]
    ax.text(
        0.05,
        0.95,
        f"r={r:.2f}\np={p_text(p)}\nq={p_text(feature_row['Pearson_p_FDR'])}\nN={n}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.88),
    )
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(FEATURE_LABELS[feature])
    ax.set_xlim(feature_values.min() - xpad, feature_values.max() + xpad)
    ax.set_ylim(score_values.min() - ypad, score_values.max() + ypad)
    ax.grid(True, alpha=0.25)
    if idx == 0:
        ax.set_ylabel(SCORE_COL)
    else:
        ax.tick_params(labelleft=False)
c_axes[0].set_title("")
c_axes[1].set_title("")
panel_c_pos = panel_c_spec.get_position(fig)
fig.text(
    (panel_c_pos.x0 + panel_c_pos.x1) / 2,
    panel_c_pos.y1 + 0.012,
    "C. MEQ-alpha feature comparisons",
    ha="center",
    va="bottom",
    fontsize=12,
)


# Panel D
ax = axes[1, 1]
plot_stats = stats.copy()
plot_stats["Feature"] = pd.Categorical(plot_stats["Feature"], KEY_FEATURES, ordered=True)
plot_stats = plot_stats.sort_values("Feature")
bar_x = np.arange(len(plot_stats))
ax.bar(bar_x, plot_stats["Pearson_r"], color=CONDITION_COLOR, alpha=0.86)
ax.axhline(0, color="black", linestyle="--", linewidth=1)
for i, row in enumerate(plot_stats.itertuples()):
    if not np.isfinite(row.Pearson_r):
        continue
    star = "*" if bool(row.Significant_FDR) else ""
    va = "bottom" if row.Pearson_r >= 0 else "top"
    offset = 0.035 if row.Pearson_r >= 0 else -0.045
    ax.text(
        i,
        row.Pearson_r + offset,
        f"p={p_text(row.Pearson_p)}\nq={p_text(row.Pearson_p_FDR)}{star}",
        ha="center",
        va=va,
        fontsize=8,
    )
ax.set_xticks(bar_x)
ax.set_xticklabels([FEATURE_LABELS_PANEL_D[f] for f in plot_stats["Feature"]], rotation=25, ha="right")
ax.set_ylabel(f"Pearson r with {SCORE_COL}")
ax.set_ylim(-1, 1)
ax.set_title("D. 4 Hz alpha feature-MEQ correlations", pad=8)
ax.grid(True, axis="y", alpha=0.25)
ax.text(
    0.0,
    -0.45,
    FEATURE_NOTE,
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=8,
)


savefig(fig, OUT_DIR / "FigureS6A_4Hz_alpha_mechanism.png")

print("\nDONE: Figure S6A 4 Hz alpha mechanism.")
print(f"Output folder:\n{OUT_DIR}")
