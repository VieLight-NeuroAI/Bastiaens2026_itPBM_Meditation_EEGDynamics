# ============================================================
# Fig05_120hz_alpha_peak_frequency_coupling.py
#
# Manuscript Figure 5 (Results 4.3):
#   Alpha peak-frequency dynamics were coupled to 120 Hz late-to-post
#   alpha-power change.
#
# Question:
#   For 120 Hz PBM, are FOOOF alpha peak-frequency (APF / PeakCF)
#   dynamics coupled to the source relative-alpha Post-Dur20 shift?
#
# Panels:
#   A. 120 Hz APF trajectory colored by MEQ
#   B. Descriptive median-split (high/low MEQ) APF trajectory
#   C. During-stimulation APF (Dur20, mean(Dur10_15_20)) vs alpha-power
#      Post-Dur20
#   D. APF Post-Dur20 and Post-mean(Dur10_15_20) vs alpha-power Post-Dur20
#
# Pearson correlations, FDR-corrected across the four APF features shown
# in panels C and D (Methods 3.7, family vi).
#
# Output files are prefixed "figure5_" / "Figure5_".
# ============================================================

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, t

from pbm_paper_common import (
    OUT_FIGS,
    OUT_STATS,
    OUT_TABLES,
    BAND_SCHEME,
    build_source_trajectory_table,
    clean_condition,
    clean_subject_id,
    fdr_bh,
    is_schaefer_brain_roi,
    load_score_table,
    savefig,
    sem,
)


# ============================================================
# SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "Data"

FOOOF_ROOT = DATA_ROOT / "source_results" / "source_fooof_standard"
WHOLE_FOOOF_FILE = FOOOF_ROOT / "source_roi_fooof_all_subjects_conditions_long.csv"
DURING_5MIN_FOOOF_FILE = FOOOF_ROOT / "during_5min_source_roi_fooof_all_subjects_conditions_long.csv"

CONDITION = "Cond4"
CONDITION_LABEL = "120 Hz"
SCORE_COL = "MEQ_Total"

PEAK_BAND = "Alpha"
FOOOF_FEATURE = "PeakCF"
POWER_BAND = "Alpha"
POWER_METRIC = "relative"
NETWORK = "all_rois"
MAX_DURING_WINDOWS = 4

CONDITION_COLOR = "#D62728"
GROUP_COLORS = {"High score": "#1f77b4", "Low score": "#ff7f0e"}

TIME_COLS = [
    "PeakCF_PRE_change",
    "PeakCF_D1_change",
    "PeakCF_D2_change",
    "PeakCF_D3_change",
    "PeakCF_D4_change",
    "PeakCF_POST_change",
]
TIME_LABELS = ["PRE", "Dur5", "Dur10", "Dur15", "Dur20", "Post"]

SCATTER_FEATURES = [
    ("PeakCF_D4_change", "AlphaPower_RecoveryFromD4"),
    ("PeakCF_LateDuringMean_change", "AlphaPower_RecoveryFromD4"),
    ("PeakCF_RecoveryFromD4", "AlphaPower_RecoveryFromD4"),
    ("PeakCF_ReboundIndex", "AlphaPower_RecoveryFromD4"),
]

FEATURE_LABELS = {
    "PeakCF_D1_change": "APF Dur5",
    "PeakCF_D2_change": "APF Dur10",
    "PeakCF_D3_change": "APF Dur15",
    "PeakCF_D4_change": "APF Dur20",
    "PeakCF_POST_change": "APF Post",
    "PeakCF_LateDuringMean_change": "APF mean(Dur10_15_20)",
    "PeakCF_RecoveryFromD4": "APF Post-Dur20",
    "PeakCF_ReboundIndex": "APF Post-mean(Dur10_15_20)",
    "AlphaPower_RecoveryFromD4": "Alpha power\nPost-Dur20",
}

OUT_DIR = OUT_FIGS / "figure5_120hz_alpha_peak_frequency_coupling"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def p_text(value):
    if pd.isna(value):
        return "NA"
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def clean_string(x):
    return str(x).strip()


def add_regression_with_ci(ax, x, y, color="black"):
    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(valid) < 3:
        return np.nan, np.nan, len(valid)

    xv = valid["x"].to_numpy(dtype=float)
    yv = valid["y"].to_numpy(dtype=float)
    if np.nanstd(xv) == 0 or np.nanstd(yv) == 0:
        return np.nan, np.nan, len(valid)

    r, p = pearsonr(xv, yv)
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
        ax.fill_between(
            xx,
            yy - tcrit * se_fit,
            yy + tcrit * se_fit,
            color="gray",
            alpha=0.22,
            linewidth=0,
        )

    ax.plot(xx, yy, color=color, linewidth=2.2)
    return r, p, len(valid)


def load_fooof_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing FOOOF file:\n{path}")

    df = pd.read_csv(path)
    required = {"Subject", "Condition", "Phase", "ROI", "PeakBand", FOOOF_FEATURE}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")

    for col in ["Subject", "Condition", "Phase", "ROI", "PeakBand"]:
        df[col] = df[col].map(clean_string)

    df["Subject"] = df["Subject"].map(clean_subject_id)
    df["Condition"] = df["Condition"].map(clean_condition)
    df = df[df["ROI"].map(is_schaefer_brain_roi)].copy()
    df[FOOOF_FEATURE] = pd.to_numeric(df[FOOOF_FEATURE], errors="coerce")

    if "FOOOFFitSuccess" in df.columns:
        df["FOOOFFitSuccess"] = (
            df["FOOOFFitSuccess"]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes"])
        )
        df = df[df["FOOOFFitSuccess"]].copy()

    df = df[(df["PeakBand"] == PEAK_BAND) & df[FOOOF_FEATURE].notna()].copy()
    df = df.rename(columns={FOOOF_FEATURE: "Value"})
    return df


def load_fooof_whole_during():
    whole = load_fooof_csv(WHOLE_FOOOF_FILE)
    during = load_fooof_csv(DURING_5MIN_FOOOF_FILE)

    whole = whole[whole["Phase"].isin(["PRE", "POST"])].copy()
    during = during[during["Phase"] == "DURING"].copy()

    during["WindowIndex"] = pd.to_numeric(during["WindowIndex"], errors="coerce")
    during = during.dropna(subset=["WindowIndex"]).copy()
    during["WindowIndex"] = during["WindowIndex"].astype(int)
    during = during[during["WindowIndex"] < MAX_DURING_WINDOWS].copy()

    return whole, during


def build_fooof_trajectory_table(whole, during):
    whole_phase = (
        whole.groupby(["Subject", "Condition", "Phase"], as_index=False)["Value"]
        .mean()
        .rename(columns={"Value": "APF"})
    )

    whole_wide = whole_phase.pivot_table(
        index=["Subject", "Condition"],
        columns="Phase",
        values="APF",
        aggfunc="mean",
    ).reset_index()

    for phase in ["PRE", "POST"]:
        if phase not in whole_wide.columns:
            whole_wide[phase] = np.nan

    during_phase = (
        during.groupby(["Subject", "Condition", "WindowIndex"], as_index=False)["Value"]
        .mean()
        .rename(columns={"Value": "APF"})
    )

    during_wide = during_phase.pivot_table(
        index=["Subject", "Condition"],
        columns="WindowIndex",
        values="APF",
        aggfunc="mean",
    ).reset_index()

    during_wide = during_wide.rename(columns={0: "D1", 1: "D2", 2: "D3", 3: "D4"})
    out = whole_wide.merge(during_wide, on=["Subject", "Condition"], how="inner")

    out["PRE_change"] = 0.0
    for col in ["D1", "D2", "D3", "D4"]:
        if col in out.columns:
            out[f"{col}_change"] = out[col] - out["PRE"]
    out["POST_change"] = out["POST"] - out["PRE"]

    late_cols = [c for c in ["D2_change", "D3_change", "D4_change"] if c in out.columns]
    out["LateDuringMean_change"] = out[late_cols].mean(axis=1)
    out["RecoveryFromD4"] = out["POST_change"] - out["D4_change"]
    out["ReboundIndex"] = out["POST_change"] - out["LateDuringMean_change"]

    feature_cols = [
        "PRE",
        "POST",
        "PRE_change",
        "D1_change",
        "D2_change",
        "D3_change",
        "D4_change",
        "POST_change",
        "LateDuringMean_change",
        "RecoveryFromD4",
        "ReboundIndex",
    ]
    feature_cols = [c for c in feature_cols if c in out.columns]
    return out[["Subject", "Condition"] + feature_cols].copy()


def prefix_features(df, prefix):
    rename = {
        c: f"{prefix}_{c}"
        for c in df.columns
        if c not in {"Subject", "Condition"}
    }
    return df.rename(columns=rename)


def selected_scatter_stats(df):
    rows = []
    for x_col, y_col in SCATTER_FEATURES:
        sub = df[[x_col, y_col]].dropna()
        row = {
            "XFeature": x_col,
            "XLabel": FEATURE_LABELS[x_col],
            "YFeature": y_col,
            "YLabel": FEATURE_LABELS[y_col],
            "N": len(sub),
            "Pearson_r": np.nan,
            "Pearson_p": np.nan,
        }
        if len(sub) >= 3 and sub[x_col].std() > 0 and sub[y_col].std() > 0:
            row["Pearson_r"], row["Pearson_p"] = pearsonr(sub[x_col], sub[y_col])
        rows.append(row)

    stats = pd.DataFrame(rows)
    q, reject = fdr_bh(stats["Pearson_p"].values, alpha=0.05)
    stats["Pearson_q_selected_family"] = q
    stats["Significant_FDR_selected_family"] = reject
    return stats


# ============================================================
# MAIN
# ============================================================

print("Loading FOOOF Alpha PeakCF/APF tables...")
fooof_whole, fooof_during = load_fooof_whole_during()
peakcf = build_fooof_trajectory_table(fooof_whole, fooof_during)
peakcf = prefix_features(peakcf, "PeakCF")

print("Loading source relative-alpha power trajectory from pbm_paper_common paths...")
alpha_power = build_source_trajectory_table(
    band=POWER_BAND,
    metric=POWER_METRIC,
    network=NETWORK,
    band_scheme=BAND_SCHEME,
)
alpha_power = prefix_features(alpha_power, "AlphaPower")

scores = load_score_table(score_col=SCORE_COL)

merged = (
    peakcf.merge(alpha_power, on=["Subject", "Condition"], how="inner")
    .merge(scores, on=["Subject", "Condition"], how="inner")
)
merged = merged[merged["Condition"] == CONDITION].copy()
merged = merged.sort_values("Subject")

stats = selected_scatter_stats(merged)

merged.to_csv(OUT_TABLES / "figure5_120hz_alpha_peakcf_power_coupling_table.csv", index=False)
stats.to_csv(OUT_STATS / "figure5_120hz_alpha_peakcf_power_coupling_selected_stats.csv", index=False)

print(f"120 Hz APF-power coupling N: {merged['Subject'].nunique()}")
print(stats)

score_median = merged["MeditationScore"].median()
merged["ScoreGroup"] = np.where(
    merged["MeditationScore"] >= score_median,
    "High score",
    "Low score",
)

time_cols = [c for c in TIME_COLS if c in merged.columns]
time_labels = [lab for lab, col in zip(TIME_LABELS, TIME_COLS) if col in time_cols]
x = np.arange(len(time_cols))

fig, axes = plt.subplots(
    2,
    2,
    figsize=(17.2, 9.6),
    gridspec_kw={"wspace": 0.30, "hspace": 0.58},
)


# Panel A: APF trajectory colored by MEQ.
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
ax.set_ylabel("Alpha peak frequency change from PRE (Hz)")
ax.set_title("A. 120 Hz APF trajectory colored by MEQ", pad=8)
ax.grid(True, axis="y", alpha=0.25)
ax.legend(frameon=False)
cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.046, pad=0.03)
cbar.set_label(SCORE_COL)


# Panel B: APF trajectory for high/low MEQ groups.
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
ax.set_ylabel("Alpha peak frequency change from PRE (Hz)")
ax.set_title("B. Descriptive high/low MEQ APF trajectory", pad=8)
ax.grid(True, axis="y", alpha=0.25)
ax.legend(frameon=False)


def draw_scatter(ax, x_col, y_col, title, show_ylabel=True):
    sub = merged[[x_col, y_col]].dropna()
    ax.scatter(
        merged[x_col],
        merged[y_col],
        s=62,
        alpha=0.86,
        color=CONDITION_COLOR,
        edgecolor="white",
        linewidth=0.7,
    )
    r, p, n = add_regression_with_ci(ax, merged[x_col], merged[y_col])
    stat_row = stats[stats["XFeature"] == x_col].iloc[0]
    ax.text(
        0.05,
        0.95,
        f"r={r:.2f}\np={p_text(p)}\nq={p_text(stat_row['Pearson_q_selected_family'])}\nN={n}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.88),
    )

    if len(sub) > 0:
        xvals = sub[x_col].to_numpy(dtype=float)
        yvals = sub[y_col].to_numpy(dtype=float)
        xpad = 0.06 * (xvals.max() - xvals.min()) if xvals.max() != xvals.min() else 0.01
        ypad = 0.06 * (yvals.max() - yvals.min()) if yvals.max() != yvals.min() else 0.01
        ax.set_xlim(xvals.min() - xpad, xvals.max() + xpad)
        ax.set_ylim(yvals.min() - ypad, yvals.max() + ypad)

    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(f"{FEATURE_LABELS[x_col]} (Hz)")
    if show_ylabel:
        ax.set_ylabel(FEATURE_LABELS[y_col])
    else:
        ax.tick_params(labelleft=False)
    ax.set_title(title, pad=4, fontsize=9.5)
    ax.grid(True, alpha=0.25)


# Panel C: During-stimulation APF state vs alpha-power Post-Dur20.
panel_c_spec = axes[1, 0].get_subplotspec()
axes[1, 0].remove()
c_axes = panel_c_spec.subgridspec(1, 2, wspace=0.14).subplots(sharey=True)
draw_scatter(
    c_axes[0],
    "PeakCF_D4_change",
    "AlphaPower_RecoveryFromD4",
    "APF Dur20 vs alpha power Post-Dur20",
    show_ylabel=True,
)
draw_scatter(
    c_axes[1],
    "PeakCF_LateDuringMean_change",
    "AlphaPower_RecoveryFromD4",
    "Late APF mean vs alpha power Post-Dur20",
    show_ylabel=False,
)
panel_c_pos = panel_c_spec.get_position(fig)
fig.text(
    (panel_c_pos.x0 + panel_c_pos.x1) / 2,
    panel_c_pos.y1 + 0.035,
    "C. During-stimulation APF state vs alpha power Post-Dur20",
    ha="center",
    va="bottom",
    fontsize=11,
)


# Panel D: APF post-stimulation change vs alpha-power Post-Dur20.
panel_d_spec = axes[1, 1].get_subplotspec()
axes[1, 1].remove()
d_axes = panel_d_spec.subgridspec(1, 2, wspace=0.14).subplots(sharey=True)
draw_scatter(
    d_axes[0],
    "PeakCF_RecoveryFromD4",
    "AlphaPower_RecoveryFromD4",
    "APF Post-Dur20 vs alpha power Post-Dur20",
    show_ylabel=True,
)
draw_scatter(
    d_axes[1],
    "PeakCF_ReboundIndex",
    "AlphaPower_RecoveryFromD4",
    "APF Post-late mean vs alpha power Post-Dur20",
    show_ylabel=False,
)
panel_d_pos = panel_d_spec.get_position(fig)
fig.text(
    (panel_d_pos.x0 + panel_d_pos.x1) / 2,
    panel_d_pos.y1 + 0.035,
    "D. APF post-stimulation change vs alpha power Post-Dur20",
    ha="center",
    va="bottom",
    fontsize=11,
)

fig.text(
    0.5,
    0.012,
    (
        "APF = alpha peak frequency from FOOOF PeakCF. Dur5, Dur10, Dur15, Dur20, and Post are changes from PRE; "
        "mean(Dur10_15_20)=mean(Dur10,Dur15,Dur20); Post-Dur20=Post minus Dur20."
    ),
    ha="center",
    va="bottom",
    fontsize=8,
)

savefig(fig, OUT_DIR / "Figure5_120Hz_alpha_peak_frequency_coupling.png")

print("\nDONE: Figure 5 120 Hz alpha peak-frequency coupling figure.")
print(f"Output folder:\n{OUT_DIR}")
