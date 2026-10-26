# ============================================================
# Fig02A-C_behavior_meq_frequency_effects.py
#
# Manuscript Figure 2 (panels A-C):
#   Frequency-specific effects of pulsed PBM on meditation experience
#   (Results 4.1).
#
# Panels:
#   A. MEQ total score distributions, 4 Hz / 10 Hz / 40 Hz / 120 Hz / sham
#   B. Planned within-participant contrasts vs sham (sign-flip p, FDR q)
#   C. MEQ subscale differences vs sham (heatmap of mean differences)
#
# Panel D (MEQ total across conditions) is drawn by
#   Code/Main_figures/Fig02D_behavior_meq_total_across_conditions.py
# Panel E (personalized PBM response matrix) is drawn by
#   Fig02E_behavior_meq_personalized_response_matrix.py
#
# Statistics exported here: active-vs-sham sign-flip permutation tests
# (10,000 permutations), FDR-corrected across the four active conditions
# within each score (family i for MEQ total). The family-wide subscale
# correction reported in Results 4.1 and the omnibus / Table S1 values come
# from Code/Supplementary/FigS1_TableS1_behavior_meq_crossover_models.py.
#
# Cond6 / no-device control is excluded, as is MED_PBM_001 (withdrew).
# Output files are prefixed "figure2_" / "Figure2A-C_".
# ============================================================

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PBM_FIGURES_DIR = Path(__file__).resolve().parent

if PBM_FIGURES_DIR.exists() and str(PBM_FIGURES_DIR) not in sys.path:
    sys.path.insert(0, str(PBM_FIGURES_DIR))


from pbm_paper_common import (  # noqa: E402
    OUT_TABLES, OUT_STATS, OUT_FIGS,
    CONDITION_LABELS,
    load_score_table, paired_condition_vs_sham, fdr_bh, sem, savefig
)


SCORE_COLUMNS = ["MEQ_Total", "Transcendence", "PositiveMood", "Ineffability", "Mystical"]
PRIMARY_SCORE = "MEQ_Total"
EXCLUDED_SUBJECTS = {"MED_PBM_001"}

FIGURE_CONDITIONS = ["Cond1", "Cond2", "Cond3", "Cond4", "Cond5"]
ACTIVE_CONDITIONS = ["Cond1", "Cond2", "Cond3", "Cond4"]
SHAM = "Cond5"

FIGURE_CONDITION_LABELS = {
    cond: CONDITION_LABELS[cond]
    for cond in FIGURE_CONDITIONS
}

FIGURE_CONDITION_ORDER_LABELS = [
    f"{cond}\n{FIGURE_CONDITION_LABELS[cond]}"
    for cond in FIGURE_CONDITIONS
]

ACTIVE_CONDITION_LABELS = [
    FIGURE_CONDITION_LABELS[cond]
    for cond in ACTIVE_CONDITIONS
]

CONDITION_COLORS = {
    "Cond1": "#4C78A8",  # 4Hz: blue
    "Cond2": "#72B7B2",  # 10Hz: teal
    "Cond3": "#F58518",  # 40Hz: orange
    "Cond4": "#E45756",  # 120Hz: red
    "Cond5": "#9D9D9D",  # Sham: gray
}

CONDITION_DOT_COLORS = {
    "Cond1": "#2F5F8F",
    "Cond2": "#3F8F8C",
    "Cond3": "#C86400",
    "Cond4": "#B63A39",
    "Cond5": "#6F6F6F",
}


def format_p(value):
    if not np.isfinite(value):
        return "NA"
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def paired_difference_table(df, value_col, conditions, sham):
    wide = df.pivot_table(
        index="Subject",
        columns="Condition",
        values=value_col,
        aggfunc="mean",
    )

    rows = []
    for cond in conditions:
        if cond not in wide.columns or sham not in wide.columns:
            continue

        diffs = (wide[cond] - wide[sham]).dropna()
        for subject, diff in diffs.items():
            rows.append({
                "Subject": subject,
                "Condition": cond,
                "Reference": sham,
                "ConditionLabel": FIGURE_CONDITION_LABELS[cond],
                "Difference": diff,
            })

    return pd.DataFrame(rows)


# ============================================================
# LOAD SCORES
# ============================================================

all_scores = []

for score_col in SCORE_COLUMNS:
    scores = load_score_table(score_col=score_col)
    scores["ScoreName"] = score_col
    all_scores.append(scores.rename(columns={"MeditationScore": "ScoreValue"}))

scores_long = pd.concat(all_scores, ignore_index=True)
scores_long = scores_long[scores_long["Condition"].isin(FIGURE_CONDITIONS)].copy()
scores_long = scores_long[~scores_long["Subject"].isin(EXCLUDED_SUBJECTS)].copy()

scores_long.to_csv(
    OUT_TABLES / "figure2_behavior_scores_long_no_control.csv",
    index=False,
)

primary = scores_long[scores_long["ScoreName"] == PRIMARY_SCORE].copy()


# ============================================================
# STATS: CONDITION VS SHAM
# ============================================================

stat_rows = []

for score_col in SCORE_COLUMNS:
    sub = scores_long[scores_long["ScoreName"] == score_col].copy()
    sub = sub.rename(columns={"ScoreValue": "MeditationScore"})

    for cond in ACTIVE_CONDITIONS:
        res = paired_condition_vs_sham(
            sub.rename(columns={"MeditationScore": "Value"}),
            value_col="Value",
            condition=cond,
            sham=SHAM,
        )
        res["ScoreName"] = score_col
        stat_rows.append(res)

stats = pd.DataFrame(stat_rows)

for score_col, idx in stats.groupby("ScoreName").groups.items():
    idx = list(idx)
    adj, reject = fdr_bh(stats.loc[idx, "P_signflip"].values, alpha=0.05)
    stats.loc[idx, "P_FDR_within_score"] = adj
    stats.loc[idx, "Significant_FDR"] = reject

stats.to_csv(
    OUT_STATS / "figure2_behavior_condition_vs_sham_stats_no_control.csv",
    index=False,
)

print(stats)


# ============================================================
# FIGURE 2, PANELS A-C
# ============================================================

fig = plt.figure(
    figsize=(16.5, 5.4),
    constrained_layout=True,
)

gs = fig.add_gridspec(
    1, 4,
    width_ratios=[1.25, 1.1, 1.15, 0.04],
    wspace=0.18,
)

axes = [
    fig.add_subplot(gs[0, 0]),
    fig.add_subplot(gs[0, 1]),
    fig.add_subplot(gs[0, 2]),
]

cax = fig.add_subplot(gs[0, 3])

rng = np.random.default_rng(7)


# ----------------------------
# Figure 2A: MEQ total distributions
# ----------------------------

ax = axes[0]

data = []
positions = np.arange(len(FIGURE_CONDITIONS))

for cond in FIGURE_CONDITIONS:
    vals = primary.loc[primary["Condition"] == cond, "ScoreValue"].dropna().values
    data.append(vals)

bp = ax.boxplot(
    data,
    positions=positions,
    widths=0.55,
    patch_artist=True,
    showfliers=False,
)

for patch, cond in zip(bp["boxes"], FIGURE_CONDITIONS):
    patch.set_facecolor(CONDITION_COLORS[cond])
    patch.set_edgecolor("black")
    patch.set_alpha(0.45)

for median in bp["medians"]:
    median.set_color("black")
    median.set_linewidth(1.5)

for i, vals in enumerate(data):
    if len(vals) == 0:
        continue

    jitter = rng.normal(0, 0.055, size=len(vals))
    ax.scatter(
        np.full(len(vals), positions[i]) + jitter,
        vals,
        s=34,
        alpha=0.75,
        edgecolor="black",
        linewidth=0.4,
        color=CONDITION_DOT_COLORS[FIGURE_CONDITIONS[i]],
    )

means = [np.nanmean(vals) if len(vals) else np.nan for vals in data]
sems = [sem(vals) if len(vals) else np.nan for vals in data]

ax.errorbar(
    positions,
    means,
    yerr=sems,
    marker="o",
    linestyle="none",
    markersize=6,
    capsize=4,
    color="black",
    label="Mean +/- SEM",
)

ax.set_xticks(positions)
ax.set_xticklabels(FIGURE_CONDITION_ORDER_LABELS)
ax.set_ylabel(PRIMARY_SCORE)
ax.set_title("A. MEQ total score distributions")
ax.grid(True, axis="y", alpha=0.25)
ax.legend(frameon=False, fontsize=8)


# ----------------------------
# Figure 2B: planned contrasts vs sham
# ----------------------------

ax = axes[1]

primary_diff = paired_difference_table(
    primary,
    value_col="ScoreValue",
    conditions=ACTIVE_CONDITIONS,
    sham=SHAM,
)

diff_data = [
    primary_diff.loc[primary_diff["Condition"] == cond, "Difference"].dropna().values
    for cond in ACTIVE_CONDITIONS
]

x2 = np.arange(len(ACTIVE_CONDITIONS))

bp = ax.boxplot(
    diff_data,
    positions=x2,
    widths=0.55,
    patch_artist=True,
    showfliers=False,
)

for patch, cond in zip(bp["boxes"], ACTIVE_CONDITIONS):
    patch.set_facecolor(CONDITION_COLORS[cond])
    patch.set_edgecolor("black")
    patch.set_alpha(0.55)

for median in bp["medians"]:
    median.set_color("black")
    median.set_linewidth(1.5)

for i, vals in enumerate(diff_data):
    if len(vals) == 0:
        continue

    jitter = rng.normal(0, 0.055, size=len(vals))
    ax.scatter(
        np.full(len(vals), x2[i]) + jitter,
        vals,
        s=34,
        alpha=0.75,
        edgecolor="black",
        linewidth=0.4,
        color=CONDITION_DOT_COLORS[ACTIVE_CONDITIONS[i]],
    )

diff_means = [np.nanmean(vals) if len(vals) else np.nan for vals in diff_data]
diff_sems = [sem(vals) if len(vals) else np.nan for vals in diff_data]

ax.errorbar(
    x2,
    diff_means,
    yerr=diff_sems,
    marker="o",
    linestyle="none",
    markersize=6,
    capsize=4,
    color="black",
)

primary_stats = stats[stats["ScoreName"] == PRIMARY_SCORE].copy()
primary_stats = (
    primary_stats
    .set_index("Condition")
    .reindex(ACTIVE_CONDITIONS)
    .reset_index()
)

finite_diffs = primary_diff["Difference"].dropna().values
if len(finite_diffs):
    ymin = min(0, np.nanmin(finite_diffs))
    ymax = max(0, np.nanmax(finite_diffs))
else:
    ymin, ymax = -1, 1

y_span = max(ymax - ymin, 1)
label_pad = 0.08 * y_span

for i, row in primary_stats.iterrows():
    vals = diff_data[i] if i < len(diff_data) else np.array([])
    if len(vals):
        y = np.nanmax(vals) + label_pad
    else:
        y = row["MeanDiff"] + label_pad

    star = "*" if bool(row["Significant_FDR"]) else ""
    ax.text(
        i,
        y,
        f"p={format_p(row['P_signflip'])}\nFDR={format_p(row['P_FDR_within_score'])}{star}",
        ha="center",
        va="bottom",
        fontsize=8,
    )

ax.axhline(0, color="black", linestyle="--", linewidth=1)
ax.set_xticks(x2)
ax.set_xticklabels(ACTIVE_CONDITION_LABELS)
ax.set_ylabel(f"{PRIMARY_SCORE}: condition - sham")
ax.set_title("B. Planned contrasts vs sham")
ax.set_ylim(ymin - 0.12 * y_span, ymax + 0.28 * y_span)
ax.grid(True, axis="y", alpha=0.25)

primary_diff.to_csv(
    OUT_TABLES / "figure2B_behavior_primary_paired_differences_vs_sham_no_control.csv",
    index=False,
)


# ----------------------------
# Figure 2C: subscale contrast heatmap
# ----------------------------

ax = axes[2]

heat = stats.copy()
heat["Label"] = heat["Condition"].map(FIGURE_CONDITION_LABELS)

mat = heat.pivot_table(
    index="ScoreName",
    columns="Label",
    values="MeanDiff",
    aggfunc="mean",
).reindex(SCORE_COLUMNS)

mat = mat[ACTIVE_CONDITION_LABELS]

im = ax.imshow(mat.values, aspect="auto", cmap="coolwarm")
ax.set_xticks(np.arange(len(mat.columns)))
ax.set_xticklabels(mat.columns)
ax.set_yticks(np.arange(len(mat.index)))
ax.set_yticklabels(mat.index)
ax.set_title("C. MEQ subscale differences vs sham")

for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        val = mat.values[i, j]
        if np.isfinite(val):
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8)

cbar = fig.colorbar(im, cax=cax)
cbar.set_label("Condition - sham")

fig.suptitle(
    "Figure 2. Behavioral effects of PBM frequency on meditation scores",
    fontsize=14,
    y=1.04,
)

savefig(fig, OUT_FIGS / "Figure2A-C_behavior_MEQ_effects_no_control.png")
