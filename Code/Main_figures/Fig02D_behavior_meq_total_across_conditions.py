# ============================================================
# Fig02D_behavior_meq_total_across_conditions.py
#
# Manuscript Figure 2D (Results 4.1):
#   Individual MEQ total trajectories across conditions, with the group
#   mean +/- SEM and N per condition.
#
# Also exports the per-condition MEQ total N, mean, SD and SEM reported in
# Results 4.1 (e.g. 120 Hz mean +/- SEM = 92.11 +/- 6.45).
#
# Uses Cond1-Cond5 only (Cond6 / no-device control removed) and excludes
# MED_PBM_001 (withdrew). Panels A-C of Figure 2 are drawn by
#   Code/Main_figures/Fig02A-C_behavior_meq_frequency_effects.py
# ============================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pbm_paper_common import (
    OUT_TABLES, OUT_STATS, OUT_FIGS,
    CONDITION_LABELS,
    load_score_table, sem, savefig
)


PRIMARY_SCORE = "MEQ_Total"
EXCLUDED_SUBJECTS = {"MED_PBM_001"}
FIGURE_CONDITIONS = ["Cond1", "Cond2", "Cond3", "Cond4", "Cond5"]

OUT_BEHAV = OUT_FIGS / "behavior_sensitivity"
OUT_BEHAV.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD MEQ TOTAL SCORES
# ============================================================

primary = load_score_table(score_col=PRIMARY_SCORE)
primary = primary.rename(columns={"MeditationScore": "ScoreValue"})
primary = primary[primary["Condition"].isin(FIGURE_CONDITIONS)].copy()
primary = primary[~primary["Subject"].isin(EXCLUDED_SUBJECTS)].copy()

primary.to_csv(
    OUT_TABLES / "figure2D_behavior_meq_total_scores_no_control.csv",
    index=False,
)

n_summary = (
    primary
    .groupby("Condition")
    .agg(
        N_subjects=("Subject", "nunique"),
        Mean=("ScoreValue", "mean"),
        SD=("ScoreValue", "std"),
        SEM=("ScoreValue", sem),
    )
    .reindex(FIGURE_CONDITIONS)
    .reset_index()
)
n_summary["ConditionLabel"] = n_summary["Condition"].map(CONDITION_LABELS)
n_summary.to_csv(
    OUT_STATS / "figure2D_behavior_meq_total_N_mean_sem_no_control.csv",
    index=False,
)


# ============================================================
# FIGURE 2D: MEQ TOTAL ACROSS CONDITIONS
# ============================================================

fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.8))

wide = primary.pivot_table(
    index="Subject",
    columns="Condition",
    values="ScoreValue",
    aggfunc="mean",
)

x = np.arange(len(FIGURE_CONDITIONS))

for _, row in wide.iterrows():
    y = [row.get(c, np.nan) for c in FIGURE_CONDITIONS]
    ax.plot(x, y, marker="o", linewidth=0.8, alpha=0.20)

means = [wide[c].mean() if c in wide.columns else np.nan for c in FIGURE_CONDITIONS]
sems = [sem(wide[c]) if c in wide.columns else np.nan for c in FIGURE_CONDITIONS]
ns = [wide[c].dropna().shape[0] if c in wide.columns else 0 for c in FIGURE_CONDITIONS]

ax.errorbar(
    x,
    means,
    yerr=sems,
    marker="o",
    linewidth=3,
    markersize=7,
    capsize=5,
    color="black",
    label="Mean +/- SEM",
)

for i, n in enumerate(ns):
    if np.isfinite(means[i]) and np.isfinite(sems[i]):
        ax.text(
            i,
            means[i] + sems[i] + 3,
            f"N={n}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

ax.set_xticks(x)
ax.set_xticklabels([CONDITION_LABELS[c] for c in FIGURE_CONDITIONS])
ax.set_ylabel(PRIMARY_SCORE)
ax.set_title("D. MEQ total across conditions")
ax.grid(True, axis="y", alpha=0.25)
ax.legend(frameon=False)

savefig(
    fig,
    OUT_BEHAV / "Figure2D_behavior_MEQ_total_across_conditions_no_control.png",
)

print("\nDONE: Figure 2D MEQ total across conditions.")
