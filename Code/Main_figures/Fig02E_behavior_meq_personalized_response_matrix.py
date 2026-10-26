# ============================================================
# Fig02E_behavior_meq_personalized_response_matrix.py
#
# Manuscript Figure 2E (Results 4.1):
#   Personalized PBM response matrix. Subject-level MEQ total scores across
#   4 Hz / 10 Hz / 40 Hz / 120 Hz / sham; white stars mark each
#   participant's highest-scoring condition. Descriptive only.
#
# Uses Cond1-Cond5 only (Cond6 / no-device control removed) and excludes
# MED_PBM_001 (withdrew), matching the other Figure 2 panels:
#   A-C  Fig02A-C_behavior_meq_frequency_effects.py
#   D    Code/Main_figures/Fig02D_behavior_meq_total_across_conditions.py
#
# Output files are prefixed "figure2E_" / "Figure2E_".
# ============================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pbm_paper_common import (
    OUT_TABLES, OUT_FIGS,
    CONDITION_LABELS,
    load_score_table,
    savefig,
)


SCORE_COL = "MEQ_Total"
FIGURE_CONDITIONS = ["Cond1", "Cond2", "Cond3", "Cond4", "Cond5"]
EXCLUDED_SUBJECTS = {"MED_PBM_001"}


# ============================================================
# LOAD SCORES
# ============================================================

merged = load_score_table(score_col=SCORE_COL)
merged = merged[merged["Condition"].isin(FIGURE_CONDITIONS)].copy()
merged = merged[~merged["Subject"].isin(EXCLUDED_SUBJECTS)].copy()

if merged.empty:
    raise RuntimeError("Score table is empty.")

merged["ConditionLabel"] = merged["Condition"].map(CONDITION_LABELS)

merged.to_csv(
    OUT_TABLES / "figure2E_behavior_meq_scores_for_personalized_matrix.csv",
    index=False,
)


# ============================================================
# FIGURE 2E: PERSONALIZED RESPONSE MATRIX
# ============================================================

def plot_meq_matrix():
    score_mat = merged.pivot_table(
        index="Subject",
        columns="Condition",
        values="MeditationScore",
        aggfunc="mean",
    ).reindex(columns=FIGURE_CONDITIONS).sort_index()

    fig, ax = plt.subplots(figsize=(9.5, max(6, 0.35 * len(score_mat))))

    cmap = plt.cm.viridis.copy()
    cmap.set_bad("#e6e6e6")
    im = ax.imshow(np.ma.masked_invalid(score_mat.values), aspect="auto", cmap=cmap)

    ax.set_xticks(np.arange(len(FIGURE_CONDITIONS)))
    ax.set_xticklabels([CONDITION_LABELS[c] for c in FIGURE_CONDITIONS])
    ax.set_yticks(np.arange(len(score_mat.index)))
    ax.set_yticklabels(score_mat.index, fontsize=7)

    # Mark best condition(s) per subject; ties would all be starred.
    for i, subject in enumerate(score_mat.index):
        row = score_mat.loc[subject]
        if row.notna().sum() == 0:
            continue
        max_val = row.max()
        for j, cond in enumerate(score_mat.columns):
            if pd.notna(row[cond]) and np.isclose(row[cond], max_val):
                ax.scatter(
                    j,
                    i,
                    marker="*",
                    s=90,
                    facecolor="none",
                    edgecolor="white",
                    linewidth=1.4,
                )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(SCORE_COL)

    ax.set_title(
        "E. Personalized PBM response matrix\n"
        "Cell color = MEQ score; white star = subject's best condition"
    )

    savefig(fig, OUT_FIGS / "Figure2E_behavior_MEQ_personalized_response_matrix.png")


plot_meq_matrix()

print("\nDONE: Figure 2E personalized PBM response matrix.")
print(f"Output folder:\n{OUT_FIGS}")
