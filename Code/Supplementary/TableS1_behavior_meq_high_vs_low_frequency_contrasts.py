# ============================================================
# TableS1_behavior_meq_high_vs_low_frequency_contrasts.py
#
# Results 4.1 (second paragraph): direct comparisons between high-frequency
# PBM (Cond3 = 40 Hz, Cond4 = 120 Hz) and low-frequency PBM
# (Cond1 = 4 Hz, Cond2 = 10 Hz) on the MEQ.
#
# Manuscript values from this script:
#   - Friedman test restricted to the four active conditions
#     (chi2(3), p, Kendall's W).
# The direct contrasts and the pooled contrast are also computed here; the
# values printed in Table S1C and Figure S1 come from
#   FigS1_TableS1_behavior_meq_crossover_models.py
# which reproduces the same contrasts within the crossover model.
#
# This script does not refit or re-correct the condition-vs-sham contrasts
# reported by
#   Code/Main_figures/Fig02A-C_behavior_meq_frequency_effects.py
# It reuses the same score table, the same excluded subject, the same
# within-subject pairing, the same 10000-iteration sign-flip permutation,
# and the same Benjamini-Hochberg helper.
#
# Note that the direct contrast is algebraically the difference between
# two sham-referenced effects, e.g.
#     (Cond3 - Cond5) - (Cond1 - Cond5) = Cond3 - Cond1
#
# Analyses
#   1. Pooled within-subject contrast
#          mean(40 Hz, 120 Hz) - mean(4 Hz, 10 Hz)
#      A single exploratory test per score (family C in Table S1).
#   2. Direct contrasts, FDR-BH corrected within the family of four
#      high-vs-low pairs, separately within each score (Methods 3.7,
#      family ii; Table S1, family B):
#          40-4, 40-10, 120-4, 120-10
#      40-vs-120 and 4-vs-10 are also computed for context but are held
#      OUTSIDE the correction family and flagged as descriptive.
#   3. Supporting omnibus: Friedman test across the four active
#      conditions, with Kendall's W as the effect size.
#
# Outputs
#   stats/behavior_meq_highlow_pooled_contrast_no_control.csv
#   stats/behavior_meq_highlow_pairwise_contrasts_no_control.csv
#   stats/behavior_meq_active_frequency_omnibus_no_control.csv
#   stats/behavior_meq_active_condition_descriptives_no_control.csv
#   figures/behavior_sensitivity/behavior_MEQ_high_vs_low_frequency_no_control.png
# ============================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import ttest_1samp, wilcoxon, friedmanchisquare, t as t_dist

from pbm_paper_common import (
    OUT_STATS, OUT_FIGS,
    CONDITION_LABELS,
    load_score_table, sign_flip_p, cohen_d_paired, fdr_bh, sem, savefig
)


SCORE_COLUMNS = ["MEQ_Total", "Transcendence", "PositiveMood", "Ineffability", "Mystical"]
PRIMARY_SCORE = "MEQ_Total"
EXCLUDED_SUBJECTS = {"MED_PBM_001"}

ACTIVE_CONDITIONS = ["Cond1", "Cond2", "Cond3", "Cond4"]
HIGH_CONDITIONS = ["Cond3", "Cond4"]   # 40 Hz, 120 Hz
LOW_CONDITIONS = ["Cond1", "Cond2"]    # 4 Hz, 10 Hz

# The four direct high-vs-low contrasts. These form the FDR family.
PRIMARY_PAIRS = [
    ("Cond3", "Cond1"),
    ("Cond3", "Cond2"),
    ("Cond4", "Cond1"),
    ("Cond4", "Cond2"),
]

# Reported for context only, deliberately kept out of the FDR family.
CONTEXT_PAIRS = [
    ("Cond4", "Cond3"),
    ("Cond2", "Cond1"),
]

N_PERM = 10000
SEED = 42
N_BOOT = 10000
ALPHA = 0.05

CONDITION_COLORS = {
    "Cond1": "#4C78A8",  # 4Hz: blue
    "Cond2": "#72B7B2",  # 10Hz: teal
    "Cond3": "#F58518",  # 40Hz: orange
    "Cond4": "#E45756",  # 120Hz: red
}

CONDITION_DOT_COLORS = {
    "Cond1": "#2F5F8F",
    "Cond2": "#3F8F8C",
    "Cond3": "#C86400",
    "Cond4": "#B63A39",
}

OUT_BEHAV = OUT_FIGS / "behavior_sensitivity"
OUT_BEHAV.mkdir(parents=True, exist_ok=True)


# ============================================================
# SMALL HELPERS
# ============================================================

def format_p(value):
    if value is None or not np.isfinite(value):
        return "NA"
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def fmt(value, digits=2):
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def contrast_label(pos, neg):
    pos_txt = " + ".join(CONDITION_LABELS[c] for c in pos)
    neg_txt = " + ".join(CONDITION_LABELS[c] for c in neg)
    if len(pos) > 1 or len(neg) > 1:
        return f"mean({pos_txt}) - mean({neg_txt})"
    return f"{pos_txt} - {neg_txt}"


def mean_t_ci(diff, alpha=ALPHA):
    """Two-sided t-based confidence interval for the mean of paired differences."""
    diff = pd.to_numeric(pd.Series(diff), errors="coerce").dropna()
    n = len(diff)
    if n < 2:
        return np.nan, np.nan

    se = diff.std(ddof=1) / np.sqrt(n)
    if not np.isfinite(se) or se == 0:
        return diff.mean(), diff.mean()

    crit = t_dist.ppf(1 - alpha / 2, n - 1)
    return diff.mean() - crit * se, diff.mean() + crit * se


def bootstrap_mean_ci(diff, n_boot=N_BOOT, seed=SEED, alpha=ALPHA):
    """Percentile bootstrap confidence interval, matching the style of scripts 02/03."""
    diff = pd.to_numeric(pd.Series(diff), errors="coerce").dropna().to_numpy(dtype=float)
    n = len(diff)
    if n < 3:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diff[idx].mean(axis=1)

    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def wilcoxon_p(diff):
    diff = pd.to_numeric(pd.Series(diff), errors="coerce").dropna().to_numpy(dtype=float)
    if len(diff) < 3 or np.allclose(diff, 0):
        return np.nan
    try:
        _, p = wilcoxon(diff)
        return float(p)
    except Exception:
        return np.nan


def paired_contrast(wide, pos, neg, n_perm=N_PERM):
    """
    Within-subject contrast mean(pos conditions) - mean(neg conditions).

    Complete cases only: a subject must have every condition entering the
    contrast, mirroring the pairing rule used by paired_condition_vs_sham.
    """
    pos = list(pos)
    neg = list(neg)
    needed = pos + neg

    row = {
        "Contrast": contrast_label(pos, neg),
        "PosConditions": "+".join(pos),
        "NegConditions": "+".join(neg),
        "N": 0,
        "Mean_pos": np.nan,
        "Mean_neg": np.nan,
        "MeanDiff": np.nan,
        "SD_diff": np.nan,
        "SEM_diff": np.nan,
        "CI95_low": np.nan,
        "CI95_high": np.nan,
        "Boot_CI95_low": np.nan,
        "Boot_CI95_high": np.nan,
        "T": np.nan,
        "df": np.nan,
        "P_ttest": np.nan,
        "P_signflip": np.nan,
        "P_wilcoxon": np.nan,
        "Cohen_dz": np.nan,
    }

    missing = [c for c in needed if c not in wide.columns]
    if missing:
        return row

    sub = wide[needed].dropna()
    row["N"] = len(sub)
    if len(sub) < 3:
        return row

    pos_mean = sub[pos].mean(axis=1)
    neg_mean = sub[neg].mean(axis=1)
    diff = pos_mean - neg_mean

    t_stat, p_t = ttest_1samp(diff.to_numpy(dtype=float), 0.0)
    ci_low, ci_high = mean_t_ci(diff)
    boot_low, boot_high = bootstrap_mean_ci(diff)

    row.update({
        "Mean_pos": pos_mean.mean(),
        "Mean_neg": neg_mean.mean(),
        "MeanDiff": diff.mean(),
        "SD_diff": diff.std(ddof=1),
        "SEM_diff": sem(diff),
        "CI95_low": ci_low,
        "CI95_high": ci_high,
        "Boot_CI95_low": boot_low,
        "Boot_CI95_high": boot_high,
        "T": float(t_stat),
        "df": len(sub) - 1,
        "P_ttest": float(p_t),
        "P_signflip": sign_flip_p(diff, n_perm=n_perm),
        "P_wilcoxon": wilcoxon_p(diff),
        "Cohen_dz": cohen_d_paired(diff),
    })

    return row


def contrast_diffs(wide, pos, neg):
    """Per-subject contrast values, used for plotting."""
    pos = list(pos)
    neg = list(neg)
    needed = pos + neg

    if any(c not in wide.columns for c in needed):
        return pd.Series(dtype=float)

    sub = wide[needed].dropna()
    if sub.empty:
        return pd.Series(dtype=float)

    return sub[pos].mean(axis=1) - sub[neg].mean(axis=1)


# ============================================================
# LOAD SCORES
# ============================================================

all_scores = []

for score_col in SCORE_COLUMNS:
    scores = load_score_table(score_col=score_col)
    scores["ScoreName"] = score_col
    all_scores.append(scores.rename(columns={"MeditationScore": "ScoreValue"}))

scores_long = pd.concat(all_scores, ignore_index=True)
scores_long = scores_long[scores_long["Condition"].isin(ACTIVE_CONDITIONS)].copy()
scores_long = scores_long[~scores_long["Subject"].isin(EXCLUDED_SUBJECTS)].copy()

wide_by_score = {
    score_col: (
        scores_long[scores_long["ScoreName"] == score_col]
        .pivot_table(index="Subject", columns="Condition", values="ScoreValue", aggfunc="mean")
    )
    for score_col in SCORE_COLUMNS
}


# ============================================================
# DESCRIPTIVES FOR THE ACTIVE CONDITIONS
# ============================================================

descriptives = (
    scores_long
    .groupby(["ScoreName", "Condition"])
    .agg(
        N_subjects=("Subject", "nunique"),
        Mean=("ScoreValue", "mean"),
        SD=("ScoreValue", "std"),
        SEM=("ScoreValue", sem),
    )
    .reset_index()
)
descriptives["ConditionLabel"] = descriptives["Condition"].map(CONDITION_LABELS)
descriptives["ScoreName"] = pd.Categorical(descriptives["ScoreName"], SCORE_COLUMNS, ordered=True)
descriptives["Condition"] = pd.Categorical(descriptives["Condition"], ACTIVE_CONDITIONS, ordered=True)
descriptives = descriptives.sort_values(["ScoreName", "Condition"])

descriptives.to_csv(
    OUT_STATS / "behavior_meq_active_condition_descriptives_no_control.csv",
    index=False,
)


# ============================================================
# 1. POOLED CONTRAST: HIGH (40, 120 Hz) - LOW (4, 10 Hz)
# ============================================================

pooled_rows = []

for score_col in SCORE_COLUMNS:
    row = paired_contrast(wide_by_score[score_col], HIGH_CONDITIONS, LOW_CONDITIONS)
    row["ScoreName"] = score_col
    row["Family"] = "Exploratory pooled high-vs-low contrast"
    pooled_rows.append(row)

pooled = pd.DataFrame(pooled_rows)

# One test per score. MEQ_Total is reported unadjusted; the adjustment
# below covers the subscale family.
adj, reject = fdr_bh(pooled["P_signflip"].values, alpha=ALPHA)
pooled["P_FDR_across_scores"] = adj
pooled["Significant_FDR_across_scores"] = reject

pooled = pooled[[
    "ScoreName", "Family", "Contrast", "PosConditions", "NegConditions", "N",
    "Mean_pos", "Mean_neg", "MeanDiff", "SD_diff", "SEM_diff",
    "CI95_low", "CI95_high", "Boot_CI95_low", "Boot_CI95_high",
    "T", "df", "P_ttest", "P_signflip", "P_wilcoxon", "Cohen_dz",
    "P_FDR_across_scores", "Significant_FDR_across_scores",
]]

pooled.to_csv(
    OUT_STATS / "behavior_meq_highlow_pooled_contrast_no_control.csv",
    index=False,
)


# ============================================================
# 2. DIRECT HIGH-VS-LOW CONTRASTS
# ============================================================

pair_rows = []

for score_col in SCORE_COLUMNS:
    wide = wide_by_score[score_col]

    for pos, neg in PRIMARY_PAIRS:
        row = paired_contrast(wide, [pos], [neg])
        row["ScoreName"] = score_col
        row["PosCondition"] = pos
        row["NegCondition"] = neg
        row["InFDRFamily"] = True
        pair_rows.append(row)

    for pos, neg in CONTEXT_PAIRS:
        row = paired_contrast(wide, [pos], [neg])
        row["ScoreName"] = score_col
        row["PosCondition"] = pos
        row["NegCondition"] = neg
        row["InFDRFamily"] = False
        pair_rows.append(row)

pairwise = pd.DataFrame(pair_rows)
pairwise["P_FDR_within_family"] = np.nan
pairwise["Significant_FDR"] = False

# FDR-BH across the four high-vs-low pairs, separately within each score.
for score_col in SCORE_COLUMNS:
    mask = (pairwise["ScoreName"] == score_col) & (pairwise["InFDRFamily"])
    idx = pairwise.index[mask]
    if len(idx) == 0:
        continue
    adj, reject = fdr_bh(pairwise.loc[idx, "P_signflip"].values, alpha=ALPHA)
    pairwise.loc[idx, "P_FDR_within_family"] = adj
    pairwise.loc[idx, "Significant_FDR"] = reject

pairwise = pairwise[[
    "ScoreName", "Contrast", "PosCondition", "NegCondition", "InFDRFamily", "N",
    "Mean_pos", "Mean_neg", "MeanDiff", "SD_diff", "SEM_diff",
    "CI95_low", "CI95_high", "Boot_CI95_low", "Boot_CI95_high",
    "T", "df", "P_ttest", "P_signflip", "P_wilcoxon", "Cohen_dz",
    "P_FDR_within_family", "Significant_FDR",
]]

pairwise.to_csv(
    OUT_STATS / "behavior_meq_highlow_pairwise_contrasts_no_control.csv",
    index=False,
)


# ============================================================
# 3. SUPPORTING OMNIBUS ACROSS THE FOUR ACTIVE CONDITIONS
#    (Results 4.1: Friedman chi2(3), Kendall's W)
# ============================================================

omnibus_rows = []

for score_col in SCORE_COLUMNS:
    wide = wide_by_score[score_col]

    row = {
        "ScoreName": score_col,
        "Test": "Friedman across Cond1-Cond4",
        "N": 0,
        "K_conditions": len(ACTIVE_CONDITIONS),
        "Chi2": np.nan,
        "df": len(ACTIVE_CONDITIONS) - 1,
        "P": np.nan,
        "Kendalls_W": np.nan,
    }

    if all(c in wide.columns for c in ACTIVE_CONDITIONS):
        sub = wide[ACTIVE_CONDITIONS].dropna()
        row["N"] = len(sub)

        if len(sub) >= 3:
            try:
                chi2, p_val = friedmanchisquare(
                    *[sub[c].to_numpy(dtype=float) for c in ACTIVE_CONDITIONS]
                )
                row["Chi2"] = float(chi2)
                row["P"] = float(p_val)
                row["Kendalls_W"] = float(chi2) / (len(sub) * (len(ACTIVE_CONDITIONS) - 1))
            except Exception as exc:
                row["Test"] = f"Friedman failed: {exc}"

    omnibus_rows.append(row)

omnibus = pd.DataFrame(omnibus_rows)
omnibus.to_csv(
    OUT_STATS / "behavior_meq_active_frequency_omnibus_no_control.csv",
    index=False,
)


# ============================================================
# WORKING FIGURE: DIRECT HIGH-VS-LOW COMPARISONS FOR MEQ TOTAL
# (not a manuscript figure; Figure S1 is drawn by
#  FigS1_TableS1_behavior_meq_crossover_models.py)
# ============================================================

primary_wide = wide_by_score[PRIMARY_SCORE]
primary_pairs_stats = (
    pairwise[(pairwise["ScoreName"] == PRIMARY_SCORE) & (pairwise["InFDRFamily"])]
    .set_index(["PosCondition", "NegCondition"])
)
primary_pooled = pooled[pooled["ScoreName"] == PRIMARY_SCORE].iloc[0]

fig = plt.figure(figsize=(12.6, 5.4), constrained_layout=True)
gs = fig.add_gridspec(1, 2, width_ratios=[1.75, 1.0])

axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
rng = np.random.default_rng(7)


# ----------------------------
# Panel A: the four cell-wise contrasts
# ----------------------------

ax = axes[0]

pair_diffs = [contrast_diffs(primary_wide, [pos], [neg]) for pos, neg in PRIMARY_PAIRS]
pair_data = [d.to_numpy(dtype=float) for d in pair_diffs]
x = np.arange(len(PRIMARY_PAIRS))

bp = ax.boxplot(
    [d if len(d) else np.array([np.nan]) for d in pair_data],
    positions=x,
    widths=0.55,
    patch_artist=True,
    showfliers=False,
)

for patch, (pos, _neg) in zip(bp["boxes"], PRIMARY_PAIRS):
    patch.set_facecolor(CONDITION_COLORS[pos])
    patch.set_edgecolor("black")
    patch.set_alpha(0.55)

for median in bp["medians"]:
    median.set_color("black")
    median.set_linewidth(1.5)

for i, vals in enumerate(pair_data):
    if len(vals) == 0:
        continue
    jitter = rng.normal(0, 0.055, size=len(vals))
    ax.scatter(
        np.full(len(vals), x[i]) + jitter,
        vals,
        s=34,
        alpha=0.75,
        edgecolor="black",
        linewidth=0.4,
        color=CONDITION_DOT_COLORS[PRIMARY_PAIRS[i][0]],
    )

pair_means = [np.nanmean(v) if len(v) else np.nan for v in pair_data]
pair_sems = [sem(v) if len(v) else np.nan for v in pair_data]

ax.errorbar(
    x,
    pair_means,
    yerr=pair_sems,
    marker="o",
    linestyle="none",
    markersize=6,
    capsize=4,
    color="black",
    label="Mean +/- SEM",
)

all_pair_vals = np.concatenate([v for v in pair_data if len(v)]) if any(len(v) for v in pair_data) else np.array([])
if all_pair_vals.size:
    ymin = min(0, np.nanmin(all_pair_vals))
    ymax = max(0, np.nanmax(all_pair_vals))
else:
    ymin, ymax = -1.0, 1.0

y_span = max(ymax - ymin, 1.0)
label_pad = 0.08 * y_span

for i, (pos, neg) in enumerate(PRIMARY_PAIRS):
    stat_row = primary_pairs_stats.loc[(pos, neg)]
    vals = pair_data[i]
    y = (np.nanmax(vals) + label_pad) if len(vals) else label_pad

    star = "*" if bool(stat_row["Significant_FDR"]) else ""
    ax.text(
        i,
        y,
        f"p={format_p(stat_row['P_signflip'])}\n"
        f"FDR={format_p(stat_row['P_FDR_within_family'])}{star}\n"
        f"dz={fmt(stat_row['Cohen_dz'])}, n={int(stat_row['N'])}",
        ha="center",
        va="bottom",
        fontsize=8,
    )

ax.axhline(0, color="black", linestyle="--", linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels([
    f"{CONDITION_LABELS[pos]}\nvs {CONDITION_LABELS[neg]}"
    for pos, neg in PRIMARY_PAIRS
])
ax.set_ylabel(f"{PRIMARY_SCORE}: high - low frequency")
ax.set_title("A. Direct high- vs low-frequency contrasts")
ax.set_ylim(ymin - 0.12 * y_span, ymax + 0.34 * y_span)
ax.grid(True, axis="y", alpha=0.25)
ax.legend(frameon=False, fontsize=8, loc="lower right")


# ----------------------------
# Panel B: pooled planned contrast
# ----------------------------

ax = axes[1]

pooled_diff = contrast_diffs(primary_wide, HIGH_CONDITIONS, LOW_CONDITIONS)
pooled_vals = pooled_diff.to_numpy(dtype=float)

bp = ax.boxplot(
    [pooled_vals if len(pooled_vals) else np.array([np.nan])],
    positions=[0],
    widths=0.42,
    patch_artist=True,
    showfliers=False,
)

bp["boxes"][0].set_facecolor("#8C6BB1")
bp["boxes"][0].set_edgecolor("black")
bp["boxes"][0].set_alpha(0.55)
bp["medians"][0].set_color("black")
bp["medians"][0].set_linewidth(1.5)

if len(pooled_vals):
    jitter = rng.normal(0, 0.045, size=len(pooled_vals))
    ax.scatter(
        jitter,
        pooled_vals,
        s=40,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.4,
        color="#5D3A8E",
    )

    ax.errorbar(
        [0],
        [np.nanmean(pooled_vals)],
        yerr=[sem(pooled_vals)],
        marker="o",
        linestyle="none",
        markersize=7,
        capsize=5,
        color="black",
    )

    pmin = min(0, np.nanmin(pooled_vals))
    pmax = max(0, np.nanmax(pooled_vals))
else:
    pmin, pmax = -1.0, 1.0

p_span = max(pmax - pmin, 1.0)

ax.text(
    0.5,
    0.97,
    f"mean diff = {fmt(primary_pooled['MeanDiff'])}\n"
    f"95% CI [{fmt(primary_pooled['CI95_low'])}, {fmt(primary_pooled['CI95_high'])}]\n"
    f"t({int(primary_pooled['df'])}) = {fmt(primary_pooled['T'])}\n"
    f"p_perm = {format_p(primary_pooled['P_signflip'])}\n"
    f"dz = {fmt(primary_pooled['Cohen_dz'])}, n = {int(primary_pooled['N'])}",
    transform=ax.transAxes,
    ha="center",
    va="top",
    fontsize=8.5,
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.88),
)

ax.axhline(0, color="black", linestyle="--", linewidth=1)
ax.set_xticks([0])
ax.set_xticklabels(["mean(40, 120Hz)\n- mean(4, 10Hz)"])
ax.set_xlim(-0.55, 0.55)
ax.set_ylim(pmin - 0.12 * p_span, pmax + 0.55 * p_span)
ax.set_ylabel(f"{PRIMARY_SCORE}: pooled high - low")
ax.set_title("B. Planned pooled contrast")
ax.grid(True, axis="y", alpha=0.25)

fig.suptitle(
    "Direct comparison of high- (40, 120 Hz) versus low-frequency (4, 10 Hz) PBM on MEQ total",
    fontsize=13,
)

savefig(fig, OUT_BEHAV / "behavior_MEQ_high_vs_low_frequency_no_control.png")


# ============================================================
# CONSOLE SUMMARY AND DRAFT RESULTS SENTENCE
# ============================================================

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

print("\n================ POOLED HIGH-VS-LOW CONTRAST ================")
print(pooled[[
    "ScoreName", "N", "MeanDiff", "CI95_low", "CI95_high",
    "T", "P_ttest", "P_signflip", "Cohen_dz", "P_FDR_across_scores",
]].to_string(index=False))

print("\n================ CELL-WISE CONTRASTS ================")
print(pairwise[[
    "ScoreName", "Contrast", "InFDRFamily", "N", "MeanDiff",
    "CI95_low", "CI95_high", "P_signflip", "P_FDR_within_family",
    "Cohen_dz", "Significant_FDR",
]].to_string(index=False))

print("\n================ OMNIBUS ACROSS ACTIVE CONDITIONS ================")
print(omnibus.to_string(index=False))

primary_family = pairwise[
    (pairwise["ScoreName"] == PRIMARY_SCORE) & (pairwise["InFDRFamily"])
]

sig_pairs = primary_family[primary_family["Significant_FDR"]]["Contrast"].tolist()
omni = omnibus[omnibus["ScoreName"] == PRIMARY_SCORE].iloc[0]

print("\n================ DRAFT RESULTS TEXT (CHECK BEFORE USE) ================")
print(
    f"To address whether the two effective frequencies outperformed the lower "
    f"frequencies rather than merely differing from sham, we ran direct "
    f"within-subject contrasts among the four active conditions. The planned "
    f"pooled contrast, mean(40 Hz, 120 Hz) - mean(4 Hz, 10 Hz), was "
    f"{fmt(primary_pooled['MeanDiff'])} MEQ total points "
    f"(95% CI [{fmt(primary_pooled['CI95_low'])}, {fmt(primary_pooled['CI95_high'])}], "
    f"t({int(primary_pooled['df'])}) = {fmt(primary_pooled['T'])}, "
    f"sign-flip permutation p = {format_p(primary_pooled['P_signflip'])}, "
    f"Cohen's dz = {fmt(primary_pooled['Cohen_dz'])}, n = {int(primary_pooled['N'])}). "
    f"A Friedman test across the four active conditions gave "
    f"chi2({int(omni['df'])}) = {fmt(omni['Chi2'])}, p = {format_p(omni['P'])}, "
    f"Kendall's W = {fmt(omni['Kendalls_W'])}, n = {int(omni['N'])}."
)

for _, row in primary_family.iterrows():
    print(
        f"  {row['Contrast']}: mean diff = {fmt(row['MeanDiff'])} "
        f"(95% CI [{fmt(row['CI95_low'])}, {fmt(row['CI95_high'])}]), "
        f"t({int(row['df'])}) = {fmt(row['T'])}, "
        f"p = {format_p(row['P_signflip'])}, "
        f"FDR q = {format_p(row['P_FDR_within_family'])}, "
        f"dz = {fmt(row['Cohen_dz'])}, n = {int(row['N'])}"
    )

if len(sig_pairs) == 0:
    print(
        "\nNOTE: no cell-wise high-vs-low contrast survived FDR correction. "
        "Do NOT claim frequency-specific superiority over 4 Hz and 10 Hz. "
        "Report the direction and the confidence intervals, and state that "
        "the study was not powered to resolve differences between active "
        "frequencies."
    )
else:
    print(f"\nNOTE: cell-wise contrasts surviving FDR: {', '.join(sig_pairs)}")

print("\nDONE: high- vs low-frequency MEQ contrasts.")
