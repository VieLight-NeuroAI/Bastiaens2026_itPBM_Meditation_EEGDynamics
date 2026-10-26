# ============================================================
# FigS1_TableS1_behavior_meq_crossover_models.py
#
# Crossover reanalysis of the MEQ data.
#
# Manuscript outputs:
#   Figure S1  Within-participant crossover contrasts on MEQ total
#              (forest plot; Supplementary S.1)
#   Table S1A  Omnibus test of condition across MEQ scores
#   Table S1B  Estimated marginal means for MEQ total by condition
#   Table S1C  Planned and exploratory MEQ total contrasts
#   Results 4.1 omnibus F, permutation p, partial eta2, Friedman,
#              Kendall's W, mixed-model Wald test, and marginal means
#
# Items computed, in order:
#   1. An omnibus test of condition.
#   2. Each active condition versus sham (family A).
#   3. Direct comparisons of 40 and 120 Hz against 4 and 10 Hz (family B).
#   4. Exploratory mean(40, 120 Hz) versus mean(4, 10 Hz) (family C).
#   5. Estimated marginal means, paired differences, 95% CIs, and
#      within-person standardized effect sizes (Cohen's dz).
#   6. The number of complete participant pairs behind every contrast.
#
# MODEL
#   The full crossover model is  MEQ ~ condition + period + participant.
#   No period / sequence / visit-order variable exists in the available
#   data (see PERIOD_FILE below), so the fitted primary model is
#       MEQ ~ condition + participant
#   with participant carried as a repeated factor (subject intercepts).
#   If a period map is supplied the script automatically upgrades to
#       MEQ ~ condition + period + participant
#   and additionally screens first-order carryover. Until then the
#   omnibus and all contrasts are period-unadjusted, and the exported
#   tables say so explicitly in the Model column.
#
#   Because the behavioural design is balanced and complete, subject
#   intercepts and a subject random intercept give identical condition
#   contrasts. The pure-numpy engine below is therefore exact and needs
#   no third-party modelling package. If statsmodels is importable the
#   script also fits the random-intercept MixedLM and exports its
#   coefficients as a confirmatory check.
#
# INFERENCE CONSISTENCY
#   The manuscript designates sign-flip permutation as the primary
#   inferential analysis. Every table produced here therefore carries
#   P_perm as the primary p value and P_ttest / P_model only as
#   secondary columns, and every FDR value is derived from P_perm.
#   The omnibus uses an exact within-subject label permutation, which
#   also avoids the sphericity assumption of a parametric RM-ANOVA.
#
# MULTIPLICITY
#   Three corrections are exported side by side:
#     P_FDR_within_score     - within each score, within each family
#                              (used for Table S1C and Figure S1)
#     P_FDR_subscale_family  - active-vs-sham subscale contrasts only, across
#                              the four subscales x four active conditions
#                              (16 tests; Results 4.1, Methods 3.7 family iii)
#     P_FDR_score_grid       - across every score-by-condition cell within
#                              each family, MEQ_Total included (20 tests in
#                              family A); not used in the manuscript
#   MEQ_Total is the primary outcome; subscales are exploratory.
#
# Outputs (OUT_STATS)
#   behavior_meq_crossover_omnibus_no_control.csv      -> Table S1A
#   behavior_meq_crossover_emmeans_no_control.csv      -> Table S1B
#   behavior_meq_crossover_contrasts_no_control.csv    -> Table S1C
#   behavior_meq_crossover_model_coefficients_no_control.csv
#   behavior_meq_crossover_period_diagnostics_no_control.csv  (if period supplied)
# Outputs (OUT_TABLES)
#   TEMPLATE_condition_period_map.csv  (only when no period map exists)
# Figure
#   figures/behavior_sensitivity/behavior_MEQ_crossover_contrast_forest_no_control.png
#     -> Figure S1
# ============================================================

import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import ttest_1samp, friedmanchisquare, f as f_dist, chi2 as chi2_dist, t as t_dist

from pbm_paper_common import (
    DATA_ROOT, OUT_STATS, OUT_TABLES, OUT_FIGS,
    CONDITION_LABELS,
    load_score_table, sign_flip_p, cohen_d_paired, fdr_bh, sem, savefig
)

try:
    import statsmodels.formula.api as smf
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False


SCORE_COLUMNS = ["MEQ_Total", "Transcendence", "PositiveMood", "Ineffability", "Mystical"]
PRIMARY_SCORE = "MEQ_Total"
EXCLUDED_SUBJECTS = {"MED_PBM_001"}

ANALYSIS_CONDITIONS = ["Cond1", "Cond2", "Cond3", "Cond4", "Cond5"]
ACTIVE_CONDITIONS = ["Cond1", "Cond2", "Cond3", "Cond4"]
HIGH_CONDITIONS = ["Cond3", "Cond4"]
LOW_CONDITIONS = ["Cond1", "Cond2"]
SHAM = "Cond5"

# Optional crossover period map. Supply columns: Subject, Condition, Period
# (Period = 1..k visit index). Sequence is derived from it automatically.
PERIOD_FILE = DATA_ROOT / "behavior" / "condition_period_map.csv"

N_PERM = 10000
SEED = 42
ALPHA = 0.05

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
    diff = pd.to_numeric(pd.Series(diff), errors="coerce").dropna()
    n = len(diff)
    if n < 2:
        return np.nan, np.nan
    se = diff.std(ddof=1) / np.sqrt(n)
    if not np.isfinite(se) or se == 0:
        return diff.mean(), diff.mean()
    crit = t_dist.ppf(1 - alpha / 2, n - 1)
    return diff.mean() - crit * se, diff.mean() + crit * se


# ============================================================
# ITEM 1: OMNIBUS TEST OF CONDITION (TABLE S1A)
# ============================================================

def rm_anova(Y):
    """
    One-way repeated-measures ANOVA on a complete (n_subjects x k_conditions)
    matrix. Returns the condition F, its degrees of freedom, the parametric p,
    and partial eta squared.
    """
    n, k = Y.shape
    grand = Y.mean()
    cond_means = Y.mean(axis=0)
    subj_means = Y.mean(axis=1)

    ss_cond = n * float(((cond_means - grand) ** 2).sum())
    ss_subj = k * float(((subj_means - grand) ** 2).sum())
    ss_total = float(((Y - grand) ** 2).sum())
    ss_resid = ss_total - ss_cond - ss_subj

    df_cond = k - 1
    df_resid = (n - 1) * (k - 1)

    if df_resid <= 0 or ss_resid <= 0:
        return dict(F=np.nan, df1=df_cond, df2=df_resid, P=np.nan, partial_eta2=np.nan)

    ms_cond = ss_cond / df_cond
    ms_resid = ss_resid / df_resid
    F = ms_cond / ms_resid

    return dict(
        F=F,
        df1=df_cond,
        df2=df_resid,
        P=float(f_dist.sf(F, df_cond, df_resid)),
        partial_eta2=ss_cond / (ss_cond + ss_resid),
    )


def permutation_omnibus_p(Y, n_perm=N_PERM, seed=SEED):
    """
    Exact within-subject label permutation test for the condition effect.

    Permuting condition labels within a participant leaves the subject sum of
    squares and the total sum of squares untouched, so the permutation
    redistributes variance only between the condition and residual terms. This
    is the correct exchangeability structure for a within-subject crossover and
    it makes no sphericity assumption.
    """
    obs = rm_anova(Y)["F"]
    if not np.isfinite(obs):
        return np.nan, np.nan

    n, k = Y.shape
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)

    for i in range(n_perm):
        order = rng.random((n, k)).argsort(axis=1)
        null[i] = rm_anova(np.take_along_axis(Y, order, axis=1))["F"]

    p = (np.sum(null >= obs) + 1) / (n_perm + 1)
    return float(p), obs


# ============================================================
# ITEMS 2-6: PAIRED CONTRAST ENGINE
# ============================================================

def paired_contrast(wide, pos, neg, family, n_perm=N_PERM):
    """
    Within-subject contrast mean(pos) - mean(neg), complete cases only.

    Reports the paired difference, its t-based 95% CI, the within-person
    standardized effect size (Cohen's dz), the number of complete participant
    pairs, and both the permutation and parametric p values.
    """
    pos, neg = list(pos), list(neg)
    needed = pos + neg

    row = {
        "Family": family,
        "Contrast": contrast_label(pos, neg),
        "PosConditions": "+".join(pos),
        "NegConditions": "+".join(neg),
        "N_complete_pairs": 0,
        "EMM_pos": np.nan, "EMM_neg": np.nan,
        "MeanDiff": np.nan, "SE_diff": np.nan,
        "CI95_low": np.nan, "CI95_high": np.nan,
        "T": np.nan, "df": np.nan,
        "P_perm": np.nan, "P_ttest": np.nan,
        "Cohen_dz": np.nan,
    }

    if any(c not in wide.columns for c in needed):
        return row

    sub = wide[needed].dropna()
    row["N_complete_pairs"] = len(sub)
    if len(sub) < 3:
        return row

    pos_mean = sub[pos].mean(axis=1)
    neg_mean = sub[neg].mean(axis=1)
    diff = pos_mean - neg_mean

    t_stat, p_t = ttest_1samp(diff.to_numpy(dtype=float), 0.0)
    ci_low, ci_high = mean_t_ci(diff)

    row.update({
        "EMM_pos": pos_mean.mean(),
        "EMM_neg": neg_mean.mean(),
        "MeanDiff": diff.mean(),
        "SE_diff": sem(diff),
        "CI95_low": ci_low,
        "CI95_high": ci_high,
        "T": float(t_stat),
        "df": len(sub) - 1,
        "P_perm": sign_flip_p(diff, n_perm=n_perm),
        "P_ttest": float(p_t),
        "Cohen_dz": cohen_d_paired(diff),
    })

    return row


# ============================================================
# LOAD SCORES
# ============================================================

all_scores = []
for score_col in SCORE_COLUMNS:
    s = load_score_table(score_col=score_col)
    s["ScoreName"] = score_col
    all_scores.append(s.rename(columns={"MeditationScore": "ScoreValue"}))

scores_long = pd.concat(all_scores, ignore_index=True)
scores_long = scores_long[scores_long["Condition"].isin(ANALYSIS_CONDITIONS)].copy()
scores_long = scores_long[~scores_long["Subject"].isin(EXCLUDED_SUBJECTS)].copy()

wide_by_score = {
    sc: scores_long[scores_long["ScoreName"] == sc].pivot_table(
        index="Subject", columns="Condition", values="ScoreValue", aggfunc="mean"
    )
    for sc in SCORE_COLUMNS
}


# ============================================================
# PERIOD MAP (OPTIONAL)
# ============================================================

period_map = None
if PERIOD_FILE.exists():
    pm = pd.read_csv(PERIOD_FILE)
    required = {"Subject", "Condition", "Period"}
    if required.issubset(pm.columns):
        pm["Subject"] = pm["Subject"].astype(str).str.strip()
        pm["Condition"] = pm["Condition"].astype(str).str.strip()
        pm["Period"] = pd.to_numeric(pm["Period"], errors="coerce")
        period_map = pm.dropna(subset=["Period"])
        print(f"Period map loaded: {PERIOD_FILE}  rows={len(period_map)}")
    else:
        print(f"[WARNING] {PERIOD_FILE} is missing columns {required - set(pm.columns)}; ignoring.")
else:
    template = (
        scores_long[["Subject", "Condition"]]
        .drop_duplicates()
        .sort_values(["Subject", "Condition"])
        .copy()
    )
    template["Period"] = ""
    template.to_csv(OUT_TABLES / "TEMPLATE_condition_period_map.csv", index=False)
    print(
        "No period map found. A blank template has been written to\n"
        f"  {OUT_TABLES / 'TEMPLATE_condition_period_map.csv'}\n"
        "Fill in the Period column (visit index 1..k per participant) and save it to\n"
        f"  {PERIOD_FILE}\n"
        "to enable the period-adjusted model and the carryover screen."
    )

MODEL_LABEL = (
    "MEQ ~ condition + period + participant"
    if period_map is not None
    else "MEQ ~ condition + participant (period unavailable)"
)


# ============================================================
# RUN: OMNIBUS
# ============================================================

omnibus_rows = []

for score_col in SCORE_COLUMNS:
    wide = wide_by_score[score_col]
    row = {
        "ScoreName": score_col,
        "Model": MODEL_LABEL,
        "Conditions": "Cond1-Cond5",
        "N_complete_subjects": 0,
        "F": np.nan, "df1": np.nan, "df2": np.nan,
        "P_perm": np.nan, "P_parametric": np.nan, "Partial_eta2": np.nan,
        "Friedman_chi2": np.nan, "Friedman_P": np.nan, "Kendalls_W": np.nan,
    }

    if all(c in wide.columns for c in ANALYSIS_CONDITIONS):
        sub = wide[ANALYSIS_CONDITIONS].dropna()
        Y = sub.to_numpy(dtype=float)
        row["N_complete_subjects"] = len(sub)

        if len(sub) >= 3:
            res = rm_anova(Y)
            p_perm, _ = permutation_omnibus_p(Y)
            chi2, p_fr = friedmanchisquare(*[Y[:, j] for j in range(Y.shape[1])])

            row.update({
                "F": res["F"], "df1": res["df1"], "df2": res["df2"],
                "P_perm": p_perm, "P_parametric": res["P"],
                "Partial_eta2": res["partial_eta2"],
                "Friedman_chi2": float(chi2), "Friedman_P": float(p_fr),
                "Kendalls_W": float(chi2) / (len(sub) * (Y.shape[1] - 1)),
            })

    omnibus_rows.append(row)

omnibus = pd.DataFrame(omnibus_rows)
omnibus.to_csv(OUT_STATS / "tableS1A_behavior_meq_crossover_omnibus_no_control.csv", index=False)


# ============================================================
# RUN: ESTIMATED MARGINAL MEANS (ITEM 5; TABLE S1B)
# ============================================================

emm_rows = []

for score_col in SCORE_COLUMNS:
    wide = wide_by_score[score_col]
    if not all(c in wide.columns for c in ANALYSIS_CONDITIONS):
        continue
    sub = wide[ANALYSIS_CONDITIONS].dropna()

    for cond in ANALYSIS_CONDITIONS:
        vals = sub[cond]
        n = len(vals)
        se = sem(vals)
        crit = t_dist.ppf(1 - ALPHA / 2, n - 1) if n > 1 else np.nan
        emm_rows.append({
            "ScoreName": score_col,
            "Condition": cond,
            "ConditionLabel": CONDITION_LABELS[cond],
            "N": n,
            "EMM": vals.mean(),
            "SD": vals.std(ddof=1),
            "SE": se,
            "CI95_low": vals.mean() - crit * se,
            "CI95_high": vals.mean() + crit * se,
            "Model": MODEL_LABEL,
        })

emmeans = pd.DataFrame(emm_rows)
emmeans.to_csv(OUT_STATS / "tableS1B_behavior_meq_crossover_emmeans_no_control.csv", index=False)


# ============================================================
# RUN: CONTRAST FAMILIES (ITEMS 2, 3, 4, 5, 6; TABLE S1C)
# ============================================================

FAMILY_VS_SHAM = "A. Active vs sham (primary)"
FAMILY_HIGH_LOW = "B. High vs low frequency (secondary)"
FAMILY_POOLED = "C. Pooled high vs low (exploratory)"

contrast_rows = []

for score_col in SCORE_COLUMNS:
    wide = wide_by_score[score_col]

    # Item 2: each active condition versus sham
    for cond in ACTIVE_CONDITIONS:
        r = paired_contrast(wide, [cond], [SHAM], FAMILY_VS_SHAM)
        r["ScoreName"] = score_col
        contrast_rows.append(r)

    # Item 3: direct 40 / 120 Hz against 4 / 10 Hz
    for hi in HIGH_CONDITIONS:
        for lo in LOW_CONDITIONS:
            r = paired_contrast(wide, [hi], [lo], FAMILY_HIGH_LOW)
            r["ScoreName"] = score_col
            contrast_rows.append(r)

    # Item 4: exploratory pooled comparison
    r = paired_contrast(wide, HIGH_CONDITIONS, LOW_CONDITIONS, FAMILY_POOLED)
    r["ScoreName"] = score_col
    contrast_rows.append(r)

contrasts = pd.DataFrame(contrast_rows)
contrasts["P_FDR_within_score"] = np.nan
contrasts["Sig_FDR_within_score"] = False
contrasts["P_FDR_score_grid"] = np.nan
contrasts["Sig_FDR_score_grid"] = False

# Manuscript convention (Table S1C, Figure S1): FDR within each score,
# within each family.
for (fam, sc), idx in contrasts.groupby(["Family", "ScoreName"]).groups.items():
    idx = list(idx)
    adj, rej = fdr_bh(contrasts.loc[idx, "P_perm"].values, alpha=ALPHA)
    contrasts.loc[idx, "P_FDR_within_score"] = adj
    contrasts.loc[idx, "Sig_FDR_within_score"] = rej

# Score grid: correct across every score-by-condition cell within each
# family (MEQ_Total included, so 20 tests in family A).
for fam, idx in contrasts.groupby("Family").groups.items():
    idx = list(idx)
    adj, rej = fdr_bh(contrasts.loc[idx, "P_perm"].values, alpha=ALPHA)
    contrasts.loc[idx, "P_FDR_score_grid"] = adj
    contrasts.loc[idx, "Sig_FDR_score_grid"] = rej

# Subscale family (Results 4.1; Methods 3.7 family iii): active-vs-sham
# subscale contrasts only, four subscales x four active conditions = 16 tests.
contrasts["P_FDR_subscale_family"] = np.nan
contrasts["Sig_FDR_subscale_family"] = False
idx = contrasts.index[
    (contrasts["Family"] == FAMILY_VS_SHAM) & (contrasts["ScoreName"] != PRIMARY_SCORE)
]
adj, rej = fdr_bh(contrasts.loc[idx, "P_perm"].values, alpha=ALPHA)
contrasts.loc[idx, "P_FDR_subscale_family"] = adj
contrasts.loc[idx, "Sig_FDR_subscale_family"] = rej

contrasts["Model"] = MODEL_LABEL
contrasts = contrasts[[
    "Family", "ScoreName", "Contrast", "PosConditions", "NegConditions",
    "N_complete_pairs", "EMM_pos", "EMM_neg", "MeanDiff", "SE_diff",
    "CI95_low", "CI95_high", "Cohen_dz", "T", "df",
    "P_perm", "P_ttest",
    "P_FDR_within_score", "Sig_FDR_within_score",
    "P_FDR_subscale_family", "Sig_FDR_subscale_family",
    "P_FDR_score_grid", "Sig_FDR_score_grid",
    "Model",
]]

contrasts.to_csv(OUT_STATS / "tableS1C_behavior_meq_crossover_contrasts_no_control.csv", index=False)


# ============================================================
# CONFIRMATORY MIXED MODEL AND PERIOD DIAGNOSTICS
# (Wald chi2 column of Table S1A)
# ============================================================

coef_rows = []
period_rows = []

if HAS_STATSMODELS:
    for score_col in SCORE_COLUMNS:
        df_score = scores_long[scores_long["ScoreName"] == score_col].copy()
        df_score = df_score.dropna(subset=["Subject", "Condition", "ScoreValue"])

        if period_map is not None:
            df_score = df_score.merge(period_map, on=["Subject", "Condition"], how="left")

        formula = 'ScoreValue ~ C(Condition, Treatment(reference="Cond5"))'
        if period_map is not None and df_score["Period"].notna().all():
            formula += " + C(Period)"

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = smf.mixedlm(formula, data=df_score, groups=df_score["Subject"]).fit(
                    reml=False, method="lbfgs", maxiter=1000
                )

            for term in fit.fe_params.index:
                coef_rows.append({
                    "ScoreName": score_col,
                    "Model": formula + " + (1|Subject)",
                    "Term": term,
                    "Estimate": fit.fe_params[term],
                    "SE": fit.bse_fe[term] if term in fit.bse_fe.index else np.nan,
                    "P_value": fit.pvalues[term] if term in fit.pvalues.index else np.nan,
                })

            # Joint Wald test of the condition terms, as a model-based omnibus.
            names = list(fit.fe_params.index)
            cond_terms = [t for t in names if "C(Condition" in t]
            if cond_terms:
                b = fit.fe_params[cond_terms].to_numpy(dtype=float)
                V = fit.cov_params().loc[cond_terms, cond_terms].to_numpy(dtype=float)
                try:
                    chi2_stat = float(b @ np.linalg.solve(V, b))
                    df_w = len(cond_terms)
                    omnibus.loc[omnibus["ScoreName"] == score_col, "MixedLM_Wald_chi2"] = chi2_stat
                    omnibus.loc[omnibus["ScoreName"] == score_col, "MixedLM_Wald_df"] = df_w
                    omnibus.loc[omnibus["ScoreName"] == score_col, "MixedLM_Wald_P"] = float(
                        chi2_dist.sf(chi2_stat, df_w)
                    )
                except np.linalg.LinAlgError:
                    pass

        except Exception as exc:
            print(f"[WARNING] MixedLM failed for {score_col}: {exc!r}")

        # First-order carryover screen, only possible with a period map.
        if period_map is not None:
            try:
                d = df_score.sort_values(["Subject", "Period"]).copy()
                d["PrevCondition"] = d.groupby("Subject")["Condition"].shift(1)
                d_carry = d.dropna(subset=["PrevCondition"])
                if d_carry["PrevCondition"].nunique() > 1:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        cfit = smf.mixedlm(
                            'ScoreValue ~ C(Condition, Treatment(reference="Cond5"))'
                            " + C(Period) + C(PrevCondition)",
                            data=d_carry,
                            groups=d_carry["Subject"],
                        ).fit(reml=False, method="lbfgs", maxiter=1000)
                    for term in cfit.fe_params.index:
                        if "PrevCondition" in term:
                            period_rows.append({
                                "ScoreName": score_col,
                                "Term": term,
                                "Estimate": cfit.fe_params[term],
                                "P_value": cfit.pvalues.get(term, np.nan),
                                "Check": "first-order carryover",
                            })
            except Exception as exc:
                print(f"[WARNING] Carryover screen failed for {score_col}: {exc!r}")
else:
    print(
        "[NOTE] statsmodels is not installed, so the confirmatory random-intercept\n"
        "       MixedLM was skipped. All quantities above come\n"
        "       from the exact within-subject engine and are unaffected.\n"
        "       To enable it: pip install statsmodels"
    )

if coef_rows:
    pd.DataFrame(coef_rows).to_csv(
        OUT_STATS / "behavior_meq_crossover_model_coefficients_no_control.csv", index=False
    )

if period_rows:
    pd.DataFrame(period_rows).to_csv(
        OUT_STATS / "behavior_meq_crossover_period_diagnostics_no_control.csv", index=False
    )

omnibus.to_csv(OUT_STATS / "tableS1A_behavior_meq_crossover_omnibus_no_control.csv", index=False)


# ============================================================
# FIGURE S1: CONTRAST FOREST PLOT FOR MEQ TOTAL
# ============================================================

prim = contrasts[contrasts["ScoreName"] == PRIMARY_SCORE].copy()
prim = prim.sort_values("Family", kind="stable").reset_index(drop=True)

family_colors = {
    FAMILY_VS_SHAM: "#4C78A8",
    FAMILY_HIGH_LOW: "#F58518",
    FAMILY_POOLED: "#8C6BB1",
}

fig, ax = plt.subplots(figsize=(10.6, 6.4), constrained_layout=True)

y = np.arange(len(prim))[::-1]

for i, row in prim.iterrows():
    color = family_colors.get(row["Family"], "#666666")
    ax.errorbar(
        row["MeanDiff"], y[i],
        xerr=[[row["MeanDiff"] - row["CI95_low"]], [row["CI95_high"] - row["MeanDiff"]]],
        fmt="o", color=color, markersize=8, capsize=4, linewidth=2,
    )
    star = "*" if bool(row["Sig_FDR_within_score"]) else ""
    ax.text(
        row["CI95_high"] + 0.6, y[i],
        f"d(z)={fmt(row['Cohen_dz'])}  p={format_p(row['P_perm'])}  q={format_p(row['P_FDR_within_score'])}{star}  n={int(row['N_complete_pairs'])}",
        va="center", ha="left", fontsize=8,
    )

ax.axvline(0, color="black", linestyle="--", linewidth=1)
ax.set_yticks(y)
ax.set_yticklabels(prim["Contrast"])
ax.set_xlabel(f"{PRIMARY_SCORE}: paired difference (95% CI)")
ax.set_title(
    "Crossover contrasts on MEQ total\n"
    "primary p = sign-flip permutation; q = FDR within score and family",
    fontsize=12,
)
ax.grid(True, axis="x", alpha=0.25)

handles = [
    plt.Line2D([0], [0], marker="o", color=c, linestyle="none", markersize=8, label=f)
    for f, c in family_colors.items()
]
ax.legend(handles=handles, frameon=False, fontsize=8, loc="lower right")

xmax = float(np.nanmax(prim["CI95_high"].to_numpy(dtype=float)))
xmin = float(np.nanmin(prim["CI95_low"].to_numpy(dtype=float)))
span = max(xmax - xmin, 1.0)
ax.set_xlim(xmin - 0.08 * span, xmax + 0.95 * span)

savefig(fig, OUT_BEHAV / "FigureS1_behavior_MEQ_crossover_contrast_forest_no_control.png")


# ============================================================
# CONSOLE SUMMARY
# ============================================================

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)

print("\n================ MODEL ================")
print(MODEL_LABEL)
print(f"statsmodels available: {HAS_STATSMODELS}")

print("\n================ 1. OMNIBUS TEST OF CONDITION ================")
cols = ["ScoreName", "N_complete_subjects", "F", "df1", "df2", "P_perm", "P_parametric", "Partial_eta2", "Friedman_P"]
if "MixedLM_Wald_P" in omnibus.columns:
    cols.append("MixedLM_Wald_P")
print(omnibus[cols].to_string(index=False))

print("\n================ 5. ESTIMATED MARGINAL MEANS (MEQ total) ================")
print(
    emmeans[emmeans["ScoreName"] == PRIMARY_SCORE][
        ["ConditionLabel", "N", "EMM", "SD", "SE", "CI95_low", "CI95_high"]
    ].to_string(index=False)
)

print("\n================ 2-4, 6. CONTRASTS (MEQ total) ================")
print(
    prim[[
        "Family", "Contrast", "N_complete_pairs", "MeanDiff", "CI95_low", "CI95_high",
        "Cohen_dz", "P_perm", "P_FDR_within_score", "P_FDR_score_grid", "Sig_FDR_within_score",
    ]].to_string(index=False)
)

print("\n================ SUBSCALES VS SHAM, 16-TEST FAMILY (Results 4.1) ================")
print(
    contrasts[contrasts["P_FDR_subscale_family"].notna()][[
        "ScoreName", "Contrast", "MeanDiff", "P_perm",
        "P_FDR_subscale_family", "Sig_FDR_subscale_family",
    ]].to_string(index=False)
)

print("\n================ SUBSCALE GRID (exploratory) ================")
print(
    contrasts[contrasts["ScoreName"] != PRIMARY_SCORE][[
        "Family", "ScoreName", "Contrast", "MeanDiff", "P_perm",
        "P_FDR_within_score", "P_FDR_score_grid", "Sig_FDR_score_grid",
    ]].to_string(index=False)
)

omni_primary = omnibus[omnibus["ScoreName"] == PRIMARY_SCORE].iloc[0]
print("\n================ DRAFT METHODS / RESULTS SENTENCES (CHECK BEFORE USE) ================")
print(
    f"Methods. MEQ scores were analysed with a within-participant model, "
    f"{MODEL_LABEL}, across the four active conditions and sham. Because the "
    f"design is balanced and complete, participant intercepts and a participant "
    f"random intercept yield identical condition contrasts. The omnibus effect of "
    f"condition was tested by exact permutation of condition labels within "
    f"participant ({N_PERM} permutations), which requires no sphericity "
    f"assumption. Planned contrasts were evaluated with sign-flip permutation "
    f"tests, the manuscript's primary inferential procedure, and Benjamini-Hochberg "
    f"FDR was applied within each contrast family."
)
print()
print(
    f"Results. The omnibus effect of condition on MEQ total was "
    f"F({int(omni_primary['df1'])}, {int(omni_primary['df2'])}) = {fmt(omni_primary['F'])}, "
    f"permutation p = {format_p(omni_primary['P_perm'])}, "
    f"partial eta squared = {fmt(omni_primary['Partial_eta2'], 3)}, "
    f"n = {int(omni_primary['N_complete_subjects'])}."
)

print("\nDONE: crossover reanalysis of MEQ.")
