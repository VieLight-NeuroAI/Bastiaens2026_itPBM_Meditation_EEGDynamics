# ============================================================
# pbm_sham_interaction_common.py
#
# Helpers for active-vs-sham alpha/MEQ coupling figures.
# The primary model tests whether the feature-MEQ slope differs between active
# PBM and sham while accounting for subject-level repeated measures. It uses
# statsmodels MixedLM when available and falls back to subject fixed intercepts:
#   MEQ ~ IsActive * Feature_z + (1 | Subject)
# ============================================================

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, pearsonr, t as t_dist

from pbm_paper_common import fdr_bh


FEATURE_LABELS = {
    "D1_change": "D1",
    "D2_change": "D2",
    "D3_change": "D3",
    "D4_change": "D4",
    "LateDuringMean_change": "D2-D4 mean",
    "POST_change": "Post",
    "RecoveryFromD4": "Post-D4",
    "ReboundIndex": "Post-(D2-D4)",
    "Trajectory_slope": "Slope",
}

FEATURE_DESCRIPTIONS = {
    "D1": "first during-stimulation window minus PRE",
    "D2": "second during-stimulation window minus PRE",
    "D3": "third during-stimulation window minus PRE",
    "D4": "change at final during window",
    "D2-D4 mean": "mean change across D2, D3, and D4",
    "Post": "post-stimulation change",
    "Post-D4": "post-stimulation change minus D4 change",
    "Post-(D2-D4)": "post-stimulation change minus D2-D4 mean",
    "Slope": "trajectory slope across PRE-DURING-POST",
}


def p_text(value):
    if pd.isna(value):
        return "NA"
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def zscore(series):
    vals = pd.to_numeric(series, errors="coerce")
    sd = vals.std(ddof=1)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=series.index)
    return (vals - vals.mean()) / sd


def fit_mixedlm_formula(formula, data, group_col="Subject"):
    import statsmodels.formula.api as smf

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.mixedlm(formula, data=data, groups=data[group_col])
        try:
            fit = model.fit(reml=False, method="lbfgs", maxiter=1000, disp=False)
            # L-BFGS can stop at a degenerate solution (non-finite
            # log-likelihood) while still reporting convergence; treat that
            # as a failed fit.
            if np.isfinite(fit.llf):
                return fit
        except Exception:
            pass
        return model.fit(reml=False, method="powell", maxiter=2000, disp=False)


def fixed_effect_covariance(fit):
    names = list(fit.fe_params.index)
    cov = fit.cov_params()
    return cov.loc[names, names]


def mixedlm_term_stats(fit, term):
    if term not in fit.fe_params.index:
        return np.nan, np.nan, np.nan

    est = fit.fe_params[term]
    se = fit.bse_fe[term] if hasattr(fit, "bse_fe") and term in fit.bse_fe.index else np.nan
    p = fit.pvalues[term] if term in fit.pvalues.index else np.nan

    return est, se, p


def mixedlm_linear_contrast(fit, weights):
    names = list(fit.fe_params.index)
    w = pd.Series(0.0, index=names)

    for name, value in weights.items():
        if name in w.index:
            w[name] = float(value)

    est = float(np.dot(w.to_numpy(), fit.fe_params.loc[names].to_numpy()))
    cov = fixed_effect_covariance(fit)
    se = float(np.sqrt(np.dot(w.to_numpy(), np.dot(cov.to_numpy(), w.to_numpy()))))

    if not np.isfinite(se) or se <= 0:
        return est, np.nan, np.nan

    z = est / se
    p = 2 * norm.sf(abs(z))

    return est, se, p


class OLSResult:
    def __init__(self, params, pvalues, bse, df_resid, cov_params=None):
        self.params = params
        self.pvalues = pvalues
        self.bse = bse
        self.df_resid = df_resid
        self.cov_params = cov_params


def fit_subject_fixed_ols(data, outcome, predictors, group_col="Subject"):
    cols = [outcome, group_col] + predictors
    sub = data[cols].dropna().copy()
    if len(sub) < 4:
        raise ValueError("Not enough rows for subject fixed-intercept OLS.")

    x_parts = [
        pd.Series(1.0, index=sub.index, name="Intercept"),
        sub[predictors].astype(float),
    ]

    dummies = pd.get_dummies(sub[group_col], prefix=group_col, drop_first=True, dtype=float)
    if not dummies.empty:
        x_parts.append(dummies)

    x_df = pd.concat(x_parts, axis=1)
    y = sub[outcome].to_numpy(dtype=float)
    x = x_df.to_numpy(dtype=float)

    beta, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    rank = np.linalg.matrix_rank(x)
    df_resid = len(y) - rank
    params = pd.Series(beta, index=x_df.columns, dtype=float)

    if df_resid <= 0:
        pvalues = pd.Series(np.nan, index=x_df.columns, dtype=float)
        bse = pd.Series(np.nan, index=x_df.columns, dtype=float)
        cov_df = pd.DataFrame(np.nan, index=x_df.columns, columns=x_df.columns)
        return OLSResult(params=params, pvalues=pvalues, bse=bse, df_resid=df_resid, cov_params=cov_df)

    resid = y - x @ beta
    sigma2 = float(resid.T @ resid) / df_resid
    cov = sigma2 * np.linalg.pinv(x.T @ x)
    se = np.sqrt(np.clip(np.diag(cov), 0, np.inf))

    with np.errstate(divide="ignore", invalid="ignore"):
        tvals = beta / se
    pvals = 2 * t_dist.sf(np.abs(tvals), df_resid)
    bse = pd.Series(se, index=x_df.columns, dtype=float)
    pvalues = pd.Series(pvals, index=x_df.columns, dtype=float)

    cov_df = pd.DataFrame(cov, index=x_df.columns, columns=x_df.columns)

    return OLSResult(params=params, pvalues=pvalues, bse=bse, df_resid=df_resid, cov_params=cov_df)


def ols_linear_contrast(fit, weights):
    names = list(fit.params.index)
    w = pd.Series(0.0, index=names)

    for name, value in weights.items():
        if name in w.index:
            w[name] = float(value)

    est = float(np.dot(w.to_numpy(), fit.params.loc[names].to_numpy()))

    if fit.cov_params is None:
        return est, np.nan, np.nan

    cov = fit.cov_params.loc[names, names]
    se = float(np.sqrt(np.dot(w.to_numpy(), np.dot(cov.to_numpy(), w.to_numpy()))))

    if not np.isfinite(se) or se <= 0 or fit.df_resid <= 0:
        return est, np.nan, np.nan

    t_value = est / se
    p = 2 * t_dist.sf(abs(t_value), fit.df_resid)

    return est, se, p


def active_vs_sham_interaction_stats(
    merged,
    features,
    active_condition,
    sham_condition="Cond5",
    score_col="MeditationScore",
):
    rows = []
    physiology_rows = []
    analysis = merged[merged["Condition"].isin([active_condition, sham_condition])].copy()
    analysis["IsActive"] = (analysis["Condition"] == active_condition).astype(float)

    for feature in features:
        sub = analysis[["Subject", "Condition", "IsActive", score_col, feature]].dropna().copy()
        sub = sub.rename(columns={score_col: "Score", feature: "Feature"})
        sub["Feature_z"] = zscore(sub["Feature"])
        sub = sub.dropna(subset=["Feature_z", "Score"])

        paired_subjects = (
            sub.groupby("Subject")["Condition"]
            .nunique()
            .loc[lambda s: s == 2]
            .index
        )
        paired_n = len(paired_subjects)

        row = {
            "Feature": feature,
            "N_rows": len(sub),
            "N_subjects": sub["Subject"].nunique(),
            "N_paired_subjects": paired_n,
            "Model": None,
            "Sham_slope": np.nan,
            "Active_minus_sham_slope": np.nan,
            "Active_minus_sham_slope_SE": np.nan,
            "Active_slope": np.nan,
            "Interaction_p": np.nan,
        }

        phys_row = {
            "Feature": feature,
            "N_rows": len(sub),
            "N_subjects": sub["Subject"].nunique(),
            "N_paired_subjects": paired_n,
            "Model": None,
            "Active_minus_sham_feature_z": np.nan,
            "Active_minus_sham_feature_z_SE": np.nan,
            "Active_vs_sham_feature_p": np.nan,
        }

        if len(sub) >= 8 and sub["Condition"].nunique() == 2 and sub["Subject"].nunique() >= 4:
            try:
                fit = fit_mixedlm_formula(
                    "Score ~ IsActive * Feature_z",
                    sub,
                )
                interaction_est, interaction_se, interaction_p = mixedlm_term_stats(
                    fit,
                    "IsActive:Feature_z",
                )
                row["Model"] = "MixedLM_random_subject_intercept_ML"
                row["Sham_slope"] = fit.fe_params.get("Feature_z", np.nan)
                row["Active_minus_sham_slope"] = interaction_est
                row["Active_minus_sham_slope_SE"] = interaction_se
                row["Active_slope"] = row["Sham_slope"] + row["Active_minus_sham_slope"]
                row["Interaction_p"] = interaction_p
            except Exception as exc:
                try:
                    sub["Interaction"] = sub["IsActive"] * sub["Feature_z"]
                    fit = fit_subject_fixed_ols(
                        sub,
                        outcome="Score",
                        predictors=["IsActive", "Feature_z", "Interaction"],
                    )
                    row["Model"] = f"OLS_subject_fixed_intercepts_fallback: {exc}"
                    row["Sham_slope"] = fit.params.get("Feature_z", np.nan)
                    row["Active_minus_sham_slope"] = fit.params.get("Interaction", np.nan)
                    row["Active_minus_sham_slope_SE"] = fit.bse.get("Interaction", np.nan)
                    row["Active_slope"] = row["Sham_slope"] + row["Active_minus_sham_slope"]
                    row["Interaction_p"] = fit.pvalues.get("Interaction", np.nan)
                except Exception as fallback_exc:
                    row["Model"] = f"failed: MixedLM={exc}; fallback={fallback_exc}"

            try:
                phys_fit = fit_mixedlm_formula(
                    "Feature_z ~ IsActive",
                    sub,
                )
                est, se_val, p_val = mixedlm_term_stats(phys_fit, "IsActive")
                phys_row["Model"] = "MixedLM_random_subject_intercept_ML"
                phys_row["Active_minus_sham_feature_z"] = est
                phys_row["Active_minus_sham_feature_z_SE"] = se_val
                phys_row["Active_vs_sham_feature_p"] = p_val
            except Exception as exc:
                try:
                    phys_fit = fit_subject_fixed_ols(
                        sub,
                        outcome="Feature_z",
                        predictors=["IsActive"],
                    )
                    phys_row["Model"] = f"OLS_subject_fixed_intercepts_fallback: {exc}"
                    phys_row["Active_minus_sham_feature_z"] = phys_fit.params.get("IsActive", np.nan)
                    phys_row["Active_minus_sham_feature_z_SE"] = phys_fit.bse.get("IsActive", np.nan)
                    phys_row["Active_vs_sham_feature_p"] = phys_fit.pvalues.get("IsActive", np.nan)
                except Exception as fallback_exc:
                    phys_row["Model"] = f"failed: MixedLM={exc}; fallback={fallback_exc}"

        rows.append(row)
        physiology_rows.append(phys_row)

    stats = pd.DataFrame(rows)
    adj, reject = fdr_bh(stats["Interaction_p"].values, alpha=0.05)
    stats["Interaction_p_FDR"] = adj
    stats["Interaction_sig_FDR"] = reject

    physiology = pd.DataFrame(physiology_rows)
    adj, reject = fdr_bh(physiology["Active_vs_sham_feature_p"].values, alpha=0.05)
    physiology["Active_vs_sham_feature_p_FDR"] = adj
    physiology["Active_vs_sham_feature_sig_FDR"] = reject

    return analysis, stats, physiology


def choose_primary_interaction_feature(stats, fallback_feature):
    valid = stats.dropna(subset=["Interaction_p"]).copy()
    if valid.empty:
        return fallback_feature, None

    fdr_hits = valid[valid["Interaction_sig_FDR"]].copy()
    if not fdr_hits.empty:
        row = fdr_hits.sort_values(["Interaction_p_FDR", "Interaction_p"]).iloc[0]
    else:
        row = valid.sort_values("Interaction_p").iloc[0]

    return row["Feature"], row


def add_group_regression(ax, x, y, color):
    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(valid) < 3:
        return np.nan, np.nan, len(valid)

    xv = valid["x"].to_numpy(dtype=float)
    yv = valid["y"].to_numpy(dtype=float)
    r, p = pearsonr(xv, yv)

    if np.nanmin(xv) == np.nanmax(xv):
        return r, p, len(valid)

    coef = np.polyfit(xv, yv, deg=1)
    xx = np.linspace(xv.min(), xv.max(), 200)
    yy = coef[0] * xx + coef[1]
    ax.plot(xx, yy, color=color, linewidth=2.2)

    return r, p, len(valid)


def plot_interaction_scatter(
    ax,
    analysis,
    feature,
    score_col,
    active_condition,
    sham_condition,
    active_label,
    sham_label,
    stats_row,
    title,
):
    colors = {
        sham_condition: "#6F7F8F",
        active_condition: "#D95F02",
    }
    labels = {
        sham_condition: sham_label,
        active_condition: active_label,
    }

    for cond in [sham_condition, active_condition]:
        g = analysis[analysis["Condition"] == cond].dropna(subset=[feature, score_col])
        ax.scatter(
            g[feature],
            g[score_col],
            s=64,
            alpha=0.82,
            color=colors[cond],
            edgecolor="white",
            linewidth=0.7,
            label=f"{labels[cond]}, n={g['Subject'].nunique()}",
        )
        add_group_regression(ax, g[feature], g[score_col], colors[cond])

    if stats_row is not None:
        star = "*" if bool(stats_row["Interaction_sig_FDR"]) else ""
        txt = (
            f"Condition x feature\n"
            f"p={p_text(stats_row['Interaction_p'])}, "
            f"FDR q={p_text(stats_row['Interaction_p_FDR'])}{star}\n"
            f"N paired={int(stats_row['N_paired_subjects'])}"
        )
    else:
        txt = "Condition x feature\nnot estimable"

    ax.text(
        0.05,
        0.95,
        txt,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.88),
    )

    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel(FEATURE_LABELS.get(feature, feature))
    ax.set_ylabel(score_col)
    ax.set_title(title, pad=8)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)


def plot_interaction_summary(ax, stats, features, title):
    plot_stats = stats.copy()
    plot_stats["Feature"] = pd.Categorical(plot_stats["Feature"], features, ordered=True)
    plot_stats = plot_stats.sort_values("Feature")

    x = np.arange(len(plot_stats))
    y = plot_stats["Active_minus_sham_slope"].to_numpy(dtype=float)
    colors = np.where(y >= 0, "#4C78A8", "#E45756")

    ax.bar(x, y, color=colors, alpha=0.86)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)

    for i, row in enumerate(plot_stats.itertuples()):
        if not np.isfinite(row.Active_minus_sham_slope):
            continue
        star = "*" if bool(row.Interaction_sig_FDR) else ""
        va = "bottom" if row.Active_minus_sham_slope >= 0 else "top"
        offset = 0.025 if row.Active_minus_sham_slope >= 0 else -0.035
        ax.text(
            i,
            row.Active_minus_sham_slope + offset,
            f"q={p_text(row.Interaction_p_FDR)}{star}",
            ha="center",
            va=va,
            fontsize=8,
        )

    tick_labels = [FEATURE_LABELS.get(f, f).replace("\n", " ") for f in plot_stats["Feature"]]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=42, ha="right")
    ax.set_ylabel("Active - sham slope difference\n(MEQ per 1 SD feature)")
    ax.set_title(title, pad=8)
    ax.grid(True, axis="y", alpha=0.25)
