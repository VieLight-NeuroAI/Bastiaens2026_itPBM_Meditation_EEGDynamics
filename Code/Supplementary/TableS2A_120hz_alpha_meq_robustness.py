# ============================================================
# TableS2A_120hz_alpha_meq_robustness.py
#
# Manuscript Table S2A (Results 4.3):
#   Robustness of the within-condition 120 Hz alpha-MEQ correlations shown
#   in Figure 4 and Table 1.
#
# Exports:
#   1. Feature-level Pearson/Spearman correlations, permutation p values
#      (10,000 permutations), bootstrap 95% CIs (10,000 resamples), and
#      leave-one-out summaries, with FDR across the six trajectory features
#      separately for each p-value family (Methods 3.7, family v).
#   2. Leave-one-out correlation estimates for each feature.
#   3. Linear-model influence diagnostics (hat values, studentized
#      residuals, Cook's distance) for all features and for the primary
#      Post-Dur20 feature.
#
# Reuses the same source-feature builder and feature definitions as
#   Code/Main_figures/Fig04_120hz_alpha_late_to_post_meq.py
#
# Internal feature names: RecoveryFromD4 = Post-Dur20,
# ReboundIndex = Post-mean(Dur10_15_20),
# LateDuringMean_change = mean(Dur10_15_20).
# Output files are prefixed "tableS2A_".
# ============================================================

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from pbm_paper_common import (
    OUT_STATS,
    OUT_TABLES,
    BAND_SCHEME,
    build_source_trajectory_table,
    load_score_table,
    fdr_bh,
)


BAND = "Alpha"
METRIC = "relative"
NETWORK = "all_rois"
CONDITION = "Cond4"
CONDITION_LABEL = "120 Hz"
SCORE_COL = "MEQ_Total"
PRIMARY_FEATURE = "RecoveryFromD4"

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

N_PERM = 10000
N_BOOT = 10000
RANDOM_SEED = 42


def finite_xy(df, feature):
    sub = df[["Subject", feature, "MeditationScore"]].dropna().copy()
    sub = sub.rename(columns={feature: "FeatureValue"})
    sub["FeatureValue"] = pd.to_numeric(sub["FeatureValue"], errors="coerce")
    sub["MeditationScore"] = pd.to_numeric(sub["MeditationScore"], errors="coerce")
    sub = sub.dropna(subset=["FeatureValue", "MeditationScore"]).copy()
    return sub


def safe_corr(x, y, method):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return np.nan, np.nan
    if np.nanstd(x, ddof=1) == 0 or np.nanstd(y, ddof=1) == 0:
        return np.nan, np.nan

    if method == "pearson":
        return pearsonr(x, y)
    if method == "spearman":
        return spearmanr(x, y)
    raise ValueError(f"Unknown method: {method}")


def permutation_corr_p(x, y, method, n_perm=N_PERM, rng=None):
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    obs, _ = safe_corr(x, y, method)
    if not np.isfinite(obs):
        return np.nan

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    null = np.empty(n_perm, dtype=float)
    valid = 0

    for i in range(n_perm):
        r, _ = safe_corr(x, rng.permutation(y), method)
        null[i] = r
        if np.isfinite(r):
            valid += 1

    null = null[np.isfinite(null)]
    if len(null) == 0:
        return np.nan

    return (np.sum(np.abs(null) >= abs(obs)) + 1) / (len(null) + 1)


def bootstrap_corr_ci(x, y, method, n_boot=N_BOOT, rng=None):
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return np.nan, np.nan, 0

    boot = []
    n = len(x)

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        r, _ = safe_corr(x[idx], y[idx], method)
        if np.isfinite(r):
            boot.append(r)

    if len(boot) == 0:
        return np.nan, np.nan, 0

    low, high = np.percentile(boot, [2.5, 97.5])
    return float(low), float(high), int(len(boot))


def leave_one_out_stats(sub, feature, method, full_r):
    method_label = {
        "pearson": "Pearson",
        "spearman": "Spearman",
    }.get(method, method)

    rows = []
    for subject in sub["Subject"].tolist():
        loo = sub[sub["Subject"] != subject]
        r, p = safe_corr(loo["FeatureValue"], loo["MeditationScore"], method)
        rows.append(
            {
                "Feature": feature,
                "FeatureLabel": FEATURE_LABELS.get(feature, feature),
                "LeftOutSubject": subject,
                "Method": method,
                "N_after_drop": int(len(loo)),
                "Correlation": r,
                "P_value": p,
            }
        )

    loo_df = pd.DataFrame(rows)
    vals = loo_df["Correlation"].dropna().to_numpy(dtype=float)

    if len(vals) == 0 or not np.isfinite(full_r):
        return loo_df, {
            f"LOO_{method_label}_min": np.nan,
            f"LOO_{method_label}_max": np.nan,
            f"LOO_{method_label}_mean": np.nan,
            f"LOO_{method_label}_sign_stability": np.nan,
        }

    if full_r == 0:
        sign_stability = float(np.mean(vals == 0))
    else:
        sign_stability = float(np.mean(np.sign(vals) == np.sign(full_r)))

    return loo_df, {
        f"LOO_{method_label}_min": float(np.min(vals)),
        f"LOO_{method_label}_max": float(np.max(vals)),
        f"LOO_{method_label}_mean": float(np.mean(vals)),
        f"LOO_{method_label}_sign_stability": sign_stability,
    }


def influence_diagnostics(sub, feature):
    x = sub["FeatureValue"].to_numpy(dtype=float)
    y = sub["MeditationScore"].to_numpy(dtype=float)
    n = len(x)

    out = sub[["Subject", "FeatureValue", "MeditationScore"]].copy()
    out["Feature"] = feature
    out["FeatureLabel"] = FEATURE_LABELS.get(feature, feature)
    out["Fitted"] = np.nan
    out["Residual"] = np.nan
    out["HatDiagonal"] = np.nan
    out["CooksDistance"] = np.nan
    out["StudentizedResidual"] = np.nan

    if n < 3 or np.nanstd(x, ddof=1) == 0:
        return out

    X = np.column_stack([np.ones(n), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    fitted = X @ beta
    resid = y - fitted

    p = X.shape[1]
    dof = n - p
    if dof <= 0:
        return out

    mse = np.sum(resid ** 2) / dof
    if mse <= 0:
        return out

    xtx_inv = np.linalg.pinv(X.T @ X)
    hat = np.sum((X @ xtx_inv) * X, axis=1)
    cooks = (resid ** 2 / (p * mse)) * (hat / ((1 - hat) ** 2))
    stud = resid / np.sqrt(mse * (1 - hat))

    out["Fitted"] = fitted
    out["Residual"] = resid
    out["HatDiagonal"] = hat
    out["CooksDistance"] = cooks
    out["StudentizedResidual"] = stud
    return out


def feature_robustness(df):
    rng_perm = np.random.default_rng(RANDOM_SEED)
    rng_boot = np.random.default_rng(RANDOM_SEED + 1)

    rows = []
    loo_rows = []
    influence_rows = []

    for feature in KEY_FEATURES:
        sub = finite_xy(df, feature)
        x = sub["FeatureValue"].to_numpy(dtype=float)
        y = sub["MeditationScore"].to_numpy(dtype=float)

        pearson_r, pearson_p = safe_corr(x, y, "pearson")
        spearman_rho, spearman_p = safe_corr(x, y, "spearman")

        pearson_perm_p = permutation_corr_p(x, y, "pearson", rng=rng_perm)
        spearman_perm_p = permutation_corr_p(x, y, "spearman", rng=rng_perm)

        pearson_boot_low, pearson_boot_high, pearson_boot_n = bootstrap_corr_ci(
            x, y, "pearson", rng=rng_boot
        )
        spearman_boot_low, spearman_boot_high, spearman_boot_n = bootstrap_corr_ci(
            x, y, "spearman", rng=rng_boot
        )

        loo_p_df, loo_p_summary = leave_one_out_stats(sub, feature, "pearson", pearson_r)
        loo_s_df, loo_s_summary = leave_one_out_stats(sub, feature, "spearman", spearman_rho)
        loo_rows.extend([loo_p_df, loo_s_df])

        influence = influence_diagnostics(sub, feature)
        influence_rows.append(influence)

        row = {
            "Condition": CONDITION,
            "ConditionLabel": CONDITION_LABEL,
            "Band": BAND,
            "Metric": METRIC,
            "Network": NETWORK,
            "Feature": feature,
            "FeatureLabel": FEATURE_LABELS.get(feature, feature),
            "N": int(len(sub)),
            "Pearson_r": pearson_r,
            "Pearson_p": pearson_p,
            "Permutation_Pearson_p": pearson_perm_p,
            "Bootstrap_Pearson_low": pearson_boot_low,
            "Bootstrap_Pearson_high": pearson_boot_high,
            "Bootstrap_Pearson_valid_N": pearson_boot_n,
            "Spearman_rho": spearman_rho,
            "Spearman_p": spearman_p,
            "Permutation_Spearman_p": spearman_perm_p,
            "Bootstrap_Spearman_low": spearman_boot_low,
            "Bootstrap_Spearman_high": spearman_boot_high,
            "Bootstrap_Spearman_valid_N": spearman_boot_n,
        }
        row.update(loo_p_summary)
        row.update(loo_s_summary)
        rows.append(row)

    stats = pd.DataFrame(rows)

    for p_col in [
        "Pearson_p",
        "Permutation_Pearson_p",
        "Spearman_p",
        "Permutation_Spearman_p",
    ]:
        q_col = p_col.replace("_p", "_q")
        sig_col = p_col.replace("_p", "_Significant_FDR")
        stats[q_col], stats[sig_col] = fdr_bh(stats[p_col].values, alpha=0.05)

    loo = pd.concat(loo_rows, ignore_index=True) if loo_rows else pd.DataFrame()
    influence = pd.concat(influence_rows, ignore_index=True) if influence_rows else pd.DataFrame()

    return stats, loo, influence


def main():
    features = build_source_trajectory_table(
        band=BAND,
        metric=METRIC,
        network=NETWORK,
        band_scheme=BAND_SCHEME,
    )
    features = features[features["Condition"] == CONDITION].copy()
    features["FeatureSource"] = "StandardCleanedInterpolatedSourcePipeline"

    scores = load_score_table(score_col=SCORE_COL)
    merged = features.merge(scores, on=["Subject", "Condition"], how="inner")

    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_STATS.mkdir(parents=True, exist_ok=True)

    merged.to_csv(
        OUT_TABLES / "tableS2A_120hz_alpha_features_with_MEQ_robustness_input.csv",
        index=False,
    )

    stats, loo, influence = feature_robustness(merged)

    stats_file = OUT_STATS / "tableS2A_120hz_alpha_MEQ_feature_robustness.csv"
    loo_file = OUT_STATS / "tableS2A_120hz_alpha_MEQ_feature_leave_one_out.csv"
    influence_file = OUT_STATS / "tableS2A_120hz_alpha_MEQ_feature_influence.csv"
    primary_influence_file = OUT_STATS / "tableS2A_120hz_alpha_PostDur20_influence.csv"

    stats.to_csv(stats_file, index=False)
    loo.to_csv(loo_file, index=False)
    influence.to_csv(influence_file, index=False)

    primary_influence = influence[influence["Feature"] == PRIMARY_FEATURE].copy()
    primary_influence.to_csv(primary_influence_file, index=False)

    print("Table S2A robustness input N:", merged["Subject"].nunique())
    print("Saved:", stats_file)
    print("Saved:", loo_file)
    print("Saved:", influence_file)
    print("Saved:", primary_influence_file)
    print()
    print(stats.sort_values("Pearson_p")[
        [
            "FeatureLabel",
            "N",
            "Pearson_r",
            "Pearson_p",
            "Pearson_q",
            "Spearman_rho",
            "Spearman_p",
            "Permutation_Spearman_p",
            "Bootstrap_Spearman_low",
            "Bootstrap_Spearman_high",
            "LOO_Spearman_min",
            "LOO_Spearman_max",
            "LOO_Spearman_sign_stability",
        ]
    ])


if __name__ == "__main__":
    main()
