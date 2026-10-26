# ============================================================
# Fig07_alpha_meq_specificity_vs_sham.py
#
# Manuscript Figure 7 (Results 4.5):
#   Sham-controlled evaluation of alpha-MEQ coupling across PBM
#   frequencies.
#
# Panels and the tables they feed:
#   A. MEQ-alpha feature correlations across all five conditions
#      -> Table S3 (FDR across six features within each condition)
#   B. Active-vs-sham timepoint contrasts, 40 Hz and 120 Hz
#      -> Table S4A (FDR across 10 timepoint-by-condition estimates)
#   C. Active-vs-sham alpha feature effects (SD units)
#      -> Table S4B (FDR across 12 feature-by-condition estimates)
#   D. Active-vs-sham alpha-MEQ coupling interactions
#      -> Table 2 (FDR across 12 feature-by-condition estimates)
#
# Cond3 / 40 Hz uses ZapLine source features.
# Cond1 / 4 Hz, Cond2 / 10 Hz, Cond4 / 120 Hz and Cond5 / sham use the
# standard source pipeline.
#
# Panels B, C and D show 95% Wald confidence intervals (estimate +/- 1.96*SE)
# on the model estimates. These are regression estimates, not correlations,
# so no Fisher z-transform is involved (unlike Figures 4 and 6).
#
# p and q values are not printed on the figure; they are reported in the
# manuscript tables. Asterisks mark estimates meeting the FDR threshold.
#
# Output files are prefixed "figure7_" / "Figure7_"; the four statistics
# files also name their panel and table (e.g. figure7D_table2_...).
# ============================================================

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.stats import t as t_dist

from pbm_paper_common import (
    OUT_TABLES, OUT_STATS, OUT_FIGS,
    BAND_SCHEME, MAX_DURING_WINDOWS, LATE_DURING_WINDOWS,
    clean_subject_id, clean_condition, is_schaefer_brain_roi, roi_to_network,
    build_source_trajectory_table, load_score_table, fdr_bh, savefig,
)

from pbm_sham_interaction_common import (
    active_vs_sham_interaction_stats,
    fit_subject_fixed_ols,
    fit_mixedlm_formula,
    mixedlm_linear_contrast,
)


BAND = "Alpha"
METRIC = "relative"
NETWORK = "all_rois"
SCORE_COL = "MEQ_Total"

# Wald multiplier for 95% intervals on model estimates
Z_CRIT = 1.96

ACTIVE_CONDITIONS = ["Cond3", "Cond4"]
SHAM_CONDITION = "Cond5"
MATRIX_CONDITIONS = ["Cond1", "Cond2", "Cond3", "Cond4", "Cond5"]
CONDITION_LABELS_SHORT = {
    "Cond1": "4 Hz",
    "Cond2": "10 Hz",
    "Cond3": "40 Hz",
    "Cond4": "120 Hz",
    "Cond5": "Sham",
}
CONDITION_COLORS = {
    "Cond1": "#4C78A8",
    "Cond2": "#9467BD",
    "Cond3": "#D95F02",
    "Cond4": "#D62728",
    "Cond5": "#6F7F8F",
}

TIME_COLS = ["D1_change", "D2_change", "D3_change", "D4_change", "POST_change"]
TIME_LABELS = ["Dur5", "Dur10", "Dur15", "Dur20", "Post"]

KEY_FEATURES = [
    "D1_change",
    "LateDuringMean_change",
    "D4_change",
    "POST_change",
    "RecoveryFromD4",
    "ReboundIndex",
]

FEATURE_LABELS_FIG = {
    "D1_change": "Dur5",
    "D2_change": "Dur10",
    "D3_change": "Dur15",
    "D4_change": "Dur20",
    "POST_change": "Post",
    "LateDuringMean_change": "mean(Dur10_15_20)",
    "RecoveryFromD4": "Post-Dur20",
    "ReboundIndex": "Post-mean(Dur10_15_20)",
}

FEATURE_LABELS_FIG_WRAPPED = {
    **FEATURE_LABELS_FIG,
    "LateDuringMean_change": "mean\n(Dur10_15_20)",
    "ReboundIndex": "Post-mean\n(Dur10_15_20)",
}

FEATURE_NOTE = (
    "Dur5, Dur10, Dur15, Dur20, and Post are changes from PRE; "
    "mean(Dur10_15_20)=mean(Dur10,Dur15,Dur20); Post-Dur20=Post minus Dur20."
)

ERROR_BAR_NOTE = (
    "Panels B, C and D: error bars are 95% Wald confidence intervals (estimate +/- 1.96*SE) on the "
    "active-minus-sham model estimate, and are uncorrected; q values are Benjamini-Hochberg "
    "FDR-corrected (Panel B across the 10 timepoint contrasts; Panels C and D across the 12 "
    "feature-by-condition estimates) and are reported in the manuscript tables. Bars and markers not "
    "meeting the FDR threshold are shown semi-transparent; asterisks mark estimates meeting the "
    "threshold. Features are algebraically interdependent (e.g. Post-Dur20 = Post - Dur20), so these "
    "are not independent tests; overlapping intervals mean features cannot be ranked by effect size. "
    "Panel A shows within-condition Pearson r and carries no error bars; its q values are "
    "FDR-corrected within each condition (6 features)."
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "Data"

ZAPLINE_PRE_POST_FEATURES_CSV = (
    DATA_ROOT
    / "source_results"
    / "source_band_power_40Hz"
    / "source_roi_band_power_all_subjects_conditions_long.csv"
)
ZAPLINE_DURING_5MIN_FEATURES_CSV = (
    DATA_ROOT
    / "source_results"
    / "source_band_power_40Hz"
    / "during_5min_source_roi_band_power_all_subjects_conditions_long.csv"
)

OUT_DIR = OUT_FIGS / "figure7_pbm_vs_sham_specificity"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_zapline_source_csv(path, require_window=False):
    if not path.exists():
        raise FileNotFoundError(f"ZapLine source feature file not found:\n{path}")

    df = pd.read_csv(path)
    required = {"Subject", "Condition", "Phase", "ROI", "Band", "Metric", "Value", "BandScheme"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"ZapLine source feature file is missing required columns: {missing}")

    for col in ["Subject", "Condition", "Phase", "ROI", "Band", "Metric", "BandScheme"]:
        df[col] = df[col].astype(str).str.strip()

    df["Subject"] = df["Subject"].apply(clean_subject_id)
    df["Condition"] = df["Condition"].apply(clean_condition)
    df["Phase"] = df["Phase"].astype(str).str.upper().str.strip()
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df[df["ROI"].apply(is_schaefer_brain_roi)].copy()
    df["Network"] = df["ROI"].apply(roi_to_network)
    df["FeatureSource"] = "ZapLine40_Cond3"

    if require_window:
        if "WindowIndex" not in df.columns:
            raise ValueError("ZapLine DURING source feature file must contain WindowIndex.")
        df["WindowIndex"] = pd.to_numeric(df["WindowIndex"], errors="coerce")
        df = df.dropna(subset=["WindowIndex"]).copy()
        df["WindowIndex"] = df["WindowIndex"].astype(int)
        if MAX_DURING_WINDOWS is not None:
            df = df[df["WindowIndex"] < MAX_DURING_WINDOWS].copy()

    return df


def build_zapline_cond3_trajectory_table(
    band=BAND,
    metric=METRIC,
    network=NETWORK,
    band_scheme=BAND_SCHEME,
):
    whole = _load_zapline_source_csv(ZAPLINE_PRE_POST_FEATURES_CSV, require_window=False)
    during = _load_zapline_source_csv(ZAPLINE_DURING_5MIN_FEATURES_CSV, require_window=True)

    whole = whole[
        (whole["Condition"] == "Cond3")
        & (whole["Phase"].isin(["PRE", "POST"]))
        & (whole["Band"] == band)
        & (whole["Metric"] == metric)
        & (whole["BandScheme"] == band_scheme)
    ].copy()
    during = during[
        (during["Condition"] == "Cond3")
        & (during["Phase"] == "DURING")
        & (during["Band"] == band)
        & (during["Metric"] == metric)
        & (during["BandScheme"] == band_scheme)
    ].copy()

    if network != "all_rois":
        whole = whole[whole["Network"] == network].copy()
        during = during[during["Network"] == network].copy()

    if whole.empty or during.empty:
        raise RuntimeError("No ZapLine Cond3 rows were available after filtering.")

    whole_phase = (
        whole.groupby(["Subject", "Condition", "Phase"], as_index=False)["Value"]
        .mean()
        .rename(columns={"Value": "MeanSourcePower"})
    )
    whole_wide = whole_phase.pivot_table(
        index=["Subject", "Condition"],
        columns="Phase",
        values="MeanSourcePower",
        aggfunc="mean",
    ).reset_index()

    for phase in ["PRE", "POST"]:
        if phase not in whole_wide.columns:
            whole_wide[phase] = np.nan

    during_phase = (
        during.groupby(["Subject", "Condition", "WindowIndex"], as_index=False)["Value"]
        .mean()
        .rename(columns={"Value": "MeanSourcePower"})
    )
    window_indices = sorted(during_phase["WindowIndex"].dropna().unique().astype(int).tolist())
    if MAX_DURING_WINDOWS is not None:
        window_indices = [w for w in window_indices if w < MAX_DURING_WINDOWS]

    during_wide = during_phase.pivot_table(
        index=["Subject", "Condition"],
        columns="WindowIndex",
        values="MeanSourcePower",
        aggfunc="mean",
    ).reset_index()
    during_wide = during_wide.rename(columns={idx: f"D{idx + 1}" for idx in window_indices})
    during_cols = [f"D{idx + 1}" for idx in window_indices]

    out = whole_wide.merge(during_wide, on=["Subject", "Condition"], how="inner")
    if out.empty:
        raise RuntimeError("ZapLine Cond3 PRE/POST and DURING rows did not overlap by subject.")

    out["PRE_change"] = 0.0
    for col in during_cols:
        if col in out.columns:
            out[f"{col}_change"] = out[col] - out["PRE"]

    out["POST_change"] = out["POST"] - out["PRE"]
    during_change_cols = [f"{c}_change" for c in during_cols if f"{c}_change" in out.columns]
    late_cols = [f"{c}_change" for c in LATE_DURING_WINDOWS if f"{c}_change" in out.columns]

    out["DURING_mean_change"] = out[during_change_cols].mean(axis=1)
    out["LateDuringMean_change"] = out[late_cols].mean(axis=1)
    out["DuringSuppression"] = -out["LateDuringMean_change"]
    out["D4_change"] = out["D4_change"] if "D4_change" in out.columns else np.nan
    out["ReboundIndex"] = out["POST_change"] - out["LateDuringMean_change"]
    out["RecoveryFromD4"] = out["POST_change"] - out["D4_change"] if "D4_change" in out.columns else np.nan

    slope_cols = ["PRE_change"] + during_change_cols + ["POST_change"]

    def slope(row):
        y = np.array([row[c] for c in slope_cols], dtype=float)
        x = np.arange(len(y), dtype=float)
        mask = np.isfinite(y)
        if mask.sum() < 2:
            return np.nan
        return np.polyfit(x[mask], y[mask], deg=1)[0]

    out["Trajectory_slope"] = out.apply(slope, axis=1)
    out["Band"] = band
    out["Metric"] = metric
    out["Network"] = network
    out["BandScheme"] = band_scheme
    out["FeatureSource"] = "ZapLine40_Cond3"

    return out


def build_feature_score_table():
    standard = build_source_trajectory_table(
        band=BAND,
        metric=METRIC,
        network=NETWORK,
        band_scheme=BAND_SCHEME,
    )
    standard = standard[standard["Condition"].isin(["Cond1", "Cond2", "Cond4", "Cond5"])].copy()
    standard["FeatureSource"] = "StandardSourcePipeline"

    zapline_40 = build_zapline_cond3_trajectory_table()
    features = pd.concat([zapline_40, standard], ignore_index=True, sort=False)
    features = features[features["Condition"].isin(MATRIX_CONDITIONS)].copy()

    scores = load_score_table(score_col=SCORE_COL)
    merged = features.merge(scores, on=["Subject", "Condition"], how="inner")

    features.to_csv(OUT_TABLES / "figure7_alpha_pbm_vs_sham_features.csv", index=False)
    merged.to_csv(OUT_TABLES / "figure7_alpha_pbm_vs_sham_features_with_MEQ.csv", index=False)

    print("Feature source by condition:")
    print(features.groupby(["Condition", "FeatureSource"])["Subject"].nunique())
    print("\nMerged N by condition:")
    print(merged.groupby("Condition")["Subject"].nunique())

    return merged


def timepoint_contrast_stats(merged):
    rows = []

    for active in ACTIVE_CONDITIONS:
        wide = merged[merged["Condition"].isin([active, SHAM_CONDITION])].copy()
        long = wide.melt(
            id_vars=["Subject", "Condition"],
            value_vars=[c for c in TIME_COLS if c in wide.columns],
            var_name="TimeColumn",
            value_name="AlphaChange",
        ).dropna(subset=["AlphaChange"])
        long["Timepoint"] = long["TimeColumn"].map(dict(zip(TIME_COLS, TIME_LABELS)))
        long["Timepoint"] = pd.Categorical(long["Timepoint"], TIME_LABELS, ordered=True)
        long["IsActive"] = (long["Condition"] == active).astype(float)

        model_name = "MixedLM_random_subject_intercept_ML"
        mixed_fit = None
        mixed_error = None

        if len(long) >= 12 and long["Condition"].nunique() == 2:
            try:
                mixed_fit = fit_mixedlm_formula(
                    "AlphaChange ~ IsActive * C(Timepoint)",
                    long,
                )
            except Exception as exc:
                mixed_error = exc

        for time_label, time_col in zip(TIME_LABELS, TIME_COLS):
            row = {
                "ActiveCondition": active,
                "ActiveLabel": CONDITION_LABELS_SHORT[active],
                "ReferenceCondition": SHAM_CONDITION,
                "ReferenceLabel": CONDITION_LABELS_SHORT[SHAM_CONDITION],
                "Timepoint": time_label,
                "Feature": time_col,
                "N_rows": int((long["Timepoint"] == time_label).sum()),
                "N_subjects": long.loc[long["Timepoint"] == time_label, "Subject"].nunique(),
                "Estimate": np.nan,
                "SE": np.nan,
                "CI_low": np.nan,
                "CI_high": np.nan,
                "P": np.nan,
                "Model": model_name,
            }

            if mixed_fit is not None:
                try:
                    weights = {"IsActive": 1.0}
                    if time_label != TIME_LABELS[0]:
                        weights[f"IsActive:C(Timepoint)[T.{time_label}]"] = 1.0

                    est, se_val, p_val = mixedlm_linear_contrast(mixed_fit, weights)
                    row["Estimate"] = est
                    row["SE"] = se_val
                    row["CI_low"] = est - Z_CRIT * se_val
                    row["CI_high"] = est + Z_CRIT * se_val
                    row["P"] = p_val
                except Exception as exc:
                    row["Model"] = f"failed MixedLM contrast: {exc}"
            else:
                sub = wide[["Subject", "Condition", time_col]].dropna()
                sub = sub.rename(columns={time_col: "AlphaChange"})
                sub["IsActive"] = (sub["Condition"] == active).astype(float)

                if len(sub) >= 6 and sub["Condition"].nunique() == 2:
                    try:
                        fit = fit_subject_fixed_ols(
                            sub,
                            outcome="AlphaChange",
                            predictors=["IsActive"],
                        )
                        est = fit.params.get("IsActive", np.nan)
                        se_val = fit.bse.get("IsActive", np.nan)
                        tcrit = t_dist.ppf(0.975, fit.df_resid) if fit.df_resid > 0 else np.nan
                        row["Estimate"] = est
                        row["SE"] = se_val
                        row["CI_low"] = est - tcrit * se_val
                        row["CI_high"] = est + tcrit * se_val
                        row["P"] = fit.pvalues.get("IsActive", np.nan)
                        row["Model"] = f"OLS_subject_fixed_intercepts_fallback: {mixed_error}"
                    except Exception as exc:
                        row["Model"] = f"failed: MixedLM={mixed_error}; fallback={exc}"

            rows.append(row)

    stats = pd.DataFrame(rows)
    adj, reject = fdr_bh(stats["P"].values, alpha=0.05)
    stats["P_FDR"] = adj
    stats["Sig_FDR"] = reject

    return stats


def physiology_feature_stats(merged):
    rows = []

    for active in ACTIVE_CONDITIONS:
        _, _, phys = active_vs_sham_interaction_stats(
            merged,
            KEY_FEATURES,
            active_condition=active,
            sham_condition=SHAM_CONDITION,
        )
        phys["ActiveCondition"] = active
        phys["ActiveLabel"] = CONDITION_LABELS_SHORT[active]
        phys["ReferenceCondition"] = SHAM_CONDITION
        phys["ReferenceLabel"] = CONDITION_LABELS_SHORT[SHAM_CONDITION]
        rows.append(phys)

    stats = pd.concat(rows, ignore_index=True)
    adj, reject = fdr_bh(stats["Active_vs_sham_feature_p"].values, alpha=0.05)
    stats["P_FDR_all_features"] = adj
    stats["Sig_FDR_all_features"] = reject

    # 95% Wald CI on the active-minus-sham feature estimate (plotted in Panel C)
    half = Z_CRIT * stats["Active_minus_sham_feature_z_SE"].astype(float)
    stats["CI_low"] = stats["Active_minus_sham_feature_z"].astype(float) - half
    stats["CI_high"] = stats["Active_minus_sham_feature_z"].astype(float) + half

    return stats


def coupling_interaction_stats(merged):
    rows = []

    for active in ACTIVE_CONDITIONS:
        _, stats, _ = active_vs_sham_interaction_stats(
            merged,
            KEY_FEATURES,
            active_condition=active,
            sham_condition=SHAM_CONDITION,
        )
        stats["ActiveCondition"] = active
        stats["ActiveLabel"] = CONDITION_LABELS_SHORT[active]
        stats["ReferenceCondition"] = SHAM_CONDITION
        stats["ReferenceLabel"] = CONDITION_LABELS_SHORT[SHAM_CONDITION]
        rows.append(stats)

    stats = pd.concat(rows, ignore_index=True)
    adj, reject = fdr_bh(stats["Interaction_p"].values, alpha=0.05)
    stats["Interaction_p_FDR_all_features"] = adj
    stats["Interaction_sig_FDR_all_features"] = reject

    # 95% Wald CI on the active-minus-sham MEQ slope (plotted in Panel D)
    half = Z_CRIT * stats["Active_minus_sham_slope_SE"].astype(float)
    stats["CI_low"] = stats["Active_minus_sham_slope"].astype(float) - half
    stats["CI_high"] = stats["Active_minus_sham_slope"].astype(float) + half

    return stats


def matrix_correlation_stats(merged):
    rows = []

    for cond in MATRIX_CONDITIONS:
        sub_cond = merged[merged["Condition"] == cond].copy()
        for feature in KEY_FEATURES:
            sub = sub_cond[[feature, "MeditationScore"]].dropna()
            row = {
                "Condition": cond,
                "ConditionLabel": CONDITION_LABELS_SHORT[cond],
                "Feature": feature,
                "N": len(sub),
                "Pearson_r": np.nan,
                "Pearson_p": np.nan,
            }
            if len(sub) >= 3:
                row["Pearson_r"], row["Pearson_p"] = pearsonr(sub[feature], sub["MeditationScore"])
            rows.append(row)

    stats = pd.DataFrame(rows)
    stats["Pearson_p_FDR_within_condition"] = np.nan
    stats["Significant_FDR_within_condition"] = False
    stats["Significant_raw_p"] = stats["Pearson_p"] < 0.05

    for cond, idx in stats.groupby("Condition").groups.items():
        idx = list(idx)
        adj, reject = fdr_bh(stats.loc[idx, "Pearson_p"].values, alpha=0.05)
        stats.loc[idx, "Pearson_p_FDR_within_condition"] = adj
        stats.loc[idx, "Significant_FDR_within_condition"] = reject

    return stats


def plot_correlation_matrix(ax, stats):
    heat = stats.pivot_table(
        index="Condition",
        columns="Feature",
        values="Pearson_r",
        aggfunc="mean",
    ).reindex(MATRIX_CONDITIONS)[KEY_FEATURES]

    raw_sig = stats.pivot_table(
        index="Condition",
        columns="Feature",
        values="Significant_raw_p",
        aggfunc="max",
    ).reindex(MATRIX_CONDITIONS)[KEY_FEATURES]

    sig = stats.pivot_table(
        index="Condition",
        columns="Feature",
        values="Significant_FDR_within_condition",
        aggfunc="max",
    ).reindex(MATRIX_CONDITIONS)[KEY_FEATURES]

    im = ax.imshow(heat.values, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)

    ax.set_xticks(np.arange(len(KEY_FEATURES)))
    ax.set_xticklabels(
        [FEATURE_LABELS_FIG_WRAPPED.get(f, f) for f in KEY_FEATURES],
        rotation=30,
        ha="right",
    )
    ax.set_yticks(np.arange(len(MATRIX_CONDITIONS)))
    ax.set_yticklabels([CONDITION_LABELS_SHORT[c] for c in MATRIX_CONDITIONS])

    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            val = heat.values[i, j]
            if not np.isfinite(val):
                continue
            star = "*" if pd.notna(sig.values[i, j]) and bool(sig.values[i, j]) else ""
            is_raw_sig = pd.notna(raw_sig.values[i, j]) and bool(raw_sig.values[i, j])
            ax.text(
                j,
                i,
                f"{val:.2f}{star}",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold" if (star or is_raw_sig) else "normal",
                color="black",
            )
            if is_raw_sig:
                rect = plt.Rectangle(
                    (j - 0.5, i - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor="#FFD400",
                    linewidth=2.6,
                )
                ax.add_patch(rect)
                inner_rect = plt.Rectangle(
                    (j - 0.46, i - 0.46),
                    0.92,
                    0.92,
                    fill=False,
                    edgecolor="black",
                    linewidth=0.9,
                )
                ax.add_patch(inner_rect)

    ax.set_title("A. MEQ-alpha feature correlations across conditions", pad=8)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Pearson r")
    ax.text(
        0.0,
        -0.30,
        FEATURE_NOTE,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )


def plot_timepoint_contrasts(ax, stats):
    x = np.arange(len(TIME_LABELS))
    offsets = {"Cond3": -0.13, "Cond4": 0.13}

    for active in ACTIVE_CONDITIONS:
        sub = stats[stats["ActiveCondition"] == active].set_index("Timepoint").reindex(TIME_LABELS)
        xpos = x + offsets[active]
        y = sub["Estimate"].to_numpy(dtype=float)
        yerr = np.vstack([
            y - sub["CI_low"].to_numpy(dtype=float),
            sub["CI_high"].to_numpy(dtype=float) - y,
        ])
        yerr = np.clip(np.nan_to_num(yerr, nan=0.0), 0, None)

        ax.errorbar(
            xpos,
            y,
            yerr=yerr,
            marker="o",
            linewidth=2.4,
            capsize=4,
            color=CONDITION_COLORS[active],
            label=f"{CONDITION_LABELS_SHORT[active]} - sham",
        )

        for i, row in enumerate(sub.itertuples()):
            if not np.isfinite(row.Estimate):
                continue
            if not (pd.notna(row.Sig_FDR) and bool(row.Sig_FDR)):
                continue
            if np.isfinite(row.CI_high) and np.isfinite(row.CI_low):
                y_text = row.CI_high + 0.004 if row.Estimate >= 0 else row.CI_low - 0.004
            else:
                y_text = row.Estimate
            ax.text(
                xpos[i],
                y_text,
                "*",
                ha="center",
                va="bottom" if row.Estimate >= 0 else "top",
                fontsize=11,
                color=CONDITION_COLORS[active],
            )

    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(TIME_LABELS)
    ax.set_ylabel("Active - sham alpha change\n(subject-adjusted estimate)")
    ax.set_title("B. Active-vs-sham trajectory contrasts", pad=8)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    ax.margins(y=0.12)


def plot_feature_summary(ax, stats):
    x = np.arange(len(KEY_FEATURES))
    width = 0.34
    offsets = {"Cond3": -width / 2, "Cond4": width / 2}
    star_pad = 0.03

    for active in ACTIVE_CONDITIONS:
        sub = stats[stats["ActiveCondition"] == active].copy()
        sub["Feature"] = pd.Categorical(sub["Feature"], KEY_FEATURES, ordered=True)
        sub = sub.sort_values("Feature")
        y = sub["Active_minus_sham_feature_z"].to_numpy(dtype=float)
        half = Z_CRIT * sub["Active_minus_sham_feature_z_SE"].to_numpy(dtype=float)
        yerr = np.clip(np.nan_to_num(half, nan=0.0), 0, None)
        xpos = x + offsets[active]

        bars = ax.bar(
            xpos,
            y,
            width=width,
            color=CONDITION_COLORS[active],
            label=f"{CONDITION_LABELS_SHORT[active]} - sham",
            yerr=yerr,
            capsize=4,
            error_kw=dict(ecolor="black", elinewidth=1.2, capthick=1.2),
        )

        # Bars not meeting the FDR threshold are drawn lighter, so the graphic
        # matches the q column rather than the (uncorrected) error bars.
        # NOTE: ax.bar(alpha=...) takes a scalar only, hence the per-bar loop.
        survives = sub["Sig_FDR_all_features"].fillna(False).to_numpy(dtype=bool)
        for bar, ok in zip(bars, survives):
            bar.set_alpha(0.86 if ok else 0.38)

        for i, row in enumerate(sub.itertuples()):
            est = row.Active_minus_sham_feature_z
            if not np.isfinite(est):
                continue
            if not (pd.notna(row.Sig_FDR_all_features) and bool(row.Sig_FDR_all_features)):
                continue
            se_val = row.Active_minus_sham_feature_z_SE
            h = Z_CRIT * se_val if np.isfinite(se_val) else 0.0
            if est >= 0:
                y_text, va = est + h + star_pad, "bottom"
            else:
                y_text, va = est - h - star_pad, "top"
            ax.text(xpos[i], y_text, "*", ha="center", va=va, fontsize=11)

    labels = [FEATURE_LABELS_FIG_WRAPPED.get(f, f) for f in KEY_FEATURES]
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Active - sham alpha feature\n(SD units)")
    ax.set_title("C. Alpha feature effects vs sham", pad=8)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    ax.margins(y=0.12)


def plot_coupling_summary(ax, stats):
    x = np.arange(len(KEY_FEATURES))
    width = 0.34
    offsets = {"Cond3": -width / 2, "Cond4": width / 2}
    star_pad = 0.4

    for active in ACTIVE_CONDITIONS:
        sub = stats[stats["ActiveCondition"] == active].copy()
        sub["Feature"] = pd.Categorical(sub["Feature"], KEY_FEATURES, ordered=True)
        sub = sub.sort_values("Feature")
        y = sub["Active_minus_sham_slope"].to_numpy(dtype=float)
        half = Z_CRIT * sub["Active_minus_sham_slope_SE"].to_numpy(dtype=float)
        yerr = np.clip(np.nan_to_num(half, nan=0.0), 0, None)
        xpos = x + offsets[active]

        bars = ax.bar(
            xpos,
            y,
            width=width,
            color=CONDITION_COLORS[active],
            label=f"{CONDITION_LABELS_SHORT[active]} - sham",
            yerr=yerr,
            capsize=4,
            error_kw=dict(ecolor="black", elinewidth=1.2, capthick=1.2),
        )

        survives = sub["Interaction_sig_FDR_all_features"].fillna(False).to_numpy(dtype=bool)
        for bar, ok in zip(bars, survives):
            bar.set_alpha(0.86 if ok else 0.38)

        for i, row in enumerate(sub.itertuples()):
            est = row.Active_minus_sham_slope
            if not np.isfinite(est):
                continue
            sig = row.Interaction_sig_FDR_all_features
            if not (pd.notna(sig) and bool(sig)):
                continue
            se_val = row.Active_minus_sham_slope_SE
            h = Z_CRIT * se_val if np.isfinite(se_val) else 0.0
            if est >= 0:
                y_text, va = est + h + star_pad, "bottom"
            else:
                y_text, va = est - h - star_pad, "top"
            ax.text(xpos[i], y_text, "*", ha="center", va=va, fontsize=11)

    labels = [FEATURE_LABELS_FIG_WRAPPED.get(f, f) for f in KEY_FEATURES]
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Active - sham MEQ slope\nper 1 SD alpha feature")
    ax.set_title("D. Alpha-MEQ coupling interactions vs sham", pad=8)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    ax.margins(y=0.12)


def plot_main_figure(matrix_stats, time_stats, feature_stats, coupling_stats):
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(18.4, 10.2),
        gridspec_kw={"wspace": 0.31, "hspace": 0.52},
    )

    plot_correlation_matrix(axes[0, 0], matrix_stats)
    plot_timepoint_contrasts(axes[0, 1], time_stats)
    plot_feature_summary(axes[1, 0], feature_stats)
    plot_coupling_summary(axes[1, 1], coupling_stats)

    fig.text(
        0.01,
        -0.02,
        ERROR_BAR_NOTE,
        ha="left",
        va="top",
        fontsize=8,
        wrap=True,
    )

    savefig(fig, OUT_DIR / "Figure7_PBM_vs_sham_specificity.png")


merged = build_feature_score_table()

matrix_stats = matrix_correlation_stats(merged)
time_stats = timepoint_contrast_stats(merged)
feature_stats = physiology_feature_stats(merged)
coupling_stats = coupling_interaction_stats(merged)

matrix_stats.to_csv(
    OUT_STATS / "figure7A_tableS3_all_condition_MEQ_alpha_feature_correlation_matrix.csv",
    index=False,
)
time_stats.to_csv(
    OUT_STATS / "figure7B_tableS4A_PBM_vs_sham_timepoint_contrasts.csv",
    index=False,
)
feature_stats.to_csv(
    OUT_STATS / "figure7C_tableS4B_PBM_vs_sham_feature_effects.csv",
    index=False,
)
coupling_stats.to_csv(
    OUT_STATS / "figure7D_table2_PBM_vs_sham_MEQ_coupling_interactions.csv",
    index=False,
)

print("\nTimepoint active-vs-sham contrasts:")
print(time_stats.sort_values("P"))
print("\nAll-condition MEQ-alpha feature correlation matrix:")
print(matrix_stats.sort_values("Pearson_p"))
print("\nFeature active-vs-sham physiology effects:")
print(feature_stats.sort_values("Active_vs_sham_feature_p"))
print("\nActive-vs-sham MEQ coupling interactions:")
print(coupling_stats.sort_values("Interaction_p"))

plot_main_figure(matrix_stats, time_stats, feature_stats, coupling_stats)

print("\nDONE: Figure 7 PBM-vs-sham specificity figure.")
print(f"Output folder:\n{OUT_DIR}")
