# ============================================================
# FigS5B_alpha_network_summary_bars.py
#
# Supplementary Figure S5B (Supplementary S.5):
#   Network-level summaries of source-alpha dynamics.
#
# Active-vs-sham estimates come from subject-adjusted mixed models with
# 95% Wald CIs, FDR-corrected across all network x feature x condition
# tests (e.g. limbic Post alpha, 40 Hz vs sham, reported in S.5).
#
# Panels:
#   A. Alpha mean(Dur10_15_20) by network and condition
#   B. Alpha Post by network and condition
#   C. Alpha Post-Dur20 by network and condition
#   D. Active-vs-sham alpha mean(Dur10_15_20) by network
#   E. Active-vs-sham alpha Post by network
#   F. Active-vs-sham alpha Post-Dur20 by network
#
# Cond3 / 40 Hz uses ZapLine source features.
# Other conditions use the cleaned/interpolated source paths from
# pbm_paper_common.py.
# ============================================================

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pbm_paper_common import (
    OUT_FIGS,
    OUT_STATS,
    OUT_TABLES,
    BAND_SCHEME,
    MAX_DURING_WINDOWS,
    LATE_DURING_WINDOWS,
    CONDITION_LABELS,
    clean_condition,
    clean_subject_id,
    fdr_bh,
    is_schaefer_brain_roi,
    load_source_during_5min,
    load_source_whole,
    roi_to_network,
    savefig,
    sem,
)

from pbm_sham_interaction_common import (
    fit_mixedlm_formula,
    fit_subject_fixed_ols,
    mixedlm_term_stats,
    p_text,
)


BAND = "Alpha"
METRIC = "relative"
FIGURE_CONDITIONS = ["Cond1", "Cond2", "Cond3", "Cond4", "Cond5"]
ACTIVE_CONDITIONS = ["Cond3", "Cond4"]
SHAM_CONDITION = "Cond5"
NETWORKS = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default", "TempPar"]

FEATURES = ["LateDuringMean_change", "POST_change", "RecoveryFromD4"]
FEATURE_LABELS = {
    "LateDuringMean_change": "mean(Dur10_15_20)",
    "POST_change": "Post",
    "RecoveryFromD4": "Post-Dur20",
}

CONDITION_COLORS = {
    "Cond1": "#4C78A8",
    "Cond2": "#9467BD",
    "Cond3": "#D95F02",
    "Cond4": "#D62728",
    "Cond5": "#6F7F8F",
}

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

OUT_DIR = OUT_FIGS / "figureS5B_alpha_network_summary_bars"
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

    if require_window:
        if "WindowIndex" not in df.columns:
            raise ValueError("ZapLine DURING source feature file must contain WindowIndex.")
        df["WindowIndex"] = pd.to_numeric(df["WindowIndex"], errors="coerce")
        df = df.dropna(subset=["WindowIndex"]).copy()
        df["WindowIndex"] = df["WindowIndex"].astype(int)
        if MAX_DURING_WINDOWS is not None:
            df = df[df["WindowIndex"] < MAX_DURING_WINDOWS].copy()

    return df


def build_trajectory_from_loaded_tables(whole, during, conditions, feature_source):
    whole = whole[
        (whole["Condition"].isin(conditions))
        & (whole["Network"].isin(NETWORKS))
        & (whole["Band"] == BAND)
        & (whole["Metric"] == METRIC)
        & (whole["BandScheme"] == BAND_SCHEME)
        & (whole["Phase"].isin(["PRE", "POST"]))
    ].copy()

    during = during[
        (during["Condition"].isin(conditions))
        & (during["Network"].isin(NETWORKS))
        & (during["Band"] == BAND)
        & (during["Metric"] == METRIC)
        & (during["BandScheme"] == BAND_SCHEME)
        & (during["Phase"] == "DURING")
    ].copy()

    if whole.empty or during.empty:
        return pd.DataFrame()

    whole_phase = (
        whole.groupby(["Subject", "Condition", "Network", "Phase"], as_index=False)["Value"]
        .mean()
        .rename(columns={"Value": "MeanSourcePower"})
    )
    whole_wide = whole_phase.pivot_table(
        index=["Subject", "Condition", "Network"],
        columns="Phase",
        values="MeanSourcePower",
        aggfunc="mean",
    ).reset_index()
    for phase in ["PRE", "POST"]:
        if phase not in whole_wide.columns:
            whole_wide[phase] = np.nan

    during_phase = (
        during.groupby(["Subject", "Condition", "Network", "WindowIndex"], as_index=False)["Value"]
        .mean()
        .rename(columns={"Value": "MeanSourcePower"})
    )
    window_indices = sorted(during_phase["WindowIndex"].dropna().unique().astype(int).tolist())
    if MAX_DURING_WINDOWS is not None:
        window_indices = [w for w in window_indices if w < MAX_DURING_WINDOWS]

    during_wide = during_phase.pivot_table(
        index=["Subject", "Condition", "Network"],
        columns="WindowIndex",
        values="MeanSourcePower",
        aggfunc="mean",
    ).reset_index()
    during_wide = during_wide.rename(columns={idx: f"D{idx + 1}" for idx in window_indices})
    during_cols = [f"D{idx + 1}" for idx in window_indices]

    out = whole_wide.merge(during_wide, on=["Subject", "Condition", "Network"], how="inner")
    if out.empty:
        return out

    out["PRE_change"] = 0.0
    for col in during_cols:
        if col in out.columns:
            out[f"{col}_change"] = out[col] - out["PRE"]

    out["POST_change"] = out["POST"] - out["PRE"]
    during_change_cols = [f"{c}_change" for c in during_cols if f"{c}_change" in out.columns]
    late_cols = [f"{c}_change" for c in LATE_DURING_WINDOWS if f"{c}_change" in out.columns]
    out["DURING_mean_change"] = out[during_change_cols].mean(axis=1)
    out["LateDuringMean_change"] = out[late_cols].mean(axis=1)
    out["RecoveryFromD4"] = out["POST_change"] - out["D4_change"] if "D4_change" in out.columns else np.nan
    out["ReboundIndex"] = out["POST_change"] - out["LateDuringMean_change"]
    out["Band"] = BAND
    out["Metric"] = METRIC
    out["BandScheme"] = BAND_SCHEME
    out["FeatureSource"] = feature_source

    return out


def build_alpha_network_table():
    print("Loading standard cleaned/interpolated source tables once...")
    standard_whole = load_source_whole()
    standard_during = load_source_during_5min()
    standard = build_trajectory_from_loaded_tables(
        standard_whole,
        standard_during,
        conditions=["Cond1", "Cond2", "Cond4", "Cond5"],
        feature_source="StandardSourcePipeline",
    )

    print("Loading ZapLine 40 Hz source tables once...")
    zapline_whole = _load_zapline_source_csv(ZAPLINE_PRE_POST_FEATURES_CSV, require_window=False)
    zapline_during = _load_zapline_source_csv(ZAPLINE_DURING_5MIN_FEATURES_CSV, require_window=True)
    cond3 = build_trajectory_from_loaded_tables(
        zapline_whole,
        zapline_during,
        conditions=["Cond3"],
        feature_source="ZapLine40_Cond3",
    )

    out = pd.concat([standard, cond3], ignore_index=True, sort=False)
    out = out[out["Condition"].isin(FIGURE_CONDITIONS)].copy()
    out["Network"] = pd.Categorical(out["Network"], NETWORKS, ordered=True)
    out["Condition"] = pd.Categorical(out["Condition"], FIGURE_CONDITIONS, ordered=True)

    return out.sort_values(["Network", "Condition", "Subject"]).reset_index(drop=True)


def active_vs_sham_stats(df):
    rows = []

    for network in NETWORKS:
        for feature in FEATURES:
            for active in ACTIVE_CONDITIONS:
                sub = df[
                    (df["Network"] == network)
                    & (df["Condition"].isin([active, SHAM_CONDITION]))
                ][["Subject", "Condition", feature]].dropna().copy()
                sub = sub.rename(columns={feature: "Value"})
                sub["IsActive"] = (sub["Condition"] == active).astype(float)

                row = {
                    "Network": network,
                    "Feature": feature,
                    "FeatureLabel": FEATURE_LABELS[feature],
                    "ActiveCondition": active,
                    "ActiveLabel": CONDITION_LABELS[active],
                    "ReferenceCondition": SHAM_CONDITION,
                    "ReferenceLabel": CONDITION_LABELS[SHAM_CONDITION],
                    "N_rows": len(sub),
                    "N_subjects": sub["Subject"].nunique(),
                    "Estimate": np.nan,
                    "SE": np.nan,
                    "CI_low": np.nan,
                    "CI_high": np.nan,
                    "P": np.nan,
                    "Model": "MixedLM_random_subject_intercept_ML",
                }

                if len(sub) >= 6 and sub["Condition"].nunique() == 2:
                    try:
                        fit = fit_mixedlm_formula("Value ~ IsActive", sub)
                        est, se_val, p_val = mixedlm_term_stats(fit, "IsActive")
                        row["Estimate"] = est
                        row["SE"] = se_val
                        row["P"] = p_val
                    except Exception as exc:
                        try:
                            fit = fit_subject_fixed_ols(
                                sub,
                                outcome="Value",
                                predictors=["IsActive"],
                            )
                            row["Estimate"] = fit.params.get("IsActive", np.nan)
                            row["SE"] = fit.bse.get("IsActive", np.nan)
                            row["P"] = fit.pvalues.get("IsActive", np.nan)
                            row["Model"] = f"OLS_subject_fixed_intercepts_fallback: {exc}"
                        except Exception as fallback_exc:
                            row["Model"] = f"failed: MixedLM={exc}; fallback={fallback_exc}"

                if np.isfinite(row["Estimate"]) and np.isfinite(row["SE"]):
                    row["CI_low"] = row["Estimate"] - 1.96 * row["SE"]
                    row["CI_high"] = row["Estimate"] + 1.96 * row["SE"]

                rows.append(row)

    stats = pd.DataFrame(rows)
    q, reject = fdr_bh(stats["P"].values, alpha=0.05)
    stats["P_FDR_all_network_feature_tests"] = q
    stats["Significant_FDR"] = reject
    stats["Significant_raw_p"] = stats["P"] < 0.05

    return stats


def plot_condition_bars(ax, df, feature, title):
    x = np.arange(len(NETWORKS))
    width = 0.15
    offsets = np.linspace(-2, 2, len(FIGURE_CONDITIONS)) * width

    for offset, cond in zip(offsets, FIGURE_CONDITIONS):
        means = []
        errors = []
        for network in NETWORKS:
            vals = df[(df["Network"] == network) & (df["Condition"] == cond)][feature]
            means.append(vals.mean())
            errors.append(sem(vals))

        ax.bar(
            x + offset,
            means,
            width=width,
            yerr=errors,
            capsize=2.5,
            color=CONDITION_COLORS[cond],
            alpha=0.86,
            label=CONDITION_LABELS[cond],
        )

    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(NETWORKS, rotation=25, ha="right")
    ax.set_ylabel("Alpha relative power\nchange from PRE")
    ax.set_title(title, pad=8)
    ax.grid(True, axis="y", alpha=0.25)
    ax.margins(y=0.18)


def plot_sham_contrast_bars(ax, stats, feature, title):
    sub_all = stats[stats["Feature"] == feature].copy()
    x = np.arange(len(NETWORKS))
    width = 0.34
    offsets = {"Cond3": -width / 2, "Cond4": width / 2}

    for active in ACTIVE_CONDITIONS:
        sub = sub_all[sub_all["ActiveCondition"] == active].copy()
        sub["Network"] = pd.Categorical(sub["Network"], NETWORKS, ordered=True)
        sub = sub.sort_values("Network")

        y = sub["Estimate"].to_numpy(dtype=float)
        ci_low = sub["CI_low"].to_numpy(dtype=float)
        ci_high = sub["CI_high"].to_numpy(dtype=float)
        yerr = np.vstack([y - ci_low, ci_high - y])
        xpos = x + offsets[active]

        ax.bar(
            xpos,
            y,
            width=width,
            yerr=yerr,
            capsize=3,
            color=CONDITION_COLORS[active],
            alpha=0.86,
            label=f"{CONDITION_LABELS[active]} - sham",
        )

        for i, row in enumerate(sub.itertuples()):
            if not np.isfinite(row.Estimate):
                continue
            star = "*" if bool(row.Significant_FDR) else ""
            if np.isfinite(row.CI_high) and np.isfinite(row.CI_low):
                y_text = row.CI_high + 0.006 if row.Estimate >= 0 else row.CI_low - 0.006
            else:
                y_text = row.Estimate
            ax.text(
                xpos[i],
                y_text,
                f"p={p_text(row.P)}\nq={p_text(row.P_FDR_all_network_feature_tests)}{star}",
                ha="center",
                va="bottom" if row.Estimate >= 0 else "top",
                fontsize=6.5,
                rotation=90,
            )

    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(NETWORKS, rotation=25, ha="right")
    ax.set_ylabel("Active - sham alpha change\n(subject-adjusted estimate)")
    ax.set_title(title, pad=8)
    ax.grid(True, axis="y", alpha=0.25)
    ax.margins(y=0.30)


def plot_figure(df, stats):
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(18.0, 12.8),
        gridspec_kw={"wspace": 0.25, "hspace": 0.50},
    )

    plot_condition_bars(
        axes[0, 0],
        df,
        "LateDuringMean_change",
        "A. Alpha mean(Dur10_15_20) by network and condition",
    )
    plot_condition_bars(
        axes[0, 1],
        df,
        "POST_change",
        "B. Alpha Post by network and condition",
    )
    plot_condition_bars(
        axes[1, 0],
        df,
        "RecoveryFromD4",
        "C. Alpha Post-Dur20 by network and condition",
    )
    plot_sham_contrast_bars(
        axes[1, 1],
        stats,
        "LateDuringMean_change",
        "D. Alpha mean(Dur10_15_20): active vs sham",
    )
    plot_sham_contrast_bars(
        axes[2, 0],
        stats,
        "POST_change",
        "E. Alpha Post: active vs sham",
    )
    plot_sham_contrast_bars(
        axes[2, 1],
        stats,
        "RecoveryFromD4",
        "F. Alpha Post-Dur20: active vs sham",
    )

    axes[0, 0].legend(frameon=False, ncol=3, fontsize=8)
    axes[1, 1].legend(frameon=False, fontsize=8)

    fig.text(
        0.5,
        0.012,
        (
            "Bars show mean +/- SEM for condition panels and model estimate +/- 95% CI for active-vs-sham panels. "
            "40 Hz uses ZapLine source features; other conditions use the cleaned/interpolated source pipeline."
        ),
        ha="center",
        va="bottom",
        fontsize=8,
    )

    return fig


features = build_alpha_network_table()
features.to_csv(OUT_TABLES / "figureS5B_alpha_network_summary_features.csv", index=False)

stats = active_vs_sham_stats(features)
stats.to_csv(OUT_STATS / "figureS5B_alpha_network_summary_active_vs_sham_stats.csv", index=False)

print("\nN by network and condition:")
print(
    features.groupby(["Network", "Condition"])["Subject"]
    .nunique()
    .reset_index(name="N")
    .to_string(index=False)
)

print("\nActive-vs-sham alpha network stats:")
print(stats.sort_values("P").to_string(index=False))

fig = plot_figure(features, stats)
savefig(fig, OUT_DIR / "FigureS5B_alpha_network_summary_bars.png")

print("\nDONE: Figure S5B alpha network summary bar figure.")
print(f"Output folder:\n{OUT_DIR}")
