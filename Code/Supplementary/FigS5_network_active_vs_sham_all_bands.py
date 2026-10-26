# ============================================================
# FigS5_network_active_vs_sham_all_bands.py
#
# Supplementary S.5 (text statistics accompanying Figures S5A/S5B):
#   All-band source-network active-vs-sham relative-power models.
#
# Tests active-vs-sham source relative-power changes in each Schaefer
# network and frequency band, for all four active conditions, using
# subject-adjusted mixed models (participant random intercept), for:
#   - LateDuringMean_change  = mean(Dur10_15_20)
#   - POST_change            = Post
#   - RecoveryFromD4         = Post-Dur20
#
# FDR is applied across all network x band x condition x feature tests
# and, separately, within each feature (both reported in S.5).
#
# Cond3 / 40 Hz uses the ZapLine-cleaned source features.
# Output files are prefixed "figureS5_" / "FigureS5_".
# ============================================================

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pbm_paper_common import (
    OUT_TABLES, OUT_STATS, OUT_FIGS,
    CONDITION_LABELS, BAND_SCHEME, MAX_DURING_WINDOWS, LATE_DURING_WINDOWS,
    clean_subject_id, clean_condition, is_schaefer_brain_roi, roi_to_network,
    build_source_trajectory_table, load_source_whole, fdr_bh, savefig,
)

from pbm_sham_interaction_common import (
    fit_mixedlm_formula,
    fit_subject_fixed_ols,
    mixedlm_term_stats,
    p_text,
)


METRIC = "relative"
SHAM_CONDITION = "Cond5"
ACTIVE_CONDITIONS = ["Cond1", "Cond2", "Cond3", "Cond4"]
FIGURE_CONDITIONS = ACTIVE_CONDITIONS + [SHAM_CONDITION]
NETWORKS = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default", "TempPar"]
PREFERRED_BANDS = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
FEATURES = ["LateDuringMean_change", "POST_change", "RecoveryFromD4"]

FEATURE_LABELS = {
    "LateDuringMean_change": "Late-DURING",
    "POST_change": "POST",
    "RecoveryFromD4": "Recovery from D4",
}

CONDITION_COLORS = {
    "Cond1": "#4C78A8",
    "Cond2": "#9467BD",
    "Cond3": "#D95F02",
    "Cond4": "#D62728",
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

OUT_DIR = OUT_FIGS / "figureS5_network_active_vs_sham_all_bands"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def available_bands():
    whole = load_source_whole()
    whole = whole[
        (whole["Metric"] == METRIC)
        & (whole["BandScheme"] == BAND_SCHEME)
    ].copy()

    found = sorted(whole["Band"].dropna().unique().tolist())
    preferred = [b for b in PREFERRED_BANDS if b in found]
    remaining = [b for b in found if b not in preferred]

    return preferred + remaining


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


def build_zapline_cond3_network_table(band, network):
    whole = _load_zapline_source_csv(ZAPLINE_PRE_POST_FEATURES_CSV, require_window=False)
    during = _load_zapline_source_csv(ZAPLINE_DURING_5MIN_FEATURES_CSV, require_window=True)

    whole = whole[
        (whole["Condition"] == "Cond3")
        & (whole["Phase"].isin(["PRE", "POST"]))
        & (whole["Band"] == band)
        & (whole["Metric"] == METRIC)
        & (whole["BandScheme"] == BAND_SCHEME)
        & (whole["Network"] == network)
    ].copy()
    during = during[
        (during["Condition"] == "Cond3")
        & (during["Phase"] == "DURING")
        & (during["Band"] == band)
        & (during["Metric"] == METRIC)
        & (during["BandScheme"] == BAND_SCHEME)
        & (during["Network"] == network)
    ].copy()

    if whole.empty or during.empty:
        return pd.DataFrame()

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
    out["DuringSuppression"] = -out["LateDuringMean_change"]
    out["RecoveryFromD4"] = out["POST_change"] - out["D4_change"] if "D4_change" in out.columns else np.nan
    out["ReboundIndex"] = out["POST_change"] - out["LateDuringMean_change"]
    out["Band"] = band
    out["Metric"] = METRIC
    out["Network"] = network
    out["BandScheme"] = BAND_SCHEME
    out["FeatureSource"] = "ZapLine40_Cond3"

    return out


def build_network_band_table(bands):
    rows = []

    for network in NETWORKS:
        for band in bands:
            standard = build_source_trajectory_table(
                band=band,
                metric=METRIC,
                network=network,
                band_scheme=BAND_SCHEME,
            )
            standard = standard[standard["Condition"].isin(["Cond1", "Cond2", "Cond4", "Cond5"])].copy()
            standard["FeatureSource"] = "StandardSourcePipeline"

            cond3 = build_zapline_cond3_network_table(band, network)
            if not cond3.empty:
                rows.append(pd.concat([standard, cond3], ignore_index=True, sort=False))
            else:
                rows.append(standard)

    out = pd.concat(rows, ignore_index=True, sort=False)
    out = out[out["Condition"].isin(FIGURE_CONDITIONS)].copy()

    return out


def active_vs_sham_stats(df):
    rows = []

    for network in NETWORKS:
        net_df = df[df["Network"] == network].copy()
        for band in sorted(net_df["Band"].dropna().unique()):
            band_df = net_df[net_df["Band"] == band].copy()
            for feature in FEATURES:
                for cond in ACTIVE_CONDITIONS:
                    sub = band_df[band_df["Condition"].isin([cond, SHAM_CONDITION])].copy()
                    sub = sub[["Subject", "Condition", feature]].dropna()
                    sub = sub.rename(columns={feature: "Value"})
                    sub["IsActive"] = (sub["Condition"] == cond).astype(float)

                    row = {
                        "Network": network,
                        "Band": band,
                        "Feature": feature,
                        "Condition": cond,
                        "ConditionLabel": CONDITION_LABELS[cond],
                        "Reference": SHAM_CONDITION,
                        "ReferenceLabel": CONDITION_LABELS[SHAM_CONDITION],
                        "N_rows": len(sub),
                        "N_subjects": sub["Subject"].nunique(),
                        "Estimate": np.nan,
                        "SE": np.nan,
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

                    rows.append(row)

    stats = pd.DataFrame(rows)

    adj, reject = fdr_bh(stats["P"].values, alpha=0.05)
    stats["P_FDR_all_network_band_condition_feature_tests"] = adj
    stats["Significant_FDR_all_tests"] = reject
    stats["Significant_raw_p"] = stats["P"] < 0.05

    for feature, idx in stats.groupby("Feature").groups.items():
        idx = list(idx)
        adj, reject = fdr_bh(stats.loc[idx, "P"].values, alpha=0.05)
        stats.loc[idx, "P_FDR_within_feature"] = adj
        stats.loc[idx, "Significant_FDR_within_feature"] = reject

    return stats


def plot_network_band_heatmap(ax, stats, feature, condition, title):
    sub = stats[(stats["Feature"] == feature) & (stats["Condition"] == condition)].copy()
    bands = [b for b in PREFERRED_BANDS if b in set(sub["Band"])]
    bands += [b for b in sorted(sub["Band"].dropna().unique()) if b not in bands]

    heat = sub.pivot_table(
        index="Network",
        columns="Band",
        values="Estimate",
        aggfunc="mean",
    ).reindex(NETWORKS)[bands]
    pvals = sub.pivot_table(
        index="Network",
        columns="Band",
        values="P",
        aggfunc="mean",
    ).reindex(NETWORKS)[bands]
    raw_sig = sub.pivot_table(
        index="Network",
        columns="Band",
        values="Significant_raw_p",
        aggfunc="max",
    ).reindex(NETWORKS)[bands]
    fdr_sig = sub.pivot_table(
        index="Network",
        columns="Band",
        values="Significant_FDR_within_feature",
        aggfunc="max",
    ).reindex(NETWORKS)[bands]

    vmax = np.nanmax(np.abs(heat.values))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1

    im = ax.imshow(heat.values, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(bands)))
    ax.set_xticklabels(bands, rotation=28, ha="right")
    ax.set_yticks(np.arange(len(NETWORKS)))
    ax.set_yticklabels(NETWORKS)
    ax.set_title(title, pad=8)

    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            val = heat.values[i, j]
            if not np.isfinite(val):
                continue
            p = pvals.values[i, j]
            star = "*" if pd.notna(fdr_sig.values[i, j]) and bool(fdr_sig.values[i, j]) else ""
            ax.text(
                j,
                i,
                f"{val:.3f}{star}\np={p_text(p)}",
                ha="center",
                va="center",
                fontsize=6.5,
                fontweight="bold" if star else "normal",
            )
            if pd.notna(raw_sig.values[i, j]) and bool(raw_sig.values[i, j]):
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="#FFD400",
                        linewidth=2.2,
                    )
                )

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Active - sham estimate")


bands = available_bands()
print("Available relative-power bands:", bands)

network_band = build_network_band_table(bands)
network_band.to_csv(
    OUT_TABLES / "figureS5_network_relative_power_band_trajectory_features.csv",
    index=False,
)

stats = active_vs_sham_stats(network_band)
stats.to_csv(
    OUT_STATS / "figureS5_network_relative_power_active_vs_sham_stats.csv",
    index=False,
)

print("\nNetwork relative-power active-vs-sham stats:")
print(stats.sort_values("P"))


fig1, axes1 = plt.subplots(
    2,
    2,
    figsize=(17.5, 12.0),
    gridspec_kw={"wspace": 0.30, "hspace": 0.34},
)

plot_network_band_heatmap(
    axes1[0, 0],
    stats,
    "LateDuringMean_change",
    "Cond3",
    "A. 40 Hz - sham late-DURING network power",
)
plot_network_band_heatmap(
    axes1[0, 1],
    stats,
    "LateDuringMean_change",
    "Cond4",
    "B. 120 Hz - sham late-DURING network power",
)
plot_network_band_heatmap(
    axes1[1, 0],
    stats,
    "POST_change",
    "Cond3",
    "C. 40 Hz - sham POST network power",
)
plot_network_band_heatmap(
    axes1[1, 1],
    stats,
    "POST_change",
    "Cond4",
    "D. 120 Hz - sham POST network power",
)

savefig(fig1, OUT_DIR / "FigureS5_network_relative_power_late_post_screen.png")


fig2, axes2 = plt.subplots(
    1,
    2,
    figsize=(17.5, 5.8),
    gridspec_kw={"wspace": 0.30},
)

plot_network_band_heatmap(
    axes2[0],
    stats,
    "RecoveryFromD4",
    "Cond3",
    "A. 40 Hz - sham recovery-from-D4 network power",
)
plot_network_band_heatmap(
    axes2[1],
    stats,
    "RecoveryFromD4",
    "Cond4",
    "B. 120 Hz - sham recovery-from-D4 network power",
)

savefig(fig2, OUT_DIR / "FigureS5_network_relative_power_recovery_screen.png")

print("\nDONE: network relative-power band screen.")
print(f"Output folder:\n{OUT_DIR}")
