# ============================================================
# FigS5A_network_relative_power_trajectories.py
#
# Supplementary Figure S5A (Supplementary S.5):
#   Source network relative-power trajectories across PBM conditions.
#
# Plots relative-power trajectories (change from PRE, mean +/- SEM) for
# each large-scale source network and frequency band. Trajectories only;
# no statistics are computed here.
#
# Figure layout:
#   rows    = networks
#   columns = frequency bands
#   lines   = PBM/sham conditions
#
# Cond3 / 40 Hz uses the ZapLine source-feature rerun.
# All other conditions use the standard cleaned/interpolated source path
# configured in pbm_paper_common.py.
# ============================================================

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pbm_paper_common import (
    OUT_FIGS,
    OUT_TABLES,
    BAND_SCHEME,
    MAX_DURING_WINDOWS,
    LATE_DURING_WINDOWS,
    CONDITION_LABELS,
    clean_condition,
    clean_subject_id,
    is_schaefer_brain_roi,
    roi_to_network,
    load_source_during_5min,
    load_source_whole,
    savefig,
    sem,
)


METRIC = "relative"
FIGURE_CONDITIONS = ["Cond1", "Cond2", "Cond3", "Cond4", "Cond5"]
NETWORKS = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default", "TempPar"]
PREFERRED_BANDS = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]

TIME_COLS = ["PRE_change", "D1_change", "D2_change", "D3_change", "D4_change", "POST_change"]
TIME_LABELS = ["PRE", "Dur5", "Dur10", "Dur15", "Dur20", "Post"]

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

OUT_DIR = OUT_FIGS / "figureS5A_all_network_relative_power_trajectories"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def available_bands(whole):
    whole = whole[
        (whole["Metric"] == METRIC)
        & (whole["BandScheme"] == BAND_SCHEME)
    ].copy()

    found = sorted(whole["Band"].dropna().unique().tolist())
    preferred = [band for band in PREFERRED_BANDS if band in found]
    remaining = [band for band in found if band not in preferred]
    return preferred + remaining


def build_trajectory_from_loaded_tables(whole, during, bands, conditions, feature_source):
    whole = whole[
        (whole["Condition"].isin(conditions))
        & (whole["Network"].isin(NETWORKS))
        & (whole["Band"].isin(bands))
        & (whole["Metric"] == METRIC)
        & (whole["BandScheme"] == BAND_SCHEME)
        & (whole["Phase"].isin(["PRE", "POST"]))
    ].copy()

    during = during[
        (during["Condition"].isin(conditions))
        & (during["Network"].isin(NETWORKS))
        & (during["Band"].isin(bands))
        & (during["Metric"] == METRIC)
        & (during["BandScheme"] == BAND_SCHEME)
        & (during["Phase"] == "DURING")
    ].copy()

    if whole.empty or during.empty:
        return pd.DataFrame()

    whole_phase = (
        whole.groupby(["Subject", "Condition", "Network", "Band", "Phase"], as_index=False)["Value"]
        .mean()
        .rename(columns={"Value": "MeanSourcePower"})
    )
    whole_wide = whole_phase.pivot_table(
        index=["Subject", "Condition", "Network", "Band"],
        columns="Phase",
        values="MeanSourcePower",
        aggfunc="mean",
    ).reset_index()
    for phase in ["PRE", "POST"]:
        if phase not in whole_wide.columns:
            whole_wide[phase] = np.nan

    during_phase = (
        during.groupby(
            ["Subject", "Condition", "Network", "Band", "WindowIndex"],
            as_index=False,
        )["Value"]
        .mean()
        .rename(columns={"Value": "MeanSourcePower"})
    )
    window_indices = sorted(during_phase["WindowIndex"].dropna().unique().astype(int).tolist())
    if MAX_DURING_WINDOWS is not None:
        window_indices = [w for w in window_indices if w < MAX_DURING_WINDOWS]

    during_wide = during_phase.pivot_table(
        index=["Subject", "Condition", "Network", "Band"],
        columns="WindowIndex",
        values="MeanSourcePower",
        aggfunc="mean",
    ).reset_index()
    during_wide = during_wide.rename(columns={idx: f"D{idx + 1}" for idx in window_indices})
    during_cols = [f"D{idx + 1}" for idx in window_indices]

    out = whole_wide.merge(
        during_wide,
        on=["Subject", "Condition", "Network", "Band"],
        how="inner",
    )
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
    out["Metric"] = METRIC
    out["BandScheme"] = BAND_SCHEME
    out["FeatureSource"] = feature_source

    return out


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


def build_network_band_table(bands, standard_whole, standard_during):
    standard = build_trajectory_from_loaded_tables(
        whole=standard_whole,
        during=standard_during,
        bands=bands,
        conditions=["Cond1", "Cond2", "Cond4", "Cond5"],
        feature_source="StandardSourcePipeline",
    )

    zapline_whole = _load_zapline_source_csv(ZAPLINE_PRE_POST_FEATURES_CSV, require_window=False)
    zapline_during = _load_zapline_source_csv(ZAPLINE_DURING_5MIN_FEATURES_CSV, require_window=True)
    cond3 = build_trajectory_from_loaded_tables(
        whole=zapline_whole,
        during=zapline_during,
        bands=bands,
        conditions=["Cond3"],
        feature_source="ZapLine40_Cond3",
    )

    out = pd.concat([standard, cond3], ignore_index=True, sort=False)
    out = out[out["Condition"].isin(FIGURE_CONDITIONS)].copy()
    out["Network"] = pd.Categorical(out["Network"], NETWORKS, ordered=True)
    out["Band"] = pd.Categorical(out["Band"], bands, ordered=True)

    return out.sort_values(["Network", "Band", "Condition", "Subject"]).reset_index(drop=True)


def plot_network_band_trajectories(df, bands):
    fig, axes = plt.subplots(
        len(NETWORKS),
        len(bands),
        figsize=(4.0 * len(bands), 2.35 * len(NETWORKS)),
        sharex=True,
        squeeze=False,
        gridspec_kw={"wspace": 0.18, "hspace": 0.35},
    )

    for row_i, network in enumerate(NETWORKS):
        for col_i, band in enumerate(bands):
            ax = axes[row_i, col_i]
            sub_panel = df[(df["Network"] == network) & (df["Band"] == band)].copy()

            for cond in FIGURE_CONDITIONS:
                sub = sub_panel[sub_panel["Condition"] == cond].copy()
                if sub.empty:
                    continue

                present_cols = [col for col in TIME_COLS if col in sub.columns]
                present_labels = [lab for lab, col in zip(TIME_LABELS, TIME_COLS) if col in present_cols]
                x = np.arange(len(present_cols))

                ax.errorbar(
                    x,
                    [sub[col].mean() for col in present_cols],
                    yerr=[sem(sub[col]) for col in present_cols],
                    marker="o",
                    markersize=3.8,
                    linewidth=1.6,
                    capsize=2.5,
                    color=CONDITION_COLORS[cond],
                    label=CONDITION_LABELS[cond],
                )

            ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
            ax.grid(True, axis="y", alpha=0.22)

            if row_i == 0:
                ax.set_title(band, fontsize=11, pad=8)
            if col_i == 0:
                ax.set_ylabel(f"{network}\nchange from PRE", fontsize=9)
            if row_i == len(NETWORKS) - 1:
                ax.set_xticks(np.arange(len(TIME_LABELS)))
                ax.set_xticklabels(TIME_LABELS, rotation=30, ha="right")
            else:
                ax.tick_params(labelbottom=False)

            ax.tick_params(axis="both", labelsize=8)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(FIGURE_CONDITIONS),
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.suptitle(
        "Source network relative-power trajectories by frequency band",
        y=1.012,
        fontsize=14,
    )
    fig.text(
        0.5,
        0.006,
        "Relative power is expressed as change from PRE. 40 Hz uses ZapLine source features; other conditions use the cleaned/interpolated source pipeline.",
        ha="center",
        va="bottom",
        fontsize=8,
    )

    return fig


print("Loading standard cleaned/interpolated source tables once...")
standard_whole = load_source_whole()
standard_during = load_source_during_5min()

bands = available_bands(standard_whole)
print("Bands:", bands)
print("Networks:", NETWORKS)

print("Building all network x band trajectory features...")
features = build_network_band_table(bands, standard_whole, standard_during)
features.to_csv(OUT_TABLES / "figureS5A_all_network_relative_power_trajectory_features.csv", index=False)

print("\nN by network, band, condition:")
print(
    features.groupby(["Network", "Band", "Condition"])["Subject"]
    .nunique()
    .reset_index(name="N")
    .to_string(index=False)
)

fig = plot_network_band_trajectories(features, bands)
savefig(fig, OUT_DIR / "FigureS5A_all_network_relative_power_trajectories.png")

print("\nDONE: Figure S5A all-network relative-power trajectory figure.")
print(f"Output folder:\n{OUT_DIR}")
