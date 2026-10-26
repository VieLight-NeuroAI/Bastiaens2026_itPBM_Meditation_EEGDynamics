# ============================================================
# Fig08_low_frequency_acute_slowing.py
#
# Manuscript Figure 8 (Results 4.6):
#   Acute source-level slowing-like spectral changes were most prominent
#   in the 4 Hz and 10 Hz conditions.
#
# Panels:
#   A. 4 Hz source relative delta/theta/alpha trajectories
#   B. 10 Hz source relative delta/theta/alpha trajectories
#   C. Acute slowing index across conditions
#   D. Mean during-stimulation band changes (heatmap)
#
# Statistics:
#   1. Within-condition band changes vs 0 (sign-flip), FDR within each
#      feature across the condition x band tests (stars in A, B, D)
#   2. Acute slowing index vs 0 within each condition, FDR across the five
#      conditions (Methods 3.7, family vii)
#   3. Acute slowing index active vs sham, FDR across the four active
#      conditions (family viii)
#
# Cond3 / 40 Hz uses the ZapLine-cleaned source features.
#
# Output files are prefixed "figure8_" / "Figure8_".
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp

from pbm_paper_common import (
    OUT_TABLES, OUT_STATS, OUT_FIGS,
    CONDITION_LABELS, NETWORK_ORDER, BAND_SCHEME, MAX_DURING_WINDOWS, LATE_DURING_WINDOWS,
    clean_subject_id, clean_condition, is_schaefer_brain_roi, roi_to_network,
    build_source_trajectory_table,
    paired_condition_vs_sham, fdr_bh, sign_flip_p, cohen_d_paired, sem, savefig
)

BANDS = ["Delta", "Theta", "Alpha", "Beta"]
TRAJECTORY_BANDS = ["Delta", "Theta", "Alpha"]
METRIC = "relative"
NETWORK = "all_rois"
SHAM = "Cond5"
FIGURE_CONDITIONS = ["Cond1", "Cond2", "Cond3", "Cond4", "Cond5"]
ACTIVE_CONDITIONS = ["Cond1", "Cond2", "Cond3", "Cond4"]
ZAPLINE_CONDITION = "Cond3"
REQUIRE_ZAPLINE_FOR_COND3 = True

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

OUT_DIR = OUT_FIGS / "figure8_acute_stats"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TIME_COLS = ["PRE_change", "D1_change", "D2_change", "D3_change", "D4_change", "POST_change"]
TIME_LABELS = ["PRE", "Dur5", "Dur10", "Dur15", "Dur20", "Post"]

BAND_COLORS = {
    "Delta": "#D95F02",
    "Theta": "#E6AB02",
    "Alpha": "#1F77B4",
    "Beta": "#7570B3",
}

CONDITION_COLORS = {
    "Cond1": "#4C78A8",
    "Cond2": "#9467BD",
    "Cond3": "#D95F02",
    "Cond4": "#D62728",
    "Cond5": "#6F7F8F",
}


def one_sample_stats(values, label):
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna()

    if len(values) < 3:
        return {
            **label,
            "N": len(values),
            "Mean": values.mean() if len(values) else np.nan,
            "SEM": values.sem() if len(values) > 1 else np.nan,
            "T": np.nan,
            "P_ttest_vs_zero": np.nan,
            "P_signflip_vs_zero": np.nan,
            "Cohen_dz_vs_zero": np.nan,
        }

    t, p = ttest_1samp(values, 0.0)

    return {
        **label,
        "N": len(values),
        "Mean": values.mean(),
        "SEM": values.sem(),
        "T": t,
        "P_ttest_vs_zero": p,
        "P_signflip_vs_zero": sign_flip_p(values, n_perm=10000, seed=42),
        "Cohen_dz_vs_zero": cohen_d_paired(values),
    }


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
    band,
    metric=METRIC,
    network=NETWORK,
    band_scheme=BAND_SCHEME,
):
    whole = _load_zapline_source_csv(
        ZAPLINE_PRE_POST_FEATURES_CSV,
        require_window=False,
    )
    during = _load_zapline_source_csv(
        ZAPLINE_DURING_5MIN_FEATURES_CSV,
        require_window=True,
    )

    whole = whole[
        (whole["Condition"] == ZAPLINE_CONDITION)
        & (whole["Phase"].isin(["PRE", "POST"]))
        & (whole["Band"] == band)
        & (whole["Metric"] == metric)
        & (whole["BandScheme"] == band_scheme)
    ].copy()

    during = during[
        (during["Condition"] == ZAPLINE_CONDITION)
        & (during["Phase"] == "DURING")
        & (during["Band"] == band)
        & (during["Metric"] == metric)
        & (during["BandScheme"] == band_scheme)
    ].copy()

    if network != "all_rois":
        whole = whole[whole["Network"] == network].copy()
        during = during[during["Network"] == network].copy()

    if whole.empty or during.empty:
        if REQUIRE_ZAPLINE_FOR_COND3:
            raise RuntimeError(
                f"No ZapLine Cond3 rows were available for {band} after filtering. "
                "Check the ZapLine paths, metric, network, and band scheme."
            )
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

    window_col_map = {idx: f"D{idx + 1}" for idx in window_indices}
    during_wide = during_wide.rename(columns=window_col_map)
    during_cols = [f"D{idx + 1}" for idx in window_indices]

    out = whole_wide.merge(during_wide, on=["Subject", "Condition"], how="inner")

    if out.empty:
        if REQUIRE_ZAPLINE_FOR_COND3:
            raise RuntimeError(f"ZapLine Cond3 PRE/POST and DURING rows did not overlap by subject for {band}.")
        return out

    out["PRE_change"] = 0.0

    for c in during_cols:
        if c in out.columns:
            out[f"{c}_change"] = out[c] - out["PRE"]

    out["POST_change"] = out["POST"] - out["PRE"]

    all_during_change_cols = [f"{c}_change" for c in during_cols if f"{c}_change" in out.columns]
    late_cols = [f"{c}_change" for c in LATE_DURING_WINDOWS if f"{c}_change" in out.columns]

    out["DURING_mean_change"] = out[all_during_change_cols].mean(axis=1)
    out["LateDuringMean_change"] = out[late_cols].mean(axis=1)
    out["DuringSuppression"] = -out["LateDuringMean_change"]
    out["DURING_min_change"] = out[all_during_change_cols].min(axis=1)
    out["DURING_max_change"] = out[all_during_change_cols].max(axis=1)
    out["ReboundIndex"] = out["POST_change"] - out["LateDuringMean_change"]

    if "D4_change" in out.columns:
        out["RecoveryFromD4"] = out["POST_change"] - out["D4_change"]
    else:
        out["RecoveryFromD4"] = np.nan

    out["Band"] = band
    out["Metric"] = metric
    out["Network"] = network
    out["BandScheme"] = band_scheme
    out["FeatureSource"] = "ZapLine40_Cond3"

    return out


def replace_cond3_with_zapline(tab, band):
    zap_cond3 = build_zapline_cond3_trajectory_table(
        band=band,
        metric=METRIC,
        network=NETWORK,
    )

    if zap_cond3.empty and REQUIRE_ZAPLINE_FOR_COND3:
        raise RuntimeError(f"ZapLine Cond3 feature table is empty for {band}.")

    standard = tab[tab["Condition"] != ZAPLINE_CONDITION].copy()
    standard["FeatureSource"] = standard.get("FeatureSource", "StandardSourcePipeline")
    combined = pd.concat([standard, zap_cond3], ignore_index=True, sort=False)

    print(f"\nFeature source by condition for {band}:")
    print(combined.groupby(["Condition", "FeatureSource"])["Subject"].nunique())

    return combined


# ============================================================
# BUILD BAND FEATURE TABLE
# ============================================================

tables = []

for band in BANDS:
    tab = build_source_trajectory_table(
        band=band,
        metric=METRIC,
        network=NETWORK,
    )

    if tab.empty:
        print(f"[SKIP] {band}")
        continue

    tab = replace_cond3_with_zapline(tab, band)

    keep_cols = [
        "Subject", "Condition",
        "DURING_mean_change", "LateDuringMean_change",
        "POST_change", "D1_change", "D2_change", "D3_change", "D4_change",
    ]
    keep_cols = [c for c in keep_cols if c in tab.columns]

    tab = tab[keep_cols].copy()
    tab["Band"] = band
    tables.append(tab)

df = pd.concat(tables, ignore_index=True)
df = df[df["Condition"].isin(FIGURE_CONDITIONS)].copy()
df.to_csv(OUT_TABLES / "figure8_band_acute_features_allrois.csv", index=False)


# ============================================================
# WITHIN-CONDITION BAND TESTS
# ============================================================

rows = []

features_to_test = [
    "DURING_mean_change",
    "LateDuringMean_change",
    "D1_change",
    "D2_change",
    "D3_change",
    "D4_change",
    "POST_change",
]

features_to_test = [f for f in features_to_test if f in df.columns]

for band in BANDS:
    for cond in FIGURE_CONDITIONS:
        sub = df[(df["Band"] == band) & (df["Condition"] == cond)].copy()

        for feat in features_to_test:
            rows.append(
                one_sample_stats(
                    sub[feat],
                    {
                        "Band": band,
                        "Condition": cond,
                        "ConditionLabel": CONDITION_LABELS.get(cond, cond),
                        "Feature": feat,
                    }
                )
            )

within_stats = pd.DataFrame(rows)

# FDR separately for each feature across condition x band tests
within_stats["P_FDR_within_feature"] = np.nan
within_stats["Significant_FDR_within_feature"] = False

for feat, idx in within_stats.groupby("Feature").groups.items():
    idx = list(idx)
    adj, reject = fdr_bh(within_stats.loc[idx, "P_signflip_vs_zero"].values, alpha=0.05)
    within_stats.loc[idx, "P_FDR_within_feature"] = adj
    within_stats.loc[idx, "Significant_FDR_within_feature"] = reject

within_stats.to_csv(OUT_STATS / "figure8_within_condition_band_tests_vs_zero.csv", index=False)


# ============================================================
# ACUTE SLOWING INDEX
# ============================================================

wide = df.pivot_table(
    index=["Subject", "Condition"],
    columns="Band",
    values="DURING_mean_change",
    aggfunc="mean",
).reset_index()

for b in BANDS:
    if b not in wide.columns:
        wide[b] = np.nan

wide["DeltaThetaIncrease"] = wide[["Delta", "Theta"]].mean(axis=1)
wide["AlphaSuppression"] = -wide["Alpha"]
wide["AcuteSlowingIndex"] = wide["DeltaThetaIncrease"] + wide["AlphaSuppression"]

wide.to_csv(OUT_TABLES / "figure8_acute_slowing_index_allrois_subject_condition.csv", index=False)

acute_rows = []

for cond in FIGURE_CONDITIONS:
    sub = wide[wide["Condition"] == cond].copy()

    for feat in ["DeltaThetaIncrease", "AlphaSuppression", "AcuteSlowingIndex"]:
        acute_rows.append(
            one_sample_stats(
                sub[feat],
                {
                    "Condition": cond,
                    "ConditionLabel": CONDITION_LABELS.get(cond, cond),
                    "Feature": feat,
                }
            )
        )

acute_within = pd.DataFrame(acute_rows)

for feat, idx in acute_within.groupby("Feature").groups.items():
    idx = list(idx)
    adj, reject = fdr_bh(acute_within.loc[idx, "P_signflip_vs_zero"].values, alpha=0.05)
    acute_within.loc[idx, "P_FDR_within_feature"] = adj
    acute_within.loc[idx, "Significant_FDR_within_feature"] = reject

acute_within.to_csv(OUT_STATS / "figure8_acute_indices_within_condition_vs_zero.csv", index=False)

# Paired vs sham
vs_sham_rows = []
for feat in ["DeltaThetaIncrease", "AlphaSuppression", "AcuteSlowingIndex"]:
    for cond in ACTIVE_CONDITIONS:
        res = paired_condition_vs_sham(
            wide,
            value_col=feat,
            condition=cond,
            sham=SHAM,
        )
        res["Feature"] = feat
        vs_sham_rows.append(res)

vs_sham = pd.DataFrame(vs_sham_rows)

for feat, idx in vs_sham.groupby("Feature").groups.items():
    idx = list(idx)
    adj, reject = fdr_bh(vs_sham.loc[idx, "P_signflip"].values, alpha=0.05)
    vs_sham.loc[idx, "P_FDR_within_feature"] = adj
    vs_sham.loc[idx, "Significant_FDR_within_feature"] = reject

vs_sham.to_csv(OUT_STATS / "figure8_acute_indices_condition_vs_sham.csv", index=False)


# ============================================================
# WORKING 3-PANEL SUMMARY (built and closed, not saved;
# Figure 8 is drawn in the next section)
# ============================================================

fig, axes = plt.subplots(
    1, 3,
    figsize=(20.5, 5.6),
    gridspec_kw={"width_ratios": [1.15, 1.15, 1.25], "wspace": 0.48},
)

# Panel A: band heatmap with within-condition stars
ax = axes[0]

heat = df.pivot_table(
    index="Condition",
    columns="Band",
    values="DURING_mean_change",
    aggfunc="mean",
).reindex(FIGURE_CONDITIONS)[BANDS]

v = np.nanpercentile(np.abs(heat.values), 98)
im = ax.imshow(heat.values, aspect="auto", cmap="coolwarm", vmin=-v, vmax=v)

ax.set_xticks(np.arange(len(BANDS)))
ax.set_xticklabels(BANDS)
ax.set_yticks(np.arange(len(FIGURE_CONDITIONS)))
ax.set_yticklabels([f"{c} ({CONDITION_LABELS[c]})" for c in FIGURE_CONDITIONS])
ax.set_title("A. DURING − PRE band changes\n* = within-condition FDR q < .05", pad=10)

for i, cond in enumerate(FIGURE_CONDITIONS):
    for j, band in enumerate(BANDS):
        val = heat.loc[cond, band]
        srow = within_stats[
            (within_stats["Condition"] == cond)
            & (within_stats["Band"] == band)
            & (within_stats["Feature"] == "DURING_mean_change")
        ]

        star = ""
        if not srow.empty and bool(srow.iloc[0]["Significant_FDR_within_feature"]):
            star = "*"

        ax.text(j, i, f"{val:.2f}{star}", ha="center", va="center", fontsize=8)

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("DURING − PRE")

# Panel B: acute index by condition
ax = axes[1]
x = np.arange(len(FIGURE_CONDITIONS))

for subject, s in wide.groupby("Subject"):
    y = [
        s.loc[s["Condition"] == c, "AcuteSlowingIndex"].mean()
        if (s["Condition"] == c).any()
        else np.nan
        for c in FIGURE_CONDITIONS
    ]
    ax.plot(x, y, color="gray", alpha=0.18, linewidth=0.8)

means = [wide.loc[wide["Condition"] == c, "AcuteSlowingIndex"].mean() for c in FIGURE_CONDITIONS]
sems = [sem(wide.loc[wide["Condition"] == c, "AcuteSlowingIndex"]) for c in FIGURE_CONDITIONS]

ax.errorbar(x, means, yerr=sems, marker="o", linewidth=3, capsize=5, color="black")

for i, cond in enumerate(FIGURE_CONDITIONS):
    srow = acute_within[
        (acute_within["Condition"] == cond)
        & (acute_within["Feature"] == "AcuteSlowingIndex")
    ]
    if not srow.empty:
        q = srow.iloc[0]["P_FDR_within_feature"]
        star = "*" if bool(srow.iloc[0]["Significant_FDR_within_feature"]) else ""
        ax.text(i, means[i] + sems[i] + 0.015, f"q={q:.3f}{star}", ha="center", fontsize=8)

ax.axhline(0, color="black", linestyle="--", linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels([CONDITION_LABELS[c] for c in FIGURE_CONDITIONS])
ax.set_ylabel("Acute slowing index")
ax.set_title("B. Acute slowing index vs zero", pad=10)
ax.grid(True, axis="y", alpha=0.25)

# Panel C: condition vs sham for acute index
ax = axes[2]

plot_vs = vs_sham[vs_sham["Feature"] == "AcuteSlowingIndex"].copy()
plot_vs = plot_vs.set_index("Condition").reindex(ACTIVE_CONDITIONS).reset_index()

x2 = np.arange(len(plot_vs))

ax.bar(x2, plot_vs["MeanDiff"], alpha=0.85)
ax.axhline(0, color="black", linestyle="--", linewidth=1)

for i, row in plot_vs.iterrows():
    if pd.notna(row["P_signflip"]):
        star = "*" if row["Significant_FDR_within_feature"] else ""
        ax.text(
            i,
            row["MeanDiff"],
            f"p={row['P_signflip']:.3f}\nq={row['P_FDR_within_feature']:.3f}{star}",
            ha="center",
            va="bottom" if row["MeanDiff"] >= 0 else "top",
            fontsize=8,
        )

ax.set_xticks(x2)
ax.set_xticklabels([CONDITION_LABELS.get(c, c) for c in plot_vs["Condition"]])
ax.set_ylabel("Acute slowing index: condition − sham")
ax.set_title("C. Acute slowing specificity vs sham", pad=10)
ax.grid(True, axis="y", alpha=0.25)

plt.close(fig)


# ============================================================
# FIGURE 8: ACUTE LOW-FREQUENCY SPECTRAL SLOWING
# ============================================================

def condition_label(cond):
    return CONDITION_LABELS.get(cond, cond).replace("Hz", " Hz")


def trajectory_star(condition, band, feature):
    srow = within_stats[
        (within_stats["Condition"] == condition)
        & (within_stats["Band"] == band)
        & (within_stats["Feature"] == feature)
    ]
    if srow.empty:
        return False
    return bool(srow.iloc[0]["Significant_FDR_within_feature"])


def plot_condition_trajectory(ax, condition, title):
    present_cols = [col for col in TIME_COLS if col in df.columns]
    present_labels = [lab for lab, col in zip(TIME_LABELS, TIME_COLS) if col in present_cols]
    x = np.arange(len(present_cols))

    for band in TRAJECTORY_BANDS:
        sub = df[(df["Condition"] == condition) & (df["Band"] == band)].copy()
        if sub.empty:
            continue

        means = np.array([sub[col].mean() for col in present_cols], dtype=float)
        errors = np.array([sem(sub[col]) for col in present_cols], dtype=float)

        ax.errorbar(
            x,
            means,
            yerr=errors,
            marker="o",
            linewidth=2.6,
            capsize=4,
            color=BAND_COLORS[band],
            label=band,
        )

        for i, col in enumerate(present_cols):
            if col == "PRE_change" or not np.isfinite(means[i]):
                continue
            if trajectory_star(condition, band, col):
                err = errors[i] if np.isfinite(errors[i]) else 0.0
                sign = 1 if means[i] >= 0 else -1
                ax.text(
                    x[i],
                    means[i] + sign * (err + 0.010),
                    "*",
                    color=BAND_COLORS[band],
                    ha="center",
                    va="bottom" if sign > 0 else "top",
                    fontsize=12,
                    fontweight="bold",
                )

    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(present_labels)
    ax.set_ylabel("Relative source power\nchange from PRE")
    ax.set_title(title, pad=8)
    ax.grid(True, axis="y", alpha=0.25)
    ax.margins(y=0.22)


def plot_acute_slowing_index(ax):
    x = np.arange(len(FIGURE_CONDITIONS))

    for _, s in wide.groupby("Subject"):
        y = [
            s.loc[s["Condition"] == cond, "AcuteSlowingIndex"].mean()
            if (s["Condition"] == cond).any()
            else np.nan
            for cond in FIGURE_CONDITIONS
        ]
        ax.plot(x, y, color="gray", alpha=0.14, linewidth=0.8, zorder=1)

    means = [
        wide.loc[wide["Condition"] == cond, "AcuteSlowingIndex"].mean()
        for cond in FIGURE_CONDITIONS
    ]
    errors = [
        sem(wide.loc[wide["Condition"] == cond, "AcuteSlowingIndex"])
        for cond in FIGURE_CONDITIONS
    ]

    for i, cond in enumerate(FIGURE_CONDITIONS):
        vals = wide.loc[wide["Condition"] == cond, "AcuteSlowingIndex"].dropna()
        ax.scatter(
            np.full(len(vals), x[i]),
            vals,
            s=28,
            color=CONDITION_COLORS[cond],
            alpha=0.35,
            edgecolor="white",
            linewidth=0.4,
            zorder=2,
        )

    ax.errorbar(
        x,
        means,
        yerr=errors,
        marker="o",
        linewidth=3,
        capsize=5,
        color="black",
        label="Mean +/- SEM",
        zorder=3,
    )

    for i, cond in enumerate(FIGURE_CONDITIONS):
        srow = acute_within[
            (acute_within["Condition"] == cond)
            & (acute_within["Feature"] == "AcuteSlowingIndex")
        ]
        if srow.empty or not np.isfinite(means[i]):
            continue
        q = srow.iloc[0]["P_FDR_within_feature"]
        star = "*" if bool(srow.iloc[0]["Significant_FDR_within_feature"]) else ""
        err = errors[i] if np.isfinite(errors[i]) else 0.0
        ax.text(
            x[i],
            means[i] + err + 0.020,
            f"q={q:.3f}{star}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([condition_label(cond) for cond in FIGURE_CONDITIONS])
    ax.set_ylabel("Acute slowing index")
    ax.set_title("C. Acute slowing index across conditions", pad=8)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    ax.margins(y=0.25)


def plot_during_band_heatmap(ax):
    heat = df.pivot_table(
        index="Condition",
        columns="Band",
        values="DURING_mean_change",
        aggfunc="mean",
    ).reindex(FIGURE_CONDITIONS)[BANDS]

    vmax = np.nanpercentile(np.abs(heat.values), 98)
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1
    im = ax.imshow(heat.values, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)

    ax.set_xticks(np.arange(len(BANDS)))
    ax.set_xticklabels(BANDS)
    ax.set_yticks(np.arange(len(FIGURE_CONDITIONS)))
    ax.set_yticklabels([condition_label(cond) for cond in FIGURE_CONDITIONS])
    ax.set_title("D. Mean during-stimulation band changes", pad=8)

    for i, cond in enumerate(FIGURE_CONDITIONS):
        for j, band in enumerate(BANDS):
            val = heat.loc[cond, band]
            if not np.isfinite(val):
                continue

            srow = within_stats[
                (within_stats["Condition"] == cond)
                & (within_stats["Band"] == band)
                & (within_stats["Feature"] == "DURING_mean_change")
            ]
            star = ""
            if not srow.empty and bool(srow.iloc[0]["Significant_FDR_within_feature"]):
                star = "*"

            display_val = 0.0 if abs(val) < 0.005 else val
            ax.text(
                j,
                i,
                f"{display_val:.2f}{star}",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold" if star else "normal",
            )

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("mean(Dur5_10_15_20) - PRE")


fig, axes = plt.subplots(
    2,
    2,
    figsize=(16.8, 9.4),
    gridspec_kw={"wspace": 0.28, "hspace": 0.36},
)

plot_condition_trajectory(
    axes[0, 0],
    "Cond1",
    "A. 4 Hz source spectral trajectories",
)
plot_condition_trajectory(
    axes[0, 1],
    "Cond2",
    "B. 10 Hz source spectral trajectories",
)
plot_acute_slowing_index(axes[1, 0])
plot_during_band_heatmap(axes[1, 1])

axes[0, 0].legend(frameon=False, fontsize=8)

fig.text(
    0.5,
    0.012,
    (
        "Asterisks mark within-condition FDR q < .05 for the corresponding band/timepoint or band summary. "
        "Acute slowing index = mean delta/theta increase plus alpha suppression."
    ),
    ha="center",
    va="bottom",
    fontsize=8,
)

savefig(fig, OUT_DIR / "Figure8_acute_slowing_main.png")

print("\nDONE: Figure 8 acute stats.")
