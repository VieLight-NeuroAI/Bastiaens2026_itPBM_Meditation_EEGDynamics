# ============================================================
# pbm_paper_common.py
#
# Shared paths, loading, feature construction, stats, and plotting helpers.
# ============================================================

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, ttest_rel, ttest_1samp, ttest_ind, mannwhitneyu


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "Data"

ROOT = PROJECT_ROOT / "outputs" / "PBM_paper_analysis"
ROOT.mkdir(parents=True, exist_ok=True)

SOURCE_FEATURE_DIR = Path(
    DATA_ROOT / "source_results" / "source_band_power_standard"
)

SOURCE_DURING_5MIN_DIR = Path(
    DATA_ROOT / "source_results" / "source_band_power_standard"
)

SENSOR_BAND_POWER_CANDIDATES = [
    DATA_ROOT / "sensor" / "band_power_theta4_8_alpha8_13_all_subjects_conditions_long.csv",
    DATA_ROOT / "sensor" / "band_power_all_schemes_all_subjects_conditions_long.csv",
    DATA_ROOT / "sensor_results" / "band_power_theta4_8_alpha8_13_all_subjects_conditions_long.csv",
    DATA_ROOT / "sensor_results" / "band_power_all_schemes_all_subjects_conditions_long.csv",
]

SENSOR_BAND_POWER_CSV = next(
    (path for path in SENSOR_BAND_POWER_CANDIDATES if path.exists()),
    SENSOR_BAND_POWER_CANDIDATES[0],
)

SCORE_FILE = Path(
    DATA_ROOT / "behavior" / "meditation_scores_long.csv"
)

OUT_TABLES = ROOT / "tables"
OUT_STATS = ROOT / "stats"
OUT_FIGS = ROOT / "figures"
OUT_PANELS = ROOT / "paper_panels"

for p in [OUT_TABLES, OUT_STATS, OUT_FIGS, OUT_PANELS]:
    p.mkdir(parents=True, exist_ok=True)


# ============================================================
# GLOBAL SETTINGS
# ============================================================

CONDITIONS = ["Cond1", "Cond2", "Cond3", "Cond4", "Cond5", "Cond6"]

CONDITION_LABELS = {
    "Cond1": "4Hz",
    "Cond2": "10Hz",
    "Cond3": "40Hz",
    "Cond4": "120Hz",
    "Cond5": "Sham",
    "Cond6": "Control",
}

CONDITION_ORDER_LABELS = [f"{c}\n{CONDITION_LABELS[c]}" for c in CONDITIONS]

BAND_SCHEME = "theta4_8_alpha8_13"
MAX_DURING_WINDOWS = 4
LATE_DURING_WINDOWS = ["D2", "D3", "D4"]

NETWORK_ORDER = [
    "Vis",
    "SomMot",
    "DorsAttn",
    "SalVentAttn",
    "Limbic",
    "Cont",
    "Default",
    "TempPar",
    "Other",
]


# ============================================================
# CLEANING HELPERS
# ============================================================

def clean_subject_id(x):
    x = str(x).strip()
    if x.startswith("MED_PBM_"):
        n = x.replace("MED_PBM_", "")
        if n.isdigit():
            return f"MED_PBM_{int(n):03d}"
        return x
    try:
        return f"MED_PBM_{int(float(x)):03d}"
    except Exception:
        return x


def clean_condition(x):
    x = str(x).strip()
    mapping = {
        "4Hz": "Cond1", "4 Hz": "Cond1",
        "10Hz": "Cond2", "10 Hz": "Cond2",
        "40Hz": "Cond3", "40 Hz": "Cond3",
        "120Hz": "Cond4", "120 Hz": "Cond4",
        "Sham": "Cond5",
        "Control": "Cond6",
    }
    return mapping.get(x, x)


def infer_metadata_from_filename(path):
    name = Path(path).name
    subj = re.search(r"MED_PBM_(\d+)", name)
    cond = re.search(r"(Cond\d+)", name)
    phase = re.search(r"_(PRE|DURING|POST)_", name)

    return (
        clean_subject_id(subj.group(1)) if subj else None,
        cond.group(1) if cond else None,
        phase.group(1) if phase else None,
    )


def is_schaefer_brain_roi(roi):
    name = str(roi).lower()
    bad_terms = ["unknown", "medial", "wall", "background", "corpuscallosum", "corpus_callosum"]
    return not any(t in name for t in bad_terms)


def roi_to_network(roi):
    """
    Collapse Schaefer 7- or 17-network ROI names into broad networks.

    Examples handled:
        17Networks_LH_ContA_Cingm_1-lh -> Cont
        17Networks_RH_DefaultB_PFCd_1-rh -> Default
        17Networks_LH_VisCent_ExStr_1-lh -> Vis
        7Networks_LH_Cont_PFCl_1-lh -> Cont
    """
    roi = str(roi).strip()
    roi_l = roi.lower()

    # Try to parse the network token directly.
    # Example: 17Networks_LH_ContA_Cingm_1-lh
    # parts[2] = ContA
    parts = roi.split("_")
    token = parts[2] if len(parts) >= 3 else roi

    token_l = token.lower()

    if token_l.startswith("vis"):
        return "Vis"

    if token_l.startswith("sommot"):
        return "SomMot"

    if token_l.startswith("dorsattn"):
        return "DorsAttn"

    if token_l.startswith("salventattn") or token_l.startswith("ventattn"):
        return "SalVentAttn"

    if token_l.startswith("limbic"):
        return "Limbic"

    if token_l.startswith("cont") or token_l.startswith("control"):
        return "Cont"

    if token_l.startswith("default"):
        return "Default"

    if token_l.startswith("temppar"):
        return "TempPar"

    # Fallback matching
    if "vis" in roi_l:
        return "Vis"
    if "sommot" in roi_l or "som_mot" in roi_l:
        return "SomMot"
    if "dorsattn" in roi_l or "dors_attn" in roi_l:
        return "DorsAttn"
    if "salventattn" in roi_l or "ventattn" in roi_l:
        return "SalVentAttn"
    if "limbic" in roi_l:
        return "Limbic"
    if "cont" in roi_l or "control" in roi_l:
        return "Cont"
    if "default" in roi_l:
        return "Default"
    if "temppar" in roi_l or "temp_par" in roi_l:
        return "TempPar"

    return "Other"


def safe_log10(x):
    x = pd.to_numeric(x, errors="coerce")
    x = x.where(x > 0, np.nan)
    return np.log10(x)


# ============================================================
# LOADERS
# ============================================================

def load_score_table(score_col="MEQ_Total"):
    scores = pd.read_csv(SCORE_FILE)

    for col in scores.columns:
        if scores[col].dtype == object:
            scores[col] = scores[col].astype(str).str.strip()

    if "Subject" not in scores.columns:
        raise ValueError(f"Score file needs Subject column. Found: {scores.columns.tolist()}")
    if "Condition" not in scores.columns:
        raise ValueError(f"Score file needs Condition column. Found: {scores.columns.tolist()}")
    if score_col not in scores.columns:
        raise ValueError(f"{score_col} not found. Found: {scores.columns.tolist()}")

    scores["Subject"] = scores["Subject"].apply(clean_subject_id)
    scores["Condition"] = scores["Condition"].apply(clean_condition)
    scores[score_col] = pd.to_numeric(scores[score_col], errors="coerce")

    scores = scores[["Subject", "Condition", score_col]].dropna()
    scores = scores.groupby(["Subject", "Condition"], as_index=False)[score_col].mean()
    scores = scores.rename(columns={score_col: "MeditationScore"})

    return scores


def load_source_csvs(feature_dir, pattern):
    feature_path = Path(feature_dir)
    if feature_path.is_file():
        files = [feature_path]
    else:
        files = sorted(feature_path.glob(pattern))
        if not files and "5min" in pattern:
            files = sorted(feature_path.glob("*during_5min_source_roi_band_power_all_subjects_conditions_long*.csv"))
        if not files:
            files = sorted(feature_path.glob("*source_roi_band_power_all_subjects_conditions_long*.csv"))
    if not files:
        raise FileNotFoundError(f"No files found in {feature_dir} with pattern {pattern}")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        subj, cond, phase = infer_metadata_from_filename(f)

        if "Subject" not in df.columns or df["Subject"].isna().all():
            df["Subject"] = subj
        if "Condition" not in df.columns or df["Condition"].isna().all():
            df["Condition"] = cond
        if "Phase" not in df.columns or df["Phase"].isna().all():
            df["Phase"] = phase

        df["SourceFile"] = f.name
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    required = {"Subject", "Condition", "Phase", "ROI", "Band", "Metric", "Value", "BandScheme"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col in ["Subject", "Condition", "Phase", "ROI", "Band", "Metric", "BandScheme"]:
        df[col] = df[col].astype(str).str.strip()

    df["Subject"] = df["Subject"].apply(clean_subject_id)
    df["Condition"] = df["Condition"].apply(clean_condition)
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df[df["ROI"].apply(is_schaefer_brain_roi)].copy()
    df["Network"] = df["ROI"].apply(roi_to_network)

    return df


def load_source_whole():
    df = load_source_csvs(SOURCE_FEATURE_DIR, "*source_roi_band_power_long*.csv")
    return df[df["Phase"].isin(["PRE", "POST"])].copy()


def load_source_during_5min():
    df = load_source_csvs(SOURCE_DURING_5MIN_DIR, "*source_roi_band_power_5min_long*.csv")

    for col in ["WindowIndex", "WindowStartSec", "WindowEndSec"]:
        if col not in df.columns:
            raise ValueError(f"Missing DURING window column: {col}")

    df["WindowIndex"] = pd.to_numeric(df["WindowIndex"], errors="coerce")
    df["WindowStartSec"] = pd.to_numeric(df["WindowStartSec"], errors="coerce")
    df["WindowEndSec"] = pd.to_numeric(df["WindowEndSec"], errors="coerce")

    df = df.dropna(subset=["WindowIndex"]).copy()
    df["WindowIndex"] = df["WindowIndex"].astype(int)

    if MAX_DURING_WINDOWS is not None:
        df = df[df["WindowIndex"] < MAX_DURING_WINDOWS].copy()

    return df[df["Phase"] == "DURING"].copy()


def load_sensor_band_power():
    df = pd.read_csv(SENSOR_BAND_POWER_CSV)

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()

    df["Subject"] = df["Subject"].apply(clean_subject_id)
    df["Condition"] = df["Condition"].apply(clean_condition)
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

    return df


# ============================================================
# FEATURE CONSTRUCTION
# ============================================================

def add_plot_value(df, metric, log10_absolute=True):
    df = df.copy()
    if metric == "absolute" and log10_absolute:
        df["PlotValue"] = safe_log10(df["Value"])
    else:
        df["PlotValue"] = df["Value"]
    return df


def build_source_trajectory_table(
    band="Alpha",
    metric="relative",
    network="all_rois",
    band_scheme=BAND_SCHEME,
    log10_absolute=True,
):
    whole = load_source_whole()
    during = load_source_during_5min()

    whole = whole[
        (whole["Band"] == band)
        & (whole["Metric"] == metric)
        & (whole["BandScheme"] == band_scheme)
    ].copy()

    during = during[
        (during["Band"] == band)
        & (during["Metric"] == metric)
        & (during["BandScheme"] == band_scheme)
    ].copy()

    if network != "all_rois":
        whole = whole[whole["Network"] == network].copy()
        during = during[during["Network"] == network].copy()

    if whole.empty or during.empty:
        return pd.DataFrame()

    whole = add_plot_value(whole, metric, log10_absolute)
    during = add_plot_value(during, metric, log10_absolute)

    whole_phase = (
        whole.groupby(["Subject", "Condition", "Phase"], as_index=False)["PlotValue"]
        .mean()
        .rename(columns={"PlotValue": "MeanSourcePower"})
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
        during.groupby(["Subject", "Condition", "WindowIndex"], as_index=False)["PlotValue"]
        .mean()
        .rename(columns={"PlotValue": "MeanSourcePower"})
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
    out = out[out["Condition"].isin(CONDITIONS)].copy()

    if out.empty:
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

    change_cols = ["PRE_change"] + all_during_change_cols + ["POST_change"]

    def slope(row, cols):
        y = np.array([row[c] for c in cols], dtype=float)
        x = np.arange(len(y), dtype=float)
        mask = np.isfinite(y)
        if mask.sum() < 2:
            return np.nan
        return np.polyfit(x[mask], y[mask], deg=1)[0]

    out["Trajectory_slope"] = out.apply(lambda r: slope(r, change_cols), axis=1)
    out["During_slope"] = out.apply(lambda r: slope(r, all_during_change_cols), axis=1)

    out["Band"] = band
    out["Metric"] = metric
    out["Network"] = network
    out["BandScheme"] = band_scheme

    return out


# ============================================================
# STATS HELPERS
# ============================================================

def sem(x):
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    if len(x) < 2:
        return np.nan
    return x.std(ddof=1) / np.sqrt(len(x))


def cohen_d_paired(diff):
    diff = pd.to_numeric(pd.Series(diff), errors="coerce").dropna()
    if len(diff) < 2 or diff.std(ddof=1) == 0:
        return np.nan
    return diff.mean() / diff.std(ddof=1)


def cohen_d_independent(a, b):
    a = pd.to_numeric(pd.Series(a), errors="coerce").dropna().values
    b = pd.to_numeric(pd.Series(b), errors="coerce").dropna().values
    if len(a) < 2 or len(b) < 2:
        return np.nan

    pooled = np.sqrt(
        ((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1))
        / (len(a) + len(b) - 2)
    )
    if pooled == 0:
        return np.nan
    return (np.mean(a) - np.mean(b)) / pooled


def sign_flip_p(diff, n_perm=10000, seed=42):
    rng = np.random.default_rng(seed)
    diff = pd.to_numeric(pd.Series(diff), errors="coerce").dropna().values
    if len(diff) < 2:
        return np.nan

    obs = np.mean(diff)
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = np.mean(diff * rng.choice([-1, 1], size=len(diff)))

    return (np.sum(np.abs(null) >= np.abs(obs)) + 1) / (n_perm + 1)


def fdr_bh(pvals, alpha=0.05):
    pvals = np.asarray(pvals, dtype=float)
    adj = np.full_like(pvals, np.nan, dtype=float)
    reject = np.full_like(pvals, False, dtype=bool)

    valid = np.isfinite(pvals)
    p = pvals[valid]
    if len(p) == 0:
        return adj, reject

    order = np.argsort(p)
    ranked = p[order]
    n = len(ranked)

    adj_ranked = ranked * n / (np.arange(n) + 1)
    adj_ranked = np.minimum.accumulate(adj_ranked[::-1])[::-1]
    adj_ranked = np.clip(adj_ranked, 0, 1)

    adj_valid = np.empty_like(adj_ranked)
    adj_valid[order] = adj_ranked

    adj[valid] = adj_valid
    reject[valid] = adj_valid < alpha

    return adj, reject


def paired_condition_vs_sham(df, value_col, condition, sham="Cond5", n_perm=10000):
    sub = df[df["Condition"].isin([condition, sham])].copy()

    wide = sub.pivot_table(
        index="Subject",
        columns="Condition",
        values=value_col,
        aggfunc="mean",
    ).dropna(subset=[condition, sham])

    if len(wide) < 3:
        return {
            "Condition": condition,
            "Reference": sham,
            "Feature": value_col,
            "N": len(wide),
            "MeanDiff": np.nan,
            "T": np.nan,
            "P_ttest": np.nan,
            "P_signflip": np.nan,
            "Cohen_dz": np.nan,
        }

    diff = wide[condition] - wide[sham]
    t, p = ttest_rel(wide[condition], wide[sham])

    return {
        "Condition": condition,
        "Reference": sham,
        "Feature": value_col,
        "N": len(wide),
        "MeanDiff": diff.mean(),
        "T": t,
        "P_ttest": p,
        "P_signflip": sign_flip_p(diff, n_perm=n_perm),
        "Cohen_dz": cohen_d_paired(diff),
    }


def corr_stats(df, x_col, y_col="MeditationScore", n_perm=10000, seed=42):
    sub = df[[x_col, y_col]].dropna()
    if len(sub) < 3:
        return {
            "Feature": x_col,
            "N": len(sub),
            "Pearson_r": np.nan,
            "Pearson_p": np.nan,
            "Spearman_rho": np.nan,
            "Spearman_p": np.nan,
            "Permutation_Spearman_p": np.nan,
        }

    x = sub[x_col].values
    y = sub[y_col].values

    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)

    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        yr = rng.permutation(y)
        null[i], _ = spearmanr(x, yr)

    p_perm = (np.sum(np.abs(null) >= np.abs(sr)) + 1) / (n_perm + 1)

    return {
        "Feature": x_col,
        "N": len(sub),
        "Pearson_r": pr,
        "Pearson_p": pp,
        "Spearman_rho": sr,
        "Spearman_p": sp,
        "Permutation_Spearman_p": p_perm,
    }


def savefig(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Successfully saved figure: {path}")
