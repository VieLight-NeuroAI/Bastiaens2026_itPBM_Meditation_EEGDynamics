# ============================================================
# FigS4_fooof_source_maps_40Hz_zapline.py
#
# Supplementary Figure S4 (Supplementary S.4), 40 Hz row: source-level
# alpha peak and aperiodic spectral changes from the ZapLine-cleaned
# 40 Hz source FOOOF features. The 4 Hz, 10 Hz, 120 Hz and sham rows are
# drawn by the companion standard script
# FigS4_fooof_source_maps_standard.py
#
# Group-level source ROI FOOOF visualization using:
#   PRE
#   DURING 5-min windows minus PRE
#   POST minus PRE
#
# This is the same basic structure as the band-power source FDR map script,
# but instead of Delta/Theta/Alpha/Beta/Gamma it plots four FOOOF features:
#   1. Alpha APF / alpha peak center frequency   (PeakBand=Alpha, PeakCF)
#   2. Alpha peak power                          (PeakBand=Alpha, PeakPower)
#   3. Aperiodic exponent                        (global ROI metric)
#   4. Aperiodic offset                          (global ROI metric)
#
# Statistics:
#   ROI-wise paired sign-flip permutation test (5,000 permutations)
#   FDR correction across the 200 ROIs within each map.
#
# Inputs expected from the portable data folder:
#   Data\source_results\source_fooof_40Hz\
#       source_roi_fooof_all_subjects_conditions_long.csv
#       during_5min_source_roi_fooof_all_subjects_conditions_long.csv
#
# Outputs:
#   1. All-ROI group mean trajectory figure
#   2. FDR-significant-only trajectory figure
#   3. ROI-wise stats CSV
#
# Saves PNG only. Does NOT display in notebook.
#
# Requirements:
#   pip install mne pandas numpy matplotlib scipy statsmodels pyvista pyvistaqt pyqt5
# ============================================================

from pathlib import Path
import re
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib as mpl
import mne

from scipy.stats import ttest_rel
from statsmodels.stats.multitest import fdrcorrection


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "Data"

FOOOF_ROOT = DATA_ROOT / "source_results" / "source_fooof_40Hz"
FOOOF_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "source_results" / "source_fooof_40Hz"

PRE_POST_FOOOF_CSV = FOOOF_ROOT / "source_roi_fooof_all_subjects_conditions_long.csv"

DURING_5MIN_FOOOF_CSV = FOOOF_ROOT / "during_5min_source_roi_fooof_all_subjects_conditions_long.csv"

OUTPUT_DIR = (
    FOOOF_OUTPUT_ROOT
    / "figures"
    / "fooof_fdr_APF_power_exp_offset"
)

STATS_DIR = (
    FOOOF_OUTPUT_ROOT
    / "stats_fooof_fdr_APF_power_exp_offset"
)

FSAVERAGE5_DIR = DATA_ROOT / "fsaverage5"

SUBJECTS_DIR = FSAVERAGE5_DIR.parent
SUBJECT = FSAVERAGE5_DIR.name


# ============================================================
# SETTINGS
# ============================================================

CONDITIONS_TO_PLOT = ["Cond1", "Cond2", "Cond3", "Cond4", "Cond5", "Cond6"]

CONDITION_LABELS = {
    "Cond1": "4Hz",
    "Cond2": "10Hz",
    "Cond3": "40Hz",
    "Cond4": "120Hz",
    "Cond5": "Sham",
    "Cond6": "Control",
}

# FOOOF features to plot.
# For peak-dependent metrics, PeakBand="Alpha" is used.
# For global aperiodic metrics, PeakBand is ignored and duplicate PeakBand rows
# are averaged so each subject x ROI contributes once.
FOOOF_FEATURES_TO_PLOT = [
    {
        "feature_id": "Alpha_APF_PeakCF",
        "label": "APF",
        "column": "PeakCF",
        "peak_band": "Alpha",
        "peak_dependent": True,
        "pre_cbar_label": "APF\nPRE",
        "contrast_cbar_label": "$\\Delta$ APF",
    },
    {
        "feature_id": "Alpha_PeakPower",
        "label": "APP",
        "column": "PeakPower",
        "peak_band": "Alpha",
        "peak_dependent": True,
        "pre_cbar_label": "APP\nPRE",
        "contrast_cbar_label": "$\\Delta$ APP",
    },
    {
        "feature_id": "AperiodicExponent",
        "label": "Aperiodic exponent",
        "column": "AperiodicExponent",
        "peak_band": None,
        "peak_dependent": False,
        "pre_cbar_label": "Exponent\nPRE",
        "contrast_cbar_label": "$\\Delta$ Exponent",
    },
    {
        "feature_id": "AperiodicOffset",
        "label": "Aperiodic offset",
        "column": "AperiodicOffset",
        "peak_band": None,
        "peak_dependent": False,
        "pre_cbar_label": "Offset\nPRE",
        "contrast_cbar_label": "$\\Delta$ Offset",
    },
]

# For APF/PeakPower, only use values when a FOOOF Gaussian peak was detected
# in the alpha band.
REQUIRE_FOOOF_GAUSSIAN_PEAK_FOR_PEAK_METRICS = True

# Number of DURING windows to include.
# 4 = first 20 minutes if windows are 0-5, 5-10, 10-15, 15-20.
MAX_DURING_WINDOWS = 4

# Figure outputs
SAVE_ALL_ROIS_FIGURE = True
SAVE_FDR_SIGNIFICANT_ONLY_FIGURE = True

# Statistics
N_PERM = 5000
ALPHA_FDR = 0.05
RANDOM_STATE = 42

# "per_map" = FDR correction across 200 ROIs separately for each window/contrast.
# "global" = FDR correction across all ROI x contrast tests together.
FDR_SCOPE = "per_map"  # "per_map" or "global"

# Optional visual thresholding.
THRESHOLD_PERCENTILE_PRE = None
THRESHOLD_PERCENTILE_CONTRAST = None


# ============================================================
# BRAIN PLOTTING SETTINGS
# ============================================================

SURFACE = "pial"
CORTEX = "low_contrast"
BACKGROUND = "white"
BRAIN_SIZE = (1100, 850)

PRE_COLORMAP = "magma"
CONTRAST_COLORMAP = "coolwarm"

PRE_VMIN_PERCENTILE = 2
PRE_VMAX_PERCENTILE = 98
CONTRAST_ABS_PERCENTILE = 98

CONTRAST_LOWER_LIM_FRAC = 0.05
CONTRAST_MID_LIM_FRAC = 0.50

DPI_SAVE = 300

VIEW_COLUMNS = [
    ("lh", "lateral", "Left"),
    ("rh", "lateral", "Right"),
    ("lh", "medial", "Left medial"),
    ("rh", "medial", "Right medial"),
]


# ============================================================
# SET 3D BACKEND
# ============================================================

mne.viz.set_3d_backend("pyvistaqt")
print("MNE 3D backend:", mne.viz.get_3d_backend())


# ============================================================
# HELPERS
# ============================================================

def clean_string(x):
    return str(x).strip()


def normalize_subject(x):
    x = str(x).strip()
    x = x.replace("MED_PBM_", "")
    if x.isdigit():
        return f"{int(x):03d}"
    return x


def safe_filename(x):
    x = str(x)
    x = re.sub(r"[^A-Za-z0-9_.-]+", "_", x)
    return x.strip("_")


def is_schaefer_brain_label(label_name):
    """
    Keep true Schaefer cortical parcels only.
    Remove unknown / medial wall / background labels.
    """
    name = str(label_name).lower()

    bad_terms = [
        "unknown",
        "medial",
        "wall",
        "background",
        "corpuscallosum",
        "corpus_callosum",
    ]

    return not any(term in name for term in bad_terms)


def bool_series(x):
    return x.astype(str).str.lower().isin(["true", "1", "yes"])


def standardize_fooof_table(df):
    required_cols = {
        "Subject", "Condition", "Phase", "ROI",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in FOOOF CSV: {missing}\n"
            f"Available columns are:\n{list(df.columns)}"
        )

    for col in ["Subject", "Condition", "Phase", "ROI"]:
        df[col] = df[col].map(clean_string)

    if "PeakBand" in df.columns:
        df["PeakBand"] = df["PeakBand"].map(clean_string)

    df["Subject"] = df["Subject"].map(normalize_subject)
    df["Condition"] = df["Condition"].str.replace(" ", "", regex=False)
    df["Phase"] = df["Phase"].str.upper()

    df = df[df["ROI"].map(is_schaefer_brain_label)].copy()

    numeric_cols = [
        "PeakCF",
        "PeakPower",
        "AperiodicExponent",
        "AperiodicOffset",
        "AperiodicKnee",
        "NFOOOFGaussianPeaksInBand",
        "NFOOOFGaussianPeaksTotal",
        "FOOOFR2",
        "FOOOFError",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "FOOOFFitSuccess" in df.columns:
        before = len(df)
        df = df[bool_series(df["FOOOFFitSuccess"])].copy()
        print(f"Keeping successful FOOOF fits: {len(df)} / {before} rows")

    return df


def load_zapline_note():
    print(
        "\nNOTE: This script reads the 40 Hz/ZapLine FOOOF aggregate CSVs from "
        f"{FOOOF_ROOT}."
    )


def load_whole_recording_fooof_features():
    if not PRE_POST_FOOOF_CSV.exists():
        raise FileNotFoundError(f"Could not find PRE/POST FOOOF CSV:\n{PRE_POST_FOOOF_CSV}")

    df = pd.read_csv(PRE_POST_FOOOF_CSV)
    df = standardize_fooof_table(df)
    df = df[df["Phase"].isin(["PRE", "POST"])].copy()

    if df.empty:
        raise ValueError(f"No PRE/POST rows found in:\n{PRE_POST_FOOOF_CSV}")

    print("\nLoaded PRE/POST FOOOF CSV:")
    print("  Path:", PRE_POST_FOOOF_CSV)
    print("  Shape:", df.shape)
    print("  Subjects:", df["Subject"].nunique())
    print("  Conditions:", sorted(df["Condition"].dropna().unique()))
    print("  Phases:", sorted(df["Phase"].dropna().unique()))
    if "PeakBand" in df.columns:
        print("  Peak bands:", sorted(df["PeakBand"].dropna().unique()))

    return df


def load_during_5min_fooof_features():
    if not DURING_5MIN_FOOOF_CSV.exists():
        raise FileNotFoundError(f"Could not find DURING FOOOF CSV:\n{DURING_5MIN_FOOOF_CSV}")

    df = pd.read_csv(DURING_5MIN_FOOOF_CSV)
    df = standardize_fooof_table(df)
    df = df[df["Phase"] == "DURING"].copy()

    required_window_cols = {"WindowIndex", "WindowStartSec", "WindowEndSec"}
    missing = required_window_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"DURING 5-min FOOOF CSV is missing required window columns: {missing}\n"
            f"Available columns are:\n{list(df.columns)}"
        )

    df["WindowIndex"] = pd.to_numeric(df["WindowIndex"], errors="coerce")
    df["WindowStartSec"] = pd.to_numeric(df["WindowStartSec"], errors="coerce")
    df["WindowEndSec"] = pd.to_numeric(df["WindowEndSec"], errors="coerce")

    df = df.dropna(subset=["WindowIndex"]).copy()
    df["WindowIndex"] = df["WindowIndex"].astype(int)

    if MAX_DURING_WINDOWS is not None:
        df = df[df["WindowIndex"] < MAX_DURING_WINDOWS].copy()

    if df.empty:
        raise ValueError(f"No DURING rows found in:\n{DURING_5MIN_FOOOF_CSV}")

    print("\nLoaded DURING 5-min FOOOF CSV:")
    print("  Path:", DURING_5MIN_FOOOF_CSV)
    print("  Shape:", df.shape)
    print("  Subjects:", df["Subject"].nunique())
    print("  Conditions:", sorted(df["Condition"].dropna().unique()))
    print("  Windows:", sorted(df["WindowIndex"].dropna().unique()))
    if "PeakBand" in df.columns:
        print("  Peak bands:", sorted(df["PeakBand"].dropna().unique()))

    return df


def prepare_values(df, condition, feature_spec):
    feature_col = feature_spec["column"]
    peak_band = feature_spec["peak_band"]

    if feature_col not in df.columns:
        print(f"[WARNING] Missing FOOOF column: {feature_col}")
        return pd.DataFrame()

    d = df[df["Condition"] == condition].copy()

    if d.empty:
        return d

    if peak_band is not None:
        if "PeakBand" not in d.columns:
            raise ValueError("PeakBand column is required for alpha APF / alpha peak power.")
        d = d[d["PeakBand"] == peak_band].copy()

        if (
            feature_spec["peak_dependent"]
            and REQUIRE_FOOOF_GAUSSIAN_PEAK_FOR_PEAK_METRICS
            and "NFOOOFGaussianPeaksInBand" in d.columns
        ):
            d.loc[d["NFOOOFGaussianPeaksInBand"] <= 0, feature_col] = np.nan

    d["PlotValue"] = pd.to_numeric(d[feature_col], errors="coerce")

    keep_cols = ["Subject", "Condition", "Phase", "ROI", "PlotValue"]
    group_cols = ["Subject", "Condition", "Phase", "ROI"]

    if "WindowIndex" in d.columns:
        keep_cols = [
            "Subject", "Condition", "Phase", "WindowIndex",
            "WindowStartSec", "WindowEndSec", "ROI", "PlotValue",
        ]
        group_cols = ["Subject", "Condition", "Phase", "WindowIndex", "ROI"]

        d = d[keep_cols].groupby(group_cols, as_index=False).agg(
            PlotValue=("PlotValue", "mean"),
            WindowStartSec=("WindowStartSec", "median"),
            WindowEndSec=("WindowEndSec", "median"),
        )
    else:
        d = d[keep_cols].groupby(group_cols, as_index=False)["PlotValue"].mean()

    return d


def load_schaefer_labels():
    label_dir = FSAVERAGE5_DIR / "label"
    surf_dir = FSAVERAGE5_DIR / "surf"

    if not label_dir.exists():
        raise FileNotFoundError(f"Missing label directory:\n{label_dir}")
    if not surf_dir.exists():
        raise FileNotFoundError(f"Missing surf directory:\n{surf_dir}")

    annot_files = sorted(label_dir.glob("*Schaefer*200*annot"))
    if len(annot_files) == 0:
        raise FileNotFoundError(f"No Schaefer-200 annot files found in:\n{label_dir}")

    lh_files = [f for f in annot_files if f.name.startswith("lh.")]
    if len(lh_files) == 0:
        raise FileNotFoundError("Could not find lh.*Schaefer*200*.annot file.")

    parc = lh_files[0].name.replace("lh.", "").replace(".annot", "")

    labels_all = mne.read_labels_from_annot(
        subject=SUBJECT,
        parc=parc,
        subjects_dir=SUBJECTS_DIR,
        hemi="both",
        sort=True,
    )

    labels = [
        lab for lab in labels_all
        if is_schaefer_brain_label(lab.name)
    ]

    removed = [
        lab.name for lab in labels_all
        if not is_schaefer_brain_label(lab.name)
    ]

    print(f"Parcellation: {parc}")
    print(f"Labels before filtering: {len(labels_all)}")
    print(f"Labels after filtering:  {len(labels)}")
    print("Removed labels:", removed)

    if len(labels) != 200:
        print(
            f"[WARNING] Expected 200 Schaefer parcels after filtering, "
            f"but got {len(labels)}."
        )

    return labels, parc


def make_label_lookup(labels):
    lookup = {}
    for lab in labels:
        variants = {
            lab.name,
            lab.name.lower(),
            lab.name.replace("-lh", ""),
            lab.name.replace("-rh", ""),
            lab.name.lower().replace("-lh", ""),
            lab.name.lower().replace("-rh", ""),
        }
        for v in variants:
            lookup[v] = lab
    return lookup


def find_label_for_roi(roi, label_lookup):
    roi = str(roi).strip()
    candidates = [
        roi,
        roi.lower(),
        roi.replace("-lh", ""),
        roi.replace("-rh", ""),
        roi.lower().replace("-lh", ""),
        roi.lower().replace("-rh", ""),
    ]
    for c in candidates:
        if c in label_lookup:
            return label_lookup[c]
    return None


def build_source_space():
    src = mne.setup_source_space(
        subject=SUBJECT,
        spacing="all",
        subjects_dir=SUBJECTS_DIR,
        add_dist=False,
        verbose=False,
    )

    lh_vertno = src[0]["vertno"]
    rh_vertno = src[1]["vertno"]

    print("LH vertices:", len(lh_vertno))
    print("RH vertices:", len(rh_vertno))

    return src, lh_vertno, rh_vertno


def roi_values_to_stc(roi_values, label_lookup, lh_vertno, rh_vertno):
    lh_data = np.full((len(lh_vertno), 1), np.nan, dtype=float)
    rh_data = np.full((len(rh_vertno), 1), np.nan, dtype=float)

    matched = 0
    unmatched = []

    for roi, val in roi_values.items():
        if not np.isfinite(val):
            continue

        lab = find_label_for_roi(roi, label_lookup)

        if lab is None:
            unmatched.append(roi)
            continue

        matched += 1

        if lab.hemi == "lh":
            _, src_idx, _ = np.intersect1d(
                lh_vertno,
                lab.vertices,
                return_indices=True,
            )
            if len(src_idx) > 0:
                lh_data[src_idx, 0] = val

        elif lab.hemi == "rh":
            _, src_idx, _ = np.intersect1d(
                rh_vertno,
                lab.vertices,
                return_indices=True,
            )
            if len(src_idx) > 0:
                rh_data[src_idx, 0] = val

    if len(unmatched) > 0:
        print(f"Unmatched ROIs ignored: {len(unmatched)}")

    lh_data = np.nan_to_num(lh_data, nan=0.0)
    rh_data = np.nan_to_num(rh_data, nan=0.0)

    brain_data = np.vstack([lh_data, rh_data])

    stc = mne.SourceEstimate(
        data=brain_data,
        vertices=[lh_vertno, rh_vertno],
        tmin=0,
        tstep=1,
        subject=SUBJECT,
    )

    return stc, matched, unmatched


def crop_white_border(img, tol=245):
    if img.ndim == 3:
        mask = np.any(img < tol, axis=2)
    else:
        mask = img < tol

    coords = np.argwhere(mask)

    if coords.size == 0:
        return img

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    return img[y0:y1, x0:x1]


def apply_threshold(roi_values, percentile):
    if percentile is None:
        return roi_values

    vals = roi_values.values.astype(float)
    vals = vals[np.isfinite(vals)]

    if len(vals) == 0:
        return roi_values

    if np.nanmin(vals) < 0:
        thr = np.nanpercentile(np.abs(vals), percentile)
        return roi_values.where(np.abs(roi_values) >= thr, 0.0)

    thr = np.nanpercentile(vals, percentile)
    return roi_values.where(roi_values >= thr, 0.0)


def get_pre_clim(values):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]

    if len(vals) == 0:
        raise ValueError("No finite PRE values.")

    vmin = np.nanpercentile(vals, PRE_VMIN_PERCENTILE)
    vmax = np.nanpercentile(vals, PRE_VMAX_PERCENTILE)

    if np.isclose(vmin, vmax):
        vmin = np.nanmin(vals)
        vmax = np.nanmax(vals)

    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6

    vmid = vmin + 0.5 * (vmax - vmin)
    clim = dict(kind="value", lims=[float(vmin), float(vmid), float(vmax)])

    return clim, vmin, vmax


def get_contrast_clim(values):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]

    if len(vals) == 0:
        raise ValueError("No finite contrast values.")

    vmax = np.nanpercentile(np.abs(vals), CONTRAST_ABS_PERCENTILE)

    if not np.isfinite(vmax) or vmax <= 0:
        vmax = np.nanmax(np.abs(vals))

    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1e-6

    clim = dict(
        kind="value",
        pos_lims=[
            float(vmax * CONTRAST_LOWER_LIM_FRAC),
            float(vmax * CONTRAST_MID_LIM_FRAC),
            float(vmax),
        ]
    )

    return clim, -vmax, vmax


def render_view(stc, hemi, view, clim, colormap):
    brain = stc.plot(
        subject=SUBJECT,
        subjects_dir=SUBJECTS_DIR,
        hemi=hemi,
        surface=SURFACE,
        views=view,
        clim=clim,
        colormap=colormap,
        cortex=CORTEX,
        background=BACKGROUND,
        foreground="black",
        time_viewer=False,
        size=BRAIN_SIZE,
        smoothing_steps=1,
        colorbar=False,
        backend="pyvistaqt",
    )

    img = brain.screenshot(mode="rgb")
    brain.close()

    return crop_white_border(img)


# ============================================================
# STATISTICS
# ============================================================

def paired_sign_flip_pvalue(diff, n_perm=5000, rng=None):
    """
    Two-sided paired sign-flip permutation test for whether mean(diff) != 0.
    """
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]

    if len(diff) < 2:
        return np.nan, np.nan

    if rng is None:
        rng = np.random.default_rng()

    obs = np.mean(diff)

    perm_means = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        signs = rng.choice([-1, 1], size=len(diff))
        perm_means[i] = np.mean(diff * signs)

    p = (np.sum(np.abs(perm_means) >= np.abs(obs)) + 1) / (n_perm + 1)

    t_stat, _ = ttest_rel(diff, np.zeros_like(diff))
    return t_stat, p


def run_roi_stats_for_all_maps(diff_long, n_perm=5000, alpha_fdr=0.05, random_state=42):
    """
    diff_long columns:
        MapName, MapOrder, ROI, Subject, Diff
    """
    rng = np.random.default_rng(random_state)

    results = []

    if diff_long is None or diff_long.empty:
        return pd.DataFrame()

    for (map_order, map_name), map_df in diff_long.groupby(["MapOrder", "MapName"], sort=True):
        for roi, roi_df in map_df.groupby("ROI"):
            diff = roi_df["Diff"].to_numpy(dtype=float)
            diff = diff[np.isfinite(diff)]

            t_stat, p_val = paired_sign_flip_pvalue(
                diff,
                n_perm=n_perm,
                rng=rng,
            )

            results.append(
                {
                    "MapOrder": int(map_order),
                    "MapName": map_name,
                    "MapID": roi_df["MapID"].iloc[0] if "MapID" in roi_df.columns else "",
                    "ROI": roi,
                    "N": int(len(diff)),
                    "MeanDiff": float(np.nanmean(diff)) if len(diff) > 0 else np.nan,
                    "T": t_stat,
                    "P_uncorrected": p_val,
                }
            )

    res = pd.DataFrame(results)

    if res.empty:
        return res

    res["P_FDR"] = np.nan
    res["Significant_FDR"] = False

    if FDR_SCOPE == "per_map":
        for map_name, idx in res.groupby("MapName").groups.items():
            idx = list(idx)
            valid = res.loc[idx, "P_uncorrected"].notna()

            if valid.sum() == 0:
                continue

            idx_valid = res.loc[idx].loc[valid].index
            reject, p_fdr = fdrcorrection(
                res.loc[idx_valid, "P_uncorrected"].values,
                alpha=alpha_fdr,
            )

            res.loc[idx_valid, "P_FDR"] = p_fdr
            res.loc[idx_valid, "Significant_FDR"] = reject

    elif FDR_SCOPE == "global":
        valid = res["P_uncorrected"].notna()

        if valid.sum() > 0:
            reject, p_fdr = fdrcorrection(
                res.loc[valid, "P_uncorrected"].values,
                alpha=alpha_fdr,
            )

            res.loc[valid, "P_FDR"] = p_fdr
            res.loc[valid, "Significant_FDR"] = reject

    else:
        raise ValueError("FDR_SCOPE must be 'per_map' or 'global'.")

    res["Significant_FDR"] = res["Significant_FDR"].fillna(False)

    return res.sort_values(["MapOrder", "P_uncorrected", "ROI"]).reset_index(drop=True)


# ============================================================
# MAP COMPUTATION
# ============================================================

def compute_maps_and_stats_for_condition(df_whole, df_during, condition, feature_spec):
    whole = prepare_values(df_whole, condition, feature_spec)
    during = prepare_values(df_during, condition, feature_spec)

    if whole.empty or during.empty:
        return None, None

    pre = whole[whole["Phase"] == "PRE"].copy()
    post = whole[whole["Phase"] == "POST"].copy()

    if pre.empty or post.empty:
        return None, None

    pre_subject = pre.rename(columns={"PlotValue": "PRE"})[
        ["Subject", "ROI", "PRE"]
    ]

    pre_group_map = (
        pre_subject.groupby("ROI")["PRE"]
        .mean()
        .sort_index()
    )

    maps_to_plot = [
        {
            "name": "PRE",
            "map_id": "PRE",
            "subtitle": "",
            "values": pre_group_map,
            "type": "pre",
            "n_subjects": pre_subject["Subject"].nunique(),
        }
    ]

    diff_rows = []
    map_order = 1

    window_info = (
        during
        .groupby("WindowIndex", as_index=False)
        .agg(
            WindowStartSec=("WindowStartSec", "median"),
            WindowEndSec=("WindowEndSec", "median"),
            NSubjectsInWindow=("Subject", "nunique"),
        )
        .sort_values("WindowIndex")
    )

    if MAX_DURING_WINDOWS is not None:
        window_info = window_info[window_info["WindowIndex"] < MAX_DURING_WINDOWS].copy()

    for _, win in window_info.iterrows():
        window_idx = int(win["WindowIndex"])
        wstart = float(win["WindowStartSec"])
        wend = float(win["WindowEndSec"])

        dwin = during[during["WindowIndex"] == window_idx].copy()
        dwin = dwin.rename(columns={"PlotValue": "DURING"})[
            ["Subject", "ROI", "DURING"]
        ]

        merged = dwin.merge(
            pre_subject,
            on=["Subject", "ROI"],
            how="inner",
        )

        if merged.empty:
            continue

        merged["Diff"] = merged["DURING"] - merged["PRE"]

        map_name = f"DURING {window_idx + 1} - PRE"
        map_id = f"DURING{window_idx + 1}_minus_PRE"

        diff_map = (
            merged.groupby("ROI")["Diff"]
            .mean()
            .sort_index()
        )

        maps_to_plot.append(
            {
                "name": map_name,
                "map_id": map_id,
                "subtitle": f"{wstart / 60:.0f}-{wend / 60:.0f} min",
                "values": diff_map,
                "type": "contrast",
                "n_subjects": merged["Subject"].nunique(),
            }
        )

        for _, row in merged.iterrows():
            diff_rows.append(
                {
                    "MapOrder": map_order,
                    "MapName": map_name,
                    "MapID": map_id,
                    "WindowIndex": window_idx,
                    "WindowStartSec": wstart,
                    "WindowEndSec": wend,
                    "Subject": row["Subject"],
                    "ROI": row["ROI"],
                    "Diff": row["Diff"],
                }
            )

        map_order += 1

    post_subject = post.rename(columns={"PlotValue": "POST"})[
        ["Subject", "ROI", "POST"]
    ]

    post_merged = post_subject.merge(
        pre_subject,
        on=["Subject", "ROI"],
        how="inner",
    )

    if not post_merged.empty:
        post_merged["Diff"] = post_merged["POST"] - post_merged["PRE"]

        post_diff_map = (
            post_merged.groupby("ROI")["Diff"]
            .mean()
            .sort_index()
        )

        maps_to_plot.append(
            {
                "name": "POST - PRE",
                "map_id": "POST_minus_PRE",
                "subtitle": "",
                "values": post_diff_map,
                "type": "contrast",
                "n_subjects": post_merged["Subject"].nunique(),
            }
        )

        for _, row in post_merged.iterrows():
            diff_rows.append(
                {
                    "MapOrder": map_order,
                    "MapName": "POST - PRE",
                    "MapID": "POST_minus_PRE",
                    "WindowIndex": np.nan,
                    "WindowStartSec": np.nan,
                    "WindowEndSec": np.nan,
                    "Subject": row["Subject"],
                    "ROI": row["ROI"],
                    "Diff": row["Diff"],
                }
            )

    diff_long = pd.DataFrame(diff_rows)

    stats = run_roi_stats_for_all_maps(
        diff_long=diff_long,
        n_perm=N_PERM,
        alpha_fdr=ALPHA_FDR,
        random_state=RANDOM_STATE,
    )

    return maps_to_plot, stats


# ============================================================
# PLOTTING
# ============================================================

def compact_row_label(item):
    if item["type"] == "pre":
        return f"PRE\nN={item['n_subjects']}"

    if item["name"] == "POST - PRE":
        return f"POST - PRE\nN={item['n_subjects']}"

    match = re.search(r"DURING\s+(\d+)", item["name"])
    if match:
        label = f"D{match.group(1)} - PRE"
    else:
        label = item["name"]

    if item.get("subtitle", ""):
        label += f"\n{item['subtitle']}"

    label += f"\nN={item['n_subjects']}"

    return label


def plot_maps_trajectory_grid(
    maps_to_plot,
    stats,
    label_lookup,
    lh_vertno,
    rh_vertno,
    condition,
    feature_spec,
    significant_only=False,
):
    condition_label = CONDITION_LABELS.get(condition, condition)
    feature_label = feature_spec["label"]
    feature_id = feature_spec["feature_id"]

    n_maps = len(maps_to_plot)

    if n_maps < 2:
        print(f"Not enough maps for {condition} | {feature_label}, skipping.")
        return

    sig_rois_by_map = {}
    if stats is not None and not stats.empty:
        for map_name, subdf in stats.groupby("MapName"):
            sig_rois_by_map[map_name] = set(
                subdf.loc[subdf["Significant_FDR"], "ROI"].astype(str)
            )

    pre_vals = maps_to_plot[0]["values"].values.astype(float)
    pre_clim, pre_vmin, pre_vmax = get_pre_clim(pre_vals)

    contrast_vals_all = []
    for item in maps_to_plot[1:]:
        vals = item["values"].values.astype(float)
        vals = vals[np.isfinite(vals)]
        if len(vals) > 0:
            contrast_vals_all.append(vals)

    if len(contrast_vals_all) == 0:
        print(f"No contrast values for {condition} | {feature_label}, skipping.")
        return

    contrast_vals_all = np.concatenate(contrast_vals_all)
    contrast_clim, contrast_vmin, contrast_vmax = get_contrast_clim(contrast_vals_all)

    fig_width = 13.5
    fig_height = max(7.5, n_maps * 2.0)

    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=False)

    fig.subplots_adjust(
        left=0.04,
        right=0.97,
        top=0.92,
        bottom=0.04,
        wspace=0.02,
        hspace=0.06,
    )

    gs = fig.add_gridspec(
        nrows=n_maps,
        ncols=6,
        width_ratios=[0.38, 1.0, 1.0, 1.0, 1.0, 0.055],
        height_ratios=[1.0] * n_maps,
        wspace=0.02,
        hspace=0.06,
    )

    for col_idx, title in enumerate(["Left", "Right", "Left medial", "Right medial"], start=1):
        ax_title = fig.add_subplot(gs[0, col_idx])
        ax_title.set_title(title, fontsize=11, pad=10)
        ax_title.axis("off")

    for map_idx, item in enumerate(maps_to_plot):
        roi_values = item["values"].copy()

        if item["type"] == "pre":
            roi_values = apply_threshold(roi_values, THRESHOLD_PERCENTILE_PRE)
            clim = pre_clim
            cmap = PRE_COLORMAP
        else:
            if significant_only:
                sig_rois = sig_rois_by_map.get(item["name"], set())
                roi_values = roi_values.where(roi_values.index.astype(str).isin(sig_rois), 0.0)

            roi_values = apply_threshold(roi_values, THRESHOLD_PERCENTILE_CONTRAST)
            clim = contrast_clim
            cmap = CONTRAST_COLORMAP

        stc, matched, unmatched = roi_values_to_stc(
            roi_values=roi_values,
            label_lookup=label_lookup,
            lh_vertno=lh_vertno,
            rh_vertno=rh_vertno,
        )

        sig_text = ""
        if item["type"] == "contrast" and significant_only:
            n_sig = len(sig_rois_by_map.get(item["name"], set()))
            sig_text = f" | Sig={n_sig}"

        print(
            f"{condition} | {feature_label} | {item['name']}: "
            f"matched {matched} ROIs | N={item['n_subjects']}{sig_text}"
        )

        ax_label = fig.add_subplot(gs[map_idx, 0])
        ax_label.axis("off")
        ax_label.text(
            0.98,
            0.5,
            compact_row_label(item),
            ha="right",
            va="center",
            fontsize=9,
        )

        for view_col, (hemi, view, view_name) in enumerate(VIEW_COLUMNS, start=1):
            img = render_view(
                stc=stc,
                hemi=hemi,
                view=view,
                clim=clim,
                colormap=cmap,
            )

            ax = fig.add_subplot(gs[map_idx, view_col])
            ax.imshow(img)
            ax.axis("off")

    cax_pre = fig.add_subplot(gs[0, 5])
    sm_pre = mpl.cm.ScalarMappable(
        norm=mpl.colors.Normalize(vmin=pre_vmin, vmax=pre_vmax),
        cmap=PRE_COLORMAP,
    )
    sm_pre.set_array([])
    cbar_pre = fig.colorbar(sm_pre, cax=cax_pre, orientation="vertical")
    cbar_pre.set_label(feature_spec["pre_cbar_label"], fontsize=25)
    cbar_pre.ax.tick_params(labelsize=25)

    if n_maps > 1:
        cax_contrast = fig.add_subplot(gs[1:n_maps, 5])
    else:
        cax_contrast = fig.add_subplot(gs[:, 5])

    sm_contrast = mpl.cm.ScalarMappable(
        norm=mpl.colors.Normalize(vmin=contrast_vmin, vmax=contrast_vmax),
        cmap=CONTRAST_COLORMAP,
    )
    sm_contrast.set_array([])
    cbar_contrast = fig.colorbar(sm_contrast, cax=cax_contrast, orientation="vertical")
    cbar_contrast.set_label(feature_spec["contrast_cbar_label"], fontsize=25)
    cbar_contrast.ax.tick_params(labelsize=25)

    if significant_only:
        suffix = f"FDRsig_{FDR_SCOPE}"
        subtitle = f"FDR-significant ROIs only for contrasts | {FDR_SCOPE} FDR"
    else:
        suffix = "all"
        subtitle = "All ROIs"

    fig.suptitle(
        f"{condition} ({condition_label}) | {feature_label} | source_fooof_40Hz\n"
        f"PRE, DURING 5-min windows relative to PRE, and POST - PRE | "
        f"{subtitle}",
        fontsize=12,
        y=0.985,
    )

    out_name = (
        f"FigureS4_fooof_zap_{condition}_{condition_label}_{safe_filename(feature_id)}_{suffix}.png"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / out_name

    fig.savefig(out_path, dpi=DPI_SAVE, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Successfully saved figure: {out_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    load_zapline_note()

    print("\nLoading whole-recording PRE/POST FOOOF CSV...")
    df_whole = load_whole_recording_fooof_features()

    print("\nLoading 5-minute DURING FOOOF CSV...")
    df_during = load_during_5min_fooof_features()

    print("\nWhole rows:", len(df_whole))
    print("DURING 5-min rows:", len(df_during))
    print("Whole subjects:", df_whole["Subject"].nunique())
    print("DURING subjects:", df_during["Subject"].nunique())

    print("\nLoading Schaefer labels...")
    labels, parc = load_schaefer_labels()
    label_lookup = make_label_lookup(labels)

    print("\nBuilding source space...")
    src, lh_vertno, rh_vertno = build_source_space()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STATS_DIR.mkdir(parents=True, exist_ok=True)

    for condition in CONDITIONS_TO_PLOT:
        for feature_spec in FOOOF_FEATURES_TO_PLOT:
            feature_id = feature_spec["feature_id"]
            feature_label = feature_spec["label"]

            print("\n" + "=" * 90)
            print(f"Processing: {condition} | {feature_label}")
            print("=" * 90)

            maps_to_plot, stats = compute_maps_and_stats_for_condition(
                df_whole=df_whole,
                df_during=df_during,
                condition=condition,
                feature_spec=feature_spec,
            )

            if maps_to_plot is None or stats is None:
                print("No data for this selection, skipping.")
                continue

            condition_label = CONDITION_LABELS.get(condition, condition)

            stats_file = STATS_DIR / (
                f"figureS4_zap_{condition}_{condition_label}_{safe_filename(feature_id)}_roi_stats_{FDR_SCOPE}FDR.csv"
            )
            stats.to_csv(stats_file, index=False)
            print(f"Successfully saved stats: {stats_file}")

            print("Maps to plot:")
            for m in maps_to_plot:
                print("  -", m["name"], m["subtitle"], "N=", m["n_subjects"])

            if SAVE_ALL_ROIS_FIGURE:
                plot_maps_trajectory_grid(
                    maps_to_plot=maps_to_plot,
                    stats=stats,
                    label_lookup=label_lookup,
                    lh_vertno=lh_vertno,
                    rh_vertno=rh_vertno,
                    condition=condition,
                    feature_spec=feature_spec,
                    significant_only=False,
                )

            if SAVE_FDR_SIGNIFICANT_ONLY_FIGURE:
                plot_maps_trajectory_grid(
                    maps_to_plot=maps_to_plot,
                    stats=stats,
                    label_lookup=label_lookup,
                    lh_vertno=lh_vertno,
                    rh_vertno=rh_vertno,
                    condition=condition,
                    feature_spec=feature_spec,
                    significant_only=True,
                )


if __name__ == "__main__":
    main()
