#!/usr/bin/env python
from __future__ import annotations

"""
Fig03_FigS3_source_fdr_brain_maps.py

Source-space FDR brain-map generator.

Manuscript figures:
    Figure 3   Source-level relative spectral power changes
               (--metrics relative; Results 4.2)
    Figure S3  Source-level absolute spectral power changes, log10
               (--metrics absolute; Supplementary S.3)

Rows of each figure are 4 Hz, 10 Hz, 40 Hz, 120 Hz and sham. 4 Hz, 10 Hz,
120 Hz and sham use the standard cleaned/interpolated features
(--feature-family bandpower_standard); 40 Hz uses the ZapLine-cleaned
features (--feature-family bandpower_zapline --conditions Cond3).

Each map is PRE, or a PRE-referenced contrast (Dur5, Dur10, Dur15, Dur20,
Post). Parcel-wise two-sided sign-flip permutation tests (5,000
permutations) are FDR-corrected across the 200 Schaefer parcels within each
map (Methods 3.7, family iv). These are within-condition tests, not
active-vs-sham contrasts.

A FOOOF mode (--feature-family fooof) applies the same workflow to FOOOF
features such as alpha peak frequency (PeakCF). The Figure S4 FOOOF maps
were drawn with the dedicated scripts in ../Supplementary.

Band-power output files are prefixed with their manuscript figure
(Figure3_ / figure3_ for relative, FigureS3_ / figureS3_ for absolute).
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp


mne = None


def get_mne():
    """Import MNE only when rendering/source-space helpers are needed."""
    global mne
    if mne is None:
        import mne as _mne

        mne = _mne
    return mne


# ============================================================
# DEFAULT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "Data"
DATA_SOURCE_RESULTS_ROOT = DATA_ROOT / "source_results"
SOURCE_RESULTS_ROOT = PROJECT_ROOT / "outputs" / "source_results"

DEFAULT_STANDARD_PREPOST_DIR = (
    DATA_SOURCE_RESULTS_ROOT / "source_band_power_standard"
)
DEFAULT_STANDARD_DURING_DIR = (
    DATA_SOURCE_RESULTS_ROOT / "source_band_power_standard"
)

DEFAULT_ZAPLINE_PREPOST_CSV = (
    DATA_SOURCE_RESULTS_ROOT
    / "source_band_power_40Hz"
    / "source_roi_band_power_all_subjects_conditions_long.csv"
)
DEFAULT_ZAPLINE_DURING_CSV = (
    DATA_SOURCE_RESULTS_ROOT
    / "source_band_power_40Hz"
    / "during_5min_source_roi_band_power_all_subjects_conditions_long.csv"
)

DEFAULT_FOOOF_PREPOST_CSV = (
    DATA_SOURCE_RESULTS_ROOT
    / "source_fooof_standard"
    / "source_roi_fooof_all_subjects_conditions_long.csv"
)
DEFAULT_FOOOF_DURING_CSV = (
    DATA_SOURCE_RESULTS_ROOT
    / "source_fooof_standard"
    / "during_5min_source_roi_fooof_all_subjects_conditions_long.csv"
)

DEFAULT_FSAVERAGE5_DIR = DATA_ROOT / "fsaverage5"


# ============================================================
# SETTINGS
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

DISPLAY_TIME_LABELS = {
    0: "Dur5",
    1: "Dur10",
    2: "Dur15",
    3: "Dur20",
}

DEFAULT_BANDS = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
DEFAULT_METRICS = ["relative"]
DEFAULT_FOOOF_FEATURES = ["PeakCF"]
DEFAULT_PEAK_BANDS = ["Alpha"]

DEFAULT_BAND_SCHEME = "theta4_8_alpha8_13"
DEFAULT_MAX_DURING_WINDOWS = 4

APERIODIC_OR_MODEL_FEATURES = {
    "AperiodicOffset",
    "AperiodicExponent",
    "AperiodicKnee",
    "FOOOFR2",
    "FOOOFError",
    "NFOOOFGaussianPeaksTotal",
}


# ============================================================
# PLOTTING SETTINGS
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


@dataclass(frozen=True)
class FeatureSpec:
    feature_family: str
    band: str | None = None
    metric: str | None = None
    fooof_feature: str | None = None
    peak_band: str | None = None
    band_scheme: str = DEFAULT_BAND_SCHEME
    log10_absolute: bool = True

    @property
    def display_name(self) -> str:
        if self.feature_family.startswith("bandpower"):
            return f"{self.band} {self.metric}"
        if self.peak_band and self.fooof_feature not in APERIODIC_OR_MODEL_FEATURES:
            return f"{self.peak_band} {self.fooof_feature}"
        return str(self.fooof_feature)

    @property
    def output_tokens(self) -> tuple[str, ...]:
        if self.feature_family.startswith("bandpower"):
            return (str(self.band), str(self.metric), self.band_scheme)
        if self.peak_band and self.fooof_feature not in APERIODIC_OR_MODEL_FEATURES:
            return (str(self.peak_band), str(self.fooof_feature), "fooof")
        return ("all", str(self.fooof_feature), "fooof")

    @property
    def colorbar_pre_label(self) -> str:
        if self.feature_family.startswith("bandpower"):
            if self.metric == "absolute" and self.log10_absolute:
                return f"log10 {self.band}\nPRE"
            return f"{self.band}\nPRE"
        return f"{self.display_name}\nPRE"

    @property
    def colorbar_contrast_label(self) -> str:
        if self.feature_family.startswith("bandpower"):
            if self.metric == "absolute" and self.log10_absolute:
                return f"d log10 {self.band}"
            return f"d {self.band}"
        return f"d {self.display_name}"


# ============================================================
# CLEANING HELPERS
# ============================================================


def clean_string(x) -> str:
    return str(x).strip()


def normalize_subject(x) -> str:
    x = str(x).strip().replace("MED_PBM_", "")
    try:
        return f"{int(float(x)):03d}"
    except Exception:
        return x


def clean_condition(x) -> str:
    x = str(x).strip()
    mapping = {
        "4Hz": "Cond1",
        "4 Hz": "Cond1",
        "10Hz": "Cond2",
        "10 Hz": "Cond2",
        "40Hz": "Cond3",
        "40 Hz": "Cond3",
        "120Hz": "Cond4",
        "120 Hz": "Cond4",
        "Sham": "Cond5",
        "Control": "Cond6",
    }
    return mapping.get(x, x.replace(" ", ""))


def safe_log10(x):
    x = pd.to_numeric(x, errors="coerce")
    x = x.where(x > 0, np.nan)
    return np.log10(x)


def is_schaefer_brain_label(label_name) -> bool:
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


def infer_metadata_from_filename(path: Path):
    name = path.name
    subject_match = re.search(r"MED_PBM_(\d+)", name)
    cond_match = re.search(r"(Cond\d+)", name)
    phase_match = re.search(r"_(PRE|DURING|POST)_", name)

    subject = normalize_subject(subject_match.group(1)) if subject_match else None
    condition = cond_match.group(1) if cond_match else None
    phase = phase_match.group(1) if phase_match else None
    return subject, condition, phase


def source_subject_label(x) -> str:
    return f"MED_PBM_{normalize_subject(x)}"


# ============================================================
# DATA LOADING
# ============================================================


def standardize_common_columns(df: pd.DataFrame) -> pd.DataFrame:
    required = {"Subject", "Condition", "Phase", "ROI"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col in ["Subject", "Condition", "Phase", "ROI"]:
        df[col] = df[col].map(clean_string)

    df["Subject"] = df["Subject"].map(normalize_subject)
    df["Condition"] = df["Condition"].map(clean_condition)
    df["Phase"] = df["Phase"].str.upper()
    df = df[df["ROI"].map(is_schaefer_brain_label)].copy()
    return df


def standardize_bandpower_table(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "Subject",
        "Condition",
        "Phase",
        "ROI",
        "Band",
        "Metric",
        "Value",
        "BandScheme",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required band-power columns: {missing}")

    df = standardize_common_columns(df)
    for col in ["Band", "Metric", "BandScheme"]:
        df[col] = df[col].map(clean_string)
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    return df


def standardize_fooof_table(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_common_columns(df)
    if "PeakBand" in df.columns:
        df["PeakBand"] = df["PeakBand"].map(clean_string)
    for col in [
        "WindowIndex",
        "WindowStartSec",
        "WindowEndSec",
        "WindowDurationSec",
        "NEpochs",
        "NWindowEpochs",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_feature_csvs(feature_dir: Path, pattern: str, table_kind: str) -> pd.DataFrame:
    feature_path = Path(feature_dir)
    if feature_path.is_file():
        csv_files = [feature_path]
    else:
        csv_files = sorted(feature_path.glob(pattern))
        if not csv_files and "5min" in pattern:
            csv_files = sorted(feature_path.glob("*during_5min_source_roi_band_power_all_subjects_conditions_long*.csv"))
        if not csv_files:
            csv_files = sorted(feature_path.glob("*source_roi_band_power_all_subjects_conditions_long*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found with pattern {pattern} in {feature_dir}"
        )

    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        subject, condition, phase = infer_metadata_from_filename(f)

        if "Subject" not in df.columns or df["Subject"].isna().all():
            df["Subject"] = subject
        if "Condition" not in df.columns or df["Condition"].isna().all():
            df["Condition"] = condition
        if "Phase" not in df.columns or df["Phase"].isna().all():
            df["Phase"] = phase

        df["SourceFile"] = f.name
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    if table_kind == "bandpower":
        return standardize_bandpower_table(df_all)
    if table_kind == "fooof":
        return standardize_fooof_table(df_all)
    raise ValueError(f"Unknown table kind: {table_kind}")


def load_csv(path: Path, table_kind: str) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing input CSV: {path}")
    df = pd.read_csv(path)
    if table_kind == "bandpower":
        return standardize_bandpower_table(df)
    if table_kind == "fooof":
        return standardize_fooof_table(df)
    raise ValueError(f"Unknown table kind: {table_kind}")


def load_standard_bandpower(prepost_dir: Path, during_dir: Path):
    whole = load_feature_csvs(
        prepost_dir,
        pattern="*source_roi_band_power_long*.csv",
        table_kind="bandpower",
    )
    during = load_feature_csvs(
        during_dir,
        pattern="*source_roi_band_power_5min_long*.csv",
        table_kind="bandpower",
    )
    return whole[whole["Phase"].isin(["PRE", "POST"])].copy(), during[
        during["Phase"] == "DURING"
    ].copy()


def load_zapline_bandpower(prepost_csv: Path, during_csv: Path):
    whole = load_csv(prepost_csv, table_kind="bandpower")
    during = load_csv(during_csv, table_kind="bandpower")
    return whole[whole["Phase"].isin(["PRE", "POST"])].copy(), during[
        during["Phase"] == "DURING"
    ].copy()


def load_fooof(prepost_csv: Path, during_csv: Path):
    whole = load_csv(prepost_csv, table_kind="fooof")
    during = load_csv(during_csv, table_kind="fooof")
    return whole[whole["Phase"].isin(["PRE", "POST"])].copy(), during[
        during["Phase"] == "DURING"
    ].copy()


def filter_during_windows(df: pd.DataFrame, max_during_windows: int | None):
    if df.empty:
        return df
    required = {"WindowIndex", "WindowStartSec", "WindowEndSec"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DURING table is missing window columns: {missing}")

    df = df.copy()
    df["WindowIndex"] = pd.to_numeric(df["WindowIndex"], errors="coerce")
    df = df.dropna(subset=["WindowIndex"]).copy()
    df["WindowIndex"] = df["WindowIndex"].astype(int)

    if max_during_windows is not None:
        df = df[df["WindowIndex"] < max_during_windows].copy()
    return df


# ============================================================
# FEATURE SELECTION
# ============================================================


def prepare_bandpower_values(
    df: pd.DataFrame,
    condition: str,
    spec: FeatureSpec,
) -> pd.DataFrame:
    d = df[
        (df["Condition"] == condition)
        & (df["Band"] == spec.band)
        & (df["Metric"] == spec.metric)
        & (df["BandScheme"] == spec.band_scheme)
    ].copy()

    if d.empty:
        return d

    if spec.metric == "absolute" and spec.log10_absolute:
        d["PlotValue"] = safe_log10(d["Value"])
    else:
        d["PlotValue"] = pd.to_numeric(d["Value"], errors="coerce")

    return d


def prepare_fooof_values(
    df: pd.DataFrame,
    condition: str,
    spec: FeatureSpec,
) -> pd.DataFrame:
    feature = str(spec.fooof_feature)
    if feature not in df.columns:
        raise ValueError(
            f"FOOOF feature {feature!r} not found. Available columns: {list(df.columns)}"
        )

    d = df[df["Condition"] == condition].copy()
    if d.empty:
        return d

    if feature not in APERIODIC_OR_MODEL_FEATURES:
        if "PeakBand" in d.columns and spec.peak_band:
            d = d[d["PeakBand"] == spec.peak_band].copy()
    else:
        # Aperiodic/model features are repeated once per PeakBand in the long
        # FOOOF table. Average duplicates after grouping so each ROI contributes
        # once per subject/condition/phase/window.
        pass

    if d.empty:
        return d

    d["PlotValue"] = pd.to_numeric(d[feature], errors="coerce")

    group_cols = ["Subject", "Condition", "Phase", "ROI"]
    if "WindowIndex" in d.columns:
        group_cols.extend(["WindowIndex", "WindowStartSec", "WindowEndSec"])

    d = (
        d.groupby(group_cols, as_index=False)["PlotValue"]
        .mean()
        .sort_values(group_cols)
    )
    return d


def prepare_values(
    df: pd.DataFrame,
    condition: str,
    spec: FeatureSpec,
) -> pd.DataFrame:
    if spec.feature_family.startswith("bandpower"):
        return prepare_bandpower_values(df, condition, spec)
    if spec.feature_family == "fooof":
        return prepare_fooof_values(df, condition, spec)
    raise ValueError(f"Unknown feature family: {spec.feature_family}")


# ============================================================
# MNE LABEL/STC HELPERS
# ============================================================


def load_schaefer_labels(fsaverage5_dir: Path):
    mne_mod = get_mne()
    subject = fsaverage5_dir.name
    subjects_dir = fsaverage5_dir.parent
    label_dir = fsaverage5_dir / "label"
    surf_dir = fsaverage5_dir / "surf"

    if not label_dir.exists():
        raise FileNotFoundError(f"Missing label directory: {label_dir}")
    if not surf_dir.exists():
        raise FileNotFoundError(f"Missing surf directory: {surf_dir}")

    annot_files = sorted(label_dir.glob("*Schaefer*200*annot"))
    if not annot_files:
        raise FileNotFoundError(f"No Schaefer-200 annot files found in {label_dir}")

    lh_files = [f for f in annot_files if f.name.startswith("lh.")]
    if not lh_files:
        raise FileNotFoundError("Could not find lh.*Schaefer*200*.annot file.")

    parc = lh_files[0].name.replace("lh.", "").replace(".annot", "")
    labels_all = mne_mod.read_labels_from_annot(
        subject=subject,
        parc=parc,
        subjects_dir=subjects_dir,
        hemi="both",
        sort=True,
    )
    labels = [lab for lab in labels_all if is_schaefer_brain_label(lab.name)]
    removed = [lab.name for lab in labels_all if not is_schaefer_brain_label(lab.name)]

    print(f"Parcellation: {parc}")
    print(f"Labels before filtering: {len(labels_all)}")
    print(f"Labels after filtering:  {len(labels)}")
    print("Removed labels:", removed)
    if len(labels) != 200:
        print(f"[WARNING] Expected 200 Schaefer parcels, got {len(labels)}.")

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
    for candidate in candidates:
        if candidate in label_lookup:
            return label_lookup[candidate]
    return None


def build_source_space(fsaverage5_dir: Path):
    mne_mod = get_mne()
    subject = fsaverage5_dir.name
    subjects_dir = fsaverage5_dir.parent
    src = mne_mod.setup_source_space(
        subject=subject,
        spacing="all",
        subjects_dir=subjects_dir,
        add_dist=False,
        verbose=False,
    )
    lh_vertno = src[0]["vertno"]
    rh_vertno = src[1]["vertno"]
    print("LH vertices:", len(lh_vertno))
    print("RH vertices:", len(rh_vertno))
    return src, lh_vertno, rh_vertno


def roi_values_to_stc(roi_values, label_lookup, lh_vertno, rh_vertno, subject: str):
    mne_mod = get_mne()
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

    if unmatched:
        print(f"Unmatched ROIs ignored: {len(unmatched)}")

    lh_data = np.nan_to_num(lh_data, nan=0.0)
    rh_data = np.nan_to_num(rh_data, nan=0.0)
    brain_data = np.vstack([lh_data, rh_data])

    return (
        mne_mod.SourceEstimate(
            data=brain_data,
            vertices=[lh_vertno, rh_vertno],
            tmin=0,
            tstep=1,
            subject=subject,
        ),
        matched,
        unmatched,
    )


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
    return dict(kind="value", lims=[float(vmin), float(vmid), float(vmax)]), vmin, vmax


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
        ],
    )
    return clim, -vmax, vmax


def render_view(stc, hemi, view, clim, colormap, fsaverage5_dir: Path):
    subject = fsaverage5_dir.name
    subjects_dir = fsaverage5_dir.parent
    brain = stc.plot(
        subject=subject,
        subjects_dir=subjects_dir,
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
    """Two-sided paired sign-flip permutation test of mean(diff) against zero."""
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
    t_stat, _ = ttest_1samp(diff, popmean=0.0)
    return t_stat, p


def fdrcorrection_bh(pvals, alpha=0.05):
    """Benjamini-Hochberg FDR correction compatible with statsmodels output."""
    pvals = np.asarray(pvals, dtype=float)
    qvals = np.full(pvals.shape, np.nan, dtype=float)
    reject = np.full(pvals.shape, False, dtype=bool)

    valid = np.isfinite(pvals)
    p = pvals[valid]
    if len(p) == 0:
        return reject, qvals

    order = np.argsort(p)
    ranked = p[order]
    n = len(ranked)
    adjusted_ranked = ranked * n / (np.arange(n) + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted_ranked = np.clip(adjusted_ranked, 0, 1)

    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    qvals[valid] = adjusted
    reject[valid] = adjusted <= alpha
    return reject, qvals


def run_roi_stats_for_all_maps(
    diff_long: pd.DataFrame,
    n_perm: int = 5000,
    alpha_fdr: float = 0.05,
    random_state: int = 42,
    fdr_scope: str = "per_map",
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    results = []

    if diff_long.empty:
        return pd.DataFrame()

    for (map_order, map_name), map_df in diff_long.groupby(
        ["MapOrder", "MapName"], sort=True
    ):
        map_label = map_df["MapLabel"].iloc[0] if "MapLabel" in map_df.columns else map_name
        map_id = map_df["MapID"].iloc[0] if "MapID" in map_df.columns else ""
        for roi, roi_df in map_df.groupby("ROI"):
            diff = roi_df["Diff"].to_numpy(dtype=float)
            diff = diff[np.isfinite(diff)]
            t_stat, p_val = paired_sign_flip_pvalue(diff, n_perm=n_perm, rng=rng)
            results.append(
                {
                    "MapOrder": int(map_order),
                    "MapName": map_name,
                    "MapLabel": map_label,
                    "MapID": map_id,
                    "ROI": roi,
                    "N": int(len(diff)),
                    "MeanDiff": float(np.nanmean(diff)) if len(diff) > 0 else np.nan,
                    "T": t_stat,
                    "P_uncorrected": p_val,
                }
            )

    res = pd.DataFrame(results)
    res["P_FDR"] = np.nan
    res["Significant_FDR"] = False
    if res.empty:
        return res

    if fdr_scope == "per_map":
        for map_name, idx in res.groupby("MapName").groups.items():
            idx = list(idx)
            valid = res.loc[idx, "P_uncorrected"].notna()
            if valid.sum() == 0:
                continue
            idx_valid = res.loc[idx].loc[valid].index
            reject, p_fdr = fdrcorrection_bh(
                res.loc[idx_valid, "P_uncorrected"].values,
                alpha=alpha_fdr,
            )
            res.loc[idx_valid, "P_FDR"] = p_fdr
            res.loc[idx_valid, "Significant_FDR"] = reject
    elif fdr_scope == "global":
        valid = res["P_uncorrected"].notna()
        if valid.sum() > 0:
            reject, p_fdr = fdrcorrection_bh(
                res.loc[valid, "P_uncorrected"].values,
                alpha=alpha_fdr,
            )
            res.loc[valid, "P_FDR"] = p_fdr
            res.loc[valid, "Significant_FDR"] = reject
    else:
        raise ValueError("fdr_scope must be 'per_map' or 'global'.")

    res["Significant_FDR"] = res["Significant_FDR"].fillna(False)
    return res.sort_values(["MapOrder", "P_uncorrected", "ROI"]).reset_index(drop=True)


# ============================================================
# MAP COMPUTATION
# ============================================================


def window_label(window_index: int) -> str:
    return DISPLAY_TIME_LABELS.get(window_index, f"Dur{(window_index + 1) * 5}")


def compute_maps_and_stats_for_condition(
    df_whole: pd.DataFrame,
    df_during: pd.DataFrame,
    condition: str,
    spec: FeatureSpec,
    n_perm: int,
    alpha_fdr: float,
    random_state: int,
    fdr_scope: str,
):
    whole = prepare_values(df_whole, condition, spec)
    during = prepare_values(df_during, condition, spec)

    if whole.empty or during.empty:
        return None, None

    whole_grouped = (
        whole.groupby(["Subject", "Condition", "Phase", "ROI"], as_index=False)[
            "PlotValue"
        ]
        .mean()
        .dropna(subset=["PlotValue"])
    )
    pre = whole_grouped[whole_grouped["Phase"] == "PRE"].copy()
    post = whole_grouped[whole_grouped["Phase"] == "POST"].copy()
    if pre.empty or post.empty:
        return None, None

    pre_subject = pre.rename(columns={"PlotValue": "PRE"})[["Subject", "ROI", "PRE"]]
    pre_group_map = pre_subject.groupby("ROI")["PRE"].mean().sort_index()

    maps_to_plot = [
        {
            "name": "PRE",
            "map_id": "PRE",
            "label": "PRE",
            "subtitle": "",
            "values": pre_group_map,
            "type": "pre",
            "n_subjects": pre_subject["Subject"].nunique(),
        }
    ]

    diff_rows = []
    map_order = 1

    during_grouped = (
        during.groupby(["Subject", "Condition", "WindowIndex", "ROI"], as_index=False)[
            "PlotValue"
        ]
        .mean()
        .dropna(subset=["PlotValue"])
    )

    window_info = (
        during.groupby("WindowIndex", as_index=False)
        .agg(
            WindowStartSec=("WindowStartSec", "median"),
            WindowEndSec=("WindowEndSec", "median"),
            NSubjectsInWindow=("Subject", "nunique"),
        )
        .sort_values("WindowIndex")
    )

    for _, win in window_info.iterrows():
        window_idx = int(win["WindowIndex"])
        wstart = float(win["WindowStartSec"])
        wend = float(win["WindowEndSec"])

        dwin = during_grouped[during_grouped["WindowIndex"] == window_idx].copy()
        dwin = dwin.rename(columns={"PlotValue": "DURING"})[
            ["Subject", "ROI", "DURING"]
        ]
        merged = dwin.merge(pre_subject, on=["Subject", "ROI"], how="inner")
        if merged.empty:
            continue

        merged["Diff"] = merged["DURING"] - merged["PRE"]
        legacy_num = window_idx + 1
        map_name = f"DURING {legacy_num} - PRE"
        map_id = f"DURING{legacy_num}_minus_PRE"
        map_label = f"{window_label(window_idx)} - PRE"
        diff_map = merged.groupby("ROI")["Diff"].mean().sort_index()

        maps_to_plot.append(
            {
                "name": map_name,
                "map_id": map_id,
                "label": map_label,
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
                    "MapLabel": map_label,
                    "MapID": map_id,
                    "WindowIndex": window_idx,
                    "WindowStartSec": wstart,
                    "WindowEndSec": wend,
                    "Subject": row["Subject"],
                    "SubjectLabel": source_subject_label(row["Subject"]),
                    "ROI": row["ROI"],
                    "Diff": row["Diff"],
                }
            )
        map_order += 1

    post_subject = post.rename(columns={"PlotValue": "POST"})[["Subject", "ROI", "POST"]]
    post_merged = post_subject.merge(pre_subject, on=["Subject", "ROI"], how="inner")
    if not post_merged.empty:
        post_merged["Diff"] = post_merged["POST"] - post_merged["PRE"]
        post_diff_map = post_merged.groupby("ROI")["Diff"].mean().sort_index()
        maps_to_plot.append(
            {
                "name": "POST - PRE",
                "map_id": "POST_minus_PRE",
                "label": "Post - PRE",
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
                    "MapLabel": "Post - PRE",
                    "MapID": "POST_minus_PRE",
                    "WindowIndex": np.nan,
                    "WindowStartSec": np.nan,
                    "WindowEndSec": np.nan,
                    "Subject": row["Subject"],
                    "SubjectLabel": source_subject_label(row["Subject"]),
                    "ROI": row["ROI"],
                    "Diff": row["Diff"],
                }
            )

    diff_long = pd.DataFrame(diff_rows)
    stats = run_roi_stats_for_all_maps(
        diff_long=diff_long,
        n_perm=n_perm,
        alpha_fdr=alpha_fdr,
        random_state=random_state,
        fdr_scope=fdr_scope,
    )

    if not stats.empty:
        stats["FeatureFamily"] = spec.feature_family
        stats["Band"] = spec.band
        stats["Metric"] = spec.metric
        stats["FOOOFFeature"] = spec.fooof_feature
        stats["PeakBand"] = spec.peak_band
        stats["BandScheme"] = spec.band_scheme
        stats["Condition"] = condition
        stats["ConditionLabel"] = CONDITION_LABELS.get(condition, condition)

    return maps_to_plot, stats


# ============================================================
# OUTPUT NAMING
# ============================================================


def default_output_dirs(args, spec: FeatureSpec):
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif spec.feature_family == "bandpower_standard":
        output_dir = (
            SOURCE_RESULTS_ROOT
            / "figures"
            / f"source_pial_interp_during_5min_trajectory_fdr_{spec.metric}"
        )
    elif spec.feature_family == "bandpower_zapline":
        output_dir = (
            SOURCE_RESULTS_ROOT
            / "figures"
            / "zapline_40_source_pial_during_5min_trajectory_fdr"
        )
    elif spec.feature_family == "fooof":
        output_dir = (
            SOURCE_RESULTS_ROOT
            / "figures"
            / "source_pial_fooof_during_5min_trajectory_fdr"
        )
    else:
        raise ValueError(f"Unknown feature family: {spec.feature_family}")

    if args.stats_dir:
        stats_dir = Path(args.stats_dir)
    elif spec.feature_family == "bandpower_standard":
        stats_dir = (
            SOURCE_RESULTS_ROOT
            / f"stats_roi_band_power_during_5min_interp_{spec.metric}"
        )
    elif spec.feature_family == "bandpower_zapline":
        stats_dir = SOURCE_RESULTS_ROOT / "stats" / "zapline_40_roi_band_power_during_5min"
    elif spec.feature_family == "fooof":
        stats_dir = SOURCE_RESULTS_ROOT / "stats_fooof_during_5min_source_maps"
    else:
        raise ValueError(f"Unknown feature family: {spec.feature_family}")

    return output_dir, stats_dir


# Manuscript figure each band-power metric belongs to. FOOOF maps are not a
# manuscript figure (Figure S4 is drawn by the scripts in ../Supplementary).
PAPER_FIGURE_IDS = {
    "relative": "Figure3",
    "absolute": "FigureS3",
}


def paper_figure_id(spec: FeatureSpec) -> str:
    if spec.feature_family.startswith("bandpower"):
        return PAPER_FIGURE_IDS.get(spec.metric, "")
    return ""


def paper_figure_prefix(spec: FeatureSpec, lower: bool = False) -> str:
    fig_id = paper_figure_id(spec)
    if not fig_id:
        return ""
    if lower:
        fig_id = "f" + fig_id[1:]
    return f"{fig_id}_"


def figure_file_name(
    condition: str,
    spec: FeatureSpec,
    significant_only: bool,
    alpha_fdr: float,
    fdr_scope: str,
) -> str:
    condition_label = CONDITION_LABELS.get(condition, condition)
    if significant_only:
        suffix = f"FDRsig_only_alpha{alpha_fdr}_{fdr_scope}"
    else:
        suffix = "all_rois"

    if spec.feature_family == "bandpower_standard":
        return (
            f"{paper_figure_prefix(spec)}group_source_{SURFACE}_{condition}_{condition_label}_"
            f"{spec.band}_{spec.metric}_{spec.band_scheme}_"
            f"trajectory_rows_views_columns_{suffix}.png"
        )

    if spec.feature_family == "bandpower_zapline":
        return (
            f"{paper_figure_prefix(spec)}group_source_zapline40_{SURFACE}_{condition}_{condition_label}_"
            f"{spec.band}_{spec.metric}_{spec.band_scheme}_"
            f"trajectory_rows_views_columns_{suffix}.png"
        )

    if spec.feature_family == "fooof":
        tokens = "_".join(spec.output_tokens)
        return (
            f"group_source_fooof_{SURFACE}_{condition}_{condition_label}_"
            f"{tokens}_trajectory_rows_views_columns_{suffix}.png"
        )

    raise ValueError(f"Unknown feature family: {spec.feature_family}")


def stats_file_name(condition: str, spec: FeatureSpec, fdr_scope: str) -> str:
    condition_label = CONDITION_LABELS.get(condition, condition)
    if spec.feature_family.startswith("bandpower"):
        return (
            f"{paper_figure_prefix(spec, lower=True)}"
            f"{condition}_{condition_label}_{spec.band}_{spec.metric}_{spec.band_scheme}_"
            f"during5min_POSTminusPRE_roi_stats_{fdr_scope}FDR.csv"
        )

    if spec.feature_family == "fooof":
        tokens = "_".join(spec.output_tokens)
        return (
            f"{condition}_{condition_label}_{tokens}_"
            f"during5min_POSTminusPRE_roi_stats_{fdr_scope}FDR.csv"
        )

    raise ValueError(f"Unknown feature family: {spec.feature_family}")


# ============================================================
# PLOTTING
# ============================================================


def compact_row_label(item):
    if item["type"] == "pre":
        return f"PRE\nN={item['n_subjects']}"

    label = item.get("label", item["name"])
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
    spec: FeatureSpec,
    output_dir: Path,
    fsaverage5_dir: Path,
    significant_only: bool = False,
    threshold_percentile_pre=None,
    threshold_percentile_contrast=None,
    alpha_fdr: float = 0.05,
    fdr_scope: str = "per_map",
    colorbar_fontsize: int = 25,
    colorbar_ticksize: int = 25,
):
    condition_label = CONDITION_LABELS.get(condition, condition)
    n_maps = len(maps_to_plot)
    if n_maps < 2:
        print(f"Not enough maps for {condition} | {spec.display_name}, skipping.")
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

    if not contrast_vals_all:
        print(f"No contrast values for {condition} | {spec.display_name}, skipping.")
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

    for col_idx, title in enumerate(
        ["Left", "Right", "Left medial", "Right medial"],
        start=1,
    ):
        ax_title = fig.add_subplot(gs[0, col_idx])
        ax_title.set_title(title, fontsize=11, pad=10)
        ax_title.axis("off")

    subject = fsaverage5_dir.name
    for map_idx, item in enumerate(maps_to_plot):
        roi_values = item["values"].copy()

        if item["type"] == "pre":
            roi_values = apply_threshold(roi_values, threshold_percentile_pre)
            clim = pre_clim
            cmap = PRE_COLORMAP
        else:
            if significant_only:
                sig_rois = sig_rois_by_map.get(item["name"], set())
                roi_values = roi_values.where(
                    roi_values.index.astype(str).isin(sig_rois),
                    0.0,
                )
            roi_values = apply_threshold(roi_values, threshold_percentile_contrast)
            clim = contrast_clim
            cmap = CONTRAST_COLORMAP

        stc, matched, _unmatched = roi_values_to_stc(
            roi_values=roi_values,
            label_lookup=label_lookup,
            lh_vertno=lh_vertno,
            rh_vertno=rh_vertno,
            subject=subject,
        )

        sig_text = ""
        if item["type"] == "contrast" and significant_only:
            n_sig = len(sig_rois_by_map.get(item["name"], set()))
            sig_text = f" | Sig={n_sig}"

        print(
            f"{condition} | {spec.display_name} | {item['name']}: "
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

        for view_col, (hemi, view, _view_name) in enumerate(VIEW_COLUMNS, start=1):
            img = render_view(
                stc=stc,
                hemi=hemi,
                view=view,
                clim=clim,
                colormap=cmap,
                fsaverage5_dir=fsaverage5_dir,
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
    cbar_pre.set_label(spec.colorbar_pre_label, fontsize=colorbar_fontsize)
    cbar_pre.ax.tick_params(labelsize=colorbar_ticksize)

    cax_contrast = fig.add_subplot(gs[1:n_maps, 5])
    sm_contrast = mpl.cm.ScalarMappable(
        norm=mpl.colors.Normalize(vmin=contrast_vmin, vmax=contrast_vmax),
        cmap=CONTRAST_COLORMAP,
    )
    sm_contrast.set_array([])
    cbar_contrast = fig.colorbar(sm_contrast, cax=cax_contrast, orientation="vertical")
    cbar_contrast.set_label(spec.colorbar_contrast_label, fontsize=colorbar_fontsize)
    cbar_contrast.ax.tick_params(labelsize=colorbar_ticksize)

    if significant_only:
        subtitle = f"FDR-significant ROIs only for contrasts | {fdr_scope} FDR"
    else:
        subtitle = "All ROIs"

    fig.suptitle(
        f"{condition} ({condition_label}) | {spec.display_name}\n"
        f"PRE, DURING 5-min windows relative to PRE, and POST - PRE | "
        f"{subtitle}",
        fontsize=12,
        y=0.985,
    )

    out_name = figure_file_name(
        condition=condition,
        spec=spec,
        significant_only=significant_only,
        alpha_fdr=alpha_fdr,
        fdr_scope=fdr_scope,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / out_name
    fig.savefig(out_path, dpi=DPI_SAVE, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Successfully saved figure: {out_path}")


# ============================================================
# CLI
# ============================================================


def make_specs(args) -> list[FeatureSpec]:
    specs: list[FeatureSpec] = []
    if args.feature_family.startswith("bandpower"):
        for band in args.bands:
            for metric in args.metrics:
                specs.append(
                    FeatureSpec(
                        feature_family=args.feature_family,
                        band=band,
                        metric=metric,
                        band_scheme=args.band_scheme,
                        log10_absolute=args.log10_absolute,
                    )
                )
        return specs

    if args.feature_family == "fooof":
        for feature in args.fooof_features:
            if feature in APERIODIC_OR_MODEL_FEATURES:
                specs.append(
                    FeatureSpec(
                        feature_family="fooof",
                        fooof_feature=feature,
                        peak_band=None,
                    )
                )
            else:
                for peak_band in args.peak_bands:
                    specs.append(
                        FeatureSpec(
                            feature_family="fooof",
                            fooof_feature=feature,
                            peak_band=peak_band,
                        )
                    )
        return specs

    raise ValueError(f"Unknown feature family: {args.feature_family}")


def load_inputs(args):
    if args.feature_family == "bandpower_standard":
        whole, during = load_standard_bandpower(
            prepost_dir=Path(args.standard_prepost_dir),
            during_dir=Path(args.standard_during_dir),
        )
    elif args.feature_family == "bandpower_zapline":
        whole, during = load_zapline_bandpower(
            prepost_csv=Path(args.zapline_prepost_csv),
            during_csv=Path(args.zapline_during_csv),
        )
    elif args.feature_family == "fooof":
        whole, during = load_fooof(
            prepost_csv=Path(args.fooof_prepost_csv),
            during_csv=Path(args.fooof_during_csv),
        )
    else:
        raise ValueError(f"Unknown feature family: {args.feature_family}")

    during = filter_during_windows(during, args.max_during_windows)
    print("Whole rows:", len(whole))
    print("DURING 5-min rows:", len(during))
    print("Whole subjects:", whole["Subject"].nunique())
    print("DURING subjects:", during["Subject"].nunique())
    print("Conditions:", sorted(set(whole["Condition"]) & set(during["Condition"])))
    return whole, during


def make_parser():
    parser = argparse.ArgumentParser(
        description="Generate source-space PRE-referenced FDR brain maps."
    )
    parser.add_argument(
        "--feature-family",
        choices=["bandpower_standard", "bandpower_zapline", "fooof"],
        default="bandpower_standard",
    )
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS)

    parser.add_argument("--bands", nargs="+", default=DEFAULT_BANDS)
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    parser.add_argument("--band-scheme", default=DEFAULT_BAND_SCHEME)
    parser.add_argument(
        "--log10-absolute",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use log10 values for absolute band-power contrasts.",
    )

    parser.add_argument("--fooof-features", nargs="+", default=DEFAULT_FOOOF_FEATURES)
    parser.add_argument("--peak-bands", nargs="+", default=DEFAULT_PEAK_BANDS)

    parser.add_argument("--standard-prepost-dir", default=str(DEFAULT_STANDARD_PREPOST_DIR))
    parser.add_argument("--standard-during-dir", default=str(DEFAULT_STANDARD_DURING_DIR))
    parser.add_argument("--zapline-prepost-csv", default=str(DEFAULT_ZAPLINE_PREPOST_CSV))
    parser.add_argument("--zapline-during-csv", default=str(DEFAULT_ZAPLINE_DURING_CSV))
    parser.add_argument("--fooof-prepost-csv", default=str(DEFAULT_FOOOF_PREPOST_CSV))
    parser.add_argument("--fooof-during-csv", default=str(DEFAULT_FOOOF_DURING_CSV))

    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--stats-dir", default=None)
    parser.add_argument("--fsaverage5-dir", default=str(DEFAULT_FSAVERAGE5_DIR))

    parser.add_argument("--max-during-windows", type=int, default=DEFAULT_MAX_DURING_WINDOWS)
    parser.add_argument("--n-perm", type=int, default=5000)
    parser.add_argument("--alpha-fdr", type=float, default=0.05)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--fdr-scope", choices=["per_map", "global"], default="per_map")

    parser.add_argument("--save-all-rois", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-fdr-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-render", action="store_true", help="Write stats only.")
    parser.add_argument("--backend", default="pyvistaqt")
    parser.add_argument("--threshold-percentile-pre", type=float, default=None)
    parser.add_argument("--threshold-percentile-contrast", type=float, default=None)
    parser.add_argument("--colorbar-fontsize", type=int, default=25)
    parser.add_argument("--colorbar-ticksize", type=int, default=25)
    return parser


def main():
    args = make_parser().parse_args()
    conditions = [clean_condition(c) for c in args.conditions]
    specs = make_specs(args)

    whole, during = load_inputs(args)

    fsaverage5_dir = Path(args.fsaverage5_dir)
    labels = label_lookup = lh_vertno = rh_vertno = None
    if not args.no_render:
        mne_mod = get_mne()
        mne_mod.viz.set_3d_backend(args.backend)
        print("MNE 3D backend:", mne_mod.viz.get_3d_backend())
        print("\nLoading Schaefer labels...")
        labels, _parc = load_schaefer_labels(fsaverage5_dir)
        label_lookup = make_label_lookup(labels)
        print("\nBuilding source space...")
        _src, lh_vertno, rh_vertno = build_source_space(fsaverage5_dir)

    for spec in specs:
        output_dir, stats_dir = default_output_dirs(args, spec)
        output_dir.mkdir(parents=True, exist_ok=True)
        stats_dir.mkdir(parents=True, exist_ok=True)

        for condition in conditions:
            print("\n" + "=" * 90)
            print(f"Processing: {condition} | {CONDITION_LABELS.get(condition, condition)} | {spec.display_name}")
            print("=" * 90)

            maps_to_plot, stats = compute_maps_and_stats_for_condition(
                df_whole=whole,
                df_during=during,
                condition=condition,
                spec=spec,
                n_perm=args.n_perm,
                alpha_fdr=args.alpha_fdr,
                random_state=args.random_state,
                fdr_scope=args.fdr_scope,
            )

            if maps_to_plot is None or stats is None:
                print("No data for this selection, skipping.")
                continue

            stats_file = stats_dir / stats_file_name(condition, spec, args.fdr_scope)
            stats.to_csv(stats_file, index=False)
            print(f"Successfully saved stats: {stats_file}")

            print("Maps to plot:")
            for m in maps_to_plot:
                print("  -", m["label"], m["subtitle"], "N=", m["n_subjects"])

            if args.no_render:
                continue

            assert label_lookup is not None
            assert lh_vertno is not None
            assert rh_vertno is not None

            if args.save_all_rois:
                plot_maps_trajectory_grid(
                    maps_to_plot=maps_to_plot,
                    stats=stats,
                    label_lookup=label_lookup,
                    lh_vertno=lh_vertno,
                    rh_vertno=rh_vertno,
                    condition=condition,
                    spec=spec,
                    output_dir=output_dir,
                    fsaverage5_dir=fsaverage5_dir,
                    significant_only=False,
                    threshold_percentile_pre=args.threshold_percentile_pre,
                    threshold_percentile_contrast=args.threshold_percentile_contrast,
                    alpha_fdr=args.alpha_fdr,
                    fdr_scope=args.fdr_scope,
                    colorbar_fontsize=args.colorbar_fontsize,
                    colorbar_ticksize=args.colorbar_ticksize,
                )

            if args.save_fdr_only:
                plot_maps_trajectory_grid(
                    maps_to_plot=maps_to_plot,
                    stats=stats,
                    label_lookup=label_lookup,
                    lh_vertno=lh_vertno,
                    rh_vertno=rh_vertno,
                    condition=condition,
                    spec=spec,
                    output_dir=output_dir,
                    fsaverage5_dir=fsaverage5_dir,
                    significant_only=True,
                    threshold_percentile_pre=args.threshold_percentile_pre,
                    threshold_percentile_contrast=args.threshold_percentile_contrast,
                    alpha_fdr=args.alpha_fdr,
                    fdr_scope=args.fdr_scope,
                    colorbar_fontsize=args.colorbar_fontsize,
                    colorbar_ticksize=args.colorbar_ticksize,
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
