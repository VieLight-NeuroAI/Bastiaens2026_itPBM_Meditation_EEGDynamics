# ============================================================
# source_config.py
#
# Shared band definitions (Methods 3.5), default paths, and analysis defaults
# for the source pipeline.
# ============================================================

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "Data"
OUTPUT_SOURCE_RESULTS_ROOT = PROJECT_ROOT / "outputs" / "source_results"


BAND_SCHEMES = {
    "theta4_7_alpha7_13": {
        "Delta": (1.0, 4.0),
        "Theta": (4.0, 7.0),
        "Alpha": (7.0, 13.0),
        "Beta": (13.0, 30.0),
        "Gamma": (30.0, 50.0),
    },
    "theta4_8_alpha8_13": {
        "Delta": (1.0, 4.0),
        "Theta": (4.0, 8.0),
        "Alpha": (8.0, 13.0),
        "Beta": (13.0, 30.0),
        "Gamma": (30.0, 50.0),
    },
}

PEAK_BANDS = {
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta": (13.0, 30.0),
    "Gamma": (30.0, 50.0),
}

DEFAULTS = {
    "epoch_duration": 8.0,
    "epoch_overlap": 0.0,
    "psd_fmin": 1.0,
    "psd_fmax": 50.0,
    "expected_sfreq": 500.0,
    "window_duration": 300.0,
    "min_window_duration": 120.0,
    "montage": "standard_1020",
    "method": "dSPM",
    "lambda2": 1.0 / 9.0,
    "label_mode": "mean_flip",
    "pick_ori": "normal",
    "loose": 0.2,
    "depth": 0.8,
    "mindist": 5.0,
    "n_jobs_forward": 1,
    "max_during_windows": 4,
}

# Default output folders for the per-recording ROI time series and features.
DEFAULT_STANDARD_PREPOST_ROOT = Path(
    OUTPUT_SOURCE_RESULTS_ROOT / "cleaned_interp"
)
DEFAULT_STANDARD_DURING_ROOT = Path(
    OUTPUT_SOURCE_RESULTS_ROOT / "cleaned_interp_during_5min"
)
DEFAULT_ZAPLINE_PREPOST_ROOT = Path(
    OUTPUT_SOURCE_RESULTS_ROOT / "zapline_40"
)
DEFAULT_ZAPLINE_DURING_ROOT = Path(
    OUTPUT_SOURCE_RESULTS_ROOT / "zapline_40_during_5min"
)
DEFAULT_FOOOF_ROOT = Path(
    OUTPUT_SOURCE_RESULTS_ROOT / "fooof"
)
DEFAULT_FSAVERAGE5_DIR = Path(
    DATA_ROOT / "fsaverage5"
)
