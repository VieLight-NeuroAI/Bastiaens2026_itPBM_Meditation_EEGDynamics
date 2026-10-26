#!/usr/bin/env python
# ============================================================
# 03_compute_fooof_from_roi_timeseries.py
#
# Methods 3.6 (Alpha peak-frequency analysis): specparam/FOOOF fits per
# parcel, aperiodic offset/exponent, and flattened-spectrum alpha peak
# frequency (APF) and alpha peak power (APP).
# Step 3 of the source pipeline; reads the .npz files from 01.
# ============================================================
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from source_config import DEFAULTS, DEFAULT_FOOOF_ROOT
from source_utils import (
    add_common_source_args,
    add_fooof_args,
    compute_roi_psd,
    fit_fooof_roi_features,
    load_roi_ts_npz,
)


def fooof_for_roi_file(roi_file: Path, args):
    roi_ts, label_names, _times, sfreq, meta = load_roi_ts_npz(roi_file)
    psds_epoch, psds_mean, freqs = compute_roi_psd(
        roi_ts=roi_ts,
        sfreq=sfreq,
        epoch_duration=float(meta.get("epoch_duration", args.epoch_duration)),
        psd_fmin=args.psd_fmin,
        psd_fmax=args.psd_fmax,
    )

    subject = str(meta.get("subject_id"))
    condition = str(meta.get("condition"))
    phase = str(meta.get("phase"))
    base = {
        "Subject": subject,
        "Condition": condition,
        "Phase": phase,
        "SourceSpace": str(meta.get("source_space", "fsaverage5")),
        "Parcellation": str(meta.get("parcellation", "")),
        "InverseMethod": str(meta.get("method", args.method)),
        "Lambda2": float(meta.get("lambda2", args.lambda2)),
        "LabelMode": str(meta.get("label_mode", args.label_mode)),
        "InputFile": str(meta.get("input_file", "")),
        "ROITimeSeriesFile": str(roi_file),
        "PSD_fmin": args.psd_fmin,
        "PSD_fmax": args.psd_fmax,
        "EpochDuration": float(meta.get("epoch_duration", args.epoch_duration)),
        "EpochOverlap": float(meta.get("epoch_overlap", args.epoch_overlap)),
    }
    if phase == "DURING" and "window_index" in meta:
        base.update({
            "WindowIndex": int(meta.get("window_index")),
            "WindowLabel": f"{float(meta.get('window_start_sec'))/60:.0f}-{float(meta.get('window_end_sec'))/60:.0f}min",
            "WindowStartSec": float(meta.get("window_start_sec")),
            "WindowEndSec": float(meta.get("window_end_sec")),
            "WindowDurationSec": float(meta.get("window_duration_sec", float(meta.get("window_end_sec")) - float(meta.get("window_start_sec")))),
            "NWindowEpochs": int(roi_ts.shape[0]),
        })
    else:
        base.update({"NEpochs": int(roi_ts.shape[0])})

    features, qc = fit_fooof_roi_features(
        psds_mean=psds_mean,
        freqs=freqs,
        label_names=label_names,
        peak_bands=args.peak_bands,
        args=args,
        base_metadata=base,
    )
    qc["n_epochs"] = int(roi_ts.shape[0])
    qc["roi_ts_shape"] = str(tuple(roi_ts.shape))
    qc["psd_shape"] = str(tuple(psds_epoch.shape))
    qc["sfreq"] = float(sfreq)
    qc["n_rois"] = len(label_names)
    return features, qc, meta


def write_prepost_fooof(roi_file: Path, fooof_root: Path, args):
    features, qc, meta = fooof_for_roi_file(roi_file, args)
    subject = str(meta.get("subject_id"))
    condition = str(meta.get("condition"))
    phase = str(meta.get("phase"))
    method = str(meta.get("method", args.method))
    label_mode = str(meta.get("label_mode", args.label_mode))

    feature_dir = fooof_root / "roi_fooof_features"
    qc_dir = fooof_root / "roi_fooof_qc"
    feature_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)
    feature_file = feature_dir / (
        f"MED_PBM_{subject}_{condition}_{phase}_"
        f"source_roi_fooof_features_long_schaefer200_fsaverage5_{method}_{label_mode}.csv"
    )
    qc_file = qc_dir / (
        f"MED_PBM_{subject}_{condition}_{phase}_"
        f"source_roi_fooof_qc_schaefer200_fsaverage5_{method}_{label_mode}.csv"
    )
    features.to_csv(feature_file, index=False)
    qc.to_csv(qc_file, index=False)
    return feature_file, qc_file


def write_during_fooof(group_key, roi_files: list[Path], fooof_root: Path, args):
    subject, condition, phase = group_key
    feature_dfs, qc_dfs = [], []
    method = args.method
    label_mode = args.label_mode
    for roi_file in roi_files:
        features, qc, meta = fooof_for_roi_file(roi_file, args)
        method = str(meta.get("method", method))
        label_mode = str(meta.get("label_mode", label_mode))
        feature_dfs.append(features)
        qc_dfs.append(qc)

    features = pd.concat(feature_dfs, ignore_index=True)
    qc = pd.concat(qc_dfs, ignore_index=True)
    feature_dir = fooof_root / "during_5min" / "during_5min_roi_fooof_features"
    qc_dir = fooof_root / "during_5min" / "during_5min_roi_fooof_qc"
    feature_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)
    feature_file = feature_dir / (
        f"MED_PBM_{subject}_{condition}_{phase}_"
        f"source_roi_fooof_5min_long_schaefer200_fsaverage5_{method}_{label_mode}.csv"
    )
    qc_file = qc_dir / (
        f"MED_PBM_{subject}_{condition}_{phase}_"
        f"source_roi_fooof_5min_qc_schaefer200_fsaverage5_{method}_{label_mode}.csv"
    )
    features.to_csv(feature_file, index=False)
    qc.to_csv(qc_file, index=False)
    return feature_file, qc_file


def make_parser():
    parser = argparse.ArgumentParser(description="Compute source ROI FOOOF/APF features from cached ROI time-series files.")
    parser.add_argument("--roi-output-root", required=True, help="Root containing roi_timeseries and/or during_5min_roi_timeseries.")
    parser.add_argument("--fooof-output-root", default=str(DEFAULT_FOOOF_ROOT), help="Root where FOOOF feature CSVs will be written.")
    parser.add_argument("--scope", choices=["all", "prepost", "during"], default="all")
    parser.add_argument("--max-during-windows", type=int, default=DEFAULTS["max_during_windows"])
    add_common_source_args(parser)
    add_fooof_args(parser)
    return parser


def main():
    args = make_parser().parse_args()
    roi_root = Path(args.roi_output_root)
    fooof_root = Path(args.fooof_output_root)
    written = []

    if args.scope in {"all", "prepost"}:
        for roi_file in sorted((roi_root / "roi_timeseries").glob("*_roi_ts_schaefer200_fsaverage5_*.npz")):
            _roi_ts, _labels, _times, _sfreq, meta = load_roi_ts_npz(roi_file)
            if str(meta.get("phase")) == "DURING":
                continue
            written.append(write_prepost_fooof(roi_file, fooof_root, args)[0])

    if args.scope in {"all", "during"}:
        groups = {}
        for roi_file in sorted((roi_root / "during_5min_roi_timeseries").glob("*_roi_ts_schaefer200_fsaverage5_*.npz")):
            _roi_ts, _labels, _times, _sfreq, meta = load_roi_ts_npz(roi_file)
            if args.max_during_windows is not None and int(meta.get("window_index")) >= args.max_during_windows:
                continue
            key = (str(meta.get("subject_id")), str(meta.get("condition")), str(meta.get("phase")))
            groups.setdefault(key, []).append(roi_file)
        for key, files in groups.items():
            written.append(write_during_fooof(key, files, fooof_root, args)[0])

    manifest = pd.DataFrame({"FOOOFFeatureFile": [str(f) for f in written]})
    manifest_file = fooof_root / "fooof_feature_manifest.csv"
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_file, index=False)
    print(f"Saved {len(written)} FOOOF feature files.")
    print(f"Manifest: {manifest_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

