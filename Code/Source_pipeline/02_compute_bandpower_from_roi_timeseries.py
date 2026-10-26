#!/usr/bin/env python
# ============================================================
# 02_compute_bandpower_from_roi_timeseries.py
#
# Methods 3.5 (Source-level power spectra): Welch PSD per 8-s epoch over
# 1-50 Hz, trapezoidal band integration, absolute and relative band power.
# Step 2 of the source pipeline; reads the .npz files from 01.
# ============================================================
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from source_config import DEFAULTS
from source_utils import add_common_source_args, build_bandpower_rows, load_roi_ts_npz


def write_prepost_feature(roi_file: Path, output_root: Path, args):
    _roi_ts, _labels, _times, _sfreq, meta = load_roi_ts_npz(roi_file)
    subject = str(meta.get("subject_id"))
    condition = str(meta.get("condition"))
    phase = str(meta.get("phase"))
    method = str(meta.get("method", args.method))
    label_mode = str(meta.get("label_mode", args.label_mode))
    df = build_bandpower_rows(roi_file, args)
    out_dir = output_root / "roi_band_power_features"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / (
        f"MED_PBM_{subject}_{condition}_{phase}_"
        f"source_roi_band_power_long_schaefer200_fsaverage5_{method}_{label_mode}.csv"
    )
    df.to_csv(out_file, index=False)
    return out_file


def write_during_feature(group_key, roi_files: list[Path], output_root: Path, args):
    subject, condition, phase = group_key
    _roi_ts, _labels, _times, _sfreq, meta = load_roi_ts_npz(roi_files[0])
    method = str(meta.get("method", args.method))
    label_mode = str(meta.get("label_mode", args.label_mode))
    dfs = [build_bandpower_rows(f, args) for f in roi_files]
    df = pd.concat(dfs, ignore_index=True)
    out_dir = output_root / "during_5min_roi_band_power_features"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / (
        f"MED_PBM_{subject}_{condition}_{phase}_"
        f"source_roi_band_power_5min_long_schaefer200_fsaverage5_{method}_{label_mode}.csv"
    )
    df.to_csv(out_file, index=False)
    return out_file


def make_parser():
    parser = argparse.ArgumentParser(description="Compute source ROI band-power CSVs from cached ROI time-series files.")
    parser.add_argument("--output-root", required=True, help="Root containing roi_timeseries and/or during_5min_roi_timeseries.")
    parser.add_argument("--scope", choices=["all", "prepost", "during"], default="all")
    parser.add_argument("--max-during-windows", type=int, default=DEFAULTS["max_during_windows"])
    add_common_source_args(parser)
    return parser


def main():
    args = make_parser().parse_args()
    output_root = Path(args.output_root)
    written = []

    if args.scope in {"all", "prepost"}:
        for roi_file in sorted((output_root / "roi_timeseries").glob("*_roi_ts_schaefer200_fsaverage5_*.npz")):
            _roi_ts, _labels, _times, _sfreq, meta = load_roi_ts_npz(roi_file)
            if str(meta.get("phase")) == "DURING":
                continue
            written.append(write_prepost_feature(roi_file, output_root, args))

    if args.scope in {"all", "during"}:
        groups = {}
        for roi_file in sorted((output_root / "during_5min_roi_timeseries").glob("*_roi_ts_schaefer200_fsaverage5_*.npz")):
            _roi_ts, _labels, _times, _sfreq, meta = load_roi_ts_npz(roi_file)
            if args.max_during_windows is not None and int(meta.get("window_index")) >= args.max_during_windows:
                continue
            key = (str(meta.get("subject_id")), str(meta.get("condition")), str(meta.get("phase")))
            groups.setdefault(key, []).append(roi_file)
        for key, files in groups.items():
            written.append(write_during_feature(key, files, output_root, args))

    manifest = pd.DataFrame({"BandPowerFile": [str(f) for f in written]})
    manifest_file = output_root / "bandpower_feature_manifest.csv"
    manifest.to_csv(manifest_file, index=False)
    print(f"Saved {len(written)} feature files.")
    print(f"Manifest: {manifest_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
