#!/usr/bin/env python
# ============================================================
# 01_extract_source_roi_timeseries.py
#
# Methods 3.4 (Source reconstruction): fsaverage5 template forward model,
# dSPM inverse, Schaefer-200 parcels (mean_flip). Saves one ROI time-series
# .npz per recording for PRE/POST, and per 5-min DURING window (Dur5-Dur20).
# Step 1 of the source pipeline; outputs feed 02 and 03.
# ============================================================
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mne
import pandas as pd

from source_config import DEFAULTS, DEFAULT_FSAVERAGE5_DIR
from source_utils import (
    add_common_source_args,
    build_channel_template_from_files,
    build_labels_and_src,
    build_windows,
    discover_set_files,
    expected_roi_ts_file,
    expected_window_roi_ts_file,
    extract_roi_ts_from_epochs,
    fetch_fsaverage_bem,
    get_or_create_inverse,
    parse_med_pbm_filename,
    prepare_raw_for_source,
    read_subjects_file,
    save_roi_ts_npz,
)


def process_whole_recording(raw, inv, labels, label_names, parc, set_file, subject, condition, phase, cache_sig, channel_qc, args):
    epochs = mne.make_fixed_length_epochs(
        raw,
        duration=args.epoch_duration,
        overlap=args.epoch_overlap,
        preload=True,
        reject_by_annotation=True,
        verbose=False,
    ).copy().pick("eeg")
    if len(epochs) == 0:
        raise RuntimeError(f"No valid epochs for {set_file}")

    roi_ts = extract_roi_ts_from_epochs(epochs, inv, labels, args)
    out_file = expected_roi_ts_file(Path(args.output_root), subject, condition, phase, args)
    metadata = {
        "subject_id": subject,
        "condition": condition,
        "phase": phase,
        "parcellation": parc,
        "source_space": "fsaverage5",
        "method": args.method,
        "lambda2": args.lambda2,
        "label_mode": args.label_mode,
        "epoch_duration": args.epoch_duration,
        "epoch_overlap": args.epoch_overlap,
        "input_file": str(set_file),
        "cache_signature": cache_sig,
    }
    save_roi_ts_npz(out_file, roi_ts=roi_ts, label_names=label_names, times=epochs.times, sfreq=epochs.info["sfreq"], metadata=metadata)
    return [{
        "Subject": subject,
        "Condition": condition,
        "Phase": phase,
        "WindowIndex": "",
        "WindowStartSec": "",
        "WindowEndSec": "",
        "InputFile": str(set_file),
        "ROITimeSeriesFile": str(out_file),
        "NEpochs": int(len(epochs)),
        "NROIs": int(len(label_names)),
        "Sfreq": float(epochs.info["sfreq"]),
        "CacheSignature": cache_sig,
        **channel_qc,
    }]


def process_during_windows(raw, inv, labels, label_names, parc, set_file, subject, condition, phase, cache_sig, channel_qc, args):
    sfreq = float(raw.info["sfreq"])
    total_duration = float(raw.times[-1])
    windows = build_windows(total_duration, args.window_duration, args.min_window_duration)
    if args.max_during_windows is not None:
        windows = [w for w in windows if w[0] < args.max_during_windows]

    rows = []
    for widx, wstart, wend in windows:
        tmax = max(wstart, wend - 1.0 / sfreq)
        raw_win = raw.copy().crop(tmin=wstart, tmax=tmax, include_tmax=True)
        epochs = mne.make_fixed_length_epochs(
            raw_win,
            duration=args.epoch_duration,
            overlap=args.epoch_overlap,
            preload=True,
            reject_by_annotation=True,
            verbose=False,
        ).copy().pick("eeg")
        if len(epochs) == 0:
            continue

        roi_ts = extract_roi_ts_from_epochs(epochs, inv, labels, args)
        out_file = expected_window_roi_ts_file(Path(args.output_root), subject, condition, phase, widx, wstart, wend, args)
        metadata = {
            "subject_id": subject,
            "condition": condition,
            "phase": phase,
            "window_index": int(widx),
            "window_start_sec": float(wstart),
            "window_end_sec": float(wend),
            "window_duration_sec": float(wend - wstart),
            "parcellation": parc,
            "source_space": "fsaverage5",
            "method": args.method,
            "lambda2": args.lambda2,
            "label_mode": args.label_mode,
            "epoch_duration": args.epoch_duration,
            "epoch_overlap": args.epoch_overlap,
            "input_file": str(set_file),
            "cache_signature": cache_sig,
        }
        save_roi_ts_npz(out_file, roi_ts=roi_ts, label_names=label_names, times=epochs.times, sfreq=epochs.info["sfreq"], metadata=metadata)
        rows.append({
            "Subject": subject,
            "Condition": condition,
            "Phase": phase,
            "WindowIndex": int(widx),
            "WindowStartSec": float(wstart),
            "WindowEndSec": float(wend),
            "InputFile": str(set_file),
            "ROITimeSeriesFile": str(out_file),
            "NEpochs": int(len(epochs)),
            "NROIs": int(len(label_names)),
            "Sfreq": float(epochs.info["sfreq"]),
            "CacheSignature": cache_sig,
            **channel_qc,
        })
    return rows


def process_one_set_file(set_file: Path, labels, label_names, src, parc, bem, args):
    subject, condition, phase = parse_med_pbm_filename(set_file)
    print("=" * 80)
    print(f"Extracting ROI time series: {subject} {condition} {phase}")
    print(set_file)

    raw = mne.io.read_raw_eeglab(str(set_file), preload=True, verbose=False).pick("eeg")
    channel_qc = prepare_raw_for_source(
        raw,
        montage=args.montage,
        channel_template_file=args.channel_template_file,
        interpolate_bads=not args.no_interpolate_bads,
    )
    sfreq = float(raw.info["sfreq"])
    if args.expected_sfreq is not None and abs(sfreq - args.expected_sfreq) > 1e-6:
        print(f"[WARNING] Expected sfreq {args.expected_sfreq}, got {sfreq}")

    raw.set_eeg_reference("average", projection=True)
    raw.apply_proj()
    _, inv, cache_sig = get_or_create_inverse(raw.info, src, bem, Path(args.output_root), args)

    if phase == "DURING" and args.split_during:
        return process_during_windows(raw, inv, labels, label_names, parc, set_file, subject, condition, phase, cache_sig, channel_qc, args)
    return process_whole_recording(raw, inv, labels, label_names, parc, set_file, subject, condition, phase, cache_sig, channel_qc, args)


def make_parser():
    parser = argparse.ArgumentParser(description="Extract and cache source ROI time series from cleaned MED_PBM .set files.")
    parser.add_argument("--input-root", required=True, help="Root containing MED_PBM_*_RELAX.set files.")
    parser.add_argument("--output-root", required=True, help="Output root for ROI time-series cache and manifests.")
    parser.add_argument("--fsaverage5-dir", default=str(DEFAULT_FSAVERAGE5_DIR))
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--subjects-file", default=None)
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument("--phases", nargs="*", default=None, help="PRE POST DURING. Defaults to all.")
    parser.add_argument("--channel-template-file", default=None)
    parser.add_argument("--build-channel-template", action="store_true", help="Build a template from selected input files if no template is supplied.")
    parser.add_argument("--no-interpolate-bads", action="store_true")
    parser.add_argument("--split-during", action="store_true", default=True)
    parser.add_argument("--no-split-during", action="store_false", dest="split_during")
    parser.add_argument("--window-duration", type=float, default=DEFAULTS["window_duration"])
    parser.add_argument("--min-window-duration", type=float, default=DEFAULTS["min_window_duration"])
    parser.add_argument("--max-during-windows", type=int, default=DEFAULTS["max_during_windows"])
    parser.add_argument("--use-cache", action="store_true", default=True)
    parser.add_argument("--no-cache", action="store_false", dest="use_cache")
    add_common_source_args(parser)
    return parser


def main():
    mne.set_log_level("WARNING")
    args = make_parser().parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    subjects = list(args.subjects or []) + read_subjects_file(args.subjects_file)
    files = discover_set_files(Path(args.input_root), subjects=subjects, conditions=args.conditions, phases=args.phases)
    if not files:
        raise FileNotFoundError("No matching MED_PBM *_RELAX.set files found.")

    if args.channel_template_file is None and args.build_channel_template:
        args.channel_template_file = str(build_channel_template_from_files(files, output_root, args.montage))

    labels, label_names, src, parc, _subject, _subjects_dir = build_labels_and_src(Path(args.fsaverage5_dir))
    bem = fetch_fsaverage_bem()

    all_rows = []
    for set_file in files:
        all_rows.extend(process_one_set_file(set_file, labels, label_names, src, parc, bem, args))

    manifest = pd.DataFrame(all_rows)
    manifest_file = output_root / "source_roi_timeseries_manifest.csv"
    manifest.to_csv(manifest_file, index=False)
    print(f"Saved manifest: {manifest_file}")
    print(f"Rows: {len(manifest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

