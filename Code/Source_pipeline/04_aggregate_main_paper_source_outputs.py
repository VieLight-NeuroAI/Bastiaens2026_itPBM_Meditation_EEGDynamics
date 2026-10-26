#!/usr/bin/env python
# ============================================================
# 04_aggregate_main_paper_source_outputs.py
#
# Step 4 of the source pipeline: combines the per-recording band-power and
# FOOOF CSVs into the aggregate tables read by the figure scripts
# (Data\source_results\...).
# ============================================================
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from source_config import (
    DEFAULT_FOOOF_ROOT,
    DEFAULT_STANDARD_DURING_ROOT,
    DEFAULT_STANDARD_PREPOST_ROOT,
    DEFAULT_ZAPLINE_DURING_ROOT,
    DEFAULT_ZAPLINE_PREPOST_ROOT,
)


def aggregate(root: Path, pattern: str, out_name: str, required: bool = False):
    files = sorted(root.glob(pattern))
    if not files:
        msg = f"No files found under {root} with pattern {pattern}"
        if required:
            raise FileNotFoundError(msg)
        print(f"[SKIP] {msg}")
        return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    out_file = root / out_name
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
    print(f"[OK] {out_file} | files={len(files)} shape={df.shape}")
    return out_file


def make_parser():
    parser = argparse.ArgumentParser(description="Aggregate per-recording source feature CSVs into the combined tables used by the figure scripts.")
    parser.add_argument("--standard-prepost-root", default=str(DEFAULT_STANDARD_PREPOST_ROOT))
    parser.add_argument("--standard-during-root", default=str(DEFAULT_STANDARD_DURING_ROOT))
    parser.add_argument("--zapline-prepost-root", default=str(DEFAULT_ZAPLINE_PREPOST_ROOT))
    parser.add_argument("--zapline-during-root", default=str(DEFAULT_ZAPLINE_DURING_ROOT))
    parser.add_argument("--fooof-root", default=str(DEFAULT_FOOOF_ROOT))
    parser.add_argument("--require-all", action="store_true")
    return parser


def main():
    args = make_parser().parse_args()
    outputs = []

    # Standard band power -> Data/source_results/source_band_power_standard
    outputs.append(aggregate(
        Path(args.standard_prepost_root),
        "roi_band_power_features/*source_roi_band_power_long*.csv",
        "source_roi_band_power_all_subjects_conditions_long.csv",
        required=args.require_all,
    ))
    outputs.append(aggregate(
        Path(args.standard_during_root),
        "during_5min_roi_band_power_features/*source_roi_band_power_5min_long*.csv",
        "during_5min_source_roi_band_power_all_subjects_conditions_long.csv",
        required=args.require_all,
    ))

    # ZapLine-cleaned 40 Hz band power -> Data/source_results/source_band_power_40Hz
    outputs.append(aggregate(
        Path(args.zapline_prepost_root),
        "roi_band_power_features/*source_roi_band_power_long*.csv",
        "source_roi_band_power_all_subjects_conditions_long.csv",
        required=args.require_all,
    ))
    outputs.append(aggregate(
        Path(args.zapline_during_root),
        "during_5min_roi_band_power_features/*source_roi_band_power_5min_long*.csv",
        "during_5min_source_roi_band_power_all_subjects_conditions_long.csv",
        required=args.require_all,
    ))

    # FOOOF/APF -> Data/source_results/source_fooof_standard, or
    # source_fooof_40Hz when --fooof-root points to the ZapLine FOOOF output.
    fooof_root = Path(args.fooof_root)
    outputs.append(aggregate(
        fooof_root,
        "roi_fooof_features/*source_roi_fooof_features_long*.csv",
        "source_roi_fooof_all_subjects_conditions_long.csv",
        required=args.require_all,
    ))
    outputs.append(aggregate(
        fooof_root / "during_5min",
        "during_5min_roi_fooof_features/*source_roi_fooof_5min_long*.csv",
        "during_5min_source_roi_fooof_all_subjects_conditions_long.csv",
        required=args.require_all,
    ))

    manifest = pd.DataFrame({"OutputFile": [str(o) for o in outputs if o is not None]})
    manifest_file = fooof_root.parent / "source_output_manifest.csv"
    manifest.to_csv(manifest_file, index=False)
    print(f"Saved output manifest: {manifest_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

