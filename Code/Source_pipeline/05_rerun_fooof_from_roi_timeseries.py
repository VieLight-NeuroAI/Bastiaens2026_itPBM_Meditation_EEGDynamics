#!/usr/bin/env python
from __future__ import annotations

"""
05_rerun_fooof_from_roi_timeseries.py

Run source-level FOOOF/APF extraction from saved ROI time series, for both
PRE/POST and the 5-min DURING windows, then aggregate the outputs.

This is a convenience wrapper around:

    03_compute_fooof_from_roi_timeseries.py
    04_aggregate_main_paper_source_outputs.py

It does not run source reconstruction. It expects the ROI time-series `.npz`
files written by 01_extract_source_roi_timeseries.py in:

    cleaned_interp/roi_timeseries
    cleaned_interp_during_5min/during_5min_roi_timeseries

Any additional FOOOF options passed to this script are forwarded to
03_compute_fooof_from_roi_timeseries.py. For example:

    python 05_rerun_fooof_from_roi_timeseries.py ^
      --fooof-output-root "..\\..\\outputs\\source_results\\fooof" ^
      --peak-bands Theta Alpha Beta Gamma ^
      --fooof-fmin 1 --fooof-fmax 50
"""

import argparse
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_SOURCE_RESULTS_ROOT = PROJECT_ROOT / "outputs" / "source_results"

DEFAULT_PREPOST_ROI_ROOT = Path(
    OUTPUT_SOURCE_RESULTS_ROOT / "cleaned_interp"
)
DEFAULT_DURING_ROI_ROOT = Path(
    OUTPUT_SOURCE_RESULTS_ROOT / "cleaned_interp_during_5min"
)

# Default FOOOF output folder; pass --fooof-output-root to write elsewhere.
DEFAULT_FOOOF_OUTPUT_ROOT = Path(
    OUTPUT_SOURCE_RESULTS_ROOT / "fooof"
)


def run_command(cmd: list[str], dry_run: bool = False):
    print("\n" + "=" * 90)
    print(" ".join(f'"{c}"' if " " in c else c for c in cmd))
    print("=" * 90)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def make_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run FOOOF/APF extraction from saved source ROI time series "
            "without rerunning source reconstruction."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--prepost-roi-root",
        default=str(DEFAULT_PREPOST_ROI_ROOT),
        help="Root containing roi_timeseries for PRE/POST ROI .npz files.",
    )
    parser.add_argument(
        "--during-roi-root",
        default=str(DEFAULT_DURING_ROI_ROOT),
        help="Root containing during_5min_roi_timeseries for DURING ROI .npz files.",
    )
    parser.add_argument(
        "--fooof-output-root",
        default=str(DEFAULT_FOOOF_OUTPUT_ROOT),
        help="Root where FOOOF feature CSVs and aggregate outputs will be written.",
    )
    parser.add_argument(
        "--max-during-windows",
        type=int,
        default=4,
        help="Maximum number of 5-min DURING windows to include.",
    )
    parser.add_argument("--skip-prepost", action="store_true")
    parser.add_argument("--skip-during", action="store_true")
    parser.add_argument("--skip-aggregate", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use for the child scripts.",
    )
    return parser


def main():
    parser = make_parser()
    args, extra_fooof_args = parser.parse_known_args()

    this_dir = Path(__file__).resolve().parent
    fooof_script = this_dir / "03_compute_fooof_from_roi_timeseries.py"
    aggregate_script = this_dir / "04_aggregate_main_paper_source_outputs.py"

    prepost_roi_root = Path(args.prepost_roi_root)
    during_roi_root = Path(args.during_roi_root)
    fooof_output_root = Path(args.fooof_output_root)

    if not fooof_script.exists():
        raise FileNotFoundError(f"Missing FOOOF extraction script: {fooof_script}")
    if not aggregate_script.exists():
        raise FileNotFoundError(f"Missing aggregation script: {aggregate_script}")

    if not args.skip_prepost:
        roi_dir = prepost_roi_root / "roi_timeseries"
        if not roi_dir.exists():
            raise FileNotFoundError(f"Missing PRE/POST ROI time-series folder: {roi_dir}")
        run_command(
            [
                args.python,
                str(fooof_script),
                "--roi-output-root",
                str(prepost_roi_root),
                "--fooof-output-root",
                str(fooof_output_root),
                "--scope",
                "prepost",
                *extra_fooof_args,
            ],
            dry_run=args.dry_run,
        )

    if not args.skip_during:
        roi_dir = during_roi_root / "during_5min_roi_timeseries"
        if not roi_dir.exists():
            raise FileNotFoundError(f"Missing DURING ROI time-series folder: {roi_dir}")
        run_command(
            [
                args.python,
                str(fooof_script),
                "--roi-output-root",
                str(during_roi_root),
                "--fooof-output-root",
                str(fooof_output_root),
                "--scope",
                "during",
                "--max-during-windows",
                str(args.max_during_windows),
                *extra_fooof_args,
            ],
            dry_run=args.dry_run,
        )

    if not args.skip_aggregate:
        run_command(
            [
                args.python,
                str(aggregate_script),
                "--fooof-root",
                str(fooof_output_root),
            ],
            dry_run=args.dry_run,
        )

    print("\nDone.")
    print("FOOOF output root:", fooof_output_root)
    print(
        "Main aggregate files should be:\n"
        f"  {fooof_output_root / 'source_roi_fooof_all_subjects_conditions_long.csv'}\n"
        f"  {fooof_output_root / 'during_5min' / 'during_5min_source_roi_fooof_all_subjects_conditions_long.csv'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
