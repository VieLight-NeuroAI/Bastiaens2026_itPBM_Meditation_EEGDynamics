# EEG Preprocessing Provenance

EEG preprocessing was performed using the RELAX pipeline through the EEGLAB graphical user interface. No custom batch script was used for the primary RELAX preprocessing step. This folder contains the saved RELAX GUI configuration and quality-control/provenance outputs associated with the preprocessing run.

## Files

| File | Description |
|---|---|
| `RELAX_cfg.mat` | Saved RELAX configuration used for preprocessing. This file contains the GUI-selected preprocessing settings, including filtering, downsampling, channel deletion, ICA method, blink/electrode settings, and artifact-detection thresholds. |
| `RawMetrics.mat` | RELAX quality-control metrics computed on the raw recordings before cleaning. |
| `CleanedMetrics.mat` | RELAX quality-control metrics computed after cleaning. |
| `ProcessingStatistics_wICA.mat` | RELAX processing statistics from the ICA-based cleaning workflow. |
| `RELAXProcessingExtremeRejectionsAllParticipants.mat` | RELAX output summarizing extreme rejection periods across participants. |
| `RELAX_issues_to_check.mat` | RELAX-generated list of recordings or metrics flagged for review. |
| `zapline.m` | MATLAB/EEGLAB script that applies Zapline-plus at 40 Hz to the RELAX-cleaned 40 Hz (Cond3) `.set` files (Methods 3.3). |

## Key RELAX Settings

The saved `RELAX_cfg.mat` indicates that recordings were:

- downsampled to 500 Hz;
- filtered with a Butterworth band-pass filter from 0.5 to 80 Hz;
- filtered for 60 Hz line noise using a Butterworth notch filter;
- cleaned after removing non-standard or non-scalp channels including `CB1`, `CB2`, `F11`, `F12`, `FT11`, and `FT12`;
- cleaned using ICA estimated with the Picard algorithm;
- evaluated with RELAX artifact-detection and quality-control metrics including blink-amplitude ratio, muscle contamination metrics, artifact-rejection ratios, and extreme-noise/rejection summaries.

Blink detection was configured around frontopolar and frontal electrodes, and horizontal eye-movement detection used the saved RELAX horizontal-eye-movement criteria in `RELAX_cfg.mat`.

## Downstream Use

The cleaned EEGLAB `.set` files generated from this RELAX GUI workflow were used as inputs for the source reconstruction and source-feature extraction scripts in the main analysis code. The 40 Hz condition was additionally processed with a targeted ZapLine-plus cleaning step for the narrow stimulation-frequency artifact before the 40 Hz source-feature analyses (`zapline.m`).

## Reproducibility Note

Because preprocessing was run through the EEGLAB/RELAX GUI, this folder should be treated as preprocessing configuration and provenance rather than executable preprocessing code. To fully rerun preprocessing from raw EEG, a user would need the original raw EEG files, EEGLAB, RELAX, the relevant EEGLAB plugins, and the saved `RELAX_cfg.mat` settings documented here.
