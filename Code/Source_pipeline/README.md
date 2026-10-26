# Source Pipeline

Source reconstruction and source-level feature extraction (Methods 3.4–3.6). The pipeline takes the cleaned EEGLAB `.set` files, reconstructs source activity once, and saves the Schaefer-200 parcel time series so that band power and FOOOF/APF features can be computed from the same intermediate files:

```text
cleaned .set file
  -> source ROI time-series .npz
  -> band-power CSVs
  -> FOOOF/APF CSVs
```

## Scripts

| Script | Purpose |
|---|---|
| `01_extract_source_roi_timeseries.py` | Source reconstruction (fsaverage5 template, dSPM, Schaefer-200 parcels); saves ROI time-series `.npz` files for PRE/POST and for each 5-min DURING window. |
| `02_compute_bandpower_from_roi_timeseries.py` | Reads the ROI `.npz` files and writes long-format absolute and relative band-power CSVs. |
| `03_compute_fooof_from_roi_timeseries.py` | Reads the ROI `.npz` files and writes FOOOF/APF feature CSVs (alpha peak frequency and power, aperiodic offset and exponent). |
| `04_aggregate_main_paper_source_outputs.py` | Combines the per-recording CSVs into the tables in `Data/source_results` read by the figure scripts. |
| `05_rerun_fooof_from_roi_timeseries.py` | Convenience wrapper that reruns steps 3 and 4 from saved ROI `.npz` files without repeating source reconstruction. |
| `source_config.py` | Frequency bands, default paths, and analysis settings. |
| `source_utils.py` | Shared source-reconstruction, PSD, FOOOF, channel-interpolation, and ROI time-series helpers. |

## Outputs

The scripts write to `outputs/source_results` at the repository root. The aggregation step (`04_aggregate_main_paper_source_outputs.py`) writes each combined table inside its output folder; these are the tables distributed in `Data.zip`:

```text
Data/source_results/source_band_power_standard
  source_roi_band_power_all_subjects_conditions_long.csv
  during_5min_source_roi_band_power_all_subjects_conditions_long.csv

Data/source_results/source_band_power_40Hz
  source_roi_band_power_all_subjects_conditions_long.csv
  during_5min_source_roi_band_power_all_subjects_conditions_long.csv

Data/source_results/source_fooof_standard
  source_roi_fooof_all_subjects_conditions_long.csv
  during_5min_source_roi_fooof_all_subjects_conditions_long.csv

Data/source_results/source_fooof_40Hz
  source_roi_fooof_all_subjects_conditions_long.csv
  during_5min_source_roi_fooof_all_subjects_conditions_long.csv
```

## Running the pipeline

Run from this folder. `Data/fsaverage5` must be set up first (see the main README).

### 1. Extract PRE/POST ROI time series

```powershell
python 01_extract_source_roi_timeseries.py `
  --input-root "path\to\cleaned_set_files" `
  --output-root "..\..\outputs\source_results\cleaned_interp" `
  --fsaverage5-dir "..\..\Data\fsaverage5" `
  --phases PRE POST `
  --build-channel-template
```

### 2. Extract DURING 5-min ROI time series

```powershell
python 01_extract_source_roi_timeseries.py `
  --input-root "path\to\cleaned_set_files" `
  --output-root "..\..\outputs\source_results\cleaned_interp_during_5min" `
  --fsaverage5-dir "..\..\Data\fsaverage5" `
  --phases DURING `
  --build-channel-template
```

### 3. Compute band-power features

```powershell
python 02_compute_bandpower_from_roi_timeseries.py `
  --output-root "..\..\outputs\source_results\cleaned_interp" `
  --scope prepost

python 02_compute_bandpower_from_roi_timeseries.py `
  --output-root "..\..\outputs\source_results\cleaned_interp_during_5min" `
  --scope during
```

### 4. Compute FOOOF/APF features

```powershell
python 03_compute_fooof_from_roi_timeseries.py `
  --roi-output-root "..\..\outputs\source_results\cleaned_interp" `
  --fooof-output-root "..\..\outputs\source_results\fooof" `
  --scope prepost

python 03_compute_fooof_from_roi_timeseries.py `
  --roi-output-root "..\..\outputs\source_results\cleaned_interp_during_5min" `
  --fooof-output-root "..\..\outputs\source_results\fooof" `
  --scope during
```

### 5. Aggregate into the combined tables

```powershell
python 04_aggregate_main_paper_source_outputs.py
```

Steps 4 and 5 can also be run together from saved ROI time series with:

```powershell
python 05_rerun_fooof_from_roi_timeseries.py `
  --fooof-output-root "..\..\outputs\source_results\fooof" `
  --peak-bands Theta Alpha Beta Gamma `
  --fooof-fmin 1 `
  --fooof-fmax 50
```

## 40 Hz condition

The 40 Hz recordings are first cleaned with `Code/Preprocessing/zapline.m`. Run steps 1–3 on the ZapLine-cleaned `.set` files with the output roots

```text
outputs\source_results\zapline_40
outputs\source_results\zapline_40_during_5min
```

and step 4 with a separate FOOOF output folder (for example `--fooof-output-root "..\..\outputs\source_results\fooof_40Hz"`). Step 5 then writes the 40 Hz band-power tables (`source_band_power_40Hz`); run it once more with `--fooof-root "..\..\outputs\source_results\fooof_40Hz"` to write the 40 Hz FOOOF tables (`source_fooof_40Hz`).
