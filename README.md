# Bastiaens2026_itPBM_Meditation_EEGDynamics

## Overview

This repository contains the code and processed data for the paper "Pulse Frequency of Intranasal-Transcranial Photobiomodulation Influences Meditation Phenomenology and EEG Dynamics: A Pilot Crossover Study" by SP Bastiaens, S Ziaee, M Zare, Z Wang, S Taghian, G Loheswaran, MD Sacchet, L Lim, and R Zomorrodi.

Long-term meditators completed a randomized, double-blind, sham-controlled crossover study of 810-nm intranasal-transcranial photobiomodulation (itPBM) pulsed at 4, 10, 40, or 120 Hz. EEG was recorded before, during, and after stimulation, and meditation phenomenology was assessed with the 30-item Mystical Experience Questionnaire (MEQ). Due to participant consent and institutional ethics restrictions, we do not include raw or preprocessed EEG data. Instead, the Data folder contains the deidentified MEQ scores and the extracted sensor- and source-level spectral features (band power and FOOOF/specparam parameters for the 200 Schaefer parcels) required to reproduce every figure and table in the paper.

## Folders

**Code**: Contains the .py and .m files for preprocessing, source reconstruction, feature extraction, statistics, and figures. It consists of 4 subfolders:

- **Preprocessing**: The saved RELAX configuration and quality-control outputs, and `zapline.m`, which removes the 40 Hz stimulation artifact from the 40 Hz recordings with Zapline-plus.
- **Source_pipeline**: Source reconstruction in MNE-Python (fsaverage5 template, dSPM, Schaefer-200 parcels) and extraction of the band-power and FOOOF/APF features. Scripts are numbered in run order and produce the tables in `Data/source_results`. See `Source_pipeline/README.md` for commands.
- **Main_figures**: One script per main-text figure (`Fig02`–`Fig08`), which also export the statistics for Tables 1 and 2.
- **Supplementary**: One script per supplementary figure or table (`FigS1`–`FigS6`, `TableS1`, `TableS2A`, `TableS2B`).

**Data**: Distributed as `Data.zip`. Unzip `Data.zip` in the repository root to create `Data/`. It contains the processed tables read by the scripts in `Code/Main_figures` and `Code/Supplementary`:

- `behavior/meditation_scores_long.csv`: MEQ total and subscale scores per participant and condition.
- `sensor/`: sensor-level band power and FOOOF features per channel, phase (PRE, DURING, POST), and condition.
- `source_results/source_band_power_standard/` and `source_results/source_band_power_40Hz/`: source-level absolute and relative band power per Schaefer parcel, for PRE/POST and for each 5-min DURING window. The `40Hz` folders hold the ZapLine-cleaned 40 Hz condition.
- `source_results/source_fooof_standard/` and `source_results/source_fooof_40Hz/`: source-level FOOOF features per parcel (alpha peak frequency and power, aperiodic offset and exponent), with the same structure.

Conditions are coded `Cond1` = 4 Hz, `Cond2` = 10 Hz, `Cond3` = 40 Hz, `Cond4` = 120 Hz, `Cond5` = sham, and `Cond6` = no-device control (not included in the analyses). Participants are identified by study number only. The behavioral table includes participant 1, who withdrew from the study and is excluded by the scripts; participant 21 has EEG but no MEQ data, so contributes only to EEG analyses that do not involve the MEQ.

All scripts write their figures, tables, and statistics to an `outputs/` folder at the repository root.

## Files

| Manuscript item | Script |
|---|---|
| Figure 1. Experimental design and EEG analysis pipeline | Schematic; no code |
| Figure 2A–C. MEQ distributions, contrasts vs sham, subscales | `Code/Main_figures/Fig02A-C_behavior_meq_frequency_effects.py` |
| Figure 2D. MEQ total across conditions | `Code/Main_figures/Fig02D_behavior_meq_total_across_conditions.py` |
| Figure 2E. Personalized PBM response matrix | `Code/Main_figures/Fig02E_behavior_meq_personalized_response_matrix.py` |
| Figure 3. Source-level relative spectral power changes | `Code/Main_figures/Fig03_FigS3_source_fdr_brain_maps.py --metrics relative` |
| Figure 4 and Table 1 (120 Hz). Late-to-post relative alpha and MEQ | `Code/Main_figures/Fig04_120hz_alpha_late_to_post_meq.py` |
| Figure 5. 120 Hz alpha peak-frequency coupling | `Code/Main_figures/Fig05_120hz_alpha_peak_frequency_coupling.py` |
| Figure 6 and Table 1 (40 Hz). During-stimulation relative alpha and MEQ | `Code/Main_figures/Fig06_40hz_alpha_during_stimulation_meq.py` |
| Figure 7, Table 2, Table S3, Table S4. Sham-controlled alpha–MEQ coupling | `Code/Main_figures/Fig07_alpha_meq_specificity_vs_sham.py` |
| Figure 8. Acute slowing-like spectral changes | `Code/Main_figures/Fig08_low_frequency_acute_slowing.py` |
| Figure S1 and Table S1. Crossover models of MEQ | `Code/Supplementary/FigS1_TableS1_behavior_meq_crossover_models.py` |
| Table S1 (supporting). Friedman test across active conditions | `Code/Supplementary/TableS1_behavior_meq_high_vs_low_frequency_contrasts.py` |
| Figure S2. Sensor-level spectral dynamics | `Code/Supplementary/FigS2_sensor_within_condition_topomap_atlases.py` |
| Figure S3. Source-level absolute spectral power changes | `Code/Main_figures/Fig03_FigS3_source_fdr_brain_maps.py --metrics absolute` |
| Figure S4. Source-level alpha peak and aperiodic changes | `Code/Supplementary/FigS4_fooof_source_maps_standard.py` (4 Hz, 10 Hz, 120 Hz, sham) and `Code/Supplementary/FigS4_fooof_source_maps_40Hz_zapline.py` (40 Hz) |
| Figure S5A. Network relative-power trajectories | `Code/Supplementary/FigS5A_network_relative_power_trajectories.py` |
| Figure S5B. Network alpha summaries | `Code/Supplementary/FigS5B_alpha_network_summary_bars.py` |
| Figure S6A. 4 Hz source alpha and MEQ | `Code/Supplementary/FigS6A_4hz_alpha_meq.py` |
| Figure S6B. 10 Hz source alpha and MEQ | `Code/Supplementary/FigS6B_10hz_alpha_meq.py` |
| Table S2A. 120 Hz alpha–MEQ robustness | `Code/Supplementary/TableS2A_120hz_alpha_meq_robustness.py` |
| Table S2B. 40 Hz alpha–MEQ robustness | `Code/Supplementary/TableS2B_40hz_alpha_meq_robustness.py` |

Output files are named after the manuscript item they produce (for example `Figure2A-C_...png`, `figure7D_table2_...csv`, `tableS1A_...csv`). Multi-panel figures (Figures 2, 3, S2, S3, and S4) were assembled by hand from the panels these scripts produce.

`Code/Main_figures` and `Code/Supplementary` each contain the shared helpers `pbm_paper_common.py` (paths, data loading, feature construction, statistics) and `pbm_sham_interaction_common.py` (active-vs-sham mixed models); `Code/Supplementary/sensor_utils.py` holds the sensor-level helpers.

### Running the scripts

Run each script from its own folder, for example:

```
cd Code/Main_figures
python Fig04_120hz_alpha_late_to_post_meq.py
```

`TableS2B_40hz_alpha_meq_robustness.py` reads a table written by `Fig06_40hz_alpha_during_stimulation_meq.py`, so run Figure 6 first. The brain maps for Figures 3 and S3 are drawn per condition and frequency band:

```
cd Code/Main_figures
# Figure 3 (4 Hz, 10 Hz, 120 Hz, sham) and its 40 Hz row (ZapLine-cleaned)
python Fig03_FigS3_source_fdr_brain_maps.py --feature-family bandpower_standard --metrics relative
python Fig03_FigS3_source_fdr_brain_maps.py --feature-family bandpower_zapline --metrics relative --conditions Cond3
# Figure S3
python Fig03_FigS3_source_fdr_brain_maps.py --feature-family bandpower_standard --metrics absolute
python Fig03_FigS3_source_fdr_brain_maps.py --feature-family bandpower_zapline --metrics absolute --conditions Cond3
```

Add `--no-render` to compute the map statistics without fsaverage5 or rendering.

## Requirements

Analyses and figures:

- Python 3.11 with numpy, pandas, scipy, matplotlib, statsmodels, mne (tested with MNE-Python 1.11.0), fooof, pyvista, and pyvistaqt:
  `pip install numpy pandas scipy matplotlib statsmodels mne fooof pyvista pyvistaqt`
- statsmodels is required. Without it, `Fig07_alpha_meq_specificity_vs_sham.py` and `FigS5B_alpha_network_summary_bars.py` silently fall back to ordinary least squares with participant intercepts and will not reproduce the reported mixed-model values.

Preprocessing (only needed to regenerate the Data tables from EEG):

- MATLAB with EEGLAB (https://sccn.ucsd.edu/eeglab), the RELAX plugin (https://github.com/NeilwBailey/RELAX), and Zapline-plus (https://github.com/MariusKlug/zapline-plus).

Source reconstruction and brain visualization:

The source pipeline and the brain maps for Figures 3, S3, and S4 need the FreeSurfer `fsaverage5` template with the Schaefer-200 (17-network) parcellation, placed at `Data/fsaverage5`:

1. Install FreeSurfer: download and installation instructions are provided at https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall. `fsaverage5` is included in the installation at `$FREESURFER_HOME/subjects/fsaverage5`.
2. Copy that whole folder to `Data/fsaverage5`, so that it contains `surf/` and `label/`.
3. Download `lh.Schaefer2018_200Parcels_17Networks_order.annot` and `rh.Schaefer2018_200Parcels_17Networks_order.annot` from https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/FreeSurfer5.3/fsaverage5/label and place them in `Data/fsaverage5/label/`.

The fsaverage boundary-element model used by `Source_pipeline` is downloaded automatically by MNE (`mne.datasets.fetch_fsaverage`). `Data/fsaverage5` is listed in `.gitignore` and is not distributed with this repository.
