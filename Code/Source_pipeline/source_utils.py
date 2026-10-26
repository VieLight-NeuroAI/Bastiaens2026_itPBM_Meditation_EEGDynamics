# ============================================================
# source_utils.py
#
# Shared source-reconstruction, PSD, FOOOF, channel-interpolation, and
# ROI-cache helpers for the source pipeline (Methods 3.4-3.6).
# ============================================================

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import re
from pathlib import Path
from typing import Iterable

import mne
import numpy as np
import pandas as pd
from mne.datasets import fetch_fsaverage
from mne.minimum_norm import (
    apply_inverse_epochs,
    make_inverse_operator,
    read_inverse_operator,
    write_inverse_operator,
)
from mne.time_frequency import psd_array_welch

from source_config import BAND_SCHEMES, PEAK_BANDS


def parse_med_pbm_filename(set_path: Path) -> tuple[str, str, str]:
    pattern = (
        r"MED_PBM_(?P<subject>\d+)_+"
        r"(?P<condition>Cond\d+)_(?P<phase>PRE|DURING|POST)_RELAX\.set$"
    )
    match = re.search(pattern, set_path.name)
    if match is None:
        raise ValueError(f"Filename does not match MED_PBM pattern: {set_path.name}")
    return match.group("subject"), match.group("condition"), match.group("phase")


def try_parse_med_pbm_filename(set_path: Path):
    try:
        return parse_med_pbm_filename(set_path)
    except ValueError:
        return None


def normalize_subject_id(x) -> str:
    x = str(x).strip().replace("MED_PBM_", "")
    return f"{int(x):03d}" if x.isdigit() else x


def normalize_subject_filter(subjects: Iterable[str] | None):
    if not subjects:
        return None
    out = set()
    for s in subjects:
        s = str(s).strip()
        if s:
            out.add(normalize_subject_id(s))
    return out


def read_subjects_file(path: str | Path | None) -> list[str]:
    if path is None:
        return []
    path = Path(path)
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def discover_set_files(input_root: Path, subjects=None, conditions=None, phases=None) -> list[Path]:
    subjects = normalize_subject_filter(subjects)
    conditions = set(conditions) if conditions else None
    phases = set(phases) if phases else None

    selected = []
    for f in sorted(Path(input_root).rglob("MED_PBM_*_RELAX.set")):
        parsed = try_parse_med_pbm_filename(f)
        if parsed is None:
            continue
        subject, condition, phase = parsed
        if subjects is not None and subject not in subjects:
            continue
        if conditions is not None and condition not in conditions:
            continue
        if phases is not None and phase not in phases:
            continue
        selected.append(f)
    return selected


def _clean_name(name: str) -> str:
    return str(name).strip()


def read_channel_template(path: str | Path | None) -> list[str]:
    if path is None:
        return []
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Channel template file does not exist: {path}")

    text = path.read_text(encoding="utf-8-sig", errors="replace")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []

    if path.suffix.lower() == ".csv" or "," in lines[0]:
        rows = list(csv.DictReader(lines))
        if rows and rows[0]:
            possible_cols = ["Channel", "channel", "ch_name", "name", "ChName"]
            col = next((c for c in possible_cols if c in rows[0]), None)
            if col is None:
                col = list(rows[0].keys())[0]
            out = [_clean_name(r[col]) for r in rows if r.get(col, "").strip()]
        else:
            out = []
    else:
        out = [_clean_name(ln.split(",")[0]) for ln in lines]

    seen, unique = set(), []
    for ch in out:
        key = ch.lower()
        if ch and key not in seen:
            seen.add(key)
            unique.append(ch)
    return unique


def _lower_map(names):
    return {_clean_name(n).lower(): _clean_name(n) for n in names}


def _set_montage_compatible(raw: mne.io.BaseRaw, montage: str):
    try:
        raw.set_montage(montage, on_missing="ignore", match_case=False, match_alias=True)
    except TypeError:
        try:
            raw.set_montage(montage, on_missing="ignore", match_case=False)
        except TypeError:
            raw.set_montage(montage, on_missing="ignore")


def _canonical_template_channels(template_channels: list[str], montage: str):
    std = mne.channels.make_standard_montage(montage)
    std_by_lower = _lower_map(std.ch_names)
    kept, skipped, seen = [], [], set()
    for ch in template_channels:
        key = _clean_name(ch).lower()
        if key in std_by_lower:
            canonical = std_by_lower[key]
            if canonical.lower() not in seen:
                seen.add(canonical.lower())
                kept.append(canonical)
        else:
            skipped.append(ch)
    return kept, skipped


def _has_valid_position(raw: mne.io.BaseRaw, ch_name: str) -> bool:
    idx = raw.ch_names.index(ch_name)
    loc = np.asarray(raw.info["chs"][idx]["loc"][:3], dtype=float)
    return bool(np.all(np.isfinite(loc)) and np.linalg.norm(loc) > 0)


def prepare_raw_for_source(
    raw: mne.io.BaseRaw,
    *,
    montage: str,
    channel_template_file: str | Path | None,
    interpolate_bads: bool,
) -> dict:
    if not raw.preload:
        raw.load_data()

    original_ch_names = list(raw.ch_names)
    original_bads = list(raw.info.get("bads", []))
    _set_montage_compatible(raw, montage)

    template_channels = read_channel_template(channel_template_file)
    template_channels, template_skipped = (
        _canonical_template_channels(template_channels, montage)
        if template_channels
        else ([], [])
    )

    existing_lower = {ch.lower(): ch for ch in raw.ch_names}
    missing_channels = [ch for ch in template_channels if ch.lower() not in existing_lower]

    if missing_channels:
        zero_data = np.zeros((len(missing_channels), raw.n_times), dtype=np.float64)
        missing_info = mne.create_info(
            ch_names=missing_channels,
            sfreq=float(raw.info["sfreq"]),
            ch_types="eeg",
        )
        raw.add_channels([mne.io.RawArray(zero_data, missing_info, verbose=False)], force_update_info=True)
        _set_montage_compatible(raw, montage)
        bad_set = set(raw.info.get("bads", []))
        bad_set.update(missing_channels)
        raw.info["bads"] = [ch for ch in raw.ch_names if ch in bad_set]

    bads_before = list(raw.info.get("bads", []))
    bads_without_position = [
        ch for ch in bads_before
        if ch in raw.ch_names and not _has_valid_position(raw, ch)
    ]

    if interpolate_bads and bads_before:
        interp_bads = [ch for ch in bads_before if ch not in bads_without_position]
        raw.info["bads"] = interp_bads
        if interp_bads:
            raw.interpolate_bads(reset_bads=True, verbose=False)
        if bads_without_position:
            raw.info["bads"] = sorted(set(raw.info.get("bads", [])) | set(bads_without_position))

    _set_montage_compatible(raw, montage)
    return {
        "n_channels_original": len(original_ch_names),
        "n_channels_after_harmonization": len(raw.ch_names),
        "n_template_channels": len(template_channels),
        "channel_template_file": str(channel_template_file) if channel_template_file else "",
        "original_bads": ";".join(original_bads),
        "bads_before_interpolation": ";".join(bads_before),
        "bads_after_interpolation": ";".join(raw.info.get("bads", [])),
        "missing_template_channels_added": ";".join(missing_channels),
        "n_missing_template_channels_added": len(missing_channels),
        "template_channels_skipped_no_montage": ";".join(template_skipped),
        "bads_without_position_not_interpolated": ";".join(bads_without_position),
        "interpolate_bads_enabled": bool(interpolate_bads),
        "montage": montage,
        "final_channel_order": ";".join(raw.ch_names),
    }


def build_channel_template_from_files(files: list[Path], output_root: Path, montage: str) -> Path:
    std = mne.channels.make_standard_montage(montage)
    std_by_lower = {ch.lower(): ch for ch in std.ch_names}

    seen, template, rows, skipped = set(), [], [], []
    for f in files:
        raw = mne.io.read_raw_eeglab(str(f), preload=False, verbose=False).pick("eeg")
        file_chs = []
        for ch in raw.ch_names:
            key = str(ch).strip().lower()
            if key in std_by_lower:
                canonical = std_by_lower[key]
                file_chs.append(canonical)
                if canonical.lower() not in seen:
                    seen.add(canonical.lower())
                    template.append(canonical)
            else:
                skipped.append((str(f), ch))
        rows.append({
            "input_file": str(f),
            "n_eeg_channels": len(raw.ch_names),
            "template_matched_channels": ";".join(file_chs),
        })

    output_root.mkdir(parents=True, exist_ok=True)
    template_file = output_root / "channel_template_from_selected_files.csv"
    pd.DataFrame({"Channel": template}).to_csv(template_file, index=False)
    pd.DataFrame(rows).to_csv(output_root / "channel_template_qc.csv", index=False)
    pd.DataFrame(skipped, columns=["input_file", "channel"]).to_csv(
        output_root / "channel_template_skipped_no_montage.csv",
        index=False,
    )
    return template_file


def build_windows(total_duration: float, window_duration: float, min_window_duration: float):
    windows, start, idx = [], 0.0, 0
    while start < total_duration:
        end = min(start + window_duration, total_duration)
        if end - start >= min_window_duration:
            windows.append((idx, start, end))
        start += window_duration
        idx += 1
    return windows


def is_schaefer_brain_label(label_name: str) -> bool:
    name = str(label_name).lower()
    bad_terms = ["unknown", "medial", "wall", "background", "corpuscallosum", "corpus_callosum"]
    return not any(term in name for term in bad_terms)


def build_labels_and_src(fsaverage5_dir: Path):
    subjects_dir = fsaverage5_dir.parent
    subject = fsaverage5_dir.name
    annot_files = sorted((fsaverage5_dir / "label").glob("*Schaefer*200*annot"))
    lh_files = [f for f in annot_files if f.name.startswith("lh.")]
    if not lh_files:
        raise FileNotFoundError(f"No lh.*Schaefer*200*.annot file found in {fsaverage5_dir / 'label'}")
    parc = lh_files[0].name.replace("lh.", "").replace(".annot", "")
    labels = mne.read_labels_from_annot(
        subject=subject,
        parc=parc,
        subjects_dir=subjects_dir,
        hemi="both",
        sort=True,
    )
    labels = [lab for lab in labels if is_schaefer_brain_label(lab.name)]
    label_names = [lab.name for lab in labels]
    src = mne.setup_source_space(
        subject=subject,
        spacing="all",
        subjects_dir=subjects_dir,
        add_dist=False,
        verbose=False,
    )
    return labels, label_names, src, parc, subject, subjects_dir


def fetch_fsaverage_bem() -> Path:
    bem = Path(fetch_fsaverage(verbose=False)) / "bem" / "fsaverage-5120-5120-5120-bem-sol.fif"
    if not bem.exists():
        raise FileNotFoundError(f"Could not find BEM file: {bem}")
    return bem


def channel_signature(info, args) -> str:
    text = (
        "|".join(info["ch_names"])
        + f"|sfreq={info['sfreq']}"
        + f"|method={args.method}|loose={args.loose}|depth={args.depth}"
        + f"|mindist={args.mindist}|pick_ori={args.pick_ori}|avgref|fsaverage5"
    )
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]


def get_or_create_inverse(raw_info, src, bem: Path, output_root: Path, args):
    cache_dir = output_root / "forward_inverse_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    sig = channel_signature(raw_info, args)
    fwd_file = cache_dir / f"fsaverage5_template_{sig}-fwd.fif"
    inv_file = cache_dir / f"fsaverage5_template_{sig}-inv.fif"

    if getattr(args, "use_cache", True) and fwd_file.exists() and inv_file.exists():
        return mne.read_forward_solution(fwd_file), read_inverse_operator(inv_file), sig

    fwd = mne.make_forward_solution(
        info=raw_info,
        trans="fsaverage",
        src=src,
        bem=str(bem),
        eeg=True,
        meg=False,
        mindist=args.mindist,
        n_jobs=args.n_jobs_forward,
        verbose=False,
    )
    inv = make_inverse_operator(
        info=raw_info,
        forward=fwd,
        noise_cov=mne.make_ad_hoc_cov(raw_info),
        loose=args.loose,
        depth=args.depth,
        verbose=False,
    )
    if getattr(args, "use_cache", True):
        mne.write_forward_solution(fwd_file, fwd, overwrite=True)
        write_inverse_operator(inv_file, inv, overwrite=True)
    return fwd, inv, sig


def save_roi_ts_npz(path: Path, *, roi_ts, label_names, times, sfreq, metadata: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "roi_ts": np.asarray(roi_ts, dtype=np.float32),
        "label_names": np.asarray(label_names),
        "times": np.asarray(times, dtype=np.float32),
        "sfreq": np.asarray(sfreq, dtype=np.float32),
    }
    for key, value in metadata.items():
        payload[key] = np.asarray(value)
    np.savez_compressed(path, **payload)


def _scalar_from_npz(data, key, default=None):
    if key not in data:
        return default
    value = data[key]
    if hasattr(value, "shape") and value.shape == ():
        return value.item()
    return value


def load_roi_ts_npz(path: Path):
    data = np.load(path, allow_pickle=True)
    roi_ts = np.asarray(data["roi_ts"], dtype=np.float32)
    label_names = data["label_names"].astype(str).tolist()
    times = np.asarray(data["times"], dtype=np.float32)
    sfreq = float(np.asarray(data["sfreq"]).item())
    metadata = {k: _scalar_from_npz(data, k) for k in data.files if k not in {"roi_ts", "label_names", "times", "sfreq"}}
    return roi_ts, label_names, times, sfreq, metadata


def expected_roi_ts_file(output_root: Path, subject: str, condition: str, phase: str, args) -> Path:
    return output_root / "roi_timeseries" / (
        f"MED_PBM_{subject}_{condition}_{phase}_"
        f"roi_ts_schaefer200_fsaverage5_{args.method}_{args.label_mode}.npz"
    )


def expected_window_roi_ts_file(
    output_root: Path,
    subject: str,
    condition: str,
    phase: str,
    window_index: int,
    window_start: float,
    window_end: float,
    args,
) -> Path:
    return output_root / "during_5min_roi_timeseries" / (
        f"MED_PBM_{subject}_{condition}_{phase}_"
        f"window{window_index:02d}_{int(window_start):04d}-{int(window_end):04d}s_"
        f"roi_ts_schaefer200_fsaverage5_{args.method}_{args.label_mode}.npz"
    )


def extract_roi_ts_from_epochs(epochs, inv, labels, args):
    stcs = apply_inverse_epochs(
        epochs,
        inverse_operator=inv,
        lambda2=args.lambda2,
        method=args.method,
        pick_ori=args.pick_ori,
        return_generator=True,
        verbose=False,
    )
    roi_ts = mne.extract_label_time_course(
        stcs,
        labels,
        src=inv["src"],
        mode=args.label_mode,
        allow_empty=False,
        return_generator=False,
        verbose=False,
    )
    return np.asarray(roi_ts, dtype=np.float32)


def compute_roi_psd(roi_ts, sfreq: float, epoch_duration: float, psd_fmin: float, psd_fmax: float):
    n_fft = int(round(sfreq * epoch_duration))
    n_fft_use = min(n_fft, roi_ts.shape[-1])
    psds, freqs = psd_array_welch(
        roi_ts,
        sfreq=sfreq,
        fmin=psd_fmin,
        fmax=psd_fmax,
        n_fft=n_fft_use,
        n_per_seg=n_fft_use,
        n_overlap=0,
        average="mean",
        verbose=False,
    )
    return psds, np.nanmean(psds, axis=0), freqs


def extract_source_band_powers(psds, freqs, bands, normalize=False, total_band=(1.0, 50.0)):
    band_epoch, band_mean = {}, {}
    if normalize:
        total_idx = np.where((freqs >= total_band[0]) & (freqs <= total_band[1]))[0]
        if len(total_idx) == 0:
            raise RuntimeError("No frequency bins found for total relative-power band.")
        total_power = np.trapezoid(psds[:, :, total_idx], freqs[total_idx], axis=-1)
    for band_name, (fmin, fmax) in bands.items():
        idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]
        if len(idx) == 0:
            values = np.full(psds.shape[:2], np.nan)
        else:
            values = np.trapezoid(psds[:, :, idx], freqs[idx], axis=-1)
            if normalize:
                values = values / (total_power + 1e-12)
        band_epoch[band_name] = values
        band_mean[band_name] = np.nanmean(values, axis=0)
    return band_epoch, band_mean


def build_bandpower_rows(roi_file: Path, args) -> pd.DataFrame:
    roi_ts, label_names, _times, sfreq, meta = load_roi_ts_npz(roi_file)
    psds, _psds_mean, freqs = compute_roi_psd(
        roi_ts,
        sfreq=sfreq,
        epoch_duration=float(meta.get("epoch_duration", args.epoch_duration)),
        psd_fmin=args.psd_fmin,
        psd_fmax=args.psd_fmax,
    )

    subject = str(meta.get("subject_id"))
    condition = str(meta.get("condition"))
    phase = str(meta.get("phase"))
    is_during_window = phase == "DURING" and "window_index" in meta

    rows = []
    for scheme_name, bands in BAND_SCHEMES.items():
        _, abs_mean = extract_source_band_powers(psds, freqs, bands, normalize=False, total_band=(args.psd_fmin, args.psd_fmax))
        _, rel_mean = extract_source_band_powers(psds, freqs, bands, normalize=True, total_band=(args.psd_fmin, args.psd_fmax))
        for band_name, (blo, bhi) in bands.items():
            for roi_name, abs_val, rel_val in zip(label_names, abs_mean[band_name], rel_mean[band_name]):
                base = {
                    "Subject": subject,
                    "Condition": condition,
                    "Phase": phase,
                    "ROI": roi_name,
                    "Band": band_name,
                    "BandScheme": scheme_name,
                    "BandLow": blo,
                    "BandHigh": bhi,
                    "PSD_fmin": args.psd_fmin,
                    "PSD_fmax": args.psd_fmax,
                    "EpochDuration": float(meta.get("epoch_duration", args.epoch_duration)),
                    "EpochOverlap": float(meta.get("epoch_overlap", args.epoch_overlap)),
                    "SourceSpace": str(meta.get("source_space", "fsaverage5")),
                    "Parcellation": str(meta.get("parcellation", "")),
                    "InverseMethod": str(meta.get("method", args.method)),
                    "Lambda2": float(meta.get("lambda2", args.lambda2)),
                    "LabelMode": str(meta.get("label_mode", args.label_mode)),
                    "InputFile": str(meta.get("input_file", "")),
                    "ROITimeSeriesFile": str(roi_file),
                }
                if is_during_window:
                    base.update({
                        "WindowIndex": int(meta.get("window_index")),
                        "WindowLabel": f"{float(meta.get('window_start_sec'))/60:.0f}-{float(meta.get('window_end_sec'))/60:.0f}min",
                        "WindowStartSec": float(meta.get("window_start_sec")),
                        "WindowEndSec": float(meta.get("window_end_sec")),
                        "WindowDurationSec": float(meta.get("window_duration_sec", float(meta.get("window_end_sec")) - float(meta.get("window_start_sec")))),
                        "NWindowEpochs": int(roi_ts.shape[0]),
                    })
                rows.append({**base, "Metric": "absolute", "Value": float(abs_val)})
                rows.append({**base, "Metric": "relative", "Value": float(rel_val)})
    return pd.DataFrame(rows)


def cleanup(*objs):
    for obj in objs:
        del obj
    gc.collect()


def add_common_source_args(parser: argparse.ArgumentParser):
    parser.add_argument("--epoch-duration", type=float, default=8.0)
    parser.add_argument("--epoch-overlap", type=float, default=0.0)
    parser.add_argument("--psd-fmin", type=float, default=1.0)
    parser.add_argument("--psd-fmax", type=float, default=50.0)
    parser.add_argument("--expected-sfreq", type=float, default=500.0)
    parser.add_argument("--montage", default="standard_1020")
    parser.add_argument("--method", default="dSPM", choices=["MNE", "dSPM", "sLORETA", "eLORETA"])
    parser.add_argument("--lambda2", type=float, default=1.0 / 9.0)
    parser.add_argument("--label-mode", default="mean", choices=["mean", "mean_flip", "pca_flip", "max"])
    parser.add_argument("--pick-ori", default="normal")
    parser.add_argument("--loose", type=float, default=0.2)
    parser.add_argument("--depth", type=float, default=0.8)
    parser.add_argument("--mindist", type=float, default=5.0)
    parser.add_argument("--n-jobs-forward", type=int, default=1)
    return parser


def _get_aperiodic_offset_exponent(fm, aperiodic_mode: str):
    params = np.asarray(fm.aperiodic_params_, dtype=float)
    if params.size == 0:
        return np.nan, np.nan, np.nan
    if aperiodic_mode == "knee" and params.size >= 3:
        return float(params[0]), float(params[2]), float(params[1])
    return float(params[0]), float(params[1]) if params.size >= 2 else np.nan, np.nan


def _get_fooof_aperiodic_fit(fm, aperiodic_mode: str):
    if hasattr(fm, "_ap_fit") and fm._ap_fit is not None:
        return np.asarray(fm._ap_fit, dtype=float)
    try:
        from fooof.sim.gen import gen_aperiodic
        return np.asarray(gen_aperiodic(fm.freqs, fm.aperiodic_params_, aperiodic_mode=aperiodic_mode), dtype=float)
    except Exception:
        freqs = np.asarray(fm.freqs, dtype=float)
        params = np.asarray(fm.aperiodic_params_, dtype=float)
        if aperiodic_mode == "knee" and params.size >= 3:
            offset, knee, exponent = params[:3]
            return offset - np.log10(knee + freqs ** exponent)
        if params.size >= 2:
            offset, exponent = params[:2]
            return offset - np.log10(freqs) * exponent
    return np.full_like(np.asarray(fm.freqs, dtype=float), np.nan)


def _find_peak_in_flattened_band(fm, band_low, band_high, aperiodic_mode, prominence=None):
    from scipy.signal import find_peaks

    freqs = np.asarray(fm.freqs, dtype=float)
    flat_spec = np.asarray(fm.power_spectrum, dtype=float) - _get_fooof_aperiodic_fit(fm, aperiodic_mode)
    mask = np.isfinite(freqs) & np.isfinite(flat_spec) & (freqs >= band_low) & (freqs <= band_high)
    if mask.sum() < 3:
        return np.nan, np.nan, 0, np.nan, np.nan, np.nan
    band_freqs = freqs[mask]
    band_flat = flat_spec[mask]
    max_idx = int(np.nanargmax(band_flat))
    peak_inds, _ = find_peaks(band_flat, **({"prominence": float(prominence)} if prominence is not None else {}))
    if len(peak_inds) == 0:
        return np.nan, np.nan, 0, float(band_freqs[max_idx]), float(band_flat[max_idx]), float(np.nanmean(band_flat))
    best_local = peak_inds[int(np.nanargmax(band_flat[peak_inds]))]
    return (
        float(band_freqs[best_local]),
        float(band_flat[best_local]),
        int(len(peak_inds)),
        float(band_freqs[max_idx]),
        float(band_flat[max_idx]),
        float(np.nanmean(band_flat)),
    )


def _count_fooof_gaussian_peaks_in_band(peak_params, band_low, band_high) -> int:
    if peak_params is None or len(peak_params) == 0:
        return 0
    peaks = np.asarray(peak_params, dtype=float)
    if peaks.ndim == 1:
        peaks = peaks.reshape(1, -1)
    if peaks.shape[1] < 1:
        return 0
    return int(np.sum((peaks[:, 0] >= band_low) & (peaks[:, 0] <= band_high)))


def fit_fooof_roi_features(psds_mean, freqs, label_names, peak_bands, args, base_metadata):
    try:
        from fooof import FOOOF
    except Exception as exc:
        raise ImportError("FOOOF is required. Install with: pip install fooof") from exc

    peak_bands = [b for b in peak_bands if b in PEAK_BANDS]
    if not peak_bands:
        raise ValueError(f"No valid peak bands requested. Valid options: {list(PEAK_BANDS)}")

    rows, qc_rows = [], []
    freq_range = [float(args.fooof_fmin), float(args.fooof_fmax)]
    prominence = getattr(args, "find_peaks_prominence", None)

    for roi_idx, roi_name in enumerate(label_names):
        psd = np.asarray(psds_mean[roi_idx, :], dtype=float)
        valid = np.isfinite(psd) & (psd > 0) & np.isfinite(freqs)
        fit_success = False
        fit_error_msg = ""
        offset = exponent = knee = r2 = error = np.nan
        n_peaks_total = 0
        peak_params = np.empty((0, 3))
        fm = None

        try:
            if valid.sum() < 5:
                raise RuntimeError("Too few valid positive PSD bins for FOOOF.")
            fm = FOOOF(
                peak_width_limits=[args.peak_width_min, args.peak_width_max],
                max_n_peaks=args.max_n_peaks,
                min_peak_height=args.min_peak_height,
                peak_threshold=args.peak_threshold,
                aperiodic_mode=args.aperiodic_mode,
                verbose=False,
            )
            fm.fit(freqs[valid], psd[valid], freq_range)
            offset, exponent, knee = _get_aperiodic_offset_exponent(fm, args.aperiodic_mode)
            r2 = float(fm.r_squared_) if fm.r_squared_ is not None else np.nan
            error = float(fm.error_) if fm.error_ is not None else np.nan
            peak_params = np.asarray(fm.peak_params_, dtype=float)
            n_peaks_total = int(len(peak_params)) if peak_params is not None else 0
            fit_success = bool(fm.has_model)
        except Exception as exc:
            fit_error_msg = repr(exc)

        qc_rows.append({
            **base_metadata,
            "ROI": roi_name,
            "ROIIndex": roi_idx,
            "FOOOFFitSuccess": fit_success,
            "FOOOFErrorMessage": fit_error_msg,
            "FOOOFR2": r2,
            "FOOOFError": error,
            "AperiodicOffset": offset,
            "AperiodicExponent": exponent,
            "AperiodicKnee": knee,
            "NFOOOFGaussianPeaksTotal": n_peaks_total,
            "FOOOF_fmin": args.fooof_fmin,
            "FOOOF_fmax": args.fooof_fmax,
            "AperiodicMode": args.aperiodic_mode,
            "MaxNPeaks": args.max_n_peaks,
            "PeakThreshold": args.peak_threshold,
            "MinPeakHeight": args.min_peak_height,
            "PeakWidthMin": args.peak_width_min,
            "PeakWidthMax": args.peak_width_max,
            "PeakMethod": "flattened_spectrum_find_peaks",
            "FindPeaksProminence": prominence,
        })

        for band_name in peak_bands:
            blo, bhi = PEAK_BANDS[band_name]
            peak_cf = peak_power = flat_max_cf = flat_max_power = band_mean_flat_power = np.nan
            n_local_peaks = 0
            n_gauss = _count_fooof_gaussian_peaks_in_band(peak_params, blo, bhi)
            if fit_success and fm is not None:
                peak_cf, peak_power, n_local_peaks, flat_max_cf, flat_max_power, band_mean_flat_power = _find_peak_in_flattened_band(
                    fm,
                    blo,
                    bhi,
                    args.aperiodic_mode,
                    prominence=prominence,
                )
            rows.append({
                **base_metadata,
                "ROI": roi_name,
                "ROIIndex": roi_idx,
                "PeakBand": band_name,
                "PeakBandLow": blo,
                "PeakBandHigh": bhi,
                "PeakCF": peak_cf,
                "PeakPower": peak_power,
                "NLocalPeaksInBand": n_local_peaks,
                "FlatMaxCF": flat_max_cf,
                "FlatMaxPower": flat_max_power,
                "BandMeanFlatPower": band_mean_flat_power,
                "NFOOOFGaussianPeaksInBand": n_gauss,
                "AperiodicOffset": offset,
                "AperiodicExponent": exponent,
                "AperiodicKnee": knee,
                "FOOOFR2": r2,
                "FOOOFError": error,
                "FOOOFFitSuccess": fit_success,
                "PeakMethod": "flattened_spectrum_find_peaks",
            })
    return pd.DataFrame(rows), pd.DataFrame(qc_rows)


def add_fooof_args(parser: argparse.ArgumentParser):
    parser.add_argument("--peak-bands", nargs="+", default=["Theta", "Alpha", "Beta", "Gamma"])
    parser.add_argument("--fooof-fmin", type=float, default=1.0)
    parser.add_argument("--fooof-fmax", type=float, default=50.0)
    parser.add_argument("--aperiodic-mode", default="fixed", choices=["fixed", "knee"])
    parser.add_argument("--peak-width-min", type=float, default=1.0)
    parser.add_argument("--peak-width-max", type=float, default=12.0)
    parser.add_argument("--max-n-peaks", type=int, default=6)
    parser.add_argument("--min-peak-height", type=float, default=0.0)
    parser.add_argument("--peak-threshold", type=float, default=2.0)
    parser.add_argument("--find-peaks-prominence", type=float, default=None)
    return parser

