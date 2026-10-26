# ============================================================
# sensor_utils.py
#
# Shared loading, feature, plotting, and export helpers for the sensor-level
# Figure S2 script (FigS2_sensor_within_condition_topomap_atlases.py).
# ============================================================

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATA_ROOT = PROJECT_ROOT / "Data"
DEFAULT_ZAPLINE_ROOT = PROJECT_ROOT / "Data" / "sensor_zapline_40"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "outputs" / "sensor_results"
DEFAULT_SCHEME = "theta4_8_alpha8_13"

CONDITIONS = ["Cond1", "Cond2", "Cond3", "Cond4", "Cond5", "Cond6"]
ACTIVE_CONDITIONS = ["Cond1", "Cond2", "Cond3", "Cond4"]
SHAM_CONDITION = "Cond5"
CONTROL_CONDITION = "Cond6"
PHASES = ["PRE", "DURING", "POST"]

CONDITION_LABELS = {
    "Cond1": "4Hz",
    "Cond2": "10Hz",
    "Cond3": "40Hz",
    "Cond4": "120Hz",
    "Cond5": "Sham",
    "Cond6": "Control",
}
LABEL_TO_CONDITION = {v.lower(): k for k, v in CONDITION_LABELS.items()}

BANDS = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
SCORES = ["MEQ_Total", "Transcendence", "PositiveMood", "Ineffability", "Mystical"]
POSTERIOR_PREFIXES = ("CP", "P", "PO", "O")
CPO_PREFIXES = ("FC", "C", "CP", "P", "PO", "O")
FRONTOCENTRAL_PREFIXES = ("F", "FC", "C")


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--zapline-root", type=Path, default=DEFAULT_ZAPLINE_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--scheme", default=DEFAULT_SCHEME)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def ensure_section_dirs(out_root: Path, section: str) -> dict[str, Path]:
    base = out_root / section
    dirs = {
        "base": base,
        "figures": base / "figures",
        "tables": base / "tables",
        "reports": base / "reports",
        "logs": base / "logs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().replace("\xa0", " ")


def normalize_subject(value) -> str:
    text = clean_text(value)
    match = re.search(r"(\d+)", text)
    if match:
        return str(int(match.group(1)))
    return text


def normalize_condition(value) -> str:
    text = clean_text(value)
    if text in CONDITION_LABELS:
        return text
    compact = re.sub(r"\s+", "", text).lower().replace("condition", "")
    mapping = {
        "cond1": "Cond1",
        "cond2": "Cond2",
        "cond3": "Cond3",
        "cond4": "Cond4",
        "cond5": "Cond5",
        "cond6": "Cond6",
        "4hz": "Cond1",
        "10hz": "Cond2",
        "40hz": "Cond3",
        "120hz": "Cond4",
        "sham": "Cond5",
        "control": "Cond6",
        "ctrl": "Cond6",
    }
    return mapping.get(compact, LABEL_TO_CONDITION.get(compact, text))


def normalize_phase(value) -> str:
    text = clean_text(value).upper()
    if text.startswith("PRE"):
        return "PRE"
    if text.startswith("DUR"):
        return "DURING"
    if text.startswith("POST"):
        return "POST"
    return text


def normalize_ch_name(ch) -> str:
    text = clean_text(ch)
    special = {
        "FP1": "Fp1",
        "FP2": "Fp2",
        "FPZ": "Fpz",
        "AFZ": "AFz",
        "FZ": "Fz",
        "FCZ": "FCz",
        "CZ": "Cz",
        "CPZ": "CPz",
        "PZ": "Pz",
        "POZ": "POz",
        "OZ": "Oz",
    }
    upper = text.upper()
    if upper in special:
        return special[upper]
    match = re.match(r"^([A-Za-z]+)(\d+)$", text)
    if match:
        letters, nums = match.groups()
        return letters.upper() + nums
    return text


def resolve_column(df: pd.DataFrame, candidates: Iterable[str], required: bool = True) -> str | None:
    norm = {str(col).strip().lower(): col for col in df.columns}
    for candidate in candidates:
        hit = norm.get(candidate.lower())
        if hit is not None:
            return hit
    if required:
        raise KeyError(f"Missing column. Tried {list(candidates)}. Available: {list(df.columns)}")
    return None


def find_one(base: Path, patterns: list[str]) -> Path | None:
    if not base.exists():
        return None
    for pattern in patterns:
        hits = sorted(base.rglob(pattern))
        if hits:
            return hits[0]
    return None


def expected_feature_paths(data_root: Path, scheme: str = DEFAULT_SCHEME) -> dict[str, Path | None]:
    figures = data_root / "Figures"
    if not figures.exists():
        figures = data_root
    band_interpolated = figures / "band_power_dual_scheme_interpolated"
    band_legacy = figures / "band_power_dual_scheme"
    fooof_dir = figures / "fooof_dual_scheme"

    band_scheme = band_interpolated / f"band_power_{scheme}_all_subjects_conditions_long.csv"
    band_combined = band_interpolated / "band_power_all_schemes_all_subjects_conditions_long.csv"
    legacy_band_scheme = band_legacy / f"band_power_{scheme}_all_subjects_conditions_long.csv"
    legacy_band_combined = band_legacy / "band_power_all_schemes_all_subjects_conditions_long.csv"
    fooof_scheme = fooof_dir / f"fooof_{scheme}_all_subjects_conditions_long.csv"
    fooof_combined = fooof_dir / "fooof_all_schemes_all_subjects_conditions_long.csv"

    return {
        "band_power_combined": band_combined if band_combined.exists() else (legacy_band_combined if legacy_band_combined.exists() else find_one(figures, ["band_power_all_schemes_all_subjects_conditions_long.csv"])),
        "band_power_scheme": band_scheme if band_scheme.exists() else (legacy_band_scheme if legacy_band_scheme.exists() else find_one(figures, [f"band_power_{scheme}_all_subjects_conditions_long.csv"])),
        "fooof_combined": fooof_combined if fooof_combined.exists() else find_one(figures, ["fooof_all_schemes_all_subjects_conditions_long.csv"]),
        "fooof_scheme": fooof_scheme if fooof_scheme.exists() else find_one(figures, [f"fooof_{scheme}_all_subjects_conditions_long.csv"]),
        "meditation_scores": find_one(figures, ["meditation_scores_long.csv"]),
        "band_cluster_stats": find_one(figures, [f"band_power_dual_scheme_interpolated/{scheme}/stats/cluster_test_summary.csv"]),
        "band_channel_stats": find_one(figures, [f"band_power_dual_scheme_interpolated/{scheme}/stats/channel_level_t_and_cluster_membership.csv"]),
        "fooof_cluster_stats": find_one(figures, [f"fooof_dual_scheme/{scheme}/continuous_stats_rerun/cluster_test_summary.csv", f"fooof_dual_scheme/{scheme}/continuous_stats/cluster_test_summary.csv"]),
        "fooof_channel_stats": find_one(figures, [f"fooof_dual_scheme/{scheme}/continuous_stats_rerun/channel_level_t_and_cluster_membership.csv", f"fooof_dual_scheme/{scheme}/continuous_stats/channel_level_t_and_cluster_membership.csv"]),
    }


def load_band_power(path: Path, scheme: str = DEFAULT_SCHEME) -> pd.DataFrame:
    df = pd.read_csv(path)
    c_subject = resolve_column(df, ["Subject"])
    c_condition = resolve_column(df, ["Condition"])
    c_phase = resolve_column(df, ["Phase"])
    c_channel = resolve_column(df, ["Channel"])
    c_band = resolve_column(df, ["Band"])
    c_metric = resolve_column(df, ["Metric"])
    c_value = resolve_column(df, ["Value", "Value_t"])
    c_scheme = resolve_column(df, ["BandScheme", "Band_Scheme"], required=False)
    out = pd.DataFrame(
        {
            "Subject": df[c_subject].map(normalize_subject),
            "Condition": df[c_condition].map(normalize_condition),
            "ConditionLabel": df[c_condition].map(normalize_condition).map(CONDITION_LABELS),
            "Phase": df[c_phase].map(normalize_phase),
            "Channel": df[c_channel].map(normalize_ch_name),
            "Band": df[c_band].astype(str),
            "Metric": df[c_metric].astype(str).str.lower(),
            "Value": pd.to_numeric(df[c_value], errors="coerce"),
            "BandScheme": df[c_scheme].astype(str) if c_scheme is not None else scheme,
        }
    )
    out = out[(out["Condition"].isin(CONDITIONS)) & (out["BandScheme"] == scheme)].copy()
    return out


def load_fooof(path: Path, scheme: str = DEFAULT_SCHEME) -> pd.DataFrame:
    df = pd.read_csv(path)
    c_subject = resolve_column(df, ["Subject"])
    c_condition = resolve_column(df, ["Condition"])
    c_phase = resolve_column(df, ["Phase"])
    c_feature = resolve_column(df, ["Feature"])
    c_value = resolve_column(df, ["Value", "Value_t"])
    c_scheme = resolve_column(df, ["BandScheme", "Band_Scheme"], required=False)
    c_channel = resolve_column(df, ["Channel"], required=False)
    c_roi = resolve_column(df, ["ROI"], required=False)
    c_type = resolve_column(df, ["FeatureType"], required=False)
    if c_channel is None and c_roi is None:
        raise KeyError("FOOOF file must include either Channel or ROI.")
    loc_col = c_channel if c_channel is not None else c_roi
    loc_kind = "Channel" if c_channel is not None else "ROI"
    out = pd.DataFrame(
        {
            "Subject": df[c_subject].map(normalize_subject),
            "Condition": df[c_condition].map(normalize_condition),
            "ConditionLabel": df[c_condition].map(normalize_condition).map(CONDITION_LABELS),
            "Phase": df[c_phase].map(normalize_phase),
            "Location": df[loc_col].map(normalize_ch_name if loc_kind == "Channel" else clean_text),
            "LocationKind": loc_kind,
            "Feature": df[c_feature].astype(str),
            "Value": pd.to_numeric(df[c_value], errors="coerce"),
            "BandScheme": df[c_scheme].astype(str) if c_scheme is not None else scheme,
            "FeatureType": df[c_type].astype(str) if c_type is not None else "",
        }
    )
    out = out[(out["Condition"].isin(CONDITIONS)) & (out["BandScheme"] == scheme)].copy()
    return out


def load_scores(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    c_subject = resolve_column(df, ["Subject"])
    c_condition = resolve_column(df, ["Condition"])
    score_cols = [col for col in SCORES if col in df.columns]
    out = df[[c_subject, c_condition] + score_cols].copy()
    out = out.rename(columns={c_subject: "Subject", c_condition: "Condition"})
    out["Subject"] = out["Subject"].map(normalize_subject)
    out["Condition"] = out["Condition"].map(normalize_condition)
    out["ConditionLabel"] = out["Condition"].map(CONDITION_LABELS)
    return out[out["Condition"].isin(CONDITIONS)].copy()


def transform_band_values(values: pd.Series | np.ndarray, metric: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if metric.lower() == "absolute":
        return np.log10(np.maximum(arr, np.finfo(float).tiny))
    return arr


def region_mask(channels: Iterable[str], region: str) -> list[bool]:
    region = region.lower()
    masks = []
    for ch in channels:
        up = normalize_ch_name(ch).upper()
        if region in {"whole", "wholescap", "wholescalp", "global"}:
            masks.append(True)
        elif region in {"posterior", "po"}:
            masks.append(up.startswith(POSTERIOR_PREFIXES))
        elif region in {"cpo", "centralparietaloccipital"}:
            masks.append(up.startswith(CPO_PREFIXES))
        elif region in {"frontocentral", "fc"}:
            masks.append(up.startswith(FRONTOCENTRAL_PREFIXES))
        else:
            masks.append(True)
    return masks


def summarize_roi_band(df: pd.DataFrame, metric: str, band: str, region: str = "global") -> pd.DataFrame:
    sub = df[(df["Metric"] == metric.lower()) & (df["Band"] == band)].copy()
    if sub.empty:
        return pd.DataFrame()
    sub = sub.loc[region_mask(sub["Channel"], region)].copy()
    sub["Value_t"] = transform_band_values(sub["Value"], metric)
    return (
        sub.groupby(["Subject", "Condition", "ConditionLabel", "Phase"], as_index=False)["Value_t"]
        .mean()
        .rename(columns={"Value_t": "Value"})
    )


def summarize_roi_fooof(df: pd.DataFrame, feature: str, region: str = "global") -> pd.DataFrame:
    sub = df[(df["Feature"] == feature) & (df["LocationKind"] == "Channel")].copy()
    if sub.empty:
        return pd.DataFrame()
    sub = sub.loc[region_mask(sub["Location"], region)].copy()
    return (
        sub.groupby(["Subject", "Condition", "ConditionLabel", "Phase"], as_index=False)["Value"]
        .mean()
    )


def phase_change(df: pd.DataFrame, phase_a: str, phase_b: str) -> pd.DataFrame:
    if df.empty:
        return df
    wide = df.pivot_table(
        index=["Subject", "Condition", "ConditionLabel"],
        columns="Phase",
        values="Value",
        aggfunc="mean",
    )
    if phase_a not in wide.columns or phase_b not in wide.columns:
        return pd.DataFrame()
    out = wide[[phase_a, phase_b]].dropna().copy()
    out["Change"] = out[phase_a] - out[phase_b]
    return out.reset_index()


def paired_change_by_channel(
    df: pd.DataFrame,
    value_col: str,
    channel_col: str,
    phase_a: str,
    phase_b: str,
) -> pd.DataFrame:
    wide = df.pivot_table(
        index=["Subject", "Condition", channel_col],
        columns="Phase",
        values=value_col,
        aggfunc="mean",
    )
    if phase_a not in wide.columns or phase_b not in wide.columns:
        return pd.DataFrame()
    wide = wide[[phase_a, phase_b]].dropna().copy()
    wide["Change"] = wide[phase_a] - wide[phase_b]
    return wide.reset_index().rename(columns={channel_col: "Channel"})


def signflip_p(values: Iterable[float], n_perm: int = 5000, seed: int = 42) -> float:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return math.nan
    observed = abs(np.mean(x))
    max_exact = 2 ** x.size
    rng = np.random.default_rng(seed)
    if max_exact <= n_perm:
        signs = np.array(np.meshgrid(*[[-1, 1]] * x.size)).T.reshape(-1, x.size)
    else:
        signs = rng.choice([-1, 1], size=(n_perm, x.size))
    null = np.abs((signs * x).mean(axis=1))
    return float((np.sum(null >= observed) + 1) / (len(null) + 1))


def paired_stats(values: Iterable[float], n_perm: int = 5000) -> dict[str, float]:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return {"n": int(x.size), "mean": math.nan, "sem": math.nan, "t": math.nan, "p_t": math.nan, "p_wilcoxon": math.nan, "p_perm": math.nan, "cohens_dz": math.nan}
    try:
        t_res = stats.ttest_1samp(x, 0.0, nan_policy="omit")
        p_t = float(t_res.pvalue)
        t_val = float(t_res.statistic)
    except Exception:
        t_val, p_t = math.nan, math.nan
    try:
        p_w = float(stats.wilcoxon(x).pvalue)
    except Exception:
        p_w = math.nan
    sd = np.nanstd(x, ddof=1)
    return {
        "n": int(x.size),
        "mean": float(np.nanmean(x)),
        "sem": float(sd / np.sqrt(x.size)),
        "t": t_val,
        "p_t": p_t,
        "p_wilcoxon": p_w,
        "p_perm": signflip_p(x, n_perm=n_perm),
        "cohens_dz": float(np.nanmean(x) / sd) if sd > 0 else math.nan,
    }


def bh_fdr(pvals: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(pvals), dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    valid = np.isfinite(p)
    if not valid.any():
        return q
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    adj = np.empty_like(ranked)
    prev = 1.0
    n = len(ranked)
    for i in range(n - 1, -1, -1):
        val = ranked[i] * n / (i + 1)
        prev = min(prev, val)
        adj[i] = prev
    restored = np.empty_like(adj)
    restored[order] = np.clip(adj, 0, 1)
    q[valid] = restored
    return q


def save_table(df: pd.DataFrame, path: Path, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {path}")
    df.to_csv(path, index=False)


def save_figure(fig, stem: Path, overwrite: bool = False, dpi: int = 300) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    path = stem.with_suffix(".png")
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {path}")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")


def markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    return df.head(max_rows).to_markdown(index=False)
