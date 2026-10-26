# ============================================================
# FigS2_sensor_within_condition_topomap_atlases.py
#
# Supplementary Figure S2 (Supplementary S.2):
#   Sensor-level spectral dynamics across PBM meditation conditions.
#
# Generates the component topomap atlases (absolute power, relative power,
# and FOOOF features) for the DURING-PRE, POST-DURING, and POST-PRE
# contrasts, and the cluster-permutation / FDR tables behind the p_cluster
# values and channel counts reported in S.2.
#
# The final Supp_sensors_2.png montage was assembled by hand from these
# atlases.
# ============================================================

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import t as t_dist

from sensor_utils import (
    BANDS,
    CONDITIONS,
    CONDITION_LABELS,
    add_common_args,
    bh_fdr,
    ensure_section_dirs,
    expected_feature_paths,
    load_band_power,
    load_fooof,
    paired_change_by_channel,
    paired_stats,
    save_figure,
    save_table,
    transform_band_values,
)

try:
    import mne
    from mne.stats import permutation_cluster_1samp_test
except Exception:
    mne = None
    permutation_cluster_1samp_test = None


CONTRASTS = [("DURING", "PRE"), ("POST", "PRE"), ("POST", "DURING")]
FOOOF_FEATURES = ["alpha_peak_cf", "alpha_peak_pw", "aperiodic_exponent", "aperiodic_offset"]
CLUSTER_FORMING_ALPHA = 0.05
CLUSTER_ALPHA = 0.05
N_CLUSTER_PERMUTATIONS = 2000
MIN_SUBJECTS = 8
MIN_CHANNELS = 12
MIN_VISIBLE_VLIM = 1e-12


def make_info(channels: list[str]):
    if mne is None:
        return None, []
    montage = mne.channels.make_standard_montage("standard_1020")
    montage_names = set(montage.ch_names)
    good = [ch for ch in channels if ch in montage_names]
    if len(good) < 3:
        return None, []
    info = mne.create_info(good, sfreq=500.0, ch_types="eeg")
    info.set_montage(montage, on_missing="ignore")
    return info, good


def within_condition_long(change: pd.DataFrame, spec: dict) -> pd.DataFrame:
    if change.empty:
        return pd.DataFrame()
    out = change[change["Condition"] == spec["condition"]].copy()
    out["kind"] = spec["kind"]
    out["condition"] = spec["condition"]
    out["condition_label"] = CONDITION_LABELS[spec["condition"]]
    out["feature"] = spec.get("feature", f"{spec.get('metric')}_{spec.get('band')}")
    out["metric"] = spec.get("metric", "")
    out["band"] = spec.get("band", "")
    out["contrast"] = f"{spec['contrast'][0]}-{spec['contrast'][1]}"
    return out.rename(columns={"Change": "WithinConditionChange"})


def within_condition_stats(change: pd.DataFrame, spec: dict) -> pd.DataFrame:
    d = within_condition_long(change, spec)
    if d.empty:
        return pd.DataFrame()
    rows = []
    for channel, sub in d.groupby("Channel"):
        row = paired_stats(sub["WithinConditionChange"])
        row.update(
            {
                "kind": spec["kind"],
                "condition": spec["condition"],
                "condition_label": CONDITION_LABELS[spec["condition"]],
                "feature": spec.get("feature", f"{spec.get('metric')}_{spec.get('band')}"),
                "metric": spec.get("metric", ""),
                "band": spec.get("band", ""),
                "contrast": f"{spec['contrast'][0]}-{spec['contrast'][1]}",
                "channel": channel,
            }
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_perm_fdr_by_panel"] = bh_fdr(out["p_perm"])
    return out


def cluster_permutation_within(change: pd.DataFrame, spec: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    common = {
        "kind": spec["kind"],
        "condition": spec["condition"],
        "condition_label": CONDITION_LABELS[spec["condition"]],
        "feature": spec.get("feature", f"{spec.get('metric')}_{spec.get('band')}"),
        "metric": spec.get("metric", ""),
        "band": spec.get("band", ""),
        "contrast": f"{spec['contrast'][0]}-{spec['contrast'][1]}",
    }
    if mne is None or permutation_cluster_1samp_test is None:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame([{**common, "status": "mne_unavailable"}])
    d = within_condition_long(change, spec)
    if d.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame([{**common, "status": "empty_change_table"}])
    mat = d.pivot_table(index="Subject", columns="Channel", values="WithinConditionChange", aggfunc="mean")
    montage = mne.channels.make_standard_montage("standard_1020")
    montage_names = set(montage.ch_names)
    initial_channels = mat.columns.tolist()
    good_channels = [ch for ch in initial_channels if ch in montage_names]
    dropped_non_montage = [ch for ch in initial_channels if ch not in montage_names]
    mat_montage = mat.reindex(columns=good_channels)
    full_data_channels = mat_montage.columns[mat_montage.notna().all(axis=0)].tolist()
    mat = mat_montage.reindex(columns=full_data_channels).dropna(axis=0, how="any")
    audit = {
        **common,
        "status": "ok",
        "n_subjects_initial": int(mat_montage.shape[0]),
        "n_channels_initial": int(len(initial_channels)),
        "n_channels_in_standard_1020": int(len(good_channels)),
        "n_channels_dropped_non_montage": int(len(dropped_non_montage)),
        "dropped_non_montage_channels": ";".join(dropped_non_montage),
        "n_channels_complete_case": int(len(full_data_channels)),
        "n_subjects_complete_case": int(mat.shape[0]),
        "channels_used_for_cluster": ";".join(full_data_channels),
    }
    if mat.shape[0] < MIN_SUBJECTS or mat.shape[1] < MIN_CHANNELS:
        audit["status"] = "too_few_subjects_or_channels"
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame([audit])

    info = mne.create_info(mat.columns.tolist(), sfreq=500.0, ch_types="eeg")
    info.set_montage(montage, on_missing="ignore")
    adjacency, _ = mne.channels.find_ch_adjacency(info, ch_type="eeg")
    threshold = t_dist.ppf(1 - CLUSTER_FORMING_ALPHA / 2, mat.shape[0] - 1)
    n_perm = min(N_CLUSTER_PERMUTATIONS, max(1, (2 ** mat.shape[0]) - 1))

    try:
        t_obs, clusters, cluster_p_values, _ = permutation_cluster_1samp_test(
            mat.to_numpy(dtype=float),
            threshold=threshold,
            adjacency=adjacency,
            n_permutations=n_perm,
            tail=0,
            out_type="mask",
            seed=42,
            verbose=False,
        )
    except Exception as exc:
        audit["status"] = "cluster_error"
        audit["cluster_error"] = str(exc)
        return pd.DataFrame(), pd.DataFrame(
            [
                {
                    **common,
                    "cluster_error": str(exc),
                }
            ]
        ), pd.DataFrame([audit])

    channel_rows = []
    summary_rows = []
    channel_list = mat.columns.tolist()
    for cluster_id, (mask, p_value) in enumerate(zip(clusters, cluster_p_values), start=1):
        mask = np.asarray(mask, dtype=bool)
        channels = [ch for ch, keep in zip(channel_list, mask) if keep]
        sig = bool(p_value < CLUSTER_ALPHA)
        summary_rows.append(
            {
                **common,
                "cluster_id": cluster_id,
                "cluster_p_value": float(p_value),
                "cluster_sig": sig,
                "n_subjects": mat.shape[0],
                "n_channels_in_cluster": len(channels),
                "channels": ";".join(channels),
                "cluster_error": "",
            }
        )
        for ch in channels:
            channel_rows.append(
                {
                    **common,
                    "channel": ch,
                    "cluster_id": cluster_id,
                    "cluster_p_value": float(p_value),
                    "cluster_sig_perm": sig,
                    "cluster_t_obs": float(t_obs[channel_list.index(ch)]),
                }
            )
    audit["n_clusters_found"] = int(len(summary_rows))
    audit["n_sig_clusters"] = int(sum(row["cluster_sig"] for row in summary_rows)) if summary_rows else 0
    return pd.DataFrame(channel_rows), pd.DataFrame(summary_rows), pd.DataFrame([audit])


def band_change_table(band: pd.DataFrame, spec: dict) -> pd.DataFrame:
    phase_a, phase_b = spec["contrast"]
    sub = band[
        (band["Condition"] == spec["condition"])
        & (band["Metric"] == spec["metric"])
        & (band["Band"] == spec["band"])
    ].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["Value_t"] = transform_band_values(sub["Value"], spec["metric"])
    return paired_change_by_channel(sub, "Value_t", "Channel", phase_a, phase_b)


def band_coverage_table(band: pd.DataFrame, spec: dict) -> pd.DataFrame:
    phase_a, phase_b = spec["contrast"]
    sub = band[
        (band["Condition"] == spec["condition"])
        & (band["Metric"] == spec["metric"])
        & (band["Band"] == spec["band"])
    ].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["Value_t"] = transform_band_values(sub["Value"], spec["metric"])
    return phase_channel_coverage(sub, "Value_t", "Channel", spec, phase_a, phase_b)


def fooof_change_table(fooof: pd.DataFrame, spec: dict) -> pd.DataFrame:
    phase_a, phase_b = spec["contrast"]
    sub = fooof[
        (fooof["Condition"] == spec["condition"])
        & (fooof["Feature"] == spec["feature"])
        & (fooof["LocationKind"] == "Channel")
    ].copy()
    if sub.empty:
        return pd.DataFrame()
    sub = sub.rename(columns={"Location": "Channel"})
    return paired_change_by_channel(sub, "Value", "Channel", phase_a, phase_b)


def fooof_coverage_table(fooof: pd.DataFrame, spec: dict) -> pd.DataFrame:
    phase_a, phase_b = spec["contrast"]
    sub = fooof[
        (fooof["Condition"] == spec["condition"])
        & (fooof["Feature"] == spec["feature"])
        & (fooof["LocationKind"] == "Channel")
    ].copy()
    if sub.empty:
        return pd.DataFrame()
    sub = sub.rename(columns={"Location": "Channel"})
    return phase_channel_coverage(sub, "Value", "Channel", spec, phase_a, phase_b)


def phase_channel_coverage(df: pd.DataFrame, value_col: str, channel_col: str, spec: dict, phase_a: str, phase_b: str) -> pd.DataFrame:
    wide = df.pivot_table(index=["Subject", channel_col], columns="Phase", values=value_col, aggfunc="mean")
    for phase in [phase_a, phase_b]:
        if phase not in wide.columns:
            wide[phase] = np.nan
    out = wide[[phase_a, phase_b]].reset_index().rename(columns={channel_col: "channel"})
    out["has_phase_a"] = np.isfinite(pd.to_numeric(out[phase_a], errors="coerce"))
    out["has_phase_b"] = np.isfinite(pd.to_numeric(out[phase_b], errors="coerce"))
    out["has_complete_change"] = out["has_phase_a"] & out["has_phase_b"]
    out["missing_reason"] = np.select(
        [
            ~out["has_phase_a"] & ~out["has_phase_b"],
            ~out["has_phase_a"],
            ~out["has_phase_b"],
        ],
        [
            f"missing_{phase_a}_and_{phase_b}",
            f"missing_{phase_a}",
            f"missing_{phase_b}",
        ],
        default="complete",
    )
    out.insert(0, "kind", spec["kind"])
    out.insert(1, "condition", spec["condition"])
    out.insert(2, "condition_label", CONDITION_LABELS[spec["condition"]])
    out.insert(3, "feature", spec.get("feature", f"{spec.get('metric')}_{spec.get('band')}"))
    out.insert(4, "metric", spec.get("metric", ""))
    out.insert(5, "band", spec.get("band", ""))
    out.insert(6, "contrast", f"{phase_a}-{phase_b}")
    return out


def summarize_coverage(coverage: pd.DataFrame) -> pd.DataFrame:
    if coverage.empty:
        return pd.DataFrame()
    return (
        coverage.groupby(["kind", "condition", "condition_label", "feature", "metric", "band", "contrast", "channel", "missing_reason"], dropna=False)
        .size()
        .reset_index(name="n_subject_channel_rows")
    )


def panel_qc(df: pd.DataFrame, title: str, vlim, status: str = "ok") -> dict:
    values = pd.to_numeric(df["mean"], errors="coerce") if "mean" in df.columns else pd.Series(dtype=float)
    finite = values[np.isfinite(values)]
    if isinstance(vlim, tuple):
        vlim_abs = max(abs(float(vlim[0])), abs(float(vlim[1])))
        vlim_text = f"{float(vlim[0]):.4g},{float(vlim[1]):.4g}"
    elif vlim is not None and np.isfinite(vlim):
        vlim_abs = abs(float(vlim))
        vlim_text = f"{float(vlim):.4g}"
    else:
        vlim_abs = np.nan
        vlim_text = ""
    return {
        "panel": title,
        "status": status,
        "n_channels": int(len(df)) if df is not None else 0,
        "n_finite_channels": int(len(finite)),
        "n_nonfinite_channels": int(len(values) - len(finite)),
        "mean_abs": float(np.nanmean(np.abs(finite))) if len(finite) else np.nan,
        "max_abs": float(np.nanmax(np.abs(finite))) if len(finite) else np.nan,
        "vlim": vlim_text,
        "may_look_white": bool(len(finite) == 0 or (np.isfinite(vlim_abs) and np.nanmax(np.abs(finite)) < 0.05 * vlim_abs)),
    }


def robust_limits(values: pd.Series | np.ndarray, center_zero: bool = True) -> tuple[float, float] | None:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    low, high = np.nanpercentile(vals, [2, 98])
    if not np.isfinite(low) or not np.isfinite(high) or abs(high - low) <= MIN_VISIBLE_VLIM:
        low, high = np.nanmin(vals), np.nanmax(vals)
    if not np.isfinite(low) or not np.isfinite(high) or abs(high - low) <= MIN_VISIBLE_VLIM:
        return None
    if center_zero and low < 0 < high:
        vmax = max(abs(low), abs(high))
        return (-float(vmax), float(vmax))
    pad = 0.05 * abs(high - low)
    return (float(low - pad), float(high + pad))


def plot_topomap(ax, df: pd.DataFrame, title: str, vlim=None, add_colorbar: bool = True):
    if mne is None or df.empty:
        ax.axis("off")
        ax.set_title(f"{title}\nmissing", fontsize=7)
        return None, panel_qc(df, title, vlim, "missing")
    channels = df["channel"].tolist()
    info, good = make_info(channels)
    if info is None:
        ax.axis("off")
        ax.set_title(f"{title}\nno montage", fontsize=7)
        return None, panel_qc(df, title, vlim, "no_montage")
    sub = df.set_index("channel").reindex(good).reset_index()
    pre_filter_qc = panel_qc(sub, title, vlim, "pre_filter")
    finite_rows = np.isfinite(pd.to_numeric(sub["mean"], errors="coerce").to_numpy(dtype=float))
    sub = sub.loc[finite_rows].copy()
    if len(sub) < 3:
        ax.axis("off")
        ax.set_title(f"{title}\n<3 finite channels", fontsize=7)
        qc = panel_qc(sub, title, vlim, "too_few_finite_channels")
        qc["n_nonfinite_channels_before_filter"] = pre_filter_qc["n_nonfinite_channels"]
        return None, qc
    info, good = make_info(sub["channel"].tolist())
    if info is None:
        ax.axis("off")
        ax.set_title(f"{title}\nno montage after finite filter", fontsize=7)
        qc = panel_qc(sub, title, vlim, "no_montage_after_filter")
        qc["n_nonfinite_channels_before_filter"] = pre_filter_qc["n_nonfinite_channels"]
        return None, qc
    sub = sub.set_index("channel").reindex(good).reset_index()
    values = sub["mean"].to_numpy(dtype=float)
    if "cluster_sig_perm" in sub.columns and sub["cluster_sig_perm"].fillna(False).astype(bool).any():
        mask = sub["cluster_sig_perm"].fillna(False).astype(bool).to_numpy()
    else:
        mask = sub["q_perm_fdr_by_panel"].to_numpy(dtype=float) < 0.05
    if vlim is None:
        vlim = robust_limits(values, center_zero=True)
    if vlim is None:
        ax.axis("off")
        ax.set_title(f"{title}\nflat/no range", fontsize=7)
        return None, panel_qc(sub, title, vlim, "flat_no_range")
    im, _ = mne.viz.plot_topomap(
        values,
        info,
        axes=ax,
        show=False,
        cmap="RdBu_r",
        vlim=vlim,
        contours=0,
        sensors="k.",
        extrapolate="head",
        image_interp="cubic",
        mask=mask,
        mask_params=dict(marker="o", markerfacecolor="none", markeredgecolor="black", linewidth=0, markersize=5),
    )
    if add_colorbar:
        cax = inset_axes(
            ax,
            width="72%",
            height="5%",
            loc="lower center",
            bbox_to_anchor=(0, -0.14, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )
        cb = plt.colorbar(im, cax=cax, orientation="horizontal")
        cb.ax.tick_params(labelsize=5, length=2, pad=1)
        cb.outline.set_linewidth(0.4)
    ax.set_title(title, fontsize=7)
    qc = panel_qc(sub, title, vlim, "ok")
    qc["n_nonfinite_channels_before_filter"] = pre_filter_qc["n_nonfinite_channels"]
    return im, qc


def make_band_atlas(stats_df: pd.DataFrame, metric: str, contrast: str, scale_mode: str):
    sub = stats_df[(stats_df["kind"] == "band") & (stats_df["metric"] == metric) & (stats_df["contrast"] == contrast)].copy()
    if sub.empty:
        return None, pd.DataFrame()
    fig, axes = plt.subplots(len(CONDITIONS), len(BANDS), figsize=(2.55 * len(BANDS), 2.35 * len(CONDITIONS)), constrained_layout=True)
    qc_rows = []
    band_vlims = {
        band_name: robust_limits(sub.loc[sub["band"] == band_name, "mean"], center_zero=True)
        for band_name in BANDS
    }
    figure_vlim = robust_limits(sub["mean"], center_zero=True)
    last_im = None
    for r, condition in enumerate(CONDITIONS):
        for c, band_name in enumerate(BANDS):
            ax = axes[r, c]
            panel = sub[(sub["condition"] == condition) & (sub["band"] == band_name)]
            title = f"{CONDITION_LABELS[condition]} {band_name}"
            if scale_mode == "panel":
                vlim = None
            elif scale_mode == "column":
                vlim = band_vlims.get(band_name)
            else:
                vlim = figure_vlim
            im, qc = plot_topomap(ax, panel, title, vlim=vlim, add_colorbar=(scale_mode == "panel"))
            if im is not None:
                last_im = im
            qc.update({"atlas": f"{metric}_{contrast}", "condition": condition, "condition_label": CONDITION_LABELS[condition], "band": band_name, "metric": metric, "contrast": contrast})
            qc_rows.append(qc)
    if scale_mode == "figure" and last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), orientation="horizontal", fraction=0.025, pad=0.035)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label(f"{metric} power change ({contrast})", fontsize=9)
    fig.suptitle(f"Within-condition {metric} power change: {contrast}", fontsize=14)
    return fig, pd.DataFrame(qc_rows)


def make_fooof_atlas(stats_df: pd.DataFrame, contrast: str, scale_mode: str):
    sub = stats_df[(stats_df["kind"] == "fooof") & (stats_df["contrast"] == contrast)].copy()
    if sub.empty:
        return None, pd.DataFrame()
    fig, axes = plt.subplots(len(CONDITIONS), len(FOOOF_FEATURES), figsize=(2.8 * len(FOOOF_FEATURES), 2.35 * len(CONDITIONS)), constrained_layout=True)
    qc_rows = []
    feature_vlims = {
        feature: robust_limits(sub.loc[sub["feature"] == feature, "mean"], center_zero=True)
        for feature in FOOOF_FEATURES
    }
    figure_vlim = robust_limits(sub["mean"], center_zero=True)
    last_im = None
    for r, condition in enumerate(CONDITIONS):
        for c, feature in enumerate(FOOOF_FEATURES):
            panel = sub[(sub["condition"] == condition) & (sub["feature"] == feature)]
            title = f"{CONDITION_LABELS[condition]} {feature}"
            if scale_mode == "panel":
                vlim = None
            elif scale_mode == "column":
                vlim = feature_vlims.get(feature)
            else:
                vlim = figure_vlim
            im, qc = plot_topomap(axes[r, c], panel, title, vlim=vlim, add_colorbar=(scale_mode == "panel"))
            if im is not None:
                last_im = im
            qc.update({"atlas": f"FOOOF_{contrast}", "condition": condition, "condition_label": CONDITION_LABELS[condition], "feature": feature, "contrast": contrast})
            qc_rows.append(qc)
    if scale_mode == "figure" and last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), orientation="horizontal", fraction=0.025, pad=0.035)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label(f"FOOOF feature change ({contrast})", fontsize=9)
    fig.suptitle(f"Within-condition FOOOF feature change: {contrast}", fontsize=14)
    return fig, pd.DataFrame(qc_rows)


def delete_stale_pdfs(fig_dir) -> int:
    removed = 0
    for path in fig_dir.glob("*.pdf"):
        path.unlink()
        removed += 1
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description="Build organized within-condition topomap atlases with cluster permutation masks.")
    add_common_args(parser)
    parser.add_argument(
        "--scale-mode",
        choices=["figure", "column", "panel"],
        default="figure",
        help="figure = one shared colorbar for the whole atlas; column = one scale per band/feature column; panel = each topomap gets its own colorbar.",
    )
    args = parser.parse_args()
    dirs = ensure_section_dirs(args.out_root, "figureS2_within_condition_topomap_atlases")
    stale_pdfs_removed = delete_stale_pdfs(dirs["figures"]) if args.overwrite else 0

    paths = expected_feature_paths(args.data_root, args.scheme)
    band_path = paths.get("band_power_scheme") or paths.get("band_power_combined")
    fooof_path = paths.get("fooof_scheme") or paths.get("fooof_combined")
    if band_path is None:
        raise FileNotFoundError("Band-power CSV not found.")
    band = load_band_power(band_path, args.scheme)
    fooof = load_fooof(fooof_path, args.scheme) if fooof_path else pd.DataFrame()

    specs = []
    for condition in CONDITIONS:
        for contrast in CONTRASTS:
            for metric in ["relative", "absolute"]:
                for band_name in BANDS:
                    specs.append({"kind": "band", "condition": condition, "metric": metric, "band": band_name, "contrast": contrast})
            for feature in FOOOF_FEATURES:
                specs.append({"kind": "fooof", "condition": condition, "feature": feature, "contrast": contrast})

    stats_frames = []
    cluster_channel_frames = []
    cluster_summary_frames = []
    cluster_audit_frames = []
    coverage_frames = []
    for spec in specs:
        coverage = band_coverage_table(band, spec) if spec["kind"] == "band" else fooof_coverage_table(fooof, spec)
        if not coverage.empty:
            coverage_frames.append(coverage)
        change = band_change_table(band, spec) if spec["kind"] == "band" else fooof_change_table(fooof, spec)
        if change.empty:
            continue
        stats_df = within_condition_stats(change, spec)
        cluster_channels, cluster_summary, cluster_audit = cluster_permutation_within(change, spec)
        if not cluster_channels.empty:
            merge_cols = ["kind", "condition", "feature", "metric", "band", "contrast", "channel"]
            stats_df = stats_df.merge(cluster_channels, on=merge_cols, how="left", suffixes=("", "_cluster"))
            cluster_channel_frames.append(cluster_channels)
        if not cluster_summary.empty:
            cluster_summary_frames.append(cluster_summary)
        if not cluster_audit.empty:
            cluster_audit_frames.append(cluster_audit)
        stats_frames.append(stats_df)

    stats_all = pd.concat([x for x in stats_frames if not x.empty], ignore_index=True) if stats_frames else pd.DataFrame()
    if not stats_all.empty and "cluster_sig_perm" not in stats_all.columns:
        stats_all["cluster_sig_perm"] = False
    elif not stats_all.empty:
        stats_all["cluster_sig_perm"] = stats_all["cluster_sig_perm"].fillna(False).astype(bool)
    cluster_channels_all = pd.concat(cluster_channel_frames, ignore_index=True) if cluster_channel_frames else pd.DataFrame()
    cluster_summary_all = pd.concat(cluster_summary_frames, ignore_index=True) if cluster_summary_frames else pd.DataFrame()
    cluster_audit_all = pd.concat(cluster_audit_frames, ignore_index=True) if cluster_audit_frames else pd.DataFrame()
    coverage_all = pd.concat(coverage_frames, ignore_index=True) if coverage_frames else pd.DataFrame()
    coverage_summary = summarize_coverage(coverage_all)

    save_table(stats_all, dirs["tables"] / "figureS2_within_condition_sensor_stats_all_features.csv", args.overwrite)
    save_table(cluster_channels_all, dirs["tables"] / "figureS2_within_condition_cluster_channel_membership.csv", args.overwrite)
    save_table(cluster_summary_all, dirs["tables"] / "figureS2_within_condition_cluster_level_results.csv", args.overwrite)
    save_table(cluster_audit_all, dirs["tables"] / "figureS2_within_condition_cluster_channel_inclusion_audit.csv", args.overwrite)
    save_table(coverage_all, dirs["tables"] / "figureS2_within_condition_subject_channel_phase_coverage_long.csv", args.overwrite)
    save_table(coverage_summary, dirs["tables"] / "figureS2_within_condition_subject_channel_missingness_summary.csv", args.overwrite)

    qc_frames = []
    for contrast in [f"{a}-{b}" for a, b in CONTRASTS]:
        for metric in ["relative", "absolute"]:
            fig, qc = make_band_atlas(stats_all, metric, contrast, args.scale_mode)
            if not qc.empty:
                qc_frames.append(qc)
            if fig is not None:
                save_figure(fig, dirs["figures"] / f"FigureS2_WithinCondition_{metric}_power_{contrast}_rows_condition_cols_band", args.overwrite)
                plt.close(fig)
        fig, qc = make_fooof_atlas(stats_all, contrast, args.scale_mode)
        if not qc.empty:
            qc_frames.append(qc)
        if fig is not None:
            save_figure(fig, dirs["figures"] / f"FigureS2_WithinCondition_FOOOF_{contrast}_rows_condition_cols_feature", args.overwrite)
            plt.close(fig)
    qc_all = pd.concat(qc_frames, ignore_index=True) if qc_frames else pd.DataFrame()
    save_table(qc_all, dirs["tables"] / "figureS2_within_condition_topomap_panel_qc.csv", args.overwrite)

    report = [
        "# Within-Condition Topomap Atlases",
        "",
        "These figures are descriptive within-condition change maps, not sham-controlled causal contrasts.",
        "Rows are conditions; columns are bands or FOOOF features.",
        "Masks use MNE cluster permutation when significant clusters are present, otherwise sensorwise FDR.",
        "MNE log lines like `number of adjacent vertices` report the number of complete-case standard-1020 channels used for that panel's cluster test, not necessarily the total channels available in the dataset.",
        "See `figureS2_within_condition_cluster_channel_inclusion_audit.csv` to check whether expected channels were dropped because they were non-standard montage names or incomplete across paired subjects.",
        "See `figureS2_within_condition_subject_channel_phase_coverage_long.csv` and `figureS2_within_condition_subject_channel_missingness_summary.csv` to identify exactly which subject/channel/phase combinations are incomplete.",
        f"Color scale mode: `{args.scale_mode}`. `figure` uses one shared colorbar for the whole atlas; `column` uses one scale per band/feature column; `panel` gives every topomap its own scale and colorbar.",
        f"Stale PDFs removed from this figure folder on this run: {stale_pdfs_removed}.",
        "See `figureS2_within_condition_topomap_panel_qc.csv` for panels that are empty, flat, or likely to look white.",
        f"Band CSV: `{band_path}`",
        f"FOOOF CSV: `{fooof_path}`" if fooof_path else "FOOOF CSV missing; FOOOF atlases skipped.",
    ]
    (dirs["reports"] / "figureS2_within_condition_topomap_atlas_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Saved within-condition topomap atlases to: {dirs['base']}")


if __name__ == "__main__":
    main()
