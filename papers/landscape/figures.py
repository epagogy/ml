#!/usr/bin/env python3
"""
Publication figures for the Leakage Landscape paper.

Usage:
    python3 figures_landscape.py
    python3 figures_landscape.py --show
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch

# Paths

PAPER = Path(__file__).parent
# Override via: python figures_landscape.py --results-dir /path/to/data
RESULTS = PAPER / "data"
V1 = RESULTS / "leakage_landscape_v1_final.jsonl"
V2 = RESULTS / "leakage_landscape_v2.jsonl"
V3_AN = RESULTS / "v3" / "v3_an.jsonl"
V3_AP = RESULTS / "v3" / "v3_ap.jsonl"
V3_AO = RESULTS / "v3" / "v3_ao_merged.jsonl"

# Design Tokens
# White canvas, zero grid, outward ticks.

# Palette — 3 semantic hues. Muted but distinct. Not "fun" — authoritative.
C = {
    "I":   "#3a8a7e",   # forest teal — stable, resolved
    "II":  "#c75637",   # earthy red — urgent, the finding
    "III": "#4a6fa5",   # slate blue — systematic, cool
}
GRAY    = "#3d3d3d"     # primary text — dark enough to read, not black
MID     = "#808080"     # secondary text, annotations
LIGHT   = "#aaaaaa"     # tertiary, disabled
SPINE   = "#cccccc"     # spine color — present but invisible
ZERO_C  = "#999999"     # zero-reference

# Typography — Helvetica Neue is the journal standard
FS_LABEL   = 9
FS_TICK    = 7.5
FS_TITLE   = 9
FS_ANNOT   = 7
FS_LEGEND  = 7

# Line weights — thinner than default, sharper
LW_SPINE   = 0.5
LW_HAIR    = 0.3
LW_THIN    = 0.6
LW_MED     = 1.0
LW_BOLD    = 1.6
LW_HEAVY   = 3.0

# Markers
MS_SM      = 3
MS_MD      = 4.5
MS_LG      = 6

# Alpha
A_GHOST    = 0.03
A_WHISPER  = 0.08
A_SUBTLE   = 0.18
A_MED      = 0.35
A_SOLID    = 0.75
A_FULL     = 1.0


# Style: Nature/Science baseline

def setup():
    plt.rcParams.update({
        # Canvas
        "figure.facecolor": "white",
        "axes.facecolor": "white",

        # Spines — present but quiet
        "axes.edgecolor": SPINE,
        "axes.linewidth": LW_SPINE,
        "axes.spines.top": False,
        "axes.spines.right": False,

        # NO grid — data speaks for itself
        "axes.grid": False,
        "axes.axisbelow": True,

        # Ticks — outward, small, thin
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": LW_SPINE,
        "ytick.major.width": LW_SPINE,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.major.pad": 4,
        "ytick.major.pad": 4,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "xtick.color": GRAY,
        "ytick.color": GRAY,

        # Typography
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial"],
        "axes.labelsize": FS_LABEL,
        "axes.titlesize": FS_TITLE,
        "axes.labelcolor": GRAY,
        "axes.titleweight": "medium",
        "mathtext.default": "regular",

        # Legend
        "legend.fontsize": FS_LEGEND,
        "legend.frameon": False,
        "legend.handlelength": 1.2,
        "legend.labelcolor": MID,

        # Output
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


# Helpers

def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def ok1(v1): return [r for r in v1 if r.get("status") == "ok"]
def ok2(v2): return [r for r in v2 if r.get("v2_status") == "ok"]
def ok3(v3): return [r for r in v3 if r.get("v3_status") == "ok"]

def cl(xs):
    return np.array([x for x in xs if x is not None and (isinstance(x, (int, float)) and not math.isnan(x))])

def dz(xs):
    xs = cl(xs)
    if len(xs) < 2 or xs.std() == 0:
        return float("nan")
    return xs.mean() / xs.std()

def _zero(ax, axis="h"):
    """Draw zero reference line."""
    if axis == "h":
        ax.axhline(0, color=ZERO_C, linewidth=LW_THIN, alpha=0.3, zorder=0)
    else:
        ax.axvline(0, color=ZERO_C, linewidth=LW_THIN, alpha=0.3, zorder=0)

def _save(fig, name):
    fig.tight_layout()
    fig_dir = PAPER / "figures"
    fig_dir.mkdir(exist_ok=True)
    fig.savefig(fig_dir / f"{name}.pdf")
    plt.close(fig)


# FIGURE 1: RAINCLOUD FOREST PLOT

def fig1_raincloud(v1_data, v2_data):
    o1 = ok1(v1_data)
    o2 = ok2(v2_data)

    exps = [
        ("PCA leakage",         "I",  cl([r.get("ab_diff") for r in o2])),
        ("Normalization (LR)",  "I",  cl([r["a_lr_gap_diff"] for r in o1 if "a_lr_gap_diff" in r])),
        ("Normalization (RF)",  "I",  cl([r["a_rf_gap_diff"] for r in o1 if "a_rf_gap_diff" in r])),
        ("Outlier removal 10%", "I",  cl([r.get("e_lr_gap_diff_10") for r in o1])),
        ("Outlier removal 30%", "I",  cl([r.get("e_lr_gap_diff_30") for r in o1])),
        ("Chained estimation",  "I",  cl([r.get("ce_diff") for r in o2])),
        ("Calibration",         "I",  cl([r.get("af_diff") for r in o2])),
        ("Seed inflation",      "II", cl([r.get("ai_inflation") for r in o2])),
        ("Peeking k=10",        "II", cl([r.get("b_infl_k10") for r in o1])),
        ("Peeking k=5",         "II", cl([r.get("b_infl_k5") for r in o1])),
        ("Early stopping",      "II", cl([r.get("bb_diff") for r in o2])),
        ("Target encoding",     "II", cl([r.get("ac_diff") for r in o2])),
        ("Algorithm screen",    "II", cl([r.get("aq_k1_optimism") for r in o2])),
        ("OOF stacking",        "II", cl([r.get("ak_diff") for r in o2])),
        ("Duplication 30% RF",  "III", cl([r.get("h_rf_30") for r in o1])),
        ("Duplication 10% RF",  "III", cl([r.get("h_rf_10") for r in o1])),
        ("Duplication 5% RF",   "III", cl([r.get("h_rf_05") for r in o1])),
        ("Duplication 10% LR",  "III", cl([r.get("h_lr_10") for r in o1])),
    ]

    exps = [(lab, cls, d) for lab, cls, d in exps if len(d) > 50]
    class_ord = {"I": 0, "II": 1, "III": 2}
    exps.sort(key=lambda e: (class_ord[e[1]], -np.median(e[2])))

    n = len(exps)
    row_h = 0.48
    fig, ax = plt.subplots(figsize=(6.5, row_h * n + 0.6))

    vheight = 0.38

    for i, (label, cls, diffs) in enumerate(exps):
        y = n - 1 - i
        color = C[cls]

        # Half-violin
        try:
            kde = gaussian_kde(diffs, bw_method=0.25)
            x_range = np.linspace(
                np.percentile(diffs, 0.5),
                np.percentile(diffs, 99.5),
                250
            )
            density = kde(x_range)
            density = density / density.max() * vheight

            ax.fill_between(x_range, y, y + density, color=color, alpha=A_SUBTLE + 0.04,
                            linewidth=0, zorder=2)
            ax.plot(x_range, y + density, color=color, linewidth=LW_THIN,
                    alpha=A_SOLID, zorder=3)
        except Exception:
            pass

        # Summary: IQR + 90% interval + median
        p5, p25, p50, p75, p95 = np.percentile(diffs, [5, 25, 50, 75, 95])

        # 90% interval (thin)
        ax.plot([p5, p95], [y, y], color=color, linewidth=LW_MED,
                solid_capstyle="round", zorder=4, alpha=A_SOLID)
        # IQR (thick)
        ax.plot([p25, p75], [y, y], color=color, linewidth=LW_HEAVY,
                solid_capstyle="round", zorder=5, alpha=0.8)
        # Median: open circle
        ax.plot(p50, y, "o", color="white", markersize=MS_SM + 1,
                zorder=7, markeredgecolor=color, markeredgewidth=LW_MED)

    # Zero reference
    ax.axvline(0, color=ZERO_C, linewidth=LW_THIN, linestyle="-", alpha=0.25, zorder=1)

    # Class bands — barely there
    for cls_key in ["I", "II", "III"]:
        indices = [i for i, (_, c, _) in enumerate(exps) if c == cls_key]
        if not indices:
            continue
        y_lo = n - 1 - max(indices) - 0.5
        y_hi = n - 1 - min(indices) + 0.5
        ax.axhspan(y_lo, y_hi, color=C[cls_key], alpha=0.025, zorder=0)

    # Y labels — colored by class
    ax.set_yticks(range(n))
    labels = [e[0] for e in exps]
    labels.reverse()
    ax.set_yticklabels(labels, fontsize=FS_TICK)

    for i, (_, cls, _) in enumerate(exps):
        tick_idx = n - 1 - i
        ax.get_yticklabels()[tick_idx].set_color(C[cls])

    ax.set_xlabel("$\\Delta$AUC (leaky $-$ clean)", fontsize=FS_LABEL, color=GRAY)
    ax.set_xlim(-0.08, 0.11)
    ax.set_ylim(-0.6, n - 0.3)
    ax.tick_params(axis="y", length=0)  # no y ticks for this chart

    # Legend: bottom right, minimal patches
    legend_elements = [
        Patch(facecolor=C["I"],   alpha=0.5, edgecolor="none", label="Class I: Estimation"),
        Patch(facecolor=C["II"],  alpha=0.5, edgecolor="none", label="Class II: Selection"),
        Patch(facecolor=C["III"], alpha=0.5, edgecolor="none", label="Class III: Memorization"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=FS_ANNOT,
              handlelength=0.8, handleheight=0.6, borderpad=0.3, labelcolor=GRAY)

    _save(fig, "fig1_raincloud")
    print(f"  fig1_raincloud.pdf: {n} experiments, raincloud forest plot")
    return fig


# FIGURE 2: N-SCALING (2×2)

def fig2_n_scaling(v3_an_data):
    o = ok3(v3_an_data)
    ns_main = [50, 100, 200, 500, 1000, 2000]
    ns_ext  = [50, 100, 200, 500, 1000, 5000, 10000]

    # Split main (n_full=2000) vs extension (n_full=10000)
    o_main = [r for r in o if r.get("an_n_full") == 2000]
    o_ext  = [r for r in o if r.get("an_n_full") == 10000]

    # All 4 types get extension points if extension data exists for them
    # Only Class II (peeking, seed) gets extension to 5K/10K:
    # - Normalization: already zero at n=200, extension adds nothing
    # - Oversampling: only 59 of 152 ext datasets have sufficient imbalance,
    #   producing survivorship bias (N drops 149→59, spurious uptick)
    has_ext = {"an_peeking_means", "an_seed_means"}

    series = {
        "Peeking (Class II)":        ("an_peeking_means",   "II"),
        "Seed (Class II)":           ("an_seed_means",       "II"),
        "Normalization (Class I)":   ("an_normalize_means",  "I"),
    }

    over_ok = [r for r in o if r.get("an_oversample_means") and
               r["an_oversample_means"][0] is not None]
    if over_ok:
        series["Oversampling (Class III)"] = ("an_oversample_means", "III")

    n_panels = len(series)
    fig, axes = plt.subplots(2, 2, figsize=(5.8, 5.2), sharey=False)
    axes = axes.flatten()

    panel_labels = "abcd"
    color_override = {"Seed (Class II)": "#d4652e"}

    def _get_val(row, field, target_n):
        """Extract value for a specific n-level from a row."""
        if field not in row or row[field] is None:
            return None
        vals = row[field]
        levels = row.get("an_n_levels", ns_main)
        if target_n not in levels:
            return None
        idx = levels.index(target_n)
        if idx < len(vals) and vals[idx] is not None and not math.isnan(vals[idx]):
            return vals[idx]
        return None

    def _extract_trajs(rows, field, n_levels):
        """Extract complete trajectories (rows with all n-levels present)."""
        trajs = []
        for r in rows:
            traj = []
            for n in n_levels:
                v = _get_val(r, field, n)
                if v is None:
                    break
                traj.append(v)
            if len(traj) == len(n_levels):
                trajs.append(traj)
        return np.array(trajs) if trajs else np.empty((0, len(n_levels)))

    for idx, (title, (field, cls)) in enumerate(series.items()):
        ax = axes[idx]
        color = color_override.get(title, C[cls])
        use_ext = field in has_ext and len(o_ext) > 0

        if "oversample" in field:
            src_main = [r for r in over_ok if r.get("an_n_full") == 2000]
            src_ext  = [r for r in over_ok if r.get("an_n_full") == 10000]
        else:
            src_main = o_main
            src_ext  = o_ext

        # Determine n-levels for this panel
        ns = ns_ext if use_ext else ns_main

        # Extract trajectories
        main_arr = _extract_trajs(src_main, field, ns_main)
        ext_arr  = _extract_trajs(src_ext, field, ns_ext) if use_ext else np.empty((0, 0))

        rng = np.random.RandomState(42)

        # Spaghetti: main rows (6 points, stop at 2K)
        n_show_main = min(len(main_arr), 200)
        if n_show_main > 0:
            idx_m = rng.choice(len(main_arr), n_show_main, replace=False)
            for j in idx_m:
                ax.plot(ns_main, main_arr[j], color=color, alpha=A_GHOST,
                        linewidth=LW_HAIR, zorder=1)

        # Spaghetti: extension rows (8 points, 50..10K)
        if use_ext and len(ext_arr) > 0:
            n_show_ext = min(len(ext_arr), 50)
            idx_e = rng.choice(len(ext_arr), n_show_ext, replace=False)
            for j in idx_e:
                ax.plot(ns_ext, ext_arr[j], color=color, alpha=A_GHOST,
                        linewidth=LW_HAIR, zorder=1)

        # Ribbon: means at each n-level from appropriate pools
        means = []
        ci_lo = []
        ci_hi = []
        for i, n in enumerate(ns):
            vals = []
            # Main rows contribute at n <= 2000
            if n <= 2000:
                mi = ns_main.index(n) if n in ns_main else None
                if mi is not None and mi < main_arr.shape[1]:
                    vals.extend(main_arr[:, mi].tolist())
            # Extension rows contribute at n <= 1000 (shared base) and n >= 5000
            if use_ext and len(ext_arr) > 0:
                ei = ns_ext.index(n) if n in ns_ext else None
                if ei is not None and ei < ext_arr.shape[1]:
                    if n <= 1000 or n >= 5000:
                        vals.extend(ext_arr[:, ei].tolist())
            if vals:
                m = np.mean(vals)
                se = np.std(vals) / np.sqrt(len(vals))
                means.append(m)
                ci_lo.append(m - 1.96 * se)
                ci_hi.append(m + 1.96 * se)
            else:
                means.append(np.nan)
                ci_lo.append(np.nan)
                ci_hi.append(np.nan)

        means = np.array(means)
        ci_lo = np.array(ci_lo)
        ci_hi = np.array(ci_hi)

        ax.fill_between(ns, ci_lo, ci_hi, color=color, alpha=A_SUBTLE, zorder=3)
        ax.plot(ns, means, color=color, linewidth=LW_BOLD, zorder=4,
                marker="o", markersize=MS_SM,
                markeredgecolor="white", markeredgewidth=LW_HAIR)

        # Annotate dataset counts at tier boundaries
        n_main_count = len(main_arr)
        i_2k = ns.index(2000) if 2000 in ns else None
        if i_2k is not None and not np.isnan(means[i_2k]):
            ax.annotate(f"N={n_main_count}", xy=(2000, means[i_2k]),
                        fontsize=FS_ANNOT, color=MID, ha="right", va="bottom",
                        xytext=(-4, 4), textcoords="offset points")
        if use_ext and len(ext_arr) > 0:
            n_ext_count = len(ext_arr)
            i_10k = ns.index(10000) if 10000 in ns else None
            if i_10k is not None and not np.isnan(means[i_10k]):
                ax.annotate(f"N={n_ext_count}", xy=(10000, means[i_10k]),
                            fontsize=FS_ANNOT, color=MID, ha="right", va="bottom",
                            xytext=(-4, 4), textcoords="offset points")

        # y-limits from all plotted data
        all_flat = list(main_arr.flatten())
        if use_ext and len(ext_arr) > 0:
            all_flat.extend(ext_arr.flatten().tolist())
        all_vals = np.array(all_flat)

        def _fmt_n(x, _):
            if x >= 1000:
                return f"{int(x/1000)}K"
            return str(int(x))
        ax.set_xscale("log")
        ax.set_xticks(ns)
        ax.get_xaxis().set_major_formatter(mticker.FuncFormatter(_fmt_n))
        ax.tick_params(axis="x", rotation=45, labelsize=FS_TICK - 0.5)
        ax.set_xlabel("$n$", fontsize=FS_LABEL - 1, color=MID)
        _zero(ax)

        ax.set_title(title, fontsize=FS_TITLE, color=color, fontweight="medium",
                      pad=6)

        ymin, ymax = np.percentile(all_vals, [1, 99])
        pad = (ymax - ymin) * 0.12
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_ylabel("$\\Delta$AUC", fontsize=FS_LABEL - 1, color=MID)

        ax.text(0.04, 0.93, panel_labels[idx], transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top", ha="left", color=GRAY)

    fig.tight_layout(w_pad=2.2, h_pad=2.8)
    _save(fig, "fig2_n_scaling")
    print(f"  fig2_n_scaling.pdf: spaghetti + ribbon, {n_panels} panels")
    return fig


# FIGURE 3: PEEKING RIDGELINE

def fig3_peeking_ridgeline(v1_data):
    o = ok1(v1_data)
    ks = [1, 2, 5, 10, 15, 19]

    fig, ax = plt.subplots(figsize=(5, 3.8))

    overlap = 0.72
    x_grid = np.linspace(-0.13, 0.13, 300)

    # Gradient from muted to saturated as k increases
    color_base = C["II"]

    for i, k in enumerate(ks):
        field = f"b_infl_k{k}"
        diffs = cl([r.get(field) for r in o])
        if len(diffs) < 50:
            continue

        baseline = i * (1 - overlap)
        # Progressive alpha: k=1 most transparent, k=19 most saturated
        fill_alpha = 0.12 + 0.06 * i

        try:
            kde = gaussian_kde(diffs, bw_method=0.22)
            density = kde(x_grid)
            density = density / density.max() * 0.80

            # Fill
            ax.fill_between(x_grid, baseline, baseline + density,
                            color=color_base, alpha=fill_alpha,
                            linewidth=0, zorder=2 + i)
            # Outline
            ax.plot(x_grid, baseline + density, color=color_base,
                    linewidth=LW_THIN, alpha=0.5 + 0.07 * i, zorder=10 + i)

            # Median tick
            med = np.median(diffs)
            ax.plot(med, baseline, "|", color=color_base, markersize=6,
                    markeredgewidth=LW_MED, zorder=15, alpha=0.7)

        except Exception:
            pass

    # Zero reference
    _zero(ax, axis="v")

    # Y labels
    y_ticks = [i * (1 - overlap) for i in range(len(ks))]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"$k = {k}$" for k in ks], fontsize=FS_TICK)
    ax.set_xlabel("$\\Delta$AUC (leaky $-$ clean)", fontsize=FS_LABEL, color=GRAY)
    ax.set_xlim(-0.11, 0.11)
    ax.set_ylim(-0.10, len(ks) * (1 - overlap) + 0.50)

    # Annotation: the non-monotonicity — naked text, no box
    med_k1 = np.median(cl([r.get("b_infl_k1") for r in o]))
    ax.annotate(
        "$k{=}1$: noise dominates\n$\\rightarrow$ negative inflation",
        xy=(med_k1, 0.02), xytext=(0.050, 0.32),
        fontsize=FS_ANNOT, color="#a04030", fontweight="medium",
        arrowprops=dict(arrowstyle="-", color="#a04030", lw=LW_THIN,
                        connectionstyle="arc3,rad=0.15"),
        zorder=20
    )

    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    _save(fig, "fig3_peeking_ridge")
    print(f"  fig3_peeking_ridge.pdf: ridgeline, {len(ks)} k-values")
    return fig


# FIGURE 4: SEED DOSE-RESPONSE

def fig4_seed(v3_ap_data):
    o = ok3(v3_ap_data)
    ks = [5, 10, 25, 50, 100]

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    rng = np.random.RandomState(42)

    for algo, color, label, marker in [
        ("lr", LIGHT, "Logistic Regression", "s"),
        ("rf", C["II"], "Random Forest", "o"),
    ]:
        means = []
        ci_lo = []
        ci_hi = []
        for k in ks:
            field = f"ap_{algo}_inflation_k{k}"
            vals = cl([r.get(field) for r in o])
            means.append(vals.mean() if len(vals) > 0 else np.nan)
            if len(vals) > 2:
                se = vals.std() / np.sqrt(len(vals))
                ci_lo.append(vals.mean() - 1.96 * se)
                ci_hi.append(vals.mean() + 1.96 * se)
            else:
                ci_lo.append(np.nan)
                ci_hi.append(np.nan)

            # Jitter cloud
            n_show = min(len(vals), 100)
            idx = rng.choice(len(vals), n_show, replace=False)
            jitter_x = k * np.exp(rng.normal(0, 0.04, n_show))
            ax.scatter(jitter_x, vals[idx], color=color, alpha=A_GHOST,
                       s=1.5, linewidth=0, zorder=1, rasterized=True)

        # Ribbon
        ax.fill_between(ks, ci_lo, ci_hi, color=color, alpha=A_WHISPER, zorder=3)
        # Mean curve
        ax.plot(ks, means, marker=marker, color=color, linewidth=LW_BOLD,
                markersize=MS_MD, markeredgecolor="white",
                markeredgewidth=LW_HAIR, label=label, zorder=5)

    # Log fit for RF
    rf_means = []
    for k in ks:
        vals = cl([r.get(f"ap_rf_inflation_k{k}") for r in o])
        rf_means.append(vals.mean())
    rf_means = np.array(rf_means)
    log_ks = np.log(np.array(ks, dtype=float))
    A = np.vstack([log_ks, np.ones(len(log_ks))]).T
    coeffs = np.linalg.lstsq(A, rf_means, rcond=None)[0]
    slope, intercept = coeffs
    ss_res = np.sum((rf_means - (slope * log_ks + intercept)) ** 2)
    ss_tot = np.sum((rf_means - rf_means.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    # Fitted curve (dotted, subtle)
    ks_smooth = np.linspace(4, 110, 200)
    ax.plot(ks_smooth, slope * np.log(ks_smooth) + intercept, ":",
            color=C["II"], linewidth=LW_THIN, alpha=0.35, zorder=2)

    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel("Seeds tried ($K$)", fontsize=FS_LABEL, color=GRAY)
    ax.set_ylabel("$\\Delta$AUC", fontsize=FS_LABEL, color=GRAY)
    ax.set_ylim(-0.005, 0.035)
    _zero(ax)

    # Legend: upper left — naked, no frame
    ax.legend(loc="upper left", fontsize=FS_LEGEND,
              handlelength=1.2, borderpad=0.2, labelcolor=GRAY)

    # R² annotation — naked text, no box
    ax.text(0.97, 0.38,
            f"$\\Delta$AUC = {slope:.5f}$\\cdot$ln($K$) + {intercept:.5f}\n$R^2$ = {r2:.3f}",
            transform=ax.transAxes, fontsize=FS_ANNOT, va="top", ha="right",
            color=C["II"])

    _save(fig, "fig4_seed")
    print("  fig4_seed.pdf: seed log curve + jitter")
    return fig


# FIGURE 5: CV COVERAGE GAP

def fig5_cv_coverage(v3_ao_data):
    o = ok3(v3_ao_data)

    fig, ax = plt.subplots(figsize=(5, 2.8))

    algos = [("lr", "LR"), ("rf", "RF"), ("dt", "DT")]
    methods = [("z", "$z$-based"), ("t", "$t$-based")]

    y = 0
    yticks = []
    ylabels = []

    for algo_key, algo_label in algos:
        for meth_key, meth_label in methods:
            field = f"ao_{algo_key}_coverage_{meth_key}"
            vals = cl([r.get(field) for r in o])
            if len(vals) < 10:
                continue

            n_obs = len(vals)
            p_hat = vals.mean()
            m = p_hat * 100

            # Wilson score interval
            z = 1.96
            denom = 1 + z**2 / n_obs
            center = (p_hat + z**2 / (2 * n_obs)) / denom
            margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n_obs)) / n_obs) / denom
            lo = max(0, center - margin) * 100
            hi = min(1, center + margin) * 100

            color = C["II"] if meth_key == "z" else C["I"]

            # CI bar — thin, muted
            ax.plot([lo, hi], [y, y], color=color, linewidth=LW_HEAVY - 0.5,
                    solid_capstyle="round", zorder=3, alpha=0.3)
            # Point — the data
            ax.plot(m, y, "o", color=color, markersize=MS_LG,
                    markeredgecolor="white", markeredgewidth=LW_THIN, zorder=5)
            # Value — bold, above
            ax.text(m, y + 0.25, f"{m:.1f}%", fontsize=FS_ANNOT, va="bottom",
                    ha="center", color=color, fontweight="bold")

            yticks.append(y)
            ylabels.append(f"{algo_label} ({meth_label})")
            y += 1

    # Nominal 95% — THE reference
    ax.axvline(95, color=C["II"], linewidth=LW_MED, linestyle="--",
               alpha=0.35, zorder=1)
    ax.text(95.3, y - 0.5, "Nominal 95%", fontsize=FS_ANNOT,
            color=C["II"], ha="left", va="center", alpha=0.6)

    # Subtle gap shading
    ax.axvspan(75, 95, color=C["II"], alpha=0.018, zorder=0)

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=FS_TICK)
    ax.set_xlabel("Actual coverage (%)", fontsize=FS_LABEL, color=GRAY)
    ax.set_xlim(48, 102)
    ax.set_ylim(-0.6, y + 0.1)
    ax.tick_params(axis="y", length=0)

    _save(fig, "fig5_cv_coverage")
    print("  fig5_cv_coverage.pdf: CV coverage (Wilson CI)")
    return fig


# FIGURE 6: CAPACITY AMPLIFICATION

def fig6_capacity(v1_data):
    o = ok1(v1_data)

    pcts = [5, 10, 30]
    fig, ax = plt.subplots(figsize=(3.8, 3.2))

    # Progressive darkness with dose
    alphas = [0.45, 0.65, 0.85]

    for i, pct in enumerate(pcts):
        lr_vals = cl([r.get(f"h_lr_{pct:02d}") for r in o])
        rf_vals = cl([r.get(f"h_rf_{pct:02d}") for r in o])

        lr_d = dz(lr_vals)
        rf_d = dz(rf_vals)

        # Connecting line
        ax.plot([0, 1], [lr_d, rf_d], color=C["III"], linewidth=LW_MED,
                alpha=alphas[i], zorder=3)
        # LR dot (gray = baseline)
        ax.plot(0, lr_d, "o", color=GRAY, markersize=MS_LG + 1,
                markeredgecolor="white", markeredgewidth=LW_THIN, zorder=5,
                alpha=alphas[i])
        # RF dot (class color = the finding)
        ax.plot(1, rf_d, "o", color=C["III"], markersize=MS_LG + 1,
                markeredgecolor="white", markeredgewidth=LW_THIN, zorder=5,
                alpha=alphas[i])

        # Label — naked text, right of RF
        ratio = rf_d / lr_d if lr_d > 0 else float("inf")
        ax.text(1.07, rf_d,
                f"{pct}% dup ({ratio:.1f}$\\times$)",
                fontsize=FS_ANNOT, ha="left", va="center", color=C["III"],
                fontweight="bold", alpha=alphas[i])

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Logistic\nRegression", "Random\nForest"],
                        fontsize=FS_TICK + 0.5, color=GRAY)
    ax.set_ylabel("Cohen's $d_z$", fontsize=FS_LABEL, color=GRAY)
    ax.set_xlim(-0.2, 1.50)
    _zero(ax)

    _save(fig, "fig6_capacity")
    print("  fig6_capacity.pdf: capacity amplification paired dots")
    return fig


# MAIN

def main():
    setup()
    show = "--show" in sys.argv

    print("=" * 60)
    print("Generating figures...")
    print("=" * 60)

    for p in [V1, V2, V3_AN, V3_AP, V3_AO]:
        if not p.exists():
            print(f"FATAL: {p} not found")
            sys.exit(1)
        print(f"  {p.name}")

    v1 = load(V1)
    v2 = load(V2)
    v3_an = load(V3_AN)
    v3_ap = load(V3_AP)
    v3_ao = load(V3_AO)

    print()
    fig1_raincloud(v1, v2)
    fig2_n_scaling(v3_an)
    fig3_peeking_ridgeline(v1)
    fig4_seed(v3_ap)
    fig5_cv_coverage(v3_ao)
    fig6_capacity(v1)

    print(f"\n6 figures saved to {PAPER}")

    if show:
        plt.show()


if __name__ == "__main__":
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--results-dir" and i < len(sys.argv) - 1:
            RESULTS = Path(sys.argv[i + 1])
            V1 = RESULTS / "leakage_landscape_v1_final.jsonl"
            V2 = RESULTS / "leakage_landscape_v2.jsonl"
            V3_AN = RESULTS / "v3" / "v3_an.jsonl"
            V3_AP = RESULTS / "v3" / "v3_ap.jsonl"
            V3_AO = RESULTS / "v3" / "v3_ao_merged.jsonl"
    main()
