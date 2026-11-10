"""Figures of the analysis stage (static PNGs for the report).

Scenes are shown as separate panels rather than as colours: there are four of them and a reader
compares shapes between panels more reliably than four hues in one. Colour is kept for the one
thing that needs it, the three predictor families in the held-out tests.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# palette: first three categorical slots (validated together), neutral inks and surface
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list("p037_blue", ["#e9f1fb", BLUE, "#0b2e5c"]).with_extremes(bad=SURF)
MODEL_STYLE = {  # fixed colour per predictor family, wherever it appears
    "uncertainty": (BLUE, "model uncertainty"),
    "proxies": (ORANGE, "landscape proxies"),
    "uncertainty+proxies": (AQUA, "both"),
}
NICE = {"slope_mean": "mean slope", "slope_p90": "slope p90", "elev_range": "elevation range",
        "hand_median": "HAND median", "hand_lt5_frac": "share HAND < 5 m", "veg_frac": "vegetation",
        "tree_frac": "tree cover", "built_frac": "built-up", "bldg_height": "building height",
        "ob_bldg_frac": "building footprints"}
TARGET_LABEL = {"err_rate": "error rate (wrong pixels / pixels)", "f1_loss": "1 - F1 (flooded tiles)",
                "miss_rate": "miss rate (missed flood / flood)"}

TARGET_SHORT = {"err_rate": "tile error rate", "f1_loss": "1 - F1", "miss_rate": "miss rate"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK, "figure.facecolor": SURF,
    "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.titlesize": 9.5, "legend.frameon": False,
})


def nice_event(event: str) -> str:
    return re.sub(r"^\d{8}_", "", str(event)).replace("_", " ").replace("NovaKakhovka", "Nova Kakhovka")


def nice_proxy(proxy: str) -> str:
    return NICE.get(proxy, proxy.replace("_", " "))


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def placement_figure(df: pd.DataFrame, proxies: list[str], events: list[str], path: Path, reference: str) -> Path:
    """Where the test tiles sit in the training distribution of each proxy.

    x is the percentile of a tile within the training tiles. Training data is flat by
    construction (dashed line), so a pile at either end is a scene outside the training range."""
    fig, axes = plt.subplots(len(proxies), len(events), figsize=(2.6 * len(events), 1.35 * len(proxies) + 0.8),
                             sharex=True, sharey=True, squeeze=False)
    edges = np.linspace(0, 100, 11)
    for i, p in enumerate(proxies):
        for j, e in enumerate(events):
            ax = axes[i, j]
            v = df.loc[df["event"] == e, f"pct__{p}"].dropna().to_numpy()
            share = np.histogram(v, edges)[0] / max(len(v), 1)
            ax.bar(edges[:-1] + 5, share, width=8.6, color=BLUE, linewidth=0)
            ax.axhline(0.1, color=MUTED, linewidth=1, linestyle=(0, (3, 2)))
            ax.grid(axis="x", visible=False)
            if i == 0:
                ax.set_title(nice_event(e))
            if j == 0:
                ax.set_ylabel(nice_proxy(p), rotation=0, ha="right", va="center", color=INK)
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_yticks([0, 0.5, 1], ["0", "50 %", "100 %"])
    axes[0, 0].set_xticks([0, 50, 100])
    fig.supxlabel(f"percentile of the tile in the training distribution ({reference})", fontsize=9, color=INK2)
    fig.suptitle("Share of each scene's tiles by position in the training distribution "
                 "(dashed = training data itself)", fontsize=10, color=INK, x=0.5, y=1.0)
    fig.tight_layout()
    return _save(fig, path)


def _binned(x: np.ndarray, y: np.ndarray, edges: np.ndarray, min_n: int = 10):
    mid, mean, lo, hi = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        m = (x >= a) & (x <= b)
        if m.sum() >= min_n:
            q = np.percentile(y[m], [25, 75])
            mid.append(np.median(x[m]))
            mean.append(y[m].mean())
            lo.append(q[0])
            hi.append(q[1])
    return np.array(mid), np.array(mean), np.array(lo), np.array(hi)


def error_vs_confidence(df: pd.DataFrame, events: list[str], unc_col: str, target: str, path: Path,
                        rho: dict[str, tuple[float, float]]) -> Path:
    """Per scene: tile error against MC-dropout variance. Dots = tiles, line = mean in 10 equal-count bins."""
    fig, axes = plt.subplots(1, len(events), figsize=(2.9 * len(events), 3.0), sharey=True, squeeze=False)
    for ax, e in zip(axes[0], events, strict=True):
        d = df[(df["event"] == e) & (df[unc_col] > 0)].dropna(subset=[unc_col, target])
        x, y = np.log10(d[unc_col].to_numpy()), d[target].to_numpy()
        ax.scatter(x, y, s=5, color=MUTED, alpha=0.25, linewidths=0, rasterized=True)
        if len(x) >= 50:
            mid, mean, _, _ = _binned(x, y, np.quantile(x, np.linspace(0, 1, 11)))
            ax.plot(mid, mean, color=BLUE, linewidth=2, marker="o", markersize=4.5, markeredgecolor=SURF,
                    markeredgewidth=1)
        r, rp = rho.get(e, (np.nan, np.nan))
        ax.set_title(f"{nice_event(e)}\nrho {r:+.2f} ({rp:+.2f} given flood share)", loc="left")
        ax.set_xlabel("MC-dropout variance (log10)")
    # square-root axis: most tiles have an error rate near 0, and a linear axis hides them
    axes[0, 0].set_yscale("function", functions=(lambda v: np.sqrt(np.clip(v, 0, None)), np.square))
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_yticks([0, 0.01, 0.05, 0.2, 0.5, 1], ["0", "0.01", "0.05", "0.2", "0.5", "1"])
    axes[0, 0].set_ylabel(TARGET_LABEL.get(target, target) + ", square-root axis")
    fig.suptitle("Does the model know where it is wrong? Tile error against its own uncertainty "
                 "(dots = tiles, line = mean of 10 equal-count bins, rho = rank correlation)",
                 fontsize=10, color=INK, x=0.01, ha="left", y=1.02)
    fig.tight_layout()
    return _save(fig, path)


def error_vs_proxy(df: pd.DataFrame, proxies: list[str], events: list[str], target: str, path: Path) -> Path:
    """Mean tile error by decile of the training distribution, per proxy (rows) and scene (columns)."""
    fig, axes = plt.subplots(len(proxies), len(events), figsize=(2.6 * len(events), 1.5 * len(proxies) + 0.9),
                             sharex=True, sharey=True, squeeze=False)
    edges = np.linspace(0, 100, 11)
    for i, p in enumerate(proxies):
        for j, e in enumerate(events):
            ax = axes[i, j]
            d = df[df["event"] == e].dropna(subset=[f"pct__{p}", target])
            mid, mean, _, _ = _binned(d[f"pct__{p}"].to_numpy(), d[target].to_numpy(), edges)
            ax.plot(mid, mean, color=BLUE, linewidth=2, marker="o", markersize=4, markeredgecolor=SURF,
                    markeredgewidth=0.8)
            if i == 0:
                ax.set_title(nice_event(e))
            if j == 0:
                ax.set_ylabel(nice_proxy(p), rotation=0, ha="right", va="center", color=INK)
    axes[0, 0].set_xticks([0, 50, 100])
    axes[0, 0].set_xlim(0, 100)
    axes[0, 0].set_ylim(bottom=0)
    fig.supxlabel("percentile of the tile in the training distribution", fontsize=9, color=INK2)
    fig.suptitle(f"Mean {TARGET_LABEL.get(target, target)} by position in the training distribution "
                 "(bins with < 10 tiles omitted)", fontsize=10, color=INK, y=1.0)
    fig.tight_layout()
    return _save(fig, path)


def weights_figure(tables: dict[str, pd.DataFrame], path: Path, target: str) -> Path:
    """Standardised regression weights with block-bootstrap 95 % intervals, one panel per score type."""
    n = max(len(t) for t in tables.values())
    fig, axes = plt.subplots(1, len(tables), figsize=(4.4 * len(tables), 0.36 * n + 1.5), sharex=True, squeeze=False)
    for ax, (title, t) in zip(axes[0], tables.items(), strict=True):
        t = t.iloc[::-1]
        y = np.arange(len(t))
        ax.barh(y, t["weight"], height=0.55, color=BLUE, linewidth=0)
        ax.hlines(y, t["lo"], t["hi"], color=INK, linewidth=1.2)
        ax.axvline(0, color=AXIS, linewidth=1)
        ax.set_yticks(y, [nice_proxy(p) for p in t["proxy"]], color=INK)
        ax.grid(axis="y", visible=False)
        ax.set_title(title, loc="left")
        ax.set_xlabel(f"change in {TARGET_SHORT.get(target, target)} per SD of the score")
    fig.suptitle("Importance weights of the proxies (bar = estimate, line = 95 % interval from resampled blocks)",
                 fontsize=10, color=INK, x=0.01, ha="left", y=1.02)
    fig.tight_layout()
    return _save(fig, path)


def held_out_figure(table: pd.DataFrame, path: Path, target: str) -> Path:
    """Rank correlation between predicted and observed tile error on the event left out of the fit."""
    events = list(dict.fromkeys(table["held_out"]))
    fig, ax = plt.subplots(figsize=(6.2, 0.62 * len(events) + 1.6))
    offsets = {"uncertainty": 0.2, "proxies": 0.0, "uncertainty+proxies": -0.2}
    for model, (colour, label) in MODEL_STYLE.items():
        t = table[table["model"] == model].set_index("held_out").reindex(events)
        y = np.arange(len(events))[::-1] + offsets[model]
        ax.scatter(t["spearman"], y, s=70, color=colour, edgecolors=SURF, linewidths=1.5, label=label, zorder=3)
        for xv, yv in zip(t["spearman"], y, strict=True):
            if np.isfinite(xv):
                ax.annotate(f"{xv:+.2f}", (xv, yv), xytext=(7, 0), textcoords="offset points", va="center",
                            fontsize=8, color=INK2)
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set_yticks(np.arange(len(events))[::-1], [nice_event(e) for e in events], color=INK)
    ax.set_ylim(-0.6, len(events) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"rank correlation of predicted and observed {TARGET_SHORT.get(target, target)} on the held-out event")
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, -0.09), ncol=3)
    ax.set_title("Predicting tile error on an event never seen by the predictor", loc="left", fontsize=10)
    return _save(fig, path)


def triage_figure(curves: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]], path: Path) -> Path:
    """Share of error pixels found against share of tiles reviewed, per held-out event."""
    events = list(curves)
    fig, axes = plt.subplots(1, len(events), figsize=(3.0 * len(events), 3.2), sharey=True, squeeze=False)
    for ax, e in zip(axes[0], events, strict=True):
        ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=(0, (3, 2)), label="random order")
        if "oracle" in curves[e]:
            ax.plot(*curves[e]["oracle"], color=MUTED, linewidth=1, label="perfect order")
        for model, (colour, label) in MODEL_STYLE.items():
            if model in curves[e]:
                ax.plot(*curves[e][model], color=colour, linewidth=2, label=label)
        ax.set_title(nice_event(e), loc="left")
        ax.set_xlabel("share of tiles reviewed")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0, 0].set_ylabel("share of wrong pixels found")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Where should a human look first? Tiles ranked by a predictor fitted on the other events",
                 fontsize=10, color=INK, x=0.01, ha="left", y=1.02)
    fig.tight_layout()
    return _save(fig, path)


def tile_maps(df: pd.DataFrame, event: str, columns: list[tuple[str, str]], path: Path) -> Path:
    """The scene's tile grid coloured by each column (north up), light = low, dark = high."""
    d = df[df["event"] == event]
    rows, cols = int(d["tile_row"].max()) + 1, int(d["tile_col"].max()) + 1
    fig, axes = plt.subplots(1, len(columns), figsize=(3.6 * len(columns), 3.6 * rows / max(cols, 1) + 1.0),
                             squeeze=False)
    for ax, (col, label) in zip(axes[0], columns, strict=True):
        grid = np.full((rows, cols), np.nan)
        grid[d["tile_row"].to_numpy(), d["tile_col"].to_numpy()] = d[col].to_numpy()
        finite = grid[np.isfinite(grid)]
        vmin, vmax = (np.percentile(finite, [2, 98]) if finite.size else (0, 1))
        im = ax.imshow(np.ma.masked_invalid(grid), cmap=SEQ, vmin=vmin, vmax=max(vmax, vmin + 1e-12),
                       interpolation="nearest")
        ax.set_title(label, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(visible=False)
        for s in ax.spines.values():
            s.set_visible(False)
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.04, fraction=0.05)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=7.5, colors=MUTED)
    fig.suptitle(f"{nice_event(event)}: one cell per 256 px tile, north up (colour scale 2nd to 98th percentile)",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, path)
