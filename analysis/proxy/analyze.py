"""Place the UrbanSARFloods test tiles in the two proxy training distributions.

Inputs : outputs/samples.csv and one or more proxy CSVs from compute_proxies.py.
Outputs (in outputs/):
  proxy_values.csv.gz        every sample x every proxy x both scales
  reference_summary.csv      n, mean, sd and quantiles of each reference distribution
  test_tile_placement.csv.gz z-score and percentile of every test tile, per proxy, scale, reference
  test_event_placement.csv   per test scene: median tile value, its z-score / percentile, and the
                             share of tiles outside the reference's 5-95 % range
  figures/dist_<proxy>.png   one distribution plot per proxy (tile and neighbourhood scale)
  figures/proxy_correlation.png, proxy_correlation.csv  Spearman correlation between proxies

Usage: python analyze.py PROXY_CSV [PROXY_CSV ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import PROXIES as BASE_PROXIES  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

PROXIES = BASE_PROXIES + ["ob_bldg_frac"]

HERE = Path(__file__).parent
OUT = HERE / "outputs"
FIG = OUT / "figures"
SCALES = ("tile", "neigh")
SCALE_LABEL = {"tile": "Tile scale (one 256 px test chip, ~4.7 km)",
               "neigh": "Neighbourhood scale (5 × 5 chips, ~24 km)"}
REFS = {"a_amazon": "(a) Amazon water-adjacent (Wagner et al. 2024 proxy)",
        "b_usf_train": "(b) UrbanSARFloods training chips"}
REF_SHORT = {"a_amazon": "a", "b_usf_train": "b"}
TEST_EVENTS = {
    "20231201_Jubba_1": "Jubba, Somalia — scene 1 (2023-12-01)",
    "20231201_Jubba_2": "Jubba, Somalia — scene 2 (2023-12-01)",
    "20210727_Weihui": "Weihui, China (2021-07-27)",
    "20230609_NovaKakhovka": "Nova Kakhovka, Ukraine (2023-06-09)",
}
LABEL = {
    "slope_mean": "Mean slope (°)",
    "slope_p90": "90th-percentile slope (°)",
    "elev_range": "Elevation range, p99 − p1 (m)",
    "hand_median": "Median HAND (m)",
    "hand_lt5_frac": "Share of pixels with HAND < 5 m",
    "veg_frac": "Vegetated fraction (WorldCover 2021)",
    "tree_frac": "Tree-cover fraction (WorldCover 2021)",
    "built_frac": "Built-up surface fraction (GHSL 2020)",
    "bldg_height": "Building height, built-weighted ANBH (m)",
    "ob_bldg_frac": "Building footprint fraction (Open Buildings v3, partial coverage)",
}
SHORT = {"slope_mean": "slope mean", "slope_p90": "slope p90", "elev_range": "elev. range",
         "hand_median": "HAND median", "hand_lt5_frac": "HAND<5 m", "veg_frac": "vegetation",
         "tree_frac": "tree cover", "built_frac": "built-up", "bldg_height": "bldg height",
         "ob_bldg_frac": "OB footprints"}
SQRT_X = {"elev_range", "hand_median", "built_frac", "slope_p90", "slope_mean", "ob_bldg_frac"}

# reference palette (dataviz skill, light mode)
C_A, C_B = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlecolor": INK, "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False,
})


def load(proxy_csvs) -> pd.DataFrame:
    s = pd.read_csv(OUT / "samples.csv.gz")
    tabs = [pd.read_csv(f) for f in proxy_csvs]
    main_ = [t for t in tabs if "slope_mean__tile" in t.columns]
    extra = [t for t in tabs if "slope_mean__tile" not in t.columns]
    p = pd.concat(main_).drop_duplicates("sample_id", keep="last")
    df = s.merge(p, on="sample_id", how="inner")
    for t in extra:  # coverage-limited proxies: missing = not covered (NaN)
        df = df.merge(t.drop_duplicates("sample_id"), on="sample_id", how="left")
    for c in (f"{p}__{sc}" for p in PROXIES for sc in SCALES):
        if c not in df:
            df[c] = np.nan
    return df


def percentile_of(ref: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Mid-rank empirical percentile (0-100) of each x within ref."""
    r = np.sort(ref)
    lo = np.searchsorted(r, x, side="left")
    hi = np.searchsorted(r, x, side="right")
    out = 100.0 * (lo + hi) / 2.0 / r.size
    out[~np.isfinite(x)] = np.nan
    return out


MAX_WATER = 0.5  # tile windows with more open water than this are "mostly water"


def placement(df: pd.DataFrame, windows: str):
    """windows='land': references and event statistics use land-dominated windows only;
    windows='all': every window. Tile-level z / percentiles are given for every test tile."""
    D = df if windows == "all" else df[~df.mostly_water]
    test_all = df[df.set == "test"]
    test = D[D.set == "test"]
    tiles = test_all[["sample_id", "event", "lon", "lat", "mostly_water"]].copy()
    summ, ev_rows = [], []
    for ref in REFS:
        R = D[D.set == ref]
        for sc in SCALES:
            for p in PROXIES:
                col = f"{p}__{sc}"
                r = R[col].dropna().to_numpy()
                if r.size < 10:  # proxy not available for this reference
                    continue
                mu, sd = r.mean(), r.std(ddof=1)
                q = np.percentile(r, [5, 25, 50, 75, 95])
                summ.append(dict(windows=windows, reference=ref, scale=sc, proxy=p, n=r.size,
                                 n_undefined=int(R[col].isna().sum()), mean=mu, sd=sd,
                                 p05=q[0], p25=q[1], p50=q[2], p75=q[3], p95=q[4]))
                xa = test_all[col].to_numpy()
                k = f"{p}__{sc}__{REF_SHORT[ref]}"
                tiles[f"z__{k}"] = (xa - mu) / sd
                tiles[f"pct__{k}"] = percentile_of(r, xa)
                x = test[col].to_numpy()
                zt, pt = (x - mu) / sd, percentile_of(r, x)
                for ev in TEST_EVENTS:
                    m = (test.event == ev).to_numpy()
                    xe = x[m & np.isfinite(x)]
                    pe = pt[m][np.isfinite(pt[m])]
                    med = np.median(xe) if xe.size else np.nan
                    ev_rows.append(dict(
                        windows=windows, event=ev, reference=ref, scale=sc, proxy=p,
                        n_tiles=int(m.sum()),
                        n_defined=int(xe.size), event_median=med,
                        z_of_median=(med - mu) / sd,
                        pct_of_median=float(percentile_of(r, np.array([med]))[0]),
                        median_tile_z=float(np.nanmedian(zt[m])) if xe.size else np.nan,
                        share_below_p05=float((pe < 5).mean()) if len(pe) else np.nan,
                        share_above_p95=float((pe > 95).mean()) if len(pe) else np.nan))
    return tiles, pd.DataFrame(summ), pd.DataFrame(ev_rows)


NICE = [0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500,
        1000, 2000]


def _xscale(ax, p, lo, hi):
    if p in SQRT_X:
        ax.set_xscale("function", functions=(lambda v: np.sqrt(np.clip(v, 0, None)),
                                             lambda v: np.square(v)))
        cand = [t for t in NICE if lo <= t <= hi]
        # keep ticks at least ~1/9 of the axis apart in sqrt space
        span, ticks = np.sqrt(hi) - np.sqrt(max(lo, 0)), []
        for t in cand:
            if not ticks or np.sqrt(t) - np.sqrt(ticks[-1]) >= span / 9:
                ticks.append(t)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:g}" for t in ticks])


def dist_plot(df: pd.DataFrame, p: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 5.6), sharex="col",
                             gridspec_kw=dict(height_ratios=[2.2, 1.6], hspace=0.08, wspace=0.12))
    test = df[df.set == "test"]
    for j, sc in enumerate(SCALES):
        col = f"{p}__{sc}"
        vals = {ref: df.loc[df.set == ref, col].dropna().to_numpy() for ref in REFS}
        allv = np.concatenate(list(vals.values()) + [test[col].dropna().to_numpy()])
        if allv.size == 0:
            ax = axes[0, j]
            ax.text(0.5, 0.5, "not available", transform=ax.transAxes, ha="center", color=MUTED)
            continue
        lo, hi = np.nanpercentile(allv, [0, 99.5])
        if p in SQRT_X:
            bins = np.square(np.linspace(np.sqrt(max(lo, 0)), np.sqrt(hi), 45))
        else:
            bins = np.linspace(lo, hi, 45)
        ax = axes[0, j]
        for ref, c in (("a_amazon", C_A), ("b_usf_train", C_B)):
            v = np.clip(vals[ref], bins[0], bins[-1])
            w = np.full(v.size, 1.0 / max(v.size, 1))
            ax.hist(v, bins=bins, weights=w, histtype="stepfilled", color=c, alpha=0.18, lw=0)
            ax.hist(v, bins=bins, weights=w, histtype="step", color=c, lw=2,
                    label=f"{REFS[ref]}  (n = {v.size:,})")
        ax.set_title(SCALE_LABEL[sc], loc="left", fontsize=9.5)
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        if j == 0:
            ax.set_ylabel("share of samples per bin")
        # test events: one row each, p10-p90 line, IQR bar, median tick
        ax2 = axes[1, j]
        for i, ev in enumerate(TEST_EVENTS):
            v = test.loc[test.event == ev, col].dropna().to_numpy()
            y = len(TEST_EVENTS) - 1 - i
            if v.size == 0:
                ax2.text(bins[0], y, "  undefined for all tiles", va="center", color=MUTED,
                         fontsize=8)
                continue
            q10, q25, q50, q75, q90 = np.percentile(v, [10, 25, 50, 75, 90])
            ax2.plot([q10, q90], [y, y], color=INK2, lw=1.2, solid_capstyle="round")
            ax2.plot([q25, q75], [y, y], color=INK, lw=6, alpha=0.35, solid_capstyle="butt")
            ax2.plot([q50], [y], marker="o", ms=8, color=INK, mec=SURF, mew=2)
            ax.axvline(q50, color=INK, lw=0.8, alpha=0.25)
        ax2.set_yticks(range(len(TEST_EVENTS)))
        ax2.set_yticklabels([] if j else list(TEST_EVENTS.values())[::-1], fontsize=8.5,
                            color=INK)
        ax2.set_ylim(-0.6, len(TEST_EVENTS) - 0.4)
        ax2.tick_params(axis="y", length=0)
        ax2.spines["left"].set_visible(False)
        ax2.xaxis.grid(True, color=GRID, lw=0.6)
        ax2.set_axisbelow(True)
        ax2.set_xlabel(LABEL[p] + ("  (square-root axis)" if p in SQRT_X else ""))
        _xscale(ax2, p, bins[0], bins[-1])
        ax2.set_xlim(bins[0], bins[-1])
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper left", bbox_to_anchor=(0.005, 0.955), ncol=2, frameon=False,
               fontsize=8.5)
    fig.suptitle(f"{LABEL[p]} — training-set proxies vs UrbanSARFloods test tiles",
                 x=0.01, y=0.995, ha="left", fontsize=11, color=INK)
    fig.text(0.01, 0.005, "Land-dominated tile windows only (open water ≤ 50 %). "
             "Test rows: dot = median tile, bar = interquartile range,\n"
             "line = 10th–90th percentile over the test tiles of the scene. "
             "Thin vertical lines in the upper panels mark the test-scene medians.",
             fontsize=7.5, color=INK2)
    fig.subplots_adjust(left=0.22, right=0.99, top=0.84, bottom=0.15)
    fig.savefig(FIG / f"dist_{p}.png", dpi=150)
    plt.close(fig)


def corr_plot(df: pd.DataFrame) -> pd.DataFrame:
    cols = [f"{p}__tile" for p in PROXIES]
    n = int((df.set == "a_amazon").sum())
    parts = []
    for s in ("a_amazon", "b_usf_train", "test"):
        d = df[df.set == s]
        parts.append(d.sample(min(n, len(d)), random_state=0) if s != "a_amazon" else d)
    pooled = pd.concat(parts)
    panels = {f"Pooled (a, b and test, {n} tiles each)": pooled,
              "(a) Amazon": df[df.set == "a_amazon"],
              "(b) UrbanSARFloods training": df[df.set == "b_usf_train"]}
    cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    rows = []
    for ax, (name, d) in zip(axes, panels.items(), strict=True):
        c = d[cols].corr(method="spearman", min_periods=30)
        for a in PROXIES:
            for b in PROXIES:
                rows.append(dict(panel=name, proxy_1=a, proxy_2=b,
                                 spearman=c.loc[f"{a}__tile", f"{b}__tile"]))
        m = c.to_numpy()
        ax.imshow(np.ma.masked_invalid(m), cmap=cmap, vmin=-1, vmax=1)
        for i in range(len(PROXIES)):
            for j in range(len(PROXIES)):
                v = m[i, j]
                txt = "–" if not np.isfinite(v) else f"{v:.2f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                        color="#ffffff" if np.isfinite(v) and abs(v) > 0.6 else INK)
        ax.set_xticks(range(len(PROXIES)))
        ax.set_xticklabels([SHORT[p] for p in PROXIES], rotation=45, ha="right", color=INK)
        ax.set_yticks(range(len(PROXIES)))
        ax.set_yticklabels([SHORT[p] for p in PROXIES] if ax is axes[0] else [], color=INK)
        ax.set_title(name, loc="left", fontsize=9.5)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-1, 1))
    cb = fig.colorbar(sm, ax=axes, fraction=0.015, pad=0.01)
    cb.set_label("Spearman ρ", color=INK2)
    cb.outline.set_visible(False)
    fig.suptitle("Correlation between proxies (tile scale, Spearman ρ; pairwise-complete samples)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.savefig(FIG / "proxy_correlation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


def main(proxy_csvs) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    df = load(proxy_csvs)
    df["mostly_water"] = df["water_frac__tile"].fillna(0) > MAX_WATER
    print(df.groupby(["set", "event"]).size().to_string())
    df.round(5).to_csv(OUT / "proxy_values.csv.gz", index=False)
    tiles, summ, ev = placement(df, "land")
    _, summ_all, ev_all = placement(df, "all")
    tiles.round(4).to_csv(OUT / "test_tile_placement.csv.gz", index=False)
    pd.concat([summ, summ_all]).round(5).to_csv(OUT / "reference_summary.csv", index=False)
    pd.concat([ev, ev_all]).round(4).to_csv(OUT / "test_event_placement.csv", index=False)
    land = df[~df.mostly_water]
    print("mostly-water windows dropped:", df[df.mostly_water].groupby("set").size().to_dict())
    for p in PROXIES:
        dist_plot(land, p)
    corr_plot(land).round(3).to_csv(OUT / "proxy_correlation.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
