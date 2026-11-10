"""Stage 5: join per-tile results with proxies and test the project's three research questions.

Input:  a per-tile results table (``tiles.csv`` of the evaluate stage, or ``analysis/usf_results.csv``
        of the zero-shot run) and the proxy table (raw proxy values per training patch and test tile).
Output: tables (CSV), figures (PNG) and ``report.md`` in ``<run folder>/analysis``.

What is computed, in the order of the report:
1. Scores per scene (pixel-pooled), to check the join against the evaluation summary.
2. Proxy placement: where each scene's tiles sit in the training distribution of each proxy.
3. Error vs confidence: does MC-dropout uncertainty rank tile error, also once the amount of
   flood in the tile is accounted for?
4. Error vs proxy: rank correlation of tile error with each proxy's position.
5. The weighted OOD score: regression of tile error on per-proxy scores (weights with intervals).
6. Held-out-event test (RQ1, RQ2): predictors fitted on the other events, tested on the one left
   out: uncertainty only, proxies only, both.
7. Triage: if a human reviews tiles in the predicted order, how fast are the wrong pixels found?

Per-tile error measures:
* err_rate  = wrong pixels / scored pixels, defined for every tile;
* f1_loss   = 1 - F1, only for tiles with at least ``min_positive_px`` flood pixels in the labels
              (F1 is undefined or meaningless on tiles without flood);
* miss_rate = missed flood pixels / flood pixels, same tiles.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import ood, plots
from .config import Config
from .datasets import REGISTRY
from .metrics import scores

log = logging.getLogger("p037.analyze")
UNCERTAINTY = ["log_mc_var", "entropy", "pred_frac"]


def md_table(df: pd.DataFrame, digits: int = 3) -> str:
    def fmt(v):
        if isinstance(v, float | np.floating):
            return "" if not np.isfinite(v) else f"{v:.{digits}f}"
        return str(v)

    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns)
    return head + "\n" + "\n".join("| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False))


def prepare(cfg: Config, tiles_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """Join results and proxies, add error measures, uncertainty features and proxy placement."""
    ds = REGISTRY[cfg.dataset]
    target = ds.target
    pre = f"{target}_{cfg.score_class}"
    tiles = pd.read_csv(tiles_csv)
    need = [f"{pre}_{k}" for k in ("tp", "fp", "fn", "tn")]
    if cfg.score_class == "any" and need[0] not in tiles:
        # older results tables (analysis/usf_results.csv) have the classes only: derive `any` from them.
        # A one-vs-rest table excludes the other classes, so its fp and tn are over non-flood pixels only.
        names = list(ds.classes.values())
        tiles[need[0]] = sum(tiles[f"{target}_{c}_tp"] for c in names)
        tiles[need[2]] = sum(tiles[f"{target}_{c}_fn"] for c in names)
        tiles[need[1]], tiles[need[3]] = tiles[f"{target}_{names[0]}_fp"], tiles[f"{target}_{names[0]}_tn"]
    if any(c not in tiles for c in need):
        raise KeyError(f"{tiles_csv} has no columns {need}")
    proxy = pd.read_csv(cfg.resolve(cfg.proxy_table))
    columns = {p: f"{p}__{cfg.proxy_scale}" for p in cfg.proxies}
    ref = proxy[proxy["set"] == cfg.reference_set]
    info = {"tiles_file": str(tiles_csv), "reference_set": cfg.reference_set, "reference_tiles": len(ref)}
    listed = cfg.out / "train_patches.csv"
    if listed.exists():  # the patches this run actually trained on (matters when events were held out)
        ids = set(pd.read_csv(listed)["patch_id"])
        if ref["sample_id"].isin(ids).any():
            ref = ref[ref["sample_id"].isin(ids)]
            info["reference_tiles"] = len(ref)
            info["reference_note"] = "restricted to the patches listed in train_patches.csv"
    if len(ref) < 10:
        raise ValueError(f"only {len(ref)} rows of set {cfg.reference_set!r} in {cfg.proxy_table}")
    keep = ["sample_id", "lon", "lat", *[c for c in ("mostly_water",) if c in proxy], *columns.values()]
    tiles = tiles.drop(columns=[c for c in ("lon", "lat") if c in tiles])  # the proxy table's are the ones used
    df = tiles.merge(proxy[keep], left_on="tile_id", right_on="sample_id", how="left")
    info["tiles"] = len(df)
    info["tiles_without_proxies"] = int(df["sample_id"].isna().sum())
    df = df[df["sample_id"].notna() & (df["n_valid"] > 0) & df[need[0]].notna()].copy()
    info["tiles_analysed"] = len(df)

    tp, fp, fn, tn = (df[c].to_numpy(float) for c in need)
    n, pos = tp + fp + fn + tn, tp + fn
    flooded = pos >= cfg.min_positive_px
    df["err_px"] = fp + fn
    df["err_rate"] = (fp + fn) / np.maximum(n, 1)
    df["gt_frac"] = pos / np.maximum(n, 1)
    df["flooded"] = flooded
    with np.errstate(invalid="ignore", divide="ignore"):
        df["f1_loss"] = np.where(flooded, 1 - 2 * tp / (2 * tp + fp + fn), np.nan)
        df["miss_rate"] = np.where(flooded, fn / pos, np.nan)
    df["log_mc_var"] = np.log10(df[f"mc_var_{target}"].to_numpy(float) + 1e-12)
    df["entropy"] = df[f"mean_entropy_{target}"]
    df["pred_frac"] = df[f"frac_{target}"]
    groups = {**ds.event_groups, **cfg.event_groups}
    df["event_group"] = df["event"].map(groups).fillna(df["event"])
    df = ood.place(df, ref, columns)
    return df, ref, columns, info


def run_analysis(cfg: Config, tiles_csv: str | Path | None = None, out_dir: str | Path | None = None) -> Path:
    tiles_csv = Path(tiles_csv) if tiles_csv else cfg.out / "tiles.csv"
    out = Path(out_dir) if out_dir else cfg.out / "analysis"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    ds = REGISTRY[cfg.dataset]
    pre = f"{ds.target}_{cfg.score_class}"
    df, ref, columns, info = prepare(cfg, tiles_csv)
    if df.empty:
        raise ValueError(f"none of the {info['tiles']} tiles of {tiles_csv} can be analysed "
                         f"({info['tiles_without_proxies']} have no proxy values in {cfg.proxy_table}): "
                         f"run `p037 proxies` first")
    events = sorted(df["event"].unique())
    proxies = list(columns)
    kept, dropped = ood.prune_collinear(ref, columns, cfg.max_proxy_corr)
    log.info("%d tiles, %d events; proxies kept %s, dropped %s", len(df), len(events), kept, list(dropped))

    # 1 ------------------------------------------------------------------ scores per scene
    rows = []
    for e, g in [*df.groupby("event"), ("all", df)]:
        s = scores(*(float(g[f"{pre}_{k}"].sum()) for k in ("tp", "fp", "fn", "tn")))
        rows.append({"event": e, "tiles": len(g), "flooded_tiles": int(g["flooded"].sum()), "f1": s["f1"],
                     "precision": s["precision"], "recall": s["recall"], "mean_err_rate": g["err_rate"].mean(),
                     "mean_f1_loss": g["f1_loss"].mean()})
    t_events = pd.DataFrame(rows)

    # 2 ------------------------------------------------------------------ placement
    rows = []
    for e, g in df.groupby("event"):
        for p in proxies:
            v = g[f"pct__{p}"].dropna()
            rows.append({"event": e, "proxy": p, "kept": p in kept, "median_percentile": v.median(),
                         "share_below_p05": float((v < 5).mean()), "share_above_p95": float((v > 95).mean())})
    t_place = pd.DataFrame(rows)

    # 3 ------------------------------------------------------------------ error vs confidence
    rows, rho = [], {}
    for e, g in [*df.groupby("event"), ("all", df)]:
        r = ood.spearman(g["err_rate"], g["log_mc_var"])
        rp = ood.partial_spearman(g["err_rate"], g["log_mc_var"], g["gt_frac"])
        rho[e] = (r, rp)
        f = g[g["flooded"]]
        rows.append({"event": e, "err_rate~mc_var": r, "err_rate~mc_var|flood_share": rp,
                     "err_rate~entropy": ood.spearman(g["err_rate"], g["entropy"]),
                     "f1_loss~mc_var (flooded)": ood.spearman(f["f1_loss"], f["log_mc_var"]),
                     "miss_rate~mc_var (flooded)": ood.spearman(f["miss_rate"], f["log_mc_var"])})
    t_conf = pd.DataFrame(rows)

    # 4 ------------------------------------------------------------------ error vs proxy
    rows = []
    for e, g in [*df.groupby("event"), ("all", df)]:
        f = g[g["flooded"]]
        for p in proxies:
            rows.append({"event": e, "proxy": p, "kept": p in kept,
                         "err_rate~position": ood.spearman(g["err_rate"], g[f"pct__{p}"]),
                         "err_rate~position|flood_share": ood.partial_spearman(g["err_rate"], g[f"pct__{p}"], g["gt_frac"]),
                         "err_rate~distance": ood.spearman(g["err_rate"], g[f"dist__{p}"]),
                         "f1_loss~position (flooded)": ood.spearman(f["f1_loss"], f[f"pct__{p}"]),
                         "f1_loss~distance (flooded)": ood.spearman(f["f1_loss"], f[f"dist__{p}"])})
    t_proxy = pd.DataFrame(rows)

    # 5 ------------------------------------------------------------------ weights and OOD score
    pos_cols, dist_cols = [f"pos__{p}" for p in kept], [f"dist__{p}" for p in kept]
    weight_tables, t_fit = {}, []
    for target in ("f1_loss", "err_rate"):
        d = df.dropna(subset=[target, *pos_cols]).reset_index(drop=True)
        if len(d) < 50:
            continue
        y, blocks = d[target].to_numpy(float), ood.spatial_blocks(d)
        for kind, cols in (("distance", dist_cols), ("position", pos_cols)):
            X = d[cols].to_numpy(float)
            model = ood.Linear().fit(X, y)
            boot = ood.bootstrap_weights(X, y, blocks, cfg.n_bootstrap, cfg.seed)
            lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
            weight_tables[(target, kind)] = pd.DataFrame({"target": target, "score": kind, "proxy": kept,
                                                          "weight": model.weights, "lo": lo, "hi": hi})
            t_fit.append({"target": target, "score": kind, "tiles": len(d), "in_sample_r2": ood.r2(y, model.predict(X)),
                          "in_sample_spearman": ood.spearman(model.predict(X), y)})
            if (target, kind) == ("err_rate", "distance"):
                df["ood_score"] = model.score(df[cols].to_numpy(float))
    t_weights = pd.concat(weight_tables.values(), ignore_index=True) if weight_tables else pd.DataFrame()
    t_fit = pd.DataFrame(t_fit)

    # 6 ------------------------------------------------------------------ held-out-event test
    feature_sets = {"uncertainty": UNCERTAINTY, "proxies_distance": dist_cols, "proxies_position": pos_cols,
                    "proxies": pos_cols + dist_cols, "uncertainty+proxies": UNCERTAINTY + pos_cols + dist_cols}
    held, preds = {}, {}
    if df["event_group"].nunique() >= 2:
        for target in ("err_rate", "f1_loss"):
            held[target], preds[target] = ood.held_out_event_test(df, feature_sets, target)
    t_held = (pd.concat([t.assign(target=k) for k, t in held.items()], ignore_index=True) if held else pd.DataFrame())

    # 7 ------------------------------------------------------------------ triage
    curves, rows = {}, []
    if "err_rate" in preds:
        p = preds["err_rate"].merge(df[["tile_id", "err_px"]], on="tile_id")
        for g, sub in p.groupby("event_group"):
            err = sub["err_px"].to_numpy(float)
            curves[g] = {"oracle": ood.triage_curve(err, err)}
            row = {"held_out": g, "tiles": len(sub), "perfect_order": ood.found_at(err, err)}
            for m in plots.MODEL_STYLE:
                if f"pred__{m}" not in sub:
                    continue
                s = sub[f"pred__{m}"].to_numpy(float)
                curves[g][m] = ood.triage_curve(s, err)
                row[m] = ood.found_at(s, err)
            rows.append(row)
    t_triage = pd.DataFrame(rows)

    # ---------------------------------------------------------------------- write tables
    tables = {"events": t_events, "placement": t_place, "confidence": t_conf, "error_vs_proxy": t_proxy,
              "weights": t_weights, "weights_fit": t_fit, "held_out": t_held, "triage_top20": t_triage}
    for name, t in tables.items():
        t.to_csv(out / f"{name}.csv", index=False)
    scored_cols = ["tile_id", "event", "event_group", "tile_row", "tile_col", "lon", "lat", "err_rate", "f1_loss",
                   "miss_rate", "gt_frac", "log_mc_var", "entropy", "pred_frac",
                   *[f"{k}__{p}" for p in proxies for k in ("pct", "pos", "dist")]]
    if "ood_score" in df:
        scored_cols.append("ood_score")
    scored = df[scored_cols]
    for target, p in preds.items():
        scored = scored.merge(p[["tile_id", *[c for c in p if c.startswith("pred__")]]].rename(
            columns=lambda c, t=target: c.replace("pred__", f"heldout_pred_{t}__")), on="tile_id", how="left")
    scored.to_csv(out / "tiles_scored.csv.gz", index=False)

    # ---------------------------------------------------------------------- figures
    fig = out / "figures"
    plots.placement_figure(df, proxies, events, fig / "placement.png", cfg.reference_set)
    plots.error_vs_confidence(df, events, f"mc_var_{ds.target}", "err_rate", fig / "error_vs_confidence.png", rho)
    plots.error_vs_proxy(df, kept, events, "err_rate", fig / "error_vs_proxy.png")
    for target in ("f1_loss", "err_rate"):
        sel = {f"{kind} from the training median" if kind == "distance" else "position (signed)": weight_tables[(target, kind)]
               for kind in ("distance", "position") if (target, kind) in weight_tables}
        if sel:
            plots.weights_figure(sel, fig / f"weights_{target}.png", target)
    for target, t in held.items():
        if len(t):
            plots.held_out_figure(t[t["model"].isin(plots.MODEL_STYLE)], fig / f"held_out_{target}.png", target)
    if curves:
        plots.triage_figure(curves, fig / "triage.png")
    map_cols = [("err_rate", "error rate"), ("log_mc_var", "MC-dropout variance (log10)")]
    if "ood_score" in df:
        map_cols.append(("ood_score", "OOD score (weighted proxy distance)"))
    for e in events:
        if (df.loc[df["event"] == e, "tile_row"] >= 0).all():
            plots.tile_maps(df, e, map_cols, fig / f"tile_maps_{e}.png")

    report = write_report(cfg, out, info, tables, kept, dropped, pre)
    log.info("analysis written to %s", out)
    return report


def compare_runs(tables: dict[str, str | Path], target: str = "flood", score_class: str = "open",
                 out_csv: str | Path | None = None) -> pd.DataFrame:
    """Pixel-pooled scores of several runs on the tiles they have in common, per event.

    ``tables`` maps a run name to its per-tile results CSV (e.g. the zero-shot results and
    Experiment B). Only tiles present in every table are used, so the runs are compared on
    exactly the same pixels. One row per (event, run)."""
    pre = f"{target}_{score_class}"
    cols = [f"{pre}_{k}" for k in ("tp", "fp", "fn", "tn")]
    loaded = {name: pd.read_csv(path).set_index("tile_id") for name, path in tables.items()}
    common = sorted(set.intersection(*(set(t.index) for t in loaded.values())))
    if not common:
        raise ValueError("the result tables have no tile_id in common")
    rows = []
    for name, t in loaded.items():
        t = t.loc[common]
        for event, g in [*t.groupby("event"), ("all", t)]:
            c = [float(g[k].sum()) for k in cols]
            s = scores(*c)
            rows.append({"event": event, "run": name, "tiles": len(g), "f1": s["f1"], "iou": s["iou"],
                         "precision": s["precision"], "recall": s["recall"], "positive_px": int(c[0] + c[2])})
    df = pd.DataFrame(rows).sort_values(["event", "run"], kind="stable").reset_index(drop=True)
    if out_csv:
        df.to_csv(out_csv, index=False)
    return df


def write_report(cfg: Config, out: Path, info: dict, tables: dict, kept: list[str], dropped: dict, pre: str) -> Path:
    t = tables
    lines = [
        f"# Analysis report: {cfg.name}", "",
        "Generated by `p037 analyze`. Numbers only; interpretation belongs in the project notes.", "",
        f"- Results table: `{Path(info['tiles_file']).name}`, score columns `{pre}_*` "
        f"({info['tiles_analysed']} of {info['tiles']} tiles analysed; {info['tiles_without_proxies']} without proxies).",
        f"- Training distribution: set `{cfg.reference_set}` of `{cfg.proxy_table}`, {info['reference_tiles']} tiles"
        + (f" ({info['reference_note']})" if "reference_note" in info else "") + f", {cfg.proxy_scale} scale.",
        f"- Proxies kept for the regressions: {', '.join(kept)}.",
        *([("- Dropped as collinear in the training distribution (|rho| > " + f"{cfg.max_proxy_corr}): "
            + "; ".join(f"{p} (rho {r:+.2f} with {k})" for p, (k, r) in dropped.items()) + ".")] if dropped else []),
        f"- A tile counts as flooded with at least {cfg.min_positive_px} labelled pixels of class `{cfg.score_class}`.",
        "", "## 1. Scores per scene", "",
        "Pixel-pooled over the analysed tiles. Compare with the evaluation summary to confirm the join.", "",
        md_table(t["events"]), "",
        "## 2. Where the test tiles sit in the training distribution", "",
        "![placement](figures/placement.png)", "",
        "Median percentile of each scene's tiles, and the share of tiles outside the central 90 % of the training data.", "",
        md_table(t["placement"], 2), "",
        "## 3. Error vs confidence", "",
        "Spearman rank correlations per scene. `|flood_share` = partial correlation given the labelled flood share "
        "of the tile, which removes the trivial effect that tiles with more flood have more pixels to get wrong.", "",
        "![error vs confidence](figures/error_vs_confidence.png)", "", md_table(t["confidence"]), "",
        "## 4. Error vs proxy", "",
        "Spearman rank correlation of tile error with each proxy's position (percentile in the training "
        "distribution) and distance (from the training median).", "",
        "![error vs proxy](figures/error_vs_proxy.png)", "", md_table(t["error_vs_proxy"]), "",
        "## 5. Importance weights and the OOD score", "",
        "Least squares of tile error on the per-proxy scores, all scenes pooled. Weights are per training standard "
        "deviation of the score. Intervals: 2.5th to 97.5th percentile over "
        f"{cfg.n_bootstrap} refits on resampled 8 x 8 tile blocks. An interval that includes 0 means the sign is not established.", "",
    ]
    for target in ("f1_loss", "err_rate"):
        if (out / "figures" / f"weights_{target}.png").exists():
            lines += [f"![weights {target}](figures/weights_{target}.png)", ""]
    if len(t["weights"]):
        lines += [md_table(t["weights"]), "", "In-sample fit (optimistic: fitted and scored on the same tiles):", "",
                  md_table(t["weights_fit"]), ""]
    lines += ["## 6. Held-out-event test", ""]
    if len(t["held_out"]):
        lines += [
            "Each predictor is fitted on the other events and scored on the event left out. `spearman` is the rank "
            "correlation between predicted and observed tile error there; `r2` can be strongly negative when the "
            "error level differs between events, which a ranking does not need.",
            f"With {t['held_out']['held_out'].nunique()} events this is {t['held_out']['held_out'].nunique()} "
            "data points per predictor: an indication, not a significance test.", ""]
        for target in t["held_out"]["target"].unique():
            if (out / "figures" / f"held_out_{target}.png").exists():
                lines += [f"![held out {target}](figures/held_out_{target}.png)", ""]
        lines += [md_table(t["held_out"]), ""]
    else:
        lines += ["Not run: needs at least two events.", ""]
    if len(t["triage_top20"]):
        lines += ["## 7. Triage", "",
                  "Share of all wrong pixels of the held-out event that lie in the 20 % of tiles ranked highest by "
                  "each predictor (random order finds 0.20).", "",
                  "![triage](figures/triage.png)", "", md_table(t["triage_top20"]), ""]
    maps = sorted((out / "figures").glob("tile_maps_*.png"))
    if maps:
        lines += ["## Tile maps", "",
                  "The OOD score shown is fitted on all scenes including the one mapped, so it is in-sample.", ""]
        lines += [f"![{m.stem}](figures/{m.name})\n" for m in maps]
    path = out / "report.md"
    path.write_text("\n".join(lines) + "\n")
    return path
