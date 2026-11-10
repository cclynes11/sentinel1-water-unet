"""Stage 3: predict, then score per tile. One row per 256 px tile: errors and the model's confidence.

Two sources, same output columns:
* whole test scenes, predicted in large windows with a context margin that is cropped away
  (``evaluate_scenes``). A window's core is cut into scoring tiles on the scene's 256 px grid,
  so ``tile_id`` = ``<scene>_ID_<row>_<col>`` matches the dataset's own test chips and the
  proxy table;
* cached patches (``evaluate_patches``), each patch being one tile, without context margin.

Two ways to get a flood map from the model (``flood_rule`` in the config):
* direct: the model was trained to output flood from its input channels (Experiment B);
* post_and_not_pre: the model outputs water; flood = water(post) AND NOT water(pre). This is the
  zero-shot rule of ``analysis/usf_eval.py``, kept so that the pretrained model runs through the
  same code.

Confidence: one deterministic pass (dropout off) gives the prediction that is scored. Then
``mc_passes`` passes with dropout on (batch norm in eval mode) give, per pixel, a mean probability
and a variance. Per tile these are averaged over the valid pixels:
``mc_var_<target>`` (mean variance) and ``mean_entropy_<target>`` (mean entropy of the MC mean, bits).

The columns match ``analysis/usf_results.csv`` wherever both exist, so the analysis stage reads
either file.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from .config import Config, save_config
from .datasets import get_dataset
from .datasets.base import INVALID, channel_indices
from .metrics import class_sets, confusion, entropy_bits, scores
from .model import load_for_eval, pick_device, set_mc_dropout

log = logging.getLogger("p037.evaluate")
PAD = 250  # 8-bit value of "no data" (0 dB), as written by db_to_uint8 for NaN


class Predictor:
    """Flood probability, prediction and MC-dropout statistics for a batch of 8-bit images."""

    def __init__(self, cfg: Config, model: torch.nn.Module, channels: list[str], device: str):
        self.model, self.device, self.cfg = model, device, cfg
        if cfg.flood_rule == "direct":
            self.index = [channel_indices(cfg.input_channels, channels)]
        else:
            self.index = [channel_indices(cfg.pre_channels, channels), channel_indices(cfg.post_channels, channels)]

    def _flood(self, x: torch.Tensor, crop: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        """(P(flood), P(water before) or None) for one forward pass, float32 on the device."""
        if len(self.index) == 1:
            return self.model(x[:, self.index[0]], crop=crop)[:, 0], None
        pre = self.model(x[:, self.index[0]], crop=crop)[:, 0]
        post = self.model(x[:, self.index[1]], crop=crop)[:, 0]
        return post * (1 - pre), torch.stack([pre, post])

    @torch.no_grad()
    def __call__(self, img8: np.ndarray, crop: int = 0) -> dict[str, np.ndarray]:
        """img8: (N, C, H, W) uint8 in the dataset's stored channel order. Returns (N, h, w) arrays."""
        t = self.cfg.threshold
        x = torch.from_numpy(np.ascontiguousarray(img8)).to(self.device).float().div_(255.0)
        set_mc_dropout(self.model, False)
        prob, parts = self._flood(x, crop)
        pred = prob >= t if parts is None else (parts[1] >= t) & ~(parts[0] >= t)
        out = {"prob": prob.cpu().numpy(), "pred": pred.cpu().numpy()}
        T = self.cfg.mc_passes
        if T >= 2:
            s1 = torch.zeros(prob.shape, dtype=torch.float64, device=self.device)
            s2 = torch.zeros_like(s1)
            set_mc_dropout(self.model, True)
            for _ in range(T):
                p = self._flood(x, crop)[0].double()
                s1 += p
                s2 += p * p
            set_mc_dropout(self.model, False)
            mean = s1 / T
            var = (s2 / T - mean * mean).clamp_(min=0) * T / (T - 1)  # unbiased
            out["mc_mean"], out["mc_var"] = mean.cpu().numpy(), var.cpu().numpy()
        return out


def tile_row(meta: dict, out: dict, i: int, gt: np.ndarray, valid: np.ndarray, classes: dict[int, str],
             target: str, mc_passes: int) -> dict:
    """Scores and confidence of one tile. ``gt`` and ``valid`` are this tile's (h, w) arrays."""
    row = dict(meta)
    row["n_valid"] = int(valid.sum())
    for value, name in classes.items():
        row[f"n_gt_{name}"] = int(np.sum(valid & (gt == value)))
    row["n_gt_nodata"] = int(np.sum(valid & (gt == INVALID)))
    if row["n_valid"] == 0:
        return row
    pred = out["pred"][i]
    for name, (positive, exclude) in class_sets(classes).items():
        tp, fp, fn, tn = confusion(pred, gt, valid, positive, exclude)
        row.update({f"{target}_{name}_tp": tp, f"{target}_{name}_fp": fp, f"{target}_{name}_fn": fn,
                    f"{target}_{name}_tn": tn})
        s = scores(tp, fp, fn, tn)
        row.update({f"{target}_{name}_{k}": s[k] for k in ("f1", "iou", "precision", "recall")})
    row[f"frac_{target}"] = float(pred[valid].mean())
    row[f"mean_prob_{target}"] = float(out["prob"][i][valid].mean())
    if "mc_mean" in out:
        mu = out["mc_mean"][i][valid]
        row[f"mc_mean_prob_{target}"] = float(mu.mean())
        row[f"mc_var_{target}"] = float(out["mc_var"][i][valid].mean())
        row[f"mean_entropy_{target}"] = float(entropy_bits(mu).mean())
    row["mc_passes"] = mc_passes
    return row


def tile_centres(transform, crs, tile_rows, tile_cols, tile: int) -> tuple[np.ndarray, np.ndarray]:
    """Longitude and latitude (WGS 84) of tile centres, from the scene's georeferencing.

    The proxies are looked up at these coordinates, so every scored tile carries them."""
    px = (np.asarray(tile_cols, float) + 0.5) * tile
    py = (np.asarray(tile_rows, float) + 0.5) * tile
    x = transform.a * px + transform.b * py + transform.c
    y = transform.d * px + transform.e * py + transform.f
    if crs is not None and not crs.is_geographic:
        from rasterio.warp import transform as warp_points

        x, y = (np.asarray(v) for v in warp_points(crs, "EPSG:4326", x.tolist(), y.tolist()))
    return x, y


def _split_tiles(a: np.ndarray, tile: int) -> np.ndarray:
    """(H, W) -> (rows * cols, tile, tile), row-major."""
    h, w = a.shape
    return a.reshape(h // tile, tile, w // tile, tile).swapaxes(1, 2).reshape(-1, tile, tile)


def evaluate_scene(cfg: Config, predictor: Predictor, ds, event: str, meta) -> list[dict]:
    import rasterio
    from rasterio.windows import Window

    files = ds.scene_files(event)
    win, crop, tile = cfg.window, cfg.crop, cfg.tile
    k = win // tile
    rows = []
    t0 = time.time()
    with rasterio.open(files["image"]) as im, rasterio.open(files["gt"]) as gtf, rasterio.open(files["valid"]) as vf:
        stored = [d or "" for d in im.descriptions]
        if all(stored) and stored != meta.channels:
            raise ValueError(f"{event}: scene bands {stored} differ from the cache channels {meta.channels}")
        H, W = im.height, im.width
        prob_out = None
        if cfg.save_probability:
            prof = dict(driver="GTiff", width=W, height=H, count=1, dtype="uint8", crs=im.crs, transform=im.transform,
                        compress="deflate", tiled=True, blockxsize=512, blockysize=512, nodata=255)
            prob_out = rasterio.open(cfg.out / f"prob_{event}.tif", "w", **prof)
            prob_out.set_band_description(1, f"P({meta.target}) * 250, deterministic pass; 255 = no data")
        origins = [(r, c) for r in range(0, H, win) for c in range(0, W, win)]
        for n, (r0, c0) in enumerate(origins, 1):
            core = Window(c0, r0, win, win)
            valid = vf.read(1, window=core, boundless=True, fill_value=0).astype(bool)
            if not valid.any():
                continue
            img = im.read(window=Window(c0 - crop, r0 - crop, win + 2 * crop, win + 2 * crop), boundless=True,
                          fill_value=PAD)
            gt = gtf.read(1, window=core, boundless=True, fill_value=INVALID)
            out = predictor(img[None], crop)
            tiles = {key: _split_tiles(v[0], tile) for key, v in out.items()}
            gts, valids = _split_tiles(gt, tile), _split_tiles(valid, tile)
            for j in range(k * k):
                if not valids[j].any():
                    continue
                tr, tc = r0 // tile + j // k, c0 // tile + j % k
                m = {"event": event, "tile_id": f"{event}_ID_{tr}_{tc}", "tile_row": tr, "tile_col": tc}
                rows.append(tile_row(m, tiles, j, gts[j], valids[j], meta.classes, meta.target, cfg.mc_passes))
            if prob_out is not None:
                h, w = min(win, H - r0), min(win, W - c0)
                p8 = np.where(valid, np.round(out["prob"][0] * 250), 255).astype(np.uint8)
                prob_out.write(p8[:h, :w], 1, window=Window(c0, r0, w, h))
            if n % 20 == 0 or n == len(origins):
                log.info("%s: window %d / %d, %.1f min", event, n, len(origins), (time.time() - t0) / 60)
        if prob_out is not None:
            prob_out.close()
        if rows and im.crs is not None:
            lon, lat = tile_centres(im.transform, im.crs, [r["tile_row"] for r in rows],
                                    [r["tile_col"] for r in rows], tile)
            for r, x, y in zip(rows, lon, lat, strict=True):
                r["lon"], r["lat"] = round(float(x), 6), round(float(y), 6)
    return rows


def summarise(df, classes: dict[int, str], target: str):
    """Pixel-pooled scores per event and overall, one row per (event, class)."""
    import pandas as pd

    rows = []
    groups = [(e, g) for e, g in df.groupby("event")] + [("all", df)]
    for event, g in groups:
        for name in [*classes.values(), "any"]:
            c = {k: float(g[f"{target}_{name}_{k}"].sum()) for k in ("tp", "fp", "fn", "tn")}
            rows.append({"event": event, "class": name, "tiles": len(g), **scores(**c),
                         "positive_px": int(c["tp"] + c["fn"])})
    return pd.DataFrame(rows)


def _finish(cfg: Config, df, meta, origin: str, name: str) -> Path:
    path = cfg.out / f"{name}.csv"
    df.to_csv(path, index=False)
    summary = summarise(df, meta.classes, meta.target)
    summary.to_csv(cfg.out / f"{name}_summary.csv", index=False)
    info = {"experiment": cfg.name, "weights": origin, "flood_rule": cfg.flood_rule, "threshold": cfg.threshold,
            "mc_passes": cfg.mc_passes, "window": cfg.window, "crop": cfg.crop, "tile": cfg.tile,
            "input_channels": cfg.input_channels, "rows": len(df)}
    (cfg.out / f"{name}_info.json").write_text(json.dumps(info, indent=1))
    log.info("wrote %s (%d tiles)\n%s", path, len(df), summary.round(3).to_string(index=False))
    return path


def evaluate_scenes(cfg: Config, device: str | None = None, overwrite: bool = False) -> Path:
    """Predict and score every test scene; writes ``tiles.csv`` in the run folder."""
    import pandas as pd

    device = pick_device(device)
    ds = get_dataset(cfg)
    meta = ds.meta()
    events = cfg.eval_events or ds.test_events
    missing = [e for e in events if not ds.has_scene(e)]
    if missing:
        raise FileNotFoundError(f"scenes not prepared: {missing}. Run `p037 data --what scenes` first.")
    model, origin = load_for_eval(cfg, device)
    log.info("%s: evaluating %s on %s with %s", cfg.name, events, device, origin)
    save_config(cfg, "config_evaluate.yaml")
    predictor = Predictor(cfg, model, meta.channels, device)
    torch.manual_seed(cfg.seed)
    parts = []
    for event in events:
        part = cfg.out / f"tiles_{event}.csv"
        if part.exists() and not overwrite:
            log.info("%s exists, reusing it", part)
        else:
            rows = evaluate_scene(cfg, predictor, ds, event, meta)
            pd.DataFrame(rows).to_csv(part, index=False)
        parts.append(pd.read_csv(part))
    df = pd.concat(parts, ignore_index=True).sort_values(["event", "tile_row", "tile_col"])
    return _finish(cfg, df, meta, origin, "tiles")


def evaluate_patches(cfg: Config, split: str = "val", device: str | None = None, batch: int = 16) -> Path:
    """Score cached patches as tiles; writes ``tiles_patches_<split>.csv``.

    Used for Experiment A (the model's own validation patches) and to check a trained model on
    its training distribution."""
    import pandas as pd

    device = pick_device(device)
    ds = get_dataset(cfg)
    meta = ds.meta()
    patches = ds.patches()
    patches = patches[patches["split"] == split].reset_index(drop=True)
    if cfg.max_val_patches:
        patches = patches.head(cfg.max_val_patches)
    model, origin = load_for_eval(cfg, device)
    predictor = Predictor(cfg, model, meta.channels, device)
    images, labels = ds.open_arrays(split)
    torch.manual_seed(cfg.seed)
    rows = []
    for i in range(0, len(patches), batch):
        sub = patches.iloc[i:i + batch]
        idx = sub["index"].to_numpy()
        img = np.stack([np.asarray(images[j]) for j in idx])
        gt = np.stack([np.asarray(labels[j]) for j in idx])
        out = predictor(img, crop=0)
        for j, r in enumerate(sub.itertuples()):
            m = {"event": r.event, "tile_id": r.patch_id, "tile_row": -1, "tile_col": -1}
            rows.append(tile_row(m, out, j, gt[j], gt[j] != INVALID, meta.classes, meta.target, cfg.mc_passes))
    return _finish(cfg, pd.DataFrame(rows), meta, origin, f"tiles_patches_{split}")
