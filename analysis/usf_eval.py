#!/usr/bin/env python
"""Evaluate the Sentinel-1 water U-Net on the UrbanSARFloods test set (flood = water(post) AND NOT water(pre)).

Dataset: Hugging Face ``S1Floodbenchmark/UrbanSARFloods_v1`` (Zhao et al. 2024, CVPR EarthVision),
test events 20210727_Weihui, 20230609_NovaKakhovka, 20231201_Jubba_1, 20231201_Jubba_2.

The steps are subcommands so they can run on a laptop-sized machine one event at a time, or as a
Slurm job array on an HPC (see ``analysis/slurm/``). Every step is restartable: outputs that already
exist are reused.

  download EVENT  (optional) pre-download scene + chips, for compute nodes without internet
  bands   EVENT   identify the 8 bands from their statistics (the GeoTIFF band descriptions are empty)
  prepare EVENT   download the full scene + GT, write a 4-band 8-bit image (pre VV, pre VH, post VV,
                  post VH) with ``s1water.preprocess.db_to_uint8``, a valid-data mask and a uint8 GT;
                  the 7-9 GB float32 scene is deleted afterwards
  chips   EVENT   download the 256 px chips (optionally one shard), run the U-Net on pre and post
                  (deterministic + MC dropout), score against GT, write a per-shard CSV; raw chips are
                  deleted as soon as they are converted
  scene   EVENT   run the U-Net on the full pre and post scenes with the original tiling (64 px crop), write
                  the pixel error map (GeoTIFF + PNG), scene metrics, and the per-tile error map figure
  merge           concatenate the per-shard CSVs into ``analysis/usf_results.csv`` and write summaries

Band identification (``bands``) is data-driven and is checked at the start of ``prepare`` and ``chips``:
if the statistics of an event disagree with ``EXPECTED_BANDS`` the run stops.

Run ``python analysis/usf_eval.py -h`` for the options.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# hf-xet keeps a chunk cache (10 GB by default) next to the downloads: disable it, disk is the bottleneck
os.environ.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", "0")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import numpy as np  # noqa: E402

log = logging.getLogger("usf_eval")

REPO_ID = "S1Floodbenchmark/UrbanSARFloods_v1"
REPO_REVISION = "0892ffa10cedb7ebd8456e1386b7a91615230429"  # dataset commit used for the results
EVENTS = {
    # event: (full-scene SAR file, full-scene GT file) inside testing_case_orig/<event>/
    "20210727_Weihui": ("20210727_Weihui_SAR.tif", "20210727_Weihui_GT.tif"),
    "20230609_NovaKakhovka": ("20230609_NovaKakhovka_SAR.tif", "20230609_NovaKakhovka_GT.tif"),
    "20231201_Jubba_1": ("20231201_Jubba_SAR.tif", "20231201_Jubba_1_GT.tif"),
    "20231201_Jubba_2": ("20231201_Jubba_2_SAR.tif", "20231201_Jubba_2_GT.tif"),
}
CHIP = 256
# 1-based intensity band indices found by `bands` on all four test events (analysis/band_identification.md).
# The GeoTIFFs carry no band descriptions and the paper lists the groups in another order (intensity
# first), so this mapping comes from the data. Bands 1-4 are coherence (values in [0, 1]); their
# pre/co-event and VV/VH roles are not needed here and are not reliably identifiable.
EXPECTED_BANDS = {"int_pre_VH": 5, "int_pre_VV": 6, "int_post_VH": 7, "int_post_VV": 8}
INTENSITY_BANDS = [EXPECTED_BANDS[k] for k in ("int_pre_VV", "int_pre_VH", "int_post_VV", "int_post_VH")]
GT_CLASSES = {1: "open", 2: "urban"}
GT_NODATA = 255
THRESHOLD = 0.5

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = Path(os.environ.get("USF_DATA", ROOT.parent / "data" / "usf"))


# --------------------------------------------------------------------------------------------- utils
def setup_torch(threads: int | None):
    import torch

    if threads:
        torch.set_num_threads(threads)
    return torch


def hf_download(filename: str, local_dir: Path, retries: int = 6) -> Path:
    """Download one dataset file to local_dir/filename, with back-off on rate limits / network errors."""
    from huggingface_hub import hf_hub_download

    delay = 5.0
    for attempt in range(retries):
        try:
            return Path(hf_hub_download(REPO_ID, filename, repo_type="dataset", revision=REPO_REVISION,
                                        local_dir=str(local_dir)))
        except Exception as err:  # noqa: BLE001 - HTTP 429 / connection resets
            if attempt == retries - 1:
                raise
            log.warning("download %s failed (%s), retry in %.0f s", filename, err, delay)
            time.sleep(delay)
            delay = min(delay * 2, 300)
    raise RuntimeError("unreachable")


def list_dataset_files() -> dict[str, int]:
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(REPO_ID, revision=REPO_REVISION, files_metadata=True)
    return {s.rfilename: (s.size or 0) for s in info.siblings}


def event_dir(data: Path, event: str) -> Path:
    d = data / event
    d.mkdir(parents=True, exist_ok=True)
    return d


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def to_uint8(db: np.ndarray) -> np.ndarray:
    """dB -> 8-bit with the model's own scaling (NaN -> 250, as SNAP no-data)."""
    from s1water.preprocess import db_to_uint8

    return db_to_uint8(db)


# ----------------------------------------------------------------------------------- band check
def band_stats(x: np.ndarray, gt: np.ndarray) -> dict:
    """Per-band statistics used to identify the bands. x: (8, N) float32, gt: (N,) labels."""
    out = {}
    for b in range(x.shape[0]):
        v = x[b]
        f = np.isfinite(v)
        q = np.percentile(v[f], [1, 50, 99]).tolist() if f.any() else [math.nan] * 3
        by = {}
        for c in (0, 1, 2):
            m = f & (gt == c)
            by[str(c)] = float(v[m].mean()) if m.sum() > 50 else None
        out[b + 1] = {"p01": q[0], "median": q[1], "p99": q[2], "min": float(np.nanmin(v)),
                      "max": float(np.nanmax(v)), "mean_by_gt": by}
    return out


def identify_bands(stats: dict) -> tuple[dict, list[str]]:
    """Infer the band roles from statistics alone. Returns (mapping, reasons)."""
    reasons = []
    coh = [b for b, s in stats.items() if s["min"] >= 0 and s["max"] <= 1.0001]
    db = [b for b, s in stats.items() if b not in coh]
    reasons.append(f"bands {coh} lie in [0, 1] (coherence); bands {db} hold negative dB values (intensity)")
    if len(coh) != 4 or len(db) != 4:
        raise ValueError(f"unexpected band ranges: coherence-like {coh}, dB-like {db}")
    db = sorted(db)
    # cross-pol (VH) backscatter is several dB below co-pol (VV): the lower median is VH
    pairs = [db[:2], db[2:]]
    roles = {}
    drops = []
    for pair in pairs:
        lo, hi = sorted(pair, key=lambda b: stats[b]["median"])
        gap = stats[hi]["median"] - stats[lo]["median"]
        reasons.append(f"pair {pair}: band {hi} median {stats[hi]['median']:.1f} dB, band {lo} "
                       f"{stats[lo]['median']:.1f} dB (gap {gap:.1f} dB) -> {hi}=VV, {lo}=VH")
        if gap < 2:
            raise ValueError(f"VV/VH gap too small in pair {pair}")
        # open flood (GT=1) darkens the post-event image: drop of the mean over GT=1 vs GT=0
        d = [stats[b]["mean_by_gt"]["1"] - stats[b]["mean_by_gt"]["0"] for b in pair]
        drops.append(float(np.mean(d)))
        roles[tuple(pair)] = {"VV": hi, "VH": lo}
    post = int(np.argmin(drops))
    pre = 1 - post
    reasons.append(f"open-flood (GT=1) minus non-flood (GT=0) mean: pair {pairs[0]} {drops[0]:+.1f} dB, "
                   f"pair {pairs[1]} {drops[1]:+.1f} dB -> the darker pair {pairs[post]} is post-event")
    mapping = {
        "int_pre_VV": roles[tuple(pairs[pre])]["VV"], "int_pre_VH": roles[tuple(pairs[pre])]["VH"],
        "int_post_VV": roles[tuple(pairs[post])]["VV"], "int_post_VH": roles[tuple(pairs[post])]["VH"],
    }
    # coherence (not used by the model): the co-event pair should decorrelate over flooded pixels (GT 1 or 2).
    # With few flood pixels and low coherence everywhere (Jubba) this is ambiguous: reported as tentative.
    coh = sorted(coh)
    cpairs = [coh[:2], coh[2:]]
    cdrop = []
    for pair in cpairs:
        d = []
        for b in pair:
            by = stats[b]["mean_by_gt"]
            fl = [by[c] for c in ("1", "2") if by[c] is not None]
            d.append(float(np.mean(fl)) - by["0"])
        cdrop.append(float(np.mean(d)))
    co = int(np.argmin(cdrop))
    reasons.append(f"TENTATIVE: flood (GT 1/2) minus GT=0 coherence: pair {cpairs[0]} {cdrop[0]:+.3f}, "
                   f"pair {cpairs[1]} {cdrop[1]:+.3f} -> pair {cpairs[co]} looks like co-event coherence "
                   "(VV/VH inside coherence pairs not identifiable; coherence is not used by the model)")
    for name, pair in (("pre", cpairs[1 - co]), ("co", cpairs[co])):
        mapping[f"coh_{name}_VH?"], mapping[f"coh_{name}_VV?"] = pair[0], pair[1]
    return mapping, reasons


def cmd_bands(args) -> dict:
    """Read strips of the full scene and identify the bands (local copy if downloaded, else HTTP range reads)."""
    import rasterio
    from rasterio.windows import Window

    os.environ.setdefault("CURL_CA_BUNDLE", os.environ.get("REQUESTS_CA_BUNDLE", ""))
    sar, gtf = EVENTS[args.event]
    base = f"/vsicurl/https://huggingface.co/datasets/{REPO_ID}/resolve/{REPO_REVISION}/testing_case_orig/{args.event}/"
    local = event_dir(args.data, args.event)
    sar_path = local / "testing_case_orig" / args.event / sar
    gt_path = local / "testing_case_orig" / args.event / gtf
    src_sar = str(sar_path) if sar_path.exists() else base + sar
    src_gt = str(gt_path) if gt_path.exists() else base + gtf
    with rasterio.open(src_sar) as s, rasterio.open(src_gt) as g:
        xs, gs = [], []
        for r in np.linspace(s.height * 0.1, s.height * 0.9, args.strips).astype(int):
            w = Window(0, int(r), s.width, 16)
            xs.append(s.read(window=w).reshape(s.count, -1))
            gs.append(g.read(1, window=w).reshape(-1))
        descriptions = list(s.descriptions)
        meta = {"width": s.width, "height": s.height, "count": s.count, "dtype": s.dtypes[0],
                "crs": str(s.crs), "res_deg": list(s.res), "bounds": list(s.bounds),
                "interleave": s.profile.get("interleave"), "nodata": s.nodata, "tags": s.tags()}
    x = np.concatenate(xs, 1)
    gt = np.concatenate(gs)
    stats = band_stats(x, gt)
    mapping, reasons = identify_bands(stats)
    # only the intensity bands are used by the model: they must match; coherence is informative only
    ok = all(mapping[k] == EXPECTED_BANDS[k] for k in EXPECTED_BANDS if k.startswith("int_"))
    out = {"event": args.event, "band_descriptions": descriptions, "meta": meta, "n_pixels_sampled": int(x.shape[1]),
           "gt_counts": {str(int(c)): int(n) for c, n in zip(*np.unique(gt[np.isfinite(gt)], return_counts=True), strict=True)},
           "stats": stats, "inferred": mapping, "reasons": reasons, "matches_expected": ok}
    path = ROOT / "band_check" / f"{args.event}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    for r in reasons:
        log.info("%s: %s", args.event, r)
    log.info("%s: inferred %s, matches EXPECTED_BANDS: %s", args.event, mapping, ok)
    if not ok:
        raise SystemExit(f"band order of {args.event} differs from EXPECTED_BANDS, see {path}")
    return out


def require_band_check(event: str):
    path = ROOT / "band_check" / f"{event}.json"
    if not path.exists() or not json.loads(path.read_text())["matches_expected"]:
        raise SystemExit(f"run `usf_eval.py bands {event}` first (band order must be confirmed per event)")


# --------------------------------------------------------------------------------------- prepare
def scene_paths(data: Path, event: str) -> dict[str, Path]:
    d = event_dir(data, event)
    return {"img8": d / f"{event}_int8_preVV_preVH_postVV_postVH.tif", "valid": d / f"{event}_valid.tif",
            "gt": d / f"{event}_GT_uint8.tif"}


def cmd_prepare(args):
    import rasterio
    from rasterio.windows import Window

    require_band_check(args.event)
    p = scene_paths(args.data, args.event)
    if all(v.exists() for v in p.values()):
        log.info("%s: prepared files exist", args.event)
        return
    sar, gtf = EVENTS[args.event]
    d = event_dir(args.data, args.event)
    log.info("%s: downloading full scene (free disk %.1f GB)", args.event, free_gb(d))
    t = time.time()
    sar_path = hf_download(f"testing_case_orig/{args.event}/{sar}", d)
    gt_path = hf_download(f"testing_case_orig/{args.event}/{gtf}", d)
    log.info("%s: downloaded in %.0f s", args.event, time.time() - t)
    with rasterio.open(sar_path) as s, rasterio.open(gt_path) as g:
        assert (s.width, s.height, s.transform) == (g.width, g.height, g.transform), "SAR and GT grids differ"
        prof = dict(driver="GTiff", width=s.width, height=s.height, crs=s.crs, transform=s.transform,
                    compress="deflate", predictor=2, tiled=True, blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER")
        rows = 512
        with rasterio.open(p["img8"], "w", count=4, dtype="uint8", **prof) as o8, \
                rasterio.open(p["valid"], "w", count=1, dtype="uint8", **prof) as ov, \
                rasterio.open(p["gt"], "w", count=1, dtype="uint8", nodata=GT_NODATA, **prof) as og:
            for r in range(0, s.height, rows):
                w = Window(0, r, s.width, min(rows, s.height - r))
                db = s.read(INTENSITY_BANDS, window=w)
                o8.write(to_uint8(db), window=w)
                ov.write(np.all(np.isfinite(db), axis=0).astype(np.uint8)[None], window=w)
                gv = g.read(1, window=w)
                gu = np.full(gv.shape, GT_NODATA, np.uint8)
                f = np.isfinite(gv)
                gu[f] = gv[f].astype(np.uint8)
                og.write(gu[None], window=w)
            for i, n in enumerate(["pre VV", "pre VH", "post VV", "post VH"], 1):
                o8.set_band_description(i, f"{n} (8-bit, (max(-49,min(1,dB))+50)*5, s1water db_to_uint8)")
            o8.update_tags(source=f"{REPO_ID}@{REPO_REVISION}:{sar}", source_bands=str(INTENSITY_BANDS))
            og.set_band_description(1, "0 non-flood, 1 open flood, 2 urban flood, 255 no data")
    if not args.keep_raw:
        sar_path.unlink()
    log.info("%s: prepared %s (free disk %.1f GB)", args.event, p["img8"].name, free_gb(d))


# -------------------------------------------------------------------------------------- model / MC
def load_unet(device: str):
    from s1water.model import load_model
    from s1water.pipeline import download_weights

    model = load_model(download_weights(), device, half=False)
    if device == "cpu":
        import torch

        model = model.to(memory_format=torch.channels_last)
    return model


def set_mc_dropout(model, on: bool):
    """Dropout layers in train mode (stochastic), everything else (batch norm) in eval mode."""
    import torch

    model.eval()
    if on:
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train()


def forward(model, img8: np.ndarray, device: str, dtype: str):
    """img8: (N, 2, H, W) uint8 VV, VH -> probabilities (N, H, W) float32. Input stacked VV, VH, VV, VH."""
    import torch

    x = torch.from_numpy(img8).to(device).float().div_(255.0)
    x = torch.cat([x, x], dim=1).contiguous(memory_format=torch.channels_last)
    with torch.inference_mode():
        if dtype == "float32":
            p = model(x)
        else:
            with torch.autocast(device_type="cpu" if device == "cpu" else "cuda", dtype=getattr(torch, dtype)):
                p = model(x)
    return p[:, 0].float().cpu().numpy()


def entropy(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, 1e-7, 1 - 1e-7)
    return -(q * np.log2(q) + (1 - q) * np.log2(1 - q))


# ----------------------------------------------------------------------------------------- metrics
def confusion(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray, cls: int) -> tuple[int, int, int, int]:
    """One-vs-rest for one GT flood class. Pixels of the *other* flood class are excluded, because the
    model has a single flood output: a detected urban-flood pixel is not a false alarm for open flood."""
    other = 3 - cls
    m = valid & (gt != GT_NODATA) & (gt != other)
    pos = gt == cls
    tp = int(np.sum(m & pred & pos))
    fp = int(np.sum(m & pred & ~pos))
    fn = int(np.sum(m & ~pred & pos))
    tn = int(np.sum(m & ~pred & ~pos))
    return tp, fp, fn, tn


def scores(tp: int, fp: int, fn: int) -> dict:
    nan = float("nan")
    prec = tp / (tp + fp) if tp + fp else nan
    rec = tp / (tp + fn) if tp + fn else nan
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else nan
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else nan
    return {"f1": f1, "iou": iou, "precision": prec, "recall": rec}


# ------------------------------------------------------------------------------------------- chips
CHIP_RE = re.compile(r"_ID_(\d+)_(\d+)_SAR\.tif$")


def chip_list(event: str, files: dict[str, int]) -> list[tuple[str, int, int, int]]:
    out = []
    for f, size in files.items():
        if f.startswith(f"testing_case_256/{event}/"):
            m = CHIP_RE.search(f)
            if m:
                out.append((f, int(m.group(1)), int(m.group(2)), size))
    return sorted(out, key=lambda t: (t[1], t[2]))


def cmd_chips(args):
    import rasterio
    from rasterio.windows import Window

    require_band_check(args.event)
    torch = setup_torch(args.threads)
    torch.manual_seed(args.seed + args.shard)
    p = scene_paths(args.data, args.event)
    if not p["gt"].exists():
        raise SystemExit(f"run `usf_eval.py prepare {args.event}` first")
    out_csv = ROOT / "results_parts" / f"{args.event}_shard{args.shard:03d}of{args.nshards:03d}.csv"
    out_csv.parent.mkdir(exist_ok=True)
    if out_csv.exists() and not args.overwrite:
        log.info("%s exists, skipping", out_csv)
        return
    files = list_dataset_files()
    chips = chip_list(args.event, files)[args.shard::args.nshards]
    if args.limit:
        chips = chips[: args.limit]
    # checkpoint: one small CSV per processed batch; a restarted job skips the chips already done
    part_dir = out_csv.with_suffix(".partial")
    if args.overwrite:
        shutil.rmtree(part_dir, ignore_errors=True)
    part_dir.mkdir(exist_ok=True)
    done_ids = set()
    import pandas as pd

    for f in sorted(part_dir.glob("batch_*.csv")):
        done_ids.update(pd.read_csv(f, usecols=["tile_id"])["tile_id"])
    if done_ids:
        log.info("resuming: %d chips already done", len(done_ids))
        chips = [c for c in chips if f"{args.event}_ID_{c[1]}_{c[2]}" not in done_ids]
    log.info("%s: %d chips in shard %d/%d (%d empty files on the Hub)", args.event, len(chips), args.shard,
             args.nshards, sum(1 for c in chips if c[3] == 0))
    d = event_dir(args.data, args.event) / f"chips_tmp_{args.shard}"
    d.mkdir(exist_ok=True)
    model = load_unet(args.device)

    gt_src = rasterio.open(p["gt"])  # read in the main thread only

    def fetch(item):
        """Read one chip (local copy, or download then delete), return (meta, 4x256x256 uint8, valid).

        Runs in worker threads: rasterio/GDAL dataset handles are not thread-safe, so every call opens
        its own handles."""
        f, r, c, size = item
        win = Window(c * CHIP, r * CHIP, CHIP, CHIP)
        meta = {"event": args.event, "tile_id": f"{args.event}_ID_{r}_{c}", "tile_row": r, "tile_col": c}
        with rasterio.open(p["img8"]) as img_src:
            scene8 = img_src.read(window=win, boundless=True, fill_value=250)
        if size == 0:
            # empty file on the Hub (997 chips of Jubba_2): use the same window of the full scene, the chips
            # being exact 256 px crops of it (checked on every downloaded chip below)
            meta["chip_source"] = "scene_crop(empty_hub_file)"
            with rasterio.open(p["valid"]) as valid_src:
                valid = valid_src.read(1, window=win, boundless=True, fill_value=0).astype(bool)
            return meta, scene8, valid
        local = event_dir(args.data, args.event) / f  # pre-downloaded with `usf_eval.py download`
        path = local if local.exists() and local.stat().st_size > 0 else hf_download(f, d)
        with rasterio.open(path) as s:
            db = s.read(INTENSITY_BANDS)
            meta["chip_bounds"] = ",".join(f"{v:.6f}" for v in s.bounds)
        if path != local:
            path.unlink()
        img8 = to_uint8(db)
        valid = np.all(np.isfinite(db), axis=0)
        meta["chip_source"] = "hub_chip"
        meta["chip_equals_scene_crop"] = bool(np.array_equal(img8, scene8))
        return meta, img8, valid

    t0 = time.time()
    bs = args.batch
    nbatch = len(list(part_dir.glob("batch_*.csv")))
    with ThreadPoolExecutor(args.download_workers) as pool:
        it = pool.map(fetch, chips)
        batch = []
        done = 0
        for item in it:
            batch.append(item)
            if len(batch) == bs or done + len(batch) == len(chips):
                tmp = part_dir / f"batch_{nbatch:05d}.tmp"
                pd.DataFrame(process_batch(model, batch, gt_src, args)).to_csv(tmp, index=False)
                tmp.replace(tmp.with_suffix(".csv"))
                nbatch += 1
                done += len(batch)
                batch = []
                el = time.time() - t0
                log.info("%s shard %d: %d/%d chips, %.2f s/chip, ETA %.0f min", args.event, args.shard, done,
                         len(chips), el / done, el / done * (len(chips) - done) / 60)
    df = pd.concat([pd.read_csv(f) for f in sorted(part_dir.glob("batch_*.csv"))], ignore_index=True)
    df = df.sort_values(["tile_row", "tile_col"])
    df.to_csv(out_csv.with_suffix(".tmp"), index=False)
    out_csv.with_suffix(".tmp").replace(out_csv)
    shutil.rmtree(part_dir, ignore_errors=True)
    shutil.rmtree(d, ignore_errors=True)
    log.info("wrote %s (%d rows) in %.0f min", out_csv, len(df), (time.time() - t0) / 60)


def process_batch(model, batch, gt_src, args) -> list[dict]:
    from rasterio.windows import Window

    metas = [b[0] for b in batch]
    img8 = np.stack([b[1] for b in batch])  # (N, 4, H, W): pre VV, pre VH, post VV, post VH
    valid = np.stack([b[2] for b in batch])
    gts = np.stack([gt_src.read(1, window=Window(m["tile_col"] * CHIP, m["tile_row"] * CHIP, CHIP, CHIP),
                                boundless=True, fill_value=GT_NODATA) for m in metas])
    n = len(batch)
    both = np.concatenate([img8[:, :2], img8[:, 2:]])  # pre chips then post chips, each (VV, VH)
    keep = valid.reshape(n, -1).any(1)  # chips with no valid pixel: no inference, metrics NaN

    set_mc_dropout(model, False)
    p_pre = np.full((n, CHIP, CHIP), np.nan, np.float32)
    p_post = p_pre.copy()
    if keep.any():
        idx = np.concatenate([np.where(keep)[0], n + np.where(keep)[0]])
        pr = forward(model, both[idx], args.device, args.dtype)
        k = keep.sum()
        p_pre[keep], p_post[keep] = pr[:k], pr[k:]

    # MC dropout: T passes, dropout on, batch norm in eval; accumulate in float64
    T = args.mc_passes
    acc = {k: np.zeros((n, CHIP, CHIP)) for k in ("pre", "pre2", "post", "post2", "flood", "flood2")}
    if T and keep.any():
        set_mc_dropout(model, True)
        for _ in range(T):
            pr = forward(model, both[idx], args.device, args.mc_dtype)
            a, b = pr[:k].astype(np.float64), pr[k:].astype(np.float64)
            fl = b * (1 - a)  # P(water post) * P(not water pre), per pass
            for name, v in (("pre", a), ("post", b), ("flood", fl)):
                acc[name][keep] += v
                acc[name + "2"][keep] += v * v
        set_mc_dropout(model, False)

    rows = []
    for i, m in enumerate(metas):
        v = valid[i]
        g = gts[i]
        row = dict(m)
        row["n_valid"] = int(v.sum())
        row["n_gt_open"] = int(np.sum(v & (g == 1)))
        row["n_gt_urban"] = int(np.sum(v & (g == 2)))
        row["n_gt_nodata"] = int(np.sum(v & (g == GT_NODATA)))
        if not keep[i]:
            rows.append(row)
            continue
        w_pre = p_pre[i] >= THRESHOLD
        w_post = p_post[i] >= THRESHOLD
        preds = {"flood": w_post & ~w_pre, "post": w_post}
        for pname, pred in preds.items():
            for cls, cname in GT_CLASSES.items():
                tp, fp, fn, tn = confusion(pred, g, v, cls)
                row.update({f"{pname}_{cname}_{k}": val for k, val in
                            {"tp": tp, "fp": fp, "fn": fn, "tn": tn}.items()})
                row.update({f"{pname}_{cname}_{k}": val for k, val in scores(tp, fp, fn).items()})
        row["frac_water_pre"] = float(w_pre[v].mean())
        row["frac_water_post"] = float(w_post[v].mean())
        row["frac_flood"] = float(preds["flood"][v].mean())
        p_flood_det = p_post[i] * (1 - p_pre[i])
        row["mean_prob_pre"] = float(p_pre[i][v].mean())
        row["mean_prob_post"] = float(p_post[i][v].mean())
        row["mean_prob_flood"] = float(p_flood_det[v].mean())
        row["mean_entropy_post_det"] = float(entropy(p_post[i][v]).mean())
        if T:
            for name in ("pre", "post", "flood"):
                mu = acc[name][i] / T
                var = np.maximum(acc[name + "2"][i] / T - mu * mu, 0) * T / (T - 1)  # unbiased
                row[f"mc_mean_prob_{name}"] = float(mu[v].mean())
                row[f"mc_var_{name}"] = float(var[v].mean())
                row[f"mean_entropy_{name}"] = float(entropy(mu[v]).mean())
        row["mc_passes"] = T
        row["det_dtype"], row["mc_dtype"] = args.dtype, args.mc_dtype
        rows.append(row)
    return rows


# ------------------------------------------------------------------------------------------- scene
ERR_CODES = {0: ("no data", (255, 255, 255)), 1: ("TN", (215, 215, 215)), 2: ("TP open", (0, 110, 255)),
             3: ("TP urban", (150, 0, 200)), 4: ("FN open", (255, 165, 0)), 5: ("FN urban", (255, 0, 160)),
             6: ("FP (non-flood)", (220, 20, 20))}


def error_map(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> np.ndarray:
    e = np.zeros(gt.shape, np.uint8)
    ok = valid & (gt != GT_NODATA)
    e[ok & ~pred & (gt == 0)] = 1
    e[ok & pred & (gt == 1)] = 2
    e[ok & pred & (gt == 2)] = 3
    e[ok & ~pred & (gt == 1)] = 4
    e[ok & ~pred & (gt == 2)] = 5
    e[ok & pred & (gt == 0)] = 6
    return e


def downsample_codes(e: np.ndarray, f: int) -> np.ndarray:
    """Block reduction keeping the most informative code (errors and hits over TN) so that sparse
    flood pixels stay visible in an overview: priority FN/FP/TP > TN > no data."""
    h, w = (e.shape[0] // f) * f, (e.shape[1] // f) * f
    b = e[:h, :w].reshape(h // f, f, w // f, f).transpose(0, 2, 1, 3).reshape(h // f, w // f, -1)
    out = np.zeros(b.shape[:2], np.uint8)
    counts = {c: (b == c).sum(-1) for c in ERR_CODES}
    out[counts[1] > 0] = 1
    for c in (2, 3, 6, 4, 5):  # later codes win: missed floods drawn on top
        out[counts[c] >= max(1, f * f // 50)] = c
    return out


def predict_full(model, img2: np.ndarray, device: str, dtype: str, tile: int = 4224) -> np.ndarray:
    """Full-image prediction with the original tiling scheme (64 px crop). tile = 4224 is the original;
    a smaller tile keeps the same 64 px context margin and fits in less memory (fp32 4224 px needs ~15 GB)."""
    from s1water.predict import CROP, predict_image

    kw = dict(return_probability=True, progress=False, tile=tile, step=tile - 2 * CROP, crop=CROP)
    if dtype != "float32":
        import torch

        with torch.autocast(device_type="cpu" if device == "cpu" else "cuda", dtype=getattr(torch, dtype)):
            return predict_image(model, img2, device, **kw)
    return predict_image(model, img2, device, **kw)


def cmd_scene(args):
    import rasterio

    setup_torch(args.threads)
    if args.device == "cpu" and args.dtype != "float32" and args.tile > 2176:
        # oneDNN fails with "could not create a primitive" for bfloat16 convolutions on 4224 px tiles
        log.warning("%s autocast is not supported on CPU for 4224 px tiles: using float32", args.dtype)
        args.dtype = "float32"
    p = scene_paths(args.data, args.event)
    d = event_dir(args.data, args.event)
    prob_path = {k: d / f"{args.event}_prob_{k}.tif" for k in ("pre", "post")}
    with rasterio.open(p["img8"]) as s:
        prof = s.profile.copy()
        img = s.read()
    model = None
    probs = {}
    for k, bands in (("pre", slice(0, 2)), ("post", slice(2, 4))):
        if prob_path[k].exists():
            with rasterio.open(prob_path[k]) as s:
                probs[k] = s.read(1)
            continue
        model = model or load_unet(args.device)
        set_mc_dropout(model, False)
        t = time.time()
        probs[k] = predict_full(model, np.ascontiguousarray(img[bands]), args.device, args.dtype, args.tile)
        log.info("%s %s: full scene predicted in %.1f min", args.event, k, (time.time() - t) / 60)
        pp = dict(prof, count=1, dtype="float32", nodata=np.nan, predictor=3)
        with rasterio.open(prob_path[k], "w", **pp) as o:
            o.write(probs[k], 1)
    del img
    with rasterio.open(p["valid"]) as s:
        valid = s.read(1).astype(bool)
    with rasterio.open(p["gt"]) as s:
        gt = s.read(1)
    # predict_image leaves the first 64 rows/cols unpredicted (NaN), like the original workflow
    valid &= np.isfinite(probs["pre"]) & np.isfinite(probs["post"])
    w_pre = probs["pre"] >= THRESHOLD
    w_post = probs["post"] >= THRESHOLD
    del probs
    preds = {"flood": w_post & ~w_pre, "post": w_post}
    rec = {"event": args.event, "n_valid": int(valid.sum()), "tile": args.tile, "dtype": args.dtype}
    for pname, pred in preds.items():
        for cls, cname in GT_CLASSES.items():
            tp, fp, fn, tn = confusion(pred, gt, valid, cls)
            rec.update({f"{pname}_{cname}_{k}": v for k, v in {"tp": tp, "fp": fp, "fn": fn, "tn": tn}.items()})
            rec.update({f"{pname}_{cname}_{k}": v for k, v in scores(tp, fp, fn).items()})
    (ROOT / "scene_metrics").mkdir(exist_ok=True)
    (ROOT / "scene_metrics" / f"{args.event}.json").write_text(json.dumps(rec, indent=1))
    for pname, pred in preds.items():
        e = error_map(pred, gt, valid)
        ep = dict(prof, count=1, dtype="uint8", nodata=0, photometric="palette")
        with rasterio.open(d / f"{args.event}_error_{pname}.tif", "w", **ep) as o:
            o.write_colormap(1, {c: (*rgb, 255) for c, (_, rgb) in ERR_CODES.items()})  # before the data
            o.write(e, 1)
            o.update_tags(codes=json.dumps({c: n for c, (n, _) in ERR_CODES.items()}))
        f = max(1, int(math.ceil(max(e.shape) / 2400)))
        small = downsample_codes(e, f)
        np.save(d / f"{args.event}_error_{pname}_small.npy", small)
        del e
    log.info("%s scene metrics: flood open F1 %.3f, urban F1 %.3f; post open F1 %.3f", args.event,
             rec["flood_open_f1"], rec["flood_urban_f1"], rec["post_open_f1"])
    plot_scene(args.event, args.data, prof)


def plot_scene(event: str, data: Path, prof: dict):
    """Figure: pixel error map (full-context prediction) + per-256 px tile metrics from the chips CSV."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    d = event_dir(data, event)
    small = np.load(d / f"{event}_error_flood_small.npy")
    parts = sorted((ROOT / "results_parts").glob(f"{event}_shard*.csv"))
    df = pd.concat([pd.read_csv(f) for f in parts]) if parts else None
    t = prof["transform"]
    ext = [t.c, t.c + t.a * prof["width"], t.f + t.e * prof["height"], t.f]
    cmap = ListedColormap([np.array(rgb) / 255 for _, (_, rgb) in sorted(ERR_CODES.items())])
    ncol = 3 if df is not None else 1
    fig, axes = plt.subplots(1, ncol, figsize=(7 * ncol, 6), constrained_layout=True)
    axes = np.atleast_1d(axes)
    axes[0].imshow(small, cmap=cmap, vmin=-0.5, vmax=6.5, extent=ext, interpolation="nearest")
    axes[0].set_title(f"{event}: flood = water(post) AND NOT water(pre)\nfull-scene prediction, pixel errors")
    axes[0].legend(handles=[Patch(color=np.array(rgb) / 255, label=n) for c, (n, rgb) in ERR_CODES.items() if c],
                   loc="lower left", fontsize=7, framealpha=0.9)
    if df is not None:
        nr = int(math.ceil(prof["height"] / CHIP))
        nc = int(math.ceil(prof["width"] / CHIP))
        for ax, col, title, cm, vmax in (
            (axes[1], "flood_err", "per-tile flood error rate (chip inference)\n(FP + FN) / valid px, open+urban",
             "magma", None),
            (axes[2], "mc_var_flood", "per-tile MC-dropout variance\nof P(flood), 20 passes", "viridis", None),
        ):
            g = np.full((nr, nc), np.nan)
            if col == "flood_err":
                fp = df["flood_open_fp"].fillna(0)  # FP on GT=0 pixels (same set for both classes)
                fn = df["flood_open_fn"].fillna(0) + df["flood_urban_fn"].fillna(0)
                vals = ((fp + fn) / df["n_valid"].where(df["n_valid"] > 0)).to_numpy()
            else:
                vals = df[col].to_numpy() if col in df else np.full(len(df), np.nan)
            g[df["tile_row"].to_numpy(), df["tile_col"].to_numpy()] = vals
            im = ax.imshow(g, cmap=cm, extent=ext, interpolation="nearest",
                           vmax=vmax if vmax else np.nanpercentile(g, 99) if np.isfinite(g).any() else None)
            ax.set_title(title)
            fig.colorbar(im, ax=ax, shrink=0.7)
    for ax in axes:
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
    out = ROOT / "figures" / f"error_map_{event}.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    log.info("wrote %s", out)


# ---------------------------------------------------------------------------------------- download
def cmd_download(args):
    """Pre-download an event (full scene + GT, and the chips unless --no-chips), for HPC compute nodes
    without internet access. Run it on a login / data-transfer node. Needs ~48 GB for all four events."""
    from huggingface_hub import snapshot_download

    d = event_dir(args.data, args.event)
    patterns = [f"testing_case_orig/{args.event}/*"] + ([] if args.no_chips else [f"testing_case_256/{args.event}/*"])
    snapshot_download(REPO_ID, repo_type="dataset", revision=REPO_REVISION, local_dir=str(d),
                      allow_patterns=patterns, max_workers=args.workers)
    log.info("%s downloaded to %s (free disk %.1f GB)", args.event, d, free_gb(d))


# ------------------------------------------------------------------------------------------- merge
def cmd_merge(args):
    import pandas as pd

    parts = sorted((ROOT / "results_parts").glob("*_shard*.csv"))
    df = pd.concat([pd.read_csv(f) for f in parts], ignore_index=True).sort_values(["event", "tile_row", "tile_col"])
    df.to_csv(ROOT / "usf_results.csv", index=False, float_format="%.6g")
    lines = ["| event | prediction | class | tiles | TP | FP | FN | precision | recall | F1 | IoU |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    summ = []
    for ev, g in [*df.groupby("event"), ("all", df)]:
        for pname in ("flood", "post"):
            for cname in GT_CLASSES.values():
                tp, fp, fn = (int(g[f"{pname}_{cname}_{k}"].sum()) for k in ("tp", "fp", "fn"))
                s = scores(tp, fp, fn)
                summ.append({"event": ev, "prediction": pname, "class": cname, "tp": tp, "fp": fp, "fn": fn, **s})
                lines.append(f"| {ev} | {pname} | {cname} | {len(g)} | {tp} | {fp} | {fn} | {s['precision']:.3f} | "
                             f"{s['recall']:.3f} | {s['f1']:.3f} | {s['iou']:.3f} |")
    scene_files = sorted((ROOT / "scene_metrics").glob("*.json"))
    if scene_files:
        sl = ["| event | prediction | class | TP | FP | FN | precision | recall | F1 | IoU |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for f in scene_files:
            rec = json.loads(f.read_text())
            for pname in ("flood", "post"):
                for cname in GT_CLASSES.values():
                    k = f"{pname}_{cname}_"
                    sl.append(f"| {rec['event']} | {pname} | {cname} | {rec[k + 'tp']} | {rec[k + 'fp']} | "
                              f"{rec[k + 'fn']} | {rec[k + 'precision']:.3f} | {rec[k + 'recall']:.3f} | "
                              f"{rec[k + 'f1']:.3f} | {rec[k + 'iou']:.3f} |")
        (ROOT / "usf_summary_scenes.md").write_text(
            "Pixel scores of the full-scene prediction (tiling with 64 px crop; tile size in the JSON), from "
            "`scene_metrics/*.json`.\n\n" + "\n".join(sl) + "\n")
    pd.DataFrame(summ).to_csv(ROOT / "usf_summary_chips.csv", index=False, float_format="%.6g")
    (ROOT / "usf_summary_chips.md").write_text(
        "Pixel-pooled (micro) scores over all 256 px chips, from `usf_results.csv`.\n\n" + "\n".join(lines) + "\n")
    log.info("merged %d rows from %d parts", len(df), len(parts))


# -------------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA, help="data/work directory (env USF_DATA)")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bands")
    b.add_argument("event", choices=EVENTS)
    b.add_argument("--strips", type=int, default=12)
    pr = sub.add_parser("prepare")
    pr.add_argument("event", choices=EVENTS)
    pr.add_argument("--keep-raw", action="store_true")
    c = sub.add_parser("chips")
    c.add_argument("event", choices=EVENTS)
    c.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    c.add_argument("--nshards", type=int, default=1)
    c.add_argument("--batch", type=int, default=16)
    c.add_argument("--mc-passes", type=int, default=20)
    c.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"],
                   help="precision of the deterministic pass")
    c.add_argument("--mc-dtype", default="float32", choices=["float32", "bfloat16", "float16"],
                   help="precision of the MC-dropout passes (bfloat16 is ~3x faster on recent x86 CPUs)")
    c.add_argument("--download-workers", type=int, default=6)
    c.add_argument("--limit", type=int, default=0, help="first N chips of the shard only (tests)")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--overwrite", action="store_true")
    s = sub.add_parser("scene")
    s.add_argument("event", choices=EVENTS)
    s.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    s.add_argument("--tile", type=int, default=4224,
                   help="tile size (step = tile - 128, 64 px crop). 4224 = original; needs ~15 GB in float32 on CPU")
    dl = sub.add_parser("download")
    dl.add_argument("event", choices=EVENTS)
    dl.add_argument("--no-chips", action="store_true")
    dl.add_argument("--workers", type=int, default=8)
    sub.add_parser("merge")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    args.data.mkdir(parents=True, exist_ok=True)
    {"bands": cmd_bands, "download": cmd_download, "prepare": cmd_prepare, "chips": cmd_chips, "scene": cmd_scene, "merge": cmd_merge}[args.cmd](args)


if __name__ == "__main__":
    main()
