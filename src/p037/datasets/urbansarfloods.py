"""UrbanSARFloods (Zhao et al. 2024, CVPR EarthVision): Hugging Face ``S1Floodbenchmark/UrbanSARFloods_v1``.

Training and validation chips (512 px, 8 bands, float32) are in one archive,
``urban_sar_floods.tar.gz`` (38 GB), under ``<folder>/SAR/<chip>_SAR.tif`` with a matching
``<chip>_GT.tif``; ``Train_dataset.txt`` and ``Valid_dataset.txt`` list the official split.
The 4 held-out test scenes are whole GeoTIFFs under ``testing_case_orig/<event>/``.

Facts established earlier in the project (``analysis/band_identification.md``,
``analysis/proxy_distributions.md``) that this adapter relies on, and re-checks where it can:

* Band order in the files: 1-4 coherence, 5 = pre VH, 6 = pre VV, 7 = post VH, 8 = post VV.
  The GeoTIFFs carry no band descriptions; the order was identified from the data on the 4 test
  scenes. ``build_cache`` re-checks it on every training event (see ``check_bands``).
* Labels: 0 non-flood, 1 open flood, 2 urban flood. They are change-based: permanent water is 0.
* The official split is random within each scene (98.7 % of validation chips touch a training
  chip), so validation scores are for checkpoint selection, not evidence of generalisation.
* Each 512 px chip is cut into 4 quadrants of 256 px (Wagner's patch size, and the size of a
  test tile). Quadrant ids ``<chip>_q0..q3`` = top-left, top-right, bottom-left, bottom-right,
  the ids used by the proxy table (``analysis/proxy/outputs/proxy_values.csv.gz``).
"""

from __future__ import annotations

import json
import logging
import re
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ..config import REPO_ROOT
from .base import INVALID, CacheMeta, Dataset, create_arrays, db_to_uint8, patch_counts, uint8_to_db

log = logging.getLogger("p037.usf")

REPO_ID = "S1Floodbenchmark/UrbanSARFloods_v1"
REPO_REVISION = "0892ffa10cedb7ebd8456e1386b7a91615230429"  # dataset commit used throughout the project
ARCHIVE = "urban_sar_floods.tar.gz"
FOOTPRINTS = REPO_ROOT / "analysis" / "proxy" / "outputs" / "usf_chip_footprints.csv.gz"

# 1-based band numbers of the intensity bands, and the order they are stored in the cache
BAND_OF = {"pre_VH": 5, "pre_VV": 6, "post_VH": 7, "post_VV": 8}
CHANNELS = ["pre_VV", "pre_VH", "post_VV", "post_VH"]
BANDS = [BAND_OF[c] for c in CHANNELS]
COHERENCE_BANDS = [1, 2, 3, 4]

TEST_SCENES = {
    # scene: (SAR file, GT file) inside testing_case_orig/<scene>/
    "20210727_Weihui": ("20210727_Weihui_SAR.tif", "20210727_Weihui_GT.tif"),
    "20230609_NovaKakhovka": ("20230609_NovaKakhovka_SAR.tif", "20230609_NovaKakhovka_GT.tif"),
    "20231201_Jubba_1": ("20231201_Jubba_SAR.tif", "20231201_Jubba_1_GT.tif"),
    "20231201_Jubba_2": ("20231201_Jubba_2_SAR.tif", "20231201_Jubba_2_GT.tif"),
}
MEMBER_RE = re.compile(r"(?P<chip>.+_ID_\d+_\d+)_(?P<kind>SAR|GT)\.tif$")
QUADRANTS = ((0, 0), (0, 1), (1, 0), (1, 1))  # (row half, column half) of q0..q3


def chip_index(footprints: Path = FOOTPRINTS):
    """One row per official chip: chip, event, folder, split (train / val) and georeferencing.

    Read from the footprint table built in PR #3 from the archive itself (Train_dataset.txt,
    Valid_dataset.txt and every chip's GeoTIFF header)."""
    import pandas as pd

    df = pd.read_csv(footprints)
    df["split"] = df["split"].replace({"valid": "val"})
    df = df[df["split"].isin(["train", "val"])].copy()
    rc = df["chip"].str.extract(r"_ID_(\d+)_(\d+)$").astype(int)
    df["chip_row"], df["chip_col"] = rc[0], rc[1]
    return df.sort_values(["split", "event", "chip_row", "chip_col"]).reset_index(drop=True)


def patch_table(index):
    """Expand the chip index to one row per 256 px quadrant, with its slot in the split's arrays."""
    import pandas as pd

    rows = []
    counters = {"train": 0, "val": 0}
    for ch in index.itertuples():
        for q, (qr, qc) in enumerate(QUADRANTS):
            rows.append({
                "patch_id": f"{ch.chip}_q{q}", "split": ch.split, "index": counters[ch.split],
                "event": ch.event, "chip": ch.chip, "folder": ch.folder, "quadrant": q,
                # centre of the quadrant (lon east, lat north)
                "lon": ch.left + (0.25 + 0.5 * qc) * ch.w * ch.sx,
                "lat": ch.top - (0.25 + 0.5 * qr) * ch.h * ch.sy,
            })
            counters[ch.split] += 1
    return pd.DataFrame(rows)


def _iter_members(source: Path):
    """Yield (file name, path or bytes) for every GeoTIFF of an extracted folder or a tar archive."""
    if source.is_dir():
        for p in sorted(source.rglob("*.tif")):
            yield p.name, p
    else:
        with tarfile.open(source, "r|*") as tar:  # streamed: a .tar.gz cannot be read at random
            for m in tar:
                if m.isfile() and m.name.endswith(".tif"):
                    yield Path(m.name).name, tar.extractfile(m).read()


def _open(payload):
    import rasterio

    if isinstance(payload, bytes | bytearray):
        return rasterio.MemoryFile(bytes(payload)).open()
    return rasterio.open(payload)


def check_bands(stats: dict, min_flood_px: int = 10_000) -> tuple[bool, list[str]]:
    """Check the band order of one event from the mean backscatter (dB) of the cached channels.

    ``stats[channel][class]`` is the mean dB over pixels of that label class (None if too few).
    VV must be at least 2 dB above VH in both dates (cross-pol is weaker). Where the event has
    enough open-flood pixels, they must darken more in the post pair than in the pre pair."""
    ok, reasons = True, []
    for date in ("pre", "post"):
        gap = stats[f"{date}_VV"]["0"] - stats[f"{date}_VH"]["0"]
        good = gap >= 2.0
        ok &= good
        reasons.append(f"{date}: VV - VH = {gap:+.1f} dB over non-flood pixels ({'ok' if good else 'TOO SMALL'})")
    if stats["n_open"] >= min_flood_px and stats["post_VV"]["1"] is not None:
        drop = {d: np.mean([stats[f"{d}_{p}"]["1"] - stats[f"{d}_{p}"]["0"] for p in ("VV", "VH")])
                for d in ("pre", "post")}
        good = drop["post"] < drop["pre"]
        ok &= good
        reasons.append(f"open flood minus non-flood: pre {drop['pre']:+.1f} dB, post {drop['post']:+.1f} dB "
                       f"({'ok: darker after the event' if good else 'WRONG WAY ROUND'})")
    else:
        reasons.append(f"pre/post not checked: only {stats['n_open']} open-flood pixels")
    return bool(ok), reasons


class UrbanSARFloods(Dataset):
    key = "urbansarfloods"
    channels = CHANNELS
    classes = {1: "open", 2: "urban"}
    target = "flood"
    patch = 256
    test_events = list(TEST_SCENES)
    event_groups = {"20231201_Jubba_1": "Jubba", "20231201_Jubba_2": "Jubba",
                    "20210727_Weihui": "Weihui", "20230609_NovaKakhovka": "NovaKakhovka"}

    # --------------------------------------------------------------------------- download
    def download(self, what: str = "archive", event: str | None = None) -> Path:
        """Download the training archive (38 GB) or one test scene to ``<root>/raw``.

        Run this on a login or data-transfer node: compute nodes usually have no internet."""
        from huggingface_hub import hf_hub_download

        raw = self.root / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        if what == "archive":
            names = [ARCHIVE]
        else:
            sar, gt = TEST_SCENES[event]
            names = [f"testing_case_orig/{event}/{sar}", f"testing_case_orig/{event}/{gt}"]
        for name in names:
            log.info("downloading %s", name)
            hf_hub_download(REPO_ID, name, repo_type="dataset", revision=REPO_REVISION, local_dir=str(raw))
        return raw

    # --------------------------------------------------------------------------- patch cache
    def build_cache(self, source: str | None = None, index=None, workers: int = 8,
                    allow_missing: bool = False) -> None:
        """Convert the official train and validation chips to the 8-bit patch cache.

        ``source``: the extracted dataset folder, or the ``.tar.gz`` archive (streamed, nothing is
        extracted). Default: ``<root>/raw/urban_sar_floods.tar.gz``, downloaded if absent.
        Disk: 6.8 GB of images + 1.7 GB of labels for the official 25,884 patches."""
        t0 = time.time()
        src = Path(source) if source else self.root / "raw" / ARCHIVE
        if not src.exists():
            if source:
                raise FileNotFoundError(src)
            self.download("archive")
        index = chip_index() if index is None else index
        patches = patch_table(index)
        first = {(r.chip): (r.split, r.index) for r in patches[patches.quadrant == 0].itertuples()}
        n = patches.groupby("split").size().to_dict()
        P = self.patch
        arrays = {s: create_arrays(self.cache_dir, s, n.get(s, 0), len(CHANNELS), P) for s in n}
        valid = {s: np.lib.format.open_memmap(self.cache_dir / f"_valid_{s}.npy", mode="w+", dtype=np.uint8,
                                              shape=(n[s], P, P)) for s in n}
        seen = {"SAR": set(), "GT": set()}
        range_checked: dict[str, int] = {}
        problems: list[str] = []
        lock = threading.Lock()
        event_of = dict(zip(index.chip, index.event, strict=True))

        def handle(item):
            name, payload = item
            m = MEMBER_RE.search(name)
            if not m or m["chip"] not in first:
                return  # test chips, text files, anything outside the official split
            chip, kind = m["chip"], m["kind"]
            split, i0 = first[chip]
            with _open(payload) as src_ds:
                if kind == "SAR":
                    with lock:  # full 8-band range check on the first 3 chips of each event
                        ev = event_of[chip]
                        do_check = range_checked.get(ev, 0) < 3
                        range_checked[ev] = range_checked.get(ev, 0) + 1
                    if do_check:
                        full = src_ds.read()
                        coh, inten = full[[b - 1 for b in COHERENCE_BANDS]], full[[b - 1 for b in BANDS]]
                        if np.nanmin(coh) < -1e-4 or np.nanmax(coh) > 1.0001 or not (np.nanmedian(inten) < -3):
                            with lock:
                                problems.append(f"{chip}: bands 1-4 are not in [0, 1] or bands 5-8 are not dB")
                        db = inten
                    else:
                        db = src_ds.read(BANDS)
                    if db.shape[1:] != (2 * P, 2 * P):
                        raise ValueError(f"{name}: expected {2 * P} px chips, got {db.shape}")
                    img8, ok = db_to_uint8(db), np.all(np.isfinite(db), axis=0).astype(np.uint8)
                    for q, (qr, qc) in enumerate(QUADRANTS):
                        sl = (slice(qr * P, (qr + 1) * P), slice(qc * P, (qc + 1) * P))
                        arrays[split][0][i0 + q] = img8[(slice(None), *sl)]
                        valid[split][i0 + q] = ok[sl]
                else:
                    g = src_ds.read(1)
                    lab = np.full(g.shape, INVALID, np.uint8)
                    good = np.isfinite(g) & np.isin(g, [0, *self.classes])
                    lab[good] = g[good].astype(np.uint8)
                    for q, (qr, qc) in enumerate(QUADRANTS):
                        arrays[split][1][i0 + q] = lab[qr * P:(qr + 1) * P, qc * P:(qc + 1) * P]
            with lock:
                seen[kind].add(chip)
                done = len(seen["SAR"]) + len(seen["GT"])
                if done % 2000 == 0:
                    log.info("%d / %d files, %.0f min", done, 2 * len(first), (time.time() - t0) / 60)

        if src.is_dir():
            with ThreadPoolExecutor(workers) as pool:
                list(pool.map(handle, _iter_members(src)))
        else:
            for item in _iter_members(src):
                handle(item)

        if problems:
            raise SystemExit("band range check failed:\n  " + "\n  ".join(problems[:10]))
        complete = seen["SAR"] & seen["GT"]
        missing = sorted(set(first) - complete)
        if missing and not allow_missing:
            raise SystemExit(f"{len(missing)} of {len(first)} chips have no SAR or no GT file in {src}, "
                             f"e.g. {missing[:3]}. The cache would be incomplete; nothing was finalised.")
        # pixels without SAR data are excluded from the loss and the scores: label 255
        for s in n:
            for i in range(0, n[s], 1024):
                lab = arrays[s][1][i:i + 1024]
                lab[np.asarray(valid[s][i:i + 1024]) == 0] = INVALID
            arrays[s][0].flush()
            arrays[s][1].flush()
            del valid[s]
            (self.cache_dir / f"_valid_{s}.npy").unlink()
        for s in n:
            m = patches["split"] == s
            for k, v in patch_counts(arrays[s][1], self.classes).items():
                patches.loc[m, k] = v[patches.loc[m, "index"].to_numpy()]
        patches = patches[patches["chip"].isin(complete)]
        count_cols = ["n_valid", *(f"n_{c}" for c in self.classes.values())]
        patches[count_cols] = patches[count_cols].astype(int)

        band_report = self._band_report(patches, arrays)
        (self.cache_dir / "band_check.json").write_text(json.dumps(band_report, indent=1))
        bad = [e for e, r in band_report.items() if not r["ok"]]
        if bad:
            raise SystemExit(f"band order check failed for {bad}: see {self.cache_dir / 'band_check.json'}")

        patches.to_csv(self.cache_dir / "patches.csv", index=False)
        meta = CacheMeta(dataset=self.key, channels=CHANNELS, patch=P, classes=self.classes, target=self.target,
                         source=f"{REPO_ID}@{REPO_REVISION} ({src.name})",
                         notes="8-bit = (max(-49, min(1, dB)) + 50) * 5; label 255 = no label or no SAR data")
        (self.cache_dir / "meta.json").write_text(meta.to_json())
        log.info("cache written to %s in %.0f min", self.cache_dir, (time.time() - t0) / 60)
        log.info("\n%s", class_balance(patches, self.classes).to_string())

    def _band_report(self, patches, arrays, max_patches: int = 400) -> dict:
        """Mean dB per cached channel and label class, per event, and the band-order verdict."""
        rng = np.random.default_rng(0)
        report = {}
        for event, sub in patches.groupby("event"):
            sums = {c: {k: [0.0, 0] for k in ("0", "1")} for c in CHANNELS}
            sub = sub.sample(min(len(sub), max_patches), random_state=int(rng.integers(1 << 30)))
            # patches with open flood first, so the pre/post test has pixels to work with
            sub = sub.sort_values("n_open", ascending=False)
            for r in sub.itertuples():
                img = uint8_to_db(arrays[r.split][0][r.index])
                lab = np.asarray(arrays[r.split][1][r.index])
                for k in ("0", "1"):
                    m = lab == int(k)
                    if m.any():
                        for ci, c in enumerate(CHANNELS):
                            sums[c][k][0] += float(img[ci][m].sum())
                            sums[c][k][1] += int(m.sum())
            stats = {c: {k: (v[0] / v[1] if v[1] > 50 else None) for k, v in d.items()} for c, d in sums.items()}
            stats["n_open"] = sums["post_VV"]["1"][1]
            ok, reasons = check_bands(stats)
            report[event] = {"ok": ok, "reasons": reasons, "mean_db": stats}
            log.info("%s: %s", event, "; ".join(reasons))
        return report

    # --------------------------------------------------------------------------- test scenes
    def prepare_scene(self, event: str, source: str | None = None, keep_raw: bool = False) -> None:
        """Write the 8-bit image, label and valid-data GeoTIFFs of one test scene.

        ``source``: a folder holding the scene's ``*_SAR.tif`` and ``*_GT.tif`` (searched
        recursively). Default: download them from Hugging Face (6-9 GB per scene)."""
        import rasterio
        from rasterio.windows import Window

        out = self.scene_files(event)
        if self.has_scene(event):
            log.info("%s: scene already prepared", event)
            return
        sar_name, gt_name = TEST_SCENES[event]
        root = Path(source) if source else self.download("scene", event)
        found = {n: next(iter(sorted(root.rglob(n))), None) for n in (sar_name, gt_name)}
        if None in found.values():
            raise FileNotFoundError(
                f"{[n for n, p in found.items() if p is None]} not found under {root}. The 4 test scenes are separate "
                f"files of the dataset (testing_case_orig/, not in the training archive). Either point --scene-source "
                f"at the folder that holds them, or download them on a node with internet: "
                f"`p037 data --config <config> --what scenes` (no --source).")
        self.scene_dir(event).mkdir(parents=True, exist_ok=True)
        with rasterio.open(found[sar_name]) as s, rasterio.open(found[gt_name]) as g:
            if (s.width, s.height, s.transform) != (g.width, g.height, g.transform):
                raise ValueError(f"{event}: SAR and GT grids differ")
            prof = dict(driver="GTiff", width=s.width, height=s.height, crs=s.crs, transform=s.transform,
                        compress="deflate", tiled=True, blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER")
            tmp = {k: p.with_suffix(".tmp.tif") for k, p in out.items()}
            with rasterio.open(tmp["image"], "w", count=len(CHANNELS), dtype="uint8", **prof) as o8, \
                    rasterio.open(tmp["valid"], "w", count=1, dtype="uint8", **prof) as ov, \
                    rasterio.open(tmp["gt"], "w", count=1, dtype="uint8", nodata=INVALID, **prof) as og:
                for r in range(0, s.height, 512):
                    w = Window(0, r, s.width, min(512, s.height - r))
                    db = s.read(BANDS, window=w)
                    o8.write(db_to_uint8(db), window=w)
                    ov.write(np.all(np.isfinite(db), axis=0).astype(np.uint8)[None], window=w)
                    gv = g.read(1, window=w)
                    gu = np.full(gv.shape, INVALID, np.uint8)
                    good = np.isfinite(gv) & np.isin(gv, [0, *self.classes])
                    gu[good] = gv[good].astype(np.uint8)
                    og.write(gu[None], window=w)
                for i, c in enumerate(CHANNELS, 1):
                    o8.set_band_description(i, c)
                o8.update_tags(source=f"{REPO_ID}@{REPO_REVISION}:{sar_name}", source_bands=str(BANDS))
        for k in out:
            tmp[k].replace(out[k])
        if not source and not keep_raw:
            found[sar_name].unlink()  # the float32 scene is 7-9 GB
        log.info("%s: scene prepared in %s", event, self.scene_dir(event))


def class_balance(patches, classes: dict[int, str]):
    """Share of patches with no positive pixel and share of positive pixels, per split."""
    import pandas as pd

    rows = []
    for split, sub in patches.groupby("split"):
        pos = sum(sub[f"n_{c}"] for c in classes.values())
        row = {"split": split, "patches": len(sub), "events": sub["event"].nunique(),
               "patches_without_positive_%": 100 * float((pos == 0).mean()),
               "positive_px_%": 100 * float(pos.sum() / max(sub["n_valid"].sum(), 1))}
        for c in classes.values():
            row[f"{c}_px_%"] = 100 * float(sub[f"n_{c}"].sum() / max(sub["n_valid"].sum(), 1))
        rows.append(row)
    return pd.DataFrame(rows).round(3)
