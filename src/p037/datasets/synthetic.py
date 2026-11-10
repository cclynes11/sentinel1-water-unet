"""A small generated dataset in the same on-disk formats as the real ones.

It exists so that every stage can be run and tested on a laptop in minutes, with no download:
unit tests, the ``smoke`` config, and dry runs of the notebooks. It is NOT a physical simulation
and nothing measured on it means anything about real floods.

What it draws (in dB, then converted with the model's 8-bit scaling):
* land: speckled backscatter around -9 dB (VV) and -16 dB (VH);
* permanent water: dark in both dates (label 0, as in UrbanSARFloods, where labels mark change);
* open flood: dark in the post-event image only (label 1);
* urban flood: slightly brighter in the post-event image, small patches (label 2);
* a strip without SAR data (NaN) on one side of every scene.
"""

from __future__ import annotations

import logging

import numpy as np

from .base import INVALID, CacheMeta, Dataset, create_arrays, db_to_uint8, patch_counts

log = logging.getLogger("p037.synthetic")
CHANNELS = ["pre_VV", "pre_VH", "post_VV", "post_VH"]


def _blobs(rng, shape, scale: float, share: float) -> np.ndarray:
    """Smooth random mask covering about ``share`` of the image."""
    from scipy.ndimage import gaussian_filter

    f = gaussian_filter(rng.standard_normal(shape), scale)
    return f > np.quantile(f, 1 - share)


def draw(rng, shape: tuple[int, int], flood_share: float = 0.12):
    """Return (db (4, H, W) float32 with NaN where no data, label (H, W) uint8)."""
    h, w = shape
    water = _blobs(rng, shape, 10, 0.08)
    flood = _blobs(rng, shape, 14, flood_share) & ~water
    urban = _blobs(rng, shape, 5, 0.02) & ~water & ~flood
    base = {"VV": -9.0, "VH": -16.0}
    db = np.empty((4, h, w), np.float32)
    for i, name in enumerate(CHANNELS):
        date, pol = name.split("_")
        x = base[pol] + 1.5 * rng.standard_normal(shape)
        dark = water | (flood if date == "post" else False)
        x[dark] = base[pol] - 9.0 + 1.0 * rng.standard_normal(int(dark.sum()))
        if date == "post":
            x[urban] += 3.0
        db[i] = x
    label = np.zeros(shape, np.uint8)
    label[flood] = 1
    label[urban] = 2
    return db, label


class Synthetic(Dataset):
    key = "synthetic"
    channels = CHANNELS
    classes = {1: "open", 2: "urban"}
    target = "flood"
    patch = 64
    test_events = ["synth_A", "synth_B"]
    event_groups = {"synth_A": "A", "synth_B": "B"}

    def build_cache(self, source: str | None = None, n_train: int = 96, n_val: int = 32, seed: int = 0,
                    patch: int | None = None, **_) -> None:
        import pandas as pd

        P = patch or self.patch
        rng = np.random.default_rng(seed)
        rows = []
        for split, n in (("train", n_train), ("val", n_val)):
            images, labels = create_arrays(self.cache_dir, split, n, len(CHANNELS), P)
            for i in range(n):
                db, lab = draw(rng, (P, P), flood_share=float(rng.choice([0.0, 0.1, 0.3])))
                images[i], labels[i] = db_to_uint8(db), lab
                event = f"train_{i % 3}"
                rows.append({"patch_id": f"{event}_{split}_{i}", "split": split, "index": i, "event": event,
                             "lon": 10 + 0.01 * i, "lat": 5.0})
            images.flush()
            labels.flush()
            counts = patch_counts(labels, self.classes)
            for k, v in counts.items():
                for r, val in zip(rows[-n:], v, strict=True):
                    r[k] = int(val)
        pd.DataFrame(rows).to_csv(self.cache_dir / "patches.csv", index=False)
        meta = CacheMeta(dataset=self.key, channels=CHANNELS, patch=P, classes=self.classes, target=self.target,
                         source=f"generated, seed {seed}", notes="synthetic: for tests and dry runs only")
        (self.cache_dir / "meta.json").write_text(meta.to_json())
        log.info("synthetic cache: %d train + %d val patches of %d px in %s", n_train, n_val, P, self.cache_dir)

    def prepare_scene(self, event: str, source: str | None = None, shape: tuple[int, int] = (300, 420),
                      seed: int | None = None, **_) -> None:
        import rasterio
        from rasterio.transform import from_origin

        if self.has_scene(event):
            return
        seed = self.test_events.index(event) + 100 if seed is None else seed
        db, label = draw(np.random.default_rng(seed), shape)
        db[:, :, -20:] = np.nan  # a strip without SAR data
        valid = np.all(np.isfinite(db), axis=0)
        label[~valid] = INVALID
        out = self.scene_files(event)
        self.scene_dir(event).mkdir(parents=True, exist_ok=True)
        prof = dict(driver="GTiff", width=shape[1], height=shape[0], crs="EPSG:4326",
                    transform=from_origin(10.0 + seed, 5.0, 1 / 6000, 1 / 6000))
        with rasterio.open(out["image"], "w", count=4, dtype="uint8", **prof) as o:
            o.write(db_to_uint8(db))
            for i, c in enumerate(CHANNELS, 1):
                o.set_band_description(i, c)
        with rasterio.open(out["gt"], "w", count=1, dtype="uint8", nodata=INVALID, **prof) as o:
            o.write(label, 1)
        with rasterio.open(out["valid"], "w", count=1, dtype="uint8", **prof) as o:
            o.write(valid.astype(np.uint8), 1)

    def generate_proxies(self, samples, columns: list[str], seed: int = 0):
        """Made-up proxy values (standard normal) for the given samples: dry runs only.

        A real dataset gets its proxies from ``analysis/proxy/compute_proxies.py``. Having this
        method is what tells the proxies stage that it may fill the table by itself."""
        import pandas as pd

        rng = np.random.default_rng(seed)
        out = pd.DataFrame({"sample_id": samples["sample_id"].to_numpy()})
        for c in columns:
            out[c] = rng.normal(size=len(out))
        return out
