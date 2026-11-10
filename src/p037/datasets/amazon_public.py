"""Experiment A, public route: Sentinel-1 scenes of the Amazon scored against public water maps.

Experiment A runs the pretrained weights on the model's own region: (a) an in-distribution score,
(b) proxy distributions of the training region vs the test tiles, (c) error vs confidence and error
vs proxy scores on the test tiles.

Why this route. Wagner et al. (2024) release neither their 59,387 training / 6,057 validation
patches nor the Planet NICFI label masks, and the paper has no separate test set, so the reported
F1 of 0.930 cannot be recomputed by anyone but the authors. What can be done with public data is
what the paper itself does as an independent check: compare the model's water with a public
Landsat water map (the paper reports F1 0.708 against JRC Global Surface Water and 0.686 against
MapBiomas Water 2022, both at 30 m over the whole basin and two years). This adapter does that
comparison one scene at a time, which is what steps (b) and (c) need: per-tile errors.

What a score from this adapter is, and is not:
* It is agreement between a 10 m radar prediction and a 30 m optical water map made by a
  different method, on a different date unless a same-month label is used. Disagreement is not
  all model error: mixed pixels along banks, flooded forest that Landsat does not map as water,
  and water that moved between the two dates all count against the model.
* It is not the paper's 0.930 (in-distribution patches with the model's own label definition),
  and not the paper's 0.708 either (that compares two-year occurrence maps over the basin, with
  a binarisation the paper does not state). Expect the per-scene figure to sit in that region if
  the pipeline is right; treat a large departure as something to explain, not as a result.

Input: a folder (``--source``) holding ``scenes.yaml`` and the files it names:

    <event name>:
      image: <8-bit, 2-band (VV, VH) GeoTIFF>   # output of `s1water preprocess`, or the word
                                                #   example  (the repository's preprocessed
                                                #   Anavilhanas scene, downloaded from Hugging Face)
      labels: [<label raster or glob>, ...]     # any grid; warped to the image grid, nearest neighbour
      label_kind: gsw_monthly                   # how to read the label values, see LABEL_KINDS
      water_min: 50                             # optional: overrides the preset
      note: "JRC GSW monthly history, 2023-10"  # optional: free text, stored with the scene

Output: the same three scene files as every other dataset (``image.tif``, ``gt.tif``,
``valid.tif``) plus ``info.json`` with the label value counts, so the rest of the pipeline runs
unchanged. There is no patch cache: nothing is trained in Experiment A.

If the authors send their validation patches, they go into the patch-cache format of
``base.py`` (channels ["VV", "VH"], classes {1: "water"}) and are scored with
``p037 evaluate --patches val``. That is the only way to reach the 0.930.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .base import INVALID, Dataset

log = logging.getLogger("p037.amazon")
CHANNELS = ["VV", "VH"]
MANIFEST = "scenes.yaml"


@dataclass
class LabelRule:
    """How raster values of a public water map become water (1), not water (0) or invalid (255).

    Either exact value lists (``water`` / ``not_water``) or a threshold (``water_min``: water where
    the value is at least this, not water below it). ``nodata`` values and ``value > valid_max``
    are invalid. With exact lists, any other value in the raster is an error: an unexpected value
    usually means the file is not the product the rule was written for."""

    water: list[int] = field(default_factory=list)
    not_water: list[int] = field(default_factory=list)
    water_min: float | None = None
    nodata: list[int] = field(default_factory=list)
    valid_max: float | None = None

    def apply(self, values: np.ndarray, covered: np.ndarray) -> np.ndarray:
        out = np.full(values.shape, INVALID, np.uint8)
        ok = covered & ~np.isin(values, self.nodata)
        if self.valid_max is not None:
            ok &= values <= self.valid_max
        if self.water_min is not None:
            out[ok] = (values[ok] >= self.water_min).astype(np.uint8)
            return out
        is_water, is_dry = np.isin(values, self.water), np.isin(values, self.not_water)
        unknown = ok & ~is_water & ~is_dry
        if unknown.any():
            found = np.unique(values[unknown])[:10].tolist()
            raise ValueError(f"label values {found} are neither water {self.water}, not water {self.not_water} "
                             f"nor no data {self.nodata}: wrong label_kind for this file?")
        out[ok & is_dry] = 0
        out[ok & is_water] = 1
        return out


# Presets. Check a new product's value coding in its own documentation before trusting a preset:
# the adapter writes the value counts it found to info.json for exactly that purpose.
LABEL_KINDS: dict[str, LabelRule] = {
    # JRC Global Surface Water, Monthly History: 0 no data, 1 not water, 2 water (one file per month).
    # The same-month label is the closest public match in time to a Sentinel-1 scene.
    "gsw_monthly": LabelRule(water=[2], not_water=[1], nodata=[0]),
    # JRC Global Surface Water, Occurrence: share (%) of valid observations with water, 0-100.
    # Long-term, not dated: water_min = 50 means "water more often than not". Values above 100 are no data.
    "gsw_occurrence": LabelRule(water_min=50, valid_max=100),
    # Any 0 / 1 water mask (e.g. an annual water layer exported as a binary raster).
    "binary": LabelRule(water=[1], not_water=[0]),
}


def read_manifest(source: str | Path) -> dict:
    import yaml

    path = Path(source) / MANIFEST
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: Experiment A needs a {MANIFEST} listing its scenes "
                                f"(format: see p037/datasets/amazon_public.py and analysis/p037/configs/"
                                f"exp_a_scenes.example.yaml)")
    scenes = yaml.safe_load(path.read_text()) or {}
    for event, entry in scenes.items():
        missing = [k for k in ("image", "labels", "label_kind") if k not in entry]
        if missing:
            raise KeyError(f"{path}: scene {event!r} lacks {missing}")
        if entry["label_kind"] not in LABEL_KINDS:
            raise KeyError(f"{path}: scene {event!r} has label_kind {entry['label_kind']!r}; "
                           f"known: {sorted(LABEL_KINDS)}")
    return scenes


def rule_for(entry: dict) -> LabelRule:
    """The preset of ``label_kind``, with any of its fields overridden in the manifest entry."""
    import dataclasses

    overrides = {f.name: entry[f.name] for f in dataclasses.fields(LabelRule) if f.name in entry}
    return dataclasses.replace(LABEL_KINDS[entry["label_kind"]], **overrides)


def warp_labels(paths: list[Path], crs, transform, width: int, height: int, block: int = 2048):
    """Label values on the image grid (nearest neighbour) and where any label file covers it.

    Several files (e.g. two 10 degree tiles) are laid over each other in the order given; a pixel
    takes the first file that covers it. Nearest neighbour keeps the label's own values: a 30 m
    label cell is repeated over the 10 m pixels inside it, nothing is averaged."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT
    from rasterio.windows import Window

    values = np.zeros((height, width), np.int32)
    covered = np.zeros((height, width), bool)
    for path in paths:
        with rasterio.open(path) as src, WarpedVRT(src, crs=crs, transform=transform, width=width, height=height,
                                                   resampling=Resampling.nearest, add_alpha=True) as vrt:
            alpha_band = vrt.count  # add_alpha appends the coverage band
            for r in range(0, height, block):
                w = Window(0, r, width, min(block, height - r))
                alpha = vrt.read(alpha_band, window=w) > 0
                take = alpha & ~covered[r:r + w.height]
                if take.any():
                    values[r:r + w.height][take] = vrt.read(1, window=w)[take]
                    covered[r:r + w.height] |= take
    return values, covered


class AmazonPublic(Dataset):
    key = "amazon_public"
    channels = CHANNELS
    classes = {1: "water"}
    target = "water"
    patch = 256
    has_training_data = False

    @property
    def test_events(self) -> list[str]:
        """Every scene already prepared (Experiment A has no fixed test set)."""
        root = self.root / "scenes"
        return sorted(p.name for p in root.iterdir() if self.has_scene(p.name)) if root.exists() else []

    def list_events(self, source: str | None = None) -> list[str]:
        return list(read_manifest(source)) if source else self.test_events

    def build_cache(self, source: str | None = None, **kwargs) -> None:
        raise NotImplementedError(
            "Experiment A has no patch cache: the Wagner et al. (2024) patches and labels are not public, and "
            "nothing is trained. Prepare scenes instead (`p037 data --what scenes --source <folder>`). "
            "If the authors' patches arrive, write them in the cache format of p037/datasets/base.py.")

    def prepare_scene(self, event: str, source: str | None = None, min_labelled: float = 0.01, **_) -> None:
        """Write image.tif (VV, VH), gt.tif (1 water, 0 not water, 255 invalid), valid.tif and info.json."""
        import rasterio

        if self.has_scene(event):
            log.info("%s: scene already prepared", event)
            return
        if not source:
            raise ValueError("Experiment A scenes are not downloaded automatically: pass --source <folder with scenes.yaml>")
        root = Path(source)
        entry = read_manifest(root)[event]
        if entry["image"] == "example":
            from s1water.pipeline import download_example_scene

            image_path = Path(download_example_scene())
        else:
            image_path = root / entry["image"]
        label_paths = [p for pattern in entry["labels"] for p in sorted(root.glob(pattern))]
        if not label_paths:
            raise FileNotFoundError(f"{event}: no label file matches {entry['labels']} under {root}")
        rule = rule_for(entry)

        with rasterio.open(image_path) as src:
            if src.dtypes[0] != "uint8" or src.count < 2:
                raise ValueError(f"{image_path}: expected the 2-band 8-bit image written by `s1water preprocess`, "
                                 f"got {src.count} band(s) of {src.dtypes[0]}")
            names = [d or "" for d in src.descriptions[:2]]
            if all(names) and not (names[0].startswith("VV") and names[1].startswith("VH")):
                raise ValueError(f"{image_path}: bands are described as {names}, expected VV then VH")
            img = src.read([1, 2])
            crs, transform, width, height = src.crs, src.transform, src.width, src.height
        # the model's own rule for pixels without SAR data and for scene borders
        from s1water.postprocess import valid_data_mask

        valid = valid_data_mask(img, transform, crs)
        values, covered = warp_labels(label_paths, crs, transform, width, height)
        gt = rule.apply(values, covered)
        gt[~valid] = INVALID

        labelled = gt != INVALID
        share = float(labelled.sum() / max(valid.sum(), 1))
        if share < min_labelled:
            raise ValueError(f"{event}: only {100 * share:.2f} % of the scene's valid pixels have a label. "
                             f"Do the label files cover the scene? ({[p.name for p in label_paths]})")
        found, counts = np.unique(values[covered & valid], return_counts=True)
        info = {"event": event, "image": str(image_path), "labels": [str(p) for p in label_paths],
                "label_kind": entry["label_kind"], "rule": rule.__dict__, "note": entry.get("note", ""),
                "pixel_size": [abs(transform.a), abs(transform.e)], "crs": str(crs),
                "valid_px": int(valid.sum()), "labelled_share_of_valid": share,
                "water_share_of_labelled": float((gt == 1).sum() / max(labelled.sum(), 1)),
                "label_value_counts": {int(v): int(c) for v, c in zip(found, counts, strict=True)}}

        out = self.scene_files(event)
        self.scene_dir(event).mkdir(parents=True, exist_ok=True)
        prof = dict(driver="GTiff", width=width, height=height, crs=crs, transform=transform, compress="deflate",
                    tiled=True, blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER")
        tmp = {k: p.with_suffix(".tmp.tif") for k, p in out.items()}
        with rasterio.open(tmp["image"], "w", count=2, dtype="uint8", **prof) as o:
            o.write(img)
            for i, c in enumerate(CHANNELS, 1):
                o.set_band_description(i, c)
        with rasterio.open(tmp["gt"], "w", count=1, dtype="uint8", nodata=INVALID, **prof) as o:
            o.write(gt, 1)
        with rasterio.open(tmp["valid"], "w", count=1, dtype="uint8", **prof) as o:
            o.write(valid.astype(np.uint8), 1)
        (self.scene_dir(event) / "info.json").write_text(json.dumps(info, indent=1))
        for k in out:
            tmp[k].replace(out[k])
        log.info("%s: prepared (%.1f %% of valid pixels labelled, %.1f %% of those water; label values %s)",
                 event, 100 * share, 100 * info["water_share_of_labelled"], info["label_value_counts"])
