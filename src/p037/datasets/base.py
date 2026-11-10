"""The contract every dataset adapter fulfils, and the on-disk formats shared by all stages.

"Only the dataset changes": a dataset adapter converts its raw files into the two formats below.
Everything after the ``data`` stage (training, evaluation, analysis) reads only these formats.

Patch cache: ``<data_root>/<dataset>/cache/``
    meta.json           channels (names, in stored order), patch size, classes, target name
    patches.csv         one row per patch: patch_id, split, index, event, lon, lat, pixel counts
    images_<split>.npy  uint8 (N, C, P, P): 8-bit backscatter, (max(-49, min(1, dB)) + 50) * 5
    labels_<split>.npy  uint8 (N, P, P): class value, 255 = invalid (no label or no SAR data)

Test scenes: ``<data_root>/<dataset>/scenes/<event>/``
    image.tif   uint8, C bands, band descriptions = channel names
    gt.tif      uint8, class value, 255 = invalid
    valid.tif   uint8, 1 where every SAR band has data

``index`` in patches.csv is the row of the patch inside its split's ``.npy`` files. The ``.npy``
files are opened memory-mapped, so a patch is read from disk only when it is used.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

INVALID = 255
SPLITS = ("train", "val")


@dataclass
class CacheMeta:
    dataset: str
    channels: list[str]
    patch: int = 256
    classes: dict[int, str] = field(default_factory=lambda: {1: "open", 2: "urban"})
    target: str = "flood"  # prefix of the score columns, e.g. flood_open_f1
    source: str = ""
    notes: str = ""

    def to_json(self) -> str:
        d = dict(self.__dict__)
        d["classes"] = {str(k): v for k, v in self.classes.items()}
        return json.dumps(d, indent=1)

    @classmethod
    def read(cls, path: Path) -> CacheMeta:
        d = json.loads(Path(path).read_text())
        d["classes"] = {int(k): v for k, v in d["classes"].items()}
        return cls(**d)


class Dataset:
    """Base class of a dataset adapter. Subclasses implement ``build_cache`` and ``prepare_scene``."""

    key = ""
    channels: list[str] = []
    classes: dict[int, str] = {1: "open", 2: "urban"}
    target = "flood"
    patch = 256
    test_events: list[str] = []
    # scene -> event, when one event is covered by several scenes (used for held-out-event tests)
    event_groups: dict[str, str] = {}

    def __init__(self, root: Path):
        self.root = Path(root)

    # ------------------------------------------------------------------ locations
    @property
    def cache_dir(self) -> Path:
        return self.root / "cache"

    def scene_dir(self, event: str) -> Path:
        return self.root / "scenes" / event

    def scene_files(self, event: str) -> dict[str, Path]:
        d = self.scene_dir(event)
        return {"image": d / "image.tif", "gt": d / "gt.tif", "valid": d / "valid.tif"}

    # ------------------------------------------------------------------ to implement
    def build_cache(self, source: str | None = None, **kwargs) -> None:
        raise NotImplementedError

    def prepare_scene(self, event: str, source: str | None = None, **kwargs) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------------ shared readers
    def meta(self) -> CacheMeta:
        return CacheMeta.read(self.cache_dir / "meta.json")

    def patches(self):
        import pandas as pd

        return pd.read_csv(self.cache_dir / "patches.csv")

    def open_arrays(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        """Memory-mapped (images, labels) of one split."""
        return (np.load(self.cache_dir / f"images_{split}.npy", mmap_mode="r"),
                np.load(self.cache_dir / f"labels_{split}.npy", mmap_mode="r"))

    def has_cache(self) -> bool:
        return (self.cache_dir / "meta.json").exists() and (self.cache_dir / "patches.csv").exists()

    def has_scene(self, event: str) -> bool:
        return all(p.exists() for p in self.scene_files(event).values())


def create_arrays(cache_dir: Path, split: str, n: int, n_channels: int, patch: int):
    """Create the writable memory-mapped arrays of one split (images zero, labels invalid)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    images = np.lib.format.open_memmap(cache_dir / f"images_{split}.npy", mode="w+", dtype=np.uint8,
                                       shape=(n, n_channels, patch, patch))
    labels = np.lib.format.open_memmap(cache_dir / f"labels_{split}.npy", mode="w+", dtype=np.uint8,
                                       shape=(n, patch, patch))
    labels[:] = INVALID
    return images, labels


def patch_counts(labels: np.ndarray, classes: dict[int, str], chunk: int = 512):
    """Per-patch pixel counts: n_valid and n_<class name> for every class."""
    n = labels.shape[0]
    out = {"n_valid": np.zeros(n, np.int64)}
    out.update({f"n_{name}": np.zeros(n, np.int64) for name in classes.values()})
    for i in range(0, n, chunk):
        lab = np.asarray(labels[i:i + chunk]).reshape(min(chunk, n - i), -1)
        out["n_valid"][i:i + chunk] = (lab != INVALID).sum(1)
        for value, name in classes.items():
            out[f"n_{name}"][i:i + chunk] = (lab == value).sum(1)
    return out


def channel_indices(names: list[str], channels: list[str]) -> list[int]:
    """Positions of the requested channel names in the stored channel order."""
    missing = [n for n in names if n not in channels]
    if missing:
        raise KeyError(f"channels {missing} are not in this dataset (it has {channels})")
    return [channels.index(n) for n in names]


def db_to_uint8(db: np.ndarray) -> np.ndarray:
    """The model's own 8-bit scaling (NaN -> 250), from ``s1water.preprocess``."""
    from s1water.preprocess import db_to_uint8 as f

    return f(db)


def uint8_to_db(u: np.ndarray) -> np.ndarray:
    return np.asarray(u, np.float32) / 5.0 - 50.0
