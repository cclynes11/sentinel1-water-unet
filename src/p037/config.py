"""Experiment configuration: one YAML file per experiment, every option documented here.

A config names a dataset, how the model is initialised and trained, how it is evaluated and how
the results are analysed. Stages read only the fields they need. Any field can be overridden on
the command line with ``--set key=value``.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    # ---------------------------------------------------------------- identity and locations
    name: str = "experiment"
    dataset: str = "urbansarfloods"  # key in p037.datasets.REGISTRY
    # Large files (caches, scenes, checkpoints). Default: $P037_DATA, else <repo>/data/p037.
    data_root: str = ""
    # Outputs of this experiment. Default: <data_root>/runs/<name>.
    out_dir: str = ""

    # ---------------------------------------------------------------- model input and target
    # Names of the cached channels fed to the 4 input channels of the U-Net, in order.
    # Change detection input (default): the labels are change-based (permanent water = 0).
    # Post-only (Wagner's layout): [post_VV, post_VH, post_VV, post_VH].
    input_channels: list[str] = field(default_factory=lambda: ["pre_VV", "pre_VH", "post_VV", "post_VH"])
    # Label values that count as the positive class when training.
    target_classes: list[int] = field(default_factory=lambda: [1, 2])
    # How a flood map is derived from the model output:
    #   direct            the model outputs P(flood) from `input_channels`
    #   post_and_not_pre  the model outputs P(water); flood = water(post) AND NOT water(pre)
    flood_rule: str = "direct"
    pre_channels: list[str] = field(default_factory=lambda: ["pre_VV", "pre_VH", "pre_VV", "pre_VH"])
    post_channels: list[str] = field(default_factory=lambda: ["post_VV", "post_VH", "post_VV", "post_VH"])

    # ---------------------------------------------------------------- model
    init: str = "random"  # random | pretrained (Wagner et al. 2024 weights from Hugging Face)
    dropout: float = 0.5
    bn_momentum: float = 0.01  # PyTorch convention; 0.01 = Keras momentum 0.99 (the original)
    # Checkpoint used by `evaluate`. Default: <out_dir>/best_<checkpoint_metric>.pt,
    # or the pretrained weights when init = pretrained and no checkpoint exists.
    weights: str = ""

    # ---------------------------------------------------------------- training (paper section 2.6)
    epochs: int = 200
    batch_size: int = 32
    lr: float = 1e-4
    flips: bool = True  # random horizontal and vertical flips
    patience: int = 30  # stop after this many epochs without improvement (0 = never)
    checkpoint_metric: str = "f1"  # f1 | accuracy (the paper kept the best-accuracy model)
    # official: the dataset's own train/validation lists.
    # events:   hold out every chip of `val_events` for validation, train on the other events.
    val_mode: str = "official"
    val_events: list[str] = field(default_factory=list)
    seed: int = 0
    num_workers: int = 4
    amp: bool = True  # mixed precision on GPU (ignored on CPU)
    max_hours: float = 0.0  # stop cleanly after this long; resubmit to resume (0 = no limit)
    max_train_patches: int = 0  # use only the first N training patches (tests; 0 = all)
    max_val_patches: int = 0

    # ---------------------------------------------------------------- evaluation
    eval_events: list[str] = field(default_factory=list)  # default: the dataset's test events
    threshold: float = 0.5
    tile: int = 256  # scoring tile (px)
    window: int = 1024  # prediction window core (px), a multiple of `tile`
    crop: int = 64  # context margin cropped from each side of a window
    mc_passes: int = 20
    save_probability: bool = True  # write the flood probability of each scene as an 8-bit GeoTIFF

    # ---------------------------------------------------------------- analysis
    # Table of raw proxy values (one row per training patch and per test tile).
    proxy_table: str = "analysis/proxy/outputs/proxy_values.csv.gz"
    reference_set: str = "b_usf_train"  # value of the `set` column that is the training distribution
    proxy_scale: str = "tile"  # tile (~4.7 km) | neigh (~24 km)
    proxies: list[str] = field(default_factory=lambda: [
        "slope_mean", "slope_p90", "elev_range", "hand_median", "hand_lt5_frac",
        "veg_frac", "tree_frac", "built_frac"])
    max_proxy_corr: float = 0.85  # drop a proxy correlated above this with one already kept
    score_class: str = "open"  # class whose errors are analysed: open | urban | any
    min_positive_px: int = 100  # tiles with fewer positive label pixels have no miss rate / IoU
    event_groups: dict[str, str] = field(default_factory=dict)  # scene -> event, for held-out tests
    n_bootstrap: int = 200

    # ------------------------------------------------------------------------------------ helpers
    def __post_init__(self):
        if not self.data_root:
            self.data_root = os.environ.get("P037_DATA", str(REPO_ROOT / "data" / "p037"))
        if not self.out_dir:
            self.out_dir = str(Path(self.data_root) / "runs" / self.name)
        if self.flood_rule not in ("direct", "post_and_not_pre"):
            raise ValueError(f"flood_rule must be direct or post_and_not_pre, got {self.flood_rule!r}")
        if self.init not in ("random", "pretrained"):
            raise ValueError(f"init must be random or pretrained, got {self.init!r}")
        if self.val_mode not in ("official", "events"):
            raise ValueError(f"val_mode must be official or events, got {self.val_mode!r}")
        if self.checkpoint_metric not in ("f1", "accuracy"):
            raise ValueError("checkpoint_metric must be f1 or accuracy")
        if self.window % self.tile:
            raise ValueError("window must be a multiple of tile")
        if (self.window + 2 * self.crop) % 32:
            raise ValueError("window + 2 * crop must be a multiple of 32 (U-Net depth)")

    @property
    def out(self) -> Path:
        p = Path(self.out_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def dataset_dir(self) -> Path:
        return Path(self.data_root) / self.dataset

    def resolve(self, path: str) -> Path:
        """Paths in a config are relative to the repository root unless absolute."""
        p = Path(path)
        return p if p.is_absolute() else REPO_ROOT / p

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _coerce(value: str, current):
    """Parse a ``--set key=value`` string to the type of the field's current value."""
    import yaml

    parsed = yaml.safe_load(value)
    if isinstance(current, bool):
        return bool(parsed)
    if isinstance(current, float) and isinstance(parsed, int):
        return float(parsed)
    if isinstance(current, str) and parsed is None:
        return ""
    if isinstance(current, str):
        return str(parsed)
    return parsed


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    """Read a YAML config; unknown keys are an error (a misspelt option must not pass silently)."""
    import yaml

    raw = {}
    if path:
        raw = yaml.safe_load(Path(path).read_text()) or {}
    known = {f.name: f for f in dataclasses.fields(Config)}
    unknown = sorted(set(raw) - set(known))
    if unknown:
        raise KeyError(f"unknown config keys in {path}: {unknown}")
    defaults = Config()
    for item in overrides or []:
        key, sep, value = item.partition("=")
        if not sep or key not in known:
            raise KeyError(f"--set expects key=value with a known key, got {item!r}")
        raw[key] = _coerce(value, getattr(defaults, key))
    return Config(**raw)


def save_config(cfg: Config, name: str = "config.yaml") -> Path:
    import yaml

    path = cfg.out / name
    path.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    return path
