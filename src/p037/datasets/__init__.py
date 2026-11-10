"""Dataset adapters. To add a dataset: subclass ``base.Dataset`` and register it here."""

from __future__ import annotations

from ..config import Config
from .amazon_wagner import AmazonWagner
from .base import Dataset
from .synthetic import Synthetic
from .urbansarfloods import UrbanSARFloods

REGISTRY: dict[str, type[Dataset]] = {d.key: d for d in (UrbanSARFloods, AmazonWagner, Synthetic)}


def get_dataset(cfg: Config) -> Dataset:
    if cfg.dataset not in REGISTRY:
        raise KeyError(f"unknown dataset {cfg.dataset!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[cfg.dataset](cfg.dataset_dir)
