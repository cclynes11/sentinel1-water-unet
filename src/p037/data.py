"""Reading the patch cache for training: split selection, the torch Dataset, the loaders."""

from __future__ import annotations

import logging

import numpy as np
import torch

from .config import Config
from .datasets import get_dataset
from .datasets.base import INVALID, Dataset, channel_indices

log = logging.getLogger("p037.data")


def select_patches(cfg: Config, ds: Dataset):
    """(train rows, validation rows) of patches.csv for this experiment.

    Patches without any valid pixel are dropped first; partly invalid ones are kept and their
    invalid pixels are masked from the loss and the scores.

    official: the dataset's own lists. In UrbanSARFloods these are random within each scene, so
              validation patches neighbour training patches: good enough to pick a checkpoint,
              not a measure of generalisation.
    events:   every patch of ``val_events`` (from both official lists) is validation; training
              uses the official training patches of all other events. Validation then measures
              transfer to unseen events, at the cost of training data.
    """
    p = ds.patches()
    empty = p["n_valid"] == 0
    if empty.any():  # e.g. swath edges: no SAR data, nothing to learn or score, and their fill value would enter batch norm
        log.info("dropping %d of %d patches without a single valid pixel (%s)", int(empty.sum()), len(p),
                 ", ".join(f"{s}: {100 * m:.1f} %" for s, m in empty.groupby(p["split"]).mean().items()))
        p = p[~empty]
    if cfg.val_mode == "official":
        train, val = p[p["split"] == "train"], p[p["split"] == "val"]
    else:
        unknown = sorted(set(cfg.val_events) - set(p["event"]))
        if unknown or not cfg.val_events:
            raise ValueError(f"val_mode=events needs val_events from {sorted(p['event'].unique())}; bad: {unknown}")
        held = p["event"].isin(cfg.val_events)
        train, val = p[(p["split"] == "train") & ~held], p[held]
    if cfg.max_train_patches:
        train = train.head(cfg.max_train_patches)
    if cfg.max_val_patches:
        val = val.head(cfg.max_val_patches)
    log.info("train: %d patches from %d events; validation (%s): %d patches from %d events",
             len(train), train["event"].nunique(), cfg.val_mode, len(val), val["event"].nunique())
    return train.reset_index(drop=True), val.reset_index(drop=True)


class PatchDataset(torch.utils.data.Dataset):
    """Patches as (x uint8 (4, P, P), target float (1, P, P), mask float (1, P, P)).

    x holds the cached channels named in ``cfg.input_channels``. target is 1 where the label is
    one of ``cfg.target_classes``. mask is 0 where the label is invalid (255): those pixels are
    left out of the loss and the scores. x stays uint8 here and is scaled to [0, 1] on the GPU.
    Nothing here is random: the flips are applied to whole batches in the training loop
    (``train.random_flips``), so they do not depend on the number of loader workers.
    """

    def __init__(self, cfg: Config, ds: Dataset, rows):
        self.ds = ds
        self.rows = rows[["split", "index"]].to_records(index=False)
        self.channels = channel_indices(cfg.input_channels, ds.meta().channels)
        self.target_classes = np.array(cfg.target_classes, np.uint8)
        self._arrays = {}  # opened lazily, once per worker process

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        split, index = self.rows[i]
        if split not in self._arrays:
            self._arrays[split] = self.ds.open_arrays(split)
        images, labels = self._arrays[split]
        x = np.asarray(images[index])[self.channels]
        lab = np.asarray(labels[index])
        y = np.isin(lab, self.target_classes).astype(np.float32)[None]
        m = (lab != INVALID).astype(np.float32)[None]
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(y), torch.from_numpy(m)


def make_loaders(cfg: Config, device: str):
    ds = get_dataset(cfg)
    train_rows, val_rows = select_patches(cfg, ds)
    kw = dict(num_workers=cfg.num_workers, pin_memory=device == "cuda", persistent_workers=cfg.num_workers > 0)
    train = torch.utils.data.DataLoader(PatchDataset(cfg, ds, train_rows),
                                        batch_size=cfg.batch_size, shuffle=True, drop_last=True, **kw)
    val = torch.utils.data.DataLoader(PatchDataset(cfg, ds, val_rows),
                                      batch_size=cfg.batch_size, shuffle=False, **kw)
    return train, val, train_rows, val_rows
