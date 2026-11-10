"""Experiment A slot: the Amazon data of Wagner et al. (2024). NOT IMPLEMENTED: the data is not public.

Experiment A runs the pretrained weights on the model's own Amazon data: (a) reproduce
F1 = 0.93, (b) proxy distributions of train vs test patches, (c) error vs confidence and
error vs proxy scores on the test tiles.

Status on 10 Nov 2026: the paper releases neither the 59,387 training / 6,057 validation patches
nor the Planet NICFI label masks, and it has no separate test set. The authors have been asked
for the validation patches, labels, locations and the split method. Until they answer, or a
fallback is agreed with the supervisor, there is nothing to convert.

What this adapter has to produce, so that the rest of the pipeline runs unchanged
(``configs/exp_a_amazon.yaml`` is already written against it):

    cache/meta.json     channels ["VV", "VH"], patch 256, classes {1: "water"}, target "water"
    cache/patches.csv   patch_id, split (train / val), index, event (e.g. the Sentinel-1 scene id),
                        lon, lat (patch centre, needed for the proxies), n_valid, n_water
    cache/images_<split>.npy   uint8 (N, 2, 256, 256), VV then VH, the model's 8-bit scaling
    cache/labels_<split>.npy   uint8 (N, 256, 256), 1 = water, 0 = not water, 255 = invalid

Three ways to fill it, in the order of the project plan:

1. Wagner's own patches (preferred). They are already 8-bit 256 px tiles: stack them into the
   arrays above and take lon / lat from their georeferencing. Without locations, proxies cannot
   be computed and only step (a) is possible.
2. Public comparison: Rio Negro scenes run through the repository's SNAP preprocessing, labelled
   with JRC Global Surface Water or MapBiomas 2022 (the paper reports F1 0.708 and 0.686 against
   them at 30 m). This reproduces a published number, but not the 0.93.
3. An independent Amazon test set: Sentinel-1 scenes + JRC monthly water history as labels.
"""

from __future__ import annotations

from .base import Dataset


class AmazonWagner(Dataset):
    key = "amazon_wagner"
    channels = ["VV", "VH"]
    classes = {1: "water"}
    target = "water"
    patch = 256

    def build_cache(self, source: str | None = None, **kwargs) -> None:
        raise NotImplementedError(
            "Experiment A has no data yet: the Wagner et al. (2024) patches and labels are not public. "
            "See the docstring of p037/datasets/amazon_wagner.py for the format to produce and the "
            "three options (authors' patches, GSW / MapBiomas comparison, independent test set).")

    def prepare_scene(self, event: str, source: str | None = None, **kwargs) -> None:
        raise NotImplementedError("Experiment A is evaluated on cached patches (`evaluate --patches val`), "
                                  "not on whole scenes, until scene-level labels exist.")
