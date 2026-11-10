"""Write the walkthrough notebooks (01 to 05) from the cell lists below.

The notebooks are generated so that their text and code are reviewed here, in one plain file.
Edit this file and run it again to change them:

    python analysis/p037/notebooks/build_notebooks.py                  # write the .ipynb files
    python analysis/p037/notebooks/build_notebooks.py --execute DIR    # also run them, executed copies go to DIR

The notebooks in this folder are kept without outputs. Running them with ``--execute`` is the
check that every cell still works (set P037_CONFIG to choose the experiment; default: dry-run data).

Your own working copies are yours: save a notebook under another name before changing it, or
this script will overwrite it the next time it runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent

SETUP = '''import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from p037 import plots                      # sets the figure style used in the report
from p037.config import REPO_ROOT, load_config
from p037.datasets import get_dataset

# The experiment this notebook works on. smoke.yaml is generated data: it runs anywhere in minutes
# and its numbers mean nothing. On BluePebble use analysis/p037/configs/exp_b_usf.yaml.
# (Setting the environment variable P037_CONFIG before starting Jupyter does the same for all notebooks.)
CONFIG = os.environ.get("P037_CONFIG", "analysis/p037/configs/smoke.yaml")
cfg = load_config(REPO_ROOT / CONFIG)
ds = get_dataset(cfg)
DRY_RUN = cfg.dataset == "synthetic"
print(f"experiment {cfg.name!r} on dataset {cfg.dataset!r}")
print("dataset files:", cfg.dataset_dir)
print("run folder:   ", cfg.out)'''

NOTEBOOKS: dict[str, list[tuple[str, str]]] = {}

# =========================================================================================== 01
NOTEBOOKS["01_data.ipynb"] = [
    ("md", '''# Stage 1: data

**Goal.** Turn a dataset's raw files into the two formats every later stage reads: a *patch cache*
for training and whole *test scenes* for evaluation. After this stage nothing depends on which
dataset is in use, which is how "only the dataset changes" is kept true.

**Command-line equivalent:** `p037 data --config <config> [--source <folder or archive>]`

**Code:** `src/p037/datasets/` (`base.py` for the formats, one module per dataset)'''),
    ("code", SETUP),
    ("md", '''## 1.1 Build the patch cache

For UrbanSARFloods this reads 6,471 chips of 512 px, converts the four intensity bands to the
model's 8-bit scale, cuts each chip into four 256 px patches and writes them to two large arrays.
It takes a while and 9 GB, so on BluePebble run it as a job (`sbatch data.sbatch <config>`) and
use this notebook to inspect the result. The cell below builds the cache here only for the dry
run, or if you point `P037_SOURCE` at the dataset on disk.'''),
    ("code", '''SOURCE = os.environ.get("P037_SOURCE")       # extracted dataset folder or the .tar.gz archive

if ds.has_cache():
    print("cache already built:", ds.cache_dir)
elif DRY_RUN or SOURCE:
    ds.build_cache(SOURCE)
else:
    print("No cache yet. Build it with the data job, or set P037_SOURCE and run this cell again.")'''),
    ("md", '''## 1.2 What is in the cache

`meta.json` names the stored channels. Configs ask for channels **by name**, never by position:
the UrbanSARFloods files carry no band names and do not follow the paper's order
(`analysis/band_identification.md`), so positions are exactly where a silent mistake would hide.'''),
    ("code", '''meta = ds.meta()
print("channels, in stored order:", meta.channels)
print("classes:", meta.classes, "| patch size:", meta.patch, "px")
print("this experiment feeds the model:", cfg.input_channels)

patches = ds.patches()
patches.head()'''),
    ("code", '''# patches per event and split
patches.pivot_table(index="event", columns="split", values="patch_id", aggfunc="count", fill_value=0)'''),
    ("md", '''## 1.3 Class balance

Flood is rare. This table is the reason the training stage keeps the checkpoint with the best
validation **F1** and not the best accuracy: a model that never predicts flood is already almost
always "right".'''),
    ("code", '''from p037.datasets.urbansarfloods import class_balance

class_balance(patches, meta.classes)'''),
    ("md", '''## 1.4 One patch, as the model sees it

The cache stores backscatter as 8-bit numbers: `(max(-49, min(1, dB)) + 50) * 5`. That is the
scaling the U-Net was designed for. `uint8_to_db` turns it back into decibels for plotting.

Open water is a smooth surface that reflects the radar pulse away from the sensor, so it is dark.
A flood therefore shows as a patch that is dark **after** the event and was not dark **before**.'''),
    ("code", '''from p037.datasets.base import INVALID, uint8_to_db

images, labels = ds.open_arrays("train")     # memory-mapped: nothing is loaded until it is used
train_rows = patches[patches["split"] == "train"]
flood_px = sum(train_rows[f"n_{c}"] for c in meta.classes.values())
row = train_rows.loc[flood_px.idxmax()]      # the training patch with the most flood
img, lab = np.asarray(images[row["index"]]), np.asarray(labels[row["index"]])

fig, axes = plt.subplots(1, len(meta.channels) + 1, figsize=(3 * (len(meta.channels) + 1), 3.3))
for ax, name, band in zip(axes, meta.channels, uint8_to_db(img)):
    im = ax.imshow(band, cmap="gray", vmin=-25, vmax=0)
    ax.set_title(f"{name} (dB)", loc="left")
shown = np.ma.masked_equal(lab, INVALID)
axes[-1].imshow(shown, cmap=plots.SEQ, vmin=0, vmax=max(meta.classes))
axes[-1].set_title("label " + str(meta.classes), loc="left")
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(visible=False)
fig.suptitle(row["patch_id"], x=0.01, ha="left")
fig.tight_layout()'''),
    ("md", '''## 1.5 Does the signal exist? Backscatter of flooded and dry pixels

A quick physical check on a sample of patches: after the event, flooded pixels should sit lower
(darker) than the rest; before the event they should not. If this plot ever shows the opposite,
the band order is wrong and nothing downstream can be trusted.'''),
    ("code", '''rng = np.random.default_rng(0)
with_flood = train_rows[flood_px > 0]
sample = with_flood.sample(min(150, len(with_flood)), random_state=0)["index"].to_numpy()
img_s = uint8_to_db(np.stack([np.asarray(images[i]) for i in np.sort(sample)]))    # (n, C, P, P)
lab_s = np.stack([np.asarray(labels[i]) for i in np.sort(sample)])
flooded, dry = np.isin(lab_s, list(meta.classes)), lab_s == 0

fig, axes = plt.subplots(1, len(meta.channels), figsize=(3 * len(meta.channels), 2.8), sharey=True)
bins = np.linspace(-35, 5, 60)
for ax, c, name in zip(axes, range(len(meta.channels)), meta.channels):
    ax.hist(img_s[:, c][dry], bins=bins, density=True, color=plots.MUTED, alpha=0.6, label="not flooded")
    ax.hist(img_s[:, c][flooded], bins=bins, density=True, color=plots.BLUE, alpha=0.6, label="flooded")
    ax.set_title(name, loc="left"); ax.set_xlabel("backscatter (dB)")
axes[0].set_ylabel("density"); axes[0].legend()
fig.tight_layout()
pd.DataFrame({"flooded median dB": [np.nanmedian(img_s[:, c][flooded]) for c in range(len(meta.channels))],
              "not flooded median dB": [np.nanmedian(img_s[:, c][dry]) for c in range(len(meta.channels))]},
             index=meta.channels).round(1)'''),
    ("md", '''## 1.6 Invalid pixels and the split used for training

Pixels without SAR data (swath edges) carry label 255. They are left out of the loss and of every
score. Patches without a single valid pixel are dropped; partly valid ones are kept.

`select_patches` applies the experiment's validation choice (`val_mode` in the config):
`official` uses the dataset's own lists, `events` holds out whole events.'''),
    ("code", '''from p037.data import select_patches

total_px = len(patches) * meta.patch ** 2
print(f"invalid pixels: {100 * (1 - patches['n_valid'].sum() / total_px):.1f} % of all cached pixels")
print(f"patches without any valid pixel: {(patches['n_valid'] == 0).sum()} of {len(patches)}")
train_sel, val_sel = select_patches(cfg, ds)
print(f"validation mode {cfg.val_mode!r}: {len(train_sel)} training patches, {len(val_sel)} validation patches")
print("validation events:", sorted(val_sel["event"].unique())[:8], "...")'''),
    ("md", '''## 1.7 Test scenes

The held-out scenes are kept whole, each with its label and a valid-data mask on the same grid.
Evaluation predicts them in large windows and scores 256 px tiles.

For UrbanSARFloods the four scenes are separate downloads (31 GB), so again: use the data job on
BluePebble, and this cell to look at the result.'''),
    ("code", '''import rasterio

SCENES = os.environ.get("P037_SCENES")       # folder holding the raw test scenes, if on disk
events = cfg.eval_events or ds.list_events(SCENES)
for event in events:
    if not ds.has_scene(event) and (DRY_RUN or SCENES):
        ds.prepare_scene(event, SCENES)
ready = [e for e in events if ds.has_scene(e)]
print("prepared scenes:", ready, "| still to prepare:", [e for e in events if e not in ready])

if ready:
    fig, axes = plt.subplots(len(ready), 2, figsize=(9, 3.4 * len(ready)), squeeze=False)
    for (a_img, a_lab), event in zip(axes, ready):
        f = ds.scene_files(event)
        with rasterio.open(f["image"]) as im, rasterio.open(f["gt"]) as gt:
            step = max(1, max(im.height, im.width) // 800)         # read a reduced overview, not the full scene
            shape = (im.height // step, im.width // step)
            last = im.count                                         # the last stored channel
            a_img.imshow(uint8_to_db(im.read(last, out_shape=shape)), cmap="gray", vmin=-25, vmax=0)
            a_img.set_title(f"{event}: {im.descriptions[last - 1]} (dB), {im.height} x {im.width} px", loc="left")
            g = gt.read(1, out_shape=shape)
            a_lab.imshow(np.ma.masked_equal(g, INVALID), cmap=plots.SEQ, vmin=0, vmax=max(meta.classes))
            a_lab.set_title("label (blank = invalid)", loc="left")
        for ax in (a_img, a_lab):
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(visible=False)
    fig.tight_layout()'''),
    ("md", '''## Check before moving on

- The channel names and the class balance match what you expect for the dataset.
- Flooded pixels are darker after the event (section 1.5).
- Every test scene is prepared.

Next: `02_train.ipynb`.'''),
]

# =========================================================================================== 02
NOTEBOOKS["02_train.ipynb"] = [
    ("md", '''# Stage 2: train

**Goal.** Train the U-Net of Wagner et al. (2024) on the patch cache with the paper's recipe, and
keep the best checkpoint.

**Command-line equivalent:** `p037 train --config <config>` (on BluePebble: `sbatch train.sbatch <config>`)

**Code:** `src/p037/model.py`, `src/p037/data.py`, `src/p037/train.py`

Experiments that use the published weights (`init: pretrained`) skip this stage.'''),
    ("code", SETUP),
    ("md", '''## 2.1 The model

`s1water.model.UNetS1Water`: a U-Net with 4 input channels and one output, the probability per
pixel. It applies the sigmoid itself, so its output is already a probability.

`init: random` builds it with fresh weights (Experiment B). `init: pretrained` would load the
published Amazon water weights instead.'''),
    ("code", '''import torch

from p037.model import build_model, pick_device

device = pick_device()
torch.manual_seed(cfg.seed)
model = build_model(cfg, init="random").to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"device: {device} | parameters: {n_params / 1e6:.1f} M | dropout: {cfg.dropout}")'''),
    ("md", '''## 2.2 One batch

Each item is three arrays: the image (8-bit, the four channels named in the config), the target
(1 where the label is one of `target_classes`) and a mask (0 where the label is invalid).
The image is divided by 255 on the device, as in the original code.'''),
    ("code", '''from p037.data import make_loaders

train_loader, val_loader, train_rows, val_rows = make_loaders(cfg, device)
x8, y, m = next(iter(train_loader))
print("image", tuple(x8.shape), x8.dtype, "| target", tuple(y.shape), "| mask", tuple(m.shape))
print(f"flood share of this batch: {100 * float((y * m).sum() / m.sum()):.2f} % of valid pixels")
x = x8.to(device).float() / 255
y, m = y.to(device), m.to(device)'''),
    ("md", '''## 2.3 The loss

Binary cross-entropy plus Dice, as in the paper (section 2.6), with two details that matter here:

- **Masked.** Invalid pixels contribute nothing.
- **Dice over the whole batch.** Most patches contain no flood, and the Dice score of an empty
  patch is decided by its smoothing constant alone, so per-patch Dice would mostly be noise.

An untrained model should give a cross-entropy near ln 2 = 0.69 (it knows nothing) plus a Dice
term between about 0.7 and 1 (little overlap yet; closer to 1 the rarer flood is in the batch).
A first loss of roughly 1.3 to 1.7 is therefore the expected starting point. A value far outside
that range before any training points to a problem in the data or the mask.'''),
    ("code", '''from p037.train import bce_dice_loss

model.train()
with torch.no_grad():
    print(f"loss of the untrained model on this batch: {float(bce_dice_loss(model(x), y, m)):.3f}")'''),
    ("md", '''## 2.4 Sanity check: can it memorise one batch?

Before spending GPU hours, check that the model, loss and optimiser are wired correctly: trained
repeatedly on a single batch, the loss must fall steadily. If it does not, something is broken
(labels misaligned, mask inverted, learning rate), and a full run would only hide it for longer.'''),
    ("code", '''STEPS = 40 if device == "cpu" else 200
opt = torch.optim.Adam(model.parameters(), lr=cfg.lr * (10 if DRY_RUN else 1))
history = []
for step in range(STEPS):
    loss = bce_dice_loss(model(x), y, m)
    opt.zero_grad(); loss.backward(); opt.step()
    history.append(float(loss.detach()))

fig, ax = plt.subplots(figsize=(6, 2.8))
ax.plot(history, color=plots.BLUE)
ax.set_xlabel("step on the same batch"); ax.set_ylabel("loss"); ax.set_title("Overfitting one batch", loc="left")
print(f"loss: {history[0]:.3f} -> {history[-1]:.3f}")'''),
    ("md", '''## 2.5 Full training

`train(cfg)` runs the real thing: every epoch it trains on all training patches, scores the
validation patches, appends a row to `log.csv` and writes `last.pt`. It keeps `best_f1.pt` and
`best_accuracy.pt`. Run again, it resumes from `last.pt`, and an epoch is the same whether the
run reached it in one job or after a resume.

On BluePebble submit it as a job. The cell below trains here only for the dry run.'''),
    ("code", '''from p037.train import train

if DRY_RUN:
    state = train(cfg, device)
    print(state)
else:
    print("Submit on BluePebble:  sbatch analysis/p037/slurm/train.sbatch", CONFIG)'''),
    ("md", '''## 2.6 Reading the training log

Two things to look for:

- **Loss.** Training and validation loss should both fall. Validation rising while training keeps
  falling is overfitting, and it is when early stopping (`patience`) ends the run.
- **F1 against accuracy.** Accuracy is high from the first epoch because most pixels are not
  flood. F1 is the curve that shows whether flood is being found.'''),
    ("code", '''log_path = cfg.out / "log.csv"
if log_path.exists():
    log = pd.read_csv(log_path)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3))
    a.plot(log["epoch"], log["train_loss"], color=plots.MUTED, label="training")
    a.plot(log["epoch"], log["val_loss"], color=plots.BLUE, label="validation")
    a.set_xlabel("epoch"); a.set_ylabel("loss"); a.legend(); a.set_title("Loss", loc="left")
    b.plot(log["epoch"], log["val_f1"], color=plots.BLUE, label="validation F1")
    b.plot(log["epoch"], log["val_accuracy"], color=plots.ORANGE, label="validation accuracy")
    best = log.loc[log["val_f1"].idxmax()] if log["val_f1"].notna().any() else None
    if best is not None:
        b.axvline(best["epoch"], color=plots.AXIS, lw=1)
    b.set_xlabel("epoch"); b.set_ylim(0, 1.02); b.legend(); b.set_title("Validation scores", loc="left")
    fig.tight_layout()
    display(log.tail(5))
    print(f"seconds per epoch (median): {log['seconds'].median():.0f}")
else:
    print("no log yet:", log_path)'''),
    ("md", '''## Check before moving on

- One batch can be memorised (section 2.4).
- Validation F1 rises above zero within the first epochs of the real run. If it stays at zero
  while accuracy is high, the model has collapsed to "no flood anywhere".
- Note which validation set the F1 refers to. With the official split it comes from patches next
  to training patches, so it is an upper bound, not a measure of how the model does elsewhere.

Next: `03_evaluate.ipynb`.'''),
]

# =========================================================================================== 03
NOTEBOOKS["03_evaluate.ipynb"] = [
    ("md", '''# Stage 3: evaluate

**Goal.** Predict the held-out scenes and produce one row per 256 px tile: how wrong the model
was there, and how confident it was. That table (`tiles.csv`) is the input of the analysis.

**Command-line equivalent:** `p037 evaluate --config <config>`

**Code:** `src/p037/evaluate.py`, `src/p037/metrics.py`'''),
    ("code", SETUP),
    ("md", '''## 3.1 Load the model

A trained experiment uses its own best checkpoint (`best_f1.pt`). An experiment with
`init: pretrained` and no checkpoint uses the published weights.'''),
    ("code", '''import rasterio
import torch
from rasterio.windows import Window

from p037.datasets.base import INVALID, uint8_to_db
from p037.evaluate import Predictor
from p037.model import load_for_eval, pick_device

device = pick_device()
model, origin = load_for_eval(cfg, device)
meta = ds.meta()
predictor = Predictor(cfg, model, meta.channels, device)
print("weights:", origin)
print("flood rule:", cfg.flood_rule)'''),
    ("md", '''## 3.2 One window, step by step

A scene is far too large for one forward pass, so it is predicted in windows. A pixel at the edge
of a window is predicted with less surrounding context than one in the middle, so each window is
read with a margin (`crop`) on every side and the margin's predictions are thrown away.

**Confidence.** The prediction that is scored comes from one pass with dropout switched off.
Then the same window is predicted `mc_passes` more times with dropout **on**: each pass removes a
different random part of the network. Where the passes disagree, the model is unsure. The variance
across passes is the per-pixel uncertainty (MC dropout).'''),
    ("code", '''events = [e for e in (cfg.eval_events or ds.test_events) if ds.has_scene(e)]
event = events[0]
f = ds.scene_files(event)
W, C = cfg.window, cfg.crop

with rasterio.open(f["image"]) as im, rasterio.open(f["gt"]) as gt, rasterio.open(f["valid"]) as va:
    # pick the window of the tile grid that holds the most labelled flood
    step = max(1, W // 64)
    small = gt.read(1, out_shape=(max(1, gt.height // step), max(1, gt.width // step)))
    rows, cols = -(-gt.height // W), -(-gt.width // W)
    counts = np.zeros((rows, cols))
    for r in range(rows):
        for c in range(cols):
            block = small[r * W // step:(r + 1) * W // step, c * W // step:(c + 1) * W // step]
            counts[r, c] = np.isin(block, list(meta.classes)).sum()
    r, c = np.unravel_index(counts.argmax(), counts.shape)
    core = Window(c * W, r * W, W, W)
    with_margin = Window(c * W - C, r * W - C, W + 2 * C, W + 2 * C)
    img = im.read(window=with_margin, boundless=True, fill_value=250)      # 250 = 0 dB, the model's padding value
    label = gt.read(1, window=core, boundless=True, fill_value=INVALID)
    valid = va.read(1, window=core, boundless=True, fill_value=0).astype(bool)

out = predictor(img[None], crop=C)                      # arrays of the core only: the margin is cropped
print("input with margin:", img.shape, "-> prediction of the core:", out["prob"][0].shape)'''),
    ("code", '''post = meta.channels.index(cfg.post_channels[0]) if cfg.post_channels[0] in meta.channels else 0
panels = [("image, " + meta.channels[post] + " (dB)", uint8_to_db(img[post, C:C + W, C:C + W]), "gray", (-25, 0)),
          ("label", np.ma.masked_equal(label, INVALID), plots.SEQ, (0, max(meta.classes))),
          ("probability of flood", out["prob"][0], plots.SEQ, (0, 1)),
          (f"prediction (threshold {cfg.threshold})", out["pred"][0], plots.SEQ, (0, 1))]
if "mc_var" in out:
    panels.append(("MC-dropout variance", out["mc_var"][0], plots.SEQ, (0, np.percentile(out["mc_var"][0], 99) + 1e-9)))
fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.5))
for ax, (title, data, cmap, (lo, hi)) in zip(axes, panels):
    ax.imshow(data, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
    ax.set_title(title, loc="left"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(visible=False)
fig.suptitle(f"{event}, window row {r}, column {c}", x=0.01, ha="left")
fig.tight_layout()'''),
    ("md", '''## 3.3 From pixels to a tile's score

Each tile gets four counts against the label: true positives (flood found), false positives
(flood claimed, none there), false negatives (flood missed) and true negatives. Everything else,
F1, precision, recall, follows from the counts.

Scoring is **one-vs-rest per class**: when open flood is scored, urban-flood pixels are left out
rather than counted as false alarms, because the model has a single flood output.'''),
    ("code", '''from p037.metrics import class_sets, confusion, scores

rows = []
for name, (positive, exclude) in class_sets(meta.classes).items():
    tp, fp, fn, tn = confusion(out["pred"][0], label, valid, positive, exclude)
    rows.append({"class": name, "tp": tp, "fp": fp, "fn": fn, "tn": tn, **scores(tp, fp, fn, tn)})
pd.DataFrame(rows).set_index("class").round(3)'''),
    ("md", '''## 3.4 All scenes

`evaluate_scenes` does the above for every window of every test scene and writes `tiles.csv`.
Scenes already scored are reused. With 1 + 20 passes per window this needs a GPU for real scenes,
so on BluePebble submit `evaluate.sbatch`; the cell runs here for the dry run or when a GPU is present.'''),
    ("code", '''from p037.evaluate import evaluate_scenes

tiles_path = cfg.out / "tiles.csv"
if DRY_RUN or device == "cuda":
    tiles_path = evaluate_scenes(cfg, device)
elif not tiles_path.exists():
    print("Submit on BluePebble:  sbatch analysis/p037/slurm/evaluate.sbatch", CONFIG)

if tiles_path.exists():
    tiles = pd.read_csv(tiles_path)
    print(len(tiles), "tiles;", len(tiles.columns), "columns")
    display(pd.read_csv(cfg.out / "tiles_summary.csv").round(3))'''),
    ("md", '''`tiles_summary.csv` pools the pixel counts of all tiles of a scene before computing the scores.
Those are the numbers to quote for a scene. An average of per-tile F1 is a different quantity and
is dominated by tiles with very little flood.

The columns of `tiles.csv`:

| Columns | Meaning |
|---|---|
| `tile_id`, `event`, `tile_row`, `tile_col`, `lon`, `lat` | which tile, and where its centre is |
| `n_valid`, `n_gt_<class>` | valid pixels and labelled pixels per class |
| `<target>_<class>_tp/fp/fn/tn` and `_f1/_iou/_precision/_recall` | counts and scores, per class and for `any` |
| `frac_<target>`, `mean_prob_<target>` | share of the tile predicted as flood, mean probability |
| `mc_var_<target>`, `mean_entropy_<target>` | MC-dropout uncertainty, averaged over the tile |'''),
    ("md", '''## 3.5 Where are the errors, and where is the model unsure?

One cell per tile, laid out as in the scene. If uncertainty were a perfect guide to error the two
maps would look the same. Where they differ, the model is wrong without knowing it, which is the
case the landscape proxies are meant to help with.'''),
    ("code", '''if tiles_path.exists():
    pre = f"{meta.target}_{cfg.score_class}"
    n = tiles[[f"{pre}_{k}" for k in ("tp", "fp", "fn", "tn")]].sum(axis=1)
    tiles["err_rate"] = (tiles[f"{pre}_fp"] + tiles[f"{pre}_fn"]) / n.where(n > 0)
    tiles["log_mc_var"] = np.log10(tiles[f"mc_var_{meta.target}"] + 1e-12)
    for event in sorted(tiles["event"].unique()):
        d = tiles[tiles["event"] == event]
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        for ax, col, title in zip(axes, ("err_rate", "log_mc_var"), ("error rate", "log10 MC-dropout variance")):
            data = d.pivot(index="tile_row", columns="tile_col", values=col).to_numpy()   # the scene's tile grid
            lo, hi = np.nanpercentile(data, [2, 98])
            ax.imshow(data, cmap=plots.SEQ, vmin=lo, vmax=max(hi, lo + 1e-9), interpolation="nearest")
            ax.set_title(f"{plots.nice_event(event)}: {title}", loc="left")
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(visible=False)
        fig.tight_layout()'''),
    ("md", '''## Check before moving on

- The pooled scores per scene are in the range you expect, and the tile count matches the scene
  size (rows x columns of 256 px tiles).
- The probability map of section 3.2 looks like the label, not like noise or like the raw image.

Next: `04_proxies.ipynb`.'''),
]

# =========================================================================================== 04
NOTEBOOKS["04_proxies.ipynb"] = [
    ("md", '''# Stage 4: proxies

**Goal.** Give every training patch and every test tile its landscape description, then express
each test tile as a **position in the training distribution**. That position is the tile's
per-proxy out-of-distribution score.

**Command-line equivalent:** `p037 proxies --config <config>`

**Code:** `src/p037/proxies.py` (coverage), `src/p037/ood.py` (placement). The proxy values
themselves are computed by `analysis/proxy/compute_proxies.py` (PR #3).'''),
    ("code", SETUP),
    ("md", '''## 4.1 Is every sample covered?

The proxy table has one row per location: training patches (the reference set) and test tiles.
`check_coverage` compares it with what this experiment needs. For anything missing it writes the
list with coordinates and prints the command that computes the values (that step downloads from
public servers, so it needs internet). For the dry run the values are simply made up.'''),
    ("code", '''from p037.proxies import check_coverage

try:
    coverage = check_coverage(cfg)
    print(f"{coverage['covered']} of {coverage['needed']} samples have proxy values")
    display(coverage["by_set"])
except FileNotFoundError as error:      # e.g. the zero-shot config: no cache and no tiles.csv in its run folder
    print("coverage not checked:", error)'''),
    ("code", '''proxy = pd.read_csv(cfg.resolve(cfg.proxy_table))
columns = {p: f"{p}__{cfg.proxy_scale}" for p in cfg.proxies}      # proxy name -> its column at the chosen scale
ref = proxy[proxy["set"] == cfg.reference_set]                      # the training distribution
test = proxy[proxy["set"] == "test"]
print(f"reference set {cfg.reference_set!r}: {len(ref)} rows | test tiles: {len(test)} | scale: {cfg.proxy_scale}")
proxy[["sample_id", "set", *columns.values()]].groupby("set").describe().T.round(3).head(16)'''),
    ("md", '''The proxies, and what each is meant to capture:

| Proxy | Source | Why it could matter for a SAR flood model |
|---|---|---|
| `slope_mean`, `slope_p90`, `elev_range` | Copernicus DEM GLO-30 | radar shadow and layover in relief; water does not pool on slopes |
| `hand_median`, `hand_lt5_frac` | GLO-30 HAND (height above nearest drainage) | how much of the tile can flood at all |
| `veg_frac`, `tree_frac` | ESA WorldCover 2021 | vegetation hides water or changes its radar signature |
| `built_frac` | GHSL | buildings brighten flooded streets (double bounce) |

WSF2019 is deliberately not used: UrbanSARFloods defines its urban-flood labels with it.'''),
    ("md", '''## 4.2 Training distribution against each test scene

The raw comparison: for each proxy, where do a scene's tiles fall relative to what the model was
trained on? A scene whose histogram sits where the grey one is empty is out of distribution on
that proxy.'''),
    ("code", '''events = sorted(test["event"].dropna().unique())
shown = list(columns)[:4]
fig, axes = plt.subplots(len(shown), len(events), figsize=(3 * len(events), 1.9 * len(shown)), squeeze=False, sharex="row")
for i, p in enumerate(shown):
    lo, hi = np.nanpercentile(proxy[columns[p]], [1, 99])
    bins = np.linspace(lo, hi if hi > lo else lo + 1, 30)
    for j, event in enumerate(events):
        ax = axes[i, j]
        ax.hist(ref[columns[p]].dropna(), bins=bins, density=True, color=plots.MUTED, alpha=0.55, label="training")
        ax.hist(test.loc[test["event"] == event, columns[p]].dropna(), bins=bins, density=True, color=plots.BLUE,
                alpha=0.6, label="test tiles")
        ax.set_yticks([])
        if i == 0:
            ax.set_title(plots.nice_event(event), loc="left")
        if j == 0:
            ax.set_ylabel(plots.nice_proxy(p), rotation=0, ha="right", va="center")
axes[0, 0].legend(fontsize=7)
fig.tight_layout()'''),
    ("md", '''## 4.3 From a raw value to a score

Three numbers per tile and proxy, all from the training distribution:

- **percentile** (0 to 100): the share of training samples with a lower value.
- **position** = (percentile - 50) / 50, from -1 to +1. Signed: "more built-up than training".
- **distance** = |position|, from 0 to 1. Unsigned: "unlike training, on either side".
  This is the out-of-distribution score proper.

Why not sigma (z-score), as first proposed? Several proxies are mostly zero with a long tail
(built-up share, tree cover in dry regions). Their training standard deviation is tiny, so a
z-score explodes for ordinary tiles. The percentile has no such problem. Both are computed, so
the two can be compared in the report.'''),
    ("code", '''from p037 import ood

placed = ood.place(test.rename(columns={"sample_id": "tile_id"}).copy(), ref, columns)
p = list(columns)[0]
placed[["tile_id", "event", columns[p], f"pct__{p}", f"pos__{p}", f"dist__{p}", f"z__{p}"]].head(8).round(3)'''),
    ("code", '''# share of each scene's tiles outside the central 90 % of the training data, per proxy
outside = pd.DataFrame({plots.nice_proxy(p): placed.groupby("event")[f"pct__{p}"].apply(
    lambda v: 100 * float(((v < 5) | (v > 95)).mean())) for p in columns}).round(0)
outside.index = [plots.nice_event(e) for e in outside.index]
print("percent of tiles outside the 5th to 95th percentile of the training data (10 % expected by chance)")
outside'''),
    ("md", '''## 4.4 Proxies that say the same thing

Mean slope, the 90th percentile of slope and elevation range all measure relief. Put together in
one regression, their weights would be unstable and uninterpretable. So before any regression,
a proxy whose rank correlation with one already kept exceeds `max_proxy_corr` (in the training
data) is dropped. The order of `proxies` in the config decides which of a pair is kept.'''),
    ("code", '''kept, dropped = ood.prune_collinear(ref, columns, cfg.max_proxy_corr)
print("kept:   ", kept)
print("dropped:", {p: f"|rho| {abs(rho):.2f} with {twin}" for p, (twin, rho) in dropped.items()})

corr = ref[list(columns.values())].corr(method="spearman")
fig, ax = plt.subplots(figsize=(5.2, 4.4))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
names = [plots.nice_proxy(p) for p in columns]
ax.set_xticks(range(len(names)), names, rotation=45, ha="right"); ax.set_yticks(range(len(names)), names)
ax.grid(visible=False)
for i in range(len(names)):
    for j in range(len(names)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7,
                color="white" if abs(corr.iloc[i, j]) > 0.6 else plots.INK)
ax.set_title("Rank correlation between proxies, training data", loc="left")
fig.colorbar(im, ax=ax, fraction=0.046)
fig.tight_layout()'''),
    ("md", '''## Things to keep in mind when writing this up

- The reference set must be the data the model under test was **trained on**. For the pretrained
  Amazon model that is `a_amazon` (600 sampled points, an approximation); for Experiment B it is
  `b_usf_train`, narrowed automatically to the patches the run actually used.
- Proxies are measured in a fixed box of about 4.7 km around the tile centre (`proxy_scale: tile`)
  or about 24 km (`neigh`). Which scale is meaningful differs by proxy.
- The land cover and building layers date from 2018 to 2021, not the year of each flood.

Next: `05_analyze.ipynb`.'''),
]

# =========================================================================================== 05
SETUP_05 = SETUP.replace(
    'CONFIG = os.environ.get("P037_CONFIG", "analysis/p037/configs/smoke.yaml")',
    '''# Default: the one real result that exists so far, the zero-shot run of PR #4 (its per-tile table is in
# the repository, so this notebook runs on a laptop). For your own run, set CONFIG to its config and
# TILES to None, which means <run folder>/tiles.csv.
CONFIG = os.environ.get("P037_CONFIG", "analysis/p037/configs/zeroshot_usf.yaml")
TILES = os.environ.get("P037_TILES", "analysis/usf_results.csv" if "zeroshot" in CONFIG else "") or None''').replace(
    '''# The experiment this notebook works on. smoke.yaml is generated data: it runs anywhere in minutes
# and its numbers mean nothing. On BluePebble use analysis/p037/configs/exp_b_usf.yaml.
# (Setting the environment variable P037_CONFIG before starting Jupyter does the same for all notebooks.)
''', "").replace(
    'print("run folder:   ", cfg.out)',
    '''tiles_csv = REPO_ROOT / TILES if TILES else cfg.out / "tiles.csv"
print("per-tile results:", tiles_csv)
print("training reference:", cfg.reference_set)''')

NOTEBOOKS["05_analyze.ipynb"] = [
    ("md", '''# Stage 5: analyze

**Goal.** Answer the project's questions from two tables, the per-tile results and the proxies:

1. Do landscape proxies predict where the model is wrong? (RQ1)
2. Do they add anything to the model's own uncertainty? (RQ2)

**Command-line equivalent:** `p037 analyze --config <config> [--tiles <results table>]`

**Code:** `src/p037/analyze.py` (the stage), `src/p037/ood.py` (the numerics), `src/p037/plots.py`

This notebook walks through the steps one at a time, then runs the stage itself, which writes the
same things as tables, figures and `report.md`.'''),
    ("code", SETUP_05),
    ("md", '''## 5.1 Join results and proxies, define "error"

`prepare` joins the two tables on `tile_id`, places every tile in the training distribution
(notebook 04) and adds three error measures:

| Measure | Definition | Defined for |
|---|---|---|
| `err_rate` | wrong pixels / scored pixels | every tile |
| `f1_loss` | 1 - F1 | tiles with at least `min_positive_px` flood pixels in the label |
| `miss_rate` | missed flood pixels / flood pixels | the same tiles |

Why three: F1 is undefined on a tile without flood, and most tiles have none. `err_rate` covers
every tile but is small wherever there is little flood to get wrong, so it partly measures *how
much flood a tile has*. That confound is handled explicitly below.'''),
    ("code", '''from p037 import ood
from p037.analyze import UNCERTAINTY, prepare

df, ref, columns, info = prepare(cfg, tiles_csv)
print(info)
print(f"flooded tiles (>= {cfg.min_positive_px} flood px): {int(df['flooded'].sum())} of {len(df)}")
df[["tile_id", "event", "err_rate", "f1_loss", "miss_rate", "gt_frac", "log_mc_var"]].describe().round(3)'''),
    ("md", '''## 5.2 Does the model's own uncertainty track its error?

Rank correlation (Spearman) between tile error rate and MC-dropout variance, per scene.

The second column removes what the amount of flood in the tile explains. A tile with more flood
has more pixels to get wrong *and* more uncertain boundary pixels, which produces a correlation
by itself. The partial correlation asks what is left once that is accounted for.'''),
    ("code", '''rows = []
for event, g in [*df.groupby("event"), ("all", df)]:
    rows.append({"event": event, "tiles": len(g),
                 "error ~ uncertainty": ood.spearman(g["err_rate"], g["log_mc_var"]),
                 "same, flood share removed": ood.partial_spearman(g["err_rate"], g["log_mc_var"], g["gt_frac"])})
pd.DataFrame(rows).set_index("event").round(2)'''),
    ("code", '''events = sorted(df["event"].unique())
fig, axes = plt.subplots(1, len(events), figsize=(3.1 * len(events), 3), sharex=True, sharey=True, squeeze=False)
for ax, event in zip(axes[0], events):
    g = df[df["event"] == event]
    ax.scatter(g["log_mc_var"], g["err_rate"], s=4, alpha=0.3, color=plots.BLUE, linewidths=0)
    ax.set_title(plots.nice_event(event), loc="left"); ax.set_xlabel("log10 MC-dropout variance")
axes[0, 0].set_ylabel("tile error rate")
fig.tight_layout()'''),
    ("md", '''## 5.3 Does error depend on where a tile sits in the training distribution?

One proxy at a time. `position` is signed (is "more of this" worse?), `distance` is unsigned (is
"unlike training" worse?). Values near zero mean the proxy alone says little about error.'''),
    ("code", '''rows = []
for p in columns:
    rows.append({"proxy": plots.nice_proxy(p),
                 "error ~ position": ood.spearman(df["err_rate"], df[f"pct__{p}"]),
                 "error ~ position, flood share removed": ood.partial_spearman(df["err_rate"], df[f"pct__{p}"], df["gt_frac"]),
                 "error ~ distance": ood.spearman(df["err_rate"], df[f"dist__{p}"])})
pd.DataFrame(rows).set_index("proxy").round(2)'''),
    ("md", '''## 5.4 The weighted OOD score

The supervisor's method: regress per-tile error on the per-proxy scores. The coefficients are
importance weights, and the weighted sum is a tile's OOD score.

Features are standardised, so a weight is the change in error rate for one training standard
deviation of that proxy's score, and weights are comparable between proxies.

The intervals come from refitting on resampled **blocks** of neighbouring tiles. Neighbouring
tiles share a river or a city and are not independent, so resampling single tiles would make the
weights look more certain than they are. A weight whose interval spans zero is not established.'''),
    ("code", '''kept, dropped = ood.prune_collinear(ref, columns, cfg.max_proxy_corr)
dist_cols = [f"dist__{p}" for p in kept]
d = df.dropna(subset=["err_rate", *dist_cols]).reset_index(drop=True)
X, y = d[dist_cols].to_numpy(float), d["err_rate"].to_numpy(float)

model = ood.Linear().fit(X, y)
boot = ood.bootstrap_weights(X, y, ood.spatial_blocks(d), cfg.n_bootstrap, cfg.seed)
lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
weights = pd.DataFrame({"weight": model.weights, "lo": lo, "hi": hi}, index=[plots.nice_proxy(p) for p in kept])
print(f"dropped as collinear: {list(dropped)}")
print(f"in-sample R2 {ood.r2(y, model.predict(X)):.3f}, rank correlation {ood.spearman(model.predict(X), y):.2f}")

fig, ax = plt.subplots(figsize=(6, 0.45 * len(kept) + 1.2))
ypos = np.arange(len(kept))[::-1]
ax.hlines(ypos, weights["lo"], weights["hi"], color=plots.BLUE, lw=2)
ax.plot(weights["weight"], ypos, "o", color=plots.BLUE)
ax.axvline(0, color=plots.AXIS, lw=1)
ax.set_yticks(ypos, weights.index); ax.set_xlabel("weight: change in tile error rate per training SD of distance")
ax.set_title("Weights of the OOD score (95 % block-bootstrap interval)", loc="left")
fig.tight_layout()
weights.round(4)'''),
    ("md", '''The fit above is **in-sample**: the same tiles choose the weights and judge them. It shows which
proxies carry weight, and says nothing about whether the score would work on a new flood.'''),
    ("md", '''## 5.5 The honest test: an event the predictor has never seen

Fit on all events but one, predict the error of every tile of the one left out, repeat for each
event. The score is the rank correlation between predicted and observed error on the unseen
event, because a ranking is what a reviewer deciding where to look needs.

Three predictors are compared:

- **uncertainty**: MC-dropout variance, entropy, predicted flood share. Needs only the model.
- **proxies**: position and distance on the kept proxies. Needs only a map of the area.
- **uncertainty + proxies**: both.

RQ1 is whether *proxies* is clearly above zero. RQ2 is whether *both* beats *uncertainty*.'''),
    ("code", '''pos_cols = [f"pos__{p}" for p in kept]
feature_sets = {"uncertainty": UNCERTAINTY, "proxies": pos_cols + dist_cols,
                "uncertainty+proxies": UNCERTAINTY + pos_cols + dist_cols}
if df["event_group"].nunique() >= 2:
    held, preds = ood.held_out_event_test(df, feature_sets, "err_rate")
    table = held.pivot(index="held_out", columns="model", values="spearman")[list(feature_sets)]
    display(table.round(2))
    fig, ax = plt.subplots(figsize=(6.5, 0.8 * len(table) + 1.3))
    for k, (name, (colour, label)) in enumerate(plots.MODEL_STYLE.items()):
        ax.plot(table[name], np.arange(len(table)) + (k - 1) * 0.2, "o", color=colour, label=label)
    ax.axvline(0, color=plots.AXIS, lw=1)
    ax.set_yticks(range(len(table)), [plots.nice_event(e) for e in table.index]); ax.invert_yaxis()
    ax.set_xlabel("rank correlation of predicted and observed tile error on the held-out event")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3)
    ax.set_title("Predicting tile error on an event never seen by the predictor", loc="left")
else:
    print("only one event: the held-out test needs at least two")'''),
    ("md", '''How to read it, and its limits:

- With three or four events, each number comes from a fit on the other two or three. Treat a
  single value as an indication. A pattern that repeats across events and across experiments is
  the evidence.
- A scene where **nothing** predicts the error is itself a finding: its failure has a cause that
  neither the model nor these proxies see.'''),
    ("md", '''## 5.6 The whole stage in one call

`run_analysis` computes everything above (plus the triage curves and per-scene maps), and writes
the tables, the figures and `report.md`. This is what `p037 analyze` runs.'''),
    ("code", '''from IPython.display import Image, Markdown

from p037.analyze import run_analysis

out_dir = cfg.out / "analysis_notebook"
report = run_analysis(cfg, tiles_csv, out_dir)
print("written to", out_dir)
print("tables: ", sorted(p.name for p in out_dir.glob("*.csv*")))
print("figures:", sorted(p.name for p in (out_dir / "figures").glob("*.png")))'''),
    ("code", '''display(Image(filename=str(out_dir / "figures" / "placement.png"), width=900))'''),
    ("code", '''triage = out_dir / "figures" / "triage.png"
if triage.exists():
    display(Image(filename=str(triage), width=900))
    display(pd.read_csv(out_dir / "triage_top20.csv").round(2))'''),
    ("md", '''**Triage** turns the ranking into something an analyst would feel: if only the top 20 % of
tiles (by predicted error) are checked by a person, what share of all wrongly mapped pixels do
they see? `perfect_order` is the best any ranking could do.'''),
    ("md", '''## 5.7 Comparing two runs on the same tiles

Once Experiment B has run, this puts it next to the zero-shot result, on exactly the tiles both
tables contain:'''),
    ("code", '''from p037.analyze import compare_runs

exp_b = Path(cfg.data_root) / "runs" / "exp_b_usf" / "tiles.csv"
zero_shot = REPO_ROOT / "analysis" / "usf_results.csv"
if exp_b.exists() and zero_shot.exists():
    display(compare_runs({"zero-shot": zero_shot, "Experiment B": exp_b}, "flood", cfg.score_class).round(3))
else:
    print("Experiment B has no tiles.csv yet:", exp_b)'''),
    ("md", '''## What goes in the report

- `placement.png`: proxy distributions, training against test (requested by the supervisor).
- `error_vs_confidence.png`, `error_vs_proxy.png`: the two "error against ..." plots.
- `weights_*.png`: the OOD weights with intervals.
- `held_out_*.png` and `triage.png`: the test of whether any of it transfers.

`report.md` holds the numbers behind each figure. It deliberately contains no interpretation:
that is yours to write.'''),
]


def build() -> list[Path]:
    paths = []
    for name, cells in NOTEBOOKS.items():
        nb = nbf.v4.new_notebook()
        nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
        nb.cells = [nbf.v4.new_markdown_cell(text) if kind == "md" else nbf.v4.new_code_cell(text)
                    for kind, text in cells]
        for i, cell in enumerate(nb.cells):  # stable ids: an unchanged cell gives an unchanged file
            cell["id"] = f"{name[:2]}-{i:02d}"
        path = HERE / name
        nbf.write(nb, path)
        paths.append(path)
    return paths


def execute(paths: list[Path], out_dir: Path) -> None:
    """Run every notebook top to bottom; executed copies are written to ``out_dir``.

    P037_CONFIG and P037_DATA are taken from the environment, as in the notebooks themselves."""
    from nbclient import NotebookClient

    out_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        nb = nbf.read(path, as_version=4)
        NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(HERE)}}).execute()
        nbf.write(nb, out_dir / path.name)
        print("executed", path.name)


if __name__ == "__main__":
    written = build()
    print("wrote", ", ".join(p.name for p in written))
    if "--execute" in sys.argv:
        only = [a for a in sys.argv[1:] if a.endswith(".ipynb")]
        target = Path(sys.argv[sys.argv.index("--execute") + 1])
        execute([p for p in written if not only or p.name in only], target)
