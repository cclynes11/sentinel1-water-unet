"""Stage 2: train the U-Net on the patch cache (Wagner et al. 2024, section 2.6 recipe).

Recipe kept from the paper: Adam, learning rate 1e-4, batch 32, binary cross-entropy + Dice loss,
random horizontal and vertical flips, no other augmentation.

Deviations, each a config switch:
* The checkpoint kept is the one with the best validation F1, not accuracy. With 1.6 % flood
  pixels a model predicting no flood anywhere is 98.4 % accurate. Both ``best_f1.pt`` and
  ``best_accuracy.pt`` are saved and both metrics are logged, so the choice can be compared.
* Pixels without a label or without SAR data are masked from the loss.
* Epoch count and early stopping are set for this dataset (the paper ran 2000 epochs on 59 k patches).

Files written to the run folder: ``log.csv`` (one row per epoch), ``last.pt`` (everything needed
to resume), ``best_f1.pt``, ``best_accuracy.pt``, ``config.yaml``.
"""

from __future__ import annotations

import csv
import logging
import time

import torch
import torch.nn.functional as F

from .config import Config, save_config
from .data import make_loaders
from .metrics import scores
from .model import build_model, pick_device

log = logging.getLogger("p037.train")


def bce_dice_loss(p: torch.Tensor, y: torch.Tensor, m: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Binary cross-entropy + (1 - soft Dice), over the pixels where the mask is 1.

    ``p`` are probabilities: the model applies the sigmoid itself. Dice is computed over the
    whole batch, not per patch: 80 % of UrbanSARFloods patches contain no flood, and the Dice of
    an empty patch is decided by the smoothing constant alone."""
    p = p.float()
    bce = (F.binary_cross_entropy(p, y, reduction="none") * m).sum() / m.sum().clamp(min=1)
    inter = (p * y * m).sum()
    dice = (2 * inter + smooth) / ((p * m).sum() + (y * m).sum() + smooth)
    return bce + (1 - dice)


def random_flips(x, y, m, gen: torch.Generator):
    """Flip each patch of a batch left-right and up-down, each with probability 1/2.

    Image, target and mask of a patch are flipped together. ``gen`` is a CPU generator, so the
    flips are the same on any device."""
    flip = torch.rand((2, x.shape[0]), generator=gen) < 0.5
    for axis, selected in ((-1, flip[0]), (-2, flip[1])):
        idx = selected.nonzero().squeeze(1).to(x.device)
        if len(idx):
            x[idx], y[idx], m[idx] = x[idx].flip(axis), y[idx].flip(axis), m[idx].flip(axis)
    return x, y, m


def _to_device(batch, device):
    x, y, m = batch
    x = x.to(device, non_blocking=True).float().div_(255.0)  # 8-bit image / 255, as in the original
    return x, y.to(device, non_blocking=True), m.to(device, non_blocking=True)


@torch.no_grad()
def validate(model, loader, device: str, threshold: float, amp: bool) -> dict[str, float]:
    """Loss and pixel-pooled scores over the validation patches, dropout off."""
    model.eval()
    tp = fp = fn = tn = 0
    loss_sum, n = 0.0, 0
    for batch in loader:
        x, y, m = _to_device(batch, device)
        with torch.autocast(device_type=torch.device(device).type, enabled=amp):
            p = model(x)
        loss_sum += float(bce_dice_loss(p, y, m)) * len(x)
        n += len(x)
        pred, pos, ok = p >= threshold, y > 0.5, m > 0.5
        tp += int((pred & pos & ok).sum())
        fp += int((pred & ~pos & ok).sum())
        fn += int((~pred & pos & ok).sum())
        tn += int((~pred & ~pos & ok).sum())
    return {"val_loss": loss_sum / max(n, 1), **{f"val_{k}": v for k, v in scores(tp, fp, fn, tn).items()}}


def train(cfg: Config, device: str | None = None, restart: bool = False) -> dict:
    device = pick_device(device)
    amp = cfg.amp and torch.device(device).type == "cuda"
    torch.manual_seed(cfg.seed)
    train_loader, val_loader, train_rows, val_rows = make_loaders(cfg, device)
    if len(train_loader) == 0:
        raise ValueError(f"{len(train_rows)} training patches is fewer than one batch of {cfg.batch_size}")
    model = build_model(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scaler = torch.amp.GradScaler(torch.device(device).type, enabled=amp)
    save_config(cfg)
    out = cfg.out
    train_rows.assign(role="train")[["patch_id", "role"]].to_csv(out / "train_patches.csv", index=False)
    val_rows.assign(role="val")[["patch_id", "role"]].to_csv(out / "val_patches.csv", index=False)

    state = {"epoch": 0, "best_f1": -1.0, "best_accuracy": -1.0, "best_epoch": 0, "stopped_early": False}
    last = out / "last.pt"
    if last.exists() and not restart:
        ck = torch.load(last, map_location="cpu", weights_only=True)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        state = ck["state"]
        log.info("resuming %s after epoch %d (best F1 %.4f)", cfg.name, state["epoch"], state["best_f1"])
        if state.get("stopped_early") or state["epoch"] >= cfg.epochs:  # e.g. a chained Slurm job arriving late
            log.info("%s already finished (early stopping or last epoch): nothing to do", cfg.name)
            return state
    elif restart and (out / "log.csv").exists():
        (out / "log.csv").unlink()

    log_path = out / "log.csv"
    fields = ["epoch", "train_loss", "val_loss", "val_f1", "val_iou", "val_precision", "val_recall",
              "val_accuracy", "seconds", "kept"]
    if not log_path.exists():
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(fields)
    log.info("%s: %d training batches per epoch on %s (mixed precision %s)", cfg.name, len(train_loader), device, amp)

    t_start = time.time()
    metric = f"best_{cfg.checkpoint_metric}"
    while state["epoch"] < cfg.epochs:
        epoch = state["epoch"] + 1
        t0 = time.time()
        model.train()
        loss_sum, n = 0.0, 0
        # Everything random in an epoch (patch order, dropout, flips) is seeded from (seed, epoch) alone,
        # so an epoch is the same whether the run reached it in one job or after a resume.
        torch.manual_seed(cfg.seed * 100_003 + epoch)
        flip_gen = torch.Generator().manual_seed(cfg.seed * 100_003 + epoch)
        for batch in train_loader:
            x, y, m = _to_device(batch, device)
            if cfg.flips:
                x, y, m = random_flips(x, y, m, flip_gen)
            with torch.autocast(device_type=torch.device(device).type, enabled=amp):
                p = model(x)
            loss = bce_dice_loss(p, y, m)  # outside autocast: BCE on probabilities needs float32
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            loss_sum += float(loss.detach()) * len(x)
            n += len(x)
        row = {"epoch": epoch, "train_loss": loss_sum / n, **validate(model, val_loader, device, cfg.threshold, amp)}
        kept = []
        for name in ("f1", "accuracy"):
            value = row[f"val_{name}"]
            if value == value and value > state[f"best_{name}"]:  # value == value: not NaN
                state[f"best_{name}"] = value
                torch.save({"model": model.state_dict(), "epoch": epoch, f"val_{name}": value}, out / f"best_{name}.pt")
                kept.append(name)
                if name == cfg.checkpoint_metric:
                    state["best_epoch"] = epoch
        state["epoch"] = epoch
        state["stopped_early"] = bool(cfg.patience) and epoch - state["best_epoch"] >= cfg.patience
        row.update(seconds=round(time.time() - t0, 1), kept="+".join(kept))
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([row[k] if not isinstance(row[k], float) else f"{row[k]:.6g}" for k in fields])
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
                    "state": state}, last)
        log.info("epoch %d: train loss %.4f, val loss %.4f, val F1 %.4f (P %.3f, R %.3f), acc %.4f, %.0f s%s",
                 epoch, row["train_loss"], row["val_loss"], row["val_f1"], row["val_precision"], row["val_recall"],
                 row["val_accuracy"], row["seconds"], f"  [kept: {row['kept']}]" if kept else "")
        if state["stopped_early"]:
            log.info("no improvement of validation %s for %d epochs: stopping", cfg.checkpoint_metric, cfg.patience)
            break
        if cfg.max_hours and (time.time() - t_start) / 3600 > cfg.max_hours:
            log.info("time budget of %.1f h used: stopping after epoch %d; run again to resume", cfg.max_hours, epoch)
            break
    log.info("%s: best validation %s %.4f at epoch %d", cfg.name, cfg.checkpoint_metric, state[metric],
             state["best_epoch"])
    return state
