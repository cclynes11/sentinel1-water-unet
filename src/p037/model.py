"""Model construction: Wagner's U-Net, randomly initialised or with the published weights."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from .config import Config

log = logging.getLogger("p037.model")


def build_model(cfg: Config, init: str | None = None) -> torch.nn.Module:
    """``s1water.model.UNetS1Water`` (34.6 M parameters, dropout 0.5 after every encoder level).

    random:     PyTorch default initialisation. The class is instantiated directly; ``load_model``
                is not used because it loads the published weights.
    pretrained: the Wagner et al. (2024) weights from Hugging Face (zero-shot runs, fine-tuning).
    """
    from s1water.model import UNetS1Water

    model = UNetS1Water(in_channels=len(cfg.input_channels), dropout=cfg.dropout)
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.momentum = cfg.bn_momentum
    if (init or cfg.init) == "pretrained":
        from safetensors.torch import load_file

        from s1water.pipeline import download_weights

        model.load_state_dict(load_file(download_weights()))
        log.info("loaded the pretrained Wagner et al. (2024) weights")
    return model


def checkpoint_path(cfg: Config) -> Path | None:
    """The checkpoint ``evaluate`` should use: cfg.weights, else the best one of this run."""
    if cfg.weights:
        return cfg.resolve(cfg.weights)
    p = cfg.out / f"best_{cfg.checkpoint_metric}.pt"
    return p if p.exists() else None


def load_for_eval(cfg: Config, device: str) -> tuple[torch.nn.Module, str]:
    """Model in eval mode and a description of where its weights came from."""
    ckpt = checkpoint_path(cfg)
    if ckpt is not None:
        model = build_model(cfg, init="random")
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"] if "model" in state else state)
        origin = f"checkpoint {ckpt} (epoch {state.get('epoch', '?')})" if "model" in state else f"weights {ckpt}"
    elif cfg.init == "pretrained":
        model, origin = build_model(cfg), "pretrained Wagner et al. (2024) weights"
    else:
        raise FileNotFoundError(f"no checkpoint in {cfg.out}: run `p037 train` first, or set `weights`")
    return model.to(device).eval(), origin


def pick_device(prefer: str | None = None) -> str:
    if prefer:
        return prefer
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_mc_dropout(model: torch.nn.Module, on: bool) -> None:
    """MC dropout: dropout layers stochastic, everything else (batch norm) in eval mode."""
    model.eval()
    if on:
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train()
