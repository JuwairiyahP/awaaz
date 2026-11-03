"""Load a trained detector and score windows of a clip."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .detector import build_model


def load_detector(ckpt_path: str | Path, device: torch.device | str = "cpu") -> torch.nn.Module:
    device = torch.device(device)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["model"], **ckpt.get("kwargs", {}))
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    # a frozen XLS-R checkpoint deliberately omits backbone weights (re-downloaded)
    missing = [k for k in missing if not k.startswith("backbone.")]
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    model.eval().to(device)
    model.awaaz_meta = {"name": ckpt["model"], "epoch": ckpt.get("epoch"), "val": ckpt.get("val")}
    return model


@torch.no_grad()
def score_windows(model: torch.nn.Module, windows: np.ndarray, batch: int = 16) -> np.ndarray:
    """Probability of synthetic speech for each window [N, T] -> [N]."""
    device = next(model.parameters()).device
    out = []
    for i in range(0, len(windows), batch):
        x = torch.from_numpy(windows[i : i + batch]).to(device)
        out.append(torch.sigmoid(model(x)).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)
