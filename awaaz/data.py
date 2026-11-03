"""Training dataset backed by the manifest CSV."""
from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from . import SAMPLE_RATE, WINDOW_SECONDS
from .audio import load_audio, peak_normalize, random_crop, trim_silence
from .augment import apply_chain, random_channel


def default_cache_dir() -> str:
    """On Kaggle the 20 GB /kaggle/working disk fills up fast; /tmp is larger and
    doesn't count against it (it's cleared on restart, which only costs a re-decode)."""
    return "/tmp/awaaz_cache" if Path("/kaggle").exists() else "data/cache"


def _cache_load(path: Path) -> np.ndarray | None:
    """Read a cached array; a truncated or corrupt file (e.g. written while the
    disk was full) is deleted and treated as a miss instead of crashing."""
    if not path.exists():
        return None
    try:
        return np.load(path).astype(np.float32)
    except Exception:
        path.unlink(missing_ok=True)
        return None


def _cache_save(path: Path, x: np.ndarray) -> None:
    """Write atomically (temp file + rename) so a crash or full disk never
    leaves a half-written file behind. Failure to cache is not an error."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "wb") as f:
            np.save(f, x.astype(np.float16))
        os.replace(tmp, path)
    except OSError:
        tmp.unlink(missing_ok=True)


def read_manifest(path: str | Path, splits: set[str] | None = None, langs: set[str] | None = None) -> list[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if splits:
        rows = [r for r in rows if r["split"] in splits]
    if langs:
        rows = [r for r in rows if r["lang"] in langs]
    for r in rows:
        r["y"] = int(r["y"])
    return rows


class ClipDataset(Dataset):
    """Random 4 s crops with optional channel augmentation (train), or a fixed
    centre crop through a named channel chain (eval)."""

    def __init__(self, rows, train: bool, augment_p: float = 0.7, chain: str = "clean",
                 seconds: float = WINDOW_SECONDS, seed: int = 0, cache_dir: str | None = None):
        self.rows, self.train, self.augment_p, self.chain = rows, train, augment_p, chain
        self.length = int(SAMPLE_RATE * seconds)
        self.seed = seed
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.rows)

    def _load(self, i):
        r = self.rows[i]
        if self.cache_dir:  # decoded 16 kHz float16 cache; decoding mp3/flac is the slow part
            c = self.cache_dir / f"{hashlib.md5(r['path'].encode()).hexdigest()}.npy"
            x = _cache_load(c)
            if x is not None:
                return x
            x = peak_normalize(trim_silence(load_audio(r["path"])))
            _cache_save(c, x)
            return x
        return peak_normalize(trim_silence(load_audio(r["path"])))

    def __getitem__(self, i):
        x = self._load(i)
        if self.train:
            rng = np.random.default_rng()
            x = random_crop(x, self.length, rng)
            if rng.random() < self.augment_p:
                x = random_channel(x, rng, p_clean=0.0)
            gain = float(rng.uniform(0.3, 1.0))  # loudness shouldn't be a cue
            x = x * gain
        else:
            # eval clips are deterministic, so cache them per channel: validation
            # re-runs every epoch and shouldn't re-encode the same clips each time
            c = None
            if self.cache_dir and self.chain != "clean":
                key = f"{self.rows[i]['path']}|{self.chain}|{self.seed + i}|{self.length}"
                c = self.cache_dir / f"eval_{hashlib.md5(key.encode()).hexdigest()}.npy"
                cached = _cache_load(c)
                if cached is not None:
                    return torch.from_numpy(cached), torch.tensor(self.rows[i]["y"], dtype=torch.float32)
            if len(x) >= self.length:
                s = (len(x) - self.length) // 2
                x = x[s : s + self.length]
            else:
                x = random_crop(x, self.length, np.random.default_rng(0))
            x = apply_chain(x, self.chain, seed=self.seed + i)
            if c is not None:
                _cache_save(c, x)
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), torch.tensor(self.rows[i]["y"], dtype=torch.float32)


def balanced_sampler_weights(rows) -> torch.Tensor:
    """Equal total weight per (language, label) group so no language or class dominates."""
    from collections import Counter

    counts = Counter((r["lang"], r["y"]) for r in rows)
    return torch.tensor([1.0 / counts[(r["lang"], r["y"])] for r in rows], dtype=torch.double)
