"""Speaker match: does this voice sound like someone the user enrolled?

Uses SpeechBrain's pretrained ECAPA-TDNN. Only 192-d embeddings are stored,
never the enrollment audio itself.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import SAMPLE_RATE
from .audio import make_windows, peak_normalize, trim_silence


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


class SpeakerEncoder:
    """Thin wrapper so the rest of the code never imports speechbrain directly."""

    def __init__(self, source: str = "speechbrain/spkrec-ecapa-voxceleb", device: str = "cpu", savedir: str = "pretrained/ecapa"):
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:  # older speechbrain
            from speechbrain.pretrained import EncoderClassifier
        self.model = EncoderClassifier.from_hparams(source=source, savedir=savedir, run_opts={"device": device})

    def embed(self, x: np.ndarray) -> np.ndarray:
        """Average of per-window embeddings: robust to short noisy notes."""
        import torch

        x = peak_normalize(trim_silence(x))
        windows, _ = make_windows(x, SAMPLE_RATE, win_s=3.0, hop_s=1.5)
        with torch.no_grad():
            e = self.model.encode_batch(torch.from_numpy(windows)).squeeze(1).cpu().numpy()
        e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)
        m = e.mean(axis=0)
        return (m / (np.linalg.norm(m) + 1e-9)).astype(np.float32)


@dataclass
class Match:
    name: str | None
    similarity: float
    all_scores: dict[str, float]


class VoiceVault:
    """Enrolled voices, stored as embeddings in a small JSON file.

    `path=None` keeps the vault in memory only (used by the public web demo,
    where each visitor's enrolled voices live in their own browser session).
    """

    def __init__(self, path: str | Path | None = "data/vault.json", people: dict | None = None):
        self.path = Path(path) if path is not None else None
        self.people: dict[str, list[list[float]]] = dict(people or {})
        if self.path is not None and self.path.exists() and people is None:
            self.people = json.loads(self.path.read_text())

    def save(self):
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.people))

    def enroll(self, name: str, embedding: np.ndarray):
        self.people.setdefault(name, []).append(embedding.astype(float).tolist())
        self.save()

    def remove(self, name: str):
        self.people.pop(name, None)
        self.save()

    def names(self) -> list[str]:
        return sorted(self.people)

    def match(self, embedding: np.ndarray, threshold: float) -> Match:
        scores = {}
        for name, embs in self.people.items():
            centroid = np.mean(np.array(embs, dtype=np.float32), axis=0)
            scores[name] = round(cosine(embedding, centroid), 4)
        if not scores:
            return Match(None, 0.0, {})
        best = max(scores, key=scores.get)
        return Match(best if scores[best] >= threshold else None, scores[best], scores)
