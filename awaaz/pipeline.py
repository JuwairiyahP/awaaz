"""End-to-end analysis of one clip: three independent checks, one verdict.

    python -m awaaz.pipeline clip.ogg --ckpt runs/xlsr/best.pt
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import SAMPLE_RATE
from .audio import load_audio, make_windows, peak_normalize, trim_silence
from .fusion import Thresholds, decide


@dataclass
class Analysis:
    verdict: dict
    synthetic: float
    window_scores: list[float]
    window_starts: list[float]
    speaker: dict
    scam: dict
    duration_s: float
    timings: dict

    def to_dict(self):
        return self.__dict__


class Awaaz:
    def __init__(self, ckpt: str | None, vault_path: str = "data/vault.json", device: str = "cpu",
                 thresholds: Thresholds | None = None, use_speaker: bool = True, use_scam: bool = True,
                 use_nli: bool = True, whisper_size: str = "small"):
        from .speaker import VoiceVault

        self.th = thresholds or Thresholds()
        self.detector = None
        if ckpt:
            from .inference import load_detector

            self.detector = load_detector(ckpt, device)
            meta_th = (self.detector.awaaz_meta or {}).get("threshold")
            if meta_th is not None and not os.environ.get("AWAAZ_TH_SYNTH"):  # explicit override wins
                self.th.synthetic = float(meta_th)
        self.vault = VoiceVault(vault_path)
        self.encoder = None
        if use_speaker:
            from .speaker import SpeakerEncoder

            self.encoder = SpeakerEncoder(device=device)
        self.scam = None
        if use_scam:
            from .scam import ScamScorer

            self.scam = ScamScorer(use_nli=use_nli, whisper_size=whisper_size, device=device)

    # ------------------------------------------------------------ enrollment
    def enroll(self, name: str, path_or_audio) -> None:
        if self.encoder is None:
            raise RuntimeError("speaker encoder disabled")
        x = load_audio(path_or_audio) if isinstance(path_or_audio, (str, Path)) else path_or_audio
        self.vault.enroll(name, self.encoder.embed(x))

    # ------------------------------------------------------------ analysis
    def analyze(self, path_or_audio) -> Analysis:
        t = {}
        t0 = time.time()
        x = load_audio(path_or_audio) if isinstance(path_or_audio, (str, Path)) else np.asarray(path_or_audio, np.float32)
        x = peak_normalize(trim_silence(x))
        duration = len(x) / SAMPLE_RATE
        t["load"] = time.time() - t0

        # A: synthetic detector, per window
        t0 = time.time()
        if self.detector is not None:
            from .inference import score_windows

            windows, starts = make_windows(x)
            ws = score_windows(self.detector, windows)
            from .inference import aggregate

            synthetic = aggregate(ws)
        else:
            ws, starts, synthetic = np.zeros(0), np.zeros(0), 0.0
        t["detector"] = time.time() - t0

        # B: speaker match
        t0 = time.time()
        speaker = {"name": None, "similarity": 0.0, "scores": {}}
        if self.encoder is not None and self.vault.people:
            m = self.vault.match(self.encoder.embed(x), self.th.speaker)
            speaker = {"name": m.name, "similarity": m.similarity, "scores": m.all_scores}
        t["speaker"] = time.time() - t0

        # C: scam script
        t0 = time.time()
        scam = {"score": 0.0, "cues": {}, "phrases": {}, "transcript": "", "translation": "", "language": ""}
        if self.scam is not None:
            r = self.scam.score_audio(x)
            scam = {"score": r.score, "cues": r.cues, "phrases": r.phrases, "transcript": r.transcript,
                    "translation": r.translation, "language": r.language}
        t["scam"] = time.time() - t0

        v = decide(synthetic, speaker["name"], speaker["similarity"], scam["score"], scam["phrases"], self.th)
        return Analysis(v.to_dict(), round(synthetic, 4), [round(float(s), 4) for s in ws],
                        [float(s) for s in starts], speaker, scam, round(duration, 2),
                        {k: round(v, 3) for k, v in t.items()})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--ckpt")
    ap.add_argument("--vault", default="data/vault.json")
    ap.add_argument("--enroll", help="enroll this clip under NAME instead of analysing")
    ap.add_argument("--no-nli", action="store_true")
    ap.add_argument("--no-speaker", action="store_true")
    ap.add_argument("--no-scam", action="store_true")
    a = ap.parse_args(argv)
    aw = Awaaz(a.ckpt, a.vault, use_nli=not a.no_nli, use_speaker=not a.no_speaker, use_scam=not a.no_scam)
    if a.enroll:
        aw.enroll(a.enroll, a.audio)
        print(f"enrolled {a.enroll}")
        return
    print(json.dumps(aw.analyze(a.audio).to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
