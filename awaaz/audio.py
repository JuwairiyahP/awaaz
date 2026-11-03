"""Audio loading, silence trimming and windowing.

Everything downstream works on float32 mono numpy arrays at 16 kHz.
ffmpeg does the decoding, so any format a phone produces (.ogg/opus from
WhatsApp or Telegram, .m4a, .mp3, .amr, .wav) works.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from . import HOP_SECONDS, SAMPLE_RATE, WINDOW_SECONDS

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac", ".amr", ".webm"}


def load_audio(path: str | Path, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Decode any audio file to mono float32 in [-1, 1] at `sr` Hz."""
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", str(path), "-ac", "1", "-ar", str(sr), "-f", "f32le", "-",
    ]
    out = subprocess.run(cmd, capture_output=True, check=False)
    if out.returncode != 0:
        raise RuntimeError(f"ffmpeg could not decode {path}: {out.stderr.decode(errors='ignore')[:300]}")
    return np.frombuffer(out.stdout, dtype=np.float32).copy()


def save_wav(path: str | Path, x: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    import soundfile as sf

    sf.write(str(path), np.clip(x, -1, 1), sr, subtype="PCM_16")


def peak_normalize(x: np.ndarray, peak: float = 0.95) -> np.ndarray:
    m = float(np.max(np.abs(x))) if x.size else 0.0
    return x if m < 1e-8 else (x / m * peak).astype(np.float32)


def trim_silence(
    x: np.ndarray,
    sr: int = SAMPLE_RATE,
    frame_ms: float = 30.0,
    threshold_db: float = -40.0,
    keep_ms: float = 150.0,
) -> np.ndarray:
    """Energy-based VAD trim: drop leading/trailing frames quieter than
    `threshold_db` relative to the loudest frame. Interior pauses are kept
    because prosody around pauses is a useful cue for synthetic speech.
    """
    if x.size == 0:
        return x
    frame = max(1, int(sr * frame_ms / 1000))
    n = len(x) // frame
    if n == 0:
        return x
    frames = x[: n * frame].reshape(n, frame)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms / (rms.max() + 1e-12))
    voiced = np.where(db > threshold_db)[0]
    if voiced.size == 0:
        return x
    pad = int(keep_ms / frame_ms)
    start = max(0, voiced[0] - pad) * frame
    end = min(n, voiced[-1] + 1 + pad) * frame
    return x[start:end]


def make_windows(
    x: np.ndarray,
    sr: int = SAMPLE_RATE,
    win_s: float = WINDOW_SECONDS,
    hop_s: float = HOP_SECONDS,
) -> tuple[np.ndarray, np.ndarray]:
    """Split into overlapping windows.

    Returns (windows [N, win], start_times [N] in seconds). Clips shorter than
    one window are repeat-padded so short voice notes still get a score.
    """
    win, hop = int(sr * win_s), int(sr * hop_s)
    if len(x) == 0:
        x = np.zeros(win, dtype=np.float32)
    if len(x) < win:
        reps = int(np.ceil(win / len(x)))
        x = np.tile(x, reps)[:win]
    starts = list(range(0, len(x) - win + 1, hop))
    if starts[-1] + win < len(x):  # cover the tail
        starts.append(len(x) - win)
    windows = np.stack([x[s : s + win] for s in starts]).astype(np.float32)
    return windows, np.array(starts, dtype=np.float32) / sr


def random_crop(x: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    """Random fixed-length crop for training; repeat-pads short clips."""
    if len(x) == 0:
        return np.zeros(length, dtype=np.float32)
    if len(x) < length:
        x = np.tile(x, int(np.ceil(length / len(x))))
    start = int(rng.integers(0, len(x) - length + 1))
    return x[start : start + length].astype(np.float32)
