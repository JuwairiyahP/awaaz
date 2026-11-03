"""Shortcut audit: could a model tell real from fake WITHOUT hearing the voice?

If real and fake clips differ in codec, sample rate, bandwidth, loudness,
duration or leading silence, a detector can score a near-perfect EER by
learning the recording pipeline instead of the voice. This script measures
those properties per class so we can fix the data before training, and so the
README can show the check was done.

    python scripts/audit_shortcuts.py --root /kaggle/working/data --n 150

`--root` must contain real/<lang>/... and fake/<lang>/... (or pass --dirs).
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from awaaz.audio import AUDIO_EXTENSIONS, load_audio  # noqa: E402


def probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=codec_name,sample_rate,bit_rate,channels:format=bit_rate,duration", "-of", "json", str(path)],
        capture_output=True, text=True)
    try:
        j = json.loads(out.stdout)
        s, f = j["streams"][0], j.get("format", {})
        br = s.get("bit_rate") or f.get("bit_rate")
        return {"codec": s.get("codec_name"), "sr": int(s.get("sample_rate", 0)),
                "kbps": round(int(br) / 1000) if br and br != "N/A" else None,
                "dur": float(f.get("duration", 0) or 0)}
    except Exception:
        return {"codec": "?", "sr": 0, "kbps": None, "dur": 0.0}


def signal_stats(path: Path) -> dict:
    """At the model's 16 kHz: energy above 4/6/7 kHz, loudness, leading silence."""
    x = load_audio(path, 16000)
    if len(x) < 1600:
        return {}
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / 16000)
    total = spec.sum() + 1e-12
    frame = 480
    n = len(x) // frame
    rms = np.sqrt((x[: n * frame].reshape(n, frame) ** 2).mean(1) + 1e-12)
    db = 20 * np.log10(rms / rms.max())
    first_voiced = int(np.argmax(db > -40)) * frame / 16000
    speech_level = np.percentile(db, 90)
    return {
        # background noise: quietest 10% of frames relative to the speech level
        # (crowd-recorded phone audio has a room/mic floor; clean TTS often doesn't)
        "noise_db": float(np.percentile(db, 10) - speech_level),
        "silent_frac": float(np.mean(db - speech_level < -60)),
        "zero_frac": float(np.mean(np.abs(x) < 1e-5)),
        "hf4k": float(spec[freqs > 4000].sum() / total),
        "hf6k": float(spec[freqs > 6000].sum() / total),
        "hf7k": float(spec[freqs > 7000].sum() / total),
        "rms_db": float(20 * np.log10(np.sqrt((x ** 2).mean()) + 1e-9)),
        "lead_sil": first_voiced,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/kaggle/working/data")
    ap.add_argument("--dirs", nargs="*", help="label:path pairs, overrides --root")
    ap.add_argument("--n", type=int, default=150, help="files sampled per class")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    rng = random.Random(a.seed)
    pools = defaultdict(list)
    if a.dirs:
        for d in a.dirs:
            label, p = d.split(":", 1)
            pools[label] += [f for f in Path(p).rglob("*") if f.suffix.lower() in AUDIO_EXTENSIONS]
    else:
        for label in ("real", "fake"):
            pools[label] = [f for f in (Path(a.root) / label).rglob("*") if f.suffix.lower() in AUDIO_EXTENSIONS]

    rows = defaultdict(list)
    for label, files in pools.items():
        for f in rng.sample(files, min(a.n, len(files))):
            rows[label].append({**probe(f), **signal_stats(f)})

    print(f"{'':10s}" + "".join(f"{lab:>22s}" for lab in rows))
    keys = ["dur", "kbps", "hf4k", "hf6k", "hf7k", "rms_db", "lead_sil", "noise_db", "silent_frac", "zero_frac"]
    for k in keys:
        cells = []
        for lab in rows:
            v = np.array([r[k] for r in rows[lab] if r.get(k) is not None], dtype=float)
            cells.append(f"{np.median(v):10.4f} ({np.percentile(v, 10):.3f}-{np.percentile(v, 90):.3f})" if len(v) else "n/a")
        print(f"{k:10s}" + "".join(f"{c:>22s}" for c in cells))
    for k in ("codec", "sr"):
        print(f"{k:10s}" + "".join(
            f"{str(dict(sorted({v: sum(1 for r in rows[lab] if r[k] == v) for v in {r[k] for r in rows[lab]}}.items(), key=lambda t: -t[1])[:3])):>22s}"
            for lab in rows))
    print("\nmedian (10th-90th percentile). hf6k = share of energy above 6 kHz at 16 kHz.")
    print("Warning signs: different codecs or sample rates, hf6k/hf7k differing by >2x, rms or lead_sil clearly apart,")
    print("noise_db differing by >10 dB or silent_frac/zero_frac much higher in one class (clean-vs-noisy shortcut).")


if __name__ == "__main__":
    main()
