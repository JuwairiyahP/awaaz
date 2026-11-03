"""Give fake clips the same recording pipeline as the real ones.

Kathbath (real) ships as AAC .m4a; IndicSynth (fake) as .wav. Re-encoding the
fakes with the same codec, sample rate and bitrate removes "AAC history" as a
shortcut, so the detector has to learn the voice, not the file format.

    python scripts/match_codec.py --src /kaggle/working/data/fake --dst /kaggle/working/data/fake_aac \
        --codec aac --sr 16000 --kbps 64 --lowpass auto --workers 8

--lowpass auto measures the real clips' bandwidth with --real-root and applies
the same cut to the fakes (AAC encoders band-limit, and a bandwidth gap is
the single easiest shortcut for a spectrogram model).
Folder layout, speaker folders and metadata.csv are preserved, with file
paths rewritten to the new .m4a files.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from awaaz.audio import AUDIO_EXTENSIONS, load_audio  # noqa: E402


def measure_cutoff(files, n=60, sr=16000) -> int:
    """Frequency below which 99.5% of energy lies, median over n real clips."""
    cuts = []
    for f in files[:n]:
        x = load_audio(f, sr)
        if len(x) < sr // 2:
            continue
        spec = np.abs(np.fft.rfft(x)) ** 2
        c = np.cumsum(spec) / (spec.sum() + 1e-12)
        cuts.append(np.fft.rfftfreq(len(x), 1 / sr)[np.searchsorted(c, 0.995)])
    return int(np.median(cuts)) if cuts else 0


def encode(src: Path, dst: Path, codec: str, sr: int, kbps: int, lowpass: int) -> bool:
    if dst.exists() and dst.stat().st_size > 0:
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    af = ["-af", ",".join([f"lowpass=f={lowpass}"] * 4)] if lowpass else []  # 4 cascaded 2-pole = steep 8th-order cut
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src), "-ac", "1", "-ar", str(sr),
           *af, "-c:a", codec, "-b:a", f"{kbps}k", str(dst)]
    return subprocess.run(cmd, capture_output=True).returncode == 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--codec", default="aac")
    ap.add_argument("--ext", default=".m4a")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--kbps", type=int, default=64)
    ap.add_argument("--lowpass", default="0", help="Hz, 0 = none, or 'auto' (needs --real-root)")
    ap.add_argument("--real-root", default="/kaggle/working/data/real")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)

    src, dst = Path(a.src), Path(a.dst)
    lowpass = 0
    if a.lowpass == "auto":
        reals = [f for f in Path(a.real_root).rglob("*") if f.suffix.lower() in AUDIO_EXTENSIONS]
        lowpass = measure_cutoff(sorted(reals)[:: max(1, len(reals) // 60)])
        print(f"real-clip 99.5% energy cutoff: {lowpass} Hz -> applying lowpass at {lowpass} Hz")
    elif int(a.lowpass):
        lowpass = int(a.lowpass)

    files = [f for f in src.rglob("*") if f.suffix.lower() in AUDIO_EXTENSIONS]
    jobs = [(f, (dst / f.relative_to(src)).with_suffix(a.ext)) for f in files]
    print(f"re-encoding {len(jobs)} files with {a.codec} {a.kbps} kbps @ {a.sr} Hz ...")
    with ThreadPoolExecutor(a.workers) as ex:
        ok = list(ex.map(lambda j: encode(j[0], j[1], a.codec, a.sr, a.kbps, lowpass), jobs))
    print(f"done: {sum(ok)}/{len(ok)} ok")

    # carry metadata.csv over with rewritten paths
    for meta in src.rglob("metadata.csv"):
        out = dst / meta.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        with meta.open(newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        for r in rows:
            if r.get("file"):
                p = Path(r["file"])
                try:
                    r["file"] = str((dst / p.relative_to(src)).with_suffix(a.ext))
                except ValueError:
                    pass
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
