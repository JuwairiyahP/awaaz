"""Score every clip in a folder with one or more detectors: the out-of-domain check.

    python scripts/score_folder.py /kaggle/working/my_real_notes --ckpt runs/xlsr/best.pt runs/specresnet/best.pt --expect real
    python scripts/score_folder.py /kaggle/working/my_clones     --ckpt runs/xlsr/best.pt runs/specresnet/best.pt --expect fake

Real WhatsApp voice notes from people outside the training data are the
honest test: a detector that learned "Kathbath recording conditions = real"
instead of "synthetic voice = fake" will flag them. Prints per-clip
probabilities and how many land on the expected side of each model's
calibrated threshold; writes a CSV next to --out.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from awaaz.audio import AUDIO_EXTENSIONS, load_audio, make_windows, peak_normalize, trim_silence  # noqa: E402
from awaaz.inference import aggregate, load_detector, score_windows  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--expect", choices=["real", "fake"], help="what these clips are, to count mistakes")
    ap.add_argument("--out", default=None, help="CSV path (default results/ood_<folder>.csv)")
    a = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files = sorted(f for f in Path(a.folder).rglob("*") if f.suffix.lower() in AUDIO_EXTENSIONS)
    if not files:
        raise SystemExit(f"no audio in {a.folder}")
    models = {}
    for c in a.ckpt:
        m = load_detector(c, device)
        name = f"{m.awaaz_meta['name']}({Path(c).parent.name})"
        models[name] = (m, m.awaaz_meta.get("threshold") if m.awaaz_meta.get("threshold") is not None else 0.5)

    rows = []
    for f in files:
        x = peak_normalize(trim_silence(load_audio(f)))
        w, _ = make_windows(x)
        r = {"file": f.name, "seconds": round(len(x) / 16000, 1)}
        for name, (m, th) in models.items():
            r[name] = round(aggregate(score_windows(m, w)), 3)
        rows.append(r)

    width = max(len(r["file"]) for r in rows)
    print(f"{'file':{width}s}  {'sec':>5s}  " + "  ".join(f"{n:>22s}" for n in models))
    for r in rows:
        cells = []
        for n, (_, th) in models.items():
            flag = "FAKE" if r[n] >= th else "real"
            cells.append(f"{r[n]:.3f} {flag:>4s}".rjust(22))
        print(f"{r['file']:{width}s}  {r['seconds']:5.1f}  " + "  ".join(cells))
    print()
    for n, (_, th) in models.items():
        s = np.array([r[n] for r in rows])
        line = f"{n}: threshold {th:.3f}, flagged fake {int((s >= th).sum())}/{len(s)}"
        if a.expect:
            wrong = int((s >= th).sum()) if a.expect == "real" else int((s < th).sum())
            line += f"  -> {wrong} mistakes on clips that are {a.expect}"
        print(line)

    out = Path(a.out or f"results/ood_{Path(a.folder).name}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
