"""Pick the detector's decision threshold on validation data and store it in the checkpoint.

    python scripts/calibrate.py --ckpt runs/xlsr/best.pt

Scores the full-length validation clips the same way the app does (4 s windows,
top-third mean), on clean and WhatsApp-Opus audio, and records:
  threshold        probability at the pooled equal-error point (used by the app)
  threshold_fpr5   probability where 5% of real clips would be flagged
Writes them into the checkpoint (a .bak copy is kept) and prints a summary.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from awaaz.audio import load_audio, make_windows, peak_normalize, trim_silence  # noqa: E402
from awaaz.augment import apply_chain  # noqa: E402
from awaaz.data import read_manifest  # noqa: E402
from awaaz.inference import aggregate, load_detector, score_windows  # noqa: E402
from awaaz.metrics import binary_report, eer  # noqa: E402


def clip_scores(model, rows, chain, max_seconds=20.0):
    out = []
    for i, r in enumerate(rows):
        x = peak_normalize(trim_silence(load_audio(r["path"])))[: int(16000 * max_seconds)]
        x = apply_chain(x, chain, seed=i)
        w, _ = make_windows(x)
        out.append(aggregate(score_windows(model, w)))
    return np.array(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--split", default="val")
    ap.add_argument("--chains", default="clean,whatsapp_opus16")
    ap.add_argument("--max-clips", type=int, default=800)
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_detector(a.ckpt, device)
    rows = read_manifest(a.manifest, {a.split})
    rng = np.random.default_rng(0)
    if len(rows) > a.max_clips:
        rows = [rows[i] for i in rng.choice(len(rows), a.max_clips, replace=False)]
    y = np.array([r["y"] for r in rows])

    all_s, all_y, per_chain = [], [], {}
    for chain in a.chains.split(","):
        s = clip_scores(model, rows, chain)
        e, t = eer(s, y)
        per_chain[chain] = {"eer": round(e * 100, 2), "threshold": round(float(t), 4)}
        all_s.append(s)
        all_y.append(y)
        print(f"{chain:18s} clip-level EER {e*100:5.2f}%  at p={t:.3f}")
    s, yy = np.concatenate(all_s), np.concatenate(all_y)
    e, thr = eer(s, yy)
    reals = np.sort(s[yy == 0])
    thr5 = float(reals[int(np.ceil(0.95 * len(reals))) - 1]) if len(reals) else thr
    rep = binary_report(s, yy, thr)
    calib = {"split": a.split, "n_clips": int(len(rows)), "chains": per_chain,
             "pooled_eer": round(e * 100, 2), "threshold": round(float(thr), 4),
             "threshold_fpr5": round(thr5, 4), "at_threshold": {k: round(v, 4) if isinstance(v, float) else v
                                                               for k, v in rep.items()}}
    print(json.dumps(calib, indent=2))

    ck = Path(a.ckpt)
    shutil.copy(ck, ck.with_suffix(".pt.bak"))
    state = torch.load(ck, map_location="cpu", weights_only=False)
    state["threshold"] = float(thr)
    state["calibration"] = calib
    torch.save(state, ck)
    print(f"saved threshold {thr:.4f} into {ck}")
    if a.out_json:
        Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out_json).write_text(json.dumps(calib, indent=2))


if __name__ == "__main__":
    main()
