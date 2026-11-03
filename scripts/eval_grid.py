"""The headline result: EER for every language x channel, plus seen vs unseen.

    python scripts/eval_grid.py --ckpt runs/xlsr/best.pt --out results/xlsr

Writes results/<name>/grid.csv (lang, split, channel, n, EER, AUC),
summary.json, and latency.json (CPU seconds per 4 s window).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from awaaz.augment import EVAL_CHAINS  # noqa: E402
from awaaz.data import ClipDataset, read_manifest  # noqa: E402
from awaaz.inference import load_detector  # noqa: E402
from awaaz.metrics import auc, eer  # noqa: E402


@torch.no_grad()
def scores_for(model, rows, chain, device, batch, workers, cache_dir):
    dl = DataLoader(ClipDataset(rows, train=False, chain=chain, cache_dir=cache_dir), batch_size=batch, num_workers=workers)
    out = []
    for x, _ in dl:
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            out.append(model(x.to(device)).float().cpu().numpy())
    return np.concatenate(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--splits", default="test,test_unseen,test_unseen_gen")
    ap.add_argument("--chains", default=",".join(EVAL_CHAINS))
    ap.add_argument("--max-per-group", type=int, default=600, help="clips per (split, lang, label)")
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cache-dir", default="/kaggle/working/cache" if Path("/kaggle").exists() else "data/cache")
    ap.add_argument("--out", default="results/run")
    a = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_detector(a.ckpt, device)
    rows = read_manifest(a.manifest, set(a.splits.split(",")))

    rng = np.random.default_rng(0)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["split"], r["lang"], r["y"])].append(r)
    picked = []
    for g in groups.values():
        idx = rng.permutation(len(g))[: a.max_per_group]
        picked += [g[i] for i in idx]

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results, pooled = [], defaultdict(lambda: ([], []))
    for chain in a.chains.split(","):
        t0 = time.time()
        s = scores_for(model, picked, chain, device, a.batch, a.workers, a.cache_dir)
        y = np.array([r["y"] for r in picked])
        keys = defaultdict(list)
        for i, r in enumerate(picked):
            keys[(r["split"], r["lang"])].append(i)
        for (split, lang), idx in sorted(keys.items()):
            ys, ss = y[idx], s[idx]
            pooled[(split, chain)][0].extend(ss)
            pooled[(split, chain)][1].extend(ys)
            if ys.min() == ys.max():
                continue
            results.append({"split": split, "lang": lang, "channel": chain, "n": len(idx),
                            "eer": round(eer(ss, ys)[0] * 100, 2), "auc": round(auc(ss, ys), 4)})
        print(f"{chain}: done in {time.time() - t0:.0f}s")

    with (out / "grid.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "lang", "channel", "n", "eer", "auc"])
        w.writeheader()
        w.writerows(results)

    summary = {}
    for (split, chain), (ss, ys) in pooled.items():
        ys = np.array(ys)
        if ys.min() != ys.max():
            summary[f"{split}/{chain}"] = round(eer(np.array(ss), ys)[0] * 100, 2)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    # CPU latency per 4 s window, the number that matters for the deployed app
    cpu_model = load_detector(a.ckpt, torch.device("cpu"))
    x = torch.randn(1, 64000)
    with torch.no_grad():
        cpu_model(x)
        t0 = time.time()
        for _ in range(5):
            cpu_model(x)
    lat = (time.time() - t0) / 5
    (out / "latency.json").write_text(json.dumps({"cpu_seconds_per_4s_window": round(lat, 3)}, indent=2))

    print("\nEER % (pooled over languages):")
    for k, v in sorted(summary.items()):
        print(f"  {k:32s} {v:6.2f}")
    print(f"CPU latency: {lat:.3f}s per 4 s window")


if __name__ == "__main__":
    main()
