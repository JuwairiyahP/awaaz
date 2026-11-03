"""Train a synthetic-voice detector on the manifest (run on a Kaggle/Colab GPU).

    # fast baseline, ~20-40 min on a T4
    python scripts/train_detector.py --model specresnet --epochs 12 --out runs/specresnet

    # main model: frozen XLS-R + head
    python scripts/train_detector.py --model xlsr --epochs 6 --batch 24 --lr 1e-3 --out runs/xlsr

Saves runs/<name>/best.pt (lowest validation EER) and a JSON log. Validation
runs on two channels, clean and WhatsApp Opus, and the checkpoint is chosen
on their average, so we don't pick a model that only works on clean audio.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from awaaz.data import ClipDataset, balanced_sampler_weights, read_manifest  # noqa: E402
from awaaz.detector import build_model, count_parameters  # noqa: E402
from awaaz.metrics import eer  # noqa: E402

VAL_CHAINS = ["clean", "whatsapp_opus16"]


@torch.no_grad()
def score_loader(model, loader, device, amp=True):
    model.eval()
    scores, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            s = model(x)
        scores.append(s.float().cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(scores), np.concatenate(labels)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--model", choices=["specresnet", "xlsr"], default="specresnet")
    ap.add_argument("--backbone", default="facebook/wav2vec2-xls-r-300m")
    ap.add_argument("--unfreeze-top", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--augment-p", type=float, default=0.7)
    ap.add_argument("--steps-per-epoch", type=int, default=0, help="0 = one pass over train")
    ap.add_argument("--max-val", type=int, default=1500, help="cap validation clips for speed")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cache-dir", default="/kaggle/working/cache" if Path("/kaggle").exists() else "data/cache")
    ap.add_argument("--out", default="runs/specresnet")
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args(argv)

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    train_rows = read_manifest(a.manifest, {"train"})
    val_rows = read_manifest(a.manifest, {"val"})
    rng = np.random.default_rng(a.seed)
    if len(val_rows) > a.max_val:
        val_rows = [val_rows[i] for i in rng.choice(len(val_rows), a.max_val, replace=False)]
    print(f"train {len(train_rows)}  val {len(val_rows)}  device {device}")

    train_ds = ClipDataset(train_rows, train=True, augment_p=a.augment_p, cache_dir=a.cache_dir)
    n_samples = a.steps_per_epoch * a.batch if a.steps_per_epoch else len(train_rows)
    sampler = WeightedRandomSampler(balanced_sampler_weights(train_rows), n_samples, replacement=True)
    train_dl = DataLoader(train_ds, batch_size=a.batch, sampler=sampler, num_workers=a.workers,
                          pin_memory=True, drop_last=True, persistent_workers=a.workers > 0)
    val_dls = {c: DataLoader(ClipDataset(val_rows, train=False, chain=c, cache_dir=a.cache_dir),
                             batch_size=a.batch, num_workers=a.workers) for c in VAL_CHAINS}

    kw = {"backbone": a.backbone, "unfreeze_top": a.unfreeze_top} if a.model == "xlsr" else {}
    model = build_model(a.model, **kw).to(device)
    total, trainable = count_parameters(model)
    print(f"{a.model}: {total/1e6:.1f}M params, {trainable/1e6:.2f}M trainable")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * len(train_dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best, log = float("inf"), []
    for epoch in range(1, a.epochs + 1):
        model.train()
        t0, run_loss, n = time.time(), 0.0, 0
        for i, (x, y) in enumerate(train_dl):
            x, y = x.to(device, non_blocking=True), y.to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item() * len(y)
            n += len(y)
            if i % 50 == 0:
                print(f"  epoch {epoch} step {i}/{len(train_dl)} loss {run_loss / max(n, 1):.4f}", flush=True)

        val = {}
        for c, dl in val_dls.items():
            s, l = score_loader(model, dl, device)
            val[c] = eer(s, l)[0]
        mean_eer = float(np.mean(list(val.values())))
        entry = {"epoch": epoch, "train_loss": run_loss / max(n, 1), "val_eer": val,
                 "val_eer_mean": mean_eer, "minutes": (time.time() - t0) / 60}
        log.append(entry)
        print(f"epoch {epoch}: loss {entry['train_loss']:.4f}  " +
              "  ".join(f"EER[{c}] {v*100:.2f}%" for c, v in val.items()) + f"  ({entry['minutes']:.1f} min)")

        state = {k: v for k, v in model.state_dict().items() if not k.startswith("backbone.")} \
            if a.model == "xlsr" and a.unfreeze_top == 0 else model.state_dict()
        ckpt = {"model": a.model, "kwargs": kw, "state_dict": state, "epoch": epoch, "val": val}
        torch.save(ckpt, out / "last.pt")
        if mean_eer < best:
            best = mean_eer
            torch.save(ckpt, out / "best.pt")
            print(f"  new best ({best*100:.2f}%) saved")
        (out / "log.json").write_text(json.dumps(log, indent=2))

    print(f"done. best mean val EER {best*100:.2f}%  -> {out/'best.pt'}")


if __name__ == "__main__":
    main()
