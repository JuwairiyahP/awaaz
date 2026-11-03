"""Build a train/val/test manifest CSV from folders of real and fake audio.

Each --source is  label:lang:path  where label is `real` or `fake`, lang is a
short code (hi, kn, ta, bn, mr, te, ml, en, ...) and path is a folder that is
searched recursively for audio files.

Example (Kaggle):
    python scripts/build_manifest.py \
        --source fake:hi:/kaggle/input/indicsynth/hindi \
        --source real:hi:/kaggle/input/indicsuperb/hindi \
        --source fake:te:/kaggle/input/indicsynth/telugu \
        --source real:te:/kaggle/input/indicsuperb/telugu \
        --holdout te,ml --max-per-group 3000 --out data/manifest.csv

Splits:
- languages in --holdout go entirely to `test_unseen` (never trained on)
- everything else is split by SPEAKER when a speaker id can be found in the
  path, so the same voice never appears in both train and test. Otherwise it
  falls back to a per-file split.
"""
from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from awaaz.audio import AUDIO_EXTENSIONS  # noqa: E402

SPEAKER_PATTERNS = [
    re.compile(r"(?:spk|speaker|spkr)[_-]?([A-Za-z0-9]+)", re.I),
]


def guess_speaker(path: Path, root: Path) -> str | None:
    rel = path.relative_to(root).as_posix()
    for pat in SPEAKER_PATTERNS:
        m = pat.search(rel)
        if m:
            return m.group(1)
    parts = path.relative_to(root).parts
    # common layout: <root>/<speaker>/<file>
    return parts[0] if len(parts) >= 2 else None


def parse_source(s: str) -> tuple[str, str, Path]:
    try:
        label, lang, path = s.split(":", 2)
    except ValueError:
        raise SystemExit(f"bad --source {s!r}; expected label:lang:path")
    if label not in {"real", "fake"}:
        raise SystemExit(f"label must be real or fake, got {label!r}")
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"path does not exist: {p}")
    return label, lang, p


def read_generators(root: Path, column: str) -> dict[str, str]:
    """metadata.csv written by fetch_hf_subset.py maps each file to its generator."""
    meta = root / "metadata.csv"
    if not meta.exists():
        return {}
    with meta.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or column not in rows[0]:
        return {}
    return {str(Path(r["file"]).resolve()): r[column] for r in rows if r.get("file")}


def collect(sources, max_per_group: int, seed: int, generator_column: str):
    rng = random.Random(seed)
    rows = []
    for label, lang, root in sources:
        files = sorted(f for f in root.rglob("*") if f.suffix.lower() in AUDIO_EXTENSIONS)
        rng.shuffle(files)
        if max_per_group:
            files = files[:max_per_group]
        gens = read_generators(root, generator_column) if label == "fake" else {}
        for f in files:
            spk = guess_speaker(f, root)
            gen = "real" if label == "real" else gens.get(str(f.resolve()), "unknown")
            rows.append({
                "path": str(f), "label": label, "y": 1 if label == "fake" else 0,
                "lang": lang, "speaker": f"{lang}_{spk}" if spk else "", "generator": gen,
            })
        extra = ""
        if gens:
            from collections import Counter
            extra = "  " + ", ".join(f"{g}={n}" for g, n in Counter(r["generator"] for r in rows[-len(files):]).items())
        print(f"  {label:4s} {lang:3s} {len(files):6d} files from {root}{extra}")
    return rows


def hold_out_generators(rows, holdout_gen: set[str]):
    """Move fakes from held-out cloning tools into `test_unseen_gen`, paired with
    copies of the real `test` clips of the same languages, so the detector is
    scored on a tool it never trained on."""
    if not holdout_gen:
        return rows
    langs = set()
    for r in rows:
        if r["label"] == "fake" and r["generator"] in holdout_gen and r["split"] != "test_unseen":
            r["split"] = "test_unseen_gen"
            langs.add(r["lang"])
    reals = [dict(r, split="test_unseen_gen") for r in rows
             if r["label"] == "real" and r["split"] == "test" and r["lang"] in langs]
    return rows + reals


def _split_speakers(speakers: list[str], val_frac: float, test_frac: float, rng) -> tuple[set, set]:
    speakers = sorted(speakers)
    rng.shuffle(speakers)
    if not speakers:
        return set(), set()
    n_val = max(1, round(len(speakers) * val_frac)) if len(speakers) >= 3 else 0
    n_test = max(1, round(len(speakers) * test_frac)) if len(speakers) >= 3 else 0
    return set(speakers[:n_val]), set(speakers[n_val : n_val + n_test])


def assign_splits(rows, holdout: set[str], val_frac: float, test_frac: float, seed: int):
    """Speaker-disjoint ACROSS classes, per language.

    In IndicSynth + Kathbath the cloned target speakers are real Kathbath
    speakers with the same ids, so a voice in the test split must be unheard in
    training, both as itself and as a clone. Speakers are pooled by which
    classes they appear in (both / fake only / real only) and each pool is
    split separately, so every split keeps real and fake clips.
    """
    rng = random.Random(seed)
    by_lang = defaultdict(list)
    for r in rows:
        if r["lang"] in holdout:
            r["split"] = "test_unseen"
        else:
            by_lang[r["lang"]].append(r)

    for lang, grp in by_lang.items():
        labels_of = defaultdict(set)
        for r in grp:
            if r["speaker"]:
                labels_of[r["speaker"]].add(r["label"])
        if len(labels_of) >= 10:
            pools = {
                "both": [s for s, l in labels_of.items() if l == {"real", "fake"}],
                "fake": [s for s, l in labels_of.items() if l == {"fake"}],
                "real": [s for s, l in labels_of.items() if l == {"real"}],
            }
            val_s, test_s = set(), set()
            for pool in pools.values():
                v, t = _split_speakers(pool, val_frac, test_frac, rng)
                val_s |= v
                test_s |= t
            for r in grp:
                s = r["speaker"]
                if not s:  # unknown speaker: random, but never leaks a known voice
                    x = rng.random()
                    r["split"] = "val" if x < val_frac else "test" if x < val_frac + test_frac else "train"
                else:
                    r["split"] = "val" if s in val_s else "test" if s in test_s else "train"
            print(f"  {lang}: speakers both={len(pools['both'])} fake-only={len(pools['fake'])} "
                  f"real-only={len(pools['real'])} -> val {len(val_s)}, test {len(test_s)}")
        else:  # too few speakers known: per-file split within each class
            for label in ("real", "fake"):
                sub = [r for r in grp if r["label"] == label]
                rng.shuffle(sub)
                n_val, n_test = int(len(sub) * val_frac), int(len(sub) * test_frac)
                for i, r in enumerate(sub):
                    r["split"] = "val" if i < n_val else "test" if i < n_val + n_test else "train"
    return rows


def summarize(rows):
    table = defaultdict(lambda: defaultdict(int))
    for r in rows:
        table[(r["split"], r["lang"])][r["label"]] += 1
    print("\nsplit        lang   real   fake")
    for (split, lang), c in sorted(table.items()):
        print(f"{split:12s} {lang:4s} {c['real']:6d} {c['fake']:6d}")
    bad = [k for k, c in table.items() if c["real"] == 0 or c["fake"] == 0]
    if bad:
        print(f"\nWARNING: these groups have only one class, so EER is undefined there: {bad}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", action="append", required=True)
    ap.add_argument("--holdout", default="", help="comma-separated languages kept out of training")
    ap.add_argument("--max-per-group", type=int, default=3000)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--generator-column", default="Generative Model")
    ap.add_argument("--holdout-generator", default="",
                    help="comma-separated cloning tools kept out of training, e.g. xtts_v2")
    ap.add_argument("--out", default="data/manifest.csv")
    a = ap.parse_args(argv)

    sources = [parse_source(s) for s in a.source]
    holdout = {h.strip() for h in a.holdout.split(",") if h.strip()}
    holdout_gen = {h.strip() for h in a.holdout_generator.split(",") if h.strip()}
    print("collecting:")
    rows = collect(sources, a.max_per_group, a.seed, a.generator_column)
    if not rows:
        raise SystemExit("no audio files found")
    rows = assign_splits(rows, holdout, a.val_frac, a.test_frac, a.seed)
    rows = hold_out_generators(rows, holdout_gen)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "y", "lang", "speaker", "generator", "split"])
        w.writeheader()
        w.writerows(rows)
    summarize(rows)
    print(f"\nwrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
