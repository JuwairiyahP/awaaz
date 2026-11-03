"""Stream a small subset of a Hugging Face audio dataset to disk, without
downloading the whole thing (IndicSynth is ~51 GB).

Step 1, look before you download (prints configs, columns, one example):
    python scripts/fetch_hf_subset.py --dataset vdivyasharma/IndicSynth --inspect

Step 2, save N clips per language:
    python scripts/fetch_hf_subset.py --dataset vdivyasharma/IndicSynth \
        --config Hindi --n 2500 --out /kaggle/working/data/fake/hi

If the dataset has one config with a language column instead, filter on it:
    ... --lang-column language --lang-value Hindi

Audio is written as the original encoded bytes (wav/flac/mp3); Awaaz decodes
anything with ffmpeg, so no re-encoding is needed. A small metadata CSV with
the non-audio columns (speaker id, generator model, ...) is written alongside.
"""
from __future__ import annotations

import argparse
import csv
import re
import io
import json
from pathlib import Path


def find_audio_column(features) -> str | None:
    from datasets import Audio

    for name, feat in features.items():
        if isinstance(feat, Audio):
            return name
    for name in ("audio", "speech", "wav", "file"):
        if name in features:
            return name
    return None


def inspect(dataset: str, token: str | None):
    from datasets import get_dataset_config_names, load_dataset

    try:
        configs = get_dataset_config_names(dataset, token=token)
    except Exception as e:
        configs = [None]
        print(f"(could not list configs: {e})")
    print(f"configs ({len(configs)}): {configs}")
    cfg = configs[0] if configs else None
    ds = load_dataset(dataset, cfg, split="train", streaming=True, token=token)
    print(f"\nfeatures of config {cfg!r}:")
    for k, v in ds.features.items():
        print(f"  {k}: {v}")
    audio_col = find_audio_column(ds.features)
    if audio_col:
        from datasets import Audio

        ds = ds.cast_column(audio_col, Audio(decode=False))
    row = next(iter(ds))
    shown = {k: (f"<{len(v['bytes'] or b'')} bytes, path={v.get('path')}>" if k == audio_col and isinstance(v, dict) else v)
             for k, v in row.items()}
    print("\nfirst row:")
    print(json.dumps(shown, indent=2, ensure_ascii=False, default=str)[:3000])
    print(f"\naudio column: {audio_col}")


def fetch(a):
    from datasets import Audio, load_dataset

    ds = load_dataset(a.dataset, a.config, split=a.split, streaming=True, token=a.token)
    audio_col = a.audio_column or find_audio_column(ds.features)
    if not audio_col:
        raise SystemExit(f"no audio column found in {list(ds.features)}; pass --audio-column")
    ds = ds.cast_column(audio_col, Audio(decode=False))
    if a.lang_column:
        ds = ds.filter(lambda r: str(r[a.lang_column]).lower() == a.lang_value.lower())
    if a.shuffle_buffer:
        ds = ds.shuffle(seed=a.seed, buffer_size=a.shuffle_buffer)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta_cols, meta_rows, saved = None, [], 0
    for i, row in enumerate(ds):
        if saved >= a.n:
            break
        audio = row[audio_col]
        data = audio.get("bytes") if isinstance(audio, dict) else None
        if not data:
            continue
        ext = Path(audio.get("path") or "x.wav").suffix or ".wav"
        if ext.lower() not in {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a"}:
            ext = ".wav"
        spk = row.get(a.speaker_column) if a.speaker_column else None
        if spk is not None and a.speaker_regex:
            m = re.search(a.speaker_regex, str(spk))
            spk = m.group(1) if m else None
        if isinstance(spk, float) and spk.is_integer():
            spk = int(spk)
        sub = out / f"spk_{spk}" if spk not in (None, "") else out
        sub.mkdir(exist_ok=True)
        fname = sub / f"{saved:06d}{ext}"
        fname.write_bytes(data)
        meta = {k: v for k, v in row.items() if k != audio_col and not isinstance(v, (bytes, dict, list))}
        meta["file"] = str(fname)
        meta_cols = meta_cols or list(meta)
        meta_rows.append(meta)
        saved += 1
        if saved % 250 == 0:
            print(f"  {saved}/{a.n} saved (scanned {i + 1})", flush=True)

    with (out / "metadata.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=meta_cols or ["file"], extrasaction="ignore")
        w.writeheader()
        w.writerows(meta_rows)
    print(f"done: {saved} clips in {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--n", type=int, default=2500)
    ap.add_argument("--out")
    ap.add_argument("--audio-column")
    ap.add_argument("--lang-column")
    ap.add_argument("--lang-value")
    ap.add_argument("--speaker-column", help="put clips in spk_<id>/ folders so the manifest splits by speaker")
    ap.add_argument("--speaker-regex", help=r"pull the id out of the speaker column, e.g. Kathbath fname: '-(\d+)-[mf]\.'")
    ap.add_argument("--shuffle-buffer", type=int, default=0, help="e.g. 2000 to avoid taking only the first speakers")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--token", default=None, help="HF token for gated datasets (or set HF_TOKEN)")
    a = ap.parse_args(argv)
    if a.inspect:
        import os
        import sys

        inspect(a.dataset, a.token)
        sys.stdout.flush()
        os._exit(0)  # streaming readers keep background threads alive; exit without waiting on them
    if not a.out:
        raise SystemExit("--out is required")
    import os
    import sys

    fetch(a)
    sys.stdout.flush()
    os._exit(0)  # same reason as --inspect: don't hang on streaming threads after we're done


if __name__ == "__main__":
    main()
