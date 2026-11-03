"""The core experiment: is "matches a known voice AND sounds synthetic" a
reliable signature of a targeted clone?

IndicSynth clones real Kathbath speakers and keeps their ids, so for each
speaker in the test split (never seen by the detector) we can:

  enroll   2 real clips of the speaker (what a family would do)
  genuine  other real clips of that speaker        -> should match, should NOT look synthetic
  clone    IndicSynth clones of that speaker        -> may match (that's the threat), should look synthetic
  impostor real clips of OTHER speakers             -> should not match

and measure:
  1. how often clones fool the speaker matcher (the reason a detector is needed)
  2. how often the fused rule (match AND synthetic) catches clones
  3. how often it falsely accuses the real person (the cost that matters most)

    python scripts/clone_signature.py --ckpt runs/xlsr/best.pt --split test --chain whatsapp_opus16

Writes results/clone_signature/{summary.json, trials.csv, similarity.png}.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from awaaz.metrics import eer  # noqa: E402


def build_trials(rows, n_enroll=2, max_genuine=5, max_clone=5, max_impostor_per_spk=3, seed=0):
    """rows: manifest rows (path, label, lang, speaker, generator). Returns
    (enroll: {spk: [paths]}, trials: [dict(kind, spk, path, generator)])."""
    rng = np.random.default_rng(seed)
    by = defaultdict(lambda: {"real": [], "fake": []})
    for r in rows:
        if r["speaker"]:
            by[r["speaker"]][r["label"]].append(r)
    enroll, trials = {}, []
    eligible = [s for s, d in by.items() if len(d["real"]) >= n_enroll + 1 and d["fake"]]
    for s in sorted(eligible):
        reals = [by[s]["real"][i] for i in rng.permutation(len(by[s]["real"]))]
        fakes = [by[s]["fake"][i] for i in rng.permutation(len(by[s]["fake"]))]
        enroll[s] = [r["path"] for r in reals[:n_enroll]]
        trials += [{"kind": "genuine", "spk": s, "path": r["path"], "generator": "real"} for r in reals[n_enroll:n_enroll + max_genuine]]
        trials += [{"kind": "clone", "spk": s, "path": r["path"], "generator": r.get("generator", "")} for r in fakes[:max_clone]]
    # impostors: real clips of other enrolled speakers in the same language
    for s in enroll:
        lang = s.split("_")[0]
        others = [o for o in enroll if o != s and o.split("_")[0] == lang]
        for o in rng.permutation(others)[:6] if others else []:
            for r in by[o]["real"][n_enroll:n_enroll + max_impostor_per_spk]:
                trials.append({"kind": "impostor", "spk": s, "path": r["path"], "generator": "real"})
    return enroll, trials


def summarize(trials: list[dict], synth_threshold: float) -> dict:
    """trials carry 'sim' (speaker similarity to the claimed speaker) and,
    for genuine/clone, 'synth' (detector clip probability)."""
    by = defaultdict(list)
    for t in trials:
        by[t["kind"]].append(t)
    g = np.array([t["sim"] for t in by["genuine"]])
    imp = np.array([t["sim"] for t in by["impostor"]])
    c = np.array([t["sim"] for t in by["clone"]])
    spk_eer, spk_th = eer(np.r_[g, imp], np.r_[np.ones(len(g)), np.zeros(len(imp))])  # genuine = positive
    out = {
        "n": {k: len(v) for k, v in by.items()},
        "speakers": len({t["spk"] for t in trials if t["kind"] == "genuine"}),
        "speaker_eer_pct": round(spk_eer * 100, 2),
        "speaker_threshold": round(float(spk_th), 4),
        "similarity_median": {"genuine": round(float(np.median(g)), 3), "clone": round(float(np.median(c)), 3),
                              "impostor": round(float(np.median(imp)), 3)},
        # 1. the threat: clones accepted as the real person by speaker matching alone
        "clones_accepted_by_speaker_match_pct": round(float(np.mean(c >= spk_th)) * 100, 1),
        "impostors_accepted_pct": round(float(np.mean(imp >= spk_th)) * 100, 1),
    }
    gs = np.array([t["synth"] for t in by["genuine"]])
    cs = np.array([t["synth"] for t in by["clone"]])
    if len(gs) and len(cs):
        g_match, c_match = g >= spk_th, c >= spk_th
        out.update({
            "synth_threshold": round(float(synth_threshold), 4),
            # detector alone
            "detector_flags_clone_pct": round(float(np.mean(cs >= synth_threshold)) * 100, 1),
            "detector_flags_genuine_pct": round(float(np.mean(gs >= synth_threshold)) * 100, 1),
            # 2. fused red verdict "likely a cloned voice of <name>"
            "red_clone_of_name_on_clones_pct": round(float(np.mean(c_match & (cs >= synth_threshold))) * 100, 1),
            # 3. the costly mistake: telling a family their real relative is a clone
            "red_clone_of_name_on_genuine_pct": round(float(np.mean(g_match & (gs >= synth_threshold))) * 100, 1),
            # of the clones that fooled speaker matching, how many the detector catches
            "detector_catches_among_accepted_clones_pct": round(float(np.mean(cs[c_match] >= synth_threshold)) * 100, 1)
            if c_match.any() else None,
        })
        by_gen = defaultdict(list)
        for t in by["clone"]:
            by_gen[t["generator"] or "unknown"].append(t)
        out["by_generator"] = {gname: {"n": len(ts),
                                       "accepted_by_speaker_match_pct": round(float(np.mean([t["sim"] >= spk_th for t in ts])) * 100, 1),
                                       "detector_flags_pct": round(float(np.mean([t["synth"] >= synth_threshold for t in ts])) * 100, 1)}
                               for gname, ts in sorted(by_gen.items())}
    return out


def plot(trials, spk_th, path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    bins = np.linspace(-0.2, 1.0, 49)
    for kind, color, label in [("impostor", "#9e9e9e", "different person (real)"),
                               ("genuine", "#2e7d32", "same person (real)"),
                               ("clone", "#c62828", "AI clone of the same person")]:
        v = [t["sim"] for t in trials if t["kind"] == kind]
        if v:
            ax.hist(v, bins=bins, alpha=0.55, color=color, label=f"{label} (n={len(v)})", density=True)
    ax.axvline(spk_th, color="black", ls="--", lw=1)
    ax.text(spk_th + 0.01, ax.get_ylim()[1] * 0.92, "match threshold", fontsize=8)
    ax.set_xlabel("speaker similarity to the enrolled voice (ECAPA cosine)")
    ax.set_yticks([])
    ax.set_title("Clones sound like the real person to a speaker matcher")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", help="detector checkpoint (calibrated); omit for speaker-only analysis")
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--split", default="test", help="comma-separated; 'test' = speakers unseen by the detector")
    ap.add_argument("--chain", default="clean", help="channel applied to genuine/clone/impostor clips (not enrollment)")
    ap.add_argument("--synth-threshold", type=float, default=None, help="override the checkpoint's calibrated threshold")
    ap.add_argument("--out", default="results/clone_signature")
    a = ap.parse_args(argv)

    import torch

    from awaaz.audio import load_audio, make_windows, peak_normalize, trim_silence
    from awaaz.augment import apply_chain
    from awaaz.data import read_manifest
    from awaaz.speaker import SpeakerEncoder, cosine

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = read_manifest(a.manifest, set(a.split.split(",")))
    enroll, trials = build_trials(rows)
    if not enroll:
        raise SystemExit("no speaker has both real and cloned clips in this split")
    print(f"{len(enroll)} speakers, {len(trials)} trials")

    enc = SpeakerEncoder(device=device)
    centroids = {}
    for s, paths in enroll.items():
        e = np.mean([enc.embed(load_audio(p)) for p in paths], axis=0)
        centroids[s] = e / (np.linalg.norm(e) + 1e-9)

    det, th = None, None
    if a.ckpt:
        from awaaz.inference import aggregate, load_detector, score_windows

        det = load_detector(a.ckpt, device)
        th = a.synth_threshold if a.synth_threshold is not None else (det.awaaz_meta.get("threshold") or 0.5)

    audio_cache = {}
    for i, t in enumerate(trials):
        key = (t["path"], a.chain)
        if key not in audio_cache:
            x = peak_normalize(trim_silence(load_audio(t["path"])))
            audio_cache[key] = apply_chain(x, a.chain, seed=i)
        x = audio_cache[key]
        t["sim"] = cosine(enc.embed(x), centroids[t["spk"]])
        if det is not None and t["kind"] != "impostor":
            w, _ = make_windows(x)
            t["synth"] = aggregate(score_windows(det, w))
        if i % 100 == 0:
            print(f"  {i}/{len(trials)}", flush=True)

    summary = summarize(trials, th if th is not None else 0.5)
    summary.update({"split": a.split, "chain": a.chain, "ckpt": a.ckpt})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"summary_{a.chain}.json").write_text(json.dumps(summary, indent=2))
    with (out / f"trials_{a.chain}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["kind", "spk", "generator", "sim", "synth", "path"], extrasaction="ignore")
        w.writeheader()
        w.writerows(trials)
    plot(trials, summary["speaker_threshold"], out / f"similarity_{a.chain}.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
