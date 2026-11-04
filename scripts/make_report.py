"""Turn everything in results/ into README-ready markdown tables and figures.

    python scripts/make_report.py            # writes results/REPORT.md and docs/figures/*.png

Missing results are skipped, so it can run at any point of the pipeline.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(os.environ.get("AWAAZ_REPORT_ROOT", Path(__file__).resolve().parents[1]))
RES = ROOT / "results"
FIG = ROOT / "docs" / "figures"

MODELS = [("xlsr", "XLS-R (shipped)"), ("xlsr_v2", "XLS-R v2: + re-record training (experiment)"),
          ("specresnet", "Log-mel ResNet (from scratch)")]


def main_run() -> str:
    """The shipped XLS-R run. v2 (re-record training) beat it on every benchmark channel,
    but its calibrated threshold (0.09) leaves too little margin on live microphone
    audio, so v1 ships and v2 is reported alongside it."""
    return "xlsr" if (RES / "xlsr" / "summary.json").exists() else "xlsr_v2"


def run_of(col: str) -> str:
    """score_folder.py names columns 'xlsr(xlsr_v2)': model type, then the run folder."""
    return col[col.index("(") + 1: col.rindex(")")] if "(" in col and col.endswith(")") else col


LABELS = {k: v for k, v in MODELS} | {"xlsr_genholdout": "XLS-R trained without XTTS-v2"}
CHANNEL_NAMES = {
    "clean": "Clean", "whatsapp_opus16": "WhatsApp (Opus 16k)", "phone_8k": "Phone (8 kHz)",
    "mp3_64": "MP3 64k", "rerecorded": "Speaker re-recorded", "rerecorded_whatsapp": "Re-recorded + WhatsApp",
    "dither": "Dither (no digital silence)", "lowpass7k": "Low-pass 7 kHz", "sanitized": "Both removed",
}
LANG_NAMES = {"hi": "Hindi", "kn": "Kannada", "ta": "Tamil", "bn": "Bengali", "mr": "Marathi",
              "te": "Telugu", "ml": "Malayalam"}


def load_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_grid(p: Path):
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["eer"] = float(r["eer"])
    return rows


def md_table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def pct(v) -> str:
    return "–" if v is None else f"{v:.2f}%"


# ------------------------------------------------------------------ sections
def section_headline(lines):
    chans = ["clean", "whatsapp_opus16", "phone_8k", "mp3_64", "rerecorded", "rerecorded_whatsapp"]
    rows = []
    for key, label in MODELS:
        s = load_json(RES / key / "summary.json")
        if not s:
            continue
        rows.append([f"**{label}** · seen languages, unseen speakers"] + [pct(s.get(f"test/{c}")) for c in chans])
        if any(k.startswith("test_unseen/") for k in s):
            rows.append([f"{label} · unseen languages (Telugu, Malayalam)"] + [pct(s.get(f"test_unseen/{c}")) for c in chans])
    if rows:
        lines += ["### Detector equal error rate (lower is better)", "",
                  md_table(["Model · test set"] + [CHANNEL_NAMES[c] for c in chans], rows), "",
                  "Test speakers were never heard in training, as themselves or as clones. "
                  "Channels are applied to the test clips to simulate how scam audio actually arrives.", ""]
        if (RES / "xlsr_v2" / "summary.json").exists() and main_run() == "xlsr":
            lines += ["**Why v2 isn't shipped:** retraining with randomised speaker re-recording improved every channel "
                      "above, but its validation-calibrated threshold came out at 0.09, against 0.81 for the shipped "
                      "model. Real out-of-domain voice notes scored up to 0.06 with it, which leaves almost no margin "
                      "for live microphone audio, so the Space runs the shipped model.", ""]


def section_per_language(lines):
    run = main_run()
    grid = load_grid(RES / run / "grid.csv")
    if not grid:
        return
    chans = [c for c in ["clean", "whatsapp_opus16", "phone_8k", "rerecorded_whatsapp"] if any(r["channel"] == c for r in grid)]
    cell = {(r["split"], r["lang"], r["channel"]): r["eer"] for r in grid}
    keys = sorted({(r["split"], r["lang"]) for r in grid if r["split"] in ("test", "test_unseen")},
                  key=lambda k: (k[0] != "test", k[1]))
    rows = [[LANG_NAMES.get(l, l) + (" *(unseen)*" if s == "test_unseen" else "")] +
            [pct(cell.get((s, l, c))) for c in chans] for s, l in keys]
    lines += [f"### {LABELS[run]} by language", "", md_table(["Language"] + [CHANNEL_NAMES[c] for c in chans], rows), ""]
    heatmap(grid, FIG / "eer_heatmap.png")
    if (FIG / "eer_heatmap.png").exists():
        lines += ["![EER by language and channel](docs/figures/eer_heatmap.png)", ""]


def heatmap(grid, path: Path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return
    chans = [c for c in CHANNEL_NAMES if any(r["channel"] == c for r in grid) and c not in ("dither", "lowpass7k", "sanitized")]
    keys = sorted({(r["split"], r["lang"]) for r in grid if r["split"] in ("test", "test_unseen")},
                  key=lambda k: (k[0] != "test", k[1]))
    if not chans or not keys:
        return
    cell = {(r["split"], r["lang"], r["channel"]): r["eer"] for r in grid}
    m = np.array([[cell.get((s, l, c), np.nan) for c in chans] for s, l in keys])
    short = {"clean": "Clean", "whatsapp_opus16": "WhatsApp\nOpus 16k", "phone_8k": "Phone\n8 kHz",
             "mp3_64": "MP3\n64k", "rerecorded": "Speaker\nre-recorded", "rerecorded_whatsapp": "Re-recorded\n+ WhatsApp"}
    fig, ax = plt.subplots(figsize=(1.15 * len(chans) + 2.6, 0.42 * len(keys) + 1.5))
    im = ax.imshow(m, cmap="Oranges", vmin=0, vmax=max(10.0, float(np.nanmax(m))), aspect="auto")
    ax.set_xticks(range(len(chans)), [short.get(c, c) for c in chans], fontsize=8)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(keys)), [LANG_NAMES.get(l, l) + (" (unseen)" if s == "test_unseen" else "") for s, l in keys], fontsize=9)
    ax.tick_params(length=0)
    n_seen = sum(1 for s, _ in keys if s == "test")
    if 0 < n_seen < len(keys):
        ax.axhline(n_seen - 0.5, color="black", lw=1.5)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if not np.isnan(m[i, j]):
                ax.text(j, i, f"{m[i, j]:.1f}", ha="center", va="center", fontsize=8.5,
                        color="white" if m[i, j] > 0.6 * im.get_clim()[1] else "black")
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title(f"{LABELS[main_run()]}: EER (%) by language and channel (lower is better)", fontsize=10, pad=34)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="EER %")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)


def section_unseen_tool(lines):
    s = load_json(RES / "xlsr_genholdout" / "summary.json")
    if not s:
        return
    rows = []
    for c in ["clean", "whatsapp_opus16"]:
        rows.append([CHANNEL_NAMES[c], pct(s.get(f"test/{c}")), pct(s.get(f"test_unseen_gen/{c}"))])
    lines += ["### Unseen cloning tool", "",
              "XLS-R retrained **without any XTTS-v2 clips**, then tested on XTTS-v2 clones "
              "(real scammers will use tools we never trained on).", "",
              md_table(["Channel", "Seen tool (FreeVC24)", "Unseen tool (XTTS-v2)"], rows), ""]


def section_ablation(lines):
    run = next((r for r in (main_run(), "xlsr") if (RES / r / "ablation" / "summary.json").exists()), None)
    if not run:
        return
    s = load_json(RES / run / "ablation" / "summary.json")
    order = ["clean", "dither", "lowpass7k", "sanitized"]
    rows = [[CHANNEL_NAMES[c], pct(s.get(f"test/{c}"))] for c in order if f"test/{c}" in s]
    lines += ["### Shortcut audit", "",
              "Real (Kathbath) and fake (IndicSynth) clips differ in two ways unrelated to the voice: real files "
              "contain ~18× more padded digital silence, and fakes resampled from 24 kHz have slightly less 7–8 kHz "
              "energy. Removing each artifact barely moves the EER, so the detector is not relying on them. "
              "Codec, bandwidth, duration, loudness and noise floor were also audited "
              "(`scripts/audit_shortcuts.py`); loudness is neutralised by normalisation.", "",
              md_table(["Test clips", f"{LABELS[run]} EER"], rows), ""]


def section_clone_signature(lines):
    rows, imgs = [], []
    th = {k: (load_json(RES / k / "calibration.json") or {}).get("threshold") for k, _ in MODELS}

    def model_of(s):
        t = s.get("synth_threshold")
        hit = [k for k, v in th.items() if v is not None and t is not None and abs(v - t) < 1e-3]
        return LABELS[hit[0]] if hit else (f"threshold {t}" if t is not None else "–")

    dirs = sorted({p.parent for p in RES.glob("**/summary_clean.json") if "clone_signature" in str(p.parent)}
                  | {p.parent for p in RES.glob("**/summary_whatsapp_opus16.json") if "clone_signature" in str(p.parent)},
                  key=lambda d: str(d))
    found = [(d, ch, load_json(d / f"summary_{ch}.json")) for d in dirs for ch in ["clean", "whatsapp_opus16"]]
    found = [(d, ch, s) for d, ch, s in found if s]
    found.sort(key=lambda t: (model_of(t[2]) != LABELS[main_run()], str(t[0]), t[1]))
    for d, ch, s in found:
        rows.append([model_of(s), CHANNEL_NAMES[ch], s.get("speakers"),
                     f"{s.get('clones_accepted_by_speaker_match_pct')}%",
                     f"{s.get('detector_catches_among_accepted_clones_pct')}%",
                     f"{s.get('red_clone_of_name_on_clones_pct')}%",
                     f"{s.get('red_clone_of_name_on_genuine_pct')}%"])
        png = d / f"similarity_{ch}.png"
        if png.exists() and not any(f"clone_similarity_{ch}.png" in i for i in imgs):
            FIG.mkdir(parents=True, exist_ok=True)
            shutil.copy(png, FIG / f"clone_similarity_{ch}.png")
            imgs.append(f"![Speaker similarity, {CHANNEL_NAMES[ch]}](docs/figures/clone_similarity_{ch}.png)")
    if rows:
        lines += ["### Clone signature: matches a known voice *and* sounds synthetic", "",
                  md_table(["Detector", "Channel", "Speakers", "Clones that fool speaker matching",
                            "…of those, caught by the detector", "Red “cloned voice of <name>” on clones",
                            "Red on the real person (false accusation)"], rows), "",
                  "Each test speaker is enrolled from 2 real clips; trials are other real clips of the same "
                  "person, IndicSynth clones of that person, and real clips of other people.", ""] + imgs + [""]


def section_scam(lines):
    rows = []
    for mode, label in [("lexicon", "Lexicon (shipped)"), ("lexicon+nli", "Lexicon + zero-shot NLI")]:
        s = load_json(RES / "scam" / f"summary_{mode}.json")
        if s:
            o = s["overall"]
            rows.append([label, f"{o['precision']:.2f}", f"{o['recall']:.2f}", f"{o['f1']:.2f}"])
    if rows:
        lines += ["### Scam-script scorer (60 labelled messages, Hindi / Hinglish / English)", "",
                  md_table(["Mode", "Precision", "Recall", "F1"], rows), "",
                  "Half the set is deliberately tricky everyday messages (money, hospitals, police, secrets, OTP "
                  "warnings). Zero-shot NLI over-fired on them, so the lexicon ships by default. "
                  "The set is small and self-written: a sanity check, not a benchmark.", ""]


def _ood_table(path: Path, expect: str, th: dict):
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None, []
    models = [k for k in rows[0] if k not in ("file", "seconds")]

    def thr(col):
        return th.get(run_of(col), th.get(col.split("(")[0], 0.5))

    def label(col):
        return LABELS.get(run_of(col), col)

    table = []
    for r in rows:
        cells = [r["file"].rsplit(".", 1)[0], r["seconds"]]
        for m in models:
            v = float(r[m])
            cells.append(f"{v:.3f} {'⚠️ fake' if v >= thr(m) else '✅ real'}")
        table.append(cells)
    summary = []
    for m in models:
        if expect == "real":
            ok = sum(float(r[m]) < thr(m) for r in rows)
            summary.append(f"{label(m)}: {ok}/{len(rows)} correctly called real")
        else:
            ok = sum(float(r[m]) >= thr(m) for r in rows)
            summary.append(f"{label(m)}: {ok}/{len(rows)} correctly called fake")
    return md_table(["Clip", "Seconds"] + [label(m) for m in models], table), summary


def section_ood(lines):
    th = {}
    for k, _ in MODELS:
        c = load_json(RES / k / "calibration.json")
        if c and c.get("threshold") is not None:
            th[k] = c["threshold"]
    parts = [("ood_real.csv", "real", "Real WhatsApp voice notes from people outside both datasets"),
             ("ood_clones.csv", "fake", "XTTS-v2 clones of a team member's voice (scam lines), made from one WhatsApp voice note"),
             ("ood_clones_neutral.csv", "fake", "XTTS-v2 clones of the same voice saying everyday sentences")]
    parts += [(f.replace(".csv", "_v2.csv"), e, d + " (v2)") for f, e, d in parts]
    found = [(f, e, d) for f, e, d in parts if (RES / f).exists()]
    if not found:
        return
    lines += ["### Out-of-domain check: voices that are in no training data", "",
              "Recorded on ordinary phones and shared over WhatsApp. None of these speakers are in Kathbath or "
              "IndicSynth, and the clones were made from a real voice note, not studio audio.", ""]
    for fname, expect, desc in found:
        table, summary = _ood_table(RES / fname, expect, th)
        if table:
            lines += [f"**{desc}** (expected: {expect})", "", table, "", "; ".join(summary) + ".", ""]


def section_cost(lines):
    rows = []
    for key, label in MODELS:
        lat = load_json(RES / key / "latency.json")
        cal = load_json(RES / key / "calibration.json")
        if lat or cal:
            rows.append([label, f"{lat['cpu_seconds_per_4s_window']} s" if lat else "–",
                         f"{cal['threshold']:.3f}" if cal else "–", pct(cal.get("pooled_eer")) if cal else "–"])
    if rows:
        lines += ["### Cost and calibration", "",
                  md_table(["Model", "CPU time per 4 s window", "Decision threshold", "Validation clip EER"], rows), ""]


def update_readme(report: str) -> bool:
    """Replace the block between the RESULTS markers in README.md."""
    readme = ROOT / "README.md"
    if not readme.exists():
        return False
    text = readme.read_text(encoding="utf-8")
    start, end = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
    if start not in text or end not in text:
        return False
    # figure paths in the report are already repo-relative
    new = text[: text.index(start) + len(start)] + "\n" + report.strip() + "\n" + text[text.index(end):]
    readme.write_text(new, encoding="utf-8")
    return True


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    update = "--update-readme" in sys.argv
    lines = ["## Results", "",
             "_Generated by `scripts/make_report.py` from `results/`. Detector metric: equal error rate (EER), "
             "the point where the false-alarm rate equals the miss rate._", ""]
    for section in (section_headline, section_per_language, section_unseen_tool, section_ablation,
                    section_clone_signature, section_scam, section_ood, section_cost):
        try:
            section(lines)
        except Exception as e:  # one broken result file shouldn't sink the report
            lines += [f"<!-- {section.__name__} skipped: {type(e).__name__}: {e} -->", ""]
    out = RES / "REPORT.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}" + (f" and figures in {FIG}" if FIG.exists() else ""))
    if update:
        print("README.md results section updated" if update_readme("\n".join(lines))
              else "README.md has no RESULTS markers; not updated")


if __name__ == "__main__":
    sys.exit(main())
