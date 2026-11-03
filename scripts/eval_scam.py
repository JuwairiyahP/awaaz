"""Evaluate the scam-script scorer on a labelled text set.

    python scripts/eval_scam.py                 # lexicon only (instant)
    python scripts/eval_scam.py --nli           # lexicon + multilingual zero-shot NLI (downloads mDeBERTa)

The default set (eval/scam_texts.csv) has 30 scam and 30 normal messages in
Hindi, Hinglish and English, written by the team. The normal half is
deliberately hard: it mentions money, hospitals, police, secrets and OTPs in
everyday contexts. It is small and self-written, so treat it as a sanity
check, not a benchmark. Hindi/Hinglish NLI needs an English translation; with
--nli the English lines are scored directly and the others use the lexicon.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from awaaz.metrics import binary_report  # noqa: E402
from awaaz.scam import ScamScorer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def run(path: Path, use_nli: bool):
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    scorer = ScamScorer(use_nli=use_nli)
    for r in rows:
        translation = r["text"] if (use_nli and r["lang"] == "en") else ""
        res = scorer.score_text(r["text"], translation)
        r["score"], r["flagged"] = res.score, int(res.flagged)
        r["cues"] = {k: v for k, v in res.cues.items() if v >= 0.5}
    return rows


def report(rows):
    y = [int(r["label"]) for r in rows]
    s = [r["score"] for r in rows]
    overall = binary_report(s, y, 0.5)
    per_lang = {}
    groups = defaultdict(list)
    for r in rows:
        groups[r["lang"]].append(r)
    for lang, g in sorted(groups.items()):
        per_lang[lang] = binary_report([r["score"] for r in g], [int(r["label"]) for r in g], 0.5)
    errors = [{"text": r["text"], "label": int(r["label"]), "score": r["score"], "kind": r["kind"], "cues": r["cues"]}
              for r in rows if r["flagged"] != int(r["label"])]
    return overall, per_lang, errors


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "eval/scam_texts.csv"))
    ap.add_argument("--nli", action="store_true")
    ap.add_argument("--out", default="results/scam")
    a = ap.parse_args(argv)

    rows = run(Path(a.data), a.nli)
    overall, per_lang, errors = report(rows)
    summary = {"mode": "lexicon+nli" if a.nli else "lexicon", "n": len(rows),
               "overall": {k: round(v, 3) if isinstance(v, float) else v for k, v in overall.items()},
               "per_language": {l: {k: round(v, 3) if isinstance(v, float) else v for k, v in m.items()}
                                for l, m in per_lang.items()},
               "errors": errors}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"summary_{summary['mode']}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    o = summary["overall"]
    print(f"{summary['mode']}: precision {o['precision']:.2f}  recall {o['recall']:.2f}  F1 {o['f1']:.2f}  "
          f"(tp {o['tp']} fp {o['fp']} fn {o['fn']} tn {o['tn']})")
    for l, m in summary["per_language"].items():
        print(f"  {l:9s} P {m['precision']:.2f} R {m['recall']:.2f} F1 {m['f1']:.2f}")
    if errors:
        print("\nmistakes:")
        for e in errors:
            print(f"  [{'missed scam' if e['label'] else 'false alarm'}] {e['text']}  (score {e['score']}, cues {e['cues']})")


if __name__ == "__main__":
    main()
