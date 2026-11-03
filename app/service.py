"""One shared Awaaz instance for the API, the bot and the dashboard.

Configured by environment variables so the same code runs on a laptop,
Kaggle or Hugging Face Spaces:

  AWAAZ_CKPT          path to the detector checkpoint (required for a real verdict)
  AWAAZ_DEVICE        cpu | cuda                         (default cpu)
  AWAAZ_VAULT         enrolled-voice store               (default data/vault.json)
  AWAAZ_SPEAKER       1/0 enable speaker match           (default 1)
  AWAAZ_SCAM          1/0 enable scam-script scorer      (default 1)
  AWAAZ_NLI           1/0 enable zero-shot NLI           (default 1)
  AWAAZ_WHISPER       faster-whisper size                (default small)
  AWAAZ_TH_SYNTH      synthetic threshold override
  AWAAZ_TH_SPEAKER    speaker-similarity threshold override
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from awaaz.fusion import Thresholds  # noqa: E402
from awaaz.pipeline import Awaaz  # noqa: E402


def _flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() not in {"0", "false", "no", ""}


def thresholds_from_env() -> Thresholds:
    th = Thresholds()
    if os.environ.get("AWAAZ_TH_SYNTH"):
        th.synthetic = float(os.environ["AWAAZ_TH_SYNTH"])
    if os.environ.get("AWAAZ_TH_SPEAKER"):
        th.speaker = float(os.environ["AWAAZ_TH_SPEAKER"])
    return th


@lru_cache(maxsize=1)
def get_awaaz() -> Awaaz:
    return Awaaz(
        ckpt=os.environ.get("AWAAZ_CKPT") or None,
        vault_path=os.environ.get("AWAAZ_VAULT", "data/vault.json"),
        device=os.environ.get("AWAAZ_DEVICE", "cpu"),
        thresholds=thresholds_from_env(),
        use_speaker=_flag("AWAAZ_SPEAKER"),
        use_scam=_flag("AWAAZ_SCAM"),
        use_nli=_flag("AWAAZ_NLI"),
        whisper_size=os.environ.get("AWAAZ_WHISPER", "small"),
    )


LEVEL_EMOJI = {"red": "🔴", "amber": "🟠", "green": "🟢"}


def telegram_text(result: dict) -> str:
    """Plain-text verdict for chat apps (Telegram/WhatsApp-style)."""
    v = result["verdict"]
    lines = [f"{LEVEL_EMOJI.get(v['level'], '')} {v['headline']}"]
    if v.get("headline_hi"):
        lines.append(v["headline_hi"])
    lines += ["", v["advice"], ""]
    lines.append(f"• Synthetic-voice score: {result['synthetic']:.2f}")
    sp = result.get("speaker") or {}
    if sp.get("scores"):
        best = max(sp["scores"], key=sp["scores"].get)
        tag = "match" if sp.get("name") else "closest, below match threshold"
        lines.append(f"• Sounds like: {best} ({sp['scores'][best]:.2f}, {tag})")
    sc = result.get("scam") or {}
    hits = [f"{cue}: {', '.join(p[:3])}" for cue, p in (sc.get("phrases") or {}).items() if p]
    if hits:
        lines.append("• Scam cues: " + "; ".join(hits))
    ws = result.get("window_scores") or []
    if len(ws) > 1:
        bars = "".join("▁▂▃▄▅▆▇█"[min(7, int(s * 8))] for s in ws)
        lines.append(f"• Timeline (2 s steps): {bars}")
    if sc.get("transcript"):
        t = sc["transcript"]
        lines += ["", f"Heard: “{t[:300]}{'…' if len(t) > 300 else ''}”"]
    lines += ["", "Awaaz is a warning tool, not proof. When in doubt, call back on a number you already have."]
    return "\n".join(lines)
