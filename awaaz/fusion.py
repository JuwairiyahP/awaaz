"""Turn three scores into one verdict a frightened parent can act on.

| synthetic | matches enrolled | scam script | verdict |
| high      | yes              | any         | RED: likely a clone of <name>       |
| high      | no               | high        | RED: AI voice making a scam request |
| high      | no               | low         | AMBER: AI voice (could be a bot)    |
| low       | any              | high        | AMBER: human voice, scam script     |
| low       | any              | low         | GREEN: no warning signs found       |

Wording rules: never say "fake for sure" or "verified real". Red verdicts
always tell the user to call back on a saved number.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Thresholds:
    synthetic: float = 0.5      # detector probability; set from validation EER threshold
    synthetic_strong: float = 0.85
    speaker: float = 0.55       # ECAPA cosine; tune on same/different speaker pairs
    scam: float = 0.5


@dataclass
class Verdict:
    level: str                  # "red" | "amber" | "green"
    headline: str
    advice: str
    reasons: list[str] = field(default_factory=list)
    headline_hi: str = ""

    def to_dict(self):
        return asdict(self)


def decide(synthetic: float, speaker_name: str | None, speaker_sim: float, scam_score: float,
           scam_phrases: dict[str, list[str]] | None = None, th: Thresholds | None = None) -> Verdict:
    th = th or Thresholds()
    is_synth = synthetic >= th.synthetic
    is_match = speaker_name is not None and speaker_sim >= th.speaker
    is_scam = scam_score >= th.scam

    reasons = [f"Synthetic-voice score {synthetic:.2f} (flag at {th.synthetic:.2f})"]
    if speaker_name is not None:
        reasons.append(f"Voice matches {speaker_name} (similarity {speaker_sim:.2f})")
    if is_scam and scam_phrases:
        hits = [f"{cue}: " + ", ".join(p[:3]) for cue, p in scam_phrases.items() if p]
        if hits:
            reasons.append("Scam cues — " + "; ".join(hits))
    elif is_scam:
        reasons.append(f"Scam-script score {scam_score:.2f}")

    call_back = "Hang up and call them back on the number saved in your phone. Don't send money or share an OTP until you've spoken to them directly."

    if is_synth and is_match:
        return Verdict("red", f"Likely a cloned voice of {speaker_name}", call_back, reasons,
                       headline_hi=f"यह {speaker_name} की नकली (AI) आवाज़ हो सकती है")
    if is_synth and is_scam:
        return Verdict("red", "AI-generated voice making a scam-style request", call_back, reasons,
                       headline_hi="AI आवाज़, धोखाधड़ी जैसी माँग")
    if is_synth:
        strength = "strongly " if synthetic >= th.synthetic_strong else ""
        return Verdict("amber", f"This voice {strength}looks AI-generated",
                       "It may be an automated bot. Don't share OTPs or pay anyone based on this message alone.",
                       reasons, headline_hi="यह आवाज़ AI से बनी लगती है")
    if is_scam:
        return Verdict("amber", "Sounds human, but follows a known scam script",
                       "Real people run these scams too. Verify through a number you already have before acting.",
                       reasons, headline_hi="आवाज़ असली लगती है, पर बातें धोखाधड़ी जैसी हैं")
    return Verdict("green", "No warning signs found",
                   "This doesn't prove the message is genuine. If anyone asks for money urgently, still call back to check.",
                   reasons, headline_hi="कोई चेतावनी संकेत नहीं मिला")
