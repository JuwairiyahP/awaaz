"""Scam-script scorer: does the message follow the voice-clone scam pattern?

Independent of acoustics, so it still works when compression has erased
synthesis artifacts, and it also catches human scammers.

Cues (from how Indian voice-clone and "digital arrest" scams are described):
  urgency   - emergency, accident, right now, quickly
  secrecy   - don't tell anyone, keep it between us
  money     - send money, UPI, transfer, account, amount
  otp       - asking for an OTP / PIN / code
  authority - police, arrest, case, court, customs, CBI

Two scorers combined:
  1. a phrase lexicon in Devanagari Hindi, romanised Hinglish and English
     (always available, instant, explainable)
  2. optional multilingual zero-shot NLI on the English translation.
     OFF by default: on our 60-message eval set it caught one more scam but
     raised 10 false alarms on everyday messages (F1 0.93 -> 0.83), because
     multi-label zero-shot scores nearly every cue high for short texts.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

CUES = ["urgency", "secrecy", "money", "otp", "authority"]

LEXICON: dict[str, list[str]] = {
    "urgency": [
        # English
        r"\burgent(ly)?\b", r"\bemergency\b", r"\bright now\b", r"\bimmediately\b", r"\bquick(ly)?\b",
        r"\bhurry\b", r"\baccident\b", r"\bhospital\b", r"\bas soon as possible\b", r"\basap\b",
        # Hinglish
        r"\bjaldi\b", r"\babhi\b", r"\bturant\b", r"\bfauran\b", r"\bekdum\b", r"\baccident ho gaya\b",
        # Hindi
        r"जल्दी", r"अभी", r"तुरंत", r"फ़ौरन|फौरन", r"एक्सीडेंट", r"हादसा", r"अस्पताल|हॉस्पिटल",
    ],
    "secrecy": [
        r"\bdon'?t tell\b", r"\bdo not tell\b", r"\bkeep (it|this) (a )?secret\b", r"\bbetween us\b",
        r"\bdon'?t inform\b", r"\bno one should know\b",
        r"\bmat batana\b", r"\bkisi ko (mat|na) bata", r"\bkisiko (mat|na) bata", r"\bghar (pe|par|mein) mat bata",
        r"\bsecret rakh",
        r"मत बताना", r"किसी को (मत|ना) बता", r"किसी से (मत|ना) कह", r"घर (पे|पर|में) (किसी को )?मत बता", r"राज़ रख|राज रख",
    ],
    "money": [
        r"\bsend (me )?(the )?money\b", r"\btransfer\b", r"\bupi\b", r"\bgpay\b", r"\bgoogle pay\b", r"\bphonepe\b",
        r"\bpaytm\b", r"\bbank account\b", r"\baccount number\b", r"\brupees\b", r"\bpay(ment)?\b", r"\bfees?\b",
        r"\bpaise\b", r"\bpaisa\b", r"\bpaise bhej", r"\bbhej do\b", r"\bupi kar", r"\bhazaar\b", r"\blakh\b",
        r"पैसे|पैसा", r"यूपीआई", r"ट्रांसफर", r"भेज दो|भेजो", r"हज़ार|हजार", r"लाख", r"भुगतान", r"खाते",
    ],
    "otp": [
        r"\botp\b", r"\bone[- ]time password\b", r"\bpin\b", r"\bverification code\b", r"\bcvv\b",
        r"ओटीपी", r"पिन", r"कोड बता",
    ],
    "authority": [
        r"\bpolice\b", r"\barrest(ed)?\b", r"\bdigital arrest\b", r"\bcase (is )?(filed|registered)\b", r"\bcourt\b",
        r"\bcbi\b", r"\bcustoms\b", r"\bnarcotics\b", r"\bwarrant\b", r"\bfir\b", r"\bjail\b",
        r"\bthana\b", r"\bkes darj\b",
        r"पुलिस", r"अरेस्ट|गिरफ़्तार|गिरफ्तार", r"केस", r"थाना|थाने", r"कोर्ट|अदालत", r"जेल", r"सीबीआई", r"कस्टम",
    ],
}

_COMPILED = {cue: [re.compile(p, re.IGNORECASE) for p in pats] for cue, pats in LEXICON.items()}

NLI_HYPOTHESES = {
    "urgency": "The speaker is describing an emergency and demands immediate action.",
    "secrecy": "The speaker asks the listener to keep this secret from others.",
    "money": "The speaker asks the listener to send money or make a payment.",
    "otp": "The speaker asks for a one-time password, PIN or verification code.",
    "authority": "The speaker claims to be police or an official, or threatens arrest.",
}


@dataclass
class ScamResult:
    score: float                     # 0..1 overall scam-pattern likelihood
    cues: dict[str, float]           # per-cue 0..1
    phrases: dict[str, list[str]]    # matched phrases per cue, for the explanation
    transcript: str = ""
    translation: str = ""
    language: str = ""
    sources: list[str] = field(default_factory=list)

    @property
    def flagged(self) -> bool:
        return self.score >= 0.5


def lexicon_cues(text: str) -> tuple[dict[str, float], dict[str, list[str]]]:
    cues, phrases = {}, {}
    for cue, pats in _COMPILED.items():
        hits = []
        for p in pats:
            hits += [m.group(0) for m in p.finditer(text)]
        phrases[cue] = sorted(set(h.strip() for h in hits))
        cues[cue] = 0.0 if not hits else min(1.0, 0.75 + 0.1 * (len(set(hits)) - 1))
    return cues, phrases


def combine(cues: dict[str, float]) -> float:
    """Scam pattern = a payment/credential ask PLUS pressure (urgency,
    secrecy or authority). Either alone is common in normal chat
    ("send me the money for dinner", "come quickly"), so neither alone
    should push the score past 0.5."""
    ask = max(cues.get("money", 0), cues.get("otp", 0))
    pressure = max(cues.get("urgency", 0), cues.get("secrecy", 0), cues.get("authority", 0))
    n_pressure = sum(cues.get(c, 0) >= 0.5 for c in ("urgency", "secrecy", "authority"))
    score = ask * pressure  # both needed
    score += 0.1 * max(0, n_pressure - 1) * ask  # several pressure tactics at once
    score = max(score, 0.45 * ask, 0.35 * pressure)  # one side alone stays below 0.5
    if cues.get("otp", 0) >= 0.5:
        score = max(score, 0.6)  # nobody legitimate asks for an OTP by voice note
    return round(min(1.0, score), 4)


class ScamScorer:
    """Lexicon always; Whisper and NLI loaded lazily if available."""

    def __init__(self, use_nli: bool = False, whisper_size: str = "small", device: str = "cpu"):
        self.use_nli, self.whisper_size, self.device = use_nli, whisper_size, device
        self._whisper = None
        self._nli = None

    # -- speech to text
    def transcribe(self, audio, sr: int = 16000) -> tuple[str, str, str]:
        """Returns (transcript in original language, English translation, language code)."""
        if self._whisper is None:
            from faster_whisper import WhisperModel

            compute = "float16" if self.device == "cuda" else "int8"
            self._whisper = WhisperModel(self.whisper_size, device=self.device, compute_type=compute)
        segs, info = self._whisper.transcribe(audio, beam_size=1, vad_filter=True)
        text = " ".join(s.text.strip() for s in segs)
        if info.language == "en":
            return text, text, "en"
        tsegs, _ = self._whisper.transcribe(audio, task="translate", beam_size=1, vad_filter=True)
        return text, " ".join(s.text.strip() for s in tsegs), info.language

    # -- text to cues
    def nli_cues(self, english: str) -> dict[str, float]:
        if not english.strip():
            return {c: 0.0 for c in CUES}
        if self._nli is None:
            from transformers import pipeline

            self._nli = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
                                 device=0 if self.device == "cuda" else -1)
        out = self._nli(english, candidate_labels=list(NLI_HYPOTHESES.values()), multi_label=True,
                        hypothesis_template="{}")
        by_label = dict(zip(out["labels"], out["scores"]))
        return {c: float(by_label[h]) for c, h in NLI_HYPOTHESES.items()}

    def score_text(self, transcript: str, translation: str = "") -> ScamResult:
        text = f"{transcript}\n{translation}"
        lex, phrases = lexicon_cues(text)
        cues, sources = dict(lex), ["lexicon"]
        if self.use_nli and translation:
            try:
                nli = self.nli_cues(translation)
                # NLI is softer evidence than an exact phrase: cap its weight
                cues = {c: max(lex[c], 0.9 * nli[c]) for c in CUES}
                sources.append("nli")
            except Exception as e:  # model download/runtime failure: fall back to lexicon
                sources.append(f"nli-unavailable: {type(e).__name__}")
        return ScamResult(combine(cues), {c: round(v, 3) for c, v in cues.items()}, phrases,
                          transcript=transcript, translation=translation, sources=sources)

    def score_audio(self, audio, sr: int = 16000) -> ScamResult:
        transcript, translation, lang = self.transcribe(audio, sr)
        r = self.score_text(transcript, translation)
        r.language = lang
        return r
