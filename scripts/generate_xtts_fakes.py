"""Fallback data: clone real Indian voices with XTTS-v2 (run on a GPU, e.g. Kaggle).

Use this if IndicSynth is too slow to download, or to add a cloning tool the
detector has not seen (a fairer test). It also makes the consented demo clips.

XTTS-v2 speaks Hindi but no other Indian language, so this produces Hindi
clones. Each real clip is used as the voice reference for one generated clip,
which mimics the scam setup: a few seconds of someone's real voice in, a
clone saying something they never said out.

    pip install coqui-tts
    python scripts/generate_xtts_fakes.py \
        --refs /kaggle/input/real-hindi --out /kaggle/working/xtts_hi --n 1500

XTTS-v2 is released under the Coqui Public Model License (non-commercial);
the clips are for research and evaluation only.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from awaaz.audio import AUDIO_EXTENSIONS  # noqa: E402

# Everyday sentences, so the detector learns the voice, not scam words.
NEUTRAL_HI = [
    "मैं आज शाम को घर थोड़ा देर से पहुँचूँगा।",
    "कल सुबह की मीटिंग दस बजे से शुरू होगी।",
    "बाज़ार से आते हुए दूध और ब्रेड ले आना।",
    "बारिश की वजह से आज ट्रैफ़िक बहुत ज़्यादा है।",
    "मैंने तुम्हें फोटो भेज दी है, एक बार देख लेना।",
    "रविवार को हम सब मामा जी के घर चलेंगे।",
    "परीक्षा के नतीजे अगले हफ़्ते आने वाले हैं।",
    "डॉक्टर ने कहा है कि रोज़ थोड़ा टहलना चाहिए।",
    "ट्रेन पंद्रह मिनट लेट चल रही है।",
    "खाना बहुत अच्छा बना था, धन्यवाद।",
    "मेरा फोन चार्ज नहीं हो रहा, शायद केबल खराब है।",
    "आज ऑफिस में बहुत काम था, मैं थक गया हूँ।",
    "दीवाली की छुट्टियों में हम गाँव जाएँगे।",
    "उसका जन्मदिन अगले महीने की पाँच तारीख को है।",
    "नया फ्लैट स्टेशन के बिलकुल पास है।",
    "मैं पहुँच कर तुम्हें फोन करता हूँ।",
    "इस बार क्रिकेट मैच देखने ज़रूर चलेंगे।",
    "बच्चों की स्कूल बस आज जल्दी आ गई।",
    "दादी की तबीयत अब पहले से बेहतर है।",
    "मुझे वो किताब लाइब्रेरी में मिल गई।",
]

# Scam-style lines, used only for the demo and the scam-script test set.
SCAM_HI = [
    "भैया का एक्सीडेंट हो गया है, जल्दी से पैसे भेज दो, घर पे किसी को मत बताना।",
    "मम्मी, मैं पुलिस स्टेशन में हूँ, अभी पचास हज़ार यूपीआई कर दो।",
    "पापा, मेरा फोन खो गया है, इस नंबर पर तुरंत पैसे भेज दो।",
    "आपके नाम पर केस दर्ज हुआ है, डिजिटल अरेस्ट से बचने के लिए अभी भुगतान करें।",
    "जो ओटीपी आया है वो मुझे बता दो, बहुत ज़रूरी है।",
]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", required=True, help="folder of real Hindi clips used as voice references")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--scam", action="store_true", help="use scam lines (demo clips) instead of neutral ones")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)

    import torch
    from TTS.api import TTS

    rng = random.Random(a.seed)
    refs = sorted(f for f in Path(a.refs).rglob("*") if f.suffix.lower() in AUDIO_EXTENSIONS)
    if not refs:
        raise SystemExit(f"no reference audio in {a.refs}")
    rng.shuffle(refs)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    lines = SCAM_HI if a.scam else NEUTRAL_HI

    made = 0
    for i in range(a.n):
        ref = refs[i % len(refs)]
        text = rng.choice(lines)
        dst = out / f"xtts_{ref.parent.name}_{ref.stem}_{i:05d}.wav"
        if dst.exists():
            made += 1
            continue
        try:
            tts.tts_to_file(text=text, speaker_wav=str(ref), language="hi", file_path=str(dst))
            made += 1
        except Exception as e:  # one bad reference clip should not stop the run
            print(f"skip {ref.name}: {e}")
        if i % 50 == 0:
            print(f"{i}/{a.n} generated")
    print(f"done: {made} clips in {out}")


if __name__ == "__main__":
    main()
