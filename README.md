# Awaaz — voice-clone scam detection for India

**Neural Storm 2026 · Track: Future Tech · Problem statement: FT-02 Deepfake & Synthetic Media Detection**

**Team:** _TODO team name_ · _TODO member names_

> Forward a suspicious voice note. Awaaz tells you whether it's AI-generated, whether it sounds like someone you know, and whether it follows a scam script — in Indian languages, on compressed WhatsApp and phone audio.

**Live demo:** _TODO Hugging Face Spaces link_ · **Demo video:** _TODO_

## The problem

_TODO: 3–4 lines with the McAfee India stats, the Indore cases, the election clips (see proposal doc)._

## What's new

Existing tools answer "AI or human?". Research shows voice clones also fool speaker-verification
systems — so a voice that **matches your relative _and_ carries synthetic artifacts** is the
fingerprint of a targeted clone attack. Awaaz checks three independent signals:

| Check | Model | Question |
| --- | --- | --- |
| A. Synthetic voice | XLS-R (frozen) + attentive pooling head; log-mel ResNet baseline | Does it sound machine-made? |
| B. Speaker match | SpeechBrain ECAPA-TDNN vs enrolled family voices | Who does it sound like? |
| C. Scam script | faster-whisper + Hindi/Hinglish/English lexicon + mDeBERTa zero-shot NLI | Urgency + secrecy + money/OTP + authority? |

Verdicts: red (likely clone of an enrolled person, or AI + scam request), amber (AI voice, or
human voice with a scam script), green (no warning signs — never "verified real").

_TODO: architecture diagram + screenshots / GIF_

## Results

_TODO: language × channel EER grid from `results/*/grid.csv`, seen vs unseen languages, XLS-R vs ResNet, clone-signature result, scam-script precision/recall, CPU latency._

## Repository layout

```
awaaz/        core package: audio, augment (channel simulation), detector, speaker, scam, fusion, pipeline
scripts/      build_manifest, generate_xtts_fakes, train_detector, eval_grid
app/          FastAPI service, Telegram bot, Streamlit dashboard
deploy/       Hugging Face Spaces config
tests/        33 tests (pytest)
KAGGLE.md     step-by-step GPU training runbook
```

## Setup

```bash
sudo apt install ffmpeg          # or brew install ffmpeg
pip install -r requirements.txt
python -m pytest -q tests
```

## Run locally

```bash
# enroll a real voice, then analyse a clip
python -m awaaz.pipeline mum_real.ogg --ckpt runs/xlsr/best.pt --enroll Mum
python -m awaaz.pipeline suspicious.ogg --ckpt runs/xlsr/best.pt
```

```bash
# dashboard
AWAAZ_CKPT=runs/xlsr/best.pt streamlit run app/streamlit_app.py

# HTTP API (POST /analyze, POST /enroll, GET /vault)
AWAAZ_CKPT=runs/xlsr/best.pt uvicorn app.api:app --port 8000

# Telegram bot (token from @BotFather)
TELEGRAM_TOKEN=... AWAAZ_CKPT=runs/xlsr/best.pt python -m app.bot
```

Training: see [KAGGLE.md](KAGGLE.md). Deployment: see [deploy/hf_space/DEPLOY.md](deploy/hf_space/DEPLOY.md).

## Limitations and responsible use

- A warning system, not proof. Red verdicts always advise calling back on a saved number.
- New cloning tools may evade the detector; the scam-script check is a second line.
- Enrolled voices are stored only as embeddings on your own deployment; no raw audio is kept.
- Demo clones were made only from a consenting team member's voice.

## Datasets and models

IndicSynth (Sharma et al., ACL 2025), IndicSuperb (AI4Bharat), ASVspoof 2019 LA, XLS-R (Meta),
ECAPA-TDNN (SpeechBrain), Whisper via faster-whisper, mDeBERTa-v3 XNLI, XTTS-v2 (Coqui, non-commercial, test data only).

## AI tools used

Claude (Anthropic) helped draft the proposal, scaffold code and write tests. _TODO: list any others._
