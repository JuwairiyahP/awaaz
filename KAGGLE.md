# Running Awaaz on Kaggle (GPU)

Settings: **Accelerator GPU T4 x2 or P100**, **Internet on**. Upload this repo as a
Kaggle dataset (or `git clone` your GitHub repo), then in a notebook:

```bash
!cp -r /kaggle/input/awaaz-code /kaggle/working/awaaz   # or: git clone <your repo>
%cd /kaggle/working/awaaz
!pip install -q av speechbrain faster-whisper sentencepiece
!python -m pytest -q tests        # 28 tests, ~15 s — confirms the env is fine
```

## 1. Data (first hour — decide fast)

Check each source in this order and stop as soon as you have real + fake audio in a few languages.

| Source | What it gives | Where to look |
| --- | --- | --- |
| IndicSynth (ACL 2025) | Fake speech, 12 Indian languages | github.com/vdivyas/IndicSynth (links to the download), IndiaAI AIKosh |
| IndicSuperb | Real speech for the same languages | AI4Bharat (IndicSynth was built from it) |
| ASVspoof 2019 LA | English real + fake | Search Kaggle datasets for "ASVspoof 2019" |
| Common Voice Hindi | Real Hindi, also XTTS voice references | Search Kaggle/HF datasets for "common voice hindi" |

Grab a **subset**: 2–3k clips per (language, real/fake) is plenty for 33 hours.
Target languages: train hi, kn, ta, bn, mr; hold out te, ml.

**Fallback at 12:30 if IndicSynth isn't flowing:** make Hindi clones yourself (GPU, ~1–2 s per clip):

```bash
!pip install -q coqui-tts
!python scripts/generate_xtts_fakes.py --refs /kaggle/input/<real-hindi-folder> --out /kaggle/working/xtts_hi --n 1500
```

Then build the manifest (one `--source label:lang:folder` per folder):

```bash
!python scripts/build_manifest.py \
   --source fake:hi:/kaggle/input/indicsynth/hindi   --source real:hi:/kaggle/input/indicsuperb/hindi \
   --source fake:kn:/kaggle/input/indicsynth/kannada --source real:kn:/kaggle/input/indicsuperb/kannada \
   --source fake:te:/kaggle/input/indicsynth/telugu  --source real:te:/kaggle/input/indicsuperb/telugu \
   --holdout te,ml --max-per-group 3000 --out data/manifest.csv
```

It prints a split table; every row needs both real and fake counts.

## 2. Train

```bash
# baseline first (~20–40 min on T4) — gives you a number early
!python scripts/train_detector.py --model specresnet --epochs 12 --batch 48 --out runs/specresnet

# main model: frozen XLS-R + head
!python scripts/train_detector.py --model xlsr --epochs 6 --batch 24 --lr 1e-3 --workers 4 --out runs/xlsr
```

First epoch is slower: it decodes every file into `/kaggle/working/cache`. Later epochs reuse it.
If XLS-R runs out of memory, drop `--batch` to 12.

## 3. Evaluate (the README headline)

```bash
!python scripts/eval_grid.py --ckpt runs/specresnet/best.pt --out results/specresnet
!python scripts/eval_grid.py --ckpt runs/xlsr/best.pt --out results/xlsr
```

Each writes `grid.csv` (language × channel EER), `summary.json`, `latency.json`.

## 4. Bring back

Download `runs/*/best.pt`, `results/`, and `runs/*/log.json`. A frozen-XLS-R checkpoint is
small (head only); the backbone re-downloads from Hugging Face at inference.

## 5. Calibrate, then the headline experiments

```bash
# decision threshold from validation, stored inside the checkpoint (the app reads it)
!python scripts/calibrate.py --ckpt runs/xlsr/best.pt --out-json results/xlsr/calibration.json
!python scripts/calibrate.py --ckpt runs/specresnet/best.pt --out-json results/specresnet/calibration.json

# clone signature: do clones fool speaker matching, and does match+synthetic catch them?
!python scripts/clone_signature.py --ckpt runs/xlsr/best.pt --split test --chain clean
!python scripts/clone_signature.py --ckpt runs/xlsr/best.pt --split test --chain whatsapp_opus16

# scam-script scorer on the labelled text set
!python scripts/eval_scam.py
!python scripts/eval_scam.py --nli
```
