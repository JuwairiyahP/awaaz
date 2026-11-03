#!/usr/bin/env bash
# Awaaz end-to-end pipeline for Kaggle (2 GPUs). Resumable: every stage is
# skipped if its output already exists, and every finished artifact is pushed
# to the Hugging Face model repo, so a wiped session just reruns this script.
#
#   export HF_TOKEN=...                       (write access)
#   export AWAAZ_HUB_REPO=<hf-user>/awaaz-models
#   bash scripts/run_all.sh
#
# Knobs (env): N_CLIPS=2500  XLSR_EPOCHS=4  RESNET_EPOCHS=10  GEN_EPOCHS=3
#              DATA=/kaggle/working/data   OOD=/kaggle/working/ood_real
set -uo pipefail
cd "$(dirname "$0")/.."

DATA=${DATA:-/kaggle/working/data}
OOD=${OOD:-/kaggle/working/ood_real}
N=${N_CLIPS:-2500}
XLSR_EPOCHS=${XLSR_EPOCHS:-4}
RESNET_EPOCHS=${RESNET_EPOCHS:-10}
GEN_EPOCHS=${GEN_EPOCHS:-3}
LANGS="hi:Hindi:hindi kn:Kannada:kannada ta:Tamil:tamil bn:Bengali:bengali mr:Marathi:marathi te:Telugu:telugu ml:Malayalam:malayalam"
SOURCES=""
for t in $LANGS; do c=${t%%:*}; SOURCES="$SOURCES --source real:$c:$DATA/real/$c --source fake:$c:$DATA/fake/$c"; done

say()  { echo "[$(date +%H:%M:%S)] ## $*"; }
push() { python scripts/hub.py push "$@" || say "WARNING: push of $* failed (will retry at the next stage)"; }
count(){ find "$1" -type f ! -name metadata.csv 2>/dev/null | wc -l; }

# ---------------------------------------------------------------- 0. resume
say "pulling any saved models/results from the hub"
python scripts/hub.py pull

# ---------------------------------------------------------------- 1. data
fetch_fake() {
  for t in $LANGS; do
    c=${t%%:*}; rest=${t#*:}; cfg=${rest%%:*}
    [ "$(count $DATA/fake/$c)" -ge "$N" ] && continue
    say "fake $cfg"
    python -u scripts/fetch_hf_subset.py --dataset vdivyasharma/IndicSynth --config "$cfg" --n "$N" \
      --shuffle-buffer 1000 --speaker-column "Target Speaker ID" --out "$DATA/fake/$c" | grep --line-buffered -E "done|Error" || true
  done
}
fetch_real() {
  split=$(python -c "from datasets import get_dataset_split_names as g; s=g('ai4bharat/Kathbath','hindi'); print(next((x for x in ['train','test','test_known'] if x in s), s[0]))")
  for t in $LANGS; do
    c=${t%%:*}; cfg=${t##*:}
    [ "$(count $DATA/real/$c)" -ge "$N" ] && continue
    say "real $cfg ($split)"
    python -u scripts/fetch_hf_subset.py --dataset ai4bharat/Kathbath --config "$cfg" --split "$split" --n "$N" \
      --shuffle-buffer 1000 --speaker-column fname --speaker-regex='-(\d+)-[mf]\.' --out "$DATA/real/$c" | grep --line-buffered -E "done|Error" || true
  done
}
say "STAGE 1: data"
fetch_fake & PF=$!
fetch_real & PR=$!
wait $PF $PR
for t in $LANGS; do c=${t%%:*}; echo "   $c real=$(count $DATA/real/$c) fake=$(count $DATA/fake/$c)"; done

# ---------------------------------------------------------------- 2. manifests
say "STAGE 2: manifests"
python scripts/build_manifest.py $SOURCES --holdout te,ml --max-per-group "$N" --out data/manifest.csv | tail -14
python scripts/build_manifest.py $SOURCES --holdout te,ml --holdout-generator xtts_v2 --max-per-group "$N" \
  --out data/manifest_genholdout.csv > /dev/null

if [ "${ONLY_DATA:-0}" = "1" ]; then say "ONLY_DATA=1: data and manifests ready, stopping here"; exit 0; fi

# ---------------------------------------------------------------- 3+. per-GPU tracks
gpu0() {  # main model and the headline experiments
  export CUDA_VISIBLE_DEVICES=0
  if [ ! -f runs/xlsr/best.pt ]; then
    say "GPU0: train XLS-R ($XLSR_EPOCHS epochs)"
    python -u scripts/train_detector.py --model xlsr --epochs "$XLSR_EPOCHS" --batch 24 --workers 2 --out runs/xlsr \
      | grep --line-buffered -E "^epoch|best|Error" ; push runs/xlsr
  fi
  if [ ! -f runs/xlsr/best.pt ]; then say "GPU0: XLS-R training FAILED, skipping its evaluation (see errors above)"; return 1; fi
  if [ ! -f results/xlsr/calibration.json ]; then
    say "GPU0: calibrate XLS-R"
    python scripts/calibrate.py --ckpt runs/xlsr/best.pt --out-json results/xlsr/calibration.json | grep --line-buffered -E "EER|saved"
    push runs/xlsr results/xlsr
  fi
  if [ ! -f results/xlsr/summary.json ]; then
    say "GPU0: eval grid XLS-R"
    python scripts/eval_grid.py --ckpt runs/xlsr/best.pt --max-per-group 300 --out results/xlsr | grep --line-buffered -vE "it/s|LOAD|UNEXPECTED|^Key|^-|^Notes|can be ignored"
    push results/xlsr
  fi
  if [ ! -f results/xlsr/ablation/summary.json ]; then
    say "GPU0: shortcut ablation XLS-R"
    python scripts/eval_grid.py --ckpt runs/xlsr/best.pt --splits test --chains clean,dither,lowpass7k,sanitized \
      --max-per-group 150 --out results/xlsr/ablation | grep --line-buffered -E "test/|latency"
    push results/xlsr
  fi
  for ch in clean whatsapp_opus16; do
    if [ ! -f results/clone_signature/summary_$ch.json ]; then
      say "GPU0: clone signature ($ch)"
      python scripts/clone_signature.py --ckpt runs/xlsr/best.pt --split test --chain $ch | tail -32
      push results/clone_signature
    fi
  done
  say "GPU0 DONE"
}
gpu1() {  # baseline + unseen-cloning-tool ablation
  export CUDA_VISIBLE_DEVICES=1
  sleep 30  # let GPU0 start decoding first; both share the cache
  if [ ! -f runs/specresnet/best.pt ]; then
    say "GPU1: train SpecResNet ($RESNET_EPOCHS epochs)"
    python -u scripts/train_detector.py --model specresnet --epochs "$RESNET_EPOCHS" --batch 48 --workers 2 \
      --out runs/specresnet | grep --line-buffered -E "^epoch|best|Error" ; push runs/specresnet
  fi
  if [ -f runs/specresnet/best.pt ]; then
  if [ ! -f results/specresnet/calibration.json ]; then
    say "GPU1: calibrate SpecResNet"
    python scripts/calibrate.py --ckpt runs/specresnet/best.pt --out-json results/specresnet/calibration.json | grep --line-buffered -E "EER|saved"
    push runs/specresnet results/specresnet
  fi
  if [ ! -f results/specresnet/summary.json ]; then
    say "GPU1: eval grid SpecResNet"
    python scripts/eval_grid.py --ckpt runs/specresnet/best.pt --max-per-group 300 --out results/specresnet | grep --line-buffered -E "/|latency"
    push results/specresnet
  fi
  else say "GPU1: SpecResNet training FAILED, skipping its evaluation"; fi
  if [ ! -f runs/xlsr_genholdout/best.pt ]; then
    say "GPU1: train XLS-R without XTTS-v2 ($GEN_EPOCHS epochs)"
    python -u scripts/train_detector.py --model xlsr --manifest data/manifest_genholdout.csv --epochs "$GEN_EPOCHS" \
      --batch 24 --workers 2 --out runs/xlsr_genholdout | grep --line-buffered -E "^epoch|best|Error" ; push runs/xlsr_genholdout
  fi
  if [ ! -f runs/xlsr_genholdout/best.pt ]; then say "GPU1: generator-holdout training FAILED, skipping its eval"; return 1; fi
  if [ ! -f results/xlsr_genholdout/summary.json ]; then
    say "GPU1: unseen-tool eval"
    python scripts/calibrate.py --ckpt runs/xlsr_genholdout/best.pt --manifest data/manifest_genholdout.csv \
      --out-json results/xlsr_genholdout/calibration.json | grep --line-buffered -E "EER|saved"
    python scripts/eval_grid.py --ckpt runs/xlsr_genholdout/best.pt --manifest data/manifest_genholdout.csv \
      --splits test,test_unseen_gen --chains clean,whatsapp_opus16 --max-per-group 300 --out results/xlsr_genholdout | grep --line-buffered -E "/|latency"
    push runs/xlsr_genholdout results/xlsr_genholdout
  fi
  say "GPU1 DONE"
}
say "STAGE 3: training + evaluation on both GPUs"
gpu0 2>&1 | sed -u 's/^/[gpu0] /' &
G0=$!
gpu1 2>&1 | sed -u 's/^/[gpu1] /' &
G1=$!
wait $G0 $G1

# ---------------------------------------------------------------- 4. out-of-domain
if [ -d "$OOD" ] && [ "$(count $OOD)" -gt 0 ]; then
  say "STAGE 4: out-of-domain real voice notes"
  python scripts/score_folder.py "$OOD" --ckpt runs/xlsr/best.pt runs/specresnet/best.pt --expect real \
    --out results/ood_real.csv
  push results
fi
say "STAGE 5: scam-script eval + report"
python scripts/eval_scam.py | head -5
python scripts/make_report.py
push results docs

say "SUMMARY"
for f in runs/xlsr/best.pt runs/specresnet/best.pt runs/xlsr_genholdout/best.pt results/xlsr/summary.json \
         results/xlsr/ablation/summary.json results/specresnet/summary.json results/xlsr_genholdout/summary.json \
         results/clone_signature/summary_clean.json results/clone_signature/summary_whatsapp_opus16.json results/ood_real.csv; do
  [ -f "$f" ] && echo "   ok       $f" || echo "   MISSING  $f"
done
push runs results
say "ALL DONE (rerun this script to retry anything MISSING; finished stages are skipped)"
