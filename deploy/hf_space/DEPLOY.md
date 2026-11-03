# Deploying the demo to Hugging Face Spaces (Gradio + ZeroGPU, free)

1. Create the Space: huggingface.co/new-space → SDK **Gradio** → **Blank** → hardware **ZeroGPU** → Public.
2. Clone it and copy in, at the Space repo root:
   - `awaaz/` and `app/` from the main repo
   - this folder's `README.md`, `requirements.txt` and `packages.txt` (ffmpeg)
   - optionally `results/` from the eval runs (shown in the "How well it works" tab)
3. Space **Settings → Variables and secrets**:
   - variable `AWAAZ_CKPT=hub:runs/xlsr/best.pt` (downloaded from the model repo at startup)
   - variable `AWAAZ_HUB_REPO=<hf-username>/awaaz-models`
   - secret `HF_TOKEN` (read access is enough; the model repo is private)
   - optionally `AWAAZ_WHISPER=base` if responses feel slow
4. Push. The first boot downloads XLS-R, Whisper and ECAPA weights (a few minutes).

Only the detector's forward pass runs on the GPU (`@spaces.GPU`, 20 s budget), so each
check uses a second or two of the free ZeroGPU quota. Enrolled voices live in the
visitor's browser session only.
