# Deploying the demo to Hugging Face Spaces (Gradio + ZeroGPU, free)

1. Create the Space: huggingface.co/new-space → SDK **Gradio** → **Blank** → hardware **ZeroGPU** → Public.
2. Clone it and copy in, at the Space repo root:
   - `awaaz/` and `app/` from the main repo
   - this folder's `README.md`, `requirements.txt` and `packages.txt` (ffmpeg)
   - the trained checkpoint as `models/best.pt`; it's tracked with Git LFS:
     `git lfs install; git lfs track "*.pt"; git add .gitattributes`
   - optionally `results/` from the eval runs (shown in the "How well it works" tab)
3. Space **Settings → Variables**: `AWAAZ_CKPT=models/best.pt`, `AWAAZ_WHISPER=small`
   (`base` if responses feel slow). NLI is off by default (it lowered scam-script F1 in our eval).
4. Push. The first boot downloads XLS-R, Whisper and ECAPA weights (a few minutes).

Only the detector's forward pass runs on the GPU (`@spaces.GPU`, 20 s budget), so each
check uses a second or two of the free ZeroGPU quota. Enrolled voices live in the
visitor's browser session only.
