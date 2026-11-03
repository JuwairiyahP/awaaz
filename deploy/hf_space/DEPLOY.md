# Deploying to Hugging Face Spaces

1. Create a Space: huggingface.co/new-space, SDK **Streamlit**, hardware **CPU basic** (free).
2. Copy into the Space repo root:
   - this folder's `README.md` and `packages.txt`
   - `awaaz/`, `app/`, `requirements.txt` from the main repo
   - your checkpoint as `models/best.pt` (track with git lfs: `git lfs track "*.pt"`)
   - `results/` from the eval runs (for the "How well it works" tab)
3. In the Space **Settings → Variables**, set:
   - `AWAAZ_CKPT=models/best.pt`
   - `AWAAZ_WHISPER=small` (use `base` if the free CPU is too slow)
4. Push. First boot downloads XLS-R / Whisper / ECAPA, so allow a few minutes.
