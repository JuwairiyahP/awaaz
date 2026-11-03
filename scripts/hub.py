"""Keep trained models and results safe in a private Hugging Face model repo.

Kaggle wipes /kaggle/working when a session ends, so every finished stage
pushes its artifacts here, and a fresh session pulls them back before
resuming. The Hugging Face Space downloads its model from the same repo.

    export AWAAZ_HUB_REPO=<hf-username>/awaaz-models    # needs HF_TOKEN with write access
    python scripts/hub.py push runs/xlsr results/xlsr
    python scripts/hub.py pull                            # everything, into the repo root
    python scripts/hub.py pull --only runs/xlsr/best.pt   # just the model (what the Space does)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# what's worth keeping: checkpoints (no last.pt/.bak), training logs, all results
KEEP = ["runs/*/best.pt", "runs/*/log.json", "results/**"]


def repo_id() -> str:
    r = os.environ.get("AWAAZ_HUB_REPO")
    if not r:
        raise SystemExit("set AWAAZ_HUB_REPO=<hf-username>/awaaz-models")
    return r


def push(paths: list[str]):
    from huggingface_hub import HfApi

    api = HfApi()
    rid = repo_id()
    api.create_repo(rid, repo_type="model", private=True, exist_ok=True)
    for p in paths:
        src = ROOT / p
        if not src.exists():
            print(f"skip {p} (missing)")
            continue
        if src.is_dir():
            api.upload_folder(folder_path=str(src), path_in_repo=p, repo_id=rid, repo_type="model",
                              ignore_patterns=["*.bak", "last.pt", "*.tmp", "__pycache__/*"],
                              commit_message=f"update {p}")
        else:
            api.upload_file(path_or_fileobj=str(src), path_in_repo=p, repo_id=rid, repo_type="model",
                            commit_message=f"update {p}")
        print(f"pushed {p} -> {rid}")


def pull(only: list[str] | None):
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import RepositoryNotFoundError

    try:
        snapshot_download(repo_id(), repo_type="model", local_dir=str(ROOT), allow_patterns=only or KEEP)
        print(f"pulled {only or KEEP} from {repo_id()}")
    except RepositoryNotFoundError:
        print(f"{repo_id()} doesn't exist yet: nothing to pull")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("push")
    p.add_argument("paths", nargs="+")
    q = sub.add_parser("pull")
    q.add_argument("--only", nargs="*")
    a = ap.parse_args(argv)
    try:
        push(a.paths) if a.cmd == "push" else pull(a.only)
    except SystemExit:
        raise
    except Exception as e:  # never let a sync hiccup kill the pipeline
        print(f"hub {a.cmd} failed: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(0 if a.cmd == "pull" else 1)


if __name__ == "__main__":
    main()
