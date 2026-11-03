"""HTTP API.

    AWAAZ_CKPT=runs/xlsr/best.pt uvicorn app.api:app --port 8000

POST /analyze   multipart file=<audio>          -> verdict JSON
POST /enroll    multipart name=<str>, file=<audio>
GET  /vault                                     -> enrolled names
DELETE /vault/{name}
GET  /health
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from app.service import get_awaaz

app = FastAPI(title="Awaaz", version="0.1.0",
              description="Voice-clone scam detection for Indian voice notes and calls.")

MAX_BYTES = 15 * 1024 * 1024  # a few minutes of compressed audio


async def _save_upload(f: UploadFile) -> Path:
    data = await f.read()
    if not data:
        raise HTTPException(400, "empty file")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "file too large (15 MB max)")
    suffix = Path(f.filename or "clip").suffix or ".bin"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(data)
    tmp.close()
    return Path(tmp.name)


@app.get("/health")
def health():
    aw = get_awaaz()
    return {"ok": True, "detector": aw.detector is not None, "speaker": aw.encoder is not None,
            "scam": aw.scam is not None, "enrolled": aw.vault.names()}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    path = await _save_upload(file)
    try:
        t0 = time.time()
        result = get_awaaz().analyze(path).to_dict()
        result["timings"]["total"] = round(time.time() - t0, 3)
        return result
    except RuntimeError as e:
        raise HTTPException(422, str(e))
    finally:
        path.unlink(missing_ok=True)  # never keep user audio


@app.post("/enroll")
async def enroll(name: str = Form(...), file: UploadFile = File(...)):
    name = name.strip()[:40]
    if not name:
        raise HTTPException(400, "name required")
    path = await _save_upload(file)
    try:
        get_awaaz().enroll(name, path)
        return {"enrolled": name, "people": get_awaaz().vault.names()}
    except RuntimeError as e:
        raise HTTPException(422, str(e))
    finally:
        path.unlink(missing_ok=True)


@app.get("/vault")
def vault():
    return {"people": get_awaaz().vault.names()}


@app.delete("/vault/{name}")
def forget(name: str):
    get_awaaz().vault.remove(name)
    return {"people": get_awaaz().vault.names()}
