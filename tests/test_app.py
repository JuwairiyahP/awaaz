"""API and chat-format tests with a randomly initialised detector (no downloads)."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from awaaz.audio import save_wav  # noqa: E402
from awaaz.detector import SpecResNet  # noqa: E402


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    d = tmp_path_factory.mktemp("app")
    ckpt = d / "rand.pt"
    torch.save({"model": "specresnet", "kwargs": {}, "state_dict": SpecResNet().state_dict(), "epoch": 0}, ckpt)
    os.environ.update({"AWAAZ_CKPT": str(ckpt), "AWAAZ_SPEAKER": "0", "AWAAZ_SCAM": "0",
                       "AWAAZ_VAULT": str(d / "vault.json")})
    from app import service

    service.get_awaaz.cache_clear()
    from fastapi.testclient import TestClient

    from app.api import app

    return TestClient(app), d


def test_health(client):
    c, _ = client
    j = c.get("/health").json()
    assert j["ok"] and j["detector"] and not j["speaker"]


def test_analyze_returns_verdict_and_deletes_upload(client, tmp_path):
    c, _ = client
    t = np.arange(16000 * 6) / 16000
    wav = tmp_path / "clip.wav"
    save_wav(wav, (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32))
    before = set(Path("/tmp").glob("tmp*.wav"))
    r = c.post("/analyze", files={"file": ("clip.wav", wav.read_bytes(), "audio/wav")})
    assert r.status_code == 200
    j = r.json()
    assert j["verdict"]["level"] in {"red", "amber", "green"}
    assert 0 <= j["synthetic"] <= 1 and len(j["window_scores"]) == len(j["window_starts"]) >= 2
    assert set(Path("/tmp").glob("tmp*.wav")) <= before  # upload not kept


def test_bad_inputs(client):
    c, _ = client
    assert c.post("/analyze", files={"file": ("x.wav", b"not audio", "audio/wav")}).status_code == 422
    assert c.post("/analyze", files={"file": ("x.wav", b"", "audio/wav")}).status_code == 400


def test_telegram_text_mentions_clone_and_cues():
    from app.service import telegram_text

    res = {"verdict": {"level": "red", "headline": "Likely a cloned voice of Bhaiya", "headline_hi": "x",
                       "advice": "Call back."},
           "synthetic": 0.91, "window_scores": [0.2, 0.9, 0.95], "window_starts": [0, 2, 4],
           "speaker": {"name": "Bhaiya", "similarity": 0.8, "scores": {"Bhaiya": 0.8}},
           "scam": {"phrases": {"money": ["UPI"], "secrecy": ["mat batana"]}, "transcript": "jaldi UPI karo"}}
    t = telegram_text(res)
    assert "🔴" in t and "Bhaiya" in t and "UPI" in t and "not proof" in t
