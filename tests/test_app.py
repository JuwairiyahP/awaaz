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


def test_xlsr_detector_trains_with_layerdrop_config(monkeypatch):
    """XLS-R ships with layerdrop>0; in train mode that used to return fewer hidden
    states than layer weights. Uses a tiny random 24-layer config (no download)."""
    transformers = pytest.importorskip("transformers")
    from transformers import Wav2Vec2Config, Wav2Vec2Model

    cfg = Wav2Vec2Config(hidden_size=32, num_hidden_layers=24, num_attention_heads=2, intermediate_size=64,
                         conv_dim=(16,) * 7, do_stable_layer_norm=True, feat_extract_norm="layer", layerdrop=0.1)
    monkeypatch.setattr(Wav2Vec2Model, "from_pretrained", classmethod(lambda cls, *a, **k: Wav2Vec2Model(cfg)))
    from awaaz.detector import XLSRDetector

    m = XLSRDetector()
    m.train()
    assert not m.backbone.training
    opt = torch.optim.Adam(m.trainable_parameters(), 1e-3)
    for _ in range(3):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(m(torch.randn(4, 16000)), torch.ones(4))
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert all(p.grad is None for p in m.backbone.parameters())


def test_gradio_demo_session_vault_and_clone_verdict(client, tmp_path):
    """Enrolled voices live in the session dict; a synthetic clip matching an
    enrolled voice gets the red 'cloned voice of <name>' verdict."""
    pytest.importorskip("gradio")
    import app.gradio_app as G

    class FakeEncoder:
        def embed(self, x):
            v = np.ones(192, dtype=np.float32)
            return v / np.linalg.norm(v)

    aw = G._aw()
    aw.encoder, aw.th.synthetic = FakeEncoder(), 0.0  # force "synthetic" with a random detector
    t = np.arange(16000 * 5) / 16000
    wav = tmp_path / "v.wav"
    save_wav(wav, (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32))
    people, md, _, _ = G.enroll("Bhaiya", str(wav), {})
    assert "Bhaiya" in md and "Bhaiya" in people
    v, fig, details, raw = G.analyze(str(wav), people)
    assert raw["verdict"]["level"] == "red" and "Bhaiya" in raw["verdict"]["headline"]
    assert not Path(aw.vault.path or "/nonexistent").exists()  # nothing written server-side
    people, md, _ = G.forget("Bhaiya", people)
    assert people == {}
    aw.encoder, aw.th.synthetic = None, 0.5
