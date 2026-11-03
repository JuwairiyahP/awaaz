"""Tests for everything that runs without downloaded models."""
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from awaaz.audio import load_audio, make_windows, save_wav, trim_silence  # noqa: E402
from awaaz.augment import CHAINS, apply_chain, random_channel  # noqa: E402
from awaaz.detector import SpecResNet, grad_cam  # noqa: E402
from awaaz.fusion import Thresholds, decide  # noqa: E402
from awaaz.metrics import auc, binary_report, eer  # noqa: E402
from awaaz.scam import ScamScorer, combine, lexicon_cues  # noqa: E402

SR = 16000


def tone(seconds=3.0, f=440.0):
    t = np.arange(int(SR * seconds)) / SR
    return (0.4 * np.sin(2 * np.pi * f * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t))).astype(np.float32)


# ------------------------------------------------------------------ audio
def test_load_roundtrip_any_format(tmp_path):
    x = tone(2.0)
    wav = tmp_path / "a.wav"
    save_wav(wav, x)
    ogg = tmp_path / "a.ogg"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(wav), "-c:a", "libopus", str(ogg)], check=True)
    for p in (wav, ogg):
        y = load_audio(p)
        assert abs(len(y) - len(x)) < 0.05 * SR


def test_trim_and_windows():
    x = np.concatenate([np.zeros(SR), tone(5), np.zeros(SR)])
    assert len(trim_silence(x)) < len(x)
    w, s = make_windows(tone(9))
    assert w.shape[1] == 4 * SR and s[0] == 0 and s[-1] + 4 <= 9.0001
    short, _ = make_windows(tone(1.0))
    assert short.shape == (1, 4 * SR)


# ------------------------------------------------------------------ augmentation
@pytest.mark.parametrize("name", list(CHAINS))
def test_chains_keep_length_and_are_finite(name):
    x = tone(4)
    y = apply_chain(x, name)
    assert y.shape == x.shape and np.isfinite(y).all()


def test_chains_change_the_signal():
    x = tone(4)
    for name in CHAINS:
        if name != "clean":
            assert not np.allclose(apply_chain(x, name), x, atol=1e-3), name


def test_codecs_work_in_process_and_via_cli(monkeypatch):
    """Both codec paths must actually encode (a silent failure means no augmentation)."""
    import awaaz.augment as A

    x = tone(3)
    for fn in (A.opus, A.mp3, A.aac):
        y_av = fn(x)
        assert y_av.shape == x.shape and not np.allclose(y_av, x, atol=1e-3), fn.__name__
    monkeypatch.setattr(A, "_av", None)  # force the ffmpeg CLI fallback
    for fn in (A.opus, A.mp3, A.aac):
        y_cli = fn(x)
        assert y_cli.shape == x.shape and not np.allclose(y_cli, x, atol=1e-3), fn.__name__


def test_eval_channel_cache(tmp_path):
    from awaaz.data import ClipDataset

    wav = tmp_path / "a.wav"
    save_wav(wav, tone(5))
    rows = [{"path": str(wav), "y": 1, "lang": "hi", "label": "fake"}]
    ds = ClipDataset(rows, train=False, chain="whatsapp_opus16", cache_dir=str(tmp_path / "cache"))
    a, _ = ds[0]
    assert len(list((tmp_path / "cache").glob("eval_*.npy"))) == 1
    b, _ = ds[0]
    assert torch.allclose(a, b, atol=1e-3)


def test_random_channel_shapes():
    rng = np.random.default_rng(0)
    for _ in range(10):
        y = random_channel(tone(4), rng)
        assert y.shape == (4 * SR,) and np.abs(y).max() <= 1.0001


# ------------------------------------------------------------------ metrics
def test_eer_perfect_and_random():
    y = np.array([0] * 50 + [1] * 50)
    assert eer(np.r_[np.zeros(50), np.ones(50)], y)[0] == pytest.approx(0.0, abs=1e-9)
    rng = np.random.default_rng(0)
    e, _ = eer(rng.random(4000), rng.integers(0, 2, 4000))
    assert 0.45 < e < 0.55
    assert auc(np.r_[np.zeros(50), np.ones(50)], y) == pytest.approx(1.0)


def test_eer_known_overlap():
    # real ~ N(0,1), fake ~ N(2,1): theoretical EER = Phi(-1) = 0.1587
    rng = np.random.default_rng(1)
    s = np.r_[rng.normal(0, 1, 20000), rng.normal(2, 1, 20000)]
    y = np.r_[np.zeros(20000), np.ones(20000)]
    assert eer(s, y)[0] == pytest.approx(0.1587, abs=0.01)
    assert binary_report(s, y, 1.0)["f1"] > 0.8


# ------------------------------------------------------------------ model
def test_specresnet_forward_backward_and_gradcam():
    m = SpecResNet()
    x = torch.randn(2, 4 * SR)
    out = m(x)
    assert out.shape == (2,)
    out.sum().backward()
    cam = grad_cam(m, torch.randn(1, 4 * SR))
    assert cam.ndim == 2 and cam.max() <= 1.0 + 1e-6


def test_specresnet_learns_a_toy_task():
    """Sanity check the training loop: separate tones from noise."""
    torch.manual_seed(0)
    m = SpecResNet(widths=(8, 16, 16, 32))
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    def batch(n=8):
        xs, ys = [], []
        for i in range(n):
            if i % 2:
                xs.append(torch.from_numpy(tone(1.0, f=float(np.random.uniform(300, 2000)))))
                ys.append(1.0)
            else:
                xs.append(0.3 * torch.randn(SR))
                ys.append(0.0)
        return torch.stack(xs), torch.tensor(ys)

    for _ in range(25):
        x, y = batch()
        loss = loss_fn(m(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    m.eval()
    x, y = batch(16)
    with torch.no_grad():
        acc = ((m(x) > 0).float() == y).float().mean().item()
    assert acc >= 0.9


# ------------------------------------------------------------------ scam scorer
SCAMS = [
    "Bhaiya ka accident ho gaya, jaldi UPI karo, ghar pe mat batana",
    "मम्मी, मैं पुलिस स्टेशन में हूँ, अभी पचास हज़ार यूपीआई कर दो।",
    "This is the police, a case is registered against you, pay the fine immediately to avoid digital arrest",
    "Please tell me the OTP you just received",
    "पापा जल्दी पैसे भेज दो, किसी को मत बताना",
]
NORMAL = [
    "I'll be home late tonight, start dinner without me",
    "Send me the money for the movie tickets whenever you can",
    "Come quickly, the match is starting!",
    "कल सुबह की मीटिंग दस बजे से शुरू होगी।",
    "Bhai kal cricket khelne chalein?",
]


@pytest.mark.parametrize("text", SCAMS)
def test_scams_flagged(text):
    r = ScamScorer(use_nli=False).score_text(text)
    assert r.flagged, (text, r.cues)


@pytest.mark.parametrize("text", NORMAL)
def test_normal_messages_not_flagged(text):
    r = ScamScorer(use_nli=False).score_text(text)
    assert not r.flagged, (text, r.cues)


def test_phrases_explain_the_flag():
    _, phrases = lexicon_cues("Bhaiya ka accident ho gaya, jaldi UPI karo, ghar pe mat batana")
    assert phrases["money"] and phrases["urgency"] and phrases["secrecy"]


def test_combine_needs_ask_and_pressure():
    assert combine({"money": 1.0}) < 0.5
    assert combine({"urgency": 1.0}) < 0.5
    assert combine({"money": 1.0, "urgency": 1.0}) >= 0.5


# ------------------------------------------------------------------ fusion
def test_verdict_matrix():
    th = Thresholds(synthetic=0.5, speaker=0.55, scam=0.5)
    assert decide(0.9, "Bhaiya", 0.8, 0.0, th=th).level == "red"           # clone of enrolled
    assert "Bhaiya" in decide(0.9, "Bhaiya", 0.8, 0.0, th=th).headline
    assert decide(0.9, None, 0.2, 0.9, th=th).level == "red"               # AI + scam
    assert decide(0.9, None, 0.2, 0.1, th=th).level == "amber"             # AI only
    assert decide(0.1, "Bhaiya", 0.9, 0.9, th=th).level == "amber"         # human scammer
    assert decide(0.1, None, 0.0, 0.1, th=th).level == "green"
    g = decide(0.1, None, 0.0, 0.1, th=th)
    assert "verified" not in g.headline.lower() and "real" not in g.headline.lower()


# ------------------------------------------------------------------ manifest
def test_build_manifest_splits(tmp_path):
    for label in ("real", "fake"):
        for lang in ("hi", "te"):
            for spk in range(12):
                d = tmp_path / label / lang / f"spk{spk:02d}"
                d.mkdir(parents=True)
                for k in range(3):
                    save_wav(d / f"{k}.wav", tone(0.5))
    out = tmp_path / "m.csv"
    srcs = sum([["--source", f"{l}:{g}:{tmp_path / l / g}"] for l in ("real", "fake") for g in ("hi", "te")], [])
    subprocess.run([sys.executable, str(ROOT / "scripts/build_manifest.py"), *srcs, "--holdout", "te",
                    "--out", str(out)], check=True, capture_output=True)
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 4 * 12 * 3
    assert all(r["split"] == "test_unseen" for r in rows if r["lang"] == "te")
    hi = [r for r in rows if r["lang"] == "hi"]
    assert {r["split"] for r in hi} == {"train", "val", "test"}
    # speaker-disjoint ACROSS classes: a voice sits in exactly one split, real or cloned
    by_spk = {}
    for r in hi:
        by_spk.setdefault(r["speaker"], set()).add(r["split"])
    assert all(len(s) == 1 for s in by_spk.values())
    for split in ("train", "val", "test"):
        assert {r["label"] for r in hi if r["split"] == split} == {"real", "fake"}


def test_build_manifest_generator_holdout(tmp_path):
    """fake clips tagged via metadata.csv; xtts_v2 held out -> test_unseen_gen with paired reals."""
    meta_rows = []
    for label in ("real", "fake"):
        for spk in range(12):
            d = tmp_path / label / f"spk_{spk}"
            d.mkdir(parents=True)
            for k in range(4):
                p = d / f"{k}.wav"
                save_wav(p, tone(0.5))
                if label == "fake":
                    meta_rows.append({"file": str(p), "Generative Model": "xtts_v2" if k % 2 else "freevc24"})
    with (tmp_path / "fake" / "metadata.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "Generative Model"])
        w.writeheader()
        w.writerows(meta_rows)
    out = tmp_path / "m.csv"
    subprocess.run([sys.executable, str(ROOT / "scripts/build_manifest.py"),
                    "--source", f"real:hi:{tmp_path / 'real'}", "--source", f"fake:hi:{tmp_path / 'fake'}",
                    "--holdout-generator", "xtts_v2", "--out", str(out)], check=True, capture_output=True)
    rows = list(csv.DictReader(out.open()))
    xtts = [r for r in rows if r["generator"] == "xtts_v2"]
    assert xtts and all(r["split"] == "test_unseen_gen" for r in xtts)
    assert not any(r["generator"] == "xtts_v2" for r in rows if r["split"] in ("train", "val", "test"))
    gen_split = [r for r in rows if r["split"] == "test_unseen_gen"]
    assert {r["label"] for r in gen_split} == {"real", "fake"}  # EER is defined
    assert {r["generator"] for r in rows if r["label"] == "real"} == {"real"}


# ------------------------------------------------------------------ clone signature
def test_clone_signature_trials_and_summary():
    sys.path.insert(0, str(ROOT / "scripts"))
    from clone_signature import build_trials, summarize

    rows = []
    for s in range(6):
        for k in range(6):
            rows.append({"path": f"r{s}_{k}", "label": "real", "lang": "hi", "speaker": f"hi_{s}", "generator": "real"})
        for k in range(4):
            rows.append({"path": f"f{s}_{k}", "label": "fake", "lang": "hi", "speaker": f"hi_{s}",
                         "generator": "xtts_v2" if k % 2 else "freevc24"})
    enroll, trials = build_trials(rows)
    assert len(enroll) == 6 and all(len(v) == 2 for v in enroll.values())
    kinds = {t["kind"] for t in trials}
    assert kinds == {"genuine", "clone", "impostor"}
    # enrollment clips are never reused as trials
    used = {t["path"] for t in trials if t["kind"] == "genuine"}
    assert not used & {p for v in enroll.values() for p in v}

    rng = np.random.default_rng(0)
    for t in trials:  # clones fool the matcher; detector separates them
        t["sim"] = {"genuine": 0.75, "clone": 0.70, "impostor": 0.10}[t["kind"]] + 0.05 * rng.standard_normal()
        if t["kind"] != "impostor":
            t["synth"] = 0.9 if t["kind"] == "clone" else 0.1
    s = summarize(trials, synth_threshold=0.5)
    assert s["clones_accepted_by_speaker_match_pct"] > 50
    assert s["red_clone_of_name_on_clones_pct"] > 50
    assert s["red_clone_of_name_on_genuine_pct"] == 0.0
    assert set(s["by_generator"]) == {"freevc24", "xtts_v2"}


def test_calibrate_writes_threshold(tmp_path):
    from awaaz.inference import load_detector

    # toy data: tones = fake, noise = real
    rows = []
    for k in range(12):
        for label, y in (("fake", 1), ("real", 0)):
            p = tmp_path / f"{label}{k}.wav"
            t = np.arange(SR * 5) / SR
            x = tone(5, 300 + 40 * k) if y else (0.2 * np.random.default_rng(k).standard_normal(SR * 5)).astype(np.float32)
            save_wav(p, x)
            rows.append({"path": str(p), "label": label, "y": y, "lang": "hi", "speaker": "", "generator": "", "split": "val"})
    man = tmp_path / "m.csv"
    with man.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    ck = tmp_path / "m.pt"
    torch.manual_seed(0)
    torch.save({"model": "specresnet", "kwargs": {"widths": (8, 16, 16, 32)},
                "state_dict": SpecResNet(widths=(8, 16, 16, 32)).state_dict()}, ck)
    subprocess.run([sys.executable, str(ROOT / "scripts/calibrate.py"), "--ckpt", str(ck), "--manifest", str(man),
                    "--chains", "clean"], check=True, capture_output=True)
    m = load_detector(ck)
    assert 0.0 <= m.awaaz_meta["threshold"] <= 1.0
    assert m.awaaz_meta["calibration"]["n_clips"] == 24
    assert (tmp_path / "m.pt.bak").exists()


def test_scam_eval_runs_and_is_reasonable():
    sys.path.insert(0, str(ROOT / "scripts"))
    from eval_scam import report, run

    rows = run(ROOT / "eval/scam_texts.csv", use_nli=False)
    overall, per_lang, _ = report(rows)
    assert len(rows) == 60 and overall["f1"] >= 0.85
