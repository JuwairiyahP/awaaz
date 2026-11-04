"""Awaaz web demo (Gradio), for Hugging Face Spaces with ZeroGPU.

    AWAAZ_CKPT=runs/xlsr/best.pt python app/gradio_app.py

Privacy by design: enrolled voices live only in the visitor's browser session
(gr.State), never on the server, and uploaded audio is deleted after analysis.
On ZeroGPU only the detector forward pass runs on the GPU (it's the only
piece that needs it), which keeps each request's GPU time to a second or two.
"""
from __future__ import annotations

import html
import json
import os
import re
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402

try:  # Hugging Face ZeroGPU; a no-op decorator anywhere else
    import spaces

    GPU = spaces.GPU(duration=10)  # a check needs ~1-2 s; a small reservation stretches visitors' quota
except Exception:  # pragma: no cover
    def GPU(fn):
        return fn

import awaaz.inference as inference  # noqa: E402
from app.service import get_awaaz  # noqa: E402
from awaaz.speaker import VoiceVault  # noqa: E402

RESULTS_DIR = Path(os.environ.get("AWAAZ_RESULTS", ROOT / "results"))
COLORS = {"red": "#C62828", "amber": "#E08E0B", "green": "#2E7D32"}
CUE_LABELS = {"urgency": "Urgency", "secrecy": "Secrecy", "money": "Money / UPI", "otp": "OTP / PIN", "authority": "Police / authority"}

# keep a handle on the original scorer, once, so re-importing this module
# never wraps the wrapper (which would recurse forever)
_cpu_score = getattr(inference, "_original_score_windows", inference.score_windows)
inference._original_score_windows = _cpu_score


def _aw():
    return get_awaaz()


@GPU
def _gpu_score_windows(windows, batch=16):
    """Runs in ZeroGPU's GPU worker. ZeroGPU pickles a function's arguments and
    return value to move them between processes, and the XLS-R module (it uses
    weight-norm parametrizations) can't be pickled. So only plain arrays cross
    the boundary; the model is the global one loaded at startup."""
    import torch

    model = _aw().detector
    if torch.cuda.is_available():
        model.to("cuda")
    return np.asarray(_cpu_score(model, np.ascontiguousarray(windows, dtype=np.float32), batch))


_ran = threading.local()  # which device the last check used, per request thread


def _score_windows(model, windows, batch=16):  # same signature the pipeline calls
    """GPU first; if ZeroGPU refuses (visitor's daily quota used up, no GPU free,
    timeout), score on the CPU copy of the same model instead of failing."""
    try:
        out = _gpu_score_windows(windows, batch)
        _ran.device = "gpu"
        return out
    except Exception as e:  # ZeroGPU raises gr.Error for quota / queue problems
        print(f"[awaaz] GPU unavailable ({type(e).__name__}: {str(e)[:120]}); scoring on CPU", flush=True)
        _ran.device = "cpu"
        import torch

        with torch.inference_mode():
            return np.asarray(_cpu_score(_aw().detector.to("cpu"), np.ascontiguousarray(windows, dtype=np.float32), batch))


inference.score_windows = _score_windows  # the pipeline looks this up at call time


# ------------------------------------------------------------------ rendering
def verdict_html(v: dict) -> str:
    c = COLORS.get(v["level"], "#555")
    return (f"<div style='border-left:8px solid {c};padding:14px 18px;border-radius:10px;"
            f"background:color-mix(in srgb,{c} 9%,transparent)'>"
            f"<div style='font-size:1.55rem;font-weight:700;color:{c}'>{html.escape(v['headline'])}</div>"
            f"<div style='font-size:1.1rem;opacity:.85;margin-top:2px'>{html.escape(v.get('headline_hi', ''))}</div>"
            f"<div style='margin-top:10px'>{html.escape(v['advice'])}</div>"
            + "".join(f"<div style='font-size:.9rem;opacity:.75;margin-top:4px'>• {html.escape(r)}</div>"
                      for r in v.get("reasons", []))
            + "</div>")


def highlight(text: str, phrases: dict) -> str:
    out = html.escape(text)
    for p in sorted({p for ps in phrases.values() for p in ps}, key=len, reverse=True):
        if p.strip():
            out = re.sub(re.escape(html.escape(p)),
                         lambda m: f"<mark style='background:#E08E0B55;border-radius:3px;padding:0 2px'>{m.group(0)}</mark>",
                         out, flags=re.I)
    return out


def details_html(r: dict) -> str:
    sp, sc = r["speaker"], r["scam"]
    parts = [f"<b>Synthetic-voice score:</b> {r['synthetic']:.2f} &nbsp; <b>Clip:</b> {r['duration_s']:.0f} s"
             f" &nbsp; <b>Analysed in:</b> {sum(r['timings'].values()):.1f} s"]
    if sp.get("scores"):
        best = max(sp["scores"], key=sp["scores"].get)
        parts.append(f"<b>Closest enrolled voice:</b> {html.escape(best)} (similarity {sp['scores'][best]:.2f}"
                     f"{'' if sp.get('name') else ', below match threshold'})")
    else:
        parts.append("<b>Speaker match:</b> nobody enrolled yet (see the Family voices tab)")
    if sc.get("transcript"):
        parts.append(f"<b>Heard</b> ({html.escape(sc.get('language') or '?')}): {highlight(sc['transcript'], sc.get('phrases', {}))}")
        if sc.get("translation") and sc.get("language") != "en":
            parts.append(f"<b>English:</b> {highlight(sc['translation'], sc.get('phrases', {}))}")
        cues = [f"{CUE_LABELS.get(k, k)} {v:.2f}" for k, v in sc.get("cues", {}).items() if v >= 0.5]
        parts.append("<b>Scam cues:</b> " + (", ".join(cues) if cues else "none"))
    return "<div style='line-height:1.6'>" + "<br>".join(parts) + "</div>"


def timeline_plot(r: dict, threshold: float):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 2.4))
    ws, st = r["window_scores"], r["window_starts"]
    if ws:
        ax.plot(st, ws, marker="o", color="#333")
        ax.fill_between(st, ws, threshold, where=np.array(ws) >= threshold, color="#C62828", alpha=0.25, interpolate=True)
    ax.axhline(threshold, ls="--", color="#C62828", lw=1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("time in clip (s)")
    ax.set_ylabel("how synthetic")
    ax.set_title("Per 4-second window", fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ actions
def analyze(audio_path, people):
    if not audio_path:
        raise gr.Error("Upload or record a voice note first.")
    aw = _aw()
    aw.vault = VoiceVault(None, people or {})
    _ran.device = None
    try:
        r = aw.analyze(audio_path).to_dict()
    except RuntimeError as e:
        if "could not decode" in str(e):
            raise gr.Error("Couldn't read that audio file. Try a WhatsApp voice note, .ogg, .mp3, .m4a or .wav.")
        raise gr.Error(f"Analysis failed: {type(e).__name__}: {e}")
    det = details_html(r)
    if getattr(_ran, "device", None) == "cpu":
        det += ("<div style='font-size:.85rem;opacity:.65;margin-top:8px'>Ran on CPU (GPU busy or daily GPU quota "
                "used up), so this one was slower. Same model, same result.</div>")
    return verdict_html(r["verdict"]), timeline_plot(r, aw.th.synthetic), det, r


def enroll(name, audio_path, people):
    name = (name or "").strip()[:40]
    if not name or not audio_path:
        raise gr.Error("Give a name and a voice note.")
    aw = _aw()
    if aw.encoder is None:
        raise gr.Error("Speaker matching is switched off on this deployment.")
    from awaaz.audio import load_audio

    emb = aw.encoder.embed(load_audio(audio_path))
    people = dict(people or {})
    people.setdefault(name, []).append(emb.astype(float).tolist())
    return people, enrolled_md(people), gr.update(choices=sorted(people)), ""


def forget(name, people):
    people = {k: v for k, v in (people or {}).items() if k != name}
    return people, enrolled_md(people), gr.update(choices=sorted(people), value=None)


def enrolled_md(people) -> str:
    return "**Enrolled in this session:** " + (", ".join(sorted(people)) if people else "nobody yet")


def results_md() -> str:
    if not RESULTS_DIR.exists():
        return "No evaluation results bundled with this deployment."
    out = []
    for d in sorted(RESULTS_DIR.iterdir()):
        if (d / "summary.json").exists():
            s = json.loads((d / "summary.json").read_text())
            out.append(f"### {d.name}\n| split / channel | EER % |\n|---|---|\n" +
                       "\n".join(f"| {k} | {v} |" for k, v in sorted(s.items())))
        for f in sorted(d.glob("summary_*.json")):
            s = json.loads(f.read_text())
            keep = {k: v for k, v in s.items() if not isinstance(v, (dict, list))}
            out.append(f"### {d.name} · {f.stem.replace('summary_', '')}\n```json\n{json.dumps(keep, indent=1)}\n```")
    return "\n\n".join(out) or "No evaluation results bundled with this deployment."


# ------------------------------------------------------------------ layout
ABOUT = """
**Why:** in McAfee's survey, 47% of Indian adults had experienced or knew someone hit by an AI voice scam,
almost double the global average. Scammers clone a relative's voice from a few seconds of social-media audio.

**How:** three independent checks.
1. **Synthetic voice:** a detector trained on Indian-language real and cloned speech (IndicSynth vs Kathbath), through WhatsApp, phone and re-recording channels.
2. **Speaker match:** compares the voice with family members you enrolled. A voice that matches *and* sounds synthetic is the signature of a targeted clone.
3. **Scam script:** transcribes the message and looks for urgency, secrecy, money/UPI, OTP and police/arrest cues.

**Limits:** a warning tool, not proof. New cloning tools can evade detectors. Always verify by calling back on a saved number.
Enrolled voices stay in your browser session only; uploaded audio is deleted after analysis.
"""

# load every model at startup, before ZeroGPU starts its worker processes,
# so the first visitor doesn't wait and the GPU function finds them in memory
_aw()

with gr.Blocks(title="Awaaz — voice-clone scam check") as demo:
    people = gr.State({})
    gr.Markdown("# 🎙️ Awaaz\nIs this really your family member, or an AI clone of them? "
                "Built for Indian languages and WhatsApp/phone audio.")
    with gr.Tab("Check a voice note"):
        with gr.Row():
            with gr.Column(scale=1):
                audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Voice note or call recording")
                go = gr.Button("Check this voice", variant="primary")
            with gr.Column(scale=2):
                verdict = gr.HTML()
                plot = gr.Plot(show_label=False)
                details = gr.HTML()
        with gr.Accordion("Raw output", open=False):
            raw = gr.JSON()
        go.click(analyze, [audio, people], [verdict, plot, details, raw])
    with gr.Tab("Family voices"):
        gr.Markdown("Enroll the **real** voice of someone you'd send money to: a normal 20–30 s voice note. "
                    "Only a 192-number voice fingerprint is kept, and only in this browser session.")
        with gr.Row():
            name = gr.Textbox(label="Name", placeholder="Mum, Bhaiya, Papa…")
            enroll_audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Their real voice")
        save = gr.Button("Save voice", variant="primary")
        listing = gr.Markdown(enrolled_md({}))
        with gr.Row():
            who = gr.Dropdown(choices=[], label="Remove someone")
            rm = gr.Button("Forget")
        save.click(enroll, [name, enroll_audio, people], [people, listing, who, name])
        rm.click(forget, [who, people], [people, listing, who])
    with gr.Tab("How well it works"):
        gr.Markdown(results_md())
    with gr.Tab("About"):
        gr.Markdown(ABOUT)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch()
