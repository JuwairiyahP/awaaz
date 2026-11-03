"""Awaaz dashboard (Streamlit). Deployable to Hugging Face Spaces.

    AWAAZ_CKPT=runs/xlsr/best.pt streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import html
import json
import os
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import altair as alt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

st.set_page_config(page_title="Awaaz — voice-clone scam check", page_icon="🎙️", layout="wide")

COLORS = {"red": "#C62828", "amber": "#E08E0B", "green": "#2E7D32"}
CUE_LABELS = {"urgency": "Urgency", "secrecy": "Secrecy", "money": "Money / UPI", "otp": "OTP / PIN", "authority": "Authority"}
RESULTS_DIR = Path(os.environ.get("AWAAZ_RESULTS", ROOT / "results"))

st.markdown("""
<style>
.verdict {border-left: 8px solid var(--c); padding: 1rem 1.25rem; border-radius: 10px;
          background: color-mix(in srgb, var(--c) 9%, transparent); margin: .5rem 0 1rem;}
.verdict h2 {margin: 0; color: var(--c); font-size: 1.6rem;}
.verdict .hi {font-size: 1.1rem; opacity: .85; margin-top: .15rem;}
.verdict p {margin: .6rem 0 0;}
mark.cue {background: color-mix(in srgb, #E08E0B 30%, transparent); padding: 0 .15em; border-radius: 3px;}
.small {font-size: .85rem; opacity: .75;}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading models (first run downloads them)…")
def load():
    from app.service import get_awaaz

    return get_awaaz()


def save_tmp(data: bytes, name: str) -> Path:
    suffix = Path(name).suffix or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(data)
    tmp.close()
    return Path(tmp.name)


def highlight(text: str, phrases: dict[str, list[str]]) -> str:
    out = html.escape(text)
    for p in sorted({p for ps in phrases.values() for p in ps}, key=len, reverse=True):
        if p.strip():
            out = re.sub(re.escape(html.escape(p)), lambda m: f"<mark class='cue'>{m.group(0)}</mark>", out, flags=re.I)
    return out


def verdict_card(v: dict):
    c = COLORS.get(v["level"], "#555")
    st.markdown(
        f"<div class='verdict' style='--c:{c}'><h2>{html.escape(v['headline'])}</h2>"
        f"<div class='hi'>{html.escape(v.get('headline_hi', ''))}</div>"
        f"<p>{html.escape(v['advice'])}</p></div>", unsafe_allow_html=True)


def timeline_chart(r: dict, threshold: float):
    if len(r["window_scores"]) < 1:
        return
    df = pd.DataFrame({"start (s)": r["window_starts"], "synthetic score": r["window_scores"]})
    base = alt.Chart(df).encode(x=alt.X("start (s):Q", title="Time in clip (s)"))
    line = base.mark_line(point=True).encode(
        y=alt.Y("synthetic score:Q", scale=alt.Scale(domain=[0, 1]), title="How synthetic (per 4 s window)"),
        tooltip=["start (s)", alt.Tooltip("synthetic score:Q", format=".2f")])
    rule = alt.Chart(pd.DataFrame({"t": [threshold]})).mark_rule(strokeDash=[4, 4], color="#C62828").encode(y="t:Q")
    st.altair_chart((line + rule).properties(height=220), width="stretch")


def gradcam_view(aw, path: Path):
    from awaaz.detector import SpecResNet, grad_cam

    if not isinstance(aw.detector, SpecResNet):
        return
    try:
        import matplotlib.pyplot as plt
        import torch

        from awaaz.audio import load_audio, make_windows, peak_normalize, trim_silence
        from awaaz.inference import score_windows

        x = peak_normalize(trim_silence(load_audio(path)))
        w, _ = make_windows(x)
        i = int(np.argmax(score_windows(aw.detector, w)))
        wav = torch.from_numpy(w[i : i + 1]).to(next(aw.detector.parameters()).device)
        spec = aw.detector.spectrogram(wav)[0, 0].cpu().numpy()
        cam = grad_cam(aw.detector, wav)
        cam_up = np.kron(cam, np.ones((int(np.ceil(spec.shape[0] / cam.shape[0])), int(np.ceil(spec.shape[1] / cam.shape[1])))))
        cam_up = cam_up[: spec.shape[0], : spec.shape[1]]
        fig, ax = plt.subplots(figsize=(9, 2.6))
        ax.imshow(spec, origin="lower", aspect="auto", cmap="gray")
        ax.imshow(cam_up, origin="lower", aspect="auto", cmap="inferno", alpha=0.45)
        ax.set_xlabel("time frames (10 ms)")
        ax.set_ylabel("mel band")
        ax.set_title("Where the detector looked (most synthetic window)")
        st.pyplot(fig, clear_figure=True)
    except Exception as e:
        st.caption(f"Grad-CAM unavailable: {e}")


# ------------------------------------------------------------------ layout
st.title("🎙️ Awaaz")
st.caption("Is this really your family member — or an AI clone of them? Built for Indian languages and WhatsApp/phone audio.")

tab_check, tab_family, tab_results, tab_about = st.tabs(["Check a voice note", "Family voices", "How well it works", "About"])

with tab_check:
    aw = load()
    if aw.detector is None:
        st.warning("No detector checkpoint loaded (set AWAAZ_CKPT). Only speaker and scam checks will run.")
    c1, c2 = st.columns(2)
    with c1:
        up = st.file_uploader("Upload a voice note or call recording", type=["ogg", "opus", "mp3", "wav", "m4a", "aac", "amr", "flac", "webm", "mp4"])
    with c2:
        rec = st.audio_input("…or record one") if hasattr(st, "audio_input") else None
    src = up or rec
    if src is not None:
        data = src.getvalue()
        st.audio(data)
        if st.button("Check this voice", type="primary"):
            path = save_tmp(data, getattr(src, "name", "rec.wav"))
            try:
                with st.spinner("Listening…"):
                    r = aw.analyze(path).to_dict()
                st.session_state["last"] = r
                st.session_state["last_path"] = str(path)
            except Exception as e:
                st.error(f"Couldn't analyse this clip: {e}")
    r = st.session_state.get("last")
    if r:
        verdict_card(r["verdict"])
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Synthetic-voice score", f"{r['synthetic']:.2f}", help=f"Flag at {aw.th.synthetic:.2f}")
        sp = r["speaker"]
        if sp.get("scores"):
            best = max(sp["scores"], key=sp["scores"].get)
            m2.metric("Sounds like", best if sp.get("name") else f"({best}?)", f"{sp['scores'][best]:.2f} similarity",
                      delta_color="off")
        else:
            m2.metric("Sounds like", "—", "nobody enrolled", delta_color="off")
        m3.metric("Scam-script score", f"{r['scam'].get('score', 0):.2f}")
        m4.metric("Analysed in", f"{sum(r['timings'].values()):.1f} s", f"{r['duration_s']:.0f} s clip", delta_color="off")

        st.subheader("When did it sound synthetic?")
        timeline_chart(r, aw.th.synthetic)

        st.subheader("What was said")
        sc = r["scam"]
        if sc.get("transcript"):
            st.markdown(f"<div>{highlight(sc['transcript'], sc.get('phrases', {}))}</div>", unsafe_allow_html=True)
            if sc.get("translation") and sc.get("language") != "en":
                st.markdown(f"<div class='small'>English: {highlight(sc['translation'], sc.get('phrases', {}))}</div>",
                            unsafe_allow_html=True)
            cues = sc.get("cues", {})
            if cues:
                cdf = pd.DataFrame({"cue": [CUE_LABELS.get(k, k) for k in cues], "strength": list(cues.values())})
                st.altair_chart(alt.Chart(cdf).mark_bar().encode(
                    x=alt.X("strength:Q", scale=alt.Scale(domain=[0, 1])), y=alt.Y("cue:N", sort=None, title=None),
                    tooltip=["cue", alt.Tooltip("strength:Q", format=".2f")]).properties(height=170),
                    use_container_width=True)
        else:
            st.caption("Scam-script check is off or nothing was transcribed.")

        with st.expander("Detector heatmap and raw output"):
            if st.session_state.get("last_path") and Path(st.session_state["last_path"]).exists():
                gradcam_view(aw, Path(st.session_state["last_path"]))
            st.json(r)

with tab_family:
    aw = load()
    st.write("Enroll **real** voices of people you'd send money to. Awaaz stores only a voice fingerprint (192 numbers), never the audio.")
    name = st.text_input("Name", placeholder="Mum, Bhaiya, Papa…")
    f = st.file_uploader("A normal voice note from them (20–30 s)", type=["ogg", "opus", "mp3", "wav", "m4a", "aac", "flac"], key="enroll")
    if st.button("Save voice", disabled=not (name and f)):
        if aw.encoder is None:
            st.error("Speaker matching is disabled on this deployment.")
        else:
            p = save_tmp(f.getvalue(), f.name)
            try:
                aw.enroll(name.strip(), p)
                st.success(f"Saved {name}.")
            finally:
                p.unlink(missing_ok=True)
    people = aw.vault.names()
    st.write("**Enrolled:** " + (", ".join(people) if people else "nobody yet"))
    if people:
        who = st.selectbox("Remove someone", [""] + people)
        if who and st.button(f"Forget {who}"):
            aw.vault.remove(who)
            st.rerun()

with tab_results:
    runs = sorted([d for d in RESULTS_DIR.glob("*") if (d / "grid.csv").exists()]) if RESULTS_DIR.exists() else []
    if not runs:
        st.info(f"No evaluation results found in {RESULTS_DIR}. Run scripts/eval_grid.py and copy results/ here.")
    else:
        run = st.selectbox("Model", [d.name for d in runs], index=len(runs) - 1)
        d = RESULTS_DIR / run
        grid = pd.read_csv(d / "grid.csv")
        st.write("**Equal error rate (%) by language and channel.** Lower is better. "
                 "*test* = languages trained on, with unseen speakers. *test_unseen* = languages never trained on. "
                 "*test_unseen_gen* = a cloning tool never trained on.")
        grid["row"] = grid["split"] + " · " + grid["lang"]
        heat = alt.Chart(grid).mark_rect().encode(
            x=alt.X("channel:N", sort=None, title="Channel"), y=alt.Y("row:N", title=None),
            color=alt.Color("eer:Q", scale=alt.Scale(scheme="orangered", domain=[0, 50]), title="EER %"),
            tooltip=["split", "lang", "channel", "n", "eer", "auc"])
        text = heat.mark_text(fontSize=12).encode(text=alt.Text("eer:Q", format=".1f"),
                                                   color=alt.condition("datum.eer > 25", alt.value("white"), alt.value("black")))
        st.altair_chart((heat + text).properties(height=max(180, 26 * grid["row"].nunique())), width="stretch")
        cols = st.columns(2)
        if (d / "summary.json").exists():
            cols[0].write("**Pooled EER %**")
            cols[0].json(json.loads((d / "summary.json").read_text()))
        if (d / "latency.json").exists():
            cols[1].write("**CPU latency**")
            cols[1].json(json.loads((d / "latency.json").read_text()))

with tab_about:
    st.markdown("""
**Why:** in McAfee's survey, 47% of Indian adults had experienced or knew someone hit by an AI voice scam,
almost double the global average. Scammers clone a relative's voice from a few seconds of social-media audio.

**How:** three independent checks.
1. **Synthetic voice:** a detector trained on Indian-language real and cloned speech, through WhatsApp, phone and re-recording channels.
2. **Speaker match:** compares the voice with family members you enrolled. A voice that matches *and* sounds synthetic is the signature of a targeted clone.
3. **Scam script:** transcribes the message and looks for urgency, secrecy, money/UPI, OTP and police/arrest cues.

**Limits:** a warning tool, not proof. New cloning tools can evade detectors. Always verify by calling back on a saved number.
Uploaded audio is deleted right after analysis.
""")
