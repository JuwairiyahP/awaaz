"""Channel simulation: what happens to a voice on its way to a victim.

Real scam audio is rarely clean. It is a WhatsApp voice note (Opus at low
bitrate), a phone call (8 kHz narrowband, mu-law), a re-shared MP3, or a
clip played on one phone's speaker and recorded on another. Detectors trained
only on clean audio learn artifacts that these channels erase, so we train
and evaluate through them.

Two entry points:
- `apply_chain(x, name)`: a fixed, named chain, used for the evaluation grid.
- `random_channel(x, rng)`: a random chain, used for training augmentation.
"""
from __future__ import annotations

import subprocess

import numpy as np
from scipy import signal

from . import SAMPLE_RATE


# ---------------------------------------------------------------- ffmpeg codecs
try:  # in-process codecs (same ffmpeg libraries, no process launch): 4-8x faster
    import av as _av
except ImportError:  # pragma: no cover
    _av = None


def _av_roundtrip(x: np.ndarray, codec: str, kbps: int, fmt: str, sr: int = SAMPLE_RATE,
                  options: dict | None = None) -> np.ndarray:
    import io

    buf = io.BytesIO()
    out = _av.open(buf, "w", format=fmt)
    st = out.add_stream(codec, rate=sr)
    st.bit_rate = kbps * 1000
    st.codec_context.layout = "mono"
    if options:
        st.codec_context.options = options
    st.codec_context.open()
    fs = st.codec_context.frame_size or 1024
    rs = _av.AudioResampler(format=st.codec_context.format.name, layout="mono", rate=sr, frame_size=fs)
    fr = _av.AudioFrame.from_ndarray(np.ascontiguousarray(x[None, :], dtype=np.float32), format="flt", layout="mono")
    fr.sample_rate = sr
    for f in rs.resample(fr) + rs.resample(None):
        for p in st.encode(f):
            out.mux(p)
    for p in st.encode(None):
        out.mux(p)
    out.close()

    inp = _av.open(io.BytesIO(buf.getvalue()), "r")
    dr = _av.AudioResampler(format="flt", layout="mono", rate=sr)
    chunks = []
    for f in inp.decode(audio=0):
        chunks += [g.to_ndarray().reshape(-1) for g in dr.resample(f)]
    chunks += [g.to_ndarray().reshape(-1) for g in dr.resample(None)]
    inp.close()
    y = np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, np.float32)
    return _match_length(y, len(x))


def _ffmpeg_roundtrip(x: np.ndarray, encode_args: list[str], fmt: str, sr: int = SAMPLE_RATE,
                      decode_fmt: str | None = None) -> np.ndarray:
    """Encode x with the ffmpeg CLI, decode it back to f32 at sr. Keeps the length.
    Fallback when PyAV is unavailable."""
    enc = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
         "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "-",
         *encode_args, "-f", fmt, "-"],
        input=x.astype(np.float32).tobytes(), capture_output=True, check=False,
    )
    if enc.returncode != 0 or not enc.stdout:
        raise RuntimeError(f"encode failed: {enc.stderr.decode(errors='ignore')[:300]}")
    dec = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
         "-f", decode_fmt or fmt, "-i", "-", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
        input=enc.stdout, capture_output=True, check=False,
    )
    if dec.returncode != 0:
        raise RuntimeError(f"decode failed: {dec.stderr.decode(errors='ignore')[:300]}")
    y = np.frombuffer(dec.stdout, dtype=np.float32).copy()
    return _match_length(y, len(x))


def _codec(x, codec, kbps, fmt, cli_args, decode_fmt=None, options=None):
    if _av is not None:
        try:
            return _av_roundtrip(x, codec, kbps, fmt, options=options)
        except Exception:
            pass  # fall through to the CLI
    return _ffmpeg_roundtrip(x, cli_args, fmt, decode_fmt=decode_fmt)


def _match_length(y: np.ndarray, n: int) -> np.ndarray:
    if len(y) >= n:
        return y[:n]
    return np.pad(y, (0, n - len(y)))


def opus(x: np.ndarray, kbps: int = 16) -> np.ndarray:
    """WhatsApp/Telegram voice notes use Opus at low bitrates."""
    return _codec(x, "libopus", kbps, "ogg", ["-c:a", "libopus", "-b:a", f"{kbps}k", "-application", "voip"],
                  options={"application": "voip"})


def mp3(x: np.ndarray, kbps: int = 64) -> np.ndarray:
    """Re-shared clips on social media are often MP3/AAC transcoded."""
    return _codec(x, "libmp3lame", kbps, "mp3", ["-c:a", "libmp3lame", "-b:a", f"{kbps}k"])


def aac(x: np.ndarray, kbps: int = 48) -> np.ndarray:
    # the ADTS stream is written with -f adts but must be read back with -f aac
    return _codec(x, "aac", kbps, "adts", ["-c:a", "aac", "-b:a", f"{kbps}k"], decode_fmt="aac")


# ---------------------------------------------------------------- numpy effects
def phone_narrowband(x: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Classic phone line: 300-3400 Hz band, 8 kHz sampling, mu-law 8-bit."""
    sos = signal.butter(6, [300, 3400], btype="bandpass", fs=sr, output="sos")
    y = signal.sosfilt(sos, x)
    y = signal.resample_poly(y, 1, 2)  # 16k -> 8k
    mu = 255.0
    y = np.clip(y, -1, 1)
    q = np.sign(y) * np.log1p(mu * np.abs(y)) / np.log1p(mu)
    q = np.round(q * 127) / 127  # 8-bit quantisation
    y = np.sign(q) * ((1 + mu) ** np.abs(q) - 1) / mu
    y = signal.resample_poly(y, 2, 1)  # back to 16k for the model
    return _match_length(y.astype(np.float32), len(x))


def synthetic_rir(sr: int = SAMPLE_RATE, rt60: float = 0.35, rng: np.random.Generator | None = None) -> np.ndarray:
    """Exponentially decaying noise impulse response, a cheap room model."""
    rng = rng or np.random.default_rng(0)
    n = int(sr * rt60)
    t = np.arange(n) / sr
    decay = np.exp(-6.9 * t / rt60)  # -60 dB at rt60
    h = rng.standard_normal(n) * decay
    h[0] = 1.0  # direct path
    return (h / np.sqrt(np.sum(h ** 2))).astype(np.float32)


def reverb(x: np.ndarray, rt60: float = 0.35, rng: np.random.Generator | None = None) -> np.ndarray:
    y = signal.fftconvolve(x, synthetic_rir(rt60=rt60, rng=rng))[: len(x)]
    peak = np.max(np.abs(y)) + 1e-8
    return (y / peak * min(0.95, np.max(np.abs(x)) + 1e-8)).astype(np.float32)


def add_noise(x: np.ndarray, snr_db: float = 15.0, rng: np.random.Generator | None = None, color: str = "pink") -> np.ndarray:
    """Add white or pink noise at a target SNR."""
    rng = rng or np.random.default_rng(0)
    n = rng.standard_normal(len(x)).astype(np.float32)
    if color == "pink":
        # 1/f shaping in the frequency domain
        spec = np.fft.rfft(n)
        f = np.arange(len(spec), dtype=np.float32)
        f[0] = 1.0
        n = np.fft.irfft(spec / np.sqrt(f), n=len(x)).astype(np.float32)
    p_sig = np.mean(x ** 2) + 1e-12
    p_noise = np.mean(n ** 2) + 1e-12
    n *= np.sqrt(p_sig / (p_noise * 10 ** (snr_db / 10)))
    return (x + n).astype(np.float32)


def speaker_rerecord(x: np.ndarray, sr: int = SAMPLE_RATE, rng: np.random.Generator | None = None) -> np.ndarray:
    """A clip played on one phone's speaker and recorded on another:
    tiny speaker (weak bass), room reverb, background noise."""
    sos = signal.butter(4, [180, 7000], btype="bandpass", fs=sr, output="sos")
    y = signal.sosfilt(sos, x).astype(np.float32)
    y = reverb(y, rt60=0.3, rng=rng)
    return add_noise(y, snr_db=20, rng=rng)


# ---------------------------------------------------------------- named chains
CHAINS = {
    "clean": lambda x, rng: x,
    "whatsapp_opus16": lambda x, rng: opus(x, 16),
    "phone_8k": lambda x, rng: phone_narrowband(x),
    "mp3_64": lambda x, rng: mp3(x, 64),
    "rerecorded": lambda x, rng: speaker_rerecord(x, rng=rng),
    # a WhatsApp voice note forwarded after being recorded off a speaker
    "rerecorded_whatsapp": lambda x, rng: opus(speaker_rerecord(x, rng=rng), 16),
}

EVAL_CHAINS = list(CHAINS.keys())


def apply_chain(x: np.ndarray, name: str, seed: int = 0) -> np.ndarray:
    if name not in CHAINS:
        raise KeyError(f"unknown chain {name!r}; choose from {EVAL_CHAINS}")
    return CHAINS[name](x, np.random.default_rng(seed)).astype(np.float32)


def random_channel(x: np.ndarray, rng: np.random.Generator, p_clean: float = 0.3) -> np.ndarray:
    """Training augmentation: random codec + optional room + optional noise.

    Bitrates and SNRs are sampled from ranges, not the fixed eval values, so
    the eval grid still measures generalisation rather than memorisation.
    """
    if rng.random() < p_clean:
        return x
    y = x
    if rng.random() < 0.3:
        y = reverb(y, rt60=float(rng.uniform(0.15, 0.6)), rng=rng)
    if rng.random() < 0.4:
        y = add_noise(y, snr_db=float(rng.uniform(8, 30)), rng=rng, color=str(rng.choice(["white", "pink"])))
    codec = rng.choice(["opus", "phone", "mp3", "aac", "none"], p=[0.35, 0.2, 0.15, 0.15, 0.15])
    try:
        if codec == "opus":
            y = opus(y, int(rng.choice([12, 16, 24, 32])))
        elif codec == "phone":
            y = phone_narrowband(y)
        elif codec == "mp3":
            y = mp3(y, int(rng.choice([32, 48, 96])))
        elif codec == "aac":
            y = aac(y, int(rng.choice([32, 64])))
    except RuntimeError:
        pass  # a codec failure should never kill a training run
    peak = np.max(np.abs(y)) + 1e-8
    return (y / max(peak, 1.0)).astype(np.float32)
