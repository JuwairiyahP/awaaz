"""Synthetic-voice detectors. Both take raw 16 kHz waveforms [B, T] and return
one logit per clip (higher = more likely synthetic).

- SpecResNet: log-mel spectrogram + small ResNet, trained from scratch.
  Fast to train, and gives Grad-CAM heatmaps over time-frequency.
- XLSRDetector: frozen multilingual wav2vec2 XLS-R backbone, learned weighted
  sum over its layers, attentive statistics pooling, small MLP head.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import SAMPLE_RATE


# ---------------------------------------------------------------- front end
def mel_filterbank(n_fft: int, n_mels: int, sr: int, fmin: float = 20.0, fmax: float | None = None) -> np.ndarray:
    fmax = fmax or sr / 2
    hz2mel = lambda f: 2595 * np.log10(1 + f / 700)  # noqa: E731
    mel2hz = lambda m: 700 * (10 ** (m / 2595) - 1)  # noqa: E731
    mels = np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2)
    bins = np.floor((n_fft + 1) * mel2hz(mels) / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        c = max(c, l + 1)
        r = max(r, c + 1)
        fb[m - 1, l:c] = (np.arange(l, c) - l) / (c - l)
        fb[m - 1, c:r] = (r - np.arange(c, r)) / (r - c)
    return fb


class LogMel(nn.Module):
    """Waveform -> log-mel [B, 1, n_mels, frames], per-utterance normalised."""

    def __init__(self, n_fft=512, hop=160, n_mels=80, sr=SAMPLE_RATE):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self.register_buffer("fb", torch.from_numpy(mel_filterbank(n_fft, n_mels, sr)), persistent=False)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(wav.float(), self.n_fft, self.hop, window=self.window, return_complex=True)
        power = spec.abs() ** 2  # [B, F, T]
        mel = torch.matmul(self.fb, power)
        logmel = torch.log(mel + 1e-6)
        logmel = (logmel - logmel.mean(dim=(1, 2), keepdim=True)) / (logmel.std(dim=(1, 2), keepdim=True) + 1e-5)
        return logmel.unsqueeze(1)


class SpecAugment(nn.Module):
    """Random time and frequency masks during training."""

    def __init__(self, freq_mask=10, time_mask=30, n=2):
        super().__init__()
        self.freq_mask, self.time_mask, self.n = freq_mask, time_mask, n

    def forward(self, x):
        if not self.training:
            return x
        b, _, f, t = x.shape
        x = x.clone()
        for _ in range(self.n):
            fw = torch.randint(0, self.freq_mask + 1, (b,))
            f0 = (torch.rand(b) * (f - fw)).long()
            tw = torch.randint(0, self.time_mask + 1, (b,))
            t0 = (torch.rand(b) * (t - tw).clamp(min=1)).long()
            for i in range(b):
                x[i, :, f0[i] : f0[i] + fw[i], :] = 0
                x[i, :, :, t0[i] : t0[i] + tw[i]] = 0
        return x


# ---------------------------------------------------------------- SpecResNet
class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.skip = (nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
                     if stride != 1 or cin != cout else nn.Identity())

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(y + self.skip(x))


class SpecResNet(nn.Module):
    def __init__(self, widths=(32, 64, 128, 256), n_mels=80, dropout=0.3):
        super().__init__()
        self.frontend = LogMel(n_mels=n_mels)
        self.specaug = SpecAugment()
        self.stem = nn.Sequential(nn.Conv2d(1, widths[0], 3, 1, 1, bias=False), nn.BatchNorm2d(widths[0]), nn.ReLU())
        layers, cin = [], widths[0]
        for i, w in enumerate(widths):
            layers += [BasicBlock(cin, w, stride=1 if i == 0 else 2), BasicBlock(w, w)]
            cin = w
        self.layers = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(cin * 2, 1))
        self._feats = None

    def features(self, wav):
        x = self.specaug(self.frontend(wav))
        return self.layers(self.stem(x))  # [B, C, F', T']

    def forward(self, wav):
        f = self.features(wav)
        self._feats = f
        pooled = torch.cat([f.mean(dim=(2, 3)), f.amax(dim=(2, 3))], dim=1)
        return self.head(pooled).squeeze(-1)

    @torch.no_grad()
    def spectrogram(self, wav):
        return self.frontend(wav)


def grad_cam(model: SpecResNet, wav: torch.Tensor) -> np.ndarray:
    """Grad-CAM over the last conv block for one clip [1, T].
    Returns a [F', T'] heatmap in [0, 1] (upsample to the spectrogram to plot)."""
    model.eval()
    wav = wav.detach().requires_grad_(False)
    feats = model.features(wav)
    feats.retain_grad()
    pooled = torch.cat([feats.mean(dim=(2, 3)), feats.amax(dim=(2, 3))], dim=1)
    logit = model.head(pooled).squeeze()
    model.zero_grad()
    logit.backward()
    weights = feats.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * feats).sum(dim=1))[0]
    cam = cam / (cam.max() + 1e-8)
    return cam.detach().cpu().numpy()


# ---------------------------------------------------------------- XLS-R
class AttentiveStatsPool(nn.Module):
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.att = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x, mask=None):  # x [B, T, D]
        w = self.att(x).squeeze(-1)
        if mask is not None:
            w = w.masked_fill(~mask, -1e4)
        w = torch.softmax(w, dim=1).unsqueeze(-1)
        mu = (w * x).sum(1)
        sd = torch.sqrt(((w * (x - mu.unsqueeze(1)) ** 2).sum(1)).clamp(min=1e-6))
        return torch.cat([mu, sd], dim=1), w.squeeze(-1)


class XLSRDetector(nn.Module):
    """Frozen XLS-R + layer weights + attentive pooling + MLP.

    `unfreeze_top` > 0 fine-tunes that many top transformer layers too
    (more accurate, slower, more GPU memory).
    """

    def __init__(self, backbone="facebook/wav2vec2-xls-r-300m", unfreeze_top=0, proj=256, dropout=0.3):
        super().__init__()
        from transformers import Wav2Vec2Model

        self.backbone = Wav2Vec2Model.from_pretrained(backbone)
        self.backbone.config.mask_time_prob = 0.0
        # LayerDrop skips random layers in train mode, which changes how many
        # hidden states come back and breaks the learned layer weighting
        self.backbone.config.layerdrop = 0.0
        for p in self.backbone.parameters():
            p.requires_grad = False
        if unfreeze_top > 0:
            for layer in self.backbone.encoder.layers[-unfreeze_top:]:
                for p in layer.parameters():
                    p.requires_grad = True
        n_layers = self.backbone.config.num_hidden_layers + 1
        dim = self.backbone.config.hidden_size
        self.layer_weights = nn.Parameter(torch.zeros(n_layers))
        self.proj = nn.Sequential(nn.Linear(dim, proj), nn.GELU(), nn.Dropout(dropout))
        self.pool = AttentiveStatsPool(proj)
        self.head = nn.Sequential(nn.Linear(proj * 2, proj), nn.GELU(), nn.Dropout(dropout), nn.Linear(proj, 1))
        self.unfreeze_top = unfreeze_top
        self.last_attention = None

    def train(self, mode: bool = True):
        super().train(mode)
        if self.unfreeze_top == 0:
            self.backbone.eval()  # frozen backbone: no dropout, deterministic features
        return self

    def forward(self, wav):
        wav = (wav - wav.mean(dim=1, keepdim=True)) / (wav.std(dim=1, keepdim=True) + 1e-5)
        ctx = torch.enable_grad() if (self.unfreeze_top > 0 and self.training) else torch.no_grad()
        with ctx:
            out = self.backbone(wav, output_hidden_states=True)
        hs = torch.stack(out.hidden_states, dim=0)  # [L, B, T, D]
        w = torch.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
        x = self.proj((w * hs).sum(0))
        pooled, att = self.pool(x)
        self.last_attention = att.detach()
        return self.head(pooled).squeeze(-1)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


def build_model(name: str, **kw) -> nn.Module:
    if name == "specresnet":
        return SpecResNet(**kw)
    if name == "xlsr":
        return XLSRDetector(**kw)
    raise ValueError(f"unknown model {name!r}")


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, train


def sigmoid(x):
    return 1 / (1 + math.exp(-x))
