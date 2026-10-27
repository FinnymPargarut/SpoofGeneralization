"""LFCC-LCNN of Wang, Yamagishi 2021, ported from the NII code of ASVspoof 2021 baseline B03.

LFCC follows the paper, not B03: full band and a 512-point FFT instead of 0-4 kHz and 1024.
"""

import numpy as np
import torch
from scipy.fft import dct
from torch import nn
from torch.nn.functional import pad

from spoof.audio import SR


def _triangle(f: torch.Tensor, a: float, b: float, c: float) -> torch.Tensor:
    """trimf of the NII code."""
    y = torch.zeros_like(f)
    rise = (a < f) & (f < b)
    y[rise] = (f[rise] - a) / (b - a)
    fall = (b < f) & (f < c)
    y[fall] = (c - f[fall]) / (c - b)
    y[f == b] = 1
    return y


def _delta(x: torch.Tensor) -> torch.Tensor:
    """Difference of the next and previous frames, edge frames repeated: (batch, frames, dim)."""
    padded = pad(x.unsqueeze(1), (0, 0, 1, 1), mode="replicate").squeeze(1)
    return padded[:, 2:] - padded[:, :-2]


class LFCC(nn.Module):
    """(batch, samples) at 16 kHz to (batch, frames, 3 * filters): LFCC, deltas, delta-deltas.

    The first coefficient is replaced by log frame energy.
    """

    def __init__(self, frame=320, shift=160, n_fft=512, filters=20):
        super().__init__()
        self.shift = shift
        self.n_fft = n_fft
        f = (SR / 2) * torch.linspace(0, 1, n_fft // 2 + 1)
        edges = torch.linspace(f.min(), f.max(), filters + 2)
        bank = torch.zeros(f.numel(), filters)
        for i in range(filters):
            bank[:, i] = _triangle(f, edges[i], edges[i + 1], edges[i + 2])
        self.register_buffer("bank", bank)
        # x @ dct_matrix is the orthonormal DCT-II of x along the last axis.
        dct_matrix = dct(np.eye(filters), norm="ortho", axis=-1)
        self.register_buffer("dct", torch.from_numpy(dct_matrix).float())
        self.register_buffer("window", torch.hamming_window(frame))

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        emphasized = torch.cat([wav[:, :1], wav[:, 1:] - 0.97 * wav[:, :-1]], dim=1)
        spec = torch.stft(
            emphasized,
            self.n_fft,
            self.shift,
            self.window.numel(),
            window=self.window,
            pad_mode="constant",
            return_complex=True,
        )
        power = spec.abs().pow(2).transpose(1, 2)
        eps = torch.finfo(torch.float32).eps
        cepstra = torch.log10(power @ self.bank + eps) @ self.dct
        energy = torch.log10(power.sum(-1) / self.n_fft + eps)
        lfcc = torch.cat([energy.unsqueeze(-1), cepstra[:, :, 1:]], dim=-1)
        first = _delta(lfcc)
        return torch.cat([lfcc, first, _delta(first)], dim=-1)


class MaxFeatureMap(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return torch.maximum(a, b)


def _cnn() -> nn.Sequential:
    def conv(in_ch, out_ch, kernel):
        return [nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2), MaxFeatureMap()]

    def norm(ch):
        return nn.BatchNorm2d(ch, affine=False)

    return nn.Sequential(
        *conv(1, 64, 5),
        nn.MaxPool2d(2),
        *conv(32, 64, 1),
        norm(32),
        *conv(32, 96, 3),
        nn.MaxPool2d(2),
        norm(48),
        *conv(48, 96, 1),
        norm(48),
        *conv(48, 128, 3),
        nn.MaxPool2d(2),
        *conv(64, 128, 1),
        norm(64),
        *conv(64, 64, 3),
        norm(32),
        *conv(32, 64, 1),
        norm(32),
        *conv(32, 64, 3),
        nn.MaxPool2d(2),
        nn.Dropout(0.7),
    )


class LCNN(nn.Module):
    """Takes 16 kHz waveforms (batch, samples), returns a score per trial: higher is bona fide."""

    def __init__(self):
        super().__init__()
        self.lfcc = LFCC()
        self.cnn = _cnn()
        # 32 channels times 60 LFCC dims after four poolings by 2.
        hidden = 32 * (60 // 16)
        self.blstm1 = nn.LSTM(hidden, hidden // 2, batch_first=True, bidirectional=True)
        self.blstm2 = nn.LSTM(hidden, hidden // 2, batch_first=True, bidirectional=True)
        self.out = nn.Linear(hidden, 1)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.lfcc(wav)
        h = self.cnn(features.unsqueeze(1)).permute(0, 2, 1, 3).flatten(2)
        lstm, _ = self.blstm1(h)
        lstm, _ = self.blstm2(lstm)
        return self.out((lstm + h).mean(1)).squeeze(1)
