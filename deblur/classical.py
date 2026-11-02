"""Classical restoration, 1D version (as in our earlier paper [2]).

1) pseudo-inverse filter, formula (2):
       W = H* / (|H|^2 + alpha^2),   F^ = W * G,   H = sin(beta*wx)/(beta*wx)
2) blind estimate of the blur length L from the cepstrum, so beta is not set by hand.
"""
from __future__ import annotations

import numpy as np

from .degradation import sinc_mtf, beta_to_length, length_to_beta

# alpha^2 values we try: 1e-8 ... 1, one order of magnitude apart
ALPHA2_GRID = [10.0 ** p for p in range(-8, 1)]   # 1e-8 ... 1


def pseudo_inverse(g: np.ndarray, beta: float, alpha2: float) -> np.ndarray:
    """Pseudo-inverse filter; output clipped to [0, 255]."""
    g = g.astype(np.float64)
    spec = np.fft.fftshift(np.fft.fft2(g))
    H = sinc_mtf(g.shape, beta)
    W = np.conj(H) / (np.abs(H) ** 2 + alpha2)
    f = np.real(np.fft.ifft2(np.fft.ifftshift(spec * W)))
    return np.clip(f, 0.0, 255.0)


def estimate_length_cepstrum(g: np.ndarray, l_min: float = 3.0, l_max: float | None = None) -> float:
    """Blind estimate of the horizontal blur length L (px) from the cepstrum.

    Zeros of sinc(beta*k) repeat every pi/beta, so log|spectrum| has a periodic part
    and its inverse FFT (the cepstrum) has a negative peak at q = L.
    """
    x = g.astype(np.float64)
    x = x - x.mean()                       # remove the big zero frequency
    h, w = x.shape
    # |FFT| of every row, averaged over rows: the blur is the same in each row,
    # averaging keeps it and smooths out letters and noise
    mag = np.abs(np.fft.fft(x, axis=1)).mean(axis=0)
    ceps = np.real(np.fft.ifft(np.log(mag + 1e-8)))
    # deepest minimum between 3 px and a quarter of the width
    if l_max is None:
        l_max = w / 4
    lo, hi = int(np.floor(l_min)), int(np.ceil(l_max))
    seg = ceps[lo:hi + 1]
    i = int(np.argmin(seg))
    q = float(lo + i)
    # parabola through the minimum and its neighbours -> sub-pixel L
    if 0 < i < len(seg) - 1:
        y0, y1, y2 = seg[i - 1], seg[i], seg[i + 1]
        den = y0 - 2 * y1 + y2
        if abs(den) > 1e-12:
            q += 0.5 * (y0 - y2) / den
    return q


def estimate_beta(g: np.ndarray, **kw) -> float:
    return length_to_beta(estimate_length_cepstrum(g, **kw), g.shape[1])


def restore_oracle(g: np.ndarray, beta_true: float, alpha2: float) -> np.ndarray:
    """Filter with the true beta (upper bound for the classical method)."""
    return pseudo_inverse(g, beta_true, alpha2)


def restore_blind(g: np.ndarray, alpha2: float) -> tuple[np.ndarray, float]:
    """Fully automatic: beta from the cepstrum + pseudo-inverse filter."""
    beta = estimate_beta(g)
    return pseudo_inverse(g, beta, alpha2), beta


__all__ = ["ALPHA2_GRID", "pseudo_inverse", "estimate_length_cepstrum", "estimate_beta",
           "restore_oracle", "restore_blind", "beta_to_length", "length_to_beta"]
