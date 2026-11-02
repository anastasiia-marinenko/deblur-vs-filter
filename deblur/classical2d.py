"""Classical restoration, 2D version: straight motion in any direction.

Differences from the 1D version (classical.py):
  1) the blur is a line of length L at any angle theta; both are found from the 2D cepstrum
  2) H is the FFT of that line kernel (like fspecial('motion', L, theta)), not the sinc formula;
     the filter itself is the same: W = H* / (|H|^2 + alpha^2)
  3) image edges are mirror-padded before filtering to reduce ringing

Why the cepstrum gives both L and theta: the spectrum of a line is sinc along the motion and
flat across it, so ln|G| is periodic along theta. Its 2D cepstrum IFFT2{ln|G|} has negative
peaks at (+-L*cos(theta), -+L*sin(theta)).
"""
from __future__ import annotations

import numpy as np

from .degradation import motion_kernel


def _hann2d(shape):
    # 2D Hann window: 1 in the centre, 0 at the edges
    return np.outer(np.hanning(shape[0]), np.hanning(shape[1]))


def cepstrum2d(g: np.ndarray) -> np.ndarray:
    # c = IFFT2{ ln|FFT2{g}| }, centred
    x = g.astype(np.float64)
    # the window removes the bright cross that sharp image borders leave in the spectrum
    x = (x - x.mean()) * _hann2d(x.shape)
    mag = np.abs(np.fft.fft2(x))
    return np.fft.fftshift(np.real(np.fft.ifft2(np.log(mag + 1e-8))))


def estimate_motion_2d(g: np.ndarray, l_min: float = 4.0, l_max: float | None = None):
    """Returns (L in px, theta in degrees, 0 <= theta < 180)."""
    c = cepstrum2d(g)
    h, w = c.shape
    cy, cx = h // 2, w // 2
    if l_max is None:
        l_max = min(h, w) / 4                       # 64 px for 512x256
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(yy - cy, xx - cx)
    # search only in a ring: closer than 4 px there is always a big hill around the centre
    mask = (r >= l_min) & (r <= l_max)
    # this line changes nothing in practice (r >= l_min already removes these points);
    # the horizontal axis itself stays in the search, that is where horizontal blur peaks
    mask &=(yy != cy) | (np.abs(xx - cx) >= l_min)
    cm = np.where(mask, c, np.inf)
    iy, ix = np.unravel_index(np.argmin(cm), cm.shape)
    dy, dx = iy - cy, ix - cx
    # sub-pixel refinement along each axis (parabola through 3 points)
    def refine(v, i, axis):
        if axis == 0 and 0 < iy < h - 1:
            y0, y1, y2 = c[iy - 1, ix], c[iy, ix], c[iy + 1, ix]
        elif axis == 1 and 0 < ix < w - 1:
            y0, y1, y2 = c[iy, ix - 1], c[iy, ix], c[iy, ix + 1]
        else:
            return float(v)
        den = y0 - 2 * y1 + y2
        return float(v) + (0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0)
    dy, dx = refine(dy, iy, 0), refine(dx, ix, 1)
    # L = distance to the peak, theta = its direction (image y goes down; theta and theta+180 are the same line)
    L = float(np.hypot(dx, dy))
    theta = float(np.degrees(np.arctan2(-dy, dx)) % 180.0)
    return L, theta


def psf_to_otf(psf: np.ndarray, shape) -> np.ndarray:
    """Small kernel -> transfer function of the image size, kernel centre moved to (0, 0) (like psf2otf)."""
    pad = np.zeros(shape, np.float64)
    kh, kw = psf.shape
    pad[:kh, :kw] = psf
    pad = np.roll(pad, (-(kh // 2), -(kw // 2)), axis=(0, 1))
    return np.fft.fft2(pad)


def deconv2d(g: np.ndarray, length_px: float, theta_deg: float, alpha2: float) -> np.ndarray:
    """2D pseudo-inverse filter with mirror padding."""
    g = g.astype(np.float64)
    m = int(np.ceil(length_px)) + 8
    # FFT treats the image as periodic; a mirrored border makes the wrap-around smooth (less ringing)
    gp = np.pad(g, m, mode="reflect")
    H = psf_to_otf(motion_kernel(length_px, theta_deg), gp.shape)
    W = np.conj(H) / (np.abs(H) ** 2 + alpha2)
    # filter, then cut the padding off
    f = np.real(np.fft.ifft2(np.fft.fft2(gp) * W))[m:-m, m:-m]
    return np.clip(f, 0.0, 255.0)


def restore_blind_2d(g: np.ndarray, alpha2: float, **kw):
    # fully automatic: L and theta from the cepstrum, then the filter
    L, th = estimate_motion_2d(g, **kw)
    return deconv2d(g, L, th, alpha2), L, th


__all__ = ["cepstrum2d", "estimate_motion_2d", "psf_to_otf", "deconv2d", "restore_blind_2d"]
