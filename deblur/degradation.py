"""Blur models: how a sharp text image f becomes a blurred image g.

Main model (horizontal motion), formula (1) in the paper:
    H(wx, wy) = sin(beta * k) / (beta * k),  k = column index from the centre of the spectrum
Blur length in pixels: L = beta * n / pi, so beta = pi * L / n (n = image width).
For n = 512 and beta = 0.194, L is about 31.6 px.

Four blur models (argument `model` of degrade() at the bottom):
    "sinc"   - multiply the spectrum by H, exactly formula (1)
    "box"    - same horizontal motion, but each pixel = mean of L neighbours
    "angle8" - straight motion at 8 degrees
    "shake"  - hand shake: random curved path + phone-like processing (gamma, noise, JPEG)
"""
from __future__ import annotations

import numpy as np


# ---- 1. sinc model

def freq_index(width: int) -> np.ndarray:
    """Column numbers from the centre: -n//2 ... n - n//2 - 1 (zero in the middle, like after fftshift)."""
    return np.arange(width) - (width // 2)


def sinc_mtf(shape: tuple[int, int], beta: float) -> np.ndarray:
    """H = sin(beta*k)/(beta*k), centred (fftshift)."""
    h, w = shape
    k = freq_index(w).astype(np.float64)
    # np.sinc(x) is sin(pi x)/(pi x), so divide by pi; at k = 0 it returns 1
    hk = np.sinc(beta * k / np.pi)
    # horizontal motion: H depends on wx only, same row for every image row
    return np.broadcast_to(hk[None, :], (h, w)).copy()


def beta_to_length(beta: float, width: int) -> float:
    return beta * width / np.pi


def length_to_beta(length_px: float, width: int) -> float:
    return np.pi * length_px / width


def blur_sinc(img: np.ndarray, beta: float) -> np.ndarray:
    """G = H * F in the frequency domain (circular convolution)."""
    f = img.astype(np.float64)
    spec = np.fft.fftshift(np.fft.fft2(f))
    # imaginary part is only rounding noise
    g = np.real(np.fft.ifft2(np.fft.ifftshift(spec * sinc_mtf(f.shape, beta))))
    return g


# ---- 2. blur in pixels (convolution with the point spread function)

def blur_box(img: np.ndarray, length_px: int) -> np.ndarray:
    """Horizontal box kernel of length L with mirrored edges.
    Not exactly the filter's model, so it is a harder test for the methods."""
    import cv2
    L = max(1, int(round(length_px)))
    # L values of 1/L: each pixel becomes the mean of L neighbours in its row
    kernel = np.full((1, L), 1.0 / L, dtype=np.float64)
    # mirrored edges (the sinc model wraps around instead)
    return cv2.filter2D(img.astype(np.float64), -1, kernel, borderType=cv2.BORDER_REFLECT)


def motion_kernel(length_px: float, angle_deg: float) -> np.ndarray:
    """Straight line of length L at an angle (anti-aliased), like fspecial('motion', L, theta)."""
    import cv2
    L = max(1.0, float(length_px))
    size = int(np.ceil(L)) + 2
    size += (size + 1) % 2                     # odd size, so there is a centre pixel
    scale = 8                                  # draw 8x bigger, then shrink = anti-aliasing
    big = np.zeros((size * scale, size * scale), np.float32)
    c = (size * scale - 1) / 2.0
    # line ends: L/2 from the centre each way; minus because image y goes down
    dx = np.cos(np.deg2rad(angle_deg)) * (L - 1) * scale / 2.0
    dy = -np.sin(np.deg2rad(angle_deg)) * (L - 1) * scale / 2.0
    cv2.line(big, (int(round(c - dx)), int(round(c - dy))), (int(round(c + dx)), int(round(c + dy))),
             1.0, thickness=scale, lineType=cv2.LINE_AA)
    k = cv2.resize(big, (size, size), interpolation=cv2.INTER_AREA).astype(np.float64)
    return k / k.sum()                          # kernel sums to 1


def blur_angle(img: np.ndarray, length_px: float, angle_deg: float) -> np.ndarray:
    """Motion at an angle: the horizontal filter model is no longer exact."""
    import cv2
    return cv2.filter2D(img.astype(np.float64), -1, motion_kernel(length_px, angle_deg),
                        borderType=cv2.BORDER_REFLECT)


# ---- 3. hand shake: random curved path

def shake_kernel(length_px: float, rng: np.random.Generator, n_steps: int = 400,
                 anxiety: float = 0.005) -> np.ndarray:
    """Kernel of a random curved camera path.
    The direction changes smoothly with rare sharp jerks; the path is scaled so that its
    largest span is L px. It is not a line, so no classical filter knows its shape."""
    # position as a complex number x + jy
    L = max(2.0, float(length_px))
    v = np.exp(1j * rng.uniform(0, 2 * np.pi))       # start direction
    x = np.zeros(n_steps, complex)                   # 400 camera positions during exposure
    for t in range(1, n_steps):
        dv = rng.normal() + 1j * rng.normal()
        if rng.random() < anxiety * 10:            # 5 % chance of a jerk 8x stronger
            dv *= 8
        # mostly keep the old direction, so the path is a smooth curve
        v = 0.995 * v + 0.06 * dv
        v /= max(abs(v), 1e-9)                       # step length stays 1
        x[t] = x[t - 1] + v
    x -= x.mean()
    # scale the larger of the x and y spans to L - 1 px
    span = max(np.ptp(x.real), np.ptp(x.imag), 1e-9)
    x *= (L - 1) / span
    size = int(np.ceil(L)) + 4
    size += (size + 1) % 2
    k = np.zeros((size, size))
    c = (size - 1) / 2.0
    # each path point adds weight 1, split between 4 neighbour pixels (bilinear)
    for z in x:
        px, py = z.real + c, z.imag + c
        i0, j0 = int(np.floor(py)), int(np.floor(px))
        fy, fx = py - i0, px - j0
        for di, wy in ((0, 1 - fy), (1, fy)):
            for dj, wx in ((0, 1 - fx), (1, fx)):
                if 0 <= i0 + di < size and 0 <= j0 + dj < size:
                    k[i0 + di, j0 + dj] += wy * wx
    return k / k.sum()


# ---- 4. phone-like processing (shake only)

def _to_linear(x):
    # sRGB -> linear light (IEC 61966-2-1)
    x = np.clip(x, 0, 255) / 255.0
    return 255.0 * np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _to_srgb(x):
    # linear light -> sRGB (gamma)
    x = np.clip(x, 0, 255) / 255.0
    return 255.0 * np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


def camera_like(img: np.ndarray, kernel: np.ndarray, sigma: float, rng: np.random.Generator,
                jpeg_quality: int = 85) -> np.ndarray:
    """Rough smartphone photo: blur in linear light, sRGB gamma, sensor noise, 8 bit, JPEG."""
    import cv2
    # the sensor collects light (linear) while moving
    lin = cv2.filter2D(_to_linear(img.astype(np.float64)), -1, kernel, borderType=cv2.BORDER_REFLECT)
    g = add_noise_and_quantize(_to_srgb(lin), sigma, rng)
    # save as JPEG (quality 85) and read back to get compression artefacts
    ok, buf = cv2.imencode(".jpg", g, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)


def add_noise_and_quantize(g: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Gaussian noise with std sigma (grey levels 0..255), then round to uint8."""
    noisy = g + rng.normal(0.0, sigma, size=g.shape) if sigma > 0 else g
    return np.clip(np.round(noisy), 0, 255).astype(np.uint8)


# ---- 5. main entry

def degrade(img: np.ndarray, model: str, length_px: float, sigma: float,
            rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """Returns (blurred uint8 image, true beta).
    True beta is saved so we can check the cepstrum estimate and run the filter with known beta."""
    width = img.shape[1]
    if model == "sinc":
        beta = length_to_beta(length_px, width)
        g = blur_sinc(img, beta)
    elif model == "box":
        L = int(round(length_px))
        beta = length_to_beta(L, width)
        g = blur_box(img, L)
    elif model.startswith("angle"):          # e.g. "angle8"
        angle = float(model[len("angle"):] or 8)
        # nominal beta = horizontal projection of the motion
        beta = length_to_beta(length_px * np.cos(np.deg2rad(angle)), width)
        g = blur_angle(img, length_px, angle)
    elif model == "shake":
        beta = length_to_beta(length_px, width)   # nominal only, there is no true beta here
        return camera_like(img, shake_kernel(length_px, rng), sigma, rng), beta
    else:
        raise ValueError(f"unknown blur model: {model}")
    # shake adds noise inside camera_like, the others add it here
    return add_noise_and_quantize(g, sigma, rng), beta
