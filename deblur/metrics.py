"""Quality metrics.

  CER  - character error rate, the main metric
  WER  - the same for words
  PSNR - restored vs sharp image; used to pick alpha^2 on validation texts
  SSIM - structural similarity, saved in quality.csv
OCR is Tesseract through pytesseract.
"""
from __future__ import annotations

import re

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# edit (Levenshtein) distance; rapidfuzz is fast
try:
    from rapidfuzz.distance import Levenshtein as _Lev

    def _edit(a, b):
        return _Lev.distance(a, b)
except ImportError:  # slow fallback, plain dynamic programming
    def _edit(a, b):
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]


def psnr(ref: np.ndarray, x: np.ndarray) -> float:
    # 10*log10(255^2 / MSE), higher = closer to the sharp image
    return float(peak_signal_noise_ratio(ref.astype(np.float64), x.astype(np.float64), data_range=255))


def ssim(ref: np.ndarray, x: np.ndarray) -> float:
    # 1 = identical images
    return float(structural_similarity(ref.astype(np.float64), x.astype(np.float64), data_range=255))


def normalize_text(s: str) -> str:
    # same dashes and quotes, single spaces, lower case, so these are not counted as errors
    s = s.replace("–", "-").replace("–", "-")
    s = s.replace("“", '"').replace("”", '"').replace("„", '"').replace("‘", "'").replace("’", "'")
    return re.sub(r"\s+", " ", s).strip().lower()


def cer(ref_text: str, hyp_text: str) -> float:
    # edit distance / length of the true text; 0 = perfect, can be > 1 if OCR adds a lot of junk
    r, h = normalize_text(ref_text), normalize_text(hyp_text)
    return _edit(r, h) / max(1, len(r))


def wer(ref_text: str, hyp_text: str) -> float:
    r, h = normalize_text(ref_text).split(), normalize_text(hyp_text).split()
    return _edit(r, h) / max(1, len(r))


def ocr(img: np.ndarray, lang: str = "eng") -> str:
    # --psm 6: treat the image as one block of text
    import pytesseract
    x = np.clip(img, 0, 255).astype(np.uint8)
    return pytesseract.image_to_string(x, lang=lang, config="--psm 6")


def ocr_confidence(img: np.ndarray, lang: str = "eng") -> float:
    """Mean Tesseract confidence (0-100), for images without ground-truth text."""
    import pytesseract
    x = np.clip(img, 0, 255).astype(np.uint8)
    d = pytesseract.image_to_data(x, lang=lang, config="--psm 6", output_type=pytesseract.Output.DICT)
    conf = [float(c) for c, t in zip(d["conf"], d["text"]) if t.strip() and float(c) >= 0]
    return float(np.mean(conf)) if conf else 0.0