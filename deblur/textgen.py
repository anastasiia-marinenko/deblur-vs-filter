"""Sharp text images with known ground-truth text.

We draw the text ourselves, so we know exactly what is written (needed for CER).
Text comes from texts/holmes_scenes.txt word by word (class Corpus).
WORDS_ENG / WORDS_UKR and random_line() are only a fallback; 01_make_dataset.py always passes a corpus.
"""
from __future__ import annotations

import glob
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# fallback word lists (not used when a corpus is given)
WORDS_ENG = (
    "the of and to in is that for it as was with be by on not he this are or his from at which "
    "but have an they you were her she there been one all we their has would when if so no will "
    "more other time some could these two may first then do any like my now over such our man me "
    "even most made after also did many before must through back years where much your way well "
    "down should because each just those people how too little state good very make world still own "
    "see men work long get here between both life being under never day same another know while "
    "last might us great old year off come since against go came right used take three energy image "
    "signal filter noise model data method result system network value pixel matrix spectrum "
    "frequency camera motion blur restore quality metric error test sample power heat server cloud"
).split()

WORDS_UKR = (
    "і в на що не з до як це та але він вона вони ми ви так вже ще був була було якщо коли де "
    "тут там дуже можна треба рік час день ніч місто людина робота життя слово світ земля вода "
    "сила точка лінія площа сигнал фільтр шум модель дані метод результат система мережа значення "
    "піксель матриця спектр частота камера рух розмиття якість помилка тест зразок енергія тепло "
    "сервер обчислення зображення текст відновлення оцінка параметр функція перетворення"
).split()

# only if matplotlib fonts are missing (on Kaggle they are always there)
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "C:/Windows/Fonts/times.ttf",
    "C:/Windows/Fonts/arial.ttf",
]


def find_fonts() -> list[str]:
    """DejaVu Serif and Sans shipped with matplotlib: same on every machine, so the images are reproducible."""
    import matplotlib
    d = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
    found = [os.path.join(d, f) for f in ("DejaVuSerif.ttf", "DejaVuSans.ttf") if os.path.exists(os.path.join(d, f))]
    if not found:
        found = [p for p in FONT_CANDIDATES if os.path.exists(p)]
    return found


def _load_font(path: str | None, size: int):
    if path:
        return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)   # Pillow >= 10.1


def random_line(rng: np.random.Generator, words: list[str], n_words: int) -> str:
    # not used in the experiment: random words, sometimes a number
    ws = list(rng.choice(words, size=n_words))
    if rng.random() < 0.5:
        ws.insert(int(rng.integers(0, len(ws))), str(int(rng.integers(10, 9999))))
    return " ".join(ws)


class Corpus:
    """Real text (passages from "The Adventures of Sherlock Holmes"), handed out word by word;
    starts again from the beginning when it runs out."""

    def __init__(self, path: str):
        self.words = open(path, encoding="utf-8").read().split()
        self.i = 0

    def next(self) -> str:
        if self.i == len(self.words):
            print("warning: corpus ended, starting again from the beginning")
        w = self.words[self.i % len(self.words)]
        self.i += 1
        return w

    def peek(self) -> str:
        # look at the next word without taking it
        return self.words[self.i % len(self.words)]


def make_text_image(rng: np.random.Generator, width: int = 512, height: int = 256,
                    lang: str = "eng", fonts: list[str] | None = None,
                    corpus: "Corpus | None" = None) -> tuple[np.ndarray, str]:
    """Black text on white, 4-6 lines. Returns (uint8 HxW image, ground-truth text)."""
    words = WORDS_UKR if lang == "ukr" else WORDS_ENG
    fonts = fonts if fonts is not None else find_fonts()
    # random but reproducible font and size 22-30
    font_path = fonts[int(rng.integers(0, len(fonts)))] if fonts else None
    size = int(rng.integers(22, 31))
    font = _load_font(font_path, size)
    img = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(img)
    margin_x, y = 16, 14
    line_h = int(size * 1.35)
    lines = []
    while y + line_h <= height - 8:
        # add words while the line still fits
        if corpus is not None:
            line = corpus.next()
            while draw.textlength(line + " " + corpus.peek(), font=font) <= width - 2 * margin_x:
                line += " " + corpus.next()
        else:
            n = int(rng.integers(3, 7))
            line = random_line(rng, words, n)
            while draw.textlength(line, font=font) > width - 2 * margin_x and len(line.split()) > 1:
                line = " ".join(line.split()[:-1])
        draw.text((margin_x, y), line, fill=0, font=font)
        lines.append(line)
        y += line_h
    return np.array(img, dtype=np.uint8), "\n".join(lines)
