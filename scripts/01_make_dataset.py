"""Step 1: sharp text images and their blurred versions for every condition.

Conditions:
  blur model : sinc   - exactly the 1D filter's model
               box    - same horizontal blur done in pixels (mean of L neighbours, mirrored edges)
               angle8 - straight motion at 8 degrees
               shake  - hand shake: curved path + phone-like processing
  L          : blur length, px: 10, 20, 32 (beta = pi * L / n)
  sigma      : noise std, grey levels: 0, 2, 5, 10
4 x 3 x 4 = 48 conditions for every sharp image.

Every test image goes through all 48 conditions (paired design). Validation images are separate
(text continues further in the corpus) and are only used to pick alpha^2.

Output in --out:
  sharp/val_000.png ... sharp/test_029.png  - 5 + 30 sharp images 512x256
  degraded/<condition>__<id>.png            - 35 x 48 = 1680 blurred (1440 of them test)
  meta.csv                                  - file, sharp id, split, model, L, sigma, true beta
  gt_text.json                              - ground-truth text of every sharp image

  python scripts/01_make_dataset.py --out data --n-test 30 --n-val 5 --corpus texts/holmes_scenes.txt
"""
import argparse
import json
import os
import sys
import zlib

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from deblur.degradation import degrade          # noqa: E402
from deblur.textgen import Corpus, make_text_image, find_fonts  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--n-test", type=int, default=10)
    ap.add_argument("--n-val", type=int, default=3)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--models", default="sinc,box,angle8,shake")
    ap.add_argument("--lengths", default="10,20,32")
    ap.add_argument("--sigmas", default="0,2,5,10")
    ap.add_argument("--lang", default="eng", choices=["eng", "ukr"])
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--corpus", default="", help="text file; empty = random words")
    a = ap.parse_args()

    # fixed seed: same images every run (the notebook checks an md5)
    rng = np.random.default_rng(a.seed)
    fonts = find_fonts()
    print("fonts:", fonts or "Pillow default")
    os.makedirs(os.path.join(a.out, "sharp"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "degraded"), exist_ok=True)

    # 1. sharp images: validation first, then test; text runs on, so no overlap
    corpus = Corpus(a.corpus) if a.corpus else None
    gt, sharp_rows = {}, []
    for split, n in (("val", a.n_val), ("test", a.n_test)):
        for i in range(n):
            sid = f"{split}_{i:03d}"
            img, text = make_text_image(rng, a.width, a.height, a.lang, fonts, corpus)
            Image.fromarray(img).save(os.path.join(a.out, "sharp", sid + ".png"))
            gt[sid] = text
            sharp_rows.append((sid, split))

    # 2. blurred versions: every sharp image x 4 models x 3 lengths x 4 noise levels
    models = a.models.split(",")
    lengths = [float(x) for x in a.lengths.split(",")]
    sigmas = [float(x) for x in a.sigmas.split(",")]
    rows = []
    for sid, split in sharp_rows:
        img = np.array(Image.open(os.path.join(a.out, "sharp", sid + ".png")))
        for m in models:
            for L in lengths:
                for s in sigmas:
                    cond = f"{m}_L{int(L)}_s{int(s)}"          # e.g. "angle8_L20_s2"
                    # own seed per (image, condition), so noise and shake path don't depend on order
                    g, beta = degrade(img, m, L, s, np.random.default_rng(zlib.crc32(f"{a.seed}|{sid}|{cond}".encode())))
                    fn = f"{cond}__{sid}.png"
                    Image.fromarray(g).save(os.path.join(a.out, "degraded", fn))
                    rows.append(dict(file=fn, sharp_id=sid, split=split, blur_model=m, length_px=L,
                                     sigma=s, beta_true=beta, condition=cond))
    # 3. table of blurred images and the true texts
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "meta.csv"), index=False)
    with open(os.path.join(a.out, "gt_text.json"), "w", encoding="utf-8") as f:
        json.dump(gt, f, ensure_ascii=False, indent=1)
    print(f"done: {len(sharp_rows)} sharp, {len(rows)} blurred images in {a.out}")


if __name__ == "__main__":
    main()