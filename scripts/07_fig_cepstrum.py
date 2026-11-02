"""Extra figure (not in the paper): centre of the 2D cepstrum of a blurred image
for three kinds of motion.

Uses real blurred images from the atlas (text test_000, sigma = 2) and the same cepstrum
function as the experiment. Straight motion gives a pair of clear negative (dark) peaks at
distance L from the centre along the motion; hand shake gives a blurry spot with no single peak.

Run after step 5:
  python scripts/07_fig_cepstrum.py --atlas atlas_out --id test_000 --lang en --out results/fig_cepstrum_en.jpg
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from deblur.classical2d import cepstrum2d, estimate_motion_2d  # noqa: E402

PICKS = [("sinc_L20_s2", {"uk": "(а) Горизонтальний рух, L = 20 пікс.", "en": "(a) Horizontal, L = 20 px"}),
         ("angle8_L20_s2", {"uk": "(б) Під кутом 8°, L = 20 пікс.", "en": "(b) At 8°, L = 20 px"}),
         ("shake_L20_s2", {"uk": "(в) Тремтіння рук, L = 20 пікс.", "en": "(c) Hand shake, L = 20 px"})]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", default="atlas_out")
    ap.add_argument("--id", default="test_000")
    ap.add_argument("--lang", default="uk")
    ap.add_argument("--half", type=int, default=32, help="half size of the shown area, px")
    ap.add_argument("--out", default="fig_cepstrum.jpg")
    a = ap.parse_args()
    lab = {"uk": "оцінка: L = {L:.1f} пікс., θ = {t:.1f}°", "en": "estimate: L = {L:.1f} px, θ = {t:.1f}°"}
    fig, axes = plt.subplots(1, len(PICKS), figsize=(6.6, 2.45))
    for ax, (cond, title) in zip(axes, PICKS):
        g = np.array(Image.open(os.path.join(a.atlas, "atlas", cond, f"{a.id}__blurred.png")).convert("L"),
                     dtype=np.float64)
        # same cepstrum as in the experiment (deblur/classical2d.py)
        c = cepstrum2d(g)
        h, w = c.shape
        cy, cx, k = h // 2, w // 2, a.half
        crop = c[cy - k:cy + k + 1, cx - k:cx + k + 1].copy()
        # central 65x65; the centre value is huge and would wash everything out, so hide it
        crop[k, k] = np.nan
        lo, hi = np.nanpercentile(crop, [0.5, 99.5])
        ax.imshow(crop, cmap="gray", vmin=lo, vmax=hi, extent=(-k - .5, k + .5, k + .5, -k - .5))
        # same L and theta estimate as in the experiment, shown under the panel
        L, t = estimate_motion_2d(g)
        if t > 90:
            t -= 180.0                            # theta and theta - 180 are the same line
        t = 0.0 if abs(t) < 0.05 else t
        ax.set_title(title[a.lang], fontsize=8, family="serif")
        txt = lab[a.lang].format(L=L, t=t)
        if a.lang == "uk":
            txt = txt.replace(f"{L:.1f}", f"{L:.1f}".replace(".", ",")).replace(f"{t:.1f}", f"{t:.1f}".replace(".", ","))
        ax.set_xlabel(txt.replace("-", "−"), fontsize=7.5, family="serif")
        ax.set_xticks([-k, 0, k]), ax.set_yticks([-k, 0, k])
        ax.tick_params(labelsize=6.5)
    fig.tight_layout(pad=0.3, w_pad=0.6)
    fig.savefig(a.out, dpi=300, pil_kwargs={"quality": 95})
    print("saved:", a.out)


if __name__ == "__main__":
    main()