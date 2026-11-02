"""Step 8: every number used in the paper.

Only reads the result files (quality.csv, efficiency.csv, stats_paired.csv)
and prints each number with a short note on how it was obtained.

Run after step 4:  python scripts/08_paper_numbers.py --res results
"""
import argparse
import os

import numpy as np
import pandas as pd

METHODS = ["blurred", "pinv_blind", "pinv2d_blind", "dgv2_mobilenet", "dgv2_inception"]
NAMES = {"blurred": "No restoration", "pinv_blind": "Pseudo-inverse 1D", "pinv2d_blind": "Pseudo-inverse 2D",
         "dgv2_mobilenet": "DeblurGAN-v2 MobileNet", "dgv2_inception": "DeblurGAN-v2 Inception"}
# sinc and box are both horizontal motion
GROUP = {"sinc": "horizontal", "box": "horizontal", "angle8": "angle 8°", "shake": "hand shake"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results")
    a = ap.parse_args()
    q = pd.read_csv(os.path.join(a.res, "quality.csv"))
    eff = pd.read_csv(os.path.join(a.res, "efficiency.csv"))
    q["grp"] = q.blur_model.map(GROUP)
    global METHODS
    METHODS = [m for m in METHODS if m in set(q.method)]   # in case a network did not load
    pd.set_option("display.width", 200)

    print("=" * 100)
    print("0. Check: Tesseract CER on the sharp images (mean over 30 texts), %:",
          round(100 * q.drop_duplicates("sharp_id").cer_sharp.mean(), 2))

    print("=" * 100)
    print("TABLE 1. CER, % - mean over all images of the group (30 texts x 3 lengths x 4 noise levels;"
          "\n horizontal also x 2 models). Images per group:",
          q[q.method == "blurred"].groupby("grp").size().to_dict())
    t1 = (100 * q[q.method.isin(METHODS)].groupby(["method", "grp"]).cer.mean()).unstack().loc[METHODS]
    t1.index = [NAMES[m] for m in METHODS]
    print(t1[["horizontal", "angle 8°", "hand shake"]].round(1).to_string())
    print("\nTime (ms per image) and energy (J per image, whole machine) from efficiency.csv:")
    print(eff[["method", "device", "ms_per_img_mean", "gross_J_per_img", "cpu_model", "gpu_model",
               "cpu_tracking"]].round(2).to_string(index=False))

    print("=" * 100)
    print("TABLE 2. CER, %, horizontal motion, by noise sigma:")
    h = q[(q.grp == "horizontal") & q.method.isin(METHODS)]
    t2 = (100 * h.groupby(["method", "sigma"]).cer.mean()).unstack().loc[METHODS]
    t2.index = [NAMES[m] for m in METHODS]
    print(t2.round(2).to_string())

    print("=" * 100)
    print("BLUR PARAMETER ERRORS (median |estimate - true|):")
    b = q[q.method == "pinv_blind"]
    e1 = (b.beta_est * 512 / np.pi - b.length_px).abs()       # 1D: beta -> L in px
    print("  1D, L, px:", e1.groupby(b.blur_model).median().round(3).to_dict())
    d = q[q.method == "pinv2d_blind"]
    eL = (d.len2d_est - d.length_px).abs()
    ang = d.angle2d_est % 180
    true = np.where(d.blur_model == "angle8", 8.0, 0.0)
    eA = np.minimum(np.abs(ang - true), 180 - np.abs(ang - true))
    print("  2D, L, px:", eL.groupby(d.blur_model).median().round(3).to_dict())
    print("  2D, theta, deg (means nothing for shake):", pd.Series(eA, index=d.index).groupby(d.blur_model)
          .median().round(3).to_dict())

    print("=" * 100)
    print("HAND SHAKE by length L, CER, % (Fig. 1c):")
    s = q[(q.blur_model == "shake") & q.method.isin(METHODS)]
    print((100 * s.groupby(["length_px", "method"]).cer.mean()).unstack()[METHODS].round(1).to_string())
    per_text = s.groupby(["sharp_id", "method"]).cer.mean().unstack()   # 30 texts x method (mean over 12 conditions)
    if "dgv2_inception" in per_text:
        print("  texts where Inception beats the 2D filter:",
              int((per_text.dgv2_inception < per_text.pinv2d_blind).sum()), "of", len(per_text))

    print("=" * 100)
    print("PAIRED COMPARISONS BY TEXT (stats_paired.csv, CER): share of texts where the first method is better, and p:")
    sp = pd.read_csv(os.path.join(a.res, "stats_paired.csv"))
    sp = sp[(sp.metric == "cer") & ~sp["compare"].str.contains("oracle")]
    print(sp[["group", "compare", "mean_a", "mean_b", "share_a_better", "p_wilcoxon"]].round(4).to_string(index=False))

    print("=" * 100)
    print("SSIM (for reference):")
    print(q[q.method.isin(METHODS)].groupby(["grp", "method"]).ssim.mean().unstack()[METHODS].round(3).to_string())


if __name__ == "__main__":
    main()