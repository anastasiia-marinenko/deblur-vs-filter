"""Fig. 1 of the paper: CER vs blur length for three kinds of motion.
Made from results/quality.csv; black and white, JPG 300 dpi.

  python scripts/06_fig_cer_vs_length.py --res results --lang en --out results/paper_fig1_en.jpg
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

T = {"en": dict(groups={"horiz": "(a) Straight horizontal motion", "angle8": "(b) Straight motion at 8°",
                        "shake": "(c) Hand shake"},
                x="Blur length L, px", y="Character error rate, %",
                m={"blurred": "No restoration", "pinv_blind": "Pseudo-inverse 1D", "pinv2d_blind": "Pseudo-inverse 2D",
                   "dgv2_mobilenet": "DeblurGAN-v2 MobileNet", "dgv2_inception": "DeblurGAN-v2 Inception"}),
     "uk": dict(groups={"horiz": "(а) Прямий горизонтальний рух", "angle8": "(б) Прямий рух під кутом 8°",
                        "shake": "(в) Тремтіння рук"},
                x="Довжина змазу L, пікс.", y="Частка помилкових символів, %",
                m={"blurred": "Без відновлення", "pinv_blind": "Псевдоінверсний 1D",
                   "pinv2d_blind": "Псевдоінверсний 2D", "dgv2_mobilenet": "DeblurGAN-v2 MobileNet",
                   "dgv2_inception": "DeblurGAN-v2 Inception"})}
STYLE = {"blurred": ("0.6", ":", "x"), "pinv_blind": ("0.0", "-", "o"), "pinv2d_blind": ("0.0", "--", "v"),
         "dgv2_mobilenet": ("0.45", "-.", "^"), "dgv2_inception": ("0.3", "-", "D")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--out", default="paper_fig1.jpg")
    ap.add_argument("--size", default="6.6,2.3", help="width,height in inches")
    a = ap.parse_args()
    t = T[a.lang]
    q = pd.read_csv(f"{a.res}/quality.csv")
    # sinc and box are both horizontal motion -> one panel; angle8 and shake separate
    q["grp"] = q.blur_model.map({"sinc": "horiz", "box": "horiz", "angle8": "angle8", "shake": "shake"})
    plt.rcParams.update({"font.family": "serif", "font.size": 7.5})
    fig, axes = plt.subplots(1, 3, figsize=tuple(float(v) for v in a.size.split(",")), sharey=True)
    for ax, g in zip(axes, ["horiz", "angle8", "shake"]):
        # mean CER (%) per method and L, over 30 texts and 4 noise levels
        d = q[q.grp == g].groupby(["method", "length_px"]).cer.mean().reset_index()
        for m, (c, ls, mk) in STYLE.items():
            dm = d[d.method == m]
            ax.plot(dm.length_px, 100 * dm.cer, color=c, ls=ls, marker=mk, ms=3.5, lw=1.1, label=t["m"][m])
        ax.set_title(t["groups"][g], fontsize=7.5)
        ax.set_xticks([10, 20, 32])
        ax.set_xlabel(t["x"])
        ax.grid(alpha=.3)
        ax.set_ylim(0, 100)
    axes[0].set_ylabel(t["y"])
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, fontsize=6.8, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.15, 1, 1), w_pad=0.6)
    fig.savefig(a.out, dpi=300, pil_kwargs={"quality": 95})
    fig.savefig(a.out.rsplit(".", 1)[0] + ".png", dpi=200)


if __name__ == "__main__":
    main()