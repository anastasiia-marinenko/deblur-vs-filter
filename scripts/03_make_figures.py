"""Step 3: extra figures and the summary table, to look through the results.

Quality vs blur length and noise, quality vs energy, restoration examples,
validation PSNR vs alpha^2. Table goes to results/table_main.csv (.md).

  python scripts/03_make_figures.py --data data --res results
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

# labels
SHORT = {"blurred": "Blurred", "pinv_oracle": "Pseudo-inverse\n(known β)", "pinv_blind": "Pseudo-inverse 1D\n(estimated)",
         "pinv2d_blind": "Pseudo-inverse 2D\n(estimated)",
         "dgv2_mobilenet": "DeblurGAN-v2\nMobileNet", "dgv2_inception": "DeblurGAN-v2\nInception"}
LABELS = {"blurred": "Blurred", "pinv_oracle": "Pseudo-inverse (known β)",
          "pinv_blind": "Pseudo-inverse 1D (estimated β)", "pinv2d_blind": "Pseudo-inverse 2D (estimated L, θ)",
          "dgv2_mobilenet": "DeblurGAN-v2 MobileNet", "dgv2_inception": "DeblurGAN-v2 Inception"}
STYLE = {"blurred": ("0.65", ":", "x"), "pinv_oracle": ("0.0", "--", "s"),
         "pinv_blind": ("0.0", "-", "o"), "pinv2d_blind": ("0.0", "-", "v"), "dgv2_mobilenet": ("0.45", "-.", "^"),
         "dgv2_inception": ("0.3", "-", "D")}
MODEL_UA = {"sinc": "sinc model", "box": "box kernel", "angle8": "motion at 8°",
            "shake": "hand shake + JPEG"}

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.grid": True, "grid.color": "0.85"})


def save(fig, out, name):
    if not fig.get_constrained_layout():
        fig.tight_layout()
    fig.savefig(os.path.join(out, name + ".jpg"), dpi=300, pil_kwargs={"quality": 95})
    fig.savefig(os.path.join(out, name + ".png"), dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--res", default="results")
    ap.add_argument("--example-file", default=None, help="blurred image file for the examples figure")
    a = ap.parse_args()
    out = os.path.join(a.res, "figures")
    os.makedirs(out, exist_ok=True)
    q = pd.read_csv(os.path.join(a.res, "quality.csv"))
    methods = [m for m in LABELS if m in q.method.unique()]
    metric_ocr = "cer" in q

    # quality vs blur length, one panel per blur model
    for metric, ylab in (("psnr", "PSNR, dB"), ("cer", "Character error rate")):
        if metric not in q:
            continue
        bms = sorted(q.blur_model.unique(), key=lambda s: list(MODEL_UA).index(s) if s in MODEL_UA else 9)
        fig, axes = plt.subplots(1, len(bms), figsize=(2.6 * len(bms), 2.5), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, bm in zip(axes, bms):
            d = q[q.blur_model == bm].groupby(["method", "length_px"])[metric].mean().reset_index()
            for m in methods:
                dm = d[d.method == m]
                c, ls, mk = STYLE[m]
                ax.plot(dm.length_px, dm[metric], color=c, ls=ls, marker=mk, ms=4, label=LABELS[m])
            ax.set_title(MODEL_UA.get(bm, bm))
            ax.set_xlabel("Blur length L, px")
        axes[0].set_ylabel(ylab)
        axes[-1].legend(fontsize=6.5, loc="best")
        save(fig, out, f"fig1_{metric}_vs_length")

    # quality vs noise
    metric = "cer" if metric_ocr else "psnr"
    d = q.groupby(["method", "sigma"])[metric].mean().reset_index()
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for m in methods:
        dm = d[d.method == m]
        c, ls, mk = STYLE[m]
        ax.plot(dm.sigma, dm[metric], color=c, ls=ls, marker=mk, ms=4, label=LABELS[m])
    ax.set_xlabel("Noise σ, grey levels")
    ax.set_ylabel("CER" if metric_ocr else "PSNR, dB")
    ax.legend(fontsize=6.5)
    save(fig, out, f"fig2_{metric}_vs_noise")

    # quality vs energy
    eff_path = os.path.join(a.res, "efficiency.csv")
    if os.path.exists(eff_path):
        eff = pd.read_csv(eff_path)
        eff_cpu = eff[eff.device == "cpu"].set_index("method")
        cond = eff.get("condition", None)
        mq = q.groupby("method")[[c for c in ("psnr", "cer") if c in q]].mean()
        fig, ax = plt.subplots(figsize=(3.4, 2.6))
        yk = "cer" if metric_ocr else "psnr"
        for m in methods:
            if m in eff_cpu.index:
                c, _, mk = STYLE[m]
                x = max(eff_cpu.loc[m, "gross_J_per_img"], 1e-4)
                ax.scatter(x, mq.loc[m, yk], color=c, marker=mk, s=40, label=LABELS[m], zorder=3)
        ax.set_xscale("log")
        ax.set_xlabel("Energy per image (CPU), J")
        ax.set_ylabel("Mean CER" if yk == "cer" else "Mean PSNR, dB")
        ax.legend(fontsize=6.5)
        save(fig, out, "fig3_quality_vs_energy")
        del cond

    # restoration examples
    test = q[q.method == "blurred"]
    ex_files = [a.example_file] if a.example_file else []
    if not ex_files:
        for cond in ("sinc_L32_s2", "angle8_L20_s5", "shake_L20_s2"):
            f = test[test.condition == cond].file
            if len(f):
                ex_files.append(f.iloc[0])
        if not ex_files:   # other conditions: take any two
            ex_files = list(test.drop_duplicates("condition").file.iloc[[0, -1]])
    cols = ["sharp"] + methods
    fig, axes = plt.subplots(len(ex_files), len(cols), figsize=(1.25 * len(cols), 1.3 * len(ex_files) + 0.3),
                             gridspec_kw={"wspace": 0.05, "hspace": 0.15})
    axes = np.atleast_2d(axes)
    for i, fn in enumerate(ex_files):
        sid = fn.split("__")[1].replace(".png", "")
        for j, c in enumerate(cols):
            p = os.path.join(a.data, "sharp", sid + ".png") if c == "sharp" else os.path.join(a.res, "restored", c, fn)
            img = np.array(Image.open(p))
            ax = axes[i, j]
            ax.imshow(img[:, : img.shape[1] // 2], cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([]), ax.set_yticks([])
            if i == 0:
                ax.set_title("Sharp" if c == "sharp" else SHORT.get(c, LABELS[c]), fontsize=6.5)
        axes[i, 0].set_ylabel(fn.split("__")[0], fontsize=6)
    save(fig, out, "fig4_examples")

    # validation PSNR vs alpha^2
    sp = os.path.join(a.res, "alpha_sensitivity.csv")
    if os.path.exists(sp):
        s = pd.read_csv(sp)
        if "filter" in s:                      # 1D filter only
            s = s[s["filter"] == "1d"]
        s = s.groupby(["sigma", "alpha2"]).psnr.mean().reset_index()
        fig, ax = plt.subplots(figsize=(3.4, 2.5))
        for k, (sig, d) in enumerate(s.groupby("sigma")):
            ax.plot(d.alpha2, d.psnr, color=str(0.15 * k), marker="os^Dv"[k % 5], ms=3.5, label=f"σ = {sig:g}")
        ax.set_xscale("log")
        ax.set_xlabel("α²")
        ax.set_ylabel("PSNR, dB (validation)")
        ax.legend(fontsize=6.5)
        save(fig, out, "fig5_alpha_sensitivity")

    # summary table
    rows = []
    eff = pd.read_csv(eff_path) if os.path.exists(eff_path) else pd.DataFrame()
    for m in methods:
        d = q[q.method == m]
        r = {"Method": LABELS[m], "PSNR, dB": f"{d.psnr.mean():.2f}", "SSIM": f"{d.ssim.mean():.3f}"}
        if metric_ocr:
            r["CER"] = f"{d.cer.mean():.3f}"
        for dev in ("cpu", "cuda"):
            e = eff[(eff.method == m) & (eff.device == dev)] if len(eff) else eff
            if len(e):
                r[f"Time {dev.upper()}, ms"] = f"{e.ms_per_img_mean.iloc[0]:.1f}"
                r[f"Energy {dev.upper()}, J"] = f"{e.gross_J_per_img.iloc[0]:.3f}"
        if len(eff) and (eff.method == m).any():
            r["Params, M"] = f"{eff[eff.method == m].n_params.iloc[0] / 1e6:.1f}"
        rows.append(r)
    t = pd.DataFrame(rows).fillna("–")
    t.to_csv(os.path.join(a.res, "table_main.csv"), index=False)
    try:
        open(os.path.join(a.res, "table_main.md"), "w", encoding="utf-8").write(t.to_markdown(index=False))
    except ImportError:
        pass
    print(t.to_string(index=False))
    print("figures:", out)


if __name__ == "__main__":
    main()