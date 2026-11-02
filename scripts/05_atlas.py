"""Step 5: "atlas" of the experiment. For every blur condition: the input, the output of each
method and the parameters it used.

For the chosen test texts (test_000 and test_001 in the notebook) it goes through all 48 conditions
(4 motions x 3 lengths x 4 noise levels) and saves:
  * atlas/<condition>/<text>__<method>.png  - full size images (sharp, blurred, restored)
  * atlas_<text>.pdf                         - one page per condition with captions
                                               (blur parameters, estimated L and theta, alpha^2, CER, time)
  * atlas_params.csv                         - the same parameters as a table
  * fig_examples_en.jpg / fig_examples_uk.jpg - Fig. 2 of the paper, b/w, 300 dpi
alpha^2 is chosen the same way as in step 2 (validation texts, separately for 1D and 2D).
This is a rerun of the same methods on the same images (CER matches quality.csv),
only now the pictures and parameters are saved.

  python scripts/05_atlas.py --data data --out atlas_out --ids test_000,test_001
only redraw the figure from a finished atlas:  python scripts/05_atlas.py --fig-only --out atlas_out
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from deblur import metrics as M  # noqa: E402
from deblur.classical import estimate_beta, pseudo_inverse, beta_to_length  # noqa: E402
from deblur.classical2d import cepstrum2d, deconv2d, estimate_motion_2d  # noqa: E402

# reuse functions from 02_run_experiment.py (alpha^2 choice, loading) so the atlas does exactly
# the same thing; the file name starts with a digit, so it is loaded by path
_spec = importlib.util.spec_from_file_location("exp02", os.path.join(os.path.dirname(__file__), "02_run_experiment.py"))
exp02 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp02)

MOTION = {"sinc": "straight horizontal (sinc model)", "box": "straight horizontal (spatial averaging)",
          "angle8": "straight at 8°", "shake": "hand shake + gamma + noise + JPEG"}
TITLE = {"sharp": "Sharp (reference)", "blurred": "Blurred input", "pinv_oracle": "Pseudo-inverse, known β",
         "pinv_blind": "Pseudo-inverse 1D (auto)", "pinv2d_blind": "Pseudo-inverse 2D (auto)",
         "dgv2_mobilenet": "DeblurGAN-v2 MobileNet", "dgv2_inception": "DeblurGAN-v2 Inception"}


def u8(x):
    return np.clip(np.round(x), 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="atlas_out")
    ap.add_argument("--ids", default="test_000", help="test texts, comma separated")
    ap.add_argument("--dg-repo", default="DeblurGANv2")
    ap.add_argument("--w-mobilenet", default="weights/fpn_mobilenet.h5")
    ap.add_argument("--w-inception", default="weights/fpn_inception.h5")
    ap.add_argument("--fig-only", action="store_true", help="only redraw the examples figure from atlas_params.csv")
    a = ap.parse_args()
    if a.fig_only:
        make_examples_figure(a.out, a.ids.split(",")[0])
        return
    os.makedirs(a.out, exist_ok=True)
    meta = pd.read_csv(os.path.join(a.data, "meta.csv"))
    gt = json.load(open(os.path.join(a.data, "gt_text.json"), encoding="utf-8"))

    print("alpha^2 on validation texts")
    lookups, sig_hat, _ = exp02.select_alpha(meta, a.data)
    pick1 = exp02.make_alpha_picker(lookups["1d"], sig_hat)
    pick2 = exp02.make_alpha_picker(lookups["2d"], sig_hat)

    nets = {}
    try:
        import torch
        from deblur.deblurgan import DeblurGANv2
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        for name, bb, w in (("dgv2_mobilenet", "fpn_mobilenet", a.w_mobilenet),
                            ("dgv2_inception", "fpn_inception", a.w_inception)):
            if os.path.exists(w):
                nets[name] = DeblurGANv2(a.dg_repo, w, bb, device=dev)
    except Exception as e:  # noqa: BLE001
        print("DeblurGAN-v2 not available:", e)
    print("networks:", list(nets) or "none")

    rows = []
    order = ["sinc", "box", "angle8", "shake"]
    # all 48 conditions of each text, order sinc -> box -> angle8 -> shake
    for sid in [s for s in a.ids.split(",") if s]:
        f = exp02.load(os.path.join(a.data, "sharp", sid + ".png"))
        sub = meta[(meta.sharp_id == sid)].copy()
        sub["o"] = sub.blur_model.map({m: i for i, m in enumerate(order)})
        sub = sub.sort_values(["o", "length_px", "sigma"])
        pdf = PdfPages(os.path.join(a.out, f"atlas_{sid}.pdf"))
        for _, r in sub.iterrows():
            g = exp02.load(os.path.join(a.data, "degraded", r.file))
            cond = r.condition
            d = os.path.join(a.out, "atlas", cond)
            os.makedirs(d, exist_ok=True)
            outs, info = {"sharp": f, "blurred": g}, {}
            # run every method, keep alpha^2, estimated L and theta, and time of one call
            a1, a2 = pick1(g), pick2(g)
            t0 = time.perf_counter(); outs["pinv_oracle"] = pseudo_inverse(g, r.beta_true, a1)
            info["pinv_oracle"] = dict(alpha2=a1, L=beta_to_length(r.beta_true, g.shape[1]), sec=time.perf_counter() - t0)
            t0 = time.perf_counter(); b = estimate_beta(g); outs["pinv_blind"] = pseudo_inverse(g, b, a1)
            info["pinv_blind"] = dict(alpha2=a1, L=beta_to_length(b, g.shape[1]), sec=time.perf_counter() - t0)
            t0 = time.perf_counter(); L2, th2 = estimate_motion_2d(g); outs["pinv2d_blind"] = deconv2d(g, L2, th2, a2)
            info["pinv2d_blind"] = dict(alpha2=a2, L=L2, theta=th2, sec=time.perf_counter() - t0)
            for n, net in nets.items():
                t0 = time.perf_counter(); outs[n] = net(g); info[n] = dict(sec=time.perf_counter() - t0)
            # save PNGs and get CER (Tesseract) for each
            for m, img in outs.items():
                Image.fromarray(u8(img)).save(os.path.join(d, f"{sid}__{m}.png"))
                c = M.cer(gt[sid], M.ocr(img))
                info.setdefault(m, {})["cer"] = c
                rows.append(dict(text=sid, condition=cond, motion=r.blur_model, L_true=r.length_px,
                                 theta_true=exp02.true_angle(r.blur_model) if r.blur_model != "shake" else np.nan,
                                 sigma=r.sigma, method=m, cer=c, **{k: v for k, v in info[m].items() if k != "cer"}))
            # A4 page: 8 panels (7 images + 2D cepstrum of the blurred one)
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.suptitle(f"{sid}  |  motion: {MOTION[r.blur_model]}  |  L = {r.length_px:g} px  |  noise σ = {r.sigma:g}"
                         f"\nα² chosen on validation: 1D = {a1:g}, 2D = {a2:g}", fontsize=10, family="serif")
            panels = ["sharp", "blurred", "pinv_oracle", "pinv_blind", "pinv2d_blind", "dgv2_mobilenet",
                      "dgv2_inception", "cepstrum"]
            for k, m in enumerate(panels):
                ax = fig.add_subplot(4, 2, k + 1)
                if m == "cepstrum":
                    # central 81x81 of the cepstrum, circles at the found peak (+-L cos, -+L sin);
                    # sign flipped (-crop), so the negative peak looks bright here
                    cp = cepstrum2d(g)
                    h, w = cp.shape
                    R = 40
                    crop = cp[h // 2 - R:h // 2 + R + 1, w // 2 - R:w // 2 + R + 1]
                    lo, hi = np.percentile(-crop, [1, 99.7])
                    ax.imshow(-crop, cmap="gray", extent=(-R, R, R, -R), vmin=lo, vmax=hi)
                    dx, dy = L2 * np.cos(np.deg2rad(th2)), -L2 * np.sin(np.deg2rad(th2))
                    ax.plot([dx, -dx], [dy, -dy], "o", mfc="none", mec="k", ms=9)
                    ax.set_title(f"2D cepstrum of the input (dark = negative)\nfound L = {L2:.1f} px, θ = {th2:.1f}°",
                                 fontsize=8, family="serif")
                    ax.tick_params(labelsize=6)
                    continue
                if m not in outs:
                    ax.axis("off")
                    ax.set_title(f"{TITLE[m]}: not available", fontsize=8)
                    continue
                ax.imshow(np.clip(outs[m], 0, 255), cmap="gray", vmin=0, vmax=255)
                ax.set_xticks([]), ax.set_yticks([])
                s = f"{TITLE[m]}\nCER = {100 * info[m]['cer']:.1f} %"
                if "L" in info[m]:
                    s += f",  L = {info[m]['L']:.1f} px"
                if "theta" in info[m]:
                    s += f", θ = {info[m]['theta']:.1f}°"
                if "alpha2" in info[m]:
                    s += f", α² = {info[m]['alpha2']:g}"
                if "sec" in info[m] and m not in ("sharp", "blurred"):
                    s += f"\ntime {1000 * info[m]['sec']:.0f} ms"
                ax.set_title(s, fontsize=8, family="serif")
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            pdf.savefig(fig)
            plt.close(fig)
            print(f"  {sid} {cond}: " + ", ".join(f"{m} {100 * info[m]['cer']:.0f}%" for m in outs if m != "sharp"))
        pdf.close()
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "atlas_params.csv"), index=False)
    make_examples_figure(a.out, a.ids.split(",")[0])


# 3 conditions (horizontal L=20, 8 deg L=20, shake L=10; sigma=2) x 6 columns.
# Only the left half of each image is shown (bigger letters); CER is for the whole image.
def make_examples_figure(out, sid):
    """Examples figure. Also works on its own: python 05_atlas.py --fig-only --out atlas_out"""
    rows = pd.read_csv(os.path.join(out, "atlas_params.csv"))
    picks = [("sinc_L20_s2", {"en": "Horizontal,\nL = 20 px", "uk": "Горизонтальний,\nL = 20 пікс."}),
             ("angle8_L20_s2", {"en": "At 8°,\nL = 20 px", "uk": "Під кутом 8°,\nL = 20 пікс."}),
             ("shake_L10_s2", {"en": "Hand shake,\nL = 10 px", "uk": "Тремтіння рук,\nL = 10 пікс."})]
    cols = ["sharp", "blurred", "pinv_blind", "pinv2d_blind", "dgv2_mobilenet", "dgv2_inception"]
    head = {"en": ["Sharp", "Blurred", "Pseudo-inverse 1D", "Pseudo-inverse 2D", "DeblurGAN-v2\nMobileNet",
                   "DeblurGAN-v2\nInception"],
            "uk": ["Різке", "Розмите", "Псевдоінверсний 1D", "Псевдоінверсний 2D", "DeblurGAN-v2\nMobileNet",
                   "DeblurGAN-v2\nInception"]}
    p = rows
    for lang in ("en", "uk"):
        fig, axes = plt.subplots(len(picks), len(cols), figsize=(7.0, 0.82 * len(picks) + 0.45))
        for i, (cond, lab) in enumerate(picks):
            for j, m in enumerate(cols):
                ax = axes[i, j]
                fp = os.path.join(out, "atlas", cond, f"{sid}__{m}.png")
                if os.path.exists(fp):
                    im = np.array(Image.open(fp))
                    ax.imshow(im[:, : im.shape[1] // 2], cmap="gray", vmin=0, vmax=255)
                    c = p[(p.condition == cond) & (p.method == m) & (p.text == sid)].cer
                    if len(c) and m != "sharp":
                        ax.set_xlabel(f"CER {100 * c.iloc[0]:.0f} %".replace(".", ",") if lang == "uk"
                                      else f"CER {100 * c.iloc[0]:.0f} %", fontsize=6.5, labelpad=1)
                ax.set_xticks([]), ax.set_yticks([])
                if i == 0:
                    ax.set_title(head[lang][j], fontsize=7, family="serif")
                if j == 0:
                    ax.set_ylabel(lab[lang], fontsize=6.5, family="serif")
        fig.tight_layout(pad=0.3, h_pad=0.4, w_pad=0.3)
        fig.savefig(os.path.join(out, f"fig_examples_{lang}.jpg"), dpi=300, pil_kwargs={"quality": 95})
        fig.savefig(os.path.join(out, f"fig_examples_{lang}.png"), dpi=200)
        plt.close(fig)
    print("done:", out)


if __name__ == "__main__":
    main()