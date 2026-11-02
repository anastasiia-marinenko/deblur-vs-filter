"""Step 2: main experiment, quality and cost (time, energy) of every method.

Methods (names as in the result files):
  blurred        - no restoration (baseline)
  pinv_oracle    - 1D pseudo-inverse filter with the true beta (upper bound for the filter)
  pinv_blind     - 1D filter, beta from the cepstrum ("Pseudo-inverse 1D")
  pinv2d_blind   - 2D filter, L and angle from the 2D cepstrum ("Pseudo-inverse 2D")
  dgv2_mobilenet - DeblurGAN-v2, MobileNet-v2 backbone (~3.3 M params)
  dgv2_inception - DeblurGAN-v2, Inception-ResNet-v2 backbone (~61 M params)

alpha^2 is not tuned on test data: it is chosen per noise level on validation images,
and on test the noise level is estimated automatically (skimage estimate_sigma).

Output in --out:
  quality.csv           - one row per (test image, method): PSNR, SSIM, CER, WER, cer_sharp, estimated beta / L / angle
  efficiency.csv        - time (ms per image) and energy (J per image), CPU and GPU models
  alpha_selection.json  - chosen alpha^2 per noise level (1D and 2D)
  alpha_sensitivity.csv - validation PSNR for every alpha^2
  summary_*.csv         - mean metrics by method and condition
  stats_wilcoxon.csv    - first comparison per image (the final one, per text, is in 04_stats.py)
  restored/<method>/... - all restored images (too big for the zip)

  python scripts/02_run_experiment.py --data data --out results --energy-seconds 30
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import wilcoxon
from skimage.restoration import estimate_sigma

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from deblur.classical import ALPHA2_GRID, estimate_beta, pseudo_inverse  # noqa: E402
from deblur.classical2d import estimate_motion_2d, deconv2d  # noqa: E402
from deblur import metrics as M  # noqa: E402
from deblur.energy import energy_per_item, measure_idle_power, time_per_item  # noqa: E402


def load(path):
    return np.array(Image.open(path), dtype=np.float64)


def true_angle(blur_model):
    # 8 for "angle8", 0 otherwise (shake has no angle)
    return float(blur_model[len("angle"):] or 8) if blur_model.startswith("angle") else 0.0


def select_alpha(meta, data_dir):
    """For each noise level, the alpha^2 with the best mean validation PSNR.
    Separately for the 1D filter (true beta) and the 2D filter (true L and angle).
    Blur parameters are known here, so only the effect of alpha^2 is measured."""
    val = meta[meta.split == "val"]
    rows = []
    for _, r in val.iterrows():
        g = load(os.path.join(data_dir, "degraded", r.file))
        f = load(os.path.join(data_dir, "sharp", r.sharp_id + ".png"))
        # keep the noise estimate, used later to map test images to a noise level
        sh = float(estimate_sigma(g))
        for a2 in ALPHA2_GRID:
            rows.append(dict(filter="1d", sigma=r.sigma, blur_model=r.blur_model, alpha2=a2,
                             psnr=M.psnr(f, pseudo_inverse(g, r.beta_true, a2)), sigma_hat=sh))
            rows.append(dict(filter="2d", sigma=r.sigma, blur_model=r.blur_model, alpha2=a2,
                             psnr=M.psnr(f, deconv2d(g, r.length_px, true_angle(r.blur_model), a2)),
                             sigma_hat=sh))
    sens = pd.DataFrame(rows)
    lookups = {}
    for flt, d in sens.groupby("filter"):
        table = d.groupby(["sigma", "alpha2"]).psnr.mean().reset_index()
        best = table.loc[table.groupby("sigma").psnr.idxmax()][["sigma", "alpha2"]]
        lookups[flt] = dict(zip(best.sigma, best.alpha2))
    # our run: 1D {0: 1e-4, 2: 1e-3, 5: 1e-2, 10: 1e-2}, 2D {0: 1e-3, 2: 1e-3, 5: 1e-2, 10: 1e-2}
    # mean noise estimate for each true level, to map sigma_hat -> level
    sig_hat = sens.groupby("sigma").sigma_hat.mean().to_dict()
    return lookups, sig_hat, sens


def make_alpha_picker(lookup, sig_hat):
    # pick(g): estimate the noise (Donoho: median of the finest wavelet coefficients / 0.6745),
    # take the closest validation level and return its alpha^2. No ground truth is used.
    levels = sorted(lookup)
    centers = np.array([sig_hat[s] for s in levels])

    def pick(g):
        s = float(estimate_sigma(g))
        return lookup[levels[int(np.argmin(np.abs(centers - s)))]]
    return pick


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--dg-repo", default="DeblurGANv2")
    ap.add_argument("--w-mobilenet", default="weights/fpn_mobilenet.h5")
    ap.add_argument("--w-inception", default="weights/fpn_inception.h5")
    ap.add_argument("--lang", default="eng")
    ap.add_argument("--no-ocr", action="store_true")
    ap.add_argument("--eff-condition", default="sinc_L32_s2",
                    help="condition used for time and energy")
    ap.add_argument("--energy-seconds", type=float, default=30.0)
    ap.add_argument("--save-restored", action="store_true", default=True)
    ap.add_argument("--only", default="", help="run only these methods (comma separated) and add them to existing results")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    meta = pd.read_csv(os.path.join(a.data, "meta.csv"))
    gt = json.load(open(os.path.join(a.data, "gt_text.json"), encoding="utf-8"))

    # 1. alpha^2 on validation
    print("1) picking alpha^2 on validation images")
    lookups, sig_hat, sens = select_alpha(meta, a.data)
    sens.to_csv(os.path.join(a.out, "alpha_sensitivity.csv"), index=False)
    json.dump({"alpha2_by_sigma_1d": {str(k): v for k, v in lookups["1d"].items()},
               "alpha2_by_sigma_2d": {str(k): v for k, v in lookups["2d"].items()},
               "sigma_hat_by_sigma": {str(k): v for k, v in sig_hat.items()}},
              open(os.path.join(a.out, "alpha_selection.json"), "w"), indent=1)
    print("   alpha^2 by noise level, 1D:", lookups["1d"], " 2D:", lookups["2d"])
    pick_alpha = make_alpha_picker(lookups["1d"], sig_hat)
    pick_alpha2d = make_alpha_picker(lookups["2d"], sig_hat)

    # 2. methods: name -> function(blurred) -> restored
    methods = {
        "blurred": lambda g: g,
        "pinv_oracle": None,  # needs beta_true, see below
        "pinv_blind": lambda g: pseudo_inverse(g, estimate_beta(g), pick_alpha(g)),
        "pinv2d_blind": None,  # see below
    }
    # both DeblurGAN-v2 nets, on GPU if there is one
    nets = {}
    try:
        import torch
        from deblur.deblurgan import DeblurGANv2
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        for name, bb, w in (("dgv2_mobilenet", "fpn_mobilenet", a.w_mobilenet),
                            ("dgv2_inception", "fpn_inception", a.w_inception)):
            if w and os.path.exists(w):
                nets[name] = DeblurGANv2(a.dg_repo, w, bb, device=dev)
                methods[name] = nets[name]
                print(f"   {name}: {nets[name].n_params / 1e6:.2f} M params, {dev}")
            else:
                print(f"   {name}: no weights at {w}, skipped")
    except Exception as e:  # noqa: BLE001
        print("   DeblurGAN-v2 not available:", e)

    # --only reruns some methods (not used in our run)
    only = [m for m in a.only.split(",") if m]
    if only:
        methods = {k: v for k, v in methods.items() if k in only}
        nets = {k: v for k, v in nets.items() if k in only}
        print("   only:", list(methods))

    # 3. quality on all 1440 test images, every method
    print("2) restoring test images")
    test = meta[meta.split == "test"].reset_index(drop=True)
    rows = []
    sharp_ocr = {}
    t0 = time.time()
    for i, r in test.iterrows():
        g = load(os.path.join(a.data, "degraded", r.file))
        f = load(os.path.join(a.data, "sharp", r.sharp_id + ".png"))
        # OCR on the sharp image once per text: OCR itself should make almost no errors
        if not a.no_ocr and r.sharp_id not in sharp_ocr:
            sharp_ocr[r.sharp_id] = M.cer(gt[r.sharp_id], M.ocr(f, a.lang))
        for name, fn in methods.items():
            extra = {}
            if name == "pinv_oracle":
                # true beta (we blurred it ourselves)
                out = pseudo_inverse(g, r.beta_true, pick_alpha(g))
            elif name == "pinv_blind":
                # beta from the cepstrum, saved to check the error later
                b = estimate_beta(g)
                extra["beta_est"] = b
                out = pseudo_inverse(g, b, pick_alpha(g))
            elif name == "pinv2d_blind":
                L2, th2 = estimate_motion_2d(g)
                extra.update(len2d_est=L2, angle2d_est=th2)
                out = deconv2d(g, L2, th2, pick_alpha2d(g))
            else:
                out = fn(g)                                       # blurred or a network
            row = dict(file=r.file, sharp_id=r.sharp_id, condition=r.condition, blur_model=r.blur_model,
                       length_px=r.length_px, sigma=r.sigma, beta_true=r.beta_true, method=name,
                       psnr=M.psnr(f, out), ssim=M.ssim(f, out), **extra)
            if not a.no_ocr:
                hyp = M.ocr(out, a.lang)
                row.update(cer=M.cer(gt[r.sharp_id], hyp), wer=M.wer(gt[r.sharp_id], hyp),
                           cer_sharp=sharp_ocr[r.sharp_id])
            rows.append(row)
            if a.save_restored:
                d = os.path.join(a.out, "restored", name)
                os.makedirs(d, exist_ok=True)
                Image.fromarray(np.clip(np.round(out), 0, 255).astype(np.uint8)).save(os.path.join(d, r.file))
        if (i + 1) % 20 == 0:
            print(f"   {i + 1}/{len(test)} images, {time.time() - t0:.0f} s")
    q = pd.DataFrame(rows)
    qpath = os.path.join(a.out, "quality.csv")
    if only and os.path.exists(qpath):
        old = pd.read_csv(qpath)
        q = pd.concat([old[~old.method.isin(only)], q], ignore_index=True)
    q.to_csv(qpath, index=False)

    # 4. time and energy on the 30 images of one condition (sinc, L = 32, sigma = 2);
    # both the filter and the nets do the same work for any blur on a 512x256 image
    print("3) time and energy")
    sub = test[test.condition == a.eff_condition]
    items = [(load(os.path.join(a.data, "degraded", r.file)), r.beta_true) for _, r in sub.iterrows()]
    idle = measure_idle_power(min(20.0, a.energy_seconds))
    print(f"   idle power: {idle:.1f} W")
    eff = []

    def bench(name, fn, device, n_params):
        # time per image -> mean ms; energy over >= 30 s loop -> J per image
        # for the filters the time includes noise estimate, cepstrum and filtering
        ts = time_per_item(fn, items)
        en = energy_per_item(fn, items, a.energy_seconds, idle, project=name)
        eff.append(dict(method=name, device=device, n_params=n_params,
                        ms_per_img_mean=1000 * ts.mean(), ms_per_img_std=1000 * ts.std(),
                        idle_power_W=idle, **en))
        print(f"   {name:15s} {device:4s} {1000 * ts.mean():9.1f} ms/img  "
              f"{en['net_J_per_img']:.3f} J/img net  [cpu: {en['cpu_tracking']}]")

    def deconv2d_blind(g):
        L2, th2 = estimate_motion_2d(g)
        return deconv2d(g, L2, th2, pick_alpha2d(g))

    # filters on CPU only
    if "pinv_oracle" in methods:
        bench("pinv_oracle", lambda it: pseudo_inverse(it[0], it[1], pick_alpha(it[0])), "cpu", 0)
    if "pinv_blind" in methods:
        bench("pinv_blind", lambda it: pseudo_inverse(it[0], estimate_beta(it[0]), pick_alpha(it[0])), "cpu", 0)
    if "pinv2d_blind" in methods:
        bench("pinv2d_blind", lambda it: deconv2d_blind(it[0]), "cpu", 0)
    # networks on CPU (separate copy) and on GPU
    for name, net in nets.items():
        from deblur.deblurgan import DeblurGANv2
        bb = net.backbone
        w = a.w_mobilenet if bb == "fpn_mobilenet" else a.w_inception
        cpu_net = net if net.device == "cpu" else DeblurGANv2(a.dg_repo, w, bb, device="cpu")
        bench(name, lambda it, n=cpu_net: n(it[0]), "cpu", net.n_params)
        if net.device.startswith("cuda"):
            bench(name, lambda it, n=net: n(it[0]), "cuda", net.n_params)
    eff = pd.DataFrame(eff)
    epath = os.path.join(a.out, "efficiency.csv")
    if only and os.path.exists(epath):
        old = pd.read_csv(epath)
        eff = pd.concat([old[~old.method.isin(only)], eff], ignore_index=True)
    eff.to_csv(epath, index=False)

    # 5. summary tables
    print("4) summary tables")
    agg_cols = [c for c in ("psnr", "ssim", "cer", "wer") if c in q]
    overall = q.groupby("method")[agg_cols].agg(["mean", "std"]).round(4)
    overall.to_csv(os.path.join(a.out, "summary_overall.csv"))
    for key in ("blur_model", "sigma", "length_px", "condition"):
        q.groupby(["method", key])[agg_cols].mean().round(4).to_csv(
            os.path.join(a.out, f"summary_by_{key}.csv"))
    # first Wilcoxon test per image; images of one text are not independent,
    # so the real comparison is per text in 04_stats.py
    stats = []
    for bname in [m for m in ("pinv_blind", "pinv2d_blind") if m in q.method.unique()]:
        base = q[q.method == bname].set_index("file")
        for other in [m for m in q.method.unique() if m.startswith("dgv2")]:
            o = q[q.method == other].set_index("file")
            for bm in ["all"] + sorted(q.blur_model.unique()):
                idx = base.index if bm == "all" else base[base.blur_model == bm].index
                for col in agg_cols:
                    x, y = base.loc[idx, col], o.loc[idx, col]
                    try:
                        p = wilcoxon(x, y).pvalue
                    except ValueError:
                        p = float("nan")
                    stats.append(dict(compare=f"{bname} vs {other}", blur_model=bm, metric=col,
                                      mean_classical=x.mean(), mean_other=y.mean(), p_wilcoxon=p, n=len(idx)))
    pd.DataFrame(stats).to_csv(os.path.join(a.out, "stats_wilcoxon.csv"), index=False)
    if "beta_est" in q:
        b = q[q.method == "pinv_blind"]
        err = (b.beta_est - b.beta_true).abs()
        print(f"   beta error: median {err.median():.4f}, 95th percentile {err.quantile(.95):.4f}")
    print(overall)
    print(f"done, results in {a.out}")


if __name__ == "__main__":
    main()