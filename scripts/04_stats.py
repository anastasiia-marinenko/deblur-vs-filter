"""Step 4: statistics, means with 95 % CIs and paired comparisons of the methods.

Each of the 30 test texts is blurred under 48 conditions, so the 1440 images are not
1440 independent samples (a text that is hard for OCR is hard under every condition).
So the unit is the source text (sharp_id), n = 30:
  * CER of each method is averaged over all conditions of one text (per blur model and overall)
  * CI by bootstrap over texts (5000 resamples)
  * methods compared with the paired Wilcoxon test over texts

Output in results/:
  stats_ci.csv            mean and 95 % CI per method, metric and blur model
  stats_paired.csv        differences classical - network, 95 % CI, p (Wilcoxon),
                          share of texts where the classical one is better (share_a_better)
  stats_2d_estimation.csv median errors of L and angle for the 2D filter

  python scripts/04_stats.py --synthetic results
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# bootstrap resamples and a fixed seed
B = 5000
RNG = np.random.default_rng(2026)
CLASSICAL = ("pinv_blind", "pinv2d_blind", "pinv1d_blind", "pinv_oracle")


# bootstrap CI of the mean: resample with replacement, take the 2.5 and 97.5 percentiles
# of the means. Returns (mean, low, high).
def boot_ci(x, b=B):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    idx = RNG.integers(0, len(x), (b, len(x)))
    m = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


# paired comparison of methods a and b over units (texts)
# piv: row = text, column = method, value = mean metric of that text
# share_a_better: share of texts where a is better (lower for CER/WER)
def paired(piv, a, b, metric):
    d = piv[[a, b]].dropna()
    if len(d) < 2:
        return None
    diff = d[a] - d[b]
    mean, lo, hi = boot_ci(diff)
    try:
        p = float(wilcoxon(d[a], d[b]).pvalue)
    except ValueError:
        p = float("nan")
    better = (d[a] < d[b]).mean() if metric in ("cer", "wer") else (d[a] > d[b]).mean()
    return dict(compare=f"{a} - {b}", metric=metric, n_units=len(d), mean_a=float(d[a].mean()),
                mean_b=float(d[b].mean()), diff=mean, diff_lo=lo, diff_hi=hi, p_wilcoxon=p,
                share_a_better=float(better))


# unit = sharp_id (text), group = blur_model; "all" = all models together
def analyse(q, unit, group, out_prefix, metrics, exclude_from_all=()):
    ci_rows, pr_rows = [], []
    groups = ["all"] + sorted(q[group].dropna().unique()) if group else ["all"]
    for g in groups:
        if g == "all":
            sub = q[~q[group].isin(exclude_from_all)] if group else q
        else:
            sub = q[q[group] == g]
        for metric in metrics:
            if metric not in sub:
                continue
            # one number per text: mean over all conditions of the group
            # (e.g. angle8: 3 lengths x 4 noise levels = 12 images)
            per_unit = sub.groupby([unit, "method"])[metric].mean().unstack("method")
            for m in per_unit.columns:
                mean, lo, hi = boot_ci(per_unit[m])
                ci_rows.append(dict(group=g, metric=metric, method=m, n_units=int(per_unit[m].notna().sum()),
                                    mean=mean, ci_lo=lo, ci_hi=hi))
            nets = [m for m in per_unit.columns if m.startswith("dgv2")]
            cls = [m for m in per_unit.columns if m in CLASSICAL]
            # every classical filter vs every network, plus 2D vs 1D
            pairs = [(c, n) for c in cls for n in nets]
            if "pinv_blind" in cls and "pinv2d_blind" in cls:
                pairs.append(("pinv2d_blind", "pinv_blind"))
            if "pinv1d_blind" in cls and "pinv2d_blind" in cls:
                pairs.append(("pinv2d_blind", "pinv1d_blind"))
            for a, b in pairs:
                r = paired(per_unit, a, b, metric)
                if r:
                    pr_rows.append(dict(group=g, **r))
    ci, pr = pd.DataFrame(ci_rows), pd.DataFrame(pr_rows)
    ci.to_csv(out_prefix + "stats_ci.csv", index=False)
    pr.to_csv(out_prefix + "stats_paired.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(ci[ci.metric == metrics[0]].round(4).to_string(index=False))
        print(pr[pr.metric == metrics[0]].round(4).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", default="results")
    ap.add_argument("--real", nargs="*", default=[])
    a = ap.parse_args()

    # synthetic experiment
    qp = os.path.join(a.synthetic, "quality.csv")
    if os.path.exists(qp):
        q = pd.read_csv(qp)
        print("=== synthetic, unit = text, n =", q.sharp_id.nunique())
        analyse(q, "sharp_id", "blur_model", os.path.join(a.synthetic, ""), ["cer", "wer", "psnr", "ssim"])
        if "beta_est" in q:
            b = q[q.method == "pinv_blind"]
            # 1D length error in px: |beta_est - beta_true| * n / pi
            L_err = (b.beta_est - b.beta_true).abs() * 512 / np.pi
            print(f"L error (1D, px): median {L_err.median():.3f}, 95th percentile {L_err.quantile(.95):.3f}")
        if "len2d_est" in q:
            b = q[q.method == "pinv2d_blind"]
            # 2D errors: length in px, angle in degrees (period 180)
            # shake has no true angle (0 is used), so its angle error means nothing
            Lerr = (b.len2d_est - b.length_px).abs()
            ang_true = np.where(b.blur_model == "angle8", 8.0, 0.0)
            aerr = np.abs(((b.angle2d_est - ang_true) + 90) % 180 - 90)
            pd.DataFrame(dict(blur_model=b.blur_model, sigma=b.sigma, L_err=Lerr, angle_err=aerr)).groupby(
                "blur_model").agg(["median", lambda s: s.quantile(.95)]).round(3).to_csv(
                os.path.join(a.synthetic, "stats_2d_estimation.csv"))
            print(f"2D error: L median {Lerr.median():.3f} px, angle median {np.median(aerr):.2f}°")
    # real photos (optional, not used in the paper)
    for r in a.real:
        rp = os.path.join(r, "real_quality.csv")
        if not os.path.exists(rp):
            print("not found:", rp)
            continue
        q = pd.read_csv(rp)
        unit = "page" if "page" in q and q.page.notna().any() else "image"
        metric = "cer" if "cer" in q and q.cer.notna().any() else "ocr_conf"
        print(f"=== real photos {r}, unit = {unit}, n = {q[unit].nunique()}")
        analyse(q, unit, "category" if "category" in q else None,
                os.path.join(r, "real_"), [metric] + (["wer"] if metric == "cer" else []),
                exclude_from_all=("sharp",))   # "all" = blurred photos only


if __name__ == "__main__":
    main()