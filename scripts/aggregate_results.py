"""Aggregate per-seed run JSONs into the paper's two tables.

The per-seed runners (v10_anchor_n10.py, v10_10m_worker.py) each write one JSON
per seed. This script combines them into:

  * table_means  - the per-view means of Table 7.1 (plus the oracle column)
  * contrasts    - the paired data-layer contrasts of Table 7.3, for all five
                   meaningful view pairs, each with a paired 95% CI and a sign
                   count over seeds

The five pairs are C2-C1 (MMP alone), C3-C1 (SSP alone), C4-C1 (both layers),
C4-C2 (SSP given MMP) and C4-C3 (MMP given SSP). C3-C2 is omitted: it swaps one
layer for another rather than adding one.

On PREDICTION metrics the censoring design forces C4-C3 == C2-C1 and
C4-C2 == C3-C1 exactly, so those two pairs carry no new information there. They
are computed anyway, because the exact equality is a free integrity check on the
censoring operators. On ECONOMIC metrics all five differ, because profit comes
from the bid rule, which multiplies the two tiers together.

Every economic row is the CLASSIFIER bidding algorithm (`clf_bidder`), which is
what the paper reports; the AFT price model's keys are deliberately not used.

Only the 1M and 10M scales are covered. The 100K `golden` profile is the
quickstart and test fixture and has no ten-seed set, so it supports no interval
and no sign count.

Usage:
    python scripts/aggregate_results.py 10m      # docs/results/v10_10m_s*.json
    python scripts/aggregate_results.py 1m       # docs/results/v10_anchor_s*.json
    python scripts/aggregate_results.py 10m --out docs/results/v10_10m_paper_tables.json

The ev_ratio contrast is scored as BIAS MAGNITUDE |ratio - 1|, so a negative
contrast means the bias fell (the improving direction), matching Table 7.3.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETS = {"10m": "docs/results/v10_10m_s*.json",
        "1m": "docs/results/v10_anchor_s*.json"}
VIEWS = ["C1", "C2", "C3", "C4"]
PAIRS = [("C2", "C1", "MMP"), ("C3", "C1", "SSP"),
         ("C4", "C1", "MMP+SSP"), ("C4", "C2", "SSP given MMP"),
         ("C4", "C3", "MMP given SSP")]

# (label, getter) - "clf" reads the clf_bidder sub-dict, "top" the condition dict
ROWS = [
    ("ev_spearman", "top", "ev_spearman"),
    ("ev_ratio", "top", "ev_ratio"),
    ("auc_click", "top", "auc_click"),
    ("auc_install", "top", "auc_install"),
    ("auc_payer", "top", "auc_payer"),
    ("auc_win", "top", "auc_win_clf"),
    ("logloss_win", "top", "logloss_win_clf"),
    ("profit_total ($)", "clf", "profit"),
    ("overpay_cpm ($)", "clf_x1000", "overpay_per_won"),
    ("ROAS", "clf", "roas"),
    ("n_won", "clf", "n_won"),
]


def read(kind, how, key):
    """Pull one metric out of one condition dict."""
    if how == "top":
        return kind.get(key)
    sub = kind.get("clf_bidder") or {}
    v = sub.get(key)
    if v is None:
        return None
    return v * 1000.0 if how == "clf_x1000" else v


def ci(vals):
    """Mean and paired 95% CI over seeds, with the count above zero."""
    v = np.array([x for x in vals if x is not None], float)
    if len(v) < 2:
        return None
    m = float(v.mean())
    se = float(v.std(ddof=1) / np.sqrt(len(v)))
    h = float(stats.t.ppf(0.975, len(v) - 1) * se)
    return {"mean": round(m, 5), "ci95": [round(m - h, 5), round(m + h, 5)],
            "n_pos": int((v > 0).sum()), "n": int(len(v))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scale", choices=sorted(SETS))
    ap.add_argument("--out", default=None, help="write the aggregate JSON here")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(ROOT, SETS[args.scale])))
    if not files:
        sys.exit(f"no per-seed files matching {SETS[args.scale]}")
    runs = [json.load(open(f, encoding="utf-8")) for f in files]
    # the per-seed JSONs carry the run tag (e.g. "v10_10m_s13"), not the seed
    seeds = []
    for r in runs:
        tag = str(r.get("profile", ""))
        suffix = tag.rsplit("_s", 1)[-1] if "_s" in tag else ""
        seeds.append(90200 + int(suffix) if suffix.isdigit() else tag)
    print(f"{len(runs)} per-seed files: {[os.path.basename(f) for f in files]}")

    # ---- Table 7.1: per-view means (+ oracle where the source defines it) ----
    table_means = {}
    for label, how, key in ROWS:
        cells = {}
        for v in VIEWS + ["oracle"]:
            vals = [read(r["conditions"].get(v, {}), how, key) for r in runs]
            vals = [x for x in vals if x is not None]
            cells[v] = round(float(np.mean(vals)), 5) if vals else None
        table_means[label] = cells

    # ---- Table 7.3: paired contrasts for every view pair ----
    contrasts = {}
    for b, a, name in PAIRS:
        block = {}
        for label, how, key in ROWS:
            d = []
            for r in runs:
                x = read(r["conditions"].get(a, {}), how, key)
                y = read(r["conditions"].get(b, {}), how, key)
                if x is not None and y is not None:
                    d.append(y - x)
            block[label] = ci(d)
        # ev_ratio scored as bias magnitude |ratio - 1|: negative = bias fell
        d = []
        for r in runs:
            x = read(r["conditions"].get(a, {}), "top", "ev_ratio")
            y = read(r["conditions"].get(b, {}), "top", "ev_ratio")
            if x is not None and y is not None:
                d.append(abs(y - 1) - abs(x - 1))
        block["ev_ratio (bias magnitude)"] = ci(d)
        # profit as a percentage change, per seed
        d = []
        for r in runs:
            x = read(r["conditions"].get(a, {}), "clf", "profit")
            y = read(r["conditions"].get(b, {}), "clf", "profit")
            if x:
                d.append(100.0 * (y / x - 1.0))
        block["profit (%)"] = ci(d)
        contrasts[f"{b}-{a} ({name})"] = block

    out = {"scale": args.scale, "n": len(runs), "seeds": seeds,
           "bidder": "classifier (clf_bidder) - the paper's reported bidding algorithm",
           "table_means": table_means, "contrasts": contrasts}

    print(f"\n===== Table 7.1: per-view means (n={len(runs)}) =====")
    print(f"{'metric':22}" + "".join(f"{v:>12}" for v in VIEWS + ["oracle"]))
    for label, _, _ in ROWS:
        c = table_means[label]
        print(f"{label:22}" + "".join(
            (f"{c[v]:>12.4f}" if isinstance(c[v], float) else f"{'-':>12}")
            for v in VIEWS + ["oracle"]))

    print("\n===== Table 7.3: paired contrasts, mean [95% CI] (seeds with a "
          "positive difference / n) =====")
    print("NB a positive difference is not always the improving direction: for "
          "the bias-magnitude\nand log-loss rows the improving direction is "
          "negative, so 0/10 there means all ten improved.")
    for pair, block in contrasts.items():
        print(f"\n{pair}")
        for label in ["ev_ratio (bias magnitude)", "auc_payer", "auc_win",
                      "profit_total ($)", "profit (%)", "overpay_cpm ($)"]:
            s = block.get(label)
            if s:
                print(f"  {label:26} {s['mean']:+10.4f}  "
                      f"[{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}]  "
                      f"({s['n_pos']}/{s['n']})")

    if args.out:
        path = os.path.join(ROOT, args.out)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
