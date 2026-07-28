"""Aggregate the per-seed run JSONs into the reported results tables.

The per-seed runners (`v10_anchor_n10.py` at 1M, `v10_10m_worker.py` at 10M) each
write one JSON per seed. Nothing in the repo turned those into the tables the
paper prints, so this script does it.

It emits ONE result set, in three parts:

  1. Main table   - per-view means for C1-C4 and the oracle, with the relevant
                    data-layer contrast folded into the same row: mean, paired
                    95% CI, sign count, direction, and a verdict. Reading a level
                    and judging whether the effect survives happen in one place.
  2. Contrast grid- all five meaningful view pairs on the economic rows, where
                    they genuinely differ.
  3. Scale check  - the headline contrasts at 1M beside 10M, which is the
                    evidence for the paper's scale-stability claim.

10M is the reported scale. 1M appears only in part 3. The 100K `golden` profile
is the quickstart and test fixture and has no ten-seed set, so it supports no
interval and is excluded.

WHAT IS REPORTED, AND WHY THESE ROWS. Every economic row is the CLASSIFIER
bidding algorithm (`clf_bidder`), which is what the paper reports. The AFT price
model is still fitted by the pipeline but is not reported, so the metrics that
only it produces (price RMSE, price CRPS, its win AUC and its calibration error)
are deliberately absent: their sole producer is a model the paper does not
describe. `rmse_spend` and the MCE rows are omitted for a different reason, that
their paired intervals span zero at n=10 and would enter the table as findings
they cannot support.

Usage:
    python scripts/aggregate_results.py                       # print only
    python scripts/aggregate_results.py --out docs/results/v10_paper_tables.json
    python scripts/aggregate_results.py --out ... --docx      # + a Word copy
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
PAIRS = [("C2", "C1", "MMP alone"), ("C3", "C1", "SSP alone"),
         ("C4", "C1", "both layers"), ("C4", "C2", "SSP given MMP"),
         ("C4", "C3", "MMP given SSP")]

# label · how to read it · key · better direction · which layer can move it · claim?
#   how:  "top"   read the condition dict
#         "clf"   read the clf_bidder sub-dict
#         "clf_k" read clf_bidder and multiply by 1000 (per-won -> CPM)
#         "bias"  |ev_ratio - 1|, so the contrast reads as bias magnitude
ROWS = [
    ("Tier 1 value model", [
        ("ev_spearman", "top", "ev_spearman", "up", "MMP", True),
        ("ev_ratio (level, 1 = unbiased)", "top", "ev_ratio", "up", "MMP", False),
        ("ev_ratio bias |ratio-1|", "bias", "ev_ratio", "down", "MMP", True),
        ("auc_click", "top", "auc_click", "up", "MMP", True),
        ("auc_install", "top", "auc_install", "up", "MMP", True),
        ("auc_payer", "top", "auc_payer", "up", "MMP", True),
    ]),
    ("Tier 2 win model, classifier", [
        ("auc_win", "top", "auc_win_clf", "up", "SSP", True),
        ("logloss_win", "top", "logloss_win_clf", "down", "SSP", True),
    ]),
    ("Economics, classifier bidding algorithm", [
        ("profit, total ($)", "clf", "profit", "up", "MMP", True),
        ("overpay CPM ($)", "clf_k", "overpay_per_won", "down", "SSP", True),
        ("revenue ($)", "clf", "revenue", None, None, False),
        ("spend ($)", "clf", "spend", None, None, False),
        ("ROAS", "clf", "roas", None, None, False),
        ("n_won", "clf", "n_won", None, None, False),
    ]),
]
CONTRAST_OF = {"MMP": ("C2", "C1"), "SSP": ("C3", "C1")}
ECON_LABELS = ["profit, total ($)", "overpay CPM ($)"]


def read(cond, how, key):
    """Pull one metric out of one condition dict."""
    if how == "top":
        return cond.get(key)
    if how == "bias":
        v = cond.get(key)
        return None if v is None else abs(v - 1.0)
    sub = cond.get("clf_bidder") or {}
    v = sub.get(key)
    if v is None:
        return None
    return v * 1000.0 if how == "clf_k" else v


def ci(vals):
    """Mean, paired 95% CI, and how many seeds moved up."""
    v = np.array([x for x in vals if x is not None], float)
    if len(v) < 2:
        return None
    m = float(v.mean())
    se = float(v.std(ddof=1) / np.sqrt(len(v)))
    h = float(stats.t.ppf(0.975, len(v) - 1) * se)
    return {"mean": round(m, 5), "ci95": [round(m - h, 5), round(m + h, 5)],
            "n_pos": int((v > 0).sum()), "n": int(len(v))}


def verdict(s, better):
    """Direction and statistical support, from the interval alone."""
    if s is None or better is None:
        return "-", "no claim"
    improved = (s["mean"] > 0) if better == "up" else (s["mean"] < 0)
    supported = s["ci95"][0] > 0 or s["ci95"][1] < 0
    return ("yes" if improved else "no",
            "supported" if supported else "not supported")


def load(scale):
    files = sorted(glob.glob(os.path.join(ROOT, SETS[scale])))
    runs = [json.load(open(f, encoding="utf-8")) for f in files]
    seeds = []
    for r in runs:
        tag = str(r.get("profile", ""))
        suffix = tag.rsplit("_s", 1)[-1] if "_s" in tag else ""
        seeds.append(90200 + int(suffix) if suffix.isdigit() else tag)
    return runs, seeds, [os.path.basename(f) for f in files]


def contrast(runs, b, a, how, key):
    d = []
    for r in runs:
        x = read(r["conditions"].get(a, {}), how, key)
        y = read(r["conditions"].get(b, {}), how, key)
        if x is not None and y is not None:
            d.append(y - x)
    return ci(d)


def build(runs):
    """The main table: means per view plus the relevant contrast, per row."""
    out = []
    for block, rows in ROWS:
        entries = []
        for label, how, key, better, layer, claim in rows:
            means = {}
            for v in VIEWS + ["oracle"]:
                vals = [read(r["conditions"].get(v, {}), how, key) for r in runs]
                vals = [x for x in vals if x is not None]
                means[v] = round(float(np.mean(vals)), 5) if vals else None
            s = None
            if claim and layer:
                b, a = CONTRAST_OF[layer]
                s = contrast(runs, b, a, how, key)
            imp, verd = verdict(s, better if claim else None)
            entries.append({"metric": label, "means": means,
                            "contrast": f"{layer} ({CONTRAST_OF[layer][0]}-{CONTRAST_OF[layer][1]})" if (claim and layer) else "-",
                            "stat": s, "improved": imp, "verdict": verd})
        out.append({"block": block, "rows": entries})
    return out


def fmt_num(v):
    if v is None:
        return "-"
    a = abs(v)
    if a >= 1000:
        return f"{v:,.0f}"
    if a >= 10:
        return f"{v:.2f}"
    return f"{v:.4f}"


def fmt_stat(s):
    if s is None:
        return "-"
    m, lo, hi = s["mean"], s["ci95"][0], s["ci95"][1]
    f = (lambda x: f"{x:+,.0f}") if abs(m) >= 1000 else (lambda x: f"{x:+.4f}")
    return f"{f(m)} [{f(lo)}, {f(hi)}] {s['n_pos']}/{s['n']}"


def write_md(path, doc):
    m = doc["main"]
    L = [f"# T9Sim reported results, 10M auctions, n = {doc['n_10m']} seeds", "",
         f"Seeds {doc['seeds_10m'][0]} to {doc['seeds_10m'][-1]}. Generated by "
         "`scripts/aggregate_results.py` from the per-seed run JSONs in "
         "`docs/results/`. Do not edit by hand.", "",
         "Every economic row is the classifier bidding algorithm, which is what the "
         "paper reports. The AFT price model is still fitted by the pipeline but is "
         "not reported, so the metrics only it produces are absent.", "",
         "## Main table", "",
         "**Contrast** names the data layer that can move the metric, and the view "
         "pair measuring it. **Improved** is direction only. **Verdict** is whether "
         "the paired 95% interval excludes zero. Rows marked *no claim* are "
         "descriptive: they carry no interval and should not be read as findings.", "",
         "| Metric | C1 | C2 | C3 | C4 | Oracle | Contrast | Mean [95% CI], seeds up | Improved | Verdict |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for blk in m:
        L.append(f"| **{blk['block']}** | | | | | | | | | |")
        for r in blk["rows"]:
            cells = " | ".join(fmt_num(r["means"][v]) for v in VIEWS + ["oracle"])
            L.append(f"| {r['metric']} | {cells} | {r['contrast']} | "
                     f"{fmt_stat(r['stat'])} | {r['improved']} | {r['verdict']} |")
    L += ["",
          "The oracle column is populated only for the Tier 1 value heads, the one "
          "block for which the source defines a latent ceiling.", "",
          "## Economic contrasts, all five view pairs", "",
          "On prediction metrics the censoring design forces C4-C3 to equal C2-C1 and "
          "C4-C2 to equal C3-C1 exactly, so those pairs carry no new information "
          "there. On economics all five differ, because profit comes from the bid "
          "rule, which multiplies the two tiers together.", "",
          "| Contrast | " + " | ".join(ECON_LABELS) + " |",
          "|---|" + "---|" * len(ECON_LABELS)]
    for key, block in doc["econ"].items():
        L.append(f"| {key} | " + " | ".join(fmt_stat(block[lab]) for lab in ECON_LABELS) + " |")
    L += ["", "## Scale check, 1M beside 10M", "",
          f"The same contrasts at 1M (n = {doc['n_1m']} seeds). This is the evidence "
          "for scale stability; 1M is not otherwise reported.", "",
          "| Metric | Contrast | 1M | 10M |", "|---|---|---|---|"]
    for row in doc["scale"]:
        L.append(f"| {row['metric']} | {row['contrast']} | {fmt_stat(row['s_1m'])} | "
                 f"{fmt_stat(row['s_10m'])} |")
    L.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


def print_console(doc):
    print(f"\n===== Main table, 10M, n={doc['n_10m']} =====")
    hdr = f"{'metric':32}" + "".join(f"{v:>10}" for v in VIEWS + ["oracle"])
    print(hdr + "  " + f"{'contrast':16}{'mean [95% CI] seeds':38}{'verdict':14}")
    for blk in doc["main"]:
        print(f"\n-- {blk['block']} --")
        for r in blk["rows"]:
            cells = "".join(f"{fmt_num(r['means'][v]):>10}" for v in VIEWS + ["oracle"])
            print(f"{r['metric']:32}{cells}  {r['contrast']:16}"
                  f"{fmt_stat(r['stat']):38}{r['verdict']:14}")
    print(f"\n===== Economic contrasts, all five pairs =====")
    for key, block in doc["econ"].items():
        print(f"  {key:26}" + "".join(f"{lab}: {fmt_stat(block[lab]):34}" for lab in ECON_LABELS))
    print(f"\n===== Scale check, 1M (n={doc['n_1m']}) beside 10M =====")
    for row in doc["scale"]:
        print(f"  {row['metric']:26}{row['contrast']:16}"
              f"1M {fmt_stat(row['s_1m']):34}10M {fmt_stat(row['s_10m'])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None,
                    help="write the JSON here; a .md twin is written alongside")
    ap.add_argument("--docx", action="store_true",
                    help="also export the markdown to .docx (needs pypandoc)")
    args = ap.parse_args()

    runs10, seeds10, files10 = load("10m")
    runs1, seeds1, files1 = load("1m")
    if not runs10:
        sys.exit("no 10M per-seed files found")
    print(f"10M: {len(runs10)} per-seed files {files10}")
    print(f"1M:  {len(runs1)} per-seed files {files1}")

    main_tbl = build(runs10)

    econ = {}
    for b, a, name in PAIRS:
        blk = {}
        for label, how, key, better, layer, claim in ROWS[2][1]:
            if label in ECON_LABELS:
                blk[label] = contrast(runs10, b, a, how, key)
        econ[f"{b}-{a} ({name})"] = blk

    scale = []
    for block, rows in ROWS:
        for label, how, key, better, layer, claim in rows:
            if not (claim and layer):
                continue
            b, a = CONTRAST_OF[layer]
            scale.append({"metric": label,
                          "contrast": f"{layer} ({b}-{a})",
                          "s_1m": contrast(runs1, b, a, how, key) if runs1 else None,
                          "s_10m": contrast(runs10, b, a, how, key)})

    doc = {"scale_reported": "10m", "n_10m": len(runs10), "seeds_10m": seeds10,
           "n_1m": len(runs1), "seeds_1m": seeds1,
           "bidder": "classifier (clf_bidder), the bidding algorithm the paper reports",
           "excluded": ("AFT-only metrics (price RMSE/CRPS, AFT win AUC and its "
                        "calibration error) because the paper does not report that "
                        "model; rmse_spend and the MCE rows because their paired "
                        "intervals span zero at n=10; the 100K profile because it "
                        "has no ten-seed set"),
           "main": main_tbl, "econ": econ, "scale": scale}

    print_console(doc)

    if args.out:
        path = os.path.join(ROOT, args.out)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2)
        md_path = os.path.splitext(path)[0] + ".md"
        write_md(md_path, doc)
        rel = lambda p: os.path.relpath(p, ROOT).replace(os.sep, "/")
        print(f"\nwrote {rel(path)}\nwrote {rel(md_path)}")
        if args.docx:
            sys.path.insert(0, os.path.join(ROOT, "src"))
            from t9sim.notebooks.docx_export import export
            docx_path = os.path.splitext(path)[0] + ".docx"
            export(md_path, docx_path)
            print(f"wrote {rel(docx_path)}")
    elif args.docx:
        sys.exit("--docx needs --out")


if __name__ == "__main__":
    main()
