"""10M method-benchmark driver: ONE FRESH PROCESS PER SEED (avoids 16 GB OOM),
then aggregate every mb10m_s*.json into docs/results/method_bench_10m_n10.json
with per-method x metric x slice means + 95% t-CIs + sign counts, paired
contrasts (IPW-reference, C3-IPW - C1-IPW), gap-closed, economics, and C2 price.

Usage:  python method_bench_driver.py            # runs any missing seeds, then aggregates
        python method_bench_driver.py --agg-only # aggregate whatever JSONs exist
"""
import sys, json, subprocess, time
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from scipy import stats

PY = sys.executable
WORKER = str(Path(__file__).parent / "method_bench_worker.py")
SEEDS = [90213, 90214, 90215, 90216, 90217, 90218, 90219, 90220, 90221, 90222]


def ci(vals):
    v = np.array([x for x in vals if x is not None], float)
    if len(v) < 2:
        return None
    m, se = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
    h = stats.t.ppf(0.975, len(v) - 1) * se
    return {"mean": round(float(m), 4), "ci95": [round(float(m - h), 4), round(float(m + h), 4)],
            "n_pos": int((v > 0).sum()), "n": len(v)}


def aggregate():
    rows = []
    for seed in SEEDS:
        p = Path(f"docs/results/mb10m_s{seed % 100}.json")
        if p.exists():
            rows.append(json.load(open(p)))
    if not rows:
        print("no per-seed JSONs found"); return
    n = len(rows)

    def cget(view, method, met):
        out = []
        for r in rows:
            row = next((x for x in r[view]["rows"] if x["method"] == method), None)
            if row is not None:
                v = row.get(met)
                if v is None and met.startswith("econ_"):
                    v = (row.get("econ") or {}).get(met[5:])
                out.append(v)
        return out

    metrics = []
    for b in ("ev_ratio", "ev_spearman", "auc_click", "auc_install", "auc_payer"):
        for sl in ("all", "won", "lost"):
            metrics.append(f"{b}_{sl}")
    econ_mets = ["econ_profit", "econ_roas", "econ_n_won"]

    summary_C1 = {}
    for m in ("floor", "reference", "ipw", "oracle"):
        summary_C1[m] = {}
        for met in metrics + econ_mets:
            c = ci(cget("C1", m, met))
            if c:
                summary_C1[m][met] = c
    summary_C3 = {"ipw": {met: ci(cget("C3", "ipw", met))
                          for met in metrics if ci(cget("C3", "ipw", met))}}

    # paired contrasts
    def paired(view_a, meth_a, view_b, meth_b, mets):
        d = {}
        for met in mets:
            diffs = []
            for r in rows:
                a = next((x.get(met) for x in r[view_a]["rows"] if x["method"] == meth_a), None)
                b = next((x.get(met) for x in r[view_b]["rows"] if x["method"] == meth_b), None)
                if a is not None and b is not None:
                    diffs.append(a - b)
            c = ci(diffs)
            if c:
                d[met] = c
        return d

    contrasts = {
        "ipw_minus_reference_C1": paired("C1", "ipw", "C1", "reference", metrics),
        "ipw_C3_minus_C1": paired("C3", "ipw", "C1", "ipw", metrics),
        "reference_minus_floor_C1": paired("C1", "reference", "C1", "floor", metrics),
    }
    # economics contrasts (profit regret vs oracle-value)
    econ_regret = {}
    for m in ("floor", "reference", "ipw"):
        diffs = []
        for r in rows:
            mp = next((x.get("econ", {}).get("profit") for x in r["C1"]["rows"]
                       if x["method"] == m), None)
            op = next((x.get("econ", {}).get("profit") for x in r["C1"]["rows"]
                       if x["method"] == "oracle"), None)
            if mp is not None and op is not None:
                diffs.append(op - mp)                     # regret (oracle - method)
        econ_regret[m] = ci(diffs)

    # gap closed (floor->oracle) on the clean rungs
    gap = {}
    for met in ("auc_install_all", "auc_install_lost", "ev_spearman_all",
                "ev_spearman_lost", "auc_click_all", "econ_profit"):
        fl = ci(cget("C1", "floor", met)); orc = ci(cget("C1", "oracle", met))
        if not fl or not orc or abs(orc["mean"] - fl["mean"]) < 1e-9:
            continue
        gap[met] = {}
        for m in ("floor", "reference", "ipw"):
            vals = []
            for r in rows:
                f = next((x.get(met) if not met.startswith("econ_")
                          else x.get("econ", {}).get(met[5:])
                          for x in r["C1"]["rows"] if x["method"] == "floor"), None)
                o = next((x.get(met) if not met.startswith("econ_")
                          else x.get("econ", {}).get(met[5:])
                          for x in r["C1"]["rows"] if x["method"] == "oracle"), None)
                v = next((x.get(met) if not met.startswith("econ_")
                          else x.get("econ", {}).get(met[5:])
                          for x in r["C1"]["rows"] if x["method"] == m), None)
                if None not in (f, o, v) and abs(o - f) > 1e-9:
                    vals.append((v - f) / (o - f))
            gap[met][m] = ci(vals)

    c2 = {k: ci([r["C2_price"].get(k) for r in rows])
          for k in rows[0]["C2_price"]}
    weights = {"mean": ci([r["C1"]["ipw_weight_stats"]["mean"] for r in rows]),
               "p99": ci([r["C1"]["ipw_weight_stats"]["p99"] for r in rows]),
               "frac_capped": ci([r["C1"]["ipw_weight_stats"]["frac_capped"] for r in rows])}
    oracle_bidder = {"profit": ci([r["oracle_bidder"]["profit"] for r in rows]),
                     "n_won": ci([r["oracle_bidder"]["n_won"] for r in rows])}

    final = {"scale": "10M", "rho": 0.8, "n_seeds": n, "seeds_present": [r["seed"] for r in rows],
             "summary_C1": summary_C1, "summary_C3": summary_C3,
             "contrasts": contrasts, "econ_regret_vs_oracle_C1": econ_regret,
             "gap_closed_C1": gap, "C2_price": c2,
             "ipw_weights": weights, "oracle_bidder": oracle_bidder, "rows": rows}
    json.dump(final, open("docs/results/method_bench_10m_n10.json", "w"),
              indent=2, default=float)
    print(f"\n===== 10M method benchmark, n={n} =====")
    print("IPW - reference (C1):")
    for met, s in contrasts["ipw_minus_reference_C1"].items():
        if met in ("auc_install_all", "ev_spearman_all", "ev_ratio_all",
                   "auc_payer_all", "auc_payer_lost"):
            print(f"  {met:20} {s['mean']:+.4f} {s['ci95']} ({s['n_pos']}/{s['n']})")
    print("C2 AFT gain over naive (CRPS lost):", c2.get("aft_gain_pct_crps_lost"))
    print("econ regret vs oracle (C1):", {m: (v['mean'] if v else None)
                                          for m, v in econ_regret.items()})
    print("Saved docs/results/method_bench_10m_n10.json")


def main():
    if "--agg-only" not in sys.argv:
        for seed in SEEDS:
            if Path(f"docs/results/mb10m_s{seed % 100}.json").exists():
                print(f"=== seed {seed} already done, skip ===", flush=True); continue
            t = time.time()
            print(f"=== 10M method-bench seed {seed} starting ===", flush=True)
            rc = subprocess.run([PY, WORKER, str(seed)]).returncode
            print(f"=== seed {seed} rc={rc} ({(time.time()-t)/60:.0f} min) ===", flush=True)
    aggregate()


if __name__ == "__main__":
    main()
