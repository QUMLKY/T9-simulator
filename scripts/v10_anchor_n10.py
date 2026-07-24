"""Extend the anchored point (rho*=0.8, 1M) to n=10 seeds: runs seeds 90218-90222
and combines with the 5-seed batch (v10_anchor_5seed.json) into an n=10 summary
(mean +/- 95% t-CI + sign counts on every metric axis).
"""
import sys, json, time
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from scipy import stats
import t9sim.simulate as sim
from t9sim.pipeline import run_profile

V7_BN = {e: True for e in ("pairing", "os_spend", "payer_timing", "hour", "exposure")}
EDGES = {**V7_BN, "rival_pool": True}
RHO, SEEDS = 0.8, [90218, 90219, 90220, 90221, 90222]

def extract(out, seed, mins):
    C, k = out["conditions"], out.get("contrasts", {})
    g = lambda c, key: C[c].get(key)
    gb = lambda c, key: (C[c].get("clf_bidder") or {}).get(key)
    return {"seed": seed,
            "auc_aft": (k.get("ssp_C3_C1_aft") or {}).get("diff"),
            "auc_clf": (k.get("ssp_C3_C1_clf") or {}).get("diff"),
            "rmse_all_gain": 100 * (1 - g("C3", "price_rmse_all") / g("C1", "price_rmse_all")),
            "rmse_lost_gain": 100 * (1 - g("C3", "price_rmse_lost") / g("C1", "price_rmse_lost")),
            "crps_all_gain": 100 * (1 - g("C3", "price_crps_all") / g("C1", "price_crps_all")),
            "crps_lost_gain": 100 * (1 - g("C3", "price_crps_lost") / g("C1", "price_crps_lost")),
            "ece_C1": g("C1", "ece_win"), "ece_C3": g("C3", "ece_win"),
            "overpay_red_aft": 100 * (1 - g("C3", "overpay_per_won") / g("C1", "overpay_per_won")),
            "surplus_up_aft": 100 * (g("C3", "surplus_per_won") / g("C1", "surplus_per_won") - 1),
            "overpay_red_clfb": (100 * (1 - gb("C3", "overpay_per_won") / gb("C1", "overpay_per_won"))
                                 if gb("C1", "overpay_per_won") else None),
            "surplus_up_clfb": (100 * (gb("C3", "surplus_per_won") / gb("C1", "surplus_per_won") - 1)
                                if gb("C1", "surplus_per_won") else None),
            "profit_up_clfb": (100 * (gb("C3", "profit") / gb("C1", "profit") - 1)
                               if gb("C1", "profit") else None),
            "mins": mins}

rows = []
for seed in SEEDS:
    tag = f"v10_anchor_s{seed % 100}"
    t = time.time()
    sim.run("test", out_name=tag, seed=seed, rho=RHO, bn_edges=EDGES)
    out = run_profile(tag, do_shap=False, hist_clearing=True)
    json.dump(out, open(f"docs/results/{tag}.json", "w"), indent=2, default=float)
    r = extract(out, seed, round((time.time() - t) / 60, 1))
    rows.append(r)
    print(f">>> seed={seed}: auc_clf={r['auc_clf']} rmse_all={r['rmse_all_gain']:.1f}% "
          f"overpay clfb={r['overpay_red_clfb'] and round(r['overpay_red_clfb'],1)}% ({r['mins']}min)")

prev = json.load(open("docs/results/v10_anchor_5seed.json"))
prev_rows = prev["rows"] if isinstance(prev, dict) else prev
all_rows = prev_rows + rows

def ci(vals):
    v = np.array([x for x in vals if x is not None], float)
    if len(v) < 2:
        return None
    m, se = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
    h = stats.t.ppf(0.975, len(v) - 1) * se
    return {"mean": round(float(m), 4), "ci95": [round(float(m - h), 4), round(float(m + h), 4)],
            "n_pos": int((v > 0).sum()), "n": len(v)}

keys = ["auc_aft", "auc_clf", "rmse_all_gain", "rmse_lost_gain", "crps_all_gain",
        "crps_lost_gain", "overpay_red_aft", "surplus_up_aft",
        "overpay_red_clfb", "surplus_up_clfb", "profit_up_clfb"]
summary = {kk: ci([r.get(kk) for r in all_rows]) for kk in keys}
summary["ece_C1"] = ci([r.get("ece_C1") for r in all_rows])
summary["ece_C3"] = ci([r.get("ece_C3") for r in all_rows])
print("\n===== ANCHORED POINT rho*=0.8, 1M, n=10: mean [95% CI] (n_pos/n) =====")
for kk in keys:
    s = summary[kk]
    if s:
        print(f"  {kk:18} {s['mean']:+8.3f}  {s['ci95']}  ({s['n_pos']}/{s['n']})")
json.dump({"rows": all_rows, "summary": summary},
          open("docs/results/v10_anchor_n10.json", "w"), indent=2, default=float)
print("Saved docs/results/v10_anchor_n10.json")
