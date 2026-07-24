"""Anchored-point robustness: rho*=0.8, 1M, n=5 seeds. Multi-seed mean +/- 95% CI
+ sign counts on EVERY metric axis (Ken's challenge: not just AUC):
ranking (AFT + CLF), price RMSE + CRPS (all/lost), calibration (ECE), and
economics from BOTH bidders (AFT win curve and CLF win curve - model-robustness
of overpay/surplus, the economics analogue of the shrink test).
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
RHO, SEEDS = 0.8, [90213, 90214, 90215, 90216, 90217]

rows = []
for seed in SEEDS:
    tag = f"v10_anchor_s{seed % 100}"
    t = time.time()
    sim.run("test", out_name=tag, seed=seed, rho=RHO, bn_edges=EDGES)
    out = run_profile(tag, do_shap=False, hist_clearing=True)
    json.dump(out, open(f"docs/results/{tag}.json", "w"), indent=2, default=float)
    C, k = out["conditions"], out.get("contrasts", {})
    g = lambda c, key: C[c].get(key)
    gb = lambda c, key: (C[c].get("clf_bidder") or {}).get(key)
    r = {"seed": seed,
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
         "profit_aft": {c: g(c, "profit") for c in ("C1", "C3")},
         "profit_clfb": {c: gb(c, "profit") for c in ("C1", "C3")},
         "mins": round((time.time() - t) / 60, 1)}
    rows.append(r)
    print(f">>> seed={seed}: auc_clf={r['auc_clf']} rmse_all={r['rmse_all_gain']:.1f}% "
          f"crps_all={r['crps_all_gain']:.1f}% overpay aft={r['overpay_red_aft']:.1f}% "
          f"clfb={r['overpay_red_clfb'] and round(r['overpay_red_clfb'],1)}% ({r['mins']}min)")
    json.dump(rows, open("docs/results/v10_anchor_5seed.json", "w"), indent=2, default=float)

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
        "overpay_red_clfb", "surplus_up_clfb"]
summary = {kk: ci([r[kk] for r in rows]) for kk in keys}
summary["ece"] = {"C1": ci([r["ece_C1"] for r in rows]), "C3": ci([r["ece_C3"] for r in rows])}
print("\n===== ANCHORED POINT (rho=0.8, 1M, n=5) mean [95% CI] (n_pos/n) =====")
for kk in keys:
    s = summary[kk]
    if s:
        print(f"  {kk:18} {s['mean']:+8.3f}  {s['ci95']}  ({s['n_pos']}/{s['n']})")
json.dump({"rows": rows, "summary": summary},
          open("docs/results/v10_anchor_5seed.json", "w"), indent=2, default=float)
print("Saved docs/results/v10_anchor_5seed.json")
