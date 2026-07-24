"""One 10M method-benchmark seed in a FRESH process (scale10m convention —
avoids cross-seed memory accumulation on 16 GB). Deletes the parquet after.

Per view, methods on the SAME censored data, all scored vs retained truth
(all / won / LOST slices) + a money-vs-truth column (profit & regret vs the
oracle bidder, win model held FIXED across methods so only the value estimate
varies):

  C1 funnel:  floor(LR) | reference(XGB) | IPW(XGB, CROSS-FIT propensities) | oracle
  C3 funnel:  IPW (weights from C3's SSP-informed win model)
  C2 price:   naive(won-rows fit) vs AFT(interval-censored)

Cross-fit: the propensity/win model is fit on the VALIDATION split and applied
to TRAIN rows (sample-split → no in-sample propensity overfit). The same model
is the shared bidder win curve for the C1 economics column.

Usage:  python method_bench_worker.py <seed> [--golden]
"""
import sys, json, shutil, gc
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb

import t9sim.simulate as sim
from t9sim import censor, schema
from t9sim.paths import OUTPUT_DIR
from t9sim.pipeline import (PipelineConfig, temporal_split, funnel_population,
                            train_tier1, predict_ev, train_tier2_aft,
                            train_tier2_clf, attach_hist_clearing, t2_feats,
                            pwin_clf, _resolve_bidder, oracle_bidder_ev,
                            _as_cat, BID_GRID)
from method_bench import (train_floor, floor_ev, train_tier1_ipw, eval_funnel,
                          price_metrics, oracle_funnel_row)

V7_BN = {e: True for e in ("pairing", "os_spend", "payer_timing", "hour", "exposure")}
EDGES = {**V7_BN, "rival_pool": True}
RHO = 0.8
IPW_PMIN = 0.05                              # weight cap = 20 (disclosed)


def xfit_weights(clf_bst, clf_feats, tr):
    """Cross-fit weight = 1/clip(p_win_at_logged_bid, IPW_PMIN, 1); p_win from a
    win model fit on an INDEPENDENT split (val), applied to train rows."""
    p = clf_bst.predict(xgb.DMatrix(_as_cat(tr, clf_feats), enable_categorical=True))
    return 1.0 / np.clip(p, IPW_PMIN, 1.0)


def econ(ev_hat, pw_grid, test):
    r = _resolve_bidder(ev_hat, pw_grid, test)
    return {"profit": r["profit"], "roas": r["roas"], "n_won": r["n_won"],
            "overpay_per_won": r["overpay_per_won"]}


def run(seed, profile):
    tag = f"mb10m_s{seed % 100}" if profile == "scale10m" else f"mbg_s{seed % 100}"
    sim.run(profile, out_name=tag, seed=seed, rho=RHO, bn_edges=EDGES)

    pc = PipelineConfig()
    path = OUTPUT_DIR / tag / "auctions.parquet"
    pf_cols = set(pq.ParquetFile(path).schema_arrow.names)
    needed = [c for c in (set(schema.NEEDED) |
              {"p_click", "p_install", "p_payer", "e_ltv"}) if c in pf_cols]
    df = pd.read_parquet(path, columns=needed)
    for c in pc.t1_cat:
        if c in df.columns:
            df[c] = df[c].astype("category")
    sp = temporal_split(df).values
    base_tr, base_va, base_te = df[sp == "train"], df[sp == "val"], df[sp == "test"]
    out = {"seed": seed, "n_rows": int(len(df)), "profile": profile}
    del df; gc.collect()

    # ---- oracle bidder (shared ceiling for the economics regret column) ----
    oracle_b = oracle_bidder_ev(pc, base_te)
    out["oracle_bidder"] = {"profit": oracle_b["profit"], "roas": oracle_b["roas"],
                            "n_won": oracle_b["n_won"]}

    # ---- C1 / C3 funnel methods ----
    for cond in ("C1", "C3"):
        tr = censor.view(base_tr, cond)
        va = censor.view(base_va, cond)
        te = base_te.copy()
        attach_hist_clearing(tr, va, te, censor.CONDITIONS[cond]["clearing"])
        feats2 = t2_feats(cond, True, view_cols=tr.columns)
        # cross-fit propensity/win model: FIT ON VAL, apply to train (+ shared bidder curve)
        clf_bst, clf_feats = train_tier2_clf(va, va, cond, feats=feats2)
        tr = tr.assign(_ipw_w=xfit_weights(clf_bst, clf_feats, tr))
        w = tr["_ipw_w"].to_numpy()
        pw_shared = pwin_clf(clf_bst, clf_feats, te, BID_GRID)   # fixed win curve

        rows = []
        if cond == "C1":
            fl_models, fl_cols = train_floor(tr, va, cond)
            r = eval_funnel("floor", floor_ev(fl_models, fl_cols, te), te)
            r["econ"] = econ(floor_ev(fl_models, fl_cols, te)[0], pw_shared, te)
            rows.append(r)
            ref = train_tier1(pc, tr, va, cond)
            r = eval_funnel("reference", predict_ev(pc, ref, te), te)
            r["econ"] = econ(predict_ev(pc, ref, te)[0], pw_shared, te)
            rows.append(r); del ref
        ipw = train_tier1_ipw(pc, tr, va, cond)
        r = eval_funnel("ipw", predict_ev(pc, ipw, te), te)
        r["econ"] = econ(predict_ev(pc, ipw, te)[0], pw_shared, te)
        rows.append(r); del ipw
        if cond == "C1":
            orow = oracle_funnel_row(te)
            # value ceiling under the SAME shared win curve → method→oracle regret
            # isolates value estimation (win model identical); out['oracle_bidder']
            # keeps the full true-EV+true-win absolute ceiling separately.
            orow["econ"] = econ(te["ev_truth"].to_numpy(), pw_shared, te)
            rows.append(orow)
        out[cond] = {"rows": rows,
                     "ipw_weight_stats": {"mean": round(float(w.mean()), 3),
                                          "p99": round(float(np.quantile(w, 0.99)), 3),
                                          "max": round(float(w.max()), 3),
                                          "frac_capped": round(float((w >= 1/IPW_PMIN - 1e-9).mean()), 4)}}
        del tr, va, te, clf_bst, pw_shared; gc.collect()

    # ---- C2 price: naive vs AFT ----
    tr = censor.view(base_tr, "C2")
    aft = train_tier2_aft(tr, "C2")
    pred_aft = np.maximum(aft.predict(xgb.DMatrix(
        _as_cat(base_te, list(schema.TIER2_FEATURES)), enable_categorical=True)), 1e-6)
    wr = tr[tr["won"] == 1]
    from sklearn.dummy import DummyRegressor           # noqa (kept import local)
    reg = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                           tree_method="hist", enable_categorical=True)
    reg.fit(_as_cat(wr, list(schema.TIER2_FEATURES)), wr["bid_price"], verbose=False)
    pred_naive = np.maximum(reg.predict(
        _as_cat(base_te, list(schema.TIER2_FEATURES))), 1e-6)
    res_tr = np.log(np.maximum(wr["bid_price"].to_numpy(), 1e-6)) - np.log(
        np.maximum(reg.predict(_as_cat(wr, list(schema.TIER2_FEATURES))), 1e-6))
    ns = float(max(res_tr.std(), 1e-3))
    c2 = {"naive_sigma": round(ns, 4)}
    c2.update(price_metrics(pred_naive, base_te, "naive", sigma=ns))
    c2.update(price_metrics(pred_aft, base_te, "aft", sigma=1.0))
    for met in ("rmse", "logrmse", "crps"):
        for sl in ("all", "won", "lost"):
            n, a = c2.get(f"naive_{met}_{sl}"), c2.get(f"aft_{met}_{sl}")
            if n and a is not None:
                c2[f"aft_gain_pct_{met}_{sl}"] = round(100.0 * (1.0 - a / n), 2)
    out["C2_price"] = c2
    del tr, base_tr, base_va, base_te; gc.collect()

    fn = f"docs/results/{tag}.json"
    json.dump(out, open(fn, "w"), indent=2, default=float)
    c1 = {x["method"]: x for x in out["C1"]["rows"]}
    print(f"WORKER DONE seed={seed}: install_auc floor={c1['floor'].get('auc_install_all')} "
          f"ref={c1['reference'].get('auc_install_all')} ipw={c1['ipw'].get('auc_install_all')} "
          f"orc={c1['oracle'].get('auc_install_all')} | profit ref={c1['reference']['econ']['profit']} "
          f"ipw={c1['ipw']['econ']['profit']} orc={out['oracle_bidder']['profit']} | "
          f"C2 crps_lost_gain={out['C2_price'].get('aft_gain_pct_crps_lost')}%", flush=True)
    # KEEP the parquet (Ken 13 Jul: HD has space, reusable for later method
    # variants without regeneration). Pass --drop to reclaim the ~2.5GB/seed.
    if "--drop" in sys.argv:
        shutil.rmtree(OUTPUT_DIR / tag, ignore_errors=True)
    else:
        print(f"  parquet kept: {OUTPUT_DIR / tag / 'auctions.parquet'}", flush=True)


if __name__ == "__main__":
    seed = int(sys.argv[1])
    prof = "golden" if "--golden" in sys.argv else "scale10m"
    run(seed, prof)
