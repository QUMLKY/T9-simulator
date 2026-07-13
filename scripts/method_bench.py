"""Method-comparison benchmark (the 'ruler' demo) at 1M, n=10 seeds.

Per view, multiple TRAINING METHODS on the SAME censored data, all scored
against the retained ground truth — including a lost-rows-only slice (the
rows no real log ever reveals):

  C1 funnel:  floor (logistic/linear heads)  <  reference (XGB, uncorrected)
              <  IPW (XGB + 1/p_win weights) <  oracle (true probabilities)
  C3 funnel:  IPW re-run (weights from C3's SSP-informed win model;
              reference/floor identical to C1 by design identity)
  C2 price:   naive floor (won-rows regression, ignores censoring)
              vs AFT incumbent (interval-censored), scored on true lost-row LU7

Estimator-side only: no schema/generator change; data byte-identical to the
anchor battery (profile 'test', rho*=0.8, all BN edges + rival_pool).

Run from repo root:  t9_sim/venv/Scripts/python.exe t9_sim/scripts/method_bench.py [--golden]
"""
import sys, json, time, shutil, argparse
sys.path.insert(0, 't9_sim/src'); sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error

import t9sim.simulate as sim
from t9sim import censor, schema
from t9sim.paths import OUTPUT_DIR
from t9sim.pipeline import (PipelineConfig, temporal_split, funnel_population,
                            train_tier1, predict_ev, train_tier2_aft,
                            train_tier2_clf, attach_hist_clearing, t2_feats,
                            _eval_pop, _as_cat, XGB_CLF, XGB_REG)

V7_BN = {e: True for e in ("pairing", "os_spend", "payer_timing", "hour", "exposure")}
EDGES = {**V7_BN, "rival_pool": True}
RHO = 0.8
SEEDS = [90213, 90214, 90215, 90216, 90217, 90218, 90219, 90220, 90221, 90222]

# LR floor: low-cardinality one-hots only (high-card ids dropped — disclosed)
LR_CATS = ["os", "os_version", "device_type", "app_category", "ad_exchange",
           "slot_format", "advertiser_scale", "ad_genre"]
LR_NUMS = ["slot_width", "slot_height", "hour_of_day", "day_of_week"]
IPW_PMIN = 0.05                      # weight cap = 1/0.05 = 20 (disclosed)


# ── LR floor ──────────────────────────────────────────────────────────
def _lr_design(df, cols=None):
    X = pd.get_dummies(df[LR_CATS].astype(str), dtype=np.float32)
    for c, s in (("slot_width", 100.0), ("slot_height", 100.0),
                 ("hour_of_day", 24.0), ("day_of_week", 7.0)):
        X[c] = (df[c].to_numpy() / s).astype(np.float32)
    if cols is not None:
        X = X.reindex(columns=cols, fill_value=0.0)
    return X, list(X.columns)


def train_floor(train, val, cond):
    """Plain LR/Ridge heads on the same funnel populations (won rows in C1/C3),
    unweighted, low-card features only. The weak-model, uncorrected floor."""
    models, cols = {}, None
    for stage, target, is_clf in (("click", "click", True),
                                  ("install", "install", True),
                                  ("payer", "is_payer", True),
                                  ("spend", "ltv_value", False)):
        tr = funnel_population(train, cond, stage)
        X, cols_ = _lr_design(tr, cols)
        cols = cols if cols is not None else cols_
        y = tr[target].to_numpy()
        if is_clf:
            m = LogisticRegression(max_iter=2000, solver="lbfgs")
            if len(np.unique(y)) < 2:                    # degenerate guard
                m = float(y.mean() if len(y) else 0.0)
            else:
                m.fit(X, y)
        else:
            m = Ridge(alpha=1.0)
            m.fit(X, y)
        models[stage] = m
    return models, cols


def floor_ev(models, cols, df):
    X, _ = _lr_design(df, cols)
    def proba(m):
        return (np.full(len(df), m) if isinstance(m, float)
                else m.predict_proba(X)[:, 1])
    pc_ = proba(models["click"]); pi_ = proba(models["install"])
    pp_ = proba(models["payer"])
    es_ = np.clip(models["spend"].predict(X), 0, None)
    return pc_ * pi_ * pp_ * es_, pc_, pi_, pp_, es_


# ── IPW-weighted XGBoost Tier-1 ───────────────────────────────────────
def train_tier1_ipw(pc, train, val, cond, w_col="_ipw_w"):
    """train_tier1's in-memory branch + per-row sample_weight (1/p_win, clipped)."""
    feats = pc.t1_feats
    models = {}
    for stage, target, is_clf in (("click", "click", True),
                                  ("install", "install", True),
                                  ("payer", "is_payer", True),
                                  ("spend", "ltv_value", False)):
        tr = funnel_population(train, cond, stage)
        va = funnel_population(val, cond, stage)
        Xtr, Xva = _as_cat(tr, feats), _as_cat(va, feats)
        params = XGB_CLF if is_clf else XGB_REG
        cls = xgb.XGBClassifier if is_clf else xgb.XGBRegressor
        m = cls(**params)
        m.fit(Xtr, tr[target], sample_weight=tr[w_col].to_numpy(),
              eval_set=[(Xva, va[target])], verbose=False)
        models[stage] = m
    return models


def ipw_weights(clf_bst, clf_feats, df):
    """w = 1 / clip(p_win_at_logged_bid, IPW_PMIN, 1). In-sample p_win from the
    view's own win classifier (all C1-visible data; no leakage)."""
    p = clf_bst.predict(xgb.DMatrix(_as_cat(df, clf_feats),
                                    enable_categorical=True))
    return 1.0 / np.clip(p, IPW_PMIN, 1.0)


# ── funnel scoring vs truth, all/won/lost slices ─────────────────────
def eval_funnel(name, ev_parts, test):
    ev_hat, p_c, p_i, p_p, _ = ev_parts
    res = {"method": name}
    masks = {"all": np.ones(len(test), bool),
             "won": test["won"].to_numpy() == 1,
             "lost": test["won"].to_numpy() == 0}
    ev_true = test["ev_truth"].to_numpy()
    preds = {"click": p_c, "install": p_i, "payer": p_p}
    for sl, m in masks.items():
        te = test[m]; ev_h, ev_t = ev_hat[m], ev_true[m]
        res[f"ev_ratio_{sl}"] = round(float(ev_h.mean() / max(ev_t.mean(), 1e-12)), 4)
        res[f"ev_spearman_{sl}"] = round(float(spearmanr(ev_h, ev_t).correlation), 4)
        for stage, target in (("click", "click"), ("install", "install"),
                              ("payer", "is_payer")):
            pop = _eval_pop(te, stage)
            if len(pop) > 50 and pop[target].nunique() > 1:
                p = preds[stage][m][_eval_pop_mask(te, stage)]
                res[f"auc_{stage}_{sl}"] = round(roc_auc_score(pop[target], p), 4)
    return res


def _eval_pop_mask(te, stage):
    if stage == "click":
        return np.ones(len(te), bool)
    if stage == "install":
        return (te["click"] == 1).to_numpy()
    return (te["install"] == 1).to_numpy()


def oracle_funnel_row(test):
    """Bayes ceiling from the saved TRUE probabilities — no training."""
    parts = (test["ev_truth"].to_numpy(), test["p_click"].to_numpy(),
             test["p_install"].to_numpy(), test["p_payer"].to_numpy(),
             test["e_ltv"].to_numpy())
    return eval_funnel("oracle", parts, test)


def gap_closed(rows, oracle, metric, higher_better=True, target=None):
    """fraction of floor→oracle gap closed, per method."""
    fl = next(r for r in rows if r["method"] == "floor")
    out = {}
    o = target if target is not None else oracle.get(metric)
    f = fl.get(metric)
    if o is None or f is None or abs(o - f) < 1e-9:
        return None
    for r in rows:
        v = r.get(metric)
        out[r["method"]] = round((v - f) / (o - f), 3) if v is not None else None
    return out


# ── C2 price floor vs AFT ────────────────────────────────────────────
def price_metrics(pred_ecpm, test, prefix, sigma=None):
    """Point RMSE (raw + log space) and, given a lognormal sigma, closed-form
    CRPS — so the censoring-aware AFT and the naive regressor are compared
    both as point predictors and as distributions."""
    from scipy.stats import norm
    lu7 = test["lu7_competing_bid"].to_numpy()
    pred = np.maximum(np.asarray(pred_ecpm, float), 1e-6)
    err = pred - lu7
    lerr = np.log(pred) - np.log(np.maximum(lu7, 1e-6))
    wm = test["won"].to_numpy() == 1
    out = {}
    for sl, m in (("all", np.ones(len(lu7), bool)), ("won", wm), ("lost", ~wm)):
        out[f"{prefix}_rmse_{sl}"] = round(float(np.sqrt(np.mean(err[m] ** 2))), 4)
        out[f"{prefix}_logrmse_{sl}"] = round(float(np.sqrt(np.mean(lerr[m] ** 2))), 4)
    if sigma is not None:
        zc = -lerr / sigma
        crps = sigma * (zc * (2 * norm.cdf(zc) - 1)
                        + 2 * norm.pdf(zc) - 1.0 / np.sqrt(np.pi))
        for sl, m in (("all", np.ones(len(lu7), bool)), ("won", wm), ("lost", ~wm)):
            out[f"{prefix}_crps_{sl}"] = round(float(np.mean(crps[m])), 4)
    return out


def run_seed(seed, profile="test"):
    tag = f"mb_s{seed % 100}"
    t0 = time.time()
    sim.run(profile, out_name=tag, seed=seed, rho=RHO, bn_edges=EDGES)

    pc = PipelineConfig()
    path = OUTPUT_DIR / tag / "auctions.parquet"
    needed = list(set(schema.NEEDED) | {"p_click", "p_install", "p_payer", "e_ltv"})
    import pyarrow.parquet as pq
    pf_cols = set(pq.ParquetFile(path).schema_arrow.names)
    df = pd.read_parquet(path, columns=[c for c in needed if c in pf_cols])
    for c in pc.t1_cat:
        if c in df.columns:
            df[c] = df[c].astype("category")
    sp = temporal_split(df)
    tr_m, va_m, te_m = (sp == s for s in ("train", "val", "test"))
    base_tr, base_va, base_te = df[tr_m.values], df[va_m.values], df[te_m.values]
    out = {"seed": seed, "n_rows": len(df)}

    # ---------- C1 / C3 funnel methods ----------
    for cond in ("C1", "C3"):
        tr = censor.view(base_tr, cond)
        va = censor.view(base_va, cond)
        te = base_te.copy()
        attach_hist_clearing(tr, va, te, censor.CONDITIONS[cond]["clearing"])
        feats2 = t2_feats(cond, True, view_cols=tr.columns)
        clf_bst, clf_feats = train_tier2_clf(tr, va, cond, feats=feats2)
        w = ipw_weights(clf_bst, clf_feats, tr)
        tr = tr.assign(_ipw_w=w)
        rows = []
        if cond == "C1":                       # floor + reference: C1 only (C3 identical by design)
            fl_models, fl_cols = train_floor(tr, va, cond)
            rows.append(eval_funnel("floor", floor_ev(fl_models, fl_cols, base_te), base_te))
            ref = train_tier1(pc, tr, va, cond)
            rows.append(eval_funnel("reference", predict_ev(pc, ref, base_te), base_te))
        ipw = train_tier1_ipw(pc, tr, va, cond)
        rows.append(eval_funnel("ipw", predict_ev(pc, ipw, base_te), base_te))
        if cond == "C1":
            rows.append(oracle_funnel_row(base_te))
        out[cond] = {"rows": rows,
                     "ipw_weight_stats": {"mean": round(float(w.mean()), 3),
                                          "max": round(float(w.max()), 3),
                                          "p99": round(float(np.quantile(w, 0.99)), 3)}}
        if cond == "C1":
            oracle_row = rows[-1]
            gc = {}
            for met, hb, tgt in (("auc_payer_all", True, None),
                                 ("auc_payer_lost", True, None),
                                 ("auc_install_all", True, None),
                                 ("ev_spearman_all", True, None),
                                 ("ev_spearman_lost", True, None),
                                 ("ev_ratio_all", True, 1.0),
                                 ("ev_ratio_lost", True, 1.0)):
                g = gap_closed(rows, oracle_row, met, hb, target=tgt)
                if g:
                    gc[met] = g
            out["C1"]["gap_closed"] = gc

    # ---------- C2 price: naive floor vs AFT incumbent ----------
    cond = "C2"
    tr = censor.view(base_tr, cond)
    va = censor.view(base_va, cond)
    aft = train_tier2_aft(tr, cond)
    pred_aft = np.maximum(aft.predict(xgb.DMatrix(
        _as_cat(base_te, list(schema.TIER2_FEATURES)), enable_categorical=True)), 1e-6)
    wr = tr[tr["won"] == 1]
    reg = xgb.XGBRegressor(**{k: v for k, v in XGB_REG.items()
                              if k != "early_stopping_rounds"})
    # C2 never sees H4; in first-price the winner's paid price IS its own bid,
    # so the naive won-rows price target is bid_price (visible in every view).
    reg.fit(_as_cat(wr, list(schema.TIER2_FEATURES)), wr["bid_price"],
            verbose=False)
    pred_naive = np.maximum(reg.predict(
        _as_cat(base_te, list(schema.TIER2_FEATURES))), 1e-6)
    # naive's best-shot distribution: lognormal around its prediction, sigma
    # fitted on its own visible (won-row) residuals
    res_tr = np.log(np.maximum(wr["bid_price"].to_numpy(), 1e-6)) - np.log(
        np.maximum(reg.predict(_as_cat(wr, list(schema.TIER2_FEATURES))), 1e-6))
    naive_sigma = float(max(res_tr.std(), 1e-3))
    c2 = {"naive_sigma": round(naive_sigma, 4)}
    c2.update(price_metrics(pred_naive, base_te, "naive", sigma=naive_sigma))
    c2.update(price_metrics(pred_aft, base_te, "aft", sigma=1.0))  # AFT_SCALE
    for met in ("rmse", "logrmse", "crps"):
        for sl in ("all", "won", "lost"):
            n, a = c2.get(f"naive_{met}_{sl}"), c2.get(f"aft_{met}_{sl}")
            if n and a is not None:
                c2[f"aft_gain_pct_{met}_{sl}"] = round(100.0 * (1.0 - a / n), 2)
    out["C2_price"] = c2

    out["mins"] = round((time.time() - t0) / 60, 1)
    shutil.rmtree(OUTPUT_DIR / tag, ignore_errors=True)   # regenerable by seed
    return out


def ci(vals):
    v = np.array([x for x in vals if x is not None], float)
    if len(v) < 2:
        return None
    m, se = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
    h = stats.t.ppf(0.975, len(v) - 1) * se
    return {"mean": round(float(m), 4),
            "ci95": [round(float(m - h), 4), round(float(m + h), 4)],
            "n_pos": int((v > 0).sum()), "n": len(v)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", action="store_true", help="100K sanity run, 1 seed")
    args = ap.parse_args()
    profile = "golden" if args.golden else "test"
    seeds = SEEDS[:1] if args.golden else SEEDS

    rows = []
    for seed in seeds:
        r = run_seed(seed, profile=profile)
        rows.append(r)
        c1 = {x["method"]: x for x in r["C1"]["rows"]}
        print(f">>> seed={seed} ({r['mins']}min) "
              f"payer_auc floor={c1['floor'].get('auc_payer_all')} "
              f"ref={c1['reference'].get('auc_payer_all')} "
              f"ipw={c1['ipw'].get('auc_payer_all')} "
              f"oracle={c1['oracle'].get('auc_payer_all')} | "
              f"C2 aft_gain crps_lost={r['C2_price'].get('aft_gain_pct_crps_lost')}% "
              f"logrmse_lost={r['C2_price'].get('aft_gain_pct_logrmse_lost')}%", flush=True)
        json.dump(rows, open(f"docs/results/method_bench_{profile}_partial.json", "w"),
                  indent=2, default=float)

    # ---- aggregate: per method x metric mean/CI + gap-closed + contrasts ----
    summary = {}
    metrics = [k for k in rows[0]["C1"]["rows"][0] if k != "method"]
    for method in ("floor", "reference", "ipw", "oracle"):
        summary[method] = {}
        for met in metrics:
            vals = []
            for r in rows:
                row = next((x for x in r["C1"]["rows"] if x["method"] == method), None)
                if row:
                    vals.append(row.get(met))
            c = ci(vals)
            if c:
                summary[method][met] = c
    ipw_gain = {}
    for met in ("auc_payer_all", "auc_payer_lost", "auc_install_all",
                "ev_spearman_all", "ev_spearman_lost", "ev_ratio_all", "ev_ratio_lost"):
        d = []
        for r in rows:
            m = {x["method"]: x.get(met) for x in r["C1"]["rows"]}
            if m.get("ipw") is not None and m.get("reference") is not None:
                d.append(m["ipw"] - m["reference"])
        ipw_gain[met] = ci(d)
    c3_vs_c1_ipw = {}
    for met in ("auc_payer_all", "auc_payer_lost", "ev_spearman_all", "ev_ratio_all"):
        d = []
        for r in rows:
            i1 = next((x.get(met) for x in r["C1"]["rows"] if x["method"] == "ipw"), None)
            i3 = next((x.get(met) for x in r["C3"]["rows"] if x["method"] == "ipw"), None)
            if i1 is not None and i3 is not None:
                d.append(i3 - i1)
        c3_vs_c1_ipw[met] = ci(d)
    c2 = {k: ci([r["C2_price"].get(k) for r in rows])
          for k in rows[0]["C2_price"]}
    gc_summary = {}
    for met in rows[0]["C1"].get("gap_closed", {}):
        gc_summary[met] = {m: ci([r["C1"]["gap_closed"].get(met, {}).get(m)
                                  for r in rows])
                           for m in ("floor", "reference", "ipw", "oracle")}

    final = {"profile": profile, "rho": RHO, "n_seeds": len(rows),
             "rows": rows, "summary_C1": summary,
             "ipw_minus_reference_C1": ipw_gain,
             "ipw_C3_minus_C1": c3_vs_c1_ipw,
             "C2_price": c2, "gap_closed_C1": gc_summary}
    fn = f"docs/results/method_bench_{'golden' if args.golden else '1m_n10'}.json"
    json.dump(final, open(fn, "w"), indent=2, default=float)
    print("\n===== IPW - reference (C1), mean [95% CI] (n_pos/n) =====")
    for met, s in ipw_gain.items():
        if s:
            print(f"  {met:20} {s['mean']:+8.4f}  {s['ci95']}  ({s['n_pos']}/{s['n']})")
    print("Saved", fn)


if __name__ == "__main__":
    main()
