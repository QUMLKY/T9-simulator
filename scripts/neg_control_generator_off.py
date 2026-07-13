"""THE leak gate (Blueprint section 4 / section 6 commit 4b / section 8 constraint 4):
the GENERATOR-OFF end-to-end negative control for v8 hist_clearing.

LOGIC
-----
Generate a master with EVERY v8 price edge OFF (so LU7 has NO competition /
placement / exchange / time structure: lu7 = max(base_e*exp(gamma*z+eps_g),
base_e*exp(eps_x)), z = focal log ev_truth). Then run the pipeline WITH
hist_clearing ON for C1 and C3 and assert:

    auc_win(C3) - auc_win(C1)  ~  0   (within seed noise, |.| <= 0.005)

WHY THIS IS THE GATE (and the row-poison unit test is necessary-but-insufficient)
---------------------------------------------------------------------------------
In the OFF generator the ONLY bucket-key channel into LU7 is base_e =
pay_shape * ecpm_target[_format] -- i.e. LU7 depends on slot_format (one of the
4 bucket keys) and on the focal z(log ev_truth) (which carries ad_genre /
campaign latents, NOT in the SSP bucket key). app_id, ad_exchange and daypart
have ZERO effect on OFF LU7 (auctions.py _prices l.394-408). Moreover
slot_format and ad_exchange are ALREADY raw Tier-2 features (schema.CAT_FEATURES
+ NUM_FEATURES). So:

  * hist_clearing_dsp (H7, from bid_price=H2 on won rows, ALL conds) and
  * hist_clearing_ssp (H6, from H4 over won + sold-lost rows, C3/C4 only)

can differ ONLY through slot_format -- which the model already has. The extra
de-biasing that H6 provides (adding sold-lost clears) carries NO app_id /
ad_exchange / daypart structure the raw features lack. Hence a CORRECT,
leak-free encoding must give auc_win C3-C1 ~ 0 on the OFF generator.

A row-poison unit test only proves self-exclusion mechanics. It CANNOT see the
high-cardinality failure mode: a sparse {app_id x slot_format x ad_exchange x
daypart} cell collapsing to a near-copy of ONE neighbour's clearing (target
laundering). That passes row-poison (no row sees its OWN target) yet fabricates
a positive C3-C1. Only THIS aggregate, OFF-generator run exposes it: any C3-C1
lift here is, by construction, leakage from the OOF/encoding machinery -- not
real SSP value -- because there is no real SSP structure to recover.

PASS THRESHOLD
--------------
  PASS  : |auc_win(C3) - auc_win(C1)|  <= 0.005   (golden 100k / 1M, 1 seed)
  WARN  : 0.005 < |.| <= 0.010                    (re-run a 2nd seed)
  FAIL  : |.| > 0.010  OR  (C3-C1) > 0.005 with the same sign across 2 seeds
          -> the encoding leaks; fix EB shrink (raise HCP_EB_K) / coarsen the
          bucket key, do NOT proceed to commit 4b.

Contrast gate (sanity that the harness is wired right): on an ON generator
(competition + ev_market) the SAME script must show C3-C1 >= +0.02 (Blueprint
4b success gate). OFF ~ 0 AND ON >> 0 together prove the lift is the FEATURE,
not the machinery.

Usage:
  ./venv/Scripts/python.exe scripts/neg_control_generator_off.py --profile golden
  ./venv/Scripts/python.exe scripts/neg_control_generator_off.py --profile test --seed 90211
  # contrast (must be >= +0.02):  ... --profile test --on
"""
from __future__ import annotations

import argparse
import sys

from t9sim import simulate
from t9sim import pipeline as P
from t9sim.paths import OUTPUT_DIR

# all six v8 price edges forced OFF (the generator-OFF state)
V8_PRICE_EDGES_OFF = {
    "competition": False, "app_quality_price": False, "exchange_price": False,
    "app_price": False, "time_competition": False, "ev_market": False,
}
# the ON contrast config (Blueprint 4b): structure an SSP feature can recover
V8_ON = {"competition": True, "ev_market": True}     # ev_market auto-forced anyway

NEG_PASS = 0.005          # |C3-C1| must be <= this for a clean PASS
NEG_WARN = 0.010          # above this -> FAIL
ON_GATE = 0.02            # ON contrast must clear this


def _auc_win(profile, out_name):
    """Run the pipeline with hist_clearing ON; return {cond: auc_win}.
    Requires Step 3b: run_profile(..., hist_clearing=True). Falls back to a
    clear error if that kwarg is not yet wired."""
    try:
        out = P.run_profile(out_name, do_shap=False, hist_clearing=True)
    except TypeError as e:
        sys.exit(f"run_profile has no hist_clearing kwarg yet (Step 3b "
                 f"unbuilt): {e}")
    return {c: out["conditions"][c].get("auc_win")
            for c in ("C1", "C2", "C3", "C4")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="golden")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--on", action="store_true",
                    help="run the ON contrast (expect C3-C1 >= +0.02)")
    args = ap.parse_args()

    edges = V8_ON if args.on else V8_PRICE_EDGES_OFF
    tag = "on_contrast" if args.on else "off_negctrl"
    out_name = f"{args.profile}_{tag}"

    print(f"== generating {out_name}: edges={edges} ==")
    ok = simulate.run(profile=args.profile, out_name=out_name,
                      seed=args.seed, bn_edges=edges)
    if not ok:
        print("WARNING: generator direction checks failed (continuing)")

    # OFF identity cross-check (only meaningful for the OFF run, default seed)
    if not args.on:
        from t9sim.fingerprint import fingerprint
        fp = fingerprint(OUTPUT_DIR / out_name)
        # NOTE: equals 0xdf0ac3e18624cf2b only for profile=golden, default seed
        print(f"OFF fingerprint = {fp:#018x}  "
              f"(expect 0xdf0ac3e18624cf2b iff profile=golden & default seed)")

    auc = _auc_win(args.profile, out_name)
    print(f"auc_win by condition: {auc}")
    delta = auc["C3"] - auc["C1"]
    print(f"\nauc_win  C3 - C1 = {delta:+.4f}")

    if args.on:
        verdict = "PASS" if delta >= ON_GATE else "FAIL"
        print(f"ON contrast gate (>= +{ON_GATE}): {verdict}")
        sys.exit(0 if verdict == "PASS" else 1)

    if abs(delta) <= NEG_PASS:
        print(f"NEG-CONTROL PASS (|delta| <= {NEG_PASS}): encoding is leak-free")
        sys.exit(0)
    if abs(delta) <= NEG_WARN:
        print(f"NEG-CONTROL WARN ({NEG_PASS} < |delta| <= {NEG_WARN}): "
              f"re-run a 2nd seed; FAIL if same sign & > {NEG_PASS}")
        sys.exit(0)
    print(f"NEG-CONTROL FAIL (|delta| > {NEG_WARN}): hist_clearing LEAKS. "
          f"Raise HCP_EB_K / coarsen the bucket key. Do NOT proceed to 4b.")
    sys.exit(1)


if __name__ == "__main__":
    main()
