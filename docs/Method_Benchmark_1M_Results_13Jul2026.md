# T9: Method-Comparison Benchmark Results (1M, n=10 seeds, 13 Jul 2026)

**Purpose.** The "ruler, not race" demonstration for the KDD paper's §5: multiple training
methods on the SAME censored view, all scored against the retained ground truth, including a
**lost-rows-only slice** (auctions the bidder lost, whose outcomes/prices no real log contains).
This is the method-discrimination evidence the red-team flagged as missing (naive floor + one
external correction), now run.

**Setup.** v10 schema, ρ\*=0.8, all BN edges + `rival_pool`, `hist_clearing` ON; profile `test`
(1M rows), seeds 90213-90222 (n=10, ~1.4 min/seed); estimator-side only (no schema change; same
generated data as the paper's headline ρ\*=0.8 ablation runs). Script: `scripts/method_bench.py`; raw:
`docs/results/method_bench_1m_n10.json` (per-seed + aggregates).

**Methods × views.**

| Method | View(s) | What it is |
|---|---|---|
| floor | C1 | plain logistic/ridge funnel heads, won rows, unweighted (low-card features only, ids dropped) |
| reference | C1 | the paper's 2-tier XGBoost funnel heads (won rows, uncorrected) |
| IPW | C1, C3 | reference + per-row weight 1/p̂(win at logged bid), p̂ from the view's own win classifier, clipped at 20 (p_min 0.05) |
| oracle | C1 | Bayes ceiling from the saved true probabilities (no training) |
| naive price | C2 | XGB regression on won-row paid price (own bid), ignores censoring; lognormal σ fitted on its own won-row residuals |
| AFT (incumbent) | C2 | the paper's interval-censored survival model |

---

## Headline 1. C2 price: the instrument discriminates, robustly (the positive)

The censoring-aware AFT recovers the **lost-row price distribution** far better than the naive
won-rows model, and only a truth-holding benchmark can measure this:

| AFT gain over naive | mean [95% CI] | sign |
|---|---|---|
| **CRPS, lost rows** | **+41.0% [38.4, 43.6]** | **10/10** |
| CRPS, all rows | +32.1% [28.0, 36.3] | 10/10 |
| log-RMSE, lost rows | +23.4% [20.4, 26.4] | 10/10 |
| raw RMSE, lost rows | +10.7% [6.8, 14.6] | 9/10 |

The teaching point: the naive model fits its own visible data almost perfectly (fitted σ ≈ 0.21;
near-zero error on won rows). It is **confidently wrong about the prices it never saw**, and on
the naive metric computed the naive way (raw RMSE, won rows) it even looks *better* than the AFT.
Only the truth-referenced, lost-row, distributional score reveals which model actually learned the
market. That is the benchmark's reason to exist, in one table.

## Headline 2. C1 funnel: textbook IPW does NOT recover the truth (the honest null)

| IPW - reference (C1) | mean [95% CI] | sign |
|---|---|---|
| payer AUC (all rows) | +0.006 [-0.028, +0.040] | 5/10, null |
| payer AUC (lost rows) | +0.005 [-0.026, +0.035] | 6/10, null |
| install AUC | **-0.015 [-0.021, -0.009]** | **0/10, consistently worse** |
| EV rank corr (Spearman) | -0.048 [-0.106, +0.009] | 2/10 |
| EV bias (ev_ratio, all) | -0.087 [-0.219, +0.045] | 3/10 (0.63 → 0.55, i.e. *more* under-prediction) |

Textbook inverse-propensity weighting (the standard prescription for "I only see outcomes on
rows I won") does **not** close the gap to truth in this market, and degrades several metrics.
The mechanism is theoretically meaningful, not an implementation accident: **T9's selection
operates on latent value** (rivals price the same latent the DSP cannot see), so propensities
estimated from *observables* cannot undo it; re-weighting mostly injects variance (mean weight
7.9, cap 20 binding at p99). This is the adverse-selection analogue of the paper's identification
story. It is precisely the kind of *negative* certification a method benchmark exists to
issue: a correction that would look plausible on observed data demonstrably fails to recover the
counterfactual truth. No real log could issue that verdict.

**C3 IPW (SSP-informed weights) adds nothing over C1 IPW** (payer AUC -0.012 [-0.026, +0.001],
2/10; EV Spearman +0.010, 6/10). Better price visibility does not repair the funnel bias,
consistent with the paper's "SSP does not fix the DSP's biased view" line.

## Gap-closed (C1, floor → oracle = 0 → 1)

| Metric | reference | IPW |
|---|---|---|
| install AUC | 0.26 | 0.21 |
| EV Spearman (all) | 0.41 | 0.36 |
| EV Spearman (lost) | 0.42 | 0.36 |
| payer AUC | -0.11 | -0.07 |

Even the reference stack closes < half of the floor→oracle gap. The rest is the censoring +
latent headroom the C2-C1 (MMP) contrast buys back. The negative payer-head cells carry a
small-sample caveat (below), not a "LR beats XGBoost" claim.

## Full C1 comparison (seed means, n = 10)

| Metric | floor | reference | IPW | oracle |
|---|---|---|---|---|
| ev_ratio (all) | **0.650** | 0.633 | 0.491 | 1.000 |
| ev_ratio (won) | 1.636 | 1.385 | **1.117** | 1.000 |
| ev_ratio (lost) | 0.547 | **0.553** | 0.420 | 1.000 |
| EV-Spearman (all) | 0.053 | **0.445** | 0.334 | 1.000 |
| EV-Spearman (won) | 0.040 | **0.412** | 0.329 | 1.000 |
| EV-Spearman (lost) | 0.057 | **0.450** | 0.334 | 1.000 |
| EV-RMSE log1p (all) | 0.081 | **0.079** | 0.080 | 0.000 |
| EV-RMSE log1p (won) | 0.048 | **0.045** | **0.045** | 0.000 |
| EV-RMSE log1p (lost) | 0.090 | **0.088** | 0.089 | 0.000 |
| click AUC (all) | 0.643 | **0.680** | 0.615 | 0.784 |
| click AUC (won) | 0.565 | **0.637** | 0.612 | 0.755 |
| click AUC (lost) | 0.649 | **0.682** | 0.613 | 0.786 |
| install AUC (all) | 0.538 | **0.622** | 0.579 | 0.861 |
| install AUC (won) | 0.545 | **0.625** | 0.576 | 0.854 |
| install AUC (lost) | 0.539 | **0.620** | 0.582 | 0.862 |
| payer AUC (all) | **0.579** | 0.556 | 0.543 | 0.836 |
| payer AUC (won) | 0.553 | 0.545 | **0.584** | 0.893 |
| payer AUC (lost) | **0.581** | 0.554 | 0.536 | 0.827 |
| click ECE (all) | **0.003** | 0.006 | 0.011 | 0.001 |
| click ECE (won) | **0.001** | **0.001** | 0.003 | 0.001 |
| click ECE (lost) | **0.004** | 0.009 | 0.015 | 0.001 |
| install ECE (all) | 0.055 | **0.045** | 0.052 | 0.016 |
| install ECE (won) | **0.025** | 0.031 | 0.033 | 0.033 |
| install ECE (lost) | 0.068 | **0.056** | 0.064 | 0.018 |
| payer ECE (all) | 0.014 | **0.011** | 0.019 | 0.006 |
| payer ECE (won) | 0.012 | **0.011** | 0.016 | 0.013 |
| payer ECE (lost) | 0.016 | **0.013** | 0.021 | 0.007 |
| profit ($) | 292 | **725** | 667 | 978 |
| regret vs oracle ($) | 686 | **252** | 311 | 0 |
| ROAS | 3.08 | 8.09 | **11.10** | 29.41 |
| n_won | 35,239 | 54,195 | 43,292 | 7,735 |

## Full C2 comparison (seed means, n = 10)

| Metric | floor | reference | oracle |
|---|---|---|---|
| ev_ratio (all) | 1.169 | **0.968** | 1.000 |
| ev_ratio (won) | 2.242 | **1.752** | 1.000 |
| ev_ratio (lost) | **1.065** | 0.890 | 1.000 |
| EV-Spearman (all) | 0.113 | **0.552** | 1.000 |
| EV-Spearman (won) | 0.083 | **0.530** | 1.000 |
| EV-Spearman (lost) | 0.120 | **0.553** | 1.000 |
| EV-RMSE log1p (all) | 0.086 | **0.080** | 0.000 |
| EV-RMSE log1p (won) | 0.050 | **0.046** | 0.000 |
| EV-RMSE log1p (lost) | 0.096 | **0.089** | 0.000 |
| click AUC (all) | 0.651 | **0.710** | 0.784 |
| click AUC (won) | 0.570 | **0.661** | 0.755 |
| click AUC (lost) | 0.657 | **0.713** | 0.786 |
| install AUC (all) | 0.548 | **0.672** | 0.861 |
| install AUC (won) | 0.555 | **0.662** | 0.854 |
| install AUC (lost) | 0.546 | **0.673** | 0.862 |
| payer AUC (all) | 0.603 | **0.614** | 0.836 |
| payer AUC (won) | 0.619 | **0.697** | 0.893 |
| payer AUC (lost) | 0.601 | **0.604** | 0.827 |
| click ECE (all) | **0.001** | 0.001 | 0.001 |
| click ECE (won) | **0.002** | 0.002 | 0.001 |
| click ECE (lost) | **0.001** | 0.002 | 0.001 |
| install ECE (all) | **0.008** | 0.018 | 0.016 |
| install ECE (won) | 0.055 | **0.052** | 0.033 |
| install ECE (lost) | **0.015** | 0.021 | 0.018 |
| payer ECE (all) | **0.006** | 0.006 | 0.006 |
| payer ECE (won) | 0.012 | **0.011** | 0.013 |
| payer ECE (lost) | **0.006** | 0.007 | 0.007 |
| profit ($) | 346 | **779** | 978 |
| regret vs oracle ($) | 632 | **199** | 0 |
| ROAS | 2.98 | **5.82** | 29.41 |
| n_won | 47,739 | 55,517 | 7,735 |

## Full C3 comparison (seed means, n = 10)

| Metric | floor | reference | IPW | oracle |
|---|---|---|---|---|
| ev_ratio (all) | **0.650** | 0.633 | 0.434 | 1.000 |
| ev_ratio (won) | 1.636 | 1.385 | **1.010** | 1.000 |
| ev_ratio (lost) | 0.547 | **0.553** | 0.370 | 1.000 |
| EV-Spearman (all) | 0.053 | **0.445** | 0.350 | 1.000 |
| EV-Spearman (won) | 0.040 | **0.412** | 0.349 | 1.000 |
| EV-Spearman (lost) | 0.057 | **0.450** | 0.349 | 1.000 |
| EV-RMSE log1p (all) | 0.081 | **0.079** | 0.079 | 0.000 |
| EV-RMSE log1p (won) | 0.048 | 0.045 | **0.044** | 0.000 |
| EV-RMSE log1p (lost) | 0.090 | **0.088** | 0.089 | 0.000 |
| click AUC (all) | 0.643 | **0.680** | 0.611 | 0.784 |
| click AUC (won) | 0.565 | **0.637** | 0.609 | 0.755 |
| click AUC (lost) | 0.649 | **0.682** | 0.610 | 0.786 |
| install AUC (all) | 0.538 | **0.622** | 0.579 | 0.861 |
| install AUC (won) | 0.545 | **0.625** | 0.581 | 0.854 |
| install AUC (lost) | 0.539 | **0.620** | 0.581 | 0.862 |
| payer AUC (all) | **0.579** | 0.556 | 0.539 | 0.836 |
| payer AUC (won) | 0.553 | 0.545 | **0.567** | 0.893 |
| payer AUC (lost) | **0.581** | 0.554 | 0.534 | 0.827 |
| click ECE (all) | **0.003** | 0.006 | 0.011 | 0.001 |
| click ECE (won) | **0.001** | **0.001** | 0.002 | 0.001 |
| click ECE (lost) | **0.004** | 0.009 | 0.015 | 0.001 |
| install ECE (all) | 0.055 | **0.045** | 0.052 | 0.016 |
| install ECE (won) | **0.025** | 0.031 | 0.026 | 0.033 |
| install ECE (lost) | 0.068 | **0.056** | 0.063 | 0.018 |
| payer ECE (all) | 0.014 | **0.011** | 0.018 | 0.006 |
| payer ECE (won) | 0.012 | **0.011** | 0.014 | 0.013 |
| payer ECE (lost) | 0.016 | **0.013** | 0.019 | 0.007 |
| profit ($) | 293 | **714** | 627 | 936 |
| regret vs oracle ($) | 643 | **222** | 309 | 0 |
| ROAS | 3.14 | 7.93 | **8.72** | 27.33 |
| n_won | 35,643 | 54,685 | 48,115 | 7,904 |

*Notes on the three full tables. Bold marks the best value among floor / reference / IPW in each row (oracle excluded): the highest for EV-Spearman, AUC, profit and ROAS; the lowest for the error rows (EV-RMSE, ECE, regret); the closest to 1.0 for ev_ratio (calibration). n_won is descriptive (no best direction), so it is not bolded. (1) IPW here is cross-fitted (propensities fit on the validation split, applied to train rows), consistent with the 10M authoritative report; this is a stronger estimator than the in-sample IPW used in the Headline 2 and gap-closed sections above, so the IPW numbers there are slightly milder. (2) C2 has no IPW column: its funnel trains on all rows, so there is no won-only selection bias to correct. (3) floor / reference / oracle funnel rows are identical for C1 and C3 by design (the funnel uses no price or SSP features); their economics differ because C3's bidder win curve is SSP-informed. (4) EV-RMSE is on log1p(EV) so near-zero impressions do not blow it up; the oracle is 0 by construction. (5) ECE is expected calibration error; the oracle's near-zero ECE confirms the true probabilities are calibrated. (6) regret = oracle profit minus method profit.*

## C2 price comparison (naive vs censoring-aware AFT, seed means, n = 10)

The funnel tables above cover the value heads. C2's distinctive challenge is price censoring, scored separately here (the clearing price is hidden on lost rows, so only a truth-holding benchmark can grade the recovery):

| Metric | slice | naive (won-rows fit) | AFT (incumbent) | AFT gain |
|---|---|---|---|---|
| CRPS | all | 0.9739 | 0.6587 | +32.1% |
| CRPS | won | 0.8794 | 0.8316 | +5.6% |
| CRPS | lost | 1.0011 | 0.5885 | +41.0% |
| log-RMSE | all | 1.3524 | 1.1749 | +12.9% |
| log-RMSE | won | 1.2544 | 1.4324 | -14.1% |
| log-RMSE | lost | 1.3769 | 1.0521 | +23.4% |
| raw RMSE | all | 29.59 | 26.51 | +9.5% |
| raw RMSE | won | 1.21 | 7.57 | -523.5% |
| raw RMSE | lost | 34.56 | 30.62 | +10.7% |

*Reading note. The won-row cells are the metric illusion: the naive model's target on won rows is its own paid price, so it looks spectacular there (raw RMSE 1.21 vs 7.57) while being confidently wrong on lost rows (fitted sigma about 0.21). A practitioner evaluating on their own observable data would pick the naive model; the truth-referenced lost-row column reverses the verdict on every metric family.*

## Caveats (report these with the numbers)

1. **Payer-head sample size:** C1's payer training population at 1M is ~75 rows (~50 payers in
   the test pop); the floor-vs-reference payer AUCs overlap ([0.56, 0.60] vs [0.53, 0.58]);
   treat all payer-AUC rows as noisy; the durable signals are install AUC, EV Spearman, ev_ratio,
   and the C2 price table.
2. **IPW is the simplest textbook variant:** in-sample propensities (the view's own classifier,
   no cross-fitting), hard weight cap at 20. The result certifies *this estimator in this market*,
   not all of IPW; doubly-robust or latent-aware corrections are exactly what the open harness
   invites (and the negative result is what makes that invitation non-trivial).
3. Single scale (1M, the shipped benchmark tier), n=10 seeds, no outlier removal.

## What goes into the paper (§5/§6)

- §5's ADD-ON (naive floor + one worked external baseline) is now **run, not promised**: the
  method table is floor → reference → IPW → oracle on C1, plus naive → AFT on C2, with the
  lost-rows slice as the featured column.
- One-sentence summary for §5: *"The benchmark's first two certifications: a censoring-aware
  price model recovers lost-row price distributions +41% better than a naive fit (10/10 seeds),
  and textbook IPW, the standard prescription for won-only funnel labels, fails to recover
  lost-row truth because the selection operates on latent value: a verdict only a
  ground-truth-holding benchmark can issue."*
- Framing discipline: the IPW null is a **feature** (the instrument catches a plausible-looking
  correction failing), not a bug; report it with the latent-selection mechanism and the
  invitation to beat it.
