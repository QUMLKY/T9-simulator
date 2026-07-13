# T9 — Method-Comparison Benchmark Results (1M, n=10 seeds, 13 Jul 2026)

**Purpose.** The "ruler, not race" demonstration for the KDD paper's §5: multiple training
methods on the SAME censored view, all scored against the retained ground truth — including a
**lost-rows-only slice** (auctions the bidder lost, whose outcomes/prices no real log contains).
This is the method-discrimination evidence the red-team flagged as missing (naive floor + one
external correction), now run.

**Setup.** v10 schema, ρ\*=0.8, all BN edges + `rival_pool`, `hist_clearing` ON; profile `test`
(1M rows), seeds 90213–90222 (n=10, ~1.4 min/seed); estimator-side only (no schema change; same
generated data as the paper's headline ρ\*=0.8 ablation runs). Script: `t9_sim/scripts/method_bench.py`; raw:
`docs/results/method_bench_1m_n10.json` (per-seed + aggregates).

**Methods × views.**

| Method | View(s) | What it is |
|---|---|---|
| floor | C1 | plain logistic/ridge funnel heads, won rows, unweighted (low-card features only — ids dropped) |
| reference | C1 | the paper's 2-tier XGBoost funnel heads (won rows, uncorrected) |
| IPW | C1, C3 | reference + per-row weight 1/p̂(win at logged bid), p̂ from the view's own win classifier, clipped at 20 (p_min 0.05) |
| oracle | C1 | Bayes ceiling from the saved true probabilities (no training) |
| naive price | C2 | XGB regression on won-row paid price (own bid), ignores censoring; lognormal σ fitted on its own won-row residuals |
| AFT (incumbent) | C2 | the paper's interval-censored survival model |

---

## Headline 1 — C2 price: the instrument discriminates, robustly (the positive)

The censoring-aware AFT recovers the **lost-row price distribution** far better than the naive
won-rows model, and only a truth-holding benchmark can measure this:

| AFT gain over naive | mean [95% CI] | sign |
|---|---|---|
| **CRPS, lost rows** | **+41.0% [38.4, 43.6]** | **10/10** |
| CRPS, all rows | +32.1% [28.0, 36.3] | 10/10 |
| log-RMSE, lost rows | +23.4% [20.4, 26.4] | 10/10 |
| raw RMSE, lost rows | +10.7% [6.8, 14.6] | 9/10 |

The teaching point: the naive model fits its own visible data almost perfectly (fitted σ ≈ 0.21;
near-zero error on won rows) — it is **confidently wrong about the prices it never saw**, and on
the naive metric computed the naive way (raw RMSE, won rows) it even looks *better* than the AFT.
Only the truth-referenced, lost-row, distributional score reveals which model actually learned the
market. That is the benchmark's reason to exist, in one table.

## Headline 2 — C1 funnel: textbook IPW does NOT recover the truth (the honest null)

| IPW − reference (C1) | mean [95% CI] | sign |
|---|---|---|
| payer AUC (all rows) | +0.006 [−0.028, +0.040] | 5/10 — null |
| payer AUC (lost rows) | +0.005 [−0.026, +0.035] | 6/10 — null |
| install AUC | **−0.015 [−0.021, −0.009]** | **0/10 — consistently worse** |
| EV rank corr (Spearman) | −0.048 [−0.106, +0.009] | 2/10 |
| EV bias (ev_ratio, all) | −0.087 [−0.219, +0.045] | 3/10 (0.63 → 0.55, i.e. *more* under-prediction) |

Textbook inverse-propensity weighting — the standard prescription for "I only see outcomes on
rows I won" — does **not** close the gap to truth in this market, and degrades several metrics.
The mechanism is theoretically meaningful, not an implementation accident: **T9's selection
operates on latent value** (rivals price the same latent the DSP cannot see), so propensities
estimated from *observables* cannot undo it; re-weighting mostly injects variance (mean weight
7.9, cap 20 binding at p99). This is the adverse-selection analogue of the paper's identification
story — and it is precisely the kind of *negative* certification a method benchmark exists to
issue: a correction that would look plausible on observed data demonstrably fails to recover the
counterfactual truth. No real log could issue that verdict.

**C3 IPW (SSP-informed weights) adds nothing over C1 IPW** (payer AUC −0.012 [−0.026, +0.001],
2/10; EV Spearman +0.010, 6/10) — better price visibility does not repair the funnel bias,
consistent with the paper's "SSP does not fix the DSP's biased view" line.

## Gap-closed (C1, floor → oracle = 0 → 1)

| Metric | reference | IPW |
|---|---|---|
| install AUC | 0.26 | 0.21 |
| EV Spearman (all) | 0.41 | 0.36 |
| EV Spearman (lost) | 0.42 | 0.36 |
| payer AUC | −0.11 | −0.07 |

Even the reference stack closes < half of the floor→oracle gap — the rest is the censoring +
latent headroom the C2−C1 (MMP) contrast buys back. The negative payer-head cells carry a
small-sample caveat (below), not a "LR beats XGBoost" claim.

## Full comparison tables (seed means, n = 10; CIs for the headline contrasts above and in the JSON)

### C1 — funnel methods × all metrics × slice (all / won / lost rows)

| Metric | floor (LR) | reference (XGB) | XGB + IPW | oracle (ceiling) |
|---|---|---|---|---|
| ev_ratio (all) | 0.6503 | 0.6326 | 0.5458 | 1.00 |
| ev_ratio (won) | 1.6364 | 1.3852 | 1.2309 | 1.00 |
| ev_ratio (lost) | 0.5472 | 0.5527 | 0.4716 | 1.00 |
| ev_spearman (all) | 0.0529 | 0.4454 | 0.3969 | 1.00 |
| ev_spearman (won) | 0.0400 | 0.4121 | 0.3843 | 1.00 |
| ev_spearman (lost) | 0.0566 | 0.4500 | 0.3972 | 1.00 |
| auc_click (all) | 0.6435 | 0.6797 | 0.6358 | 0.7843 |
| auc_click (won) | 0.5647 | 0.6368 | 0.6227 | 0.7550 |
| auc_click (lost) | 0.6495 | 0.6825 | 0.6346 | 0.7862 |
| auc_install (all) | 0.5376 | 0.6217 | 0.6065 | 0.8609 |
| auc_install (won) | 0.5455 | 0.6246 | 0.6133 | 0.8540 |
| auc_install (lost) | 0.5393 | 0.6202 | 0.6055 | 0.8619 |
| auc_payer (all) | 0.5792 | 0.5563 | 0.5621 | 0.8357 |
| auc_payer (won) | 0.5531 | 0.5449 | 0.6339 | 0.8930 |
| auc_payer (lost) | 0.5807 | 0.5542 | 0.5588 | 0.8265 |

*Reading notes.* (1) The floor's seemingly-decent all-rows ev_ratio (0.65) is an average of two
large opposite errors — it over-predicts won rows by +64% and under-predicts lost rows by −45%;
the slice columns expose what the blended number hides. (2) IPW helps payer discrimination on the
rows it reweights toward being representative of (won: 0.545 → 0.634) but not on the lost rows
themselves — the correction improves the *seen* distribution's fit without recovering the
*unseen* one; that gap is the latent-selection mechanism made visible. (3) Payer rows carry the
small-n caveat below.

### C3 — same funnel task; only the IPW weights differ (SSP-informed win model)

C3's funnel view is identical to C1's by design (funnel labels on won rows only in both), so the
floor, reference, and oracle rows are C1's by construction; the one measured difference is IPW
with weights from C3's SSP-informed win classifier.

| Metric | floor (≡ C1) | reference (≡ C1) | XGB + IPW (C3, measured) | XGB + IPW (C1, for comparison) | oracle |
|---|---|---|---|---|---|
| ev_ratio (all) | 0.6503 | 0.6326 | 0.5312 | 0.5458 | 1.00 |
| ev_ratio (won) | 1.6364 | 1.3852 | 1.1700 | 1.2309 | 1.00 |
| ev_ratio (lost) | 0.5472 | 0.5527 | 0.4614 | 0.4716 | 1.00 |
| ev_spearman (all) | 0.0529 | 0.4454 | 0.4070 | 0.3969 | 1.00 |
| ev_spearman (won) | 0.0400 | 0.4121 | 0.3915 | 0.3843 | 1.00 |
| ev_spearman (lost) | 0.0566 | 0.4500 | 0.4082 | 0.3972 | 1.00 |
| auc_click (all) | 0.6435 | 0.6797 | 0.6357 | 0.6358 | 0.7843 |
| auc_click (won) | 0.5647 | 0.6368 | 0.6231 | 0.6227 | 0.7550 |
| auc_click (lost) | 0.6495 | 0.6825 | 0.6344 | 0.6346 | 0.7862 |
| auc_install (all) | 0.5376 | 0.6217 | 0.6068 | 0.6065 | 0.8609 |
| auc_install (won) | 0.5455 | 0.6246 | 0.6084 | 0.6133 | 0.8540 |
| auc_install (lost) | 0.5393 | 0.6202 | 0.6065 | 0.6055 | 0.8619 |
| auc_payer (all) | 0.5792 | 0.5563 | 0.5499 | 0.5621 | 0.8357 |
| auc_payer (won) | 0.5531 | 0.5449 | 0.6226 | 0.6339 | 0.8930 |
| auc_payer (lost) | 0.5807 | 0.5542 | 0.5466 | 0.5588 | 0.8265 |

*Reading note.* The C3 and C1 IPW columns agree to ≈0.01 on every metric (paired contrast CIs in
the headline section): better price visibility does not repair the funnel's selection bias.

### C2 — price methods × all metrics × slice (naive vs censoring-aware AFT)

| Metric | slice | naive (won-rows fit) | AFT (incumbent) | AFT gain |
|---|---|---|---|---|
| CRPS | all | 0.9739 | 0.6587 | **+32.1%** |
| CRPS | won | 0.8794 | 0.8316 | +5.6% |
| CRPS | lost | 1.0011 | 0.5885 | **+41.0%** |
| log-RMSE | all | 1.3524 | 1.1749 | +12.9% |
| log-RMSE | won | 1.2544 | 1.4324 | −14.1% |
| log-RMSE | lost | 1.3769 | 1.0521 | **+23.4%** |
| raw RMSE | all | 29.59 | 26.51 | +9.5% |
| raw RMSE | won | 1.21 | 7.57 | −523.5% |
| raw RMSE | lost | 34.56 | 30.62 | +10.7% |

*Reading note.* The won-row cells are the metric illusion on display: the naive model's target on
won rows is its own paid price, so it looks spectacular there (raw RMSE 1.21 vs 7.57 — a −524%
"win") while being confidently wrong on lost rows (fitted σ ≈ 0.21). A practitioner evaluating on
their own observable data would pick the naive model; the truth-referenced lost-row column reverses
the verdict on every metric family.

## Caveats (report these with the numbers)

1. **Payer-head sample size:** C1's payer training population at 1M is ~75 rows (~50 payers in
   the test pop); the floor-vs-reference payer AUCs overlap ([0.56, 0.60] vs [0.53, 0.58]) —
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
  price model recovers lost-row price distributions +41% better than a naive fit (10/10 seeds) —
  and textbook IPW, the standard prescription for won-only funnel labels, fails to recover
  lost-row truth because the selection operates on latent value: a verdict only a
  ground-truth-holding benchmark can issue."*
- Framing discipline: the IPW null is a **feature** (the instrument catches a plausible-looking
  correction failing), not a bug; report it with the latent-selection mechanism and the
  invitation to beat it.
