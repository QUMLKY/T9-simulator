# T9: Method-Comparison Benchmark, 10M, n = 10 (authoritative; 13 Jul 2026)

The KDD paper's method-comparison evidence: multiple *training methods* on the **same** censored view of
data, all scored against the retained ground truth, on an all / won / **lost**-row split, plus an
**economics** column (bidder profit and regret vs the oracle). Lost = the auctions the bidder lost. No
real-world logs contain lost-auction user action outcomes. It certifies that the benchmark *discriminates
methods and catches failures real data cannot*.

**Setup.** v10 schema, ρ\*=0.8, all BN edges + `rival_pool`; profile `scale10m` (10M rows), **seeds
90213-90222 (n=10)**, one fresh OS process per seed (16 GB). Estimator-side only: no schema/generator
change, so this uses the same generated data as the paper's headline ρ\*=0.8 ablation runs, and the
per-seed parquets are **kept** (`output/mb10m_s*/`) for later method variants. Script:
`scripts/method_bench_{worker,driver}.py`; raw: `docs/results/method_bench_10m_n10.json` (+
`mb10m_s*.json`). Cells are seed means over n=10; the contrast tables carry 95% t-CIs + sign counts.

**Methods.**

| Method | C1 | C2 | C3 | What it is |
|---|---|---|---|---|
| floor (LR) | ✓ | - | ✓ | weak linear baseline |
| reference (XGB) | ✓ | - | ✓ | main 2-tier model |
| IPW | ✓ | - | ✓ | selection-bias correction |
| naive price | - | ✓ | - | ignores price censoring |
| AFT | - | ✓ | - | censoring-aware price model |
| oracle | ✓ | ✓ | ✓ | true-probability ceiling |

*Specifics (condensed from the cells above): floor uses low-cardinality features only (ids dropped),
unweighted; floor/reference train on won rows; IPW = cross-fitted propensities (win model fit on the
validation split, applied to train rows), weights capped at 20; AFT = interval-censored. floor / reference
/ oracle are identical for C1 ≡ C3 by design. Only IPW's weights differ (they use SSP data); the oracle is
view-independent.*

Economics: each method's EV run through the profit-max bidder with the **win model held fixed** across
methods (so profit differences isolate value-estimation quality). Regret = oracle-value profit minus
method profit.

---

## At a glance (condensed from the Full C1 table below; 10M, n = 10)

| Metric | floor (LR) | IPW | our 2-tier | oracle (ceiling) |
|---|---|---|---|---|
| click AUC | 0.644 | 0.698 | **0.709** | 0.784 |
| install AUC | 0.541 | 0.655 | **0.672** | 0.860 |
| payer AUC | **0.592** | 0.565 | 0.586 | 0.835 |
| EV rank corr (Spearman) | 0.092 | 0.549 | **0.589** | 1.000 |
| profit ($) | 2,658 | 4,936 | **5,119** | 10,783 |

Every method sits between a weak floor and a latent-truth ceiling no real dataset can provide, and the
benchmark catches the textbook correction (IPW, inverse propensity weighting) landing *below* the
uncorrected 2-tier model. The payer head is the one noisy row (few payers even at 10M; the three trained
methods are within noise of each other there). The Tier-2 win model has no per-method row because it is
held fixed across methods by design (it is the shared bidder win curve); its own AUC is reported in
`docs/v10_Training_Results.md`.

## Three verdicts only ground truth can issue

### Verdict 1. Model quality: XGBoost decisively beats the floor

The floor→reference step is clean and significant at 10M (the payer-head noise that muddied 1M is gone for
the ranking metrics):

| reference - floor (C1) | mean [95% CI] | Seed count = 10 |
|---|---|---|
| **EV rank corr (Spearman, all)** | **+0.497 [0.452, 0.541]** | **10/10** |
| **install AUC (all)** | **+0.131 [0.128, 0.134]** | **10/10** |
| **click AUC (all)** | **+0.065 [0.060, 0.070]** | **10/10** |
| payer AUC (all) | -0.006 [-0.023, +0.010] | 4 positive and 6 negative, null |
| ev_ratio (all) | -0.051 [-0.179, +0.077] | 4 positive and 6 negative, null |

XGBoost's structured value model is worth a large, tight margin on ranking (install AUC, click AUC, and
EV-Spearman, the Spearman rank correlation between predicted and true expected value). The exceptions are
informative: on the *payer head* even 10M is thin (a few thousand training payers), and on *ev_ratio* (mean
predicted EV / mean true EV; 1.0 = unbiased) both models are miscalibrated in different directions, so
neither dominates. This is exactly why the benchmark reports slices and multiple metrics rather than one
number.

### Verdict 2. Bias correction: cross-fitted IPW does what mean-reweighting does, and no more

The single most important result, and the one no real dataset could produce. Cross-fitted IPW (the textbook
fix for "outcomes observed only on won rows") **improves the aggregate bias metric it targets but degrades
ranking and does not recover the lost-row truth:**

| IPW - reference (C1) | mean [95% CI] | sign | reading |
|---|---|---|---|
| ev_ratio (all) | +0.129 [-0.126, +0.384] | 6/10 | moves the mean-bias toward truth (0.524 → 0.653; oracle = 1.0), directional |
| **EV rank corr (Spearman)** | **-0.040 [-0.066, -0.015]** | **0/10** | **significantly worse ranking** |
| **install AUC (all)** | **-0.017 [-0.018, -0.015]** | **0/10** | **significantly worse** |
| click AUC (all) | -0.011 [-0.012, -0.010] | 0/10 | worse |
| payer AUC (all) | -0.020 [-0.039, -0.001] | 3/10 | worse (CI excludes 0) |

And it does **not close the gap to truth**. It closes *less* of the floor→oracle gap than the uncorrected
reference:

| gap closed (floor → oracle) | reference | IPW |
|---|---|---|
| install AUC | 0.41 | **0.36** |
| EV-Spearman | 0.55 | **0.50** |
| profit | 0.33 | **0.29** |

The mechanism is the paper's identification story, made empirical: **T9's selection operates on *latent*
value** (rivals price the same latent the DSP cannot observe), so a propensity built from *observables* can
shift the aggregate mean (raising ev_ratio) but cannot recover the per-impression structure. The
reweighting also injects variance (mean weight 8.6, **29% of rows hit the cap**), which costs ranking. The
lost-row numbers make the failure explicit: after correction, ev_ratio(lost) = 0.60 and Spearman(lost) =
0.55, still far from the oracle's 1.0. **This is a plausible, standard correction, certified by ground
truth as failing to recover the counterfactual. It is the benchmark's reason to exist.** Cross-fitting
rules out "you overfit the propensities"; the cap sweep below rules out "you capped too hard."

**Cap-robustness (the obvious objection, closed).** The one-line attack on this null, *"IPW fails only
because you capped the weights at 20,"* is wrong. Sweeping the cap over {10, 20, 50, 100, none} (1M, n=10,
same cross-fit method) leaves IPW **significantly worse than the uncorrected reference on ranking at every
cap**, and looser caps make it *strictly worse*:

| weight cap | frac. capped | mean weight | IPW - reference: install AUC [95% CI] | IPW - reference: EV-Spearman |
|---|---|---|---|---|
| 10 | 43% | 6.1 | -0.035 [-0.043, -0.027] (0/10) | -0.099 (1/10) |
| **20 (baseline)** | 36% | 10.0 | -0.043 [-0.051, -0.035] (0/10) | -0.111 (1/10) |
| 50 | 29% | 19.6 | -0.056 [-0.065, -0.047] (0/10) | -0.132 (1/10) |
| 100 | 25% | 33.1 | -0.057 [-0.065, -0.049] (0/10) | -0.128 (1/10) |
| none | 0% | 1,215 | -0.061 [-0.069, -0.052] (0/10) | -0.192 (1/10) |

There is no cap at which the correction recovers ranking; the cap is the only thing preventing a variance
blow-up (uncapped mean weight 1,215). Deltas are larger here than the 10M -0.017 because reweighting
injects more variance at 1M. The *direction* is cap- and scale-invariant. Raw:
`docs/results/ipw_cap_sweep_1m_n10.json` (script `scripts/ipw_cap_sweep.py`).

### Verdict 3. Price censoring: the censoring-aware model recovers lost-row prices; the naive one is confidently wrong

Rock-solid, every seed (C2, where MMP removes the funnel bias so the *only* challenge is price censoring):

| AFT gain over naive | slice | mean [95% CI] | sign |
|---|---|---|---|
| **CRPS** | **lost** | **+41.6% [39.6, 43.6]** | **10/10** |
| CRPS | all | +32.6% [29.0, 36.2] | 10/10 |
| log-RMSE | lost | +26.3% [24.2, 28.5] | 10/10 |
| raw RMSE | lost | +17.1% [15.1, 19.2] | 10/10 |
| raw RMSE | **won** | **-290.7%** | 0/10 |

The won-row cell shows why scoring on observable data alone misleads: the naive model's target on won rows
is its own paid price, so it is accurate only on the observed (won) price distribution (raw RMSE 1.21 vs
7.57) and a practitioner scoring on their own *observable* data would choose it, while it is confidently
wrong on the prices it never saw (fitted σ ≈ 0.27). Only the truth-referenced, lost-row score reverses the
verdict. That reversal is uncomputable on any real log.

---

## Economics (profit and regret vs. the oracle)

Bidder profit and regret vs the oracle, win model fixed across methods (so this isolates value):

| Method | profit ($) | ROAS | regret vs oracle-value [95% CI] |
|---|---|---|---|
| floor | 2,658 | 2.72 | 8,124 [5,915, 10,333] |
| reference | 5,119 | 4.80 | 5,664 [3,426, 7,902] |
| IPW | 4,936 | 4.68 | 5,846 [3,723, 7,970] |
| oracle (value, shared win) | 10,783 | 26.05 | 0 |
| *oracle bidder (true win, absolute ceiling)* | *19,202* | - | - |

Reading: **better value estimation roughly halves regret** (floor → reference), and **IPW adds no profit**
(regret 5,846 ≈ reference 5,664, CIs overlap). A single-seed peek had suggested IPW *beat* reference on
profit; n=10 dissolved it. The smaller-scale caution earned its keep. Note the oracle wins *fewer* auctions
(83K vs ~455K) at far higher ROAS (26 vs ~4.7): it concentrates spend on genuinely high-value impressions,
which is precisely the targeting a good value model provides and a censoring-corrected-but-still-biased
model cannot.

## C3: does SSP-informed reweighting repair the funnel bias? No.

C3's funnel view is identical to C1's by design, so the funnel models (floor / reference / oracle) are
unchanged. What differs in C3 comes from its SSP-informed win model: (a) the IPW weights and (b) the
bidder's win curve. Every **funnel** contrast is within noise of C1's IPW:

| IPW: C3 - C1 | mean [95% CI] | sign |
|---|---|---|
| install AUC (all) | -0.0003 [-0.0015, +0.0009] | 4/10 |
| EV-Spearman (all) | -0.015 [-0.038, +0.008] | 2/10 |
| payer AUC (won) | +0.025 [-0.0003, +0.0495] | 8/10 (CI grazes 0) |

Better price/market visibility does not fix a funnel selection bias. SSP visibility and the funnel bias are
independent axes, the same result the main ablation reports, now shown at the method level.

**Economics differ too, but within noise.** Because the C3 bidder's win curve is SSP-informed, profit /
ROAS / n_won are *not* identical to C1's; they shift even where the value estimate is unchanged. For IPW
(the only method for which C3 economics were computed here), the shift is within seed noise:

| IPW economics | C1 | C3 | C3 - C1 | sign |
|---|---|---|---|---|
| profit ($) | 4,936 | 4,832 | -105 | 4/10 |
| ROAS | 4.68 | 4.82 | +0.14 | 6/10 |
| n_won | 477,421 | 443,715 | -33,706 | 5/10 |

floor / reference / oracle economics were run for C1 only; under C3's win curve they would differ from C1
likewise (even though their value estimates are identical), so the C1 economics rows must not be read as
C3's.

---

## Full C1 comparison (seed means, n = 10)

| Metric | floor | reference | IPW | oracle |
|---|---|---|---|---|
| ev_ratio (all) | 0.575 | 0.524 | **0.653** | 1.000 |
| ev_ratio (won) | 1.490 | **0.991** | 1.169 | 1.000 |
| ev_ratio (lost) | 0.488 | 0.475 | **0.601** | 1.000 |
| EV-Spearman (all) | 0.092 | **0.589** | 0.549 | 1.000 |
| EV-Spearman (won) | 0.070 | **0.577** | 0.535 | 1.000 |
| EV-Spearman (lost) | 0.102 | **0.588** | 0.547 | 1.000 |
| EV-RMSE log1p (all) | 0.079 | **0.078** | 0.084 | 0.000 |
| EV-RMSE log1p (won) | 0.044 | **0.043** | 0.048 | 0.000 |
| EV-RMSE log1p (lost) | 0.088 | **0.087** | 0.094 | 0.000 |
| click AUC (all) | 0.644 | **0.709** | 0.698 | 0.784 |
| click AUC (won) | 0.566 | **0.664** | 0.659 | 0.753 |
| click AUC (lost) | 0.651 | **0.712** | 0.699 | 0.787 |
| install AUC (all) | 0.541 | **0.672** | 0.655 | 0.860 |
| install AUC (won) | 0.551 | **0.662** | 0.647 | 0.853 |
| install AUC (lost) | 0.544 | **0.672** | 0.656 | 0.860 |
| payer AUC (all) | **0.592** | 0.586 | 0.565 | 0.835 |
| payer AUC (won) | **0.588** | 0.574 | 0.567 | 0.859 |
| payer AUC (lost) | **0.596** | 0.585 | 0.565 | 0.830 |
| click ECE (all) | **0.003** | 0.004 | 0.005 | 0.000 |
| click ECE (won) | **0.001** | **0.001** | 0.001 | 0.000 |
| click ECE (lost) | **0.003** | 0.005 | 0.007 | 0.000 |
| install ECE (all) | 0.060 | **0.038** | 0.047 | 0.005 |
| install ECE (won) | **0.012** | 0.016 | 0.014 | 0.011 |
| install ECE (lost) | 0.074 | **0.047** | 0.057 | 0.005 |
| payer ECE (all) | 0.011 | **0.010** | 0.012 | 0.002 |
| payer ECE (won) | 0.003 | **0.003** | 0.004 | 0.005 |
| payer ECE (lost) | 0.013 | **0.011** | 0.014 | 0.003 |
| profit ($) | 2,658 | **5,119** | 4,936 | 10,783 |
| regret vs oracle ($) | 8,124 | **5,664** | 5,846 | 0 |
| ROAS | 2.72 | **4.80** | 4.68 | 26.05 |
| n_won | 431,989 | 454,830 | 477,422 | 83,486 |

## Full C2 comparison (seed means, n = 10)

| Metric | floor | reference | oracle |
|---|---|---|---|
| ev_ratio (all) | **1.035** | 0.889 | 1.000 |
| ev_ratio (won) | 2.102 | **1.513** | 1.000 |
| ev_ratio (lost) | **0.926** | 0.823 | 1.000 |
| EV-Spearman (all) | 0.143 | **0.633** | 1.000 |
| EV-Spearman (won) | 0.098 | **0.606** | 1.000 |
| EV-Spearman (lost) | 0.152 | **0.639** | 1.000 |
| EV-RMSE log1p (all) | 0.079 | **0.072** | 0.000 |
| EV-RMSE log1p (won) | 0.045 | **0.041** | 0.000 |
| EV-RMSE log1p (lost) | 0.088 | **0.080** | 0.000 |
| click AUC (all) | 0.652 | **0.721** | 0.784 |
| click AUC (won) | 0.569 | **0.673** | 0.753 |
| click AUC (lost) | 0.658 | **0.725** | 0.787 |
| install AUC (all) | 0.547 | **0.699** | 0.860 |
| install AUC (won) | 0.546 | **0.688** | 0.853 |
| install AUC (lost) | 0.547 | **0.700** | 0.860 |
| payer AUC (all) | 0.617 | **0.683** | 0.835 |
| payer AUC (won) | 0.598 | **0.682** | 0.859 |
| payer AUC (lost) | 0.619 | **0.682** | 0.830 |
| click ECE (all) | **0.000** | 0.000 | 0.000 |
| click ECE (won) | 0.002 | **0.001** | 0.000 |
| click ECE (lost) | 0.001 | **0.001** | 0.000 |
| install ECE (all) | **0.004** | 0.010 | 0.005 |
| install ECE (won) | 0.052 | **0.033** | 0.011 |
| install ECE (lost) | **0.011** | 0.012 | 0.005 |
| payer ECE (all) | **0.002** | 0.005 | 0.002 |
| payer ECE (won) | 0.009 | **0.006** | 0.005 |
| payer ECE (lost) | **0.003** | 0.006 | 0.003 |
| profit ($) | 3,918 | **7,366** | 10,783 |
| regret vs oracle ($) | 6,864 | **3,417** | 0 |
| ROAS | 2.71 | **5.47** | 26.05 |
| n_won | 594,385 | 490,578 | 83,486 |

## Full C3 comparison (seed means, n = 10)

| Metric | floor | reference | IPW | oracle |
|---|---|---|---|---|
| ev_ratio (all) | 0.575 | 0.524 | **0.621** | 1.000 |
| ev_ratio (won) | 1.490 | **0.991** | 1.109 | 1.000 |
| ev_ratio (lost) | 0.488 | 0.475 | **0.573** | 1.000 |
| EV-Spearman (all) | 0.092 | **0.589** | 0.534 | 1.000 |
| EV-Spearman (won) | 0.070 | **0.577** | 0.518 | 1.000 |
| EV-Spearman (lost) | 0.102 | **0.588** | 0.533 | 1.000 |
| EV-RMSE log1p (all) | 0.079 | **0.078** | 0.085 | 0.000 |
| EV-RMSE log1p (won) | 0.044 | **0.043** | 0.048 | 0.000 |
| EV-RMSE log1p (lost) | 0.088 | **0.087** | 0.095 | 0.000 |
| click AUC (all) | 0.644 | **0.709** | 0.697 | 0.784 |
| click AUC (won) | 0.566 | **0.664** | 0.659 | 0.753 |
| click AUC (lost) | 0.651 | **0.712** | 0.698 | 0.787 |
| install AUC (all) | 0.541 | **0.672** | 0.655 | 0.860 |
| install AUC (won) | 0.551 | **0.662** | 0.645 | 0.853 |
| install AUC (lost) | 0.544 | **0.672** | 0.656 | 0.860 |
| payer AUC (all) | **0.592** | 0.586 | 0.569 | 0.835 |
| payer AUC (won) | 0.588 | 0.574 | **0.591** | 0.859 |
| payer AUC (lost) | **0.596** | 0.585 | 0.567 | 0.830 |
| click ECE (all) | **0.003** | 0.004 | 0.005 | 0.000 |
| click ECE (won) | **0.001** | **0.001** | 0.001 | 0.000 |
| click ECE (lost) | **0.003** | 0.005 | 0.008 | 0.000 |
| install ECE (all) | 0.060 | **0.038** | 0.047 | 0.005 |
| install ECE (won) | **0.012** | 0.016 | 0.016 | 0.011 |
| install ECE (lost) | 0.074 | **0.047** | 0.056 | 0.005 |
| payer ECE (all) | 0.011 | **0.010** | 0.011 | 0.002 |
| payer ECE (won) | 0.003 | **0.003** | 0.004 | 0.005 |
| payer ECE (lost) | 0.013 | **0.011** | 0.013 | 0.003 |
| profit ($) | 3,051 | **5,133** | 4,832 | 10,973 |
| regret vs oracle ($) | 7,922 | **5,840** | 6,142 | 0 |
| ROAS | 2.92 | 4.75 | **4.82** | 24.18 |
| n_won | 436,493 | 459,788 | 443,715 | 85,861 |

*Notes on the three full tables. Bold marks the best value among floor / reference / IPW in each row (oracle excluded): the highest for EV-Spearman, AUC, profit and ROAS; the lowest for the error rows (EV-RMSE, ECE, regret); the closest to 1.0 for ev_ratio (calibration). n_won is descriptive (no best direction), so it is not bolded. (1) IPW is cross-fitted (propensities fit on the validation split, applied to train rows), weights capped at 20. (2) C2 has no IPW column: its funnel trains on all rows, so there is no won-only selection bias to correct. (3) floor / reference / oracle funnel rows are identical for C1 and C3 by design (the funnel uses no price or SSP features); their economics differ because C3's bidder win curve is SSP-informed. (4) EV-RMSE is on log1p(EV) so near-zero impressions do not blow it up; the oracle is 0 by construction. (5) ECE is expected calibration error; the oracle's near-zero ECE confirms the true probabilities are calibrated. (6) regret = oracle profit minus method profit. (7) Validation: these C1 cells reproduce the original 10M run exactly (profit 2,658 / 5,119 / 4,936 / 10,783), confirming the expanded pipeline is consistent with the headline results. Raw: docs/results/full_tables_10m_agg.json; script scripts/full_tables.py.*

## Full C2 price comparison (seed means, n = 10)

| Metric | slice | naive | AFT | AFT gain |
|---|---|---|---|---|
| CRPS | all | **0.9423** | 0.6332 | +32.6% |
| CRPS | won | **0.8553** | 0.8045 | +6.1% |
| CRPS | lost | **0.9666** | 0.5632 | **+41.6%** |
| log-RMSE | all | **1.3436** | 1.1304 | +15.7% |
| log-RMSE | won | 1.2538 | **1.3890** | -10.7% |
| log-RMSE | lost | **1.3651** | 1.0042 | **+26.3%** |
| raw RMSE | all | **29.35** | 24.59 | +16.6% |
| raw RMSE | won | 1.21 | **4.74** | -290.7% |
| raw RMSE | lost | **34.28** | 28.54 | +17.1% |
| (naive fitted σ) | | 0.272 | | |

---

## Caveats (report with the numbers)

1. **Payer head** is the noisiest even at 10M (~thousands of training payers): reference - floor on payer
   AUC is null (4/10). Lead the model-quality claim with install AUC / click AUC / EV-Spearman (all 10/10,
   tight); treat payer AUC as directional.
2. **IPW = cross-fitted (sample-split), cap 20**, 29% of rows capped. This certifies *this* estimator; the
   cross-fit rules out in-sample-propensity overfit, and the cap sweep (Verdict 2) rules out cap-tuning.
   The null is invariant across caps, and looser caps only worsen it. **Doubly-robust (AIPW)** is the
   natural next entrant and would inherit the *same* latent-selection bias. Its outcome-regression term
   trains on the same won-only funnel, so it is robust to model misspecification, not to the MNAR mechanism
   at work here. This is precisely why the open benchmark invites the community to try it, and the null is
   what makes that invitation non-trivial.
3. **Economics** uses one shared, validation-fit win model across methods (isolates value; absolute profit
   is lower than the headline training-fit bidder, but the method ranking is the point). Both-bidder
   robustness is available if needed.
4. Single scale here (10M, n=10); the 1M n=10 companion agrees on all three verdicts (C2 CRPS-lost gain
   +41.0% there vs +41.6% here).

## What goes into the paper (§5 / §6)

The §5 addition (naive floor + gap-closed + a worked external correction) is **run, at 10M, n=10**. The
method table is floor → reference → IPW → oracle on C1, naive → AFT on C2, with the **lost-rows slice** and
the **economics column** as the featured evidence. One-paragraph summary:

> *T9 issues three ground-truth verdicts a real log cannot: (i) a structured value model beats a linear
> floor on ranking by a wide, tight margin (install AUC +0.131, EV-Spearman +0.497, 10/10 seeds); (ii)
> cross-fitted inverse-propensity weighting (the standard correction for won-rows-only funnel labels)
> improves the aggregate bias it targets but degrades ranking, closes less of the floor→oracle gap than
> the uncorrected model, and recovers no lost-row truth, because the selection is on latent value that no
> observable propensity can capture; and (iii) a censoring-aware price model recovers lost-row price
> distributions +41.6% better (CRPS, 10/10) than a naive fit that looks superior on its own observable
> metric. The second and third verdicts are uncomputable on any real dataset. This is the benchmark's
> reason to exist.*
