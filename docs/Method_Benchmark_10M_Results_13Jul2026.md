# T9 — Method-Comparison Benchmark, 10M, n = 10 (authoritative; 13 Jul 2026)

The KDD paper's method-comparison evidence: multiple *training methods* on the
**same** censored view, all scored against the retained ground truth — on an all / won / **lost**-row
split (lost = auctions the bidder lost, whose outcomes no real log contains) — plus an **economics** column
(bidder profit and regret vs the oracle). It certifies that the benchmark *discriminates methods
and catches failures real data cannot*.

**Setup.** v10 schema, ρ\*=0.8, all BN edges + `rival_pool`; profile `scale10m` (10M rows), **seeds
90213–90222 (n=10)**, one fresh OS process per seed (16 GB). Estimator-side only — no schema/generator
change, so this uses the same generated data as the paper's headline ρ\*=0.8 ablation runs, and the per-seed parquets are **kept**
(`t9_sim/output/mb10m_s*/`) for later method variants. Script: `t9_sim/scripts/method_bench_{worker,driver}.py`;
raw: `docs/results/method_bench_10m_n10.json` (+ `mb10m_s*.json`). Cells are seed means over n=10; the
contrast tables carry 95% t-CIs + sign counts.

**Methods.**

| Method | C1 | C2 | C3 | What it is |
|---|---|---|---|---|
| floor (LR) | ✓ | – | ✓ | weak linear baseline |
| reference (XGB) | ✓ | – | ✓ | main 2-tier model |
| IPW | ✓ | – | ✓ | selection-bias correction |
| naive price | – | ✓ | – | ignores price censoring |
| AFT | – | ✓ | – | censoring-aware price model |
| oracle | ✓ | ✓ | ✓ | true-probability ceiling |

*Specifics (condensed from the cells above): floor uses low-cardinality features only (ids dropped),
unweighted; floor/reference train on won rows; IPW = cross-fitted propensities (win model fit on the
validation split, applied to train rows), weights capped at 20; AFT = interval-censored. floor / reference
/ oracle are identical for C1 ≡ C3 by design — only IPW's weights differ (they use SSP data); the oracle is
view-independent.*

Economics: each method's EV run through the profit-max bidder with the **win model held fixed** across
methods (so profit differences isolate value-estimation quality). Regret = oracle-value profit − method
profit.

---

## Three verdicts only ground truth can issue

### Verdict 1 — model quality: XGBoost decisively beats the floor

The floor→reference step is clean and significant at 10M (the payer-head noise that muddied 1M is
gone for the ranking metrics):

| reference − floor (C1) | mean [95% CI] | sign |
|---|---|---|
| **EV rank corr (Spearman, all)** | **+0.497 [0.452, 0.541]** | **10/10** |
| **install AUC (all)** | **+0.131 [0.128, 0.134]** | **10/10** |
| **click AUC (all)** | **+0.065 [0.060, 0.070]** | **10/10** |
| payer AUC (all) | −0.006 [−0.023, +0.010] | 4/10 — null |
| ev_ratio (all) | −0.051 [−0.179, +0.077] | 4/10 — null |

XGBoost's structured value model is worth a large, tight margin on ranking (install AUC, click AUC, and
EV-Spearman — the Spearman rank correlation between predicted and true expected value). The exceptions are
informative: on the *payer head* even 10M is thin (a few thousand training payers), and on *ev_ratio*
(mean predicted EV / mean true EV; 1.0 = unbiased) both models are miscalibrated in different directions,
so neither dominates — which is
exactly why the benchmark reports slices and multiple metrics rather than one number.

### Verdict 2 — bias correction: cross-fitted IPW does what mean-reweighting does, and no more

The single most important result, and the one no real dataset could produce. Cross-fitted IPW — the
textbook fix for "outcomes observed only on won rows" — **improves the aggregate bias metric it targets
but degrades ranking and does not recover the lost-row truth:**

| IPW − reference (C1) | mean [95% CI] | sign | reading |
|---|---|---|---|
| ev_ratio (all) | +0.129 [−0.126, +0.384] | 6/10 | moves the mean-bias toward truth (0.524 → 0.653; oracle = 1.0) — directional |
| **EV rank corr (Spearman)** | **−0.040 [−0.066, −0.015]** | **0/10** | **significantly worse ranking** |
| **install AUC (all)** | **−0.017 [−0.018, −0.015]** | **0/10** | **significantly worse** |
| click AUC (all) | −0.011 [−0.012, −0.010] | 0/10 | worse |
| payer AUC (all) | −0.020 [−0.039, −0.001] | 3/10 | worse (CI excludes 0) |

And it does **not close the gap to truth** — it closes *less* of the floor→oracle gap than the
uncorrected reference:

| gap closed (floor → oracle) | reference | IPW |
|---|---|---|
| install AUC | 0.41 | **0.36** |
| EV-Spearman | 0.55 | **0.50** |
| profit | 0.33 | **0.29** |

The mechanism is the paper's identification story, made empirical: **T9's selection operates on *latent*
value** (rivals price the same latent the DSP cannot observe), so a propensity built from *observables*
can shift the aggregate mean (raising ev_ratio) but cannot recover the per-impression structure — and the
reweighting injects variance (mean weight 8.6, **29% of rows hit the cap**), which costs ranking. The
lost-row numbers make the failure explicit: after correction, ev_ratio(lost) = 0.60 and Spearman(lost) =
0.55, still far from the oracle's 1.0. **This is a plausible, standard correction, certified by ground
truth as failing to recover the counterfactual — the benchmark's reason to exist.** Cross-fitting rules
out "you overfit the propensities"; the cap sweep below rules out "you capped too hard."

**Cap-robustness (the obvious objection, closed).** The one-line attack on this null — *"IPW fails only
because you capped the weights at 20"* — is wrong. Sweeping the cap over {10, 20, 50, 100, none} (1M,
n=10, same cross-fit method) leaves IPW **significantly worse than the uncorrected reference on ranking at
every cap**, and looser caps make it *strictly worse*:

| weight cap | frac. capped | mean weight | IPW − reference: install AUC [95% CI] | IPW − reference: EV-Spearman |
|---|---|---|---|---|
| 10 | 43% | 6.1 | −0.035 [−0.043, −0.027] (0/10) | −0.099 (1/10) |
| **20 (baseline)** | 36% | 10.0 | −0.043 [−0.051, −0.035] (0/10) | −0.111 (1/10) |
| 50 | 29% | 19.6 | −0.056 [−0.065, −0.047] (0/10) | −0.132 (1/10) |
| 100 | 25% | 33.1 | −0.057 [−0.065, −0.049] (0/10) | −0.128 (1/10) |
| none | 0% | 1,215 | −0.061 [−0.069, −0.052] (0/10) | −0.192 (1/10) |

There is no cap at which the correction recovers ranking; the cap is the only thing preventing a variance
blow-up (uncapped mean weight 1,215). Deltas are larger here than the 10M −0.017 because reweighting
injects more variance at 1M — the *direction* is cap- and scale-invariant. Raw:
`docs/results/ipw_cap_sweep_1m_n10.json` (script `t9_sim/scripts/ipw_cap_sweep.py`).

### Verdict 3 — price censoring: the censoring-aware model recovers lost-row prices; the naive one is confidently wrong

Rock-solid, every seed (C2, where MMP removes the funnel bias so the *only* challenge is price
censoring):

| AFT gain over naive | slice | mean [95% CI] | sign |
|---|---|---|---|
| **CRPS** | **lost** | **+41.6% [39.6, 43.6]** | **10/10** |
| CRPS | all | +32.6% [29.0, 36.2] | 10/10 |
| log-RMSE | lost | +26.3% [24.2, 28.5] | 10/10 |
| raw RMSE | lost | +17.1% [15.1, 19.2] | 10/10 |
| raw RMSE | **won** | **−290.7%** | 0/10 |

The won-row cell shows why scoring on observable data alone misleads: the naive model's target on won
rows is its own paid price, so it is accurate only on the observed (won) price distribution (raw RMSE 1.21
vs 7.57) and a practitioner scoring on their own *observable* data would choose it — while it is
confidently wrong on the prices it never saw (fitted σ ≈
0.27). Only the truth-referenced, lost-row score reverses the verdict. That reversal is uncomputable on
any real log.

---

## Economics (profit and regret vs. the oracle)

Bidder profit and regret vs the oracle, win model fixed across methods (so this isolates value):

| Method | profit ($) | ROAS | regret vs oracle-value [95% CI] |
|---|---|---|---|
| floor | 2,658 | 2.72 | 8,124 [5,915, 10,333] |
| reference | 5,119 | 4.80 | 5,664 [3,426, 7,902] |
| IPW | 4,936 | 4.68 | 5,846 [3,723, 7,970] |
| oracle (value, shared win) | 10,783 | 26.05 | 0 |
| *oracle bidder (true win, absolute ceiling)* | *19,202* | — | — |

Reading: **better value estimation roughly halves regret** (floor → reference), and **IPW adds no profit**
(regret 5,846 ≈ reference 5,664, CIs overlap). A single-seed peek had suggested IPW *beat* reference on
profit; n=10 dissolved it — the smaller-scale caution earned its keep. Note the oracle wins *fewer*
auctions (83K vs ~455K) at far higher ROAS (26 vs ~4.7): it concentrates spend on genuinely high-value
impressions, which is precisely the targeting a good value model provides and a censoring-corrected-but-still-
biased model cannot.

## C3 — does SSP-informed reweighting repair the funnel bias? No.

C3's funnel view is identical to C1's by design, so the funnel models (floor / reference / oracle) are
unchanged. What differs in C3 comes from its SSP-informed win model: (a) the IPW weights and (b) the
bidder's win curve. Every **funnel** contrast is within noise of C1's IPW:

| IPW: C3 − C1 | mean [95% CI] | sign |
|---|---|---|
| install AUC (all) | −0.0003 [−0.0015, +0.0009] | 4/10 |
| EV-Spearman (all) | −0.015 [−0.038, +0.008] | 2/10 |
| payer AUC (won) | +0.025 [−0.0003, +0.0495] | 8/10 (CI grazes 0) |

Better price/market visibility does not fix a funnel selection bias — SSP visibility and the funnel bias
are independent axes, the same result the main ablation reports, now shown at the method level.

**Economics differ too — but within noise.** Because the C3 bidder's win curve is SSP-informed, profit /
ROAS / n_won are *not* identical to C1's; they shift even where the value estimate is unchanged. For IPW
(the only method for which C3 economics were computed here), the shift is within seed noise:

| IPW economics | C1 | C3 | C3 − C1 | sign |
|---|---|---|---|---|
| profit ($) | 4,936 | 4,832 | −105 | 4/10 |
| ROAS | 4.68 | 4.82 | +0.14 | 6/10 |
| n_won | 477,421 | 443,715 | −33,706 | 5/10 |

floor / reference / oracle economics were run for C1 only; under C3's win curve they would differ from C1
likewise (even though their value estimates are identical), so the C1 economics rows must not be read as
C3's.

---

## Full C1 comparison (seed means, n = 10)

| Metric | floor | reference | IPW | oracle |
|---|---|---|---|---|
| ev_ratio (all) | 0.575 | 0.524 | **0.653** | 1.000 |
| ev_ratio (won) | **1.490** | 0.991 | 1.169 | 1.000 |
| ev_ratio (lost) | 0.488 | 0.475 | **0.601** | 1.000 |
| EV-Spearman (all) | 0.092 | **0.589** | 0.549 | 1.000 |
| EV-Spearman (won) | 0.070 | **0.577** | 0.535 | 1.000 |
| EV-Spearman (lost) | 0.103 | **0.588** | 0.547 | 1.000 |
| click AUC (all) | 0.644 | **0.709** | 0.698 | 0.784 |
| click AUC (won) | 0.566 | **0.664** | 0.659 | 0.753 |
| click AUC (lost) | 0.651 | **0.712** | 0.699 | 0.787 |
| install AUC (all) | 0.541 | **0.672** | 0.655 | 0.860 |
| install AUC (won) | 0.551 | **0.663** | 0.647 | 0.853 |
| install AUC (lost) | 0.544 | **0.673** | 0.656 | 0.860 |
| payer AUC (all) | **0.592** | 0.586 | 0.565 | 0.835 |
| payer AUC (won) | **0.588** | 0.574 | 0.567 | 0.859 |
| payer AUC (lost) | **0.596** | 0.585 | 0.565 | 0.830 |
| profit ($) | 2,658 | **5,119** | 4,936 | 10,783 |
| ROAS | 2.72 | **4.80** | 4.68 | 26.05 |
| n_won | 431,989 | 454,830 | **477,422** | 83,486 |

## Full C2 price comparison (seed means, n = 10)

| Metric | slice | naive | AFT | AFT gain |
|---|---|---|---|---|
| CRPS | all | **0.9423** | 0.6332 | +32.6% |
| CRPS | won | **0.8553** | 0.8045 | +6.1% |
| CRPS | lost | **0.9666** | 0.5632 | **+41.6%** |
| log-RMSE | all | **1.3436** | 1.1304 | +15.7% |
| log-RMSE | won | 1.2538 | **1.3890** | −10.7% |
| log-RMSE | lost | **1.3651** | 1.0042 | **+26.3%** |
| raw RMSE | all | **29.35** | 24.59 | +16.6% |
| raw RMSE | won | 1.21 | **4.74** | −290.7% |
| raw RMSE | lost | **34.28** | 28.54 | +17.1% |
| (naive fitted σ) | | 0.272 | | |

---

## Caveats (report with the numbers)

1. **Payer head** is the noisiest even at 10M (~thousands of training payers): reference − floor on payer
   AUC is null (4/10). Lead the model-quality claim with install AUC / click AUC / EV-Spearman (all 10/10,
   tight); treat payer AUC as directional.
2. **IPW = cross-fitted (sample-split), cap 20**, 29% of rows capped. This certifies *this* estimator; the
   cross-fit rules out in-sample-propensity overfit and the cap sweep (Verdict 2) rules out cap-tuning —
   the null is invariant across caps and looser caps only worsen it. **Doubly-robust (AIPW)** is the
   natural next entrant and would inherit the *same* latent-selection bias — its outcome-regression term
   trains on the same won-only funnel, so it is robust to model misspecification, not to the MNAR
   mechanism at work here — which is precisely why the open benchmark invites the community to try it, and
   the null is what makes that invitation non-trivial.
3. **Economics** uses one shared, validation-fit win model across methods (isolates value; absolute profit
   is lower than the headline training-fit bidder, but the method ranking is the point). Both-bidder robustness
   is available if needed.
4. Single scale here (10M, n=10); the 1M n=10 companion agrees on all three verdicts (C2 CRPS-lost gain
   +41.0% there vs +41.6% here).

## What goes into the paper (§5 / §6)

The §5 addition (naive floor + gap-closed + a worked external correction) is **run, at 10M, n=10** — the
method table is floor → reference → IPW → oracle on C1, naive → AFT on C2, with the **lost-rows slice**
and the **economics column** as the featured evidence. One-paragraph summary:

> *T9 issues three ground-truth verdicts a real log cannot: (i) a structured value model beats a linear
> floor on ranking by a wide, tight margin (install AUC +0.131, EV-Spearman +0.497, 10/10 seeds); (ii)
> cross-fitted inverse-propensity weighting — the standard correction for won-rows-only funnel labels —
> improves the aggregate bias it targets but degrades ranking, closes less of the floor→oracle gap than
> the uncorrected model, and recovers no lost-row truth, because the selection is on latent value that no
> observable propensity can capture; and (iii) a censoring-aware price model recovers lost-row price
> distributions +41.6% better (CRPS, 10/10) than a naive fit that looks superior on its own observable
> metric. The second and third verdicts are uncomputable on any real dataset — which is the benchmark's
> reason to exist.*
