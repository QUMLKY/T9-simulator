# T9 — Bayesian-Network Formalisation (Readable Summary)

*Scannable companion to `BN_Formalisation.md` (the full source-of-truth). Same content,
reorganised into bullets and tables. Schema column codes in parentheses. Full derivations,
propositions, and citations live in the source doc; build detail (flags, knobs) lives in the
schema docs.*

---

## 1. The model in one page

The simulator is a **structural causal model**. It draws four pools of persistent entities once,
then generates each auction row by ancestral sampling. Every row carries the full ground truth;
the four experimental conditions **C1–C4** are just different masks over that one master table.

- **Pools (drawn once):** users, apps, campaigns, and — new in v10 — a pool of rival bidders.
  Each entity carries hidden **latent traits** plus visible **observable features**.
- **Per auction:** pick a (user, app, campaign), draw the time/slot context, run the **funnel**
  (click → install → payer → spend), draw the **competing bid** from the rival market, then
  **resolve the auction** (win/lose, clearing price).
- **The economics live in one asymmetry:** the market prices *hidden* value that our own bid
  cannot see, so we systematically win cheap (low-value) impressions and lose the expensive
  (high-value) ones — the **"DSP's biased view"** (selection bias / adverse selection). The data
  layers each reveal a different slice of what we were blind to.
- **What each layer adds:** **MMP** (C2/C4) reveals the *funnel outcomes* of the auctions we lost;
  **SSP** (C3/C4) reveals the *prices and competition density* of those auctions.

---

## 2. Variables

### 2.1 Latents — hidden from every condition (LU / LA / LC / LR)

| Symbol | Code | Name | Domain / distribution |
|---|---|---|---|
| κ_u | LU1 | archetype | {whale, engaged, casual, time_filler, inactive} ~ Cat(π) |
| q_u^clk | LU2 | click propensity | [0,1] ~ Beta(·\|κ_u) |
| q_u^ins | LU3 | install propensity | [0,1] ~ Beta(·\|κ_u) |
| q_u^pay | LU4 | payer probability | [0,1] ~ Beta(·\|κ_u) |
| m_u | LU5 | LTV multiplier | ℝ₊ ~ LogNormal(·\|κ_u) |
| ι_u | LU6 | interest vector | simplex Δ^{\|G\|} ~ Dirichlet(·\|κ_u) |
| λ_a | LA1 | app quality | ℝ₊ ~ LogNormal(0, σ_app) |
| α_a | LA2 | app audience profile | simplex Δ⁵ ~ Dirichlet(k_aud·c[g_a]) |
| η_c | LC1 | creative appeal | ℝ₊ ~ LogNormal(0, σ_cre) |
| χ_c | LC2 | game quality | ℝ₊ ~ LogNormal(0, σ_game) |
| V | LU7 | competing_bid | ℝ₊ · per-auction: V = max_k B_ik (§5) |
| w_k | LR1 | rival value loading | scalar loading on z=z(logEV) (v10) |
| R_{s,k} | LR2 | retargeting shifters | ~ N(0,1) per segment s = os×device (v10) |
| p_k(d), F_k(d) | LR3 | pacing / flights | AR(1) day chain; F ∈ {0,1} (v10) |
| π_k(e) | LR4 | participation | per-exchange propensity (v10) |

Genre taxonomy `G = {casual, strategy, rpg, hypercasual}`. Latents are *ground truth to the
simulator, latent to every model* — they parameterise the draws but never appear as columns.

### 2.2 Observables & outcomes — visible columns (domains consolidated from the schema)

| Family | Codes | Fields | Type / domain |
|---|---|---|---|
| **A** user | A1–A5 | region, city, os, os_version, device_type | int codes · cat (iOS/Android) · string · cat (phone/tablet) |
| **B** app/context | B1–B6 | app_id, app_category, ad_exchange, slot_w, slot_h, slot_format | string · cat(G) · cat · int · int · int (banner/interstitial/rewarded) |
| **C** campaign | C1–C4 | advertiser_id, advertiser_scale, campaign_id, ad_genre | string · cat (indie/mid/major) · string · cat(G) |
| **D** time | D1–D3 | timestamp, hour_of_day, day_of_week | int (unix s) · int 0–23 · int 0–6 |
| **E** click | E1–E2 | click, click_ts | {0,1} · int (−1 if none) |
| **F** install | F1–F2 | install, install_ts | {0,1} · int (−1 if none) |
| **G** LTV | G1–G4 | is_payer, ltv_value, ltv_7d, ltv_30d | {0,1} · ℝ≥0 · ℝ≥0 · ℝ≥0 |
| **H** auction | H1–H4, H9 | floor, bid, won, clearing, bid_density | ℝ₊ · ℝ₊ · {0,1} · ℝ₊∪NaN · ℤ≥0 |

*Full cardinalities and calibration sources: schema docs. E/F/G and H4/H9 are **outcome labels**
(censored per condition); A/B/C/D and H1/H2/H3 are always visible in every condition; latents
never appear.*

---

## 3. The generative process (ancestral sampling)

**Once, up front:** draw the user, app, campaign, and rival pools (latents + observables), plus each
rival's 28-day pacing/flight calendar.

**Then for each auction `i`:**

1. **Assign entities** — draw (user, app) jointly via the IPF pairing `Π^pair`; draw the campaign
   via value-optimised exposure `Π^exp` (BN edges #1, #2 — both marginal-preserving).
2. **Draw context** `ξ_i` — exchange, slot (B3–B6), and time (D1–D3), with the archetype→hour tilt
   (edge #5).
3. **Run the funnel** — click → install → payer → spend, on **every** row (§4). Lost-row outcomes
   are the declared counterfactual; no funnel formula reads the win flag.
4. **Draw the rival market** — each rival's participation, then its bid; the competing bid is the
   maximum (§5).
5. **Resolve the auction** — win rule + first-price clearing (§5).

The joint law factorises exactly in this order (product over pools × product over auctions); the
full factorisation is §2 of the source doc.

---

## 4. The funnel (Tier-1 value side)

Relevance `r = ι_u[g_c]`; stage multiplier `m_st(r) = (1−w_st) + w_st·r`; slot quality
`v = fmt_wt(B6)·size_wt(B4,B5)`.

| Stage | True probability (estimand) | Drawn as |
|---|---|---|
| Click (E1) | ν^clk = β_clk·q^clk·v·m_clk(r)·λ_a·η_c | Bern(ν^clk) |
| Install (F1) \| click | ν^ins = β_ins·q^ins·ease(g_a)·m_ins(r)·λ_a | Bern(ν^ins) |
| Payer (G1) \| install | ν^pay = β_pay·q^pay·m_pay(r)·t_pay(h,δ)  *(edge #4)* | Bern(ν^pay) |
| Spend (G2) \| payer | y^spd ~ LogNormal(μ'_{g_a}, σ)·m_u·χ_c·plat(os)  *(edge #3)* | LogNormal draw |

**Impression value (the oracle target):**

```
EV = ν^clk · ν^ins · ν^pay · ē          where ē = E[spend | payer]        (ev_truth)
```

---

## 5. The auction / pricing mechanism (v10 Private-Rival Market)

**Participation** — who bids (product clipped at 1):

```
Z_ik ~ Bern( min(1, π_k(e)·F_k(d)·gate(p_k(d))) )      A_i = {k : Z_ik=1}      N_i = |A_i|   (H9)
```

**Rival bid** — shared value core + ρ-scaled private structure + noise:

```
log B_ik = log β_0(i) + ρ·( w_k·z + β_R·R_{s,k} + p_k(d) ) + σ_k·ε_ik ,   ε iid N(0,1)
```

- `z = z(log EV)` — the latent impression value (user + app + campaign latents).
- `β_0(i)` — format eCPM base (shared, DSP-predictable core).
- `ρ` — **private-structure amplitude dial**, externally anchored at **ρ\* ≈ 0.8** (not tuned; ρ
  inflates variance rather than raking it — declared limitation, see §8 P3).

**Resolve — win rule and first-price payment:**

```
V   = max_{k ∈ A_i} B_ik                              (LU7 — top competing bid; A_i ≠ ∅ always)
won = 1[ b_own ≥ max(V, floor) ]                      (H3 — win rule)
clr = b_own            if won                         (H4 — first-price: winner pays own bid)
    = V                if lost and V ≥ floor          (competitor's price, revealed to SSP)
    = NaN              if max(b_own, V) < floor       (unsold)
```

Our own bid `b_own` (H2) is a function of **visible features only** — never a latent — so
`b_own ⊥ V | visibles`: the win-curve problem is single-agent, and exploration noise supplies the
bid variation Tier-2 needs.

---

## 6. The four conditions as masking operators

One master row `W = (visible features, labels, won)`. Each condition is an observation map `O_C`
applied row-by-row — **masks, never copies**; identical rows and temporal split across all four.

| Condition | Layers | Funnel labels E/F/G | Price labels H4, H9 | Measures (vs C1) |
|---|---|---|---|---|
| **C1** | DSP | won rows only | hidden | — (baseline) |
| **C2** | DSP+MMP | **all rows** | hidden | value of cross-network attribution |
| **C3** | DSP+SSP | won rows only | **all rows** | value of price / competition visibility |
| **C4** | all | **all rows** | **all rows** | both layers |

```
O_C(W) = ( visible features, won,
           E,F,G      if  MMP_C ∨ won=1   else NaN,      ← funnel mask
           H4, H9     if  SSP_C           else NaN )     ← price mask
```

- **MMP_C = 1** for C2/C4; **SSP_C = 1** for C3/C4.
- **The funnel mask is outcome-dependent (MNAR)** — this is the biased view: because winning
  depends on value, `E[EV | won=1] < E[EV]`, so C1/C3's funnel training data is value-biased
  downward. MMP restores the complete-data funnel (the C1→C2 lift).

**Sensitivity arms (flag-gated, not part of the operative v10 model — reported alongside it).**
Two minimal modifications, OFF = the model above: **B1 (`min_bid_to_win` / H10)** adds an observable
`M = max(V, floor)` on won rows (C3/C4), which upgrades the win-row censoring of `V` from an interval
to a point label — the SSP-full-feedback *upper bound* on price transparency. **B2 (`explore_traffic`)**
perturbs only the bid policy (a ~5% wide-jitter slice on `b_own`), which lifts win-curve quality in all
conditions and marginally compresses the C3−C1 contrast (exploration partially substitutes for SSP).
Neither touches the generative law. Formal statement: `BN_Formalisation.md` §4.1.

---

## 7. Estimands & what the pipeline learns

| Object | Definition | Reported as |
|---|---|---|
| Impression value | EV = ν^clk·ν^ins·ν^pay·ē | ev_ratio (level), ev_spearman (rank) |
| Win curve | W(b;x) = P(V < b \| x) | auc_win, CRPS, price RMSE, ECE |
| P(win) | P(win\|b,x,floor) = 1[b≥floor]·W(b;x) | — |
| Bidder | b\* = argmax_b (EV̂ − b)·Ŵ(b;x) | profit, overpay, surplus |

**Observation structure of the win curve (all conditions):** each row is a *current-status*
observation — win ⇒ `V < b_own` (left-censored); loss **with `b_own > floor`** ⇒ `V > b_own`
(right-censored); floor-caused losses (`b_own ≤ floor`) are uninformative about `V`. C3/C4 add
**exact point prices** `V = clr` on sold losses. Two heads estimate W (AFT + binary classifier);
in C1/C2 they provably coincide (a built-in check).

**Validation targets** (the master must hit these; stated as bands here, formal-constraint version
pending): population CTR 2–5% · click→install 25–40% · install→payer 2–5% · whale share (top-5%
payers) 55–65% · median payer spend ~\$6 · win rate ~30%.

---

## 8. Key properties (propositions, in plain terms)

| # | Claim | Why it matters |
|---|---|---|
| P1 | CPT tilts preserve every calibrated marginal (IPF; error < 1e−12) | turning the BN on never moves a validation target |
| P2 | The pairing/exposure couplings live in the transportation polytope | app_id becomes an **archetype** proxy with **no** marginal footprint |
| P3 | v10's ρ is an **amplitude** dial (build does not rake variance) | honest limitation; shapes the anchor mapping |
| P4 | Value-priced market ⇒ `E[EV\|won] < E[EV]` | the adverse-selection / biased-view mechanism, formally |
| P5 | The win curve is identified from win/loss alone in C1/C2 | SSP buys efficiency + tail, not identifiability |
| P6 | v7/v8 = conditional-IPV ⇒ one price is sufficient | the **structural reason the v8 SSP layer was null** |
| P7 | v10 exits IPV (rival vector + non-reconstructable value term) | private *level* footprints are SSP-recoverable; the latent-value term is recoverable by no one (footprint rule) |
| P8 | ρ anchored to the published value-of-lost-price (Wang 2023) ⇒ ρ\*≈0.8 | the operating point is external, not tuned |

**One-line verdict:** *SSP data improves the model (win/price side); MMP data makes the money
(value side).* Reason 3 is axis-specific — falsified for prediction, sustained for economics.

---

*Diagram: `Schema diagrams/T9_BN_Formal_DAG.svg`. Full doc with all equations, factorisation,
proofs and citations: `docs/BN_Formalisation.md`.*
