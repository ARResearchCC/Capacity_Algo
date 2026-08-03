# SI rebuild — instructions for the writing agent

**Task.** Rewrite the Supplementary Information (SI) into the 8-section structure below.
The current draft lives in `manuscript/SI.md` (near-final prose to *reuse and polish*, not
rewrite from scratch). This document gives (a) the target section order, (b) which existing
SI material feeds each section, (c) which figure(s) each section owns, and (d) the one or two
points the brief discussion must make. Every SI figure has an assigned home — leave none unplaced.

## Keep it SHORT (the SI is already very long)
- **Every figure gets a self-contained caption** — adapt the vetted draft caption in the `.txt`
  beside each figure (`paper_figures/si/` or `main/`), trimmed to: one lead sentence + what is
  plotted (axes, colour/marker, error bars, *n*) + at most one takeaway. The caption carries the
  detail.
- **Discussion prose is minimal — 1–2 sentences per section.** Do NOT restate the caption in the
  body. If a point is already in the caption, don't repeat it in prose.
- Do not add new analyses, panels, or claims.

## How to use this
- Pull prose from the matching *current* SI section (mapping per section) and polish/shorten it.
- Figures are numbered **S1, S2, …** in order of first appearance (renumber as you place them).
  Main-text figures are **Fig 1–6**; cite them as "Fig N (main text)" where a section leans on one.

## Global style (match the main text)
- **UK spelling** (-ise/-our): optimise, minimise, behaviour (harmonise the current mixed SI).
- Past tense for methods/results. Keep SO-CVaR / LP-Avg / LP-Worst; VoLL; PCM.
- **Do not invent numbers** — use the verified values quoted below (from the figure captions).
  Loss of load is essentially all thermal (electrical ≈ 0). Never call SO-CVaR *cheaper* than
  LP-Avg — it is "at comparable cost."

---

## Figure → section placement (master map)

| Figure file | Shows | SI section |
|---|---|---|
| `si/si_fig_variability` | Interannual variability of PV yield / heating / cooling by site (25 yr) | **S1** |
| `si/si_fig_capacities` | Sized PV / battery / hot+cold PCM by method × architecture × climate | **S2** |
| *(tables + equations only)* | Component + cost parameters, CRF | **S3** |
| `si/si_fig_voll` | VoLL **sensitivity** (California): cost & unmet vs Low/Med/High VoLL | **S4** |
| *(equations — you have them)* | LP-Avg / LP-Worst / SO-CVaR formulations | **S5** |
| `si/si_fig_risk_params` | (λ, α) robustness via nested CV | **S6** |
| `si/si_fig_generalization` | Train→test dumbbell (cost & unmet) | **S7** |
| `si/si_fig_loss_of_load` | Delivered loss of load by climate (energy + worst-event) | **S7** |
| `si/si_fig_frontier` | Cost–reliability frontier by climate | **S7** |
| `si/si_fig_dispatch` | Hourly dispatch trace | **pending — not generated; omit** |

**Naming trap:** `si_fig_voll` (VoLL *sensitivity*) ≠ `si_fig_loss_of_load` (delivered loss-of-load
*outcomes*). S4 gets `si_fig_voll`; S7 gets `si_fig_loss_of_load`.

---

## S1 — Sites and climate
**Reuse:** current SI §S1 (site/weather intro only — the detailed input models are dropped, see
note). **Figure:** `si_fig_variability` (→ Fig S1).

- **Site & climate table** (already done): five sites with latitude, longitude, Köppen–Geiger
  zone, elevation, UTC offset, cold→hot: Alaska (Dfc, 59.25°N, −154.62°, 699 m, UTC−9),
  Minnesota (Dfa/Dfb, 44.97°N, −93.26°, 256 m, UTC−6), California (Csb, 37.49°N, −122.42°,
  88 m, UTC−8), Arizona (BWh, 33.45°N, −112.06°, 334 m, UTC−7), Florida (Am/Aw, 25.77°N,
  −80.18°, 7 m, UTC−5). Source: NSRDB metadata (v3.2.2); 25 weather years 1998–2022.
- **Fig S1** (`si_fig_variability`): caption per the draft.
- Discussion (≤2 sentences): variability is modest but real (sizing-relevant CV ≈ 5–13%) and
  the polar site's solar is both least abundant and least predictable — the premise for
  multi-year stochastic sizing and the cross-validation in S7.

## S2 — System optimal capacity
**Figure:** `si_fig_capacities` (→ Fig S2).
- Caption per draft.
- Discussion (≤2 sentences): size follows a risk spectrum (LP-Worst largest, LP-Avg smallest,
  SO-CVaR intermediate); **PCM displaces battery, not PV** (battery −38–86% across climates,
  largest in Minnesota ≈ −86%), and the optimiser recovers the correct thermal regime endogenously.

## S3 — Techno-economic parameters, cost model, and annualisation
**Reuse:** current SI §S5 (component + cost parameters) and §S6 (annualisation). **No figure.**
- Component physical parameters (heat pump COP 3.5 / 10 kW; battery η 0.98, DoD 0.8,
  self-discharge 0.01/24 per h, 0.25 C; hot/cold PCM) and the cost table (PV \$1,500/kW; battery
  \$500/kWh; heat pump \$1,000/kW; PCM \$70/kWh; O&M as listed).
- **Keep the "representative-estimate" note already in current §S5** (order-of-magnitude values
  consistent with NREL ATB / cost literature, applied identically across methods so comparisons
  don't hinge on exact values; if citing ATB, cite the commercial/utility-scale storage line).
- Annualisation: CRF, lifetime 20 yr, discount 0.03 (CRF ≈ 0.067); total annual cost = annualised
  capital + fixed O&M + VoLL penalties.

## S4 — Value of lost load
**Reuse:** current SI §S7 (VoLL table + rationale). **Figure:** `si_fig_voll` (→ Fig S3).
- Three-level VoLL table: Low (thermal \$1 / critical \$30 /kWh), Med (\$3 / \$100), High (\$10 / \$300);
  electrical ≫ thermal (mission-critical electrical vs flexible HVAC).
- **Fig S3** (`si_fig_voll`): caption per draft (shown for California — representative marine).
- Discussion (≤2 sentences): method cost gaps are within fold bands at Low/Med and separate only
  at High, where with PCM SO-CVaR becomes cheapest while its unmet tracks LP-Worst.

## S5 — Optimisation formulation
**Reuse:** current SI §S8 (you have the equations). **No figure.**
- Shared hourly dispatch physics; methods differ only in uncertainty handling: LP-Avg (mean of
  per-year optima), LP-Worst (worst training year), SO-CVaR (two-stage stochastic with
  Rockafellar–Uryasev CVaR linearisation; λ = 0.9, α = 0.9 fixed a priori).

## S6 — Comparison to plain (risk-neutral) stochastic optimisation
**Figure:** `si_fig_risk_params` (→ Fig S4). Cross-reference main **Fig 5**.
- Rationale (1–2 sentences): to isolate the *risk term* from *stochastic optimisation*, SO-CVaR
  (λ = 0.9) is compared against the risk-neutral program (λ = 0); per **Fig 5 (main)** the risk
  term lowers out-of-sample mean unmet **−40%** and worst-fold unmet **−19%** (15/15 cells) for a
  **+1.2%** cost premium.
- **Fig S4** (`si_fig_risk_params`): caption per draft.
- Discussion (≤2 sentences): λ, α are a fixed a-priori risk posture (not tuned); the regret surface
  is smooth, so conclusions do not hinge on the exact values.

## S7 — Out-of-sample evaluation
**Reuse:** current SI §S9 (cross-validation procedure). **Figures:** `si_fig_generalization`
(→ Fig S5), `si_fig_loss_of_load` (→ Fig S6), `si_fig_frontier` (→ Fig S7). Cross-reference main **Fig 3**.
- Procedure (brief): 25 years → five contiguous 5-year blocks; train on 20 yr, test on the held-out
  5 yr; capacities fixed then re-simulated out of sample.
- **Fig S5** (`si_fig_generalization`): caption per draft. Discussion (≤1 sentence): SO-CVaR has the
  smallest train→test reliability gap (**+7.6%**) vs LP-Avg (**+17.2%**, overfits) and LP-Worst
  (**−51.4%**, over-conservative).
- **Fig S6** (`si_fig_loss_of_load`): caption per draft. Discussion (≤1 sentence): loss of load is
  essentially all thermal and SO-CVaR lowers thermal unmet energy vs LP-Avg in every climate
  (worst-event *duration* is a coarse proxy, not the optimised quantity).
- **Fig S7** (`si_fig_frontier`): caption per draft. Discussion (≤1 sentence): SO-CVaR/PCM is on the
  cost–reliability frontier in all five climates and dominates LP-Worst/PCM in Florida.

## S8 — Diesel benchmark
**Reuse:** current SI §S10 (configuration, sizing, dispatch, fuel/cost model, break-even definition).
Cross-reference main **Fig 6**. **No SI figure.**
- Brief: all-diesel config (no PV/battery/PCM); generator sized to peak + 20% reserve; load-dependent
  linear fuel curve; delivered-fuel tiers (\$13 / \$45 / \$400 per gal); price-independent break-even.
  Result (Fig 6): break-even **\$0.78–8.90/gal** across climates — below even routine forward-delivered
  fuel. Keep the "finalise generator cost/fuel-curve against a cited source" note.

---

## Open decisions for the author (defaults chosen; change if you disagree)
1. **S7 has three figures** — `si_fig_generalization` overlaps the main-text calibration figure
   (Fig 3; same train→test data as a dumbbell). Keep it as the per-climate SI companion, or drop it
   for a leaner S7. **Default = keep.**
2. **Component physical parameters** sit in S3 with the costs. Alternative: move to the S5 formulation
   preamble. **Default = S3.**
3. **Dispatch trace** (`si_fig_dispatch`) is not generated (needs the Gurobi env). **Default = omit.**

## Main-text cross-references to update (author handles main text)
Figures moved, so update SI callouts in Results/Discussion to the new numbering: capacities → Fig S2;
VoLL sensitivity → Fig S3; risk-parameter robustness → Fig S4; loss-of-load → Fig S6; frontier → Fig S7.

## Note on what was removed
The detailed **input-generation models** (solar PV, passive thermal, stochastic electrical load —
current SI §S2, §S3, §S4) are **dropped from the SI** at the author's instruction. S1 now covers only
the sites, climate, and interannual weather variability. (If the main-text Methods does not already
describe these models, re-add a compact version — otherwise they are gone.)
