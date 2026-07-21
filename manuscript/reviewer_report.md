# Critical reviewer report — full manuscript

A single referee report on the assembled paper (Abstract, Introduction, Methods+SI,
Results, Discussion, Conclusion), synthesizing a five-lens panel (methodology/statistics,
narrative & honesty, results/figures, novelty/positioning, language/structure) with my own
read on top. Written as the tough-but-fair referee this paper will actually get.

---

## Overall recommendation: **MAJOR REVISION** — but the core is sound and the honesty is a genuine asset

This is an unusually *honest* manuscript. It refuses the easy over-claims (never "cheaper,"
never a "knee," never a VoLL-crossover), it owns the fact that the proposed method carries a
small cost premium, and it front-loads its own caveats. The CVaR formulation is standard and
correctly specified, the out-of-sample cross-validation is more rigorous than the
single-representative-year sizing the field usually does, and the **out-of-sample calibration
idea — delivered-vs-planned reliability — is a genuine, publishable contribution.** Two of the
five lenses said minor revision on that basis.

But three lenses said major revision, and they are right, for reasons that cluster into **two
issues that dominate the rest.** Neither requires new physics; one requires a single modest
re-run. If they are fixed, this becomes a strong paper. If they are not, a methods-aware
referee will likely reject or bounce it.

---

## The two issues that matter most (fix these before anything else)

### 1. The main comparison uses the wrong baseline — the risk-neutral stochastic program (λ=0) is missing
*(novelty lens, severity 4; my #1 concern)*

The paper's thesis is that **risk-aware** (CVaR) planning is the contribution. But the
head-to-head comparison is CVaR versus two *deterministic* heuristics — LP-Avg (mean of
per-year optima) and LP-Worst (single worst year). The natural scientific baseline — the
**risk-neutral two-stage stochastic program (λ=0)**, which is exactly what Macmillan [8] and
standard SO represent, and which already exists in your own model (it is the λ=0 corner of the
risk sweep) — is relegated to an SI robustness check.

Why this is the crux, not a nitpick: your headline result (calibration) may well be a
**"stochastic optimization beats deterministic heuristics"** result rather than a
**"risk-aware beats risk-neutral"** result. A proper SO solved over all scenarios should
generalize far better than LP-Avg's plug-in average — so λ=0 might be *just as well-calibrated*
as λ=0.9. If so, the calibration lead does **not** support the CVaR-specific claim; it supports
"use stochastic optimization at all." The *risk* term's distinctive value would then live only
in the **tail/variance** result (§3.4), where λ>0 genuinely should beat λ=0.

**This was the single most likely reason the paper gets rejected** — and it has now been
**resolved in the paper's favour by the data.**

> **RESOLVED (2026-07-20, `fig_riskterm`).** Comparing CVaR (λ=0.9) directly against the
> risk-neutral SO (λ=0) within the same nested CV (PCM, 15 cells), CVaR delivers **lower
> out-of-sample unmet in 15/15 cells (median −40%)** and **lower tail unmet in 15/15 (median
> −19%)**, scaling with VoLL, for a **median +1.2% cost**. So the risk term earns its keep
> against a *proper risk-neutral stochastic baseline*, not merely against deterministic
> heuristics. This is now shown in a new main figure and stated in Results §3.4, Discussion
> §4.2, and the Conclusion. Remaining (optional): a λ=0 mains re-run would let λ=0 also appear
> on the train-vs-test *calibration* figure and extend the comparison to the PV–battery
> architecture; the reliability-level/tail win above already establishes the risk-specific
> contribution.

**What to do (and it's tractable):**
- Promote **λ=0 (risk-neutral SO)** into the main comparison for the calibration and risk
  figures. Present LP-Avg/LP-Worst as *deployed heuristics* and λ=0 as *the risk-neutral SO
  baseline*.
- **Check whether CVaR (λ=0.9) actually beats λ=0** on calibration and on the tail,
  out-of-sample. The tail/level comparison is available now from the sweep; the *calibration*
  comparison (train-vs-test for the λ=0 design) needs a one-line re-run of the mains at λ=0
  (a single Sherlock job).
- Frame the answer honestly whichever way it falls: if λ=0 is also well-calibrated, the
  contribution is "stochastic sizing is calibrated; the risk knob further controls the tail";
  if λ=0.9 clearly beats λ=0 on the tail, that is the risk-specific contribution, cleanly
  demonstrated.

*(I can pull the λ=0-vs-λ=0.9 tail/level comparison from the sweep now, and script the λ=0
mains re-run for calibration, on your go-ahead.)*

### 2. The stated contribution is mis-centered
*(novelty lens, severity 4)*

The Introduction (contribution paragraph) and Conclusion lead with **"planning method and
storage technology as coupled design decisions"** — but the paper's own honest analysis guts
that claim: the frontier result is conceded to be "largely an architecture-level effect," and
the sizing×PCM interaction holds in only 3/5 climates. Meanwhile the genuinely novel,
well-supported finding — **out-of-sample calibration** — is absent from the contribution
statement.

**Fix:** re-center the contribution on (a) the calibration finding (islanded-sizing heuristics
are miscalibrated on unseen years; a risk-aware design keeps its promise) and (b) risk
reduction at comparable cost. Demote "coupled decisions" to what the data show: PCM
independently shifts the frontier outward (an architecture effect). Rewrite the intro
contribution paragraph and the Conclusion opening accordingly.

---

## Major issues (consolidated across lenses)

| # | Issue | Lens(es) | Sev | Fix |
|---|---|---|---|---|
| 1 | Missing λ=0 risk-neutral SO baseline (above) | novelty | 4 | promote λ=0; test CVaR>λ=0 on calibration/tail |
| 2 | Contribution mis-centered on "coupled decisions" (above) | novelty | 4 | re-center on calibration + risk |
| 3 | **No formal significance test; the "N/30 cells" argument is invalid because the cells are dependent** (same 25 weather-years reused across 3 VoLL × 2 architectures). A naïve 30-cell sign count wildly overstates evidence. | methodology, narrative, language | 4 | Treat the **5 climates (or 5 folds)** as the independent units. Run a paired test respecting the structure (per-fold paired bootstrap of SO-CVaR−LP-Avg; or block bootstrap over weather-year blocks; or Wilcoxon across the 5 climates at fixed VoLL/arch). Stop implying 30/30 *is* the evidence. Replace "statistically indistinguishable" (Results 3.2) with "within fold-to-fold variability." |
| 4 | **Calibration metric is (a) undefined in Methods/SI, (b) symmetric.** |ratio−1| penalizes LP-Worst's *safe* over-provisioning (0.51) as harshly as LP-Avg's *dangerous* under-provisioning (1.11) — backwards for a mission-critical base — and is partly "by construction" (CVaR optimizes the tail it is then praised for matching). | methodology, narrative, results | 4 | Define the ratio precisely in Methods/SI, **method-consistently** (e.g. re-simulate each chosen design on its 20 training years). Separate the under-provision failure (reliability axis) from over-provision waste (cost axis) instead of collapsing to |ratio−1|. Neutralize "by construction" by showing calibration also holds on a metric CVaR does *not* directly optimize (event count or duration). |
| 5 | **Tail statistic is a max-of-5-fold-means, not a tail quantile** — noisy and dependent on the arbitrary contiguous-block layout; the −21%/−48% numbers are not reproducible under a different partition. | methodology, results | 4→3 | Compute a proper **per-year** tail (empirical CVaR₀.₉ or 90th-pct of realized annual unmet/cost) over the 25 held-out realizations, as the Conclusion's future-work already proposes; demote worst-fold to a cross-check. |
| 6 | **Florida drives the aggregates.** SO-CVaR under-provisions in the tropical/low-VoLL cells (ratio up to ~2.1), which dominate its mean |ratio−1| and the +17% pooled degradation. | results | 4→3 | Lead with the robust **median** (done), disclose the tropical exception (now done in Results 3.3 + the 30-cell figure shows it), and report median degradation alongside the mean. |
| 7 | **Delineation from Mühlbauer [7] (same group, same testbed) is thin**, and the nearest tail-risk sizing work Xie [24] is dismissed in a clause. | novelty | 3 | Add an explicit sentence: what is *reused* from [7] (physical pipeline, cost data, climates, PCM model) vs *new* (CVaR objective, out-of-sample CV, the calibration finding). Distinguish [24] on substance (CVaR vs DRO; OOS calibration test), not on "conservatism." |
| 8 | **FOB framing vs modeled load mismatch.** The load is a small building (10 kW HP, sub-kW plug loads, ≤4 occupants), not a "base"; results tie to one archetype at 5 sites. | novelty | 3 | Soften "base" to a small deployable/off-grid installation, and argue the transferable contribution is the *method* (calibration, risk-aware sizing), not the FOB per se. |
| 9 | **Capacity confound + the "unconfounded" claim.** The tail-unmet reduction is partly bought with more capacity (the paper says ~1% — it is ~1% *total* but ~5% *capital*; now corrected), and lower cost-variance could also be partly a capacity effect. | methodology, results | 3 | Add an **equal-capital control** (scale LP-Avg up ~5% capital, or SO-CVaR down) and report the residual tail/variance difference at matched capital as the clean risk-allocation effect. |
| 10 | **Stationarity untested.** Contiguous-block CV and equiprobable-scenario CVaR both assume the 25 years are exchangeable; a climate trend would bias both. | methodology | 3 | Report a Mann-Kendall/Sen-slope test on PV yield and heating/cooling per site; if a trend exists, detrend/reweight or bound its effect. |
| 11 | **Journal-structural gaps.** No Impact Statement (a Cambridge Prisms requirement); data/code availability buried; no nomenclature block despite heavy acronym load; methods named two ways (acronyms vs prose) across sections. | language | 4 | Add Impact Statement (~120 words) + end-matter declarations + nomenclature table; pick one method-naming convention and use it throughout. |

---

## Minor / editorial (fix in the revision pass)

- **Abstract** ~285 words vs ~200 target, with a ~70-word mega-sentence — split and trim; name LP-Avg as the cost minimizer and use "comparable cost" once.
- **"least risky" superlative** in the Conclusion → "substantially less risky than the average-year design" *(fixed)*.
- **RQ1 mild circularity** — "at comparable cost" is baked into the question then reported as a finding; reword so the premium is *evaluated*, not assumed.
- **VoLL/break-even decoupling** stated too absolutely — scope it (VoLL enters the break-even only via the renewable side's small residual penalty; non-trivial in Alaska) *(partly addressed in Results 3.5)*.
- **RQ labels** — the Introduction poses three unlabeled questions but the Conclusion answers "RQ1/2/3"; label them explicitly in the intro.
- **Discussion redundancy** — §4 re-quotes Results magnitudes; state each once by direction and lead on mechanism.
- **References** — out of citation order; [16] orphan; placeholders [38]/[41]/[48]/[51]; [42]/[43] "VERIFY"; confirm Cambridge Prisms style (author-date vs numbered) and convert. PCM $70/kWh needs a real cost citation (unlikely to be an ATB line item).
- **[38] "PHTM Paper" placeholder** underpins *all* thermal loads (the entire cost-reliability story) — this citation is load-bearing and must be filled.
- Consistency nits: "break-even"/"breakeven", "/gal"/"per gallon", US/UK spelling, "PV+battery"/"PVB"/"solar-battery", "$1-9/gal" vs the exact "$0.78-8.90/gal", define "cell" once (30 = 5×2×3; 10 = one VoLL slice), standardize climate labels (place vs Köppen).

---

## What is genuinely working (keep it)

- **The honesty is a competitive advantage, not a liability.** Owning "not cheaper," confining the trustworthy-reliability claim to the reliability axis (Results 3.3), and framing the cost premium as what a risk measure *should* produce ("cause for suspicion, not celebration") pre-empt the sharpest attacks. Do not sand this off.
- **The calibration framing is the best idea in the paper** — delivered-vs-planned reliability, heuristics as systematically under/over-provisioning. It goes beyond "CVaR reduces tail risk" and is the thing to build the contribution around (Issue 2).
- **Correct CVaR formulation** (R-U linearization, shared first-stage capacities, α=0.9 small-sample rationale) and a **defensible price-independent diesel break-even.**
- **Out-of-sample CV over 25 real weather years** directly targets the overfitting question the field usually ducks.

---

## Per-lens verdicts

| Lens | Verdict | Headline |
|---|---|---|
| Methodology & statistics | major | dependence/no-test; calibration-metric definition + asymmetry; tail proxy |
| Narrative, framing & honesty | minor | coherent & honest; fix "least risky", "statistically", symmetric-loss framing |
| Results vs claims & figures | minor | numbers check out; disclose Florida; fix "~1% capital"; 30-cell calibration figure |
| Novelty & positioning | major | **missing λ=0 baseline; re-center contribution**; delineate [7]/[24] |
| Language & structure | major | method-naming consistency; Impact Statement/nomenclature; abstract length; refs |

---

## My recommended order of operations

1. **Decide the λ=0 baseline (Issue 1).** Pull the λ=0-vs-λ=0.9 tail/level comparison from the sweep now; if the calibration comparison is wanted (it is), re-run the mains at λ=0. This determines the paper's actual contribution — do it *before* finalizing the narrative.
2. **Re-center the contribution** on calibration + risk (Issue 2); rewrite the intro contribution paragraph and Conclusion opening.
3. **Add the paired significance test** respecting dependence (Issue 3) and define + justify the calibration metric (Issue 4).
4. **Compute the per-year tail metric** (Issue 5) and the equal-capital control (Issue 9).
5. **Delineate [7]/[24], soften the FOB-scale framing, fill [38] and the reference gaps** (Issues 7, 8, 11).
6. Editorial pass: abstract length, naming, Impact Statement, RQ labels, references style.

Items 1, 3, 4, 5, 9 are the difference between "interesting but not yet convincing" and "accept." None require new physics; only item 1 (calibration half) needs a re-run, and it is a one-line parameter change.

---

# Reviewer 2 — second section-by-section pass (2026-07-21)

A fresh 7-lens pass over the *revised* paper. All seven sections came back "needs-work" (near-final, fixable). It caught several concrete bugs the first pass missed; the correctness/framing ones are now **fixed directly**, the rest are flagged for the writing pass.

## Fixed in this pass (applied to the section files)
- **Abstract: the diesel sentence was inverted** ("undercuts diesel *below* $9/gal" reads backwards) → now "beats diesel wherever delivered fuel *exceeds* roughly $9/gal." Added the λ=0 baseline clause, named LP-Avg as the cost minimizer, split the mega-sentence, softened "resolve"→"ease."
- **Results §3.3: "best-calibrated in 29/30" was wrong** — on the figure's ratio metric it is **26/30** (four tropical exceptions; verified from `fig_calibration.csv`). Fixed here and propagated to Discussion, `paper_narrative.md`, `figure_narratives.md`. Removed the false "ranking intact" clause, corrected the "lowest valuation" misattribution, and scoped "only the risk-aware design was calibrated" to the three *deployed* methods with a forward-reference to §3.4.
- **Discussion §4.1: LP-Avg mischaracterization** ("sized to mean historical conditions") → corrected to the true mechanism + heavy-tail explanation. **§4.2:** λ=0 claim scoped to "15 PCM cells / +1.2%"; "unconfounded" → "less confounded" with the equal-capital caveat. **§4.5:** added the capital-confound and stationarity limitations.
- **Conclusion:** opening re-centred on calibration (was "coupled design decisions"); stale "promote λ=0" future-work replaced (already done); "~1%" → "~1.2%".
- **Methods:** "heat pump adequate" (contradicted polar-unservability) → "nominal 10 kW"; added the a-priori λ/α justification; **defined the calibration metric and risk statistics in §2.4**; added λ=0 as a compared arm; clarified "operating cost"; "nested" → "common" cross-validation in Results §3.4.
- **Introduction:** re-centred the contribution on calibration + risk (λ=0 evidence); added λ=0 to the method list; fixed the para-6 miscitation ([28,7,8] → [28]).

## Deferred to the writing pass / you (need a value, a decision, or a computation)
- **SI:** mirror the calibration + risk-metric definitions into S9; supply PCM power rating / round-trip efficiency / standing loss in S5 (from the model); add exchangeability + empirical-distribution notes; add figure callouts and a risk-parameter SI section; fill refs [38]/[48]/[51].
- **Introduction:** label RQ1–RQ3 explicitly; reword RQ2 (architecture-shift vs sizing-interaction) and RQ1 (evaluate the premium); state what is reused from [7] vs new; distinguish [24] on substance; drop "reliability" from RQ3; convert to author-date + reorder; resolve orphan [16].
- **Results:** cite Fig 4 and Fig 5 in-text or confirm demotion; reconcile the capacities figure (cited SI, lives as main Fig 2); add the dependence caveat at the first "N/30"; soften "scaling with VoLL."
- **Discussion:** redundancy pass; climate-label consistency.
- **Paper-wide:** the paired significance test (Issue 3); Impact Statement + nomenclature + declarations; method-naming and UK-spelling consistency; abstract length trim to ~200 words.

The correctness of the *science* is settled; these remaining items are editorial + reproducibility polish, already routed to the writing agent by ASSEMBLY_INSTRUCTIONS.md.
