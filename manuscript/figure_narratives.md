# Figure narratives — what each figure says, where it belongs, what to write

Companion to `paper_narrative.md`. For each figure: recommended placement, the one-line
message, the non-obvious insights (with numbers mined from the plotted CSVs), a draft
narrative paragraph a writer can adapt, and the problems that must be fixed before
submission. Numbers are out-of-sample (test split), Med VoLL unless noted.

**Golden rules for all figure text** (see `paper_narrative.md`):
1. Never claim SO-CVaR is *cheaper*. On mean cost LP-Avg is marginally cheapest; SO-CVaR's
   value is reliability at the *knee* (near-LP-Worst reliability at near-LP-Avg cost).
2. Method-to-method gaps are usually **smaller than fold-to-fold weather noise** — the
   credible signal is the *consistent direction across cells*, not any single bar. Say so.
3. λ=0.9/α=0.9 are **fixed a priori** (a risk preference), not tuned; S3 is a robustness
   check, not an optimality claim.
4. Loss of load is **almost entirely thermal/HVAC**; critical/electrical is near-must-served
   (unmet ≈ 0). Attribute the reliability tradeoff to HVAC.
5. VoLL (demand-side value) and diesel price (supply-side benchmark) are **decoupled**.

---

## Recommended MAIN vs SI lineup

| Figure | Now | Recommend | Why |
|---|---|---|---|
| Fig 1 workflow | main | **main** (trim) | reader's map; drop the α/λ sweep box from step 5 |
| Fig 2 capacities | main | **SI** (lift 2 sentences to main) | low ink-efficiency; obvious dominant message |
| Fig 3 total cost | main | **main** (reframe as *parity*) | the cost half of the thesis |
| Fig 4 loss of load | main | **main** (restructure) | reliability = the whole value proposition |
| Fig 5 generalization | main | **main** | the out-of-sample backbone; pre-empts "overfit?" |
| Fig 6 diesel breakeven | main | **main** (panel b; rework a) | the "so what" figure |
| S1 frontier | SI | **SI** (fix caption) | tradeoff geometry; one nugget → main prose |
| S2 VoLL sweep | SI | **SI** (fix numbers) | supporting; 2 nuggets → main |
| S3 risk params | SI | **SI** (add reliability payoff) | defensive robustness/rebuttal |
| S5 variability | SI | **SI** | input characterization |

Net main text: **Figs 1, 3, 4, 5, 6** (five), with Fig 2's two real insights folded in as
prose. This is a tighter spine than six data-heavy main figures.

---

## Cross-cutting problems to fix everywhere

- **Fold noise > method gaps.** In nearly every figure the ±1 SD across 5 folds exceeds the
  between-method differences (e.g. Alaska cost SD ~$2,200–2,370/yr vs a 3-method spread of
  ~$735; Florida unmet 65.2 ± 66.3 kWh/yr). Fix: (a) state explicitly that method gaps are
  within fold noise; (b) report a **paired sign/rank test across the 10–15 cells** to turn
  "10/10 same direction" into a defensible significance statement; (c) prefer paired
  per-fold differences (tight CI) over independent level SD bars.
- **Per-climate y-axes hide the Alaska magnitude story** (polar HVAC is ~40–170× the warm
  climates and structurally unservable). Add a normalized/relative panel where it matters.
- **Electrical/critical ≈ 0 clutters Figs 3 & 4.** Max electrical unmet 0.72 kWh/yr, worst
  event 1.16 h (one CA-PCM cell). Drop the sub-bars to an SI footnote and relabel the
  penalty as *thermal/HVAC*.
- **Only Med VoLL shown in main figures.** State how gaps scale with VoLL (or show High).
- **PCM has two dissociable effects** — energy reduction (warm climates) vs worst-event
  duration buffering (cold). Don't sell "PCM improves reliability" as one thing.
- **Underused strong result:** SO-CVaR/PCM *dominates* LP-Worst out-of-sample on **both**
  cost and reliability in Florida (17.2 kWh/yr, $2141 vs 19.4 kWh/yr, $2148) and on energy
  in Arizona — direct evidence worst-case fitting overfits. Lift into main prose.

---

## Fig 1 — analysis workflow schematic  (MAIN, trim)

**Message.** The reader's map: NSRDB weather → three sizing methods → common 5-fold CV →
cost + three reliability metrics → sensitivities + decoupled diesel benchmark.

**Insights.** (i) 20 train + 5 test years × 5 folds exactly consumes 1998–2022, each year
held out once. (ii) All three methods run through one identical pipeline → apples-to-apples.
(iii) One cost metric vs **three** reliability metrics signals the reliability thesis up
front. (iv) SO-CVaR is visually flagged as hero while LP-Avg/LP-Worst are equal-status
baselines (not strawmen).

**What to fix (decided):**
- **Drop the α/λ sweep from step 5.** We claim λ,α are fixed a priori; showing a "sweep to
  choose parameters" box contradicts that and re-raises "why 0.9?". Relabel step 5 to the
  evaluation + sensitivities (VoLL levels) and point to S3 only as a *robustness check*.
- If risk params are shown at all, show **both** λ=0.9 and α=0.9 with "(fixed a priori)".
- Break the diesel benchmark into a **separate lane** ("external benchmark, price-independent
  break-even") — not nested under VoLL sensitivities — to reflect the decoupling.
- Add a compact **design-grid callout** (5 climates × {PVB, PCM} × 3 VoLL = 30 configs); the
  scope is currently invisible.
- Clarify Fig 1a (architecture schematic) vs 1b (this workflow) so Fig 1 reads as one unit.

---

## Fig 2 — sized capacities  (→ SI; lift 2 sentences to main)

**Message.** PCM is a **battery** substitute, not a PV substitute, and the displacement is
strongly climate-dependent; SO-CVaR's storage sizing tracks cheap LP-Avg (not LP-Worst's
over-build) in the cost-dominant cold climates.

**Non-obvious insights (the only reasons to keep it):**
- **PCM displaces battery, not PV.** PV changes ≤4% adding PCM (even +3.3% in CA, because
  PCM charging draws power), while battery collapses 38–86%: MN 64.9→9.2 kWh (−86%),
  FL −77%, CA −63%, AK −59%, AZ −38%. → corrects the "obvious PCM reduces PV/battery" worry.
- **Risk spectrum is physical:** LP-Worst builds biggest, LP-Avg smallest, SO-CVaR between.
  AK PVB battery: LP-Avg 72.4 → SO-CVaR 73.2 (+1.1%) → LP-Worst 95.5 (+32%). SO-CVaR gets
  near-LP-Avg storage cost while avoiding LP-Worst's over-build where it's most expensive.
- **LP-Worst is the least fold-stable** (largest ±SD on nearly every component: AK hot-PCM
  SD 18.4 vs SO-CVaR 9.9 vs LP-Avg 2.4 kWh_th) — a stability argument *against* the robust
  heuristic. Worth keeping.
- Optimizer recovers the correct thermal regime endogenously (hot-PCM only in AK/MN,
  cold-PCM only in FL, both in AZ) — a physical sanity check.

**Why SI.** 6 of 20 panels are empty/≈0; the dominant visual message is the obvious one; the
real insights require arithmetic off the bars. Move the full grid to SI as the "what got
built" reference; fold the PCM-displaces-battery and risk-spectrum points into two main-text
sentences citing the SI figure.

**If a compact main panel is wanted:** (a) battery displacement % by climate, and (b)
SO-CVaR's fractional position between LP-Avg and LP-Worst per component.

---

## Fig 3 — out-of-sample total cost  (MAIN, reframe as *parity*)

**Message.** On mean total cost the three methods are effectively tied — SO-CVaR ≤ +1.5% vs
LP-Avg and below LP-Worst everywhere — so the figure establishes that reliability is
**nearly free**, not that SO-CVaR is cheaper.

**Insights.**
- SO-CVaR vs LP-Avg +0.21% (MN PVB) to +1.49% (FL PVB), median ~+0.45%; vs LP-Worst negative
  in all 10 cells (to −3.31%, AK PVB). LP-Avg lowest in every cell.
- **Gaps are within fold noise:** AK across-fold SD $2,222–2,370/yr (~13% of mean) vs a
  3-method spread of $735. Methods are statistically indistinguishable on total cost.
- Mechanism: LP-Worst buys robustness with **capital**, not lower total cost (AK capital
  $9.6k→$13.1k; penalty $8.6k→$5.9k). SO-CVaR picks an intermediate capital point.
- Penalty share climbs cold-ward: FL 7.9% → AK 47% (LP-Avg); even robust AK is 31% penalty →
  the polar case is the binding stress case.
- Critical/electrical penalty is a phantom component (0 in all PVB cells, <$75/yr worst).
- PCM lowers total cost everywhere, most in MN (−14%) and FL (−11%), least in AK (−4%).

**Must-fix problems:**
- Caption must **state the method gaps are within fold SD** (currently invites over-reading
  sub-1% bar orderings). Consider a paired SO-CVaR−LP-Avg per-fold difference panel.
- Plotted penalty is the **expected (mean)** penalty, not tail — a reviewer will ask why a
  CVaR method shows mean penalty. State the division of labor with Fig 4 explicitly.
- Drop/footnote the critical penalty; relabel the light shade as *expected HVAC penalty*.
- 7× different per-climate y-axes hide the 8%→47% penalty-share gradient — add a normalized
  view or annotate %Δ vs LP-Avg/LP-Worst on the bars.

---

## Fig 4 — out-of-sample loss of load  (MAIN, restructure)

**Message.** Loss of load is essentially all thermal; SO-CVaR sits at the reliability knee
(closes 29–89% of the LP-Avg→LP-Worst thermal-energy gap) — but the worst-event panel
contains an honest reversal that the current label overclaims.

**Insights.**
- SO-CVaR < LP-Avg thermal unmet in **all 10 cells** (9.2% AK → 63.6–68.4% FL); even beats
  LP-Worst out-of-sample in **AZ-PVB** (74.6 vs 76.3) and **FL-PCM** (17.2 vs 19.4) — but NOT
  AZ-PCM (60.2 vs 55.8), so always specify the architecture → worst-year fitting overfits,
  stochastic generalizes.
- **Polar is structurally unservable:** AK thermal unmet ~2.6 MWh/yr (all methods), ~150×
  the warm climates; SO-CVaR dents it only ~9% → method choice is second-order there.
- **PCM's two effects dissociate:** AK LP-Avg PVB→PCM worst event 100.6→45.6 h (−55%) but
  energy +0.5%; CA PVB→PCM energy 121.7→78.6 (−35%) *and* worst event 46.3→6.6 h.
- Electrical/critical is a non-story (max 0.72 kWh/yr, 1.16 h — one CA-PCM cell).

**Must-fix problems (important):**
- **The panel-2 label "the tail metric SO-CVaR targets" is inaccurate.** In Marine/CA + PCM
  the worst single outage is **longest** for SO-CVaR (18.6 h) and **shortest** for LP-Avg
  (6.6 h) — the risk-averse method loses on the very metric labeled. The objective minimises
  CVaR of *energy-priced penalty*, not event duration. Relabel as a duration **proxy**,
  acknowledge the CA-PCM reversal, and ideally also plot the actual CVaR-of-penalty.
- Caption currently cherry-picks the monotone SO-CVaR<LP-Avg result and omits the LP-Worst
  comparison and the reversal — make it balanced.
- Drop electrical sub-bars to an SI note; use the width for thermal box/violin over the 25
  test-year realizations (mean±SD understates a max-statistic's noise).
- Report a paired test across the 10 cells (10/10 direction) for a defensible claim.
- Expose the Alaska magnitude (normalized/log panel) so "polar HVAC is structurally unmet"
  is visible.

---

## Fig 5 — generalization (train → test)  (MAIN)

**Message.** SO-CVaR's reliability is the one that holds up out-of-sample: its train→test
unmet gap is near zero (+0.4 to +2.0% in 8/10 cells) while risk-neutral LP-Avg is
optimistic in-sample and disappoints out-of-sample (+17.2% pooled) — all while test costs
converge to within ~1%.

**Insights.**
- **LP-Avg overfits reliability, not cost:** unmet rises out-of-sample in all 10 cells
  (pooled +17.2%); SO-CVaR nearly unbiased (+7.6%); LP-Worst wildly conservative (−51.4%).
  This is the direct answer to "did you overfit 25 weather years?"
- Cost gap direction confirms the knee: LP-Avg overruns test cost everywhere (+1.84%
  pooled), SO-CVaR/LP-Worst come in under budget — yet all converge to ~1%.
- **Honest nuance:** LP-Worst has the *lowest* test unmet in every climate. The pitch is
  "near-worst reliability at near-average cost" (knee), **not** "best reliability." The
  displayed small-gap metric rewards stability, which can be misread as best-level.
- Warm climates capture 70–105% of LP-Worst's unmet reduction; cold only ~30%.

**Must-fix problems (important — factual):**
- **Caption numbers are stale:** LP-Avg pooled cost degradation is **+1.84%**, not the stated
  +2.7%; SO-CVaR unmet is **+7.62%**, not +7.4%. Regenerate the caption from the printed
  pooled stats so it can't drift. (A reviewer recomputing from the table will catch this.)
- Level-SD error bars are huge vs the featured *gap*; use the **paired per-fold gap** and its
  own CI (train/test are paired within a fold → much tighter).
- Equal-weight % pooling is dominated by tiny-denominator FL cells (14.6→20.6 kWh = +41%);
  report absolute kWh too and caveat.
- Annotate SO-CVaR's fractional position on the LP-Avg→LP-Worst span ("72% of the reliability
  gain at ~1% of the cost gap") so it doesn't imply SO-CVaR is most reliable.
- State the near-zero SO-CVaR gap is currently an invisible dumbbell (marker on the training
  tick) — add a zero-gap reference.

---

## Fig 6 — diesel break-even  (MAIN; keep panel b, rework a)

**Message.** The renewable microgrid breaks even against an all-diesel genset at only
**$0.78–8.90/gal** across every climate/architecture — it beats diesel at essentially any
fuel price, and the conclusion doesn't even need the contested $100–600/gal band.

**Insights.**
- 4 of 5 climates break even **below ordinary commercial diesel (~$3–5/gal)**: FL 1.07
  (PCM 0.78), AZ 1.40, CA 1.69, MN 3.78; only AK (8.90) approaches a real pump price — still
  11× below the $100 floor. **No convoy premium needed to justify renewables.**
- Order-of-magnitude climate gradient is driven by **renewable capital** (weak polar solar),
  not diesel: AK renewable $18.4k/yr vs FL $2.5k/yr.
- PCM lowers break-even 5–27% (amplified vs its 4–12% total-cost effect, because the saving
  acts on the fuel-only numerator).
- Margin swamps uncertainty: AK worst fold 9.9/gal, still 10× under the band; 11× (100/8.9)
  to 767× (600/0.78) cheaper across the band.

**Must-fix problems:**
- **Caption asserts "method-independent break-even (1–3%)" but the figure plots only
  SO-CVaR** — it can't support its own claim. Either add faint LP-Avg/LP-Worst ticks, or move
  the 1–3% number to the text with the actual computed spread.
- **Foreground the robust point:** most climates beat diesel at ordinary ~$3–5/gal. Anchoring
  only to $100–600 concedes the debatable premise. Add a lower "routine delivery $13–45/gal"
  band (and ~$3–5 commercial line) so the win is visible below even peacetime fuel. *(This is
  the fig6 band improvement offered separately.)*
- Panel (a) is near-redundant with (b) (CA only, two lines nearly coincident, crossings
  crammed left). Replace with something additive (cost multiplier across the band, or gallons
  avoided), or drop it and widen (b).
- State the **genset spec** (efficiency, O&M, replacement, N+1) — break-even is sensitive to
  it in low-break-even climates where diesel fixed cost is a big share of the numerator.
- Lower panel (b) y-floor below 0.78 so the sub-$1 FL result renders honestly.

---

## S1 — cost-vs-unmet frontier  (SI; fix caption)

**Message.** SO-CVaR/PCM is on the Pareto frontier in all five climates; in Florida it
**dominates** LP-Worst on both axes.

**Insights.**
- SO-CVaR is Pareto-efficient in all 5 climates **only with PCM**; the PVB-SO-CVaR variant
  is dominated everywhere (CA: PVB 89.5 kWh/$2238 dominated by PCM 45.7/$2225).
- **FL: SO-CVaR/PCM strictly dominates LP-Worst/PCM** (17.2 kWh/$2141 vs 19.4/$2148) — lift
  to main prose as overfitting evidence. Also cuts unmet 64% below LP-Avg for +6% cost.
- Knee is real but **climate-dependent** — sharp in FL/AZ, weak in CA (pays 75% of cost
  premium for 68% of reliability gain, below linear). Don't over-generalize "knee."

**Must-fix (factual):**
- **Caption claim "PCM points dominate their PV+battery counterparts in every climate" is
  FALSE** — the figure's own Pareto flags show PVB-LP-Avg and PVB-LP-Worst are efficient in
  AK, and PVB-LP-Worst is the single most-reliable point in FL. Rewrite: PCM sweeps the
  frontier in MN/CA/AZ; PVB stays efficient in AK (2 pts) and FL (1 pt); SO-CVaR/PCM is on
  the frontier in all five and dominates LP-Worst/PCM in FL.
- Label the x-axis as effectively thermal (electrical ≈ 0).
- Flag which Pareto distinctions exceed ±1 fold SD (the FL dominance is ~2 kWh vs ~18 kWh SD).

---

## S2 — VoLL sensitivity (California)  (SI; fix numbers)

**Message.** As VoLL rises, PCM lets SO-CVaR flip from a small premium to the *cheapest*
option while nearly matching LP-Worst reliability — but it's architecture-specific and
mostly within fold noise.

**Insights.**
- **PCM is the enabler:** with PCM, SO-CVaR at High is cheapest (−$50.9/yr vs LP-Avg,
  −$142.0 vs LP-Worst); **without PCM it stays a premium** (+$102.8/yr vs LP-Avg at High).
- Under-sold nugget — **cost stability:** LP-Avg fold SD balloons 95→157→280 $/yr with VoLL;
  SO-CVaR stays 79→67→74 (~3.8× tighter at High). A variance-reduction argument distinct
  from tail-unmet control.
- SO-CVaR tracks LP-Worst reliability (closes 76% of the LP-Avg→LP-Worst gap) at below-LP-Avg
  cost, but its unmet is **above** LP-Worst (below LP-Avg only).

**Must-fix (factual):**
- Caption "$100–180/yr cheaper at High" **overstates** — it's −$50.9 vs LP-Avg / −$142 vs
  LP-Worst. And "unmet stays at or below the LP methods" is wrong (SO-CVaR unmet exceeds
  LP-Worst at every VoLL). Fix both.
- "Cost advantage grows with VoLL" is non-monotone vs LP-Avg (premium at Low/Med, flips only
  at High) — state per-comparator.
- Single climate — scope the claim to CA or add small multiples.

---

## S3 — risk-parameter robustness  (SI; add reliability payoff)  [reframed]

**Message.** Fixing (λ,α)=(0.9,0.9) a priori costs only a small, bounded mean-cost premium
(mean +1.58%, max +6.27% at AK-High) and no cell's cost-optimum is near it, so conclusions
don't hinge on the value. *(Already reframed: caption/suptitle/docstring now say robustness,
not optimality.)*

**Insights.**
- Fixed-minus-tuned test-cost gap positive in all 15 cells; mean +1.58%, concentrated in
  **High-VoLL** cells (mean +3.16% vs +0.7–0.9% Low/Med) — mechanistically sensible.
- Cost-optimal λ never exceeds 0.25 (11/15 pick λ=0; 0/15 pick 0.9) → (0.9,0.9) is a
  declared risk preference far from the cost frontier, not a disguised optimizer.
- **The payoff lives in unplotted CSV columns:** vs the cost-optimal params, the fixed
  setting cuts tail unmet ~40–53% in every High-VoLL cell (AK-High 524→316, −40%; FL-Low
  143→68, −53%). This *is* the knee thesis inside the sensitivity figure.
- Per-fold selection is unstable (modal pick backed by only 1–2 of 5 folds in 5/15 cells) →
  itself an argument for fixing a priori rather than tuning.

**Must-fix problems:**
- **Panel (b) shows only the cost penalty** — in isolation it makes the hero look strictly
  worse. Add the ~40–53% tail-unmet reduction (fixed vs selected) as a twin/overlaid series
  so cost and its reliability return read together.
- Caption "flat basin ~1%" overstates: λ=0 column is α-invariant (so the "optimum" is a
  column, not a point) and the star sits on a monotone λ-slope at ~1.4%. Correct this;
  annotate the star's actual regret.
- State which architecture the sweep covers (n=15 = 5×3, no PVB/PCM dimension).
- *(Optional, only if a reviewer presses on the exact value: on a tail-cost metric the
  per-cell optima move interior, λ≈0.5–0.9, and λ=1 is worse — see risk write-up footnote.)*

---

## S5 — interannual variability  (SI)

**Message.** Weather variability is real but modest (7–13% CV) for each site's dominant load
and solar; critically, solar is least abundant exactly where it's least predictable (polar)
— the raw justification for multi-year stochastic sizing.

**Insights.**
- **Solar reliability inversely tracks abundance:** Arid 1978 kWh/kW-yr at CV 1.5% vs Polar
  808 at CV 11.6% (7.8× more variable, 2.45× less yield). The base most reliant on storage
  faces the least predictable supply.
- The eye-catching high CVs sit on immaterial loads (Tropical heating CV 75% but ~1% of its
  cooling); sizing-relevant CVs are 5–13%.
- The ~2-of-25 tail years α=0.9 targets bite on both sides: polar heating +22%, polar PV
  −21% in the extreme year → motivates CVaR.
- Cross-climate magnitude (heating 178× polar vs tropical) dwarfs within-site spread →
  climate is first-order; per-climate sizing warranted.

**Improvements:** annotate CVs on the boxes (the key contrast is invisible on the plot); add
a per-year PV-yield-vs-heating scatter for cold climates with the CVaR tail year highlighted
(the co-occurrence of cold+dark is the load-bearing, currently unshown, question); add
electrical-demand variability and a no-trend (Mann-Kendall) check to justify fold
exchangeability.

---

# UPDATED LINEUP (2026-07-20) — calibration-lead narrative (SUPERSEDES the table at the top)

After stress-testing every candidate result against all 30 cells, the paper's spine flips
from a *cost* claim (which the data do not support — SO-CVaR is NOT cheaper than LP-Avg, there
is NO knee, and the high-VoLL cost win holds in only 1/30 cells) to **trustworthy reliability
at comparable cost.** Two new main figures were built; two old ones demote.

| Figure | Recommend | Role in the new story |
|---|---|---|
| Fig 1 workflow | **main** (trim) | reader's map; drop the α/λ sweep from step 5 |
| Fig 3 total cost | **main** | **cost parity** — CVaR at *comparable* cost, not cheaper |
| **`fig_calibration` (NEW)** | **main — HERO** | **lead: SO-CVaR delivers its planned reliability out-of-sample (median test/plan 1.02); LP-Avg under-provisions (1.11), LP-Worst over-provisions (0.51); best-calibrated 26/30 (4 tropical exceptions); 30 cells** |
| **`fig_risk` (NEW)** | **main** | supporting: SO-CVaR cuts the reliability tail (worst-fold unmet −21%, 30/30) and cost variance (−31%, 30/30); tail cut grows with VoLL |
| **`fig_riskterm` (NEW)** | **main** | **referee-critical: CVaR (λ=0.9) beats the risk-neutral SO (λ=0) on OOS unmet (−40%, 15/15) and tail (−19%, 15/15) for +1.2% cost — the risk term earns its keep vs a proper stochastic baseline, not just vs heuristics (PCM, 15 cells)** |
| Fig 6 diesel breakeven | **main** | the "so what" — break-even $0.78–8.90/gal |
| Fig 2 capacities | **SI** | fold 2 sentences into main (PCM displaces battery; risk spectrum) |
| Fig 4 loss of load | **SI** (or main) | thermal-only + worst-event; fold into calibration/risk text |
| Fig 5 generalization (dumbbell) | **SI** | **superseded** by `fig_calibration` (same train→test unmet data, sharper) |
| S1 frontier | **SI** (or main) | SO-CVaR never dominated 30/30; LP-Worst overfits (dominated 5/30); **no knee**; PCM-coupling = RQ2 (partly an architecture effect — caveat) |
| S2 VoLL · S3 risk params · S5 variability | **SI** | sensitivities / robustness |

**Recommended tight main spine (5):** Fig 1 (workflow) · Fig 3 (cost parity) · **`fig_calibration` (hero)** · `fig_risk` · Fig 6 (diesel). S1 frontier an optional 6th.

## `fig_calibration` — HERO (out-of-sample reliability calibration)
- **Message.** SO-CVaR is the only method whose *delivered* reliability matches what it was
  *planned* for on unseen years. Train-vs-test unmet scatter vs the y=x line + a test/plan
  ratio panel.
- **Numbers (verified, 30 cells).** Median test/plan unmet ratio **SO-CVaR 1.02, LP-Avg 1.11,
  LP-Worst 0.51**; best-calibrated (min |ratio−1|) in **26/30** cells (4 tropical exceptions);
  SO-CVaR ≤ LP-Avg in 28/30; mean |ratio−1| 0.14 vs 0.27 vs 0.45. LP-Avg under-provisions
  (test>plan) in 30/30; LP-Worst over-provisions (test<plan) in 30/30.
- **What to write.** This is the answer to "did you just overfit 25 weather years?" — and the
  common heuristics *do*: the average-year design promises reliability it doesn't deliver, the
  worst-year design wastes capacity. Only the risk-aware design is calibrated.
- **Caveats.** (i) Reliability is ~all thermal — label the axis honestly. (ii) COST calibration
  reverses (LP-Avg best), so keep this figure on the *reliability* axis. (iii) Partly "by
  construction" (CVaR targets the tail) — frame as an out-of-sample *validation* of design intent.

## `fig_risk` — supporting (reliability tail + cost stability)
- **Message.** Relative to LP-Avg, SO-CVaR cuts both the reliability tail and year-to-year cost
  variance — every one of 30 cells below the y=x line in both panels.
- **Numbers (verified).** Worst-fold unmet −21% median (30/30; scales Low 12% → Med 20% →
  High 48%); across-fold cost SD −31% median (30/30).
- **Caveats.** Worst-of-5-folds is a coarse tail proxy (max of 5-yr means). The tail cut is
  **partly bought with ~1% more capital** (frontier movement); the *unconfounded* win is the
  **cost-variance** reduction (panel b), not paid for by extra mean spend.
