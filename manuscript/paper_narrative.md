# Paper narrative — the overall arc and the honest framing

Orientation doc for whoever writes the manuscript. Pairs with `figure_narratives.md`
(per-figure) and the three topic write-ups (`risk_parameters_writeup.md`, `voll_writeup.md`,
`diesel_cost_writeup.md`). Target journal: Cambridge Prisms (author–date refs).

---

## One-sentence thesis

For a resilience-critical off-grid Forward Operating Base microgrid (PV + battery + PCM
thermal storage), a **risk-averse CVaR stochastic sizing** method produces designs whose
out-of-sample reliability is **trustworthy** — the loss of load they deliver on unseen
weather years matches what they were planned for (median test/plan unmet ≈ 1.0) — whereas
the average-year heuristic systematically **under-provisions** and the worst-year heuristic
**over-provisions**; it also lowers the reliability **tail** and year-to-year **cost
variability**, all at **comparable total cost**, and the microgrid beats a diesel genset at
any realistic fuel price. (It is NOT cheaper on mean total cost — that is LP-Avg — there is
NO sharp cost–reliability "knee", and the high-VoLL cost win holds in only 1/30 cells; those
framings do not survive the data and must not be used.)

## What is genuinely novel (lead with these)

1. **Risk-averse (CVaR) capacity planning applied to the FOB resilience problem** — pure
   stochastic optimization for microgrids is well-trodden; the risk-aware framing under
   contested-theatre conditions is the contribution. (Do NOT reframe the method as pure SO.)
2. **Honest out-of-sample cross-validation** over 25 weather years (1998–2022), which most
   sizing papers skip — and which exposes that the common heuristics (average-year,
   worst-year LP) mis-generalise.
3. **PCM thermal storage as a battery substitute** in this application, quantified by climate.
4. **Diesel dominance framed as a price-independent break-even**, robust from commodity fuel
   upward — not a rigged comparison at an extreme price.

## The five honest framing rules (do not violate)

1. **Never say SO-CVaR is cheaper, and never claim a "knee" or a VoLL-crossover.** On mean
   total cost LP-Avg is marginally cheapest (SO-CVaR +0.2 to +1.5%; below LP-Worst). The
   frontier is ~linear (NO sharp knee — 0/30 cells), and SO-CVaR wins total cost at high VoLL
   in only 1/30 cells (California-PCM) — do NOT generalise it. SO-CVaR's real, robust value is
   **calibration** — delivered reliability matches planned (median test/plan unmet ratio 1.02
   vs LP-Avg 1.11 = under-provisions, LP-Worst 0.51 = over-provisions; best-calibrated in
   26/30 cells, 4 tropical exceptions) — and **risk reduction** (worst-fold unmet −21%, cost variance −31%, both
   30/30) at comparable total cost.
2. **Method gaps are within weather noise.** Report the *consistent direction across cells*
   (e.g. SO-CVaR < LP-Avg unmet in 10/10 cells) via a paired sign/rank test, and state that
   single-cell differences are within fold SD. This pre-empts the strongest reviewer attack.
3. **λ=0.9, α=0.9 are fixed a priori** (a risk preference for a mission-critical base), not
   tuned. On mean cost the optimum is λ=0; S3 is a *robustness* check showing conclusions
   don't hinge on the value (fixed costs +1.6% mean/+6.3% max over tuning) and that the
   premium buys −34% tail unmet. α=0.9 is the finest resolvable tail at N≈20 (2 tail years).
   See `risk_parameters_writeup.md`.
4. **Reliability = thermal.** Critical/electrical load is near-must-served (unmet ≈ 0
   everywhere; the high critical VoLL acts as a near-hard constraint). The entire
   cost–reliability tradeoff lives in HVAC. Attribute it correctly.
5. **VoLL and diesel price are decoupled.** VoLL = demand-side value of lost load; diesel
   price = supply-side benchmark cost. The renewable model has no diesel, so there is no
   "run-diesel-vs-shed" choice and the site is never left unpowered. Break-even is
   VoLL-independent. See `diesel_cost_writeup.md` §3b.

## Story arc (suggested section flow)

1. **Motivation.** FOB fuel logistics are deadly and expensive (FBCF $100–600/gal contested;
   ~1 casualty per 24 fuel convoys, AEPI 2009). Renewables + storage can cut fuel dependence,
   but sizing under weather uncertainty for a resilience-critical base is the open problem.
2. **Approach.** Mean-CVaR two-stage stochastic program vs two LP heuristics (average-year,
   worst-year), the two ends of the risk spectrum; 5 climates × 2 architectures × 3 VoLL;
   honest 5-fold CV over 1998–2022. → **Fig 1** (workflow), **Fig 2**→SI (what gets built).
3. **Cost: parity, not saving.** The three methods are within fold noise on mean total cost;
   SO-CVaR is a small premium over LP-Avg (+0.2–1.5%) and below LP-Worst. Establishes that
   CVaR's reliability advantages come at *comparable* cost. → **Fig 3** (frame as parity).
4. **Reliability you can trust — the LEAD result.** Out-of-sample calibration: SO-CVaR's
   delivered unmet matches its planned value (median test/plan 1.01); LP-Avg under-provisions
   (1.08), LP-Worst over-provisions (0.50). The one method that keeps its promise on unseen
   years. → **Fig (calibration)** — the new hero figure.
5. **Risk reduction + loss of load.** Loss of load is ~all thermal; SO-CVaR cuts the
   reliability tail (worst-fold unmet −21%, 30/30) and year-to-year cost variance (−31%,
   30/30). Honest caveat: partly bought with ~5% more capital (~1% total spend), so the
   unconfounded win is the cost-variance reduction. → **`fig_risk`**.
5b. **Risk term vs risk-neutral SO (λ=0) — the referee-critical baseline.** Within the same
   nested CV, CVaR (λ=0.9) beats the risk-neutral SO on OOS unmet (−40%, 15/15) and tail
   (−19%, 15/15) for +1.2% cost → the benefit is risk aversion, not just stochastic
   optimization. → **`fig_riskterm`** (PCM, 15 cells).
6. **Frontier (supporting).** On separate cost/reliability axes SO-CVaR is never dominated
   (30/30); the worst-year heuristic overfits and is dominated in 5/30; PCM-coupling puts the
   design on the frontier across climates (RQ2). NO "knee". → **S1 frontier**.
7. **So what: diesel.** Break-even $0.78–8.90/gal — 11–767× below contested fuel, below even
   routine forward delivery in every climate. → **Fig 6**.
8. **Robustness & honesty.** VoLL sensitivity (S2 — the advantage does NOT grow with VoLL on
   total cost; keep as a sensitivity only, not a claim), risk-parameter robustness (S3), input
   variability (S5). Limitations: fold noise > method gaps; polar HVAC structurally
   unservable; single-architecture risk sweep; tail rests on a worst-fold proxy.

## Headline numbers (out-of-sample, Med VoLL unless noted)

- **Calibration (LEAD):** median test/plan unmet ratio SO-CVaR **1.02**, LP-Avg 1.11
  (under-provisions), LP-Worst 0.51 (over-provisions); best-calibrated (min |ratio−1|) in 26/30
  cells (4 tropical exceptions); SO-CVaR ≤ LP-Avg in 28/30; mean |ratio−1| 0.14 vs 0.27 vs 0.45.
- **Risk:** worst-fold unmet −21% vs LP-Avg (30/30), across-fold cost variance −31% (30/30);
  tail cut scales with VoLL (Low ~12% → High ~48%). Caveat: partly from ~5% more capital / ~1% total spend.
- **Risk term vs risk-neutral SO (λ=0), the key baseline (PCM, 15 cells):** CVaR (λ=0.9) lowers
  OOS mean unmet in 15/15 (median −40%) and tail unmet in 15/15 (median −19%) for a median +1.2%
  cost — so the benefit is risk aversion, NOT just "SO beats heuristics." (fig_riskterm)
- Cost: SO-CVaR vs LP-Avg +0.2 to +1.5% (median +0.45%); vs LP-Worst −0.1 to −3.3% (below in
  all 10 cells at Med). Method gaps < across-fold SD in all cells (SD median ~5% of mean, up
  to ~13% in Alaska; method spread median ~1.6%) → statistically tied.
- Generalization: pooled train→test unmet degradation LP-Avg +17.2%, SO-CVaR +7.6%, LP-Worst
  −51.4%; cost degradation LP-Avg +1.84%, SO-CVaR −8.0%, LP-Worst −9.1%.
- Reliability (levels): SO-CVaR < LP-Avg thermal unmet 10/10 cells (9% AK → 68% FL). PCM cuts
  worst-event duration (AK 100.6→45.6 h) and, in warm climates, unmet energy (CA 121.7→78.6).
- Frontier: SO-CVaR Pareto-efficient 30/30; LP-Worst dominated 5/30; no sharp knee (0/30).
- PCM: displaces battery 38–86% (MN −86%), PV ≲4% (LP-Avg; MN 4.2%); lowers total cost 4–14%.
- Diesel break-even $0.78–8.90/gal.
- Risk params: fixed (0.9,0.9) costs +1.58% mean / +6.27% max vs tuning; vs λ=0 buys −34%
  tail unmet pooled (−45 to −53% in High-VoLL cells); cost-optimal λ ≤ 0.25 in all 15 cells.

## The λ question — how to handle in writing (settled)

Do **not** make λ a research thread. State it as a fixed risk preference with a one-line
justification where the CVaR objective is introduced (risk posture + α=0.9 sample-size
argument), and cite S3 as a robustness check. If a reviewer presses on the exact value, the
escape hatch (do not put in the main paper unless asked) is: on a *tail-cost* metric the
per-cell optima move interior (λ≈0.5–0.9) and λ=1 is worse — i.e. on the resilience-relevant
objective the risk-averse choice is genuinely preferred; a per-test-year re-run would show a
clean interior optimum. Scoped but shelved.

## Must-fix caption errors found in the audit (publication risk)

- **S1 frontier:** "PCM dominates PVB in every climate" is FALSE (PVB efficient in AK & FL).
- **Fig 4:** panel-2 label "the tail metric SO-CVaR targets" is wrong (SO-CVaR has the
  *longest* worst outage in CA-PCM). Relabel as a duration proxy; acknowledge the reversal.
- **Fig 5:** stale numbers — cost degradation +1.84% (not +2.7%), unmet +7.62% (not +7.4%).
- **Fig 6:** "method-independent break-even" claim not supported (only SO-CVaR plotted).
- **S2 VoLL:** "$100–180/yr cheaper" overstates (−$51/−$142); "unmet ≤ LP methods" wrong.
- **S3:** "flat basin ~1%" overstates; star sits on a monotone slope at ~1.4%.
