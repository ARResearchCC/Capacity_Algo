# How to assemble the paper with your other Claude agent

This is a step-by-step guide for driving a second Claude agent to turn the material in
`manuscript/` into a finished Cambridge Prisms paper, section by section, that you paste
together. Every section already has a solid draft here; the agent's job is to **finalize**
each one — polish the prose, apply the referee fixes, and keep it honest — not invent claims.

---

## 0. What each file is (the map)

**Framing & rules (read for every section):**
- `paper_narrative.md` — the thesis, the five honesty rules, the story arc, and every verified
  headline number in one place. **This is the source of truth for framing and numbers.**

**Section drafts (the agent finalizes these):**
- `abstract.md`, `introduction.md`, `methods_and_data.md`, `SI.md`, `results.md`,
  `discussion.md`, `conclusion.md`

**Figures (numbers + what each figure says):**
- `figure_narratives.md` — per-figure message, exact numbers, and the recommended main/SI
  lineup (see the "UPDATED LINEUP" section at the bottom — that is the current plan).

**Accuracy & fixes (the checklists the agent must satisfy):**
- `factcheck_report.md` — every claim verified against code/results; the verified numbers.
- `reviewer_report.md` — the referee findings and the prioritized fixes to apply.
- (Reviewer-2 section notes are folded into `reviewer_report.md` under "Reviewer 2".)

**Citations & justification (for wording and references):**
- `references.md` (the bibliography; note the placeholders flagged in the reports),
  `risk_parameters_writeup.md`, `voll_writeup.md`, `diesel_cost_writeup.md`.

**Figures themselves:** `paper_figures/main/*.png` and `paper_figures/si/*.png`, each with a
`.txt` caption and `.csv` of the plotted data.

---

## 1. Paste this GLOBAL BLOCK at the top of EVERY section prompt

> You are finalizing one section of a Cambridge Prisms journal paper, "Risk-Aware Capacity
> Planning and Phase Change Material Storage for Fully Renewable Islanded Microgrids." Produce
> publication-ready prose. Obey these NON-NEGOTIABLE rules (they were hard-won against the data):
>
> **Honest framing (never violate):**
> 1. Never say the proposed method (SO-CVaR) is *cheaper*. On mean total cost the average-year
>    design (LP-Avg) is marginally cheapest; SO-CVaR is +0.2–1.5% and below the worst-year design
>    (LP-Worst). Always "at comparable cost," never "cost saving."
> 2. Never claim a cost–reliability "knee" (the frontier is ~linear) or that the advantage "grows
>    with VoLL" on total cost (that holds in only 1/30 cells — do not use it).
> 3. Method-to-method gaps are within fold-to-fold weather noise; report the *consistent direction
>    across cells*, and do NOT write "statistically" anything unless a paired test is reported.
> 4. Loss of load is essentially all thermal (critical/electrical is near-must-served, ≈0).
> 5. λ=0.9, α=0.9 are fixed a priori (a risk preference), not tuned.
>
> **The contribution to lead with:** (a) out-of-sample *calibration* — SO-CVaR delivers the
> reliability it was planned for, while LP-Avg under-provisions and LP-Worst over-provisions; and
> (b) the *risk term* beats the risk-neutral stochastic program (λ=0), not just the deterministic
> heuristics. NOT "coupled planning×storage decisions" (that is largely an architecture effect).
>
> **Verified numbers (use these; do not invent):** calibration median test/plan unmet ratio
> SO-CVaR 1.02 / LP-Avg 1.11 / LP-Worst 0.51 (best-calibrated 29/30 cells); risk-term vs λ=0:
> −40% mean unmet & −19% tail, 15/15 cells, +1.2% cost (PCM); risk vs LP-Avg: worst-fold unmet
> −21% & cost variance −31%, 30/30; cost SO-CVaR +0.2–1.5% vs LP-Avg; generalization pooled
> train→test unmet +17.2% (LP-Avg) / +7.6% (SO-CVaR) / −51.4% (LP-Worst); PCM battery −38–86%,
> cost −4–14%; diesel break-even $0.78–8.90/gal.
>
> **Style:** past tense for Methods/Results; UK spelling (Cambridge Prisms is a UK publisher —
> use -ise/-our/"behaviour"); one method-naming convention throughout (define SO-CVaR / LP-Avg /
> LP-Worst at first use and keep the acronyms); refer to figures as "Figure N" (final numbering
> below); keep the acronym list consistent (VoLL, PCM, PV, CVaR, VaR, CRF).
>
> Write only the section requested. Do not add claims beyond the source files. Please output your
> writing/update of the current section in a google doc file, and keep adding to the same google doc
> as you are working on more sections (as more steps/prompts being sent).

---

## 2. Per-section prompts (run in this order)

For each step, attach the listed files and paste the GLOBAL BLOCK, then the section prompt.

### Step A — Methods and Data (do first; it fixes terminology everyone else inherits)
Attach: `methods_and_data.md`, `SI.md`, `paper_narrative.md`, `factcheck_report.md`.
Prompt: *"Finalize the Methods and Data section from `methods_and_data.md`. Add a precise, one-
sentence definition of the calibration metric (test-year ÷ planning-year unmet load, where
planning-year unmet is the chosen design re-simulated on its training years) and state that
λ=0 is the risk-neutral stochastic baseline. Keep detailed formulations in the SI. Ensure the
VoLL levels (critical 30/100/300, HVAC 1/3/10 \$/kWh) and diesel prices (13/45/400 \$/gal)
match the SI. Output final prose + a short list of any SI edits needed."*

### Step B — Results
Attach: `results.md`, `figure_narratives.md`, `paper_narrative.md`, `reviewer_report.md`.
Prompt: *"Finalize the Results section from `results.md`. Keep the calibration lead and the
λ=0 risk-term comparison (fig_riskterm). Apply the referee fixes in `reviewer_report.md`:
disclose the tropical cells drive the calibration aggregates, replace any 'statistically'
wording with 'within fold-to-fold variability,' report absolute-kWh alongside percentages for
the pooled generalization numbers, and add one sentence noting that a paired sign/rank test
across the five climates (not the 30 dependent cells) is used to support the direction claims.
Output final prose."*

### Step C — Discussion (incl. Limitations)
Attach: `discussion.md`, `results.md`, `paper_narrative.md`, `reviewer_report.md`.
Prompt: *"Finalize the Discussion from `discussion.md`. Interpret, do not restate Results
magnitudes. Keep the Limitations subsection complete and honest (dependence/no formal test yet;
thermal-only; sweep PCM-only; worst-fold tail proxy; ~5%-capital confound; single load
archetype). Ensure the PCM 'coupled decisions' point carries the architecture-effect caveat.
Output final prose."*

### Step D — Conclusion
Attach: `conclusion.md`, `paper_narrative.md`.
Prompt: *"Finalize the Conclusion from `conclusion.md`. Answer RQ1–RQ3 honestly and precisely,
centre the contribution on calibration + risk-at-comparable-cost (not 'coupled decisions'), and
keep the future-work list (λ=0 calibration re-run, per-year tail metric, PVB risk sweep).
Output final prose."*

### Step E — Introduction
Attach: `introduction.md`, `paper_narrative.md`, `references.md`, `reviewer_report.md`.
Prompt: *"Finalize the Introduction from `introduction.md`. Re-centre the contribution paragraph
on the calibration finding and the risk-term-beats-risk-neutral-SO result (not 'coupled
decisions'). Tie the gap to the finding (avoid novelty-by-conjunction). Add one sentence
delineating what is reused from Mühlbauer [7] vs new here, and distinguish Xie [24] on substance.
Label the three research questions RQ1–RQ3 explicitly. Output final prose."*

### Step F — Abstract (do last; it summarizes the finished body)
Attach: `abstract.md`, the finalized Results/Conclusion, `paper_narrative.md`.
Prompt: *"Finalize the Abstract from `abstract.md` in ≤200 words. Lead with calibration, state
'comparable cost' (name LP-Avg as the cost minimizer), split the long results sentence into two,
and trim the diesel motivation to one sentence. No claim absent from the body."*

---

## 3. Also generate (not yet drafted) — one prompt each

- **Impact statement** (Cambridge Prisms requires it): *"Write a ~120-word non-technical impact
  statement for this paper (calibration + risk-aware sizing for renewable islanded microgrids;
  diesel replacement), for a broad readership."*
- **Nomenclature/abbreviations table** (SO-CVaR, LP-Avg, LP-Worst, VoLL, PCM, PV, CVaR, VaR, CRF,
  FBCF, COP, NSRDB).
- **End-matter declarations:** data & code availability, author contributions, funding, competing
  interests.

---

## 4. Before you submit — remaining fixes the agent cannot invent (need you or a re-run)

- **References:** fill placeholders [38] (the PHTM/passive-thermal paper — load-bearing), [41]
  (Lagrange — unresolvable, supply or drop), [48] NREL ATB / a dedicated PCM-cost source, [51]
  HOMER; cite or delete the orphan [16]; renumber to order of appearance; confirm the required
  Cambridge Prisms citation style.
- **A paired significance test** across the five climates (see `reviewer_report.md` Issue 3) —
  turns the "direction across cells" claims into a defensible statement.
- **Optional strengthening (a one-line λ=0 re-run of the mains):** would put the risk-neutral SO
  on the train-vs-test calibration figure and extend the risk-term comparison to the PV–battery
  architecture. The reliability-level/tail win (fig_riskterm) already establishes the point.

---

## 5. Final figure lineup (main text)

1. **Figure 1** — workflow schematic (`fig1_workflow`)
2. **Figure 2** — out-of-sample cost, parity (`fig3_total_cost`)
3. **Figure 3** — reliability calibration, HERO (`fig_calibration`)
4. **Figure 4** — risk reduction: tail + cost stability (`fig_risk`)
5. **Figure 5** — risk term vs risk-neutral SO (`fig_riskterm`)
6. **Figure 6** — diesel break-even (`fig6_diesel_breakeven`)

SI: capacities (`fig2_capacities`), loss-of-load (`fig4_loss_of_load`), generalization dumbbell
(`fig5_generalization`, superseded by Figure 3), frontier (`si_fig_frontier`), VoLL sensitivity
(`si_fig_voll`), risk-parameter robustness (`si_fig_risk_params`), interannual variability
(`si_fig_variability`). (Renumber SI figures S1–S7 to match their order of citation.)
