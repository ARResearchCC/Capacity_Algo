# How to assemble the paper (lean workflow)

The section drafts in `manuscript/` are already **near-final prose**. The other agent's job is
light **polishing + a few small fixes per section** — not writing from scratch, and not reading
the whole project. Follow the three rules below and it will be fast and won't overflow context.

## The three workflow rules (this is what fixes the "conversation too long" problem)
1. **One NEW chat per section.** Do not do all sections in one conversation — context piles up and
   the chat dies. Finish a section, copy its output, start a fresh chat for the next.
2. **Attach ONE file per chat** — only that section's draft (e.g. `results.md`). Do **not** attach
   `paper_narrative.md`, `figure_narratives.md`, or `reviewer_report.md`; their content is already
   baked into the drafts and the rules card below.
3. **Paste the RULES CARD once** at the top of each chat, then the short section prompt. That's it.

---

## RULES CARD  (copy-paste this block at the top of every section chat)

```
You are finalizing ONE section of a Cambridge Prisms journal paper: risk-aware (CVaR) capacity
planning + PCM thermal storage for a fully renewable islanded forward-operating-base microgrid
(PV + battery + heat pump, with/without hot & cold PCM). The text I give you is a near-final draft —
polish it to publication quality and apply the fixes I list. Output ONLY the finished section.

Non-negotiable rules:
- Never say the CVaR method is "cheaper." On mean cost the average-year design (LP-Avg) is cheapest;
  CVaR is +0.2–1.5% and below the worst-year design (LP-Worst). Always "at comparable cost."
- No cost–reliability "knee"; do NOT say the advantage "grows with VoLL" on cost.
- Method-to-method gaps are within fold-to-fold weather noise → report the consistent DIRECTION
  across cells; never write "statistically ..." (no significance test was run).
- Loss of load is essentially all thermal (critical/electrical ≈ 0, near-must-served).
- Lead the story on CALIBRATION — SO-CVaR delivers the reliability it was sized for, while LP-Avg
  under-provisions and LP-Worst over-provisions — plus the risk term beating a risk-neutral
  stochastic baseline (λ=0). NOT "coupled planning×storage decisions" (that's an architecture effect).

Verified numbers (use as-is; do not invent):
- Calibration median test/plan unmet ratio SO-CVaR 1.02, LP-Avg 1.11, LP-Worst 0.51; SO-CVaR
  best-calibrated in 26/30 cells (4 tropical exceptions).
- Risk term vs risk-neutral SO (λ=0): −40% mean unmet, −19% tail, +1.2% cost (15 PCM cells).
- Risk vs LP-Avg: worst-fold unmet −21%, year-to-year cost variance −31% (30/30 cells).
- Cost: SO-CVaR +0.2–1.5% vs LP-Avg, below LP-Worst. Generalization train→test unmet:
  LP-Avg +17.2% / SO-CVaR +7.6% / LP-Worst −51.4%.
- PCM: displaces battery 38–86%, lowers cost 4–14%. Diesel break-even $0.78–8.90/gal — renewables
  win when delivered fuel EXCEEDS the break-even (i.e. essentially always).

Style: UK spelling (-ise/-our); past tense for Methods/Results; keep the acronyms SO-CVaR / LP-Avg /
LP-Worst; refer to figures as "Figure N." Do not add any claim not in the draft.
```

---

## Per-section prompts (fresh chat each; attach the one file; paste RULES CARD, then this)

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

## Extra pieces (one short chat each, no file needed — just the RULES CARD + this)
- **Impact statement:** "Write a ~120-word non-technical impact statement for this paper (calibration + risk-aware sizing for renewable islanded microgrids; diesel replacement)."
- **Nomenclature list:** "List and expand these abbreviations as a table: SO-CVaR, LP-Avg, LP-Worst, VoLL, PCM, PV, CVaR, VaR, CRF, FBCF, COP, NSRDB."
- **Declarations:** "Draft short data & code availability, author contributions, funding, and competing-interests statements."

---

## These need YOU (not the agent — they need a value, a decision, or a computation)
- **References:** fill placeholders — [38] passive-thermal/PHTM paper (load-bearing), [41] Lagrange
  (supply or delete), [48] a real PV/battery/**PCM** cost source, [51] HOMER; resolve orphan [16];
  reorder to citation order; confirm Cambridge Prisms style (likely author–date).
- **SI:** add the PCM power rating / round-trip efficiency / standing loss values (from your model)
  in S5; mirror the calibration + risk-metric definitions into S9; add SI figure callouts.
- **Paired significance test** across the five climates (turns "direction across cells" into a real
  statement). Optional: the one-line λ=0 mains re-run for the calibration figure + PV-battery case.

## Final figure lineup (main text)
Fig 1 workflow (`fig1_workflow`) · Fig 2 cost-parity (`fig2_total_cost`) · **Fig 3 calibration
(`fig3_calibration`)** · Fig 4 risk (`fig4_risk`) · **Fig 5 risk-term vs λ=0 (`fig5_riskterm`)** ·
Fig 6 diesel (`fig6_diesel_breakeven`). SI: capacities (`si_fig_capacities`), loss-of-load
(`si_fig_loss_of_load`), generalization dumbbell (`si_fig_generalization`), frontier, VoLL
sensitivity, risk-parameter robustness, variability. All main-text image files live in
`paper_figures/main/` numbered `fig1`…`fig6` in paste order.

---
*Tip: if even one section chat feels heavy, you don't need an agent at all — the draft `.md` files
are already close to submission quality. You can paste them into the paper directly and hand-apply
the short fix list above.*
