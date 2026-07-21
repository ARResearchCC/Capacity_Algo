# Completeness & gaps — what still needs writing for the full manuscript

Companion to `factcheck_report.md`. Inventory of what exists vs what's missing for a
complete Cambridge Prisms submission, with the **source** for each missing piece so the
writing Claude knows where to pull from. (Accuracy of what exists is in the fact-check
report; this doc is about *coverage*.)

---

## What already exists

| Piece | File | State |
|---|---|---|
| Title + Abstract | `abstract.md` | ✅ prose, ⚠️ **findings sentence is a `[placeholder]`** |
| Introduction (incl. 3 RQs, contribution) | `introduction.md` | ✅ complete & well-cited (⚠️ RQ1 reword — see fact-check) |
| Methods & Data (Section 2) | `methods_and_data.md` | ✅ complete (defers detail to SI) |
| Supplementary S1–S10 | `SI.md` | ✅ complete drafts (🔧 S7/S10 data fixed; ⚠️ some stubs) |
| References | `references.md` | ✅ [1]–[51] present; ⚠️ gaps/placeholders (see fact-check §D) |
| Figures 1–6 (main) + S1/S2/S3/S5 | `paper_figures/` | ✅ generated & fact-checked |
| Per-figure narrative + numbers | `figure_narratives.md`, `paper_narrative.md` | ✅ source material for Results/Discussion |

---

## HARD GAPS — unwritten prose (the main remaining work)

### 1. Abstract findings sentence  ·  source: `figure_narratives.md`, `fig*.csv`
~35 words. Honest template (see fact-check §B for the rules):
> *"Across five climates and 25 unseen weather years, loss of load is almost entirely thermal; CVaR sizing cuts thermal unmet load 9–68% below the average-year design for a 0.2–1.5% mean-cost premium (and below the worst-year design's cost), PCM lowers system cost 4–14% while displacing 38–86% of battery capacity, and the renewable microgrid breaks even with diesel at $0.8–8.9/gal — far below in-theater fuel."*
Trim to fit; keep "cost change", not "saving"; do not headline electrical LOL.

### 2. Section 3 — Results  ·  source: `figure_narratives.md` (per-figure) + `paper_narrative.md`
Not drafted. Structure keyed to the figures:
- **3.1 Sized capacities** (Fig 2→SI; lift 2 sentences: PCM displaces battery not PV; risk spectrum).
- **3.2 Out-of-sample cost** (Fig 3): parity — SO-CVaR +0.2–1.5% vs LP-Avg, below LP-Worst; gaps within fold noise.
- **3.3 Loss of load / reliability** (Fig 4): thermal-only; knee; 29–89% gap-closed; PCM's two effects.
- **3.4 Generalization** (Fig 5): LP-Avg overfits reliability (+17%), SO-CVaR nearly unbiased (+7.6%).
- **3.5 Diesel break-even** (Fig 6): $0.78–8.90/gal, below routine delivery in every climate.
- (SI figures S1 frontier, S2 VoLL, S3 risk params, S5 variability referenced here.)
Must obey the "golden rules" in `paper_narrative.md` (never "cheaper"; state within-noise; thermal).

### 3. Section 4 — Discussion (incl. Limitations)  ·  source: `paper_narrative.md`, `figure_narratives.md` cross-cutting notes
Not drafted. Suggested subsections:
- PCM as a **battery substitute** (mechanism + climate dependence).
- **Risk-aware planning**: knee/robustness framing (NOT cost-optimality); a-priori λ,α.
- **Deployment / diesel replacement** (RQ3).
- **Limitations** (explicit subsection): method gaps within fold noise (no significance test yet); polar HVAC structurally unservable regardless of method; risk sweep covers **only the PCM architecture**; VoLL & fuel-price are assumptions (report as sensitivity/break-even); PCM modeled as an energy reservoir (no power-rating detail); single FOB load model.

### 4. Section 5 — Conclusion + future work  ·  source: `paper_narrative.md`
Not drafted. Future work: extend the λ/α sweep to the PVB architecture; add the hourly dispatch trace (S4 fig); tail-metric risk selection (the shelved analysis).

---

## SI GAPS

- ⚠️ **No figure callouts in `SI.md`.** S1–S10 reference no figures, but SI figures exist (frontier→S1, VoLL sensitivity→S7, variability→S5, risk params, dispatch). Add callouts + captions (regenerate caption text from the CSVs so numbers can't drift).
- ⚠️ **Risk-parameter robustness section is missing from `SI.md`.** A full draft exists in `risk_parameters_writeup.md` (the nested-CV robustness section for `si_fig_risk_params`). Fold it in as a new SI section (e.g. **S11**) and reconcile its numbers (+1.58% mean / +6.27% max; −34% vs λ=0). **Its ~10 methodology references** (R-U 2002, Krokhmal 2002, Sarykalin 2008, Yamai-Yoshiba 2005, Conejo 2010, Moazeni 2015, Shapiro et al., Varma-Simon 2006, Cawley-Talbot 2010) are **not** in `references.md` yet — add them when this section is folded in.
- ⚠️ **SI stubs to finalize** (also in fact-check §D): [38] PHTM (S3 depends on it), [48] ATB cost source (esp. PCM $70/kWh), [51] HOMER fuel-curve/genset cost.
- ⚠️ **S3 internal-gains sentence** (occupant + electrical waste heat added to passive Q_net) — recommended addition.

---

## FIGURE / DATA GAPS

- ⚠️ **S4 hourly dispatch figure NOT generated.** `Dispatch_Trace_Results/` is empty; `si_run_dispatch_trace.py` needs the model env (Gurobi) — run under `.venv`, then `si_fig_dispatch.py`, then add the figure + an SI callout. (Currently `SI.md` has no dispatch reference at all.)
- ⚠️ **S5 variability lacks the peak-electrical-load panel** (`si_fig_variability.py` omits it — exports only annual energy, not peak power). Either export per-year peak `E_Load` and add the 4th panel, or state peak load is out of scope in the caption.
- ⚠️ **Risk (λ/α) sweep is PCM-architecture only** (`si_run_risk_sweep.py` scenario='FOB'; n=15 = 5×3). Either extend to PVB or state the limitation explicitly in the SI + caption.

---

## STRUCTURAL / JOURNAL ITEMS (not yet audited for Cambridge Prisms requirements)

Confirm against the *Cambridge Prisms* author guidelines whether these are required, and add if so:
- **Impact statement** (Cambridge Prisms typically requires one).
- **Nomenclature / symbol table** (the model has many symbols across Methods + SI S8).
- **Data & code availability** — a line exists (`methods_and_data.md:15`, "DOI added on acceptance"); confirm it meets the Open Research policy.
- **Author contributions, Funding, Conflict of interest, Acknowledgements.**
- Figure/table numbering + in-text cross-references once Results/Discussion are drafted.

---

## Suggested order of operations

1. **Settle RQ1 wording + the honest cost framing** (fact-check §B) — everything inherits it.
2. **Fill the abstract findings sentence.**
3. **Resolve [41] Lagrange and [38] PHTM** (load-bearing placeholders you alone can source).
4. **Draft Results (§3)** from `figure_narratives.md`, obeying the golden rules.
5. **Draft Discussion (§4) + Limitations**, then **Conclusion (§5)**.
6. **Fold the risk-parameter robustness section into SI (S11)** + add its references + SI figure callouts.
7. **Generate the S4 dispatch figure** (Gurobi run) if you want it in.
8. **Finalize references** (normalize format; fill [16]/[29]–[33]/[48]/[51]; add FBCF primaries if the ladder tiers are stated); **run a Scite pass** for DOI confirmation.
9. Add the **journal structural items** (impact statement, nomenclature, availability, contributions).
