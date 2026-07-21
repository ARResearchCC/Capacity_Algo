# Fact-check report — manuscript vs code, results, figures, citations

Pre-submission accuracy pass over the full manuscript (`abstract.md`, `introduction.md`,
`methods_and_data.md`, `SI.md`, `references.md`) against the actual code, the result
workbooks, the figure CSVs, and the cited literature. **145 claims checked across 9
verification tracks.** Status legend: ✅ confirmed · 🔧 fixed by me · ⚠️ needs your
attention (authorial or unresolvable).

**Bottom line:** the model, formulation, parameters, and nearly every headline number are
**accurate and trace cleanly to the code and workbooks.** The defects are (a) two stale SI
data tables (now fixed), (b) one framing inconsistency (RQ1), (c) an unwritten abstract
findings sentence, and (d) a handful of reference-list gaps/typos. No fabricated citations
and no substantive model-vs-text divergence were found.

---

## 0. What I already fixed (this pass)

| Fix | File | Detail |
|---|---|---|
| 🔧 SI §S7 VoLL | `SI.md` | critical `$100/$300/$600` → **`$30/$100/$300`** (thermal $1/$3/$10 was already right). Confirmed against `FOB.py` VOLL_SCENARIOS, `data.py` VOLL_TABLE, both workbooks, and all figure captions. |
| 🔧 SI §S10 fuel prices | `SI.md` | `$100/$400/$600` → **`$13/$45/$400`** (peacetime ground / protected convoy / contested). Matches `FOB_Diesel.py` FUEL_PRICE_SCENARIOS + diesel workbook. Reworded the JASON sentence so $100–600 is cited as *context*, not the run values. |
| 🔧 SI §S1 Alaska | `SI.md` | "polar/subarctic" → **"subarctic (Dfc)"** (coords 59.25,−154.62 are Dfc, not polar; only site lacking a Köppen code). |
| 🔧 SI §S4 load | `SI.md` | kettle set `{0,0.13,0.27}` → **`{0,0.133,0.266}`** (exact code values); removed "plug loads" from the FOB load (code adds lighting+computer+kettle only; plug load 0.1 kW is unused). |
| 🔧 SI §S3 clustering | `SI.md` | clarified SHGC clusters on 6 variables but **RCC on only 4** (WS, RH, Ta, Cloud) — matches `Passive_Model.py`. |
| 🔧 refs [39], [45] | `references.md` | fixed a find/replace corruption: Gorman vol. `[35]`→**`35`** (+DOI); JASON `JSR-06-1[35]`→**`JSR-06-135`**. |
| 🔧 refs [42], [43] | `references.md` | filled placeholders from `voll_writeup.md`: Schröder & Kuckshinrichs 2015 (Frontiers in Energy Research 3:55); Anderson et al. **2021** (IEEE Systems J.; year corrected from the placeholder's 2018). Marked "VERIFY". |
| 🔧 source docs | `figure_narratives.md`, `paper_narrative.md` | "beats LP-Worst in AZ" → **"AZ-PVB"** (AZ-PCM does not); so the other Claude writes it correctly. |

---

## A. Methods + SI vs code — ✅ essentially all confirmed

The strongest result of the audit: **the formulation and every parameter match the code.**
- ✅ **CVaR objective (SI S8 / Methods 2.3)** is implemented term-for-term in `SO_CVaR.py:196–218`; R-U linearization (`ξ_s ≥ Z_s−η, ξ_s ≥ 0`, η free) at `:163–176`; `Z_s` = VoLL-weighted thermal (G2H) + critical-electrical (G2E) unmet at `:166–171`. Shared first-stage capacities, scenario-indexed dispatch. λ=0.9, α=0.9 (`Input_Parameters.py:134–135`). Pyomo + Gurobi.
- ✅ **LP-Avg / LP-Worst** = mean of per-year optimal capacities / capacities of the single highest-cost training year (`FOB.py:122–146`); identical dispatch physics to the stochastic model.
- ✅ **PV/Fuentes (S2):** every parameter matches — a=0.83, ε=0.84, NOCT=45, Γ=−0.004, T_ref=25, η_PV=0.96, τ_opt=0.96, h=5.7+3.8v, IAM, inverter 0.94.
- ✅ **Costs (S5):** PV $1500/kW+$15, battery $500/kWh+$5, HP $1000/kW+2%, hot/cold PCM $70/kWh+4%; HP fixed 10 kW, COP 3.5, battery 0.98/DoD 0.8/self-disch 0.01·24⁻¹/0.25C — all confirmed.
- ✅ **CRF (S6):** L=20, d=0.03 → 0.0672, ×hours/8760. ✅ **CV (S9):** 5×5-year blocks, 20/5.
- ✅ **Diesel (S10):** GenSize=(1+0.20)·max[E+P_hp], crit-then-HP dispatch, F0=0.0215/F1=0.065, CGen $800/kW, gen O&M 3%, var $0.02/kWh, gen life 15 yr, HP 20-yr CRF — all confirmed; break-even formula matches.
- ✅ Five sites (CA/AK/AZ/MN/FL), NSRDB 1998–2022, 125 location-years (verified: exactly 25 CSVs × 5 folders), per-year seeding.

**Remaining Methods/SI items (⚠️ your call):**
- ⚠️ **Ref [38] "PHTM Paper" is a placeholder** cited 3× (Methods 2.2, SI S3/S4) and it underpins the "experimentally validated" passive-thermal claim. The citations agent found a likely match — *"High-precision passive heat-transfer modeling for a microgrid research station"* (ScienceDirect S235271022502220X) — **please confirm and fill [38]**; only keep "experimentally validated" if [38] documents validation.
- ⚠️ **S3 omits internal gains.** The *served* thermal load adds occupant metabolic heat + electrical-equipment waste heat to the passive Q_net (`Data_Conversion.py:168`) before the heating/cooling split. Recommend one sentence in S3 (I left the wording to you since it's a model-description addition, not a wrong number).
- ⚠️ **Symbol mismatch:** Methods 2.3 writes the capacity term `C_cap`; SI S8 writes `C_first-stage`. Standardize to one.
- ⚠️ **PV azimuth unstated:** the "three orientations" in `Solar_Generation.py` all use azimuth 165° (≈ due south); consider stating azimuth in S2.
- ⚠️ **PCM power rating:** S5 models PCM as an energy reservoir with no stated charge/discharge power cap — confirm whether the model imposes one (add to the parameter table if so).

---

## B. Consistency (abstract & intro vs results)

- ⚠️ **RQ1 — the single most important fix.** Intro RQ1 asks *"Does CVaR achieve a **lower** out-of-sample **total system cost** than … LP-Avg and LP-Worst?"* The data say **no vs LP-Avg** (SO-CVaR is +0.2 to +1.5% at Med VoLL, in all 10 cells) and **yes vs LP-Worst** (below it everywhere); it is the outright cost minimizer only at **High VoLL with PCM**. As worded, RQ1 presupposes a two-sided cost win the results contradict.
  **Suggested reword:** *"Does CVaR-based planning improve the out-of-sample cost–reliability balance relative to the average-year and worst-year benchmarks — trading a small mean-cost premium for reduced tail loss-of-load and more reliable out-of-sample performance?"* Then answer explicitly: not lower cost than LP-Avg at nominal VoLL; consistently below LP-Worst; cost-minimizer only under High VoLL + PCM.
- ⚠️ **Abstract findings sentence (placeholder) — do not over-claim.** When filled: (a) **do not** headline an *electrical* loss-of-load reduction — critical/electrical unmet is ~0 across all methods, so the reliability story is **thermal**; (b) phrase CVaR's reliability gain as "**lower unmet than LP-Avg**" (LP-Worst still has the least unmet, at the highest cost); (c) keep "**cost change**," not "cost saving." Suggested numbers in the completeness doc.
- ⚠️ **RQ2 "wider margin with PCM" — frame via the frontier, not per-cell margins.** The reliability margin over LP-Avg *widens* with PCM in only 3/5 climates and *narrows* in AZ/FL. The robust support is the **Pareto frontier**: SO-CVaR is efficient in **5/5** climates with PCM vs **0/5** without (`si_fig_frontier.csv`), and is the cost-minimizer only with PCM at High VoLL. Argue RQ2 from frontier membership + the FL dominance, not a uniform margin.
- ✅ **RQ3** (diesel competitiveness by climate/reliability/fuel price) — fully supported by the break-even ($0.78–8.90/gal).
- ✅ **Contribution statement** ("value measured in reliability, not only cost") — this is the honest framing; anchor RQ1's answer to it.
- ✅ Abstract framing ("asks whether", "cost change") and keywords — consistent.

---

## C. Results & analysis — ✅ numbers confirmed; a few precision fixes

**Confirmed exactly** against figure CSVs + workbooks: cost gaps (+0.2/+1.5% vs LP-Avg, median +0.45%; −0.1/−3.3% below LP-Worst, 10/10); thermal-unmet reductions 9–68% (10/10) and 29–89% gap-closed; generalization (unmet **+17.2 / +7.6 / −51.4%**; cost **+1.84 / −8.0 / −9.1%**, computed as mean of per-cell deg_pct); PCM battery displacement 38–86%; PCM cost reduction 4–14%; diesel break-even **$0.78–8.90/gal**; FL SO-CVaR/PCM dominance of LP-Worst/PCM (17.2 kWh/$2141 vs 19.4/$2148); risk premium **+1.58% mean / +6.27% max**; cost-optimal λ≤0.25 in all 15 cells. **Traceability confirmed:** top-level workbooks == `NEW_RESULTS/` copies == figure CSVs (spot-checked).

**Precision fixes for whoever writes Results (⚠️ small):**
- "beats LP-Worst in **AZ**" → **AZ-PVB only** (AZ-PCM: 60.2 > 55.8). *(Fixed in my source docs.)*
- "**−40 to −53%** tail unmet" mixes baselines: vs λ=0 it's **−45 to −53%** (High-VoLL) / **−34%** pooled; vs cost-optimal selection it's **−40 to −47%**. Pick one baseline and state it.
- "**PV ≤4%**" is LP-Avg-only and MN is actually −4.2% → write "**≲4% (LP-Avg basis)**".
- Fold-SD "**~13%** vs spread **~4%**" are Alaska (largest-cost) values; medians are ~5% vs ~1.6%. Say "Alaska" or use medians. The SD>spread relationship holds in all 10 cells.
- FL dominance $2141/$2148 are **capital (cost-ex-penalty)**; total costs are $2193 vs $2207 — clarify. Gap (~2 kWh, ~$7) is within fold SD → present as directional, not statistically resolved.
- **`fold noise > method gaps` is asserted but not tested.** Recommend a paired sign/rank test across the 10 cells to make the "10/10 direction" defensible (currently rests on point estimates).
- Fig 6 "method-independent break-even": still only SO-CVaR is plotted; the ~1–3% cost spread → ~3–4% break-even spread. Either add LP ticks or keep it in text (already attributed to Fig 3). Relabel the CSV column `best_method`→`design_method`.

---

## D. Citations

**✅ Good news:** every checkable cited work is real and supports its sentence; **no fabricated references**; ~24 entries verified exactly; the [44] Rockafellar-Uryasev locator (2(3):21–41, 2000) is correct. *(Scite was unavailable — token expired — so verification used web/DOI resolution; a Scite pass before submission is advisable.)*

**🔧 Fixed:** [39]/[45] "[35]" corruption; [42]/[43] filled (Schröder 2015; Anderson **2021** not 2018).

**⚠️ Still needs you:**
- **[41] Lagrange et al. — unresolvable.** It appears nowhere in your own VoLL bibliography (`voll_writeup.md`) and can't be identified. **Supply the intended source or drop it** (cited at [40–43] in Methods 2.3 / SI S7). Do not leave a placeholder.
- **[16] Zhou et al. is an orphan** — in the list but never cited in the body. Cite it (fits the CVaR/risk-averse-planning paragraph) or delete + renumber.
- **[29]–[33]** are missing authors/volume/article-number and use ScienceDirect PIIs instead of DOIs; **[29]** likely has a 2025→2026 year issue. Normalize to the [1]–[27] format.
- **[38] PHTM, [48] NREL ATB, [51] HOMER** — finalize (year/access-date). **[48]:** confirm ATB actually backs the PV/battery/HP costs and especially the **$70/kWh PCM** cost (PCM is unlikely to be an ATB line item — may need a separate PCM-cost citation). **[51]:** confirm it backs F0/F1 and the $800/kW genset cost.
- **[40] LBNL ICE** — fill the access date.
- **FBCF lower tiers:** if the paper states the $2.8/$13/$42 ladder tiers (now in SI S10's neighborhood), the [45][46][47] set is insufficient — add **National Defense Magazine (2010)** (primary ladder source) and **GAO-09-300/388T (2009)** at minimum (optionally OSD FBCE 2013, PolitiFact 2011, DSB 2001). None are currently in `references.md`.
- **[46] Deloitte 2009** is an advocacy source that's been methodologically criticized — caveat it or lean on the DoD/GAO primaries for load-bearing numbers.
- **Citation-format inconsistency:** [1]–[27] give DOIs, [28]–[33] give PIIs, [34]–[51] mixed. Normalize before submission.
- If the **α=0.9 sample-size justification** (from `risk_parameters_writeup.md`) is moved into Methods/SI, add its refs (R-U 2002, Yamai-Yoshiba 2005, Sarykalin 2008); the body currently uses only [44] (2000), which is correct and sufficient as written.

---

## Priority triage (do these before the other Claude drafts Results/Discussion)

1. ⚠️ **Reword RQ1** (and decide the honest framing of the cost result) — everything downstream inherits this.
2. ⚠️ **Fill the abstract findings sentence** honestly (thermal LOL; cost change not saving).
3. ⚠️ **Resolve [41] Lagrange** and **[38] PHTM** (both load-bearing placeholders).
4. ⚠️ Decide RQ2's frontier framing.
5. 🔧 Already done: SI S7/S10 data, [39]/[42]/[43]/[45], the small SI factual fixes.

*(All ✅/🔧 items are settled; the ⚠️ items are authorial decisions or need a source only you have.)*
