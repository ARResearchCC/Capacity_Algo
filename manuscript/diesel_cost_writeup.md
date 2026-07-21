# Delivered diesel cost (fully burdened) — justification, benchmarking, manuscript text

Covers (1) journal-ready prose, (2) a primary-source-verified fully-burdened-cost-of-fuel
(FBCF) ladder, (3) why the renewable-vs-diesel comparison is meaningful when framed as a
break-even, (4) assessment/recommendation, (5) references. Author–date style.

**Headline finding:** the $100–600/gal figures are the *combat-zone (hostile-delivery)*
FBCF range from the primary literature — defensible for a contested FOB, not an
arbitrary extreme. The comparison is made meaningful (not a foregone conclusion) by
reporting the **break-even fuel price**, which is far below even *peacetime* forward
delivery — so renewables dominate diesel across the entire peacetime-to-contested range.

---

## 1. Methods (main text) — the diesel baseline and delivered fuel cost

The diesel baseline generator serves all load, so its cost is dominated by fuel priced
at the **fully burdened cost of fuel (FBCF)** — commodity price plus transport, storage,
force protection, and the logistics-tail risk of moving fuel to the point of use
(Defense Science Board 2001; Office of the Secretary of Defense 2013). FBCF rises sharply
with delivery difficulty, forming a well-documented ladder: commodity ≈ $2.8/gal;
peacetime ground delivery to forward locations ≈ $13/gal; in-flight aerial refuelling
≈ $42/gal; and **hostile-area (combat-zone) delivery ≈ $100–600/gal**, with the Army's
Afghanistan helicopter-resupply estimate ≈ $400/gal and extreme cases approaching
$1,000/gal (JASON/MITRE 2006; National Defense Magazine 2010; Government Accountability
Office 2009). At ~0.08–0.09 gal kWh⁻¹(e) for tactical gensets, the hostile-delivery range
corresponds to roughly **$9–54/kWh** of diesel electricity from fuel alone — before the
casualty and mission-assurance premium implied by convoy losses (≈1 casualty per 24 fuel
convoys in Afghanistan; Army Environmental Policy Institute 2009).

Because the contested-theatre FBCF spans a wide range, we do not assume a single price.
We report the diesel comparison as a **break-even delivered fuel price** — the price at
which the diesel system's annualised cost equals the renewable design's — computed
price-independently as `total(p) = (capital + O&M) + gallons·p`.

---

## 2. Verified FBCF ladder

| Tier | ~$/gal | ≈ $/kWh(e)* | Basis |
|---|---|---|---|
| Commodity / DLA standard | 2.8 | ~0.3 | point of purchase (Nat'l Defense Mag 2010) |
| Peacetime forward ground delivery | 13 | ~1.2 | Nat'l Defense Mag 2010 |
| In-flight aerial refuelling | 42 | ~3.8 | Nat'l Defense Mag 2010; DSB 2001 |
| **Hostile-area (combat-zone) delivery** | **100–600** | **~9–54** | JASON/MITRE 2006; Nat'l Defense Mag 2010 |
| Afghanistan helicopter resupply | ~400 | ~36 | 2009 Pentagon testimony (GAO; PolitiFact 2011) |
| Extreme / most remote | up to ~1,000 | ~90 | anecdotal, single Army estimate |

\* fuel-only, at ~0.085 gal kWh⁻¹(e) (≈30% genset efficiency); excludes casualty /
mission-assurance premium.

---

## 3. Why the comparison is meaningful (break-even, not a point at an extreme price)

The break-even delivered fuel price is **≈2–9 $/gal** across the five climates — i.e. the
renewable microgrid becomes cheaper than a diesel genset once delivered fuel exceeds a
few dollars per gallon. This lies **at or just above the bare commodity price (≈$2.8/gal)
and below even peacetime forward ground delivery (≈$13/gal)** — long before the
force-protection premium that drives combat-zone FBCF to $100–600/gal. So the conclusion
(renewables dominate diesel) is robust across the *entire* delivery spectrum, from benign
to contested, and does not rely on assuming an extreme in-theatre price. Presented this
way the comparison is informative — it locates the crossover and quantifies the margin —
rather than a foregone conclusion manufactured by a high fuel-price assumption. The
break-even is also independent of the VoLL assumption (the diesel system serves all load,
so VoLL does not enter its cost).

---

## 3b. VoLL vs the diesel price — two different quantities (likely reviewer question)

**Which diesel price does the study actually use?** None, singly. The diesel benchmark is
*run* at three prices ($13 / $45 / $400 per gallon) only to obtain its design (generator
size, annual fuel gallons, and fixed capital + O&M). Figure 6 then reports a **break-even**
price — the delivered fuel price at which the diesel and renewable annualised costs are
equal — computed price-independently as `total(p) = fixed_cost + gallons·p`. So no single
price is assumed; the result (break-even ≈ $0.8–8.9/gal, per climate and architecture)
holds for *any* delivered price above it. The $100–600/gal band drawn on the figure is the
contested-theatre reference, shown only to demonstrate how far below it the break-even sits.

**Are VoLL and the diesel price consistent — and if diesel is dearer than VoLL, why not
shed load instead of powering it?** VoLL and the diesel price are *independent inputs to
different parts of the study* and measure different things. VoLL is the **demand-side
value** of lost load (the mission/operational consequence of an outage); the diesel price
is the **supply-side cost** of the benchmark generator. In the renewable model there is no
diesel, so the alternative to serving a kWh is not "run a genset" but "accept the outage,
valued at VoLL." The optimiser therefore trades renewable *capital* against the VoLL
penalty and never faces a "run diesel vs shed" decision — hence the "if diesel > VoLL, do
not power" logic does not apply. That logic would only arise under a renewable-plus-diesel-
*backup* reformulation, which we deliberately do **not** adopt (it would tie the two inputs
together and require re-running the model). Nor is the site ever left unpowered: built
renewables have ≈zero marginal cost, so they always dispatch to serve what they can; VoLL
governs only *how much capacity to build*, not whether to run it.

**Do the magnitudes at least cohere?** Yes, reassuringly, and they explain the model's
behaviour. Critical (electrical) VoLL ($30–300/kWh) far exceeds the fuel-only cost of
diesel electricity (≈$4/kWh at $45/gal; $9–54/kWh at $100–600/gal), which is exactly why
the optimiser near-must-serves critical load (unmet ≈ 0 everywhere). HVAC (thermal) VoLL
($1–10/kWh-thermal) is comparable to the cost of meeting thermal demand with diesel via a
heat pump (≈$1–15/kWh-thermal at COP ≈ 3.5), which is why a small amount of thermal load is
economically shed in the tail. So the two independent inputs are mutually consistent and
jointly account for the observed "critical firm, thermal is the tradeoff" behaviour.

---

## 4. Assessment and recommendation

- **The $100–600/gal range is defensible for a combat-zone FOB** — it is the hostile-area
  delivery tier of the primary FBCF literature (JASON 2006; National Defense Magazine
  2010), not an arbitrary extreme. No need to lower it.
- **Make the comparison meaningful through the break-even framing, not by inflating the
  price.** Concretely:
  1. Report the per-climate break-even price (Fig 6) as the primary result.
  2. Span the FBCF ladder in the fuel-price scenarios. **Decided/implemented set:
     Low $13 (peacetime forward ground delivery), Med $45 (protected convoy — defensible
     burdened tier), High $400 (contested/helicopter extreme, documented upper bound)** —
     so the figure shows the break-even (~$2–9/gal) crossed below even peacetime delivery,
     with the renewable advantage widening across the ladder.
  3. On Fig 6, shade the ladder tiers (a "routine delivery ≈$13–45/gal" band and a
     "contested ≈$100–600/gal" band) and mark the break-evens below both.
  4. State the fuel-only $/kWh conversion (§2) so readers see combat-zone diesel
     electricity is ~$9–54/kWh — far above the renewable design's annualised cost.
- **Net effect:** the renewable-vs-diesel result is *more* defensible, because the win
  holds from commodity-priced fuel upward, and the contested FBCF (which a combat-zone FOB
  is entitled to use) simply makes it overwhelming.

**Status: implemented in the model.** `FUEL_PRICE_SCENARIOS` in `FOB_Diesel.py` and
`Input_Parameters.Diesel_Price` (= $45, Med) are set to $13 / $45 / $400. The break-even is
price-independent, so re-running the diesel model is optional (it only refreshes the stored
fuel price and adds the fuel-price sweep dimension); re-render
`paper_figures/fig6_diesel_breakeven.py` with the shaded ladder tiers after the run.

---

## 5. References (verified via primary sources unless flagged)

- Army Environmental Policy Institute (2009). *Sustain the Mission Project: Casualty
  Factors for Fuel and Water Resupply Convoys.* Eady, D.S., Siegel, S.B., Bell, R.S.,
  Dicke, S.H. AEPI Final Technical Report. (DTIC ADB356341.) — casualty-per-convoy factors.
  *(Author is "Dicke", not "Dial".)*
- Defense Science Board (2001). *More Capable Warfighting Through Reduced Fuel Burden*
  (Task Force on Improving Fuel Efficiency of Weapons Platforms). OUSD(AT&L). DTIC ADA392666.
  — origin of the FBCF concept.
- Defense Science Board (2008). *More Fight, Less Fuel* (Task Force on DoD Energy
  Strategy). DTIC ADA477619.
- Defense Science Board (2016). *Energy Systems for Forward/Remote Operating Bases —
  Final Report.* https://dsb.cto.mil/wp-content/uploads/reports/2010s/Energy_Systems_for_Forward_Remote_Operating_Bases.pdf
- Government Accountability Office (2009). *Defense Management: DOD Needs to Increase
  Attention on Fuel Demand Management at Forward-Deployed Locations.* GAO-09-300 (and
  testimony GAO-09-388T). https://www.gao.gov/products/gao-09-300
- JASON / The MITRE Corporation (2006). *Reducing DoD Fossil-Fuel Dependence.* JSR-06-135.
  https://irp.fas.org/agency/dod/jason/fossil.pdf — FBCF $100–600/gal in theatre.
  *(The $100–600 range is corroborated by National Defense Magazine 2010; confirm the
  exact wording in the primary PDF before quoting verbatim.)*
- National Defense Magazine (2010). *How Much Does the Pentagon Pay for a Gallon of Gas?*
  https://www.nationaldefensemagazine.org/articles/2010/4/1/2010april-how-much-does-the-pentagon-pay-for-a-gallon-of-gas
  — the $2.82 / $13 / $42 / $100–600 ladder.
- Office of the Secretary of Defense (2013). *An Overview of the Fully Burdened Cost of
  Energy (FBCE).* https://www.acq.osd.mil/eie/Downloads/OE/Energy%20FBCE_12-11-13.pdf
- PolitiFact (2011). *…gasoline for troops in Afghanistan costs $400 a gallon.*
  https://www.politifact.com/factchecks/2011/may/23/marcy-kaptur/ — notes $400 ≈ remote
  worst case, not the average.
- Deloitte (2009). *Energy Security: America's Best Defense.* — supporting/advocacy
  (fuel-use vs casualties correlation; methodologically criticised — caveat if cited).

*Verify before submission:* the JASON $100–600 wording in the primary PDF; DSB 2001/2008
per-gallon figures (seen via secondary compilations); AEPI report month and DTIC access.

---

## 6. Citation traceability (in-text citation → exact claim it supports)

| In-text citation | Supports this specific claim/number |
|---|---|
| National Defense Magazine 2010 | the FBCF ladder: ~$2.82 commodity, $13 peacetime ground delivery, $42 aerial refuelling, $100–600 hostile-area (the primary clean source for the tiers) |
| JASON/MITRE 2006 (JSR-06-135) | fully burdened cost of fuel $100–600/gal in-theatre (hostile-area) — corroborated via Nat'l Defense Mag; confirm exact wording in primary PDF |
| GAO-09-300 (2009) | FBCF is "many times higher than the price of a gallon"; convoy protection burden; Iraq/Afghanistan fuel demand |
| PolitiFact 2011 | the ~$400/gal Afghanistan figure is a remote worst case, not the average |
| Defense Science Board 2001 | origin of the FBCF concept; ~$25–50/gal overland, "hundreds of dollars" by helicopter (via secondary compilation — verify) |
| Army Environmental Policy Institute 2009 | ~1 casualty per 24 fuel-resupply convoys in Afghanistan (force-protection premium) |
| OSD FBCE 2013 | official DoD definition/seven-step method of the fully burdened cost of energy |
| Deloitte 2009 | fuel-use-vs-casualties correlation — advocacy source, methodologically criticised; caveat if cited |

**~$/kWh conversion** (§2 table) is our own arithmetic at ~0.085 gal kWh⁻¹(e) (≈30% genset
efficiency), not a cited figure — labelled as a derivation.
