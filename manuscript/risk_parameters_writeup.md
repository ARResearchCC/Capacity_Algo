# Risk-parameter choice and validation — manuscript text

Draft text for the main Methods and the Supplementary Information, plus the
reference list. Author–date citations; adapt to the target journal's style
(Cambridge Prisms uses an author–date/Harvard reference format). Placeholders
`[Fig. Sx]` and numeric values `⟨…⟩` are to be filled from the completed sweep.

---

## Methods (main text) — risk-averse capacity planning

We size the microgrid with a two-stage mean–CVaR stochastic program in which each
historical weather year is an equiprobable scenario. First-stage variables are the
component capacities (PV, battery, hot/cold PCM); second-stage variables are the
hourly dispatch in each scenario. Writing `Z_s` for the annual value of lost load
(unserved energy priced at its VoLL) in scenario *s*, the second-stage cost is a
convex combination of its expectation and its Conditional Value-at-Risk (CVaR):

> minimise  C_capital + (1 − λ)·E[Z] + λ·CVaR_α(Z),

with CVaR linearised by the auxiliary VaR variable η following Rockafellar and
Uryasev (2000, 2002). The weight λ ∈ [0, 1] sets risk aversion (λ = 0 recovers a
risk-neutral stochastic program; λ = 1 is a pure-CVaR, tail-only objective), and
α ∈ (0, 1) is the CVaR confidence level, so CVaR_α(Z) is the mean loss over the
worst (1 − α) fraction of scenarios. The two linear-program baselines are limiting
cases of this risk spectrum: LP-Avg (capacities averaged over per-year deterministic
solves) approximates the risk-neutral end, and LP-Worst (capacities from the
highest-cost year) approximates the worst-case end.

We fix the risk parameters **a priori** at λ = 0.9 and α = 0.9 for all sites and
value-of-lost-load levels, rather than tuning them per scenario. Two considerations
motivate this. First, a high λ encodes the strongly risk-averse posture appropriate
to a resilience-critical forward operating base, where the cost of tail outages
dominates mission value; mean–CVaR weights are decision-maker preferences without a
universal convention and are conventionally reported through a sensitivity analysis
(Krokhmal et al. 2002; Conejo et al. 2010). Second, α = 0.9 is the highest confidence
that still averages more than one scenario at our sample size: with N ≈ 20 training
years the CVaR tail contains ⌈N(1 − α)⌉ scenarios, i.e. two years at α = 0.9 but only
one at α = 0.95, at which point CVaR degenerates to the single worst year and
coincides with the worst-case baseline (Rockafellar and Uryasev 2002). Because
tail-average (expected-shortfall) estimates require more samples than quantile (VaR)
estimates for comparable accuracy (Yamai and Yoshiba 2005; Sarykalin et al. 2008),
α = 0.9 is the statistically prudent choice for ~20 weather-year scenarios. Fixing
(λ, α) a priori also keeps SO-CVaR on equal footing with the parameter-free LP
baselines. We stress that λ = 0.9 is a *risk preference*, not a cost minimiser: because
the objective prices reliability through VoLL, the risk-neutral setting (λ = 0) minimises
expected out-of-sample cost, and the risk-averse setting trades a small mean-cost premium
for lower tail loss of load. The Supplementary Information confirms that the paper's
conclusions are robust to the exact (λ, α) — the out-of-sample cost surface is smooth and
fixing (0.9, 0.9) carries only a small, bounded premium over per-fold tuning — and
quantifies the tail-reliability that this premium buys [Fig. Sx].

---

## Supplementary Information — validation of the a-priori risk parameters

**Nested cross-validation.** To confirm that the fixed (λ, α) is defensible without
selecting parameters on the data used to report results, we evaluate the risk-parameter
grid λ ∈ {0, 0.25, 0.5, 0.75, 0.9, 1.0} × α ∈ {0.80, 0.90, 0.95} by nested
cross-validation, treating each five-year block of weather years as a group to avoid
temporal leakage. In the outer loop we hold out one block as a **test** set; within the
remaining twenty years an inner leave-one-block-out loop fits capacities on fifteen
**training** years and scores each (λ, α) on five **validation** years. The test block
is never used to choose parameters, so the per-fold performance of the
validation-selected parameters is an unbiased out-of-sample estimate (Varma and Simon
2006; Cawley and Talbot 2010). We assess each (λ, α) on out-of-sample total system cost —
the same criterion used to report the main results. The procedure is repeated for all
five climates and three VoLL levels (15 cells), so the choice is examined across the whole
design space rather than a single case. **This is a robustness check, not a selection
procedure:** the risk parameters remain fixed a priori.

**Results.** The assessment yields three findings [Fig. Sx]. (i) The pooled out-of-sample
cost surface — each cell's cost expressed as relative regret against its own minimum and
averaged across cells — is smooth and, as expected for a risk-averse objective, decreases
toward the risk-neutral corner: minimising expected *cost* favours λ = 0. We therefore do
**not** claim (λ, α) = (0.9, 0.9) is cost-optimal; it sits on a broad, flat basin only
≈1% above the cost minimum, so the choice is not knife-edge and the ranking of the three
methods is unchanged across the low-λ region. (ii) The unbiased nested-CV test cost of the
fixed (0.9, 0.9) exceeds that of per-fold cost-optimal selection by only ≈1.6% on average
(maximum ≈6.3%, in the coldest/high-VoLL cell) — a small, bounded premium — so the main
comparison is not advantaged by selecting parameters on the evaluation data. (iii) That
premium is the deliberate price of tail reliability: relative to the risk-neutral λ = 0,
the risk-averse setting reduces out-of-sample unserved energy by ≈34% (see the
loss-of-load figure), the quantity CVaR is designed to control. Because the LP baselines
have no tunable parameters, fixing (λ, α) a priori keeps all methods on the same footing.

*(Optional, if a reviewer presses on the specific value of λ: evaluated on a tail-cost
metric — the worst weather-year cost rather than the mean — the per-cell optima move into
the interior, λ ≈ 0.5–0.9, and λ = 1 is worse than the interior; i.e. on the resilience-
relevant objective the risk-averse choice is genuinely preferred. We report the more
conservative mean-cost view above and keep λ fixed a priori.)*

---

## References

- Cawley, G.C. and Talbot, N.L.C. (2010). On over-fitting in model selection and
  subsequent selection bias in performance evaluation. *Journal of Machine Learning
  Research*, 11, 2079–2107.
- Conejo, A.J., Carrión, M. and Morales, J.M. (2010). *Decision Making Under
  Uncertainty in Electricity Markets*. International Series in Operations Research &
  Management Science, vol. 153. Springer. doi:10.1007/978-1-4419-7421-1.
- Krokhmal, P., Palmquist, J. and Uryasev, S. (2002). Portfolio optimization with
  conditional value-at-risk objective and constraints. *Journal of Risk*, 4(2), 43–68.
  doi:10.21314/JOR.2002.057.
- Moazeni, S., Powell, W.B. and Hajimiragha, A.H. (2015). Mean-conditional value-at-risk
  optimal energy storage operation in the presence of transaction costs. *IEEE
  Transactions on Power Systems*, 30(3), 1222–1232. doi:10.1109/TPWRS.2014.2341642.
- Rockafellar, R.T. and Uryasev, S. (2000). Optimization of conditional value-at-risk.
  *Journal of Risk*, 2(3), 21–41. doi:10.21314/JOR.2000.038.
- Rockafellar, R.T. and Uryasev, S. (2002). Conditional value-at-risk for general loss
  distributions. *Journal of Banking & Finance*, 26(7), 1443–1471.
  doi:10.1016/S0378-4266(02)00271-6.
- Sarykalin, S., Serraino, G. and Uryasev, S. (2008). Value-at-risk vs. conditional
  value-at-risk in risk management and optimization. In *Tutorials in Operations
  Research*, INFORMS, 270–294. doi:10.1287/educ.1080.0052.
- Shapiro, A., Dentcheva, D. and Ruszczyński, A. (2021). *Lectures on Stochastic
  Programming: Modeling and Theory*, 3rd edn. SIAM. doi:10.1137/1.9781611976595.
- Varma, S. and Simon, R. (2006). Bias in error estimation when using cross-validation
  for model selection. *BMC Bioinformatics*, 7, 91. doi:10.1186/1471-2105-7-91.
- Yamai, Y. and Yoshiba, T. (2005). Value-at-risk versus expected shortfall: a practical
  perspective. *Journal of Banking & Finance*, 29(4), 997–1015.
  doi:10.1016/j.jbankfin.2004.08.010.

*Verify before submission:* the Rockafellar–Uryasev (2000) locator (2(3):21–41; some
indexers show vol. 3 / 21–42); Varma & Simon and Cawley & Talbot page/DOI details.

---

## Citation traceability (in-text citation → exact claim it supports)

| In-text citation | Supports this specific claim/number |
|---|---|
| Rockafellar & Uryasev 2000 | the CVaR minimisation formulation with the auxiliary VaR variable η (the LP our model implements) |
| Rockafellar & Uryasev 2002 | CVaR for discrete/finite-scenario losses; CVaR_α averages the worst ⌈N(1−α)⌉ scenarios → the α-vs-sample-size argument (2 tail years at α=0.9, 1 at α=0.95 for N≈20) |
| Yamai & Yoshiba 2005 | Expected Shortfall / CVaR needs larger samples than VaR for equal accuracy → α=0.9 is the statistically prudent tail for ~20 scenarios |
| Sarykalin et al. 2008 | conventional CVaR confidence levels (0.90–0.99); estimation-stability considerations |
| Krokhmal et al. 2002 | CVaR-as-objective and the mean–CVaR efficient frontier (basis for the λ weight; no universal λ convention) |
| Conejo et al. 2010 | power-systems risk-averse SP: E[cost] weighted against CVaR via a risk parameter, reported over a range/frontier |
| Varma & Simon 2006; Cawley & Talbot 2010 | nested cross-validation gives unbiased performance when hyperparameters are selected (the leakage rebuttal) |
| Moazeni et al. 2015 | an energy application explicitly formulated as mean-CVaR (precedent for the objective form) |
| Shapiro, Dentcheva & Ruszczyński | coherent/mean-risk measures and sample complexity of risk-averse SP (theory backing) |
