# A-priori choice of the SO-CVaR risk parameters (λ, α)

This note justifies fixing the two risk parameters of the mean-CVaR capacity model
**a priori** (before looking at the results), so the SO-CVaR comparison against the
LP baselines and diesel is not advantaged by tuning on the evaluation data. The
nested-CV sweep (SI Fig S3, `si_run_risk_sweep.py`) is a **sensitivity / robustness
analysis** of that fixed choice, not a selection step.

## Formulation
Following Rockafellar & Uryasev, the second-stage objective is a mean-CVaR
combination of the outage cost `Z_s` over the N training-year scenarios:

```
min  capital  +  (1 − λ)·(1/N)·Σ_s Z_s  +  λ·[ η + (1 / ((1 − α)·N))·Σ_s ξ_s ]
```

where `η` (VaR) and `ξ_s ≥ Z_s − η` linearize `CVaR_α(Z)`. `λ = 0` is expected-cost
(risk-neutral) stochastic optimization; `λ = 1` is pure CVaR. So SO-CVaR spans a
**risk spectrum** whose endpoints are approached by the two LP baselines
(LP-Avg ≈ risk-neutral, LP-Worst ≈ worst-case), with (λ, α) selecting the interior.

## Why α = 0.9
- **Convention.** CVaR confidence levels of 0.90–0.99 are the entrenched norm in
  operations research / finance (Sarykalin et al. 2008) and in energy-system CVaR
  planning (e.g. Xuan et al. 2021; Greenough et al. 2024 use CVaR₀.₉₅).
- **Sample size (the decisive argument).** With a finite, equiprobable scenario set,
  `CVaR_α` is the average of the worst `⌈N(1−α)⌉` scenarios (Rockafellar & Uryasev
  2002). With N = 20 training years:
  - α = 0.90 → worst **2** years averaged (a genuine tail mean);
  - α = 0.95 → worst **1** year → `CVaR₀.₉₅` collapses to the single worst scenario,
    i.e. a worst-case/max objective — no longer distinct from LP-Worst;
  - α = 0.80 → worst 4 years.
  Tail-average (Expected-Shortfall/CVaR) estimates already need more samples than VaR
  for the same accuracy (Yamai & Yoshiba 2005). So **α = 0.9 is the highest confidence
  that still averages more than one scenario** — the most defensible choice at this
  sample size; higher α degenerates toward worst-case and is statistically fragile.

## Why λ = 0.9
- λ is a **risk-preference weight**, and the literature does **not** prescribe a
  universal value — it is conventionally reported as a sensitivity sweep / mean-CVaR
  efficient frontier (Krokhmal et al. 2002; Conejo et al. 2010). A small (1 − λ)
  expected-cost term is retained (rather than pure λ = 1) to break ties and avoid the
  degeneracy of a pure worst-case posture.
- λ = 0.9 is a **strongly risk-averse** setting, appropriate for a resilience-critical
  forward operating base where tail outages dominate the mission cost. Because there
  is no convention, we justify it as an explicit preference **and** show (SI Fig S3)
  that the result is robust: the validation-cost basin is flat around (0.9, 0.9) and
  the fixed choice matches per-fold cross-validated selection to within fold noise.

## Reviewer-facing points
- **"You selected (λ, α) on the same data used to evaluate."** No — the parameters are
  fixed a priori; SO-CVaR therefore has no data-tuned knob, on equal footing with the
  parameter-free LP baselines. The nested-CV sweep additionally shows the fixed choice
  performs the same (within fold SD) as parameters selected by inner-CV that never see
  the test fold, so no selection advantage is possible (Varma & Simon 2006; Cawley &
  Talbot 2010 on nested CV for unbiased selection).
- **"Why not α = 0.95?"** With ~20 scenarios it leaves <2 years in the tail, collapsing
  CVaR toward the worst-case LP and destabilizing the estimate (above).
- **"Is 0.9 special?"** No — Fig S3(a) shows a flat regret basin; (b) shows fixing it
  costs ≈0 vs tuning; (c) shows per-cell optima cluster near it.

## References (verified; check flagged items before final submission)
1. Rockafellar, R.T. & Uryasev, S. (2000). *Optimization of Conditional Value-at-Risk.*
   Journal of Risk **2(3)**, 21–41. doi:10.21314/JOR.2000.038. *(Some indexers show
   "vol. 3"/"21–42"; canonical is 2(3):21–41.)* — foundational CVaR LP with the VaR
   auxiliary variable.
2. Rockafellar, R.T. & Uryasev, S. (2002). *Conditional value-at-risk for general loss
   distributions.* Journal of Banking & Finance **26(7)**, 1443–1471.
   doi:10.1016/S0378-4266(02)00271-6. — discrete/finite-scenario CVaR; basis for the
   `⌈N(1−α)⌉` tail-count / small-sample degeneracy.
3. Krokhmal, P., Palmquist, J. & Uryasev, S. (2002). *Portfolio optimization with CVaR
   objective and constraints.* Journal of Risk **4(2)**, 43–68. doi:10.21314/JOR.2002.057.
   — CVaR-as-objective and the **mean–CVaR frontier** (basis for the λ weight).
4. Sarykalin, S., Serraino, G. & Uryasev, S. (2008). *VaR vs. CVaR in Risk Management
   and Optimization.* INFORMS Tutorials in OR, 270–294. doi:10.1287/educ.1080.0052.
   — conventional α levels; estimation-stability considerations.
5. Yamai, Y. & Yoshiba, T. (2005). *Value-at-risk versus expected shortfall: a practical
   perspective.* Journal of Banking & Finance **29(4)**, 997–1015.
   doi:10.1016/j.jbankfin.2004.08.010. — Expected Shortfall needs larger samples than
   VaR (small-sample caveat).
6. Conejo, A.J., Carrión, M. & Morales, J.M. (2010). *Decision Making Under Uncertainty
   in Electricity Markets.* Springer, vol. 153. doi:10.1007/978-1-4419-7421-1.
   — power-systems risk-averse SP: E[cost] weighted against CVaR via a risk parameter,
   explored over a range/frontier.
7. Moazeni, S., Powell, W.B. & Hajimiragha, A.H. (2015). *Mean-CVaR Optimal Energy
   Storage Operation…* IEEE Trans. Power Systems **30(3)**, 1222–1232.
   doi:10.1109/TPWRS.2014.2341642. — energy application explicitly formulated as mean-CVaR.
8. Shapiro, A., Dentcheva, D. & Ruszczyński, A. *Lectures on Stochastic Programming*,
   SIAM (2009/2014/2021). — coherent/mean-risk measures; sample complexity of risk-averse SP.
9. Xuan, A., Shen, X., Guo, Q. & Sun, H. (2021). *A CVaR-based planning model for
   integrated energy system…* Applied Energy **294**, 116971.
   doi:10.1016/j.apenergy.2021.116971. — CVaR IES investment+operation planning (close
   structural analogue). *Exact case-study α not verified to the digit — check before citing a value.*
10. Greenough, R. et al. (2024). *Wildfire Resilient Unit Commitment under Uncertain
    Demand.* arXiv:2403.09903. — energy SP fixing CVaR₀.₉₅ with a mean-CVaR weight.
    *Preprint — verify the published version if a peer-reviewed cite is required.*

(Nested-CV bias references for the rebuttal: Varma & Simon 2006, *BMC Bioinformatics*
7:91; Cawley & Talbot 2010, *JMLR* 11:2079–2107 — verify before citing.)
