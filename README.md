# DAC-to-SNG case study: f-TRFu on a direct-air-capture to synthetic-natural-gas flowsheet

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23173532.svg)](https://doi.org/10.5281/zenodo.23173532)

This repository reproduces the f-TRFu results of the **DAC-to-SNG case study,
Section 8.2** of the thesis: the run of Figure 8.2(a) (37 outer iterations, 20
f-type and 17 θ-type steps, no rejections) and the f-TRFu column of Table 8.1
(total annualised cost 4.26 million USD per year).

The flowsheet captures CO2 from ambient air on a dual-function material (DFM) and
methanates it in situ with hydrogen from solid oxide electrolysers. The DFM
capture-and-methanation unit is the **truth model**, embedded in an otherwise
glass-box Pyomo model through `ExternalFunction`s, with sensitivities obtained by
finite differences. f-TRFu builds the corrected Taylor surrogate of thesis (6.1)
from those values and sensitivities and poses the trust region on seven decision
variables only.

| Method | What it is | Implementation |
|---|---|---|
| **f-TRFu** | the feasible trust-region funnel method, Algorithm 6.1 | Pyomo's trust-region solver, `globalization_strategy='funnel'` |
| **s-TRFi** | the simplified trust-region filter method of Yoshio and Biegler | Pyomo's trust-region solver, `globalization_strategy='filter'` (optional, see below) |

f-TRFu is part of Pyomo from release 6.10.1 (`pyomo.contrib.trustregion`), so no
copy of the solver is kept here.

## Source of the model

The process model is adapted from the published superstructure formulation of

> M. Dolat, K. Keynejad, M. S. Duyar and M. Short, "Superstructure optimisation of
> direct air capture integrated with synthetic natural gas production," *Applied
> Energy*, vol. 384, p. 125413, 2025. https://doi.org/10.1016/j.apenergy.2025.125413

(see also M. Dolat, M. S. Duyar and M. Short, *Computer Aided Chemical Engineering*,
vol. 53, pp. 1213–1218, 2024, https://doi.org/10.1016/B978-0-443-28824-1.50203-9).
The published model fixes the DFM route, the adsorbent mass and the adsorption time
and selects the air-intake and hydrogen-production systems with binary variables.
This version fixes those selections at the published optimum (vane axial fan, solid
oxide electrolysers), keeps the continuous flowsheet, and represents the DFM unit as
an external truth model, so that the adsorption time and the DFM sizing become
optimisation variables. The grey-box formulation and its optimisation are those of

> G. Hameed, T. Chen, A. del Rio Chanona, L. T. Biegler and M. Short, "Optimisation
> of Synthetic Natural Gas Production via Direct Air Capture and Utilisation using
> Reduced Models under a Novel Trust-Region Funnel Method," *Systems and Control
> Transactions*, vol. 5, PSE Press, 2026, pp. 1551–1557.

Equipment costs follow Towler and Sinnott (2021) and Seider et al. (2016), as cited in
the model file. The problem has 378 variables, of which 9 are truth-model inputs and
10 truth-model outputs, with 208 equality and 39 inequality constraints.

## Layout

```
model/
  dac_to_sng.py        the flowsheet model (building it is importing it), the truth
                       model TRF_blackbox and its finite-difference sensitivities,
                       the trust-region decision variables and the surrogate basis
data/
  steam_table.csv      saturated-steam table used for the steam levels
experiments/
  run_case_study.py    solves the flowsheet with f-TRFu (or s-TRFi)
  make_figure.py       convergence plot of a run (Figure 8.2(a) for f-TRFu)
results/reference/     the reported results (see "Reference results")
environment.yml        the environment (Python 3.12, Pyomo 6.10.1, IPOPT)
```

## Requirements

```
conda env create -f environment.yml      # creates "dac_to_sng"
conda activate dac_to_sng
```

The run needs the IPOPT executable (the reported runs used IPOPT 3.14.13 with
MUMPS). On Linux and macOS the environment file installs it; on Windows the
conda-forge package provides only the library, so put an `ipopt.exe` on the `PATH`,
for example from the COIN-OR Ipopt releases on GitHub.

All commands below are run from the repository root.

## Reproducing the f-TRFu results

```
python experiments/run_case_study.py
python experiments/make_figure.py
```

`run_case_study.py` solves the flowsheet with f-TRFu from the initial point of the
model, with initial trust-region radius 1, and takes under a minute. It prints the
iteration count, the step types and the total annualised cost, and writes

- `results/funnel.log`, the solver's iteration record;
- `results/funnel_solution.csv`, the iteration and step counts and the quantities
  of the f-TRFu column of Table 8.1.

`make_figure.py` draws the convergence of that run (objective, infeasibility,
trust-region radius and step norm against the iteration, each step marked by its
type) to `results/funnel.png` and `.svg`.

Expected result: **37 iterations, 20 f-type and 17 θ-type steps, no rejections,
total annualised cost 4,259,085 USD per year**, with the DFM total capital cost
23,512,996 USD and the adsorption time at its lower bound of 500 s.

## Settings of the run

All parameters are the defaults of Pyomo's trust-region solver (Table A.1 of the
thesis), with one exception: the funnel contraction factor κ_fd
(`funnel_param_kappa_f`) is **0.75**, the value of the solver version that produced
the reported run. Pyomo 6.10.1 releases the method with κ_fd = 0.25. With that
value the run reaches the same solution, to within 0.01 USD per year, in 194
iterations instead of 37:

```
python experiments/run_case_study.py --kappa-fd 0.25
```

The trust region is imposed on seven variables: the adsorption time, the
temperature and hydrogen flow of the electrolyser product stream, the air flow
rate, the external heat supplied to the DFM unit, and the unit diameter and length (`DEGREES_OF_FREEDOM` in `model/dac_to_sng.py`). The
surrogate basis is zero, so the surrogate is the first-order corrected Taylor
expansion of the truth model.

## Testing the filter (s-TRFi), optionally

The thesis compares f-TRFu with s-TRFi on this flowsheet. s-TRFi reaches the same
solution but takes **40,153 iterations** in the reported run, settling into a limit
cycle of θ-type and f-type steps (Section 8.2.2), so its run takes a long time and
is not part of the reproduction. To run it:

```
python experiments/run_case_study.py --strategy filter
python experiments/make_figure.py --log results/filter.log
```

The iteration budget for the filter is 100,000 (`--max-iter` changes it), and
`--trust-radius` sets the initial radius; the thesis also reports the run with
radius 100, which takes 40,330 iterations.

## Reference results

`results/reference/` holds the reported run:

- `funnel_thesis_run.log` is the log of the run reported in the thesis, made with
  the trust-region solver of the working code before its release in Pyomo;
- `funnel.log` and `funnel_solution.csv` are the same run reproduced with Pyomo
  6.10.1 by the commands above. The two logs agree iteration by iteration in step
  type and trust-region radius, and in the objective to the precision of the
  finite-difference sensitivities;
- `funnel.png`, `funnel_thesis_run.png` (and `.svg`) are their convergence plots.

To draw the plot from the reference run instead of a new one:

```
python experiments/make_figure.py --log results/reference/funnel.log
```

## Citation

Release 1.0.0 of this repository accompanies G. Hameed, *Rigorous Trust-Region Algorithms for Surrogate-based Grey-box Optimisation*, PhD thesis, University of Surrey, 2026; publications that use later releases are listed here as they appear. Please cite the release you used with the metadata in `CITATION.cff`; each release has its own Zenodo DOI. Release 1.0.0: [10.5281/zenodo.23173533](https://doi.org/10.5281/zenodo.23173533); all releases: [10.5281/zenodo.23173532](https://doi.org/10.5281/zenodo.23173532).

## Licence

The code in this repository is released under the MIT licence (`LICENSE`); the published model it is adapted from remains the work of its authors (see "Source of the model").
