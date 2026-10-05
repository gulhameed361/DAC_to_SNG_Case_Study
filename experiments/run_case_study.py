# -*- coding: utf-8 -*-
"""Solve the DAC-to-SNG flowsheet (thesis Section 8.2) with Pyomo's trust-region solver.

    python experiments/run_case_study.py                      # f-TRFu, as reported
    python experiments/run_case_study.py --strategy filter    # s-TRFi (long; see README)

f-TRFu is `globalization_strategy='funnel'`, s-TRFi is `'filter'`, both in
`pyomo.contrib.trustregion` from Pyomo 6.10.1, at their default parameters. The run log
(the solver's verbose iteration record) is written to results/<strategy>.log, and a
summary of the solution to results/<strategy>_solution.csv.
"""
import argparse
import contextlib
import csv
import io
import os
import re
import sys
import time

import pyomo.environ as pyo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'model'))
RESULTS = os.path.join(ROOT, 'results')


class _Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def step_counts(log_text):
    """Iterations and step types from the solver's verbose log."""
    iters = len(re.findall(r'\*{6} Iteration (\d+) \*{6}', log_text)) - 1
    kinds = re.findall(r'INFO: (f-type|theta-type) step|INFO: step (rejected)', log_text)
    kinds = [a or b for a, b in kinds]
    return iters, {k: kinds.count(k) for k in ('f-type', 'theta-type', 'rejected')}


def solution_rows(m):
    """The quantities of thesis Table 8.1 (f-TRFu column) and the headline numbers."""
    v = pyo.value
    out = m.trf_out
    rows = [
        ('Total annualised cost (objective)', 'USD/year', v(m.TAC)),
        ('VAF total capital cost', 'USD', v(m.CAPEX_FAN)),
        ('VAF operational cost', 'USD/year', v(m.OPEX_FAN)),
        ('SOEL total capital cost', 'USD', v(m.CAPEX_SOEL)),
        ('SOEL operational cost', 'USD/year', v(m.OPEX_SOEL)),
        ('DFM total capital cost', 'USD', v(out['CAPEX_DFM'])),
        ('DFM operational cost', 'USD/year', v(out['OPEX_DFM'])),
        ('DFM units in each train', '-', v(out['N_react'])),
        ('DFM unit size (each)', 'm3', v(out['S_react'])),
        ('Air velocity inside channels', 'm/s', v(out['v'])),
        ('Unit diameter', 'm', v(m.D)),
        ('Unit height/length', 'm', v(m.L)),
        ('Pressure drop in each unit', 'Pa', v(out['DP'])),
        ('Total adsorbent weight', 'kg', v(out['DFM_Weight'])),
        ('Adsorption time', 's', v(m.t_ads)),
        ('Net CO2 uptake', 'kmol/kg', v(out['q_dfm'])),
        ('Boiler total capital cost', 'USD', v(m.CAPEX_Boiler)),
        ('Air-cooler total capital cost', 'USD', v(m.CAPEX_Aircooler)),
        ('Air-cooler operational cost', 'USD/year', v(m.OPEX_Aircooler)),
        ('Separator total capital cost', 'USD', v(m.CAPEX_Separator)),
        ('Pressure vessel total capital cost', 'USD', v(m.CAPEX_PVessel)),
        ('Profit (product and steam)', 'USD/year', v(m.Profit)),
    ]
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--strategy', choices=['funnel', 'filter'], default='funnel')
    ap.add_argument('--trust-radius', type=float, default=1.0,
                    help='initial trust-region radius (thesis: 1.0)')
    ap.add_argument('--kappa-fd', type=float, default=0.75,
                    help='funnel width contraction kappa_fd (funnel_param_kappa_f); the reported '
                         'run used 0.75, the Pyomo default is 0.25 (see README)')
    ap.add_argument('--max-iter', type=int, default=None,
                    help='iteration budget (default 500 for funnel, 100000 for filter)')
    args = ap.parse_args()
    max_it = args.max_iter or (500 if args.strategy == 'funnel' else 100000)

    import dac_to_sng as model_module          # builds the model on import
    m = model_module.m

    os.makedirs(RESULTS, exist_ok=True)
    log_path = os.path.join(RESULTS, f'{args.strategy}.log')
    solver = pyo.SolverFactory('trustregion', globalization_strategy=args.strategy,
                               trust_radius=args.trust_radius,
                               maximum_iterations=max_it, verbose=True)
    solver.config.solver = 'ipopt'
    if args.strategy == 'funnel':
        solver.config.funnel_param_kappa_f = args.kappa_fd

    buf = io.StringIO()
    t0 = time.time()
    with open(log_path, 'w', encoding='utf-8') as log, \
            contextlib.redirect_stdout(_Tee(log, buf)):
        # the solver works on a clone of m and returns it with the solution
        m = solver.solve(m, model_module.DEGREES_OF_FREEDOM(m),
                         ext_fcn_surrogate_map_rule=model_module.basis_rule)
    wall = time.time() - t0

    iters, kinds = step_counts(buf.getvalue())
    rows = solution_rows(m)
    sol_path = os.path.join(RESULTS, f'{args.strategy}_solution.csv')
    with open(sol_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['quantity', 'unit', 'value'])
        w.writerow(['Outer trust-region iterations', '-', iters])
        for k, n in kinds.items():
            w.writerow([f'{k} steps', '-', n])
        w.writerow(['Wall time', 's', round(wall, 1)])
        w.writerows(rows)

    print(f'\n{args.strategy}: {iters} iterations, steps {kinds}, '
          f'TAC = {pyo.value(m.TAC):,.0f} USD/year, {wall:.0f} s')
    print(f'log: {log_path}\nsolution: {sol_path}')


if __name__ == '__main__':
    main()
