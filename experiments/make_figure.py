# -*- coding: utf-8 -*-
"""Convergence plot of a run (thesis Figure 8.2(a) for the funnel run).

    python experiments/make_figure.py                         # results/funnel.log
    python experiments/make_figure.py --log results/reference/funnel.log

Reads the solver's verbose log and plots the objective, the infeasibility, the
trust-region radius and the step norm against the outer iteration, with the step type
of each iteration marked. Writes <log>.png and <log>.svg next to the log.
"""
import argparse
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MARK = {'f-type': ('o', 'tab:green'), 'theta-type': ('s', 'tab:blue'),
        'rejected': ('x', 'tab:red'), '': ('.', 'black')}


def parse(path):
    text = open(path, encoding='utf-8', errors='ignore').read()
    parts = re.split(r'\*{6} Iteration (\d+) \*{6}', text)[1:]
    rows = []
    for i in range(0, len(parts), 2):
        block = parts[i + 1]

        def num(key):
            mt = re.search(key + r' = ([-\d.eE+]+)', block)
            return float(mt.group(1)) if mt else float('nan')
        kind = re.search(r'(f-type|theta-type) step|step (rejected)|step type: (\S+)', block)
        kind = next((g for g in kind.groups() if g), '') if kind else ''
        rows.append((int(parts[i]), num('objectiveValue'), num('feasibility'),
                     num('trustRadius'), num('stepNorm'), kind))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', default=os.path.join(ROOT, 'results', 'funnel.log'))
    a = ap.parse_args()
    rows = parse(a.log)
    k = [r[0] for r in rows]
    series = [('Objective (USD/year)', 1, False), ('Infeasibility θ', 2, True),
              ('Trust-region radius', 3, True), ('Step norm', 4, True)]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
    for ax, (label, col, log) in zip(axes.flat, series):
        pts = [(r[0], r[col]) for r in rows if not (log and r[col] <= 0)]
        ax.plot(*zip(*pts), color='0.6', lw=1)
        for kind, (mk, c) in MARK.items():
            pts = [(r[0], r[col]) for r in rows if r[5] == kind and r[0] > 0]
            if pts:
                ax.scatter(*zip(*pts), marker=mk, color=c, s=18, zorder=3,
                           label=kind.replace('theta', 'θ') if kind else None)
        if log:
            ax.set_yscale('log')
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel('Iteration')
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    stem = os.path.splitext(a.log)[0]
    for ext in ('png', 'svg'):
        fig.savefig(f'{stem}.{ext}', dpi=300)
    print(f'wrote {stem}.png and {stem}.svg')


if __name__ == '__main__':
    main()
