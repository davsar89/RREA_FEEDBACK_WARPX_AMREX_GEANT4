#!/usr/bin/env python3
"""Separate solver accuracy from output sampling for the six FGF presets.

Writes observable summaries, not large solution arrays. Acceptance is 0.5%
for each amplitude/yield and two fine-grid steps for chronological NR peaks
above 1% of each run's maximum. Two successive sampling refinements must pass.
"""
import argparse
import json
import platform
from pathlib import Path
import numpy as np
import scipy
from scipy.signal import find_peaks
import fgf_module as FGF
from egu2025poster import PRESETS, configure_case

OBSERVABLES = ('NR_peak', 'radio_peak', 'filtered_radio_peak', 'gamma_total', 'optical_total')


def measure(sol, samples):
    t = np.linspace(sol.t[0], sol.t[-1], samples)
    v = FGF.sol2values(sol, t=t)
    radio, filtered = FGF.get_Erad(t, v['Jtot'])
    gamma = FGF.get_Fgrays(v['NR'], v['E'])
    optical = FGF.get_337(v['E'], v['NR'])
    peaks, _ = find_peaks(v['NR'], height=.01*np.max(v['NR']))
    return dict(samples=samples, dt=float(t[1]-t[0]), NR_peak=float(np.max(v['NR'])),
                radio_peak=float(np.max(abs(radio))), filtered_radio_peak=float(np.max(abs(filtered))),
                gamma_total=float(np.trapezoid(gamma, t)), optical_total=float(np.trapezoid(optical, t)),
                peak_times=t[peaks].tolist(), dominant_peak_time=float(t[np.argmax(v['NR'])]))


def compare(candidate, reference):
    errors = {key: abs(candidate[key]-reference[key])/max(abs(reference[key]), 1e-300)
              for key in OBSERVABLES}
    a, b = np.array(candidate['peak_times']), np.array(reference['peak_times'])
    timing = float(np.max(abs(a-b), initial=0)) if len(a) == len(b) else None
    step = min(candidate['dt'], reference['dt'])
    return dict(relative_errors=errors, max_peak_time_error_s=timing,
                peak_counts=[len(a), len(b)],
                dominant_peak_shift_s=candidate['dominant_peak_time']-reference['dominant_peak_time'],
                passed=bool(max(errors.values()) < .005 and timing is not None and timing <= 2.000001*step))


def check_case(case):
    configure_case(case)
    FGF.num_time_points = 100000  # stored solver output; refinement below uses its dense solution
    FGF.init()
    method = FGF.ode_method
    production, elapsed = FGF.solve_odes()
    tight, tight_elapsed = FGF.solve_odes(rtol=1e-11, atol=np.array(FGF.DEFAULT_ATOL)/100)
    FGF.ode_method = 'Radau' if method == 'LSODA' else 'LSODA'
    independent, independent_elapsed = FGF.solve_odes(rtol=1e-11, atol=np.array(FGF.DEFAULT_ATOL)/100)
    # Bound internal steps as a separate check against missed dynamics.
    bounded, bounded_elapsed = FGF.solve_odes(rtol=1e-11, atol=np.array(FGF.DEFAULT_ATOL)/100,
                                            max_step=FGF.tau/100)
    FGF.ode_method = method
    summaries, refinements = [], []
    samples = 100000
    selected = None
    for _ in range(7):
        summaries.append(measure(tight, samples))
        if len(summaries) >= 2:
            refinements.append(compare(summaries[-2], summaries[-1]))
        print(case, 'samples', samples, 'radio peak', summaries[-1]['radio_peak'],
              'sampling passed', refinements[-1]['passed'] if refinements else None, flush=True)
        if len(refinements) >= 2 and all(row['passed'] for row in refinements[-2:]):
            selected = summaries[-3]['samples']
            break
        samples = 2*samples-1
    fine = summaries[-1]
    accuracy = {label: compare(measure(solution, fine['samples']), fine) for label, solution in
                [('production_vs_tight', production), ('independent_vs_tight', independent),
                 ('bounded_steps_vs_tight', bounded)]}
    return dict(method=method, selected_samples=selected,
                runtime_s=dict(production=elapsed, tight=tight_elapsed,
                               independent=independent_elapsed, bounded_steps=bounded_elapsed),
                sampling=summaries, sampling_comparisons=refinements, accuracy=accuracy,
                passed=selected is not None and all(row['passed'] for row in accuracy.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cases', nargs='*', help='Preset names; omitted means all six')
    parser.add_argument('--output', default='convergence.json')
    args = parser.parse_args()
    results = dict(environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
                   acceptance=dict(relative_observable_error=.005, peak_timing_fine_steps=2), cases={})
    for case in args.cases or PRESETS:
        results['cases'][case] = check_case(case)
        Path(args.output).write_text(json.dumps(results, indent=2)+'\n')
    if not all(case['passed'] for case in results['cases'].values()):
        raise SystemExit('Convergence criteria failed; see '+args.output)


if __name__ == '__main__':
    main()
