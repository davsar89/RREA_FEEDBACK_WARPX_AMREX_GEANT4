#!/usr/bin/env python3
"""FGF preset simulations and source-emission figures for the EGU example."""
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as phys
from scipy.integrate import cumulative_trapezoid
import fgf_module as FGF

# Lcloud (m), KEext, tau (s), solver, output samples.
# Sampling defaults are verified by check_convergence.py.
PRESETS = {
    'GRG': (3000., 1., 1., 'Radau', 100000),
    'FGF': (5000., 1., .01, 'Radau', 100000),
    'FGFstrong': (10000., 10., .01, 'LSODA', 100000),
    'TGF': (1000., 1., .01, 'LSODA', 100000),
    'TGF2': (300., 20., .01, 'LSODA', 399997),
    'TGF3': (1000., 20., .01, 'LSODA', 199999),
}


def configure_case(case):
    """Reset the physical and numerical settings that define one complete preset."""
    if case not in PRESETS:
        raise ValueError(f'Unknown case {case!r}; choose from {", ".join(PRESETS)}')
    FGF.Lcloud, FGF.KEext, FGF.tau, FGF.ode_method, FGF.num_time_points = PRESETS[case]
    FGF.preset_name = case
    FGF.Katm, FGF.E0Et, FGF.KJext = 1., 1.02, 0.
    FGF.Eext_shape, FGF.Jext_shape = 'sigmoid', 'gaussian'
    FGF.tau_cloud, FGF.S = 1., 10.
    FGF.l_ion_att, FGF.Tcloud, FGF.Mi = 3., 200., 62.
    FGF.use_ions, FGF.do_feedback, FGF.constant_delta = True, True, True
    FGF.instantaneous_NR = FGF.minimal_model = False
    FGF.manual_feedback_parameters = FGF.reload_Dwyer2007 = False
    FGF.debugging_plots = False
    FGF.plot_only_NR = FGF.plot_logNR = False
    FGF.plot_Nsec = FGF.plot_currents = FGF.plot_label_threshold_fields = True
    FGF.T1, FGF.T2 = -10*FGF.tau, 10*FGF.tau


def get_results(case, **solver_options):
    configure_case(case)
    return FGF.init_and_solve(**solver_options)


def main(case='FGF', output_dir=None):
    if output_dir is not None:
        FGF.figdir = str(Path(output_dir))
    values = get_results(case)
    t = values['t']
    fig = FGF.plot(values)
    FGF.save_figure(fig, ['pdf'])

    radio, filtered = FGF.get_Erad(t, values['Jtot'], D=10000.)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(t/1e-3, radio, label='Unfiltered')
    ax.plot(t/1e-3, filtered, label='Cloud-size filtered')
    ax.set(xlabel='Time (ms)', ylabel='Radio field at 10 km (V/m)', title=case)
    ax.legend()
    fig.tight_layout()
    FGF.save_figure(fig, ['pdf'], suffix='_Erad')

    gamma = FGF.get_Fgrays(values['NR'], values['E'], minTGFeV=300000.)
    count = cumulative_trapezoid(gamma, t, initial=0)
    fig, ax = plt.subplots()
    ax.plot(t/1e-3, count)
    ax.set(xlabel='Time (ms)', ylabel='Source photons above 300 keV')
    print(f'Integrated source gamma production: {count[-1]:.6g} photons')

    optical = FGF.get_337(values['E'], values['NR'])
    fig, ax = plt.subplots()
    irradiance = optical/(4*np.pi*10000.**2)*(phys.h*phys.c/337e-9)
    ax.plot(t/1e-3, irradiance)
    ax.set(xlabel='Time (ms)', ylabel='Optical irradiance (W/m²)',
           title='Nominal 337 nm, 10 km; isotropic, unattenuated estimate')
    print(f'Peak optical source production: {np.max(optical):.6g} photons/s')
    return values


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', nargs='?', default='FGF', choices=PRESETS)
    parser.add_argument('--output-dir')
    args = parser.parse_args()
    main(args.case, args.output_dir)
