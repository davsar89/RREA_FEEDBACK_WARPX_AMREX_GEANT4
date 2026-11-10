#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 17:00:16 2024

Optical and x-ray emission in SPM

@author: nle003
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.integrate import solve_ivp
import bethe
import scipy.constants as phys
from model_helpers import ave, latex_float


def streamer_model():
    """Load the optional scientific model only when a streamer calculation is requested."""
    try:
        import streamer_parameters_v1_9_3 as SPM
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError('Streamer example requires streamer_parameters_v1_9_3.py') from exc
    return SPM


def solve_runaway_transport(xi, growth_rate, source_rate, velocity):
    """Solve velocity*dn/dxi = growth_rate*n + source_rate with zero inflow.

    xi increases in metres; growth_rate is s^-1, source_rate is m^-3 s^-1,
    velocity is a nonzero constant in m/s. Tabulated rates are interpolated
    linearly. Output density (m^-3) follows the original xi order.
    No external streamer module or data are needed for this equation.
    """
    xi = np.asarray(xi, dtype=float)
    growth, source = np.broadcast_arrays(growth_rate, source_rate, xi)[:2]
    if (xi.ndim != 1 or xi.size < 2 or not np.all(np.isfinite(xi))
            or not np.all(np.diff(xi) > 0) or growth.shape != xi.shape
            or not np.all(np.isfinite(growth)) or not np.all(np.isfinite(source))
            or np.any(source < 0) or not np.isfinite(velocity) or velocity == 0):
        raise ValueError('Require increasing finite xi, finite rates, nonnegative source and nonzero velocity')
    path = xi if velocity > 0 else xi[::-1]
    def rhs(x, density):
        return (np.interp(x, xi, growth)*density + np.interp(x, xi, source))/velocity
    sol = solve_ivp(rhs, (path[0], path[-1]), [0.], t_eval=path,
                    method='DOP853', rtol=1e-10, atol=1e-12,
                    max_step=np.min(np.diff(xi)))  # do not step over a localized tabulated source
    if not sol.success or not np.all(np.isfinite(sol.y)) or np.any(sol.y < 0):
        raise RuntimeError(f'Runaway transport failed: {sol.message}')
    return sol.y[0] if velocity > 0 else sol.y[0, ::-1]


# The emission rate is I = A/(A + Q[N2]*N[N2] + Q[O2]*O2) R = C*R where
# A = 2.38e7 s^(-1), Q[N2] = 0.13e-16 m^3s^-1, Q[O2] = 3e-16 m^3s^-1 [Bonaventura et al, 2011]

# C = 1e-2 at sea level
# C = 5e-2 at 10 km altitude

#%% The townsend fit
def getNm(hkm):
    return 2.5e25*np.exp(-hkm/7.64)

class Line:
    def __init__(self, A, Q):
        self.A = A
        self.Q = Q
    def tau(self, hkm):
        return 1/(self.A + self.Q*getNm(hkm))
    def quenching(self, hkm):
        return self.A*self.tau(hkm)
    def excite(self, E):
        raise NotImplementedError

line2PN2 = Line(A=2.38e7, Q=0.2*3e-16 + 0.8*0.13e-16)
line1PN2 = Line(A=1.67e5, Q=5.2e-17)
line1NN2p = Line(A=1.6e7, Q=2.7e-16)

def nu2PN2_fun(E):
    SPM = streamer_model()
    return SPM.mue_fun(E)*SPM.townsend(E, 4.73e12, 14.6e6)
line2PN2.excite = nu2PN2_fun
def nu1NN2p_fun(E):
    raise NotImplementedError('N2+ first-negative excitation has no implemented rate')
line1NN2p.excite = nu1NN2p_fun

def get_2PN2(plot=False, **params):
    SPM = streamer_model()
    Ee = params['Ee']; L=params['L']; sign=params['sign']
    a = params['a']; V = params['V']; d = params['d'];
    dopt = V*line2PN2.tau(0);
    xib = np.linspace(-5*dopt,20*d, 1001)
    xic = ave(xib)
    E,n = SPM.E_n_streamer_smooth(xic, **params)
    dxi = np.diff(xib)
    tmp = np.r_[0., np.cumsum(np.exp(-xic/dopt)*line2PN2.excite(E)*n/V*dxi)]
    nN2_C = np.exp(xib/dopt)*(tmp[-1]-tmp) # concentration of excited species in m^-3
    emission_rate_2PN2 = nN2_C*line2PN2.A
    total2PN2 = np.sum(ave(emission_rate_2PN2)*dxi)*np.pi*a**2 # photons per second
    if plot:
        plt.figure()
        plt.subplot(2,1,1)
        Emax = np.max(E)
        nmax = params['ns']
        plt.plot(xic/1e-2, E/Emax, label='$E/E_m$')
        plt.plot(xic/1e-2, n/nmax, label='$n/n_s$')
        plt.grid(True)
        plt.legend()
        plt.ylabel('$E(x), n(x)$')
        title1 = SPM.sign_str(sign)+', $E_e='+latex_float(Ee/1e6)+'$ MV/m, $L=' + latex_float(L/1e-2) + '$ cm:'
        title2 = '$E_m='+latex_float(Emax/1e6,digits=3)+'$ MV/m, $n_s='+latex_float(nmax,digits=3)+'$ m$^{-3}$'
        plt.title(title1+'\n'+title2)
        # The rate
        plt.subplot(2,1,2)
        plt.plot(xib/1e-2, emission_rate_2PN2)
        plt.grid(True)
        plt.ylabel(r'I(2PN2), m$^{-3}$s$^{-1}$')
        plt.xlabel('x, cm')
    return total2PN2

def main():
    SPM = streamer_model()
    data = Path(__file__).with_name('ExcitationCoef_N2_C3Pi_u.txt')
    if not data.is_file():
        raise FileNotFoundError(f'Streamer excitation table is missing: {data}')
    table = np.loadtxt(data, delimiter=',')
    plt.semilogy(table[:, 0], table[:, 1])
    results = SPM.calculate_streamer_modes(sign=-1, Ee=1.5e6, L=80e-3)
    params = SPM.apply_maxV(**results)
    flux = get_2PN2(plot=True, **params)*1e-4/(4*np.pi*1e4**2)
    a, V, d, sign = (params[key] for key in ('a', 'V', 'd', 'sign'))
    xi = np.r_[np.linspace(-10*d, 10*d, 1001)[:-1], np.linspace(10*d, 2*a, 1001)]
    E, n = SPM.E_n_streamer_smooth(xi, **params)
    vr = 0.9*phys.c
    Vcor = -V-sign*vr
    nr = solve_runaway_transport(xi, bethe.nu_rrea_Dwyer(E), bethe.nu_bakhov(E)*n, Vcor)
    outflow = nr[-1] if Vcor > 0 else nr[0]
    # Flux relative to the moving integration boundary, in particles/s.
    src = abs(Vcor)*outflow*np.pi*a**2
    plt.figure()
    plt.plot(xi, nr)
    plt.xlabel('Position (m)')
    plt.ylabel('Runaway density (m^-3)')
    return flux, src


if __name__ == '__main__':
    main()
