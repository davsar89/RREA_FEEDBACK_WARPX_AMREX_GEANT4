#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 17:58:44 2024

A very simple model of FGF

Version 16: Added calculations of gamma rays, radio and optical emissions

@author: nle003
"""

import os
import time
import pickle
import warnings
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.integrate import solve_ivp
from scipy.special import exprel
from model_helpers import ave, neat_float, latex_float
from bethe import bremsstrahlung_3BN
import scipy.constants as phys


version = 16

# Plotting layout


def fix_margins(fig):
    "Set the same absolute margins as for a default figure size"
    width, height = fig.get_size_inches()
    dwidth, dheight = mpl.rcParams['figure.figsize']
    rhoriz = dwidth/width
    rvert = dheight/height
    left = rhoriz*mpl.rcParams['figure.subplot.left']
    bottom = rvert*mpl.rcParams['figure.subplot.bottom']
    right = 1 - rhoriz*(1 - mpl.rcParams['figure.subplot.right'])
    top = 1 - rvert*(1 - mpl.rcParams['figure.subplot.top'])
    fig.subplots_adjust(left=left, bottom=bottom, right=right, top=top)

# Subroutines from SPM

def sigmoid(x):
    "A well-known function"
    x = np.asarray(x)
    sw = np.zeros(x.shape) # switch wave
    ep = np.exp(-x[x>0])
    em = np.exp(x[x<=0])
    sw[x>0] = 1/(1+ep)
    sw[x<=0] = em/(1+em)
    return sw[()]


# The chemistry functions
N0 = phys.physical_constants['Loschmidt constant (273.15 K, 101.325 kPa)'][0] # Loschmidt constant 2.687e25
Katm = 1 # Air number density relative to N0; fitted feedback supports only Katm=1.

def mue_fun(E):
    "Mobility"
    if np.asarray(E<0).any(): raise ValueError('E<0')
    Ered = E/Katm
    return (Ered/3e6)**(-.17)*0.044/Katm

def vd_fun(E):
    "Drift velocity"
    return mue_fun(E)*E

def townsend(E,ax,Ex):
    "Townsend approximation"
    return ax*np.exp(-Ex/np.abs(E))

def alphaa_fun(E):
    "Spatial attachment rate, two body (in Townsend approximation) and three-body"
    Ered = E/Katm # sea-level equivalent
    eta3 = 4.7778e-69*N0**2*((Ered/N0+2e-23)*1e4)**(-1.2749)
    return Katm*townsend(Ered,1.7e3,3.2e6) + Katm**2*eta3

# Derived functions
def nua_fun(E):
    return alphaa_fun(E)*vd_fun(E)


# Preference constants, with given recommended defaults
# By how much to scale
KJext = 0         # play with this dimensionless number 0.1-1e4, there are different pulses!
KEext = 1
Jext_shape = 'gaussian'
Eext_shape = 'sigmoid'
tau = 1         # the enhancement time scale
T1 = -10*tau         # how many seconds to relax the system before enhancement
T2 = 10*tau
E0Et = 1.02          # =E0/Et; Aim for E0 in the relaxed state. Probably good idea to have E0/Et<1
Lcloud = 3e3        # runaway region vertical size, somewhat arbitrary. Longer L -> smoother pulses
# Note that Lcloud also determines the default transverse area Aperp = Lcloud**2 for calculations of total values
tau_cloud = 1.0      # Maxwellian relaxation time for background cloud, around 1-10 seconds, can vary
S = 1e1              # cosmic ray source of runaway electrons, per cubic meter per second, default = 10
use_ions = True      # default = True
# The feedback description preferences
do_feedback = True         # default = True
instantaneous_NR = False    # default = False
# Used only without feedback; computes the no-feedback steady density.
constant_delta = True       # False is unsupported; see init2_particles.

# Plotting preferences
plot_only_NR = False
plot_logNR = False
plot_Nsec = True
plot_currents = True
Eslack = 2e4
plot_label_threshold_fields = True
do_save_figures = False
figdir = os.path.join('fgf','v'+str(version))
preset_name = None

# Feedback calibration and diagnostic plotting
manual_feedback_parameters = False
reload_Dwyer2007 = False     # for True, must have file 'Dwyer2007_Fig3_L_Egamma1.txt'
debugging_plots = False
# Choose a solver
ode_method = 'Radau' # Preset choices and convergence evidence: egu2025poster.py, RESULTS.md


# The "minimal" model which shows oscillations -- only ion currents and ions are just accumulated (eta=0)
minimal_model = False

def get_figpath():
    def _neat(v): return str(neat_float(v))
    basefname =  'fgf_Lcloud' + _neat(Lcloud) + 'm' + \
    ('_E' + Eext_shape + 'KE' + _neat(KEext) if KEext!=0 else '') + \
    ('_J' + Jext_shape + 'KJ' + _neat(KJext) if KJext!=0 else '') + \
    '_tau' + _neat(tau/1e-3) + 'ms' + ('' if do_feedback else '_nofeedback') + \
        ('_instantNR' if instantaneous_NR else '') + ('_logNR' if plot_logNR else '') + \
            ('_onlyNR' if plot_only_NR else '')
    basefname += '_Katm' + _neat(Katm) + '_E0Et' + _neat(E0Et)
    if preset_name:
        basefname = preset_name + '_' + basefname
    return os.path.join(figdir, basefname)

def save_figure(fig, formats=(), *, suffix=""):
    if len(formats)==0:
        return
    figpath = get_figpath() + suffix
    Path(figpath).parent.mkdir(parents=True, exist_ok=True)
    print('Saving figure to', figpath)
    if 'pdf' in formats:
        fig.savefig(figpath + '.pdf')
    if 'png' in formats:
        fig.savefig(figpath + '.png')
    if 'pkl' in formats:
        with open(figpath + '.pkl', 'wb') as stream:
            pickle.dump(fig, stream)

def init1_prefs():
    global do_feedback, instantaneous_NR
    if minimal_model:
        # Simplify other stuff
        if do_feedback:
            print('Warning:', 'removing feedback for the minimal model!')
        do_feedback = False
        instantaneous_NR = True

    if do_feedback:
        if instantaneous_NR:
            print('Warning:', 'NR cannot be instantaneous with feedback, resetting!')
        instantaneous_NR = False


#%% Runaway electrons and ions

# NOTE: higher source NR -> smoother pulses (see comments)
UR = 7.3e6      # The runaway potential, so that lambda_R = UR/(E-Et)
def get_NRext(E):
    "NR without any feedbacks"
    # Downstream density from dn/dx = R*n + S/c, n(0)=0.
    # exprel(x) = expm1(x)/x, with the exact limit exprel(0)=1.
    x = (np.asarray(E)-Et)*Lcloud/UR
    return S*Lcloud/phys.c*exprel(x)

#  Ion information -- from Dwyer (2005)
l_ion_att = 3 # attachment mean free path
Tcloud = 200  # ion temperature inside the cloud
# Ion mass in atomic units
Mi = 14 + 3*16 # NO3-

def init2_particles():
    """Constant-delta feedback parameters; fitted values have reference Katm=1."""
    global Et, alphai, mui, Ef, Es, A, delta, LDwyer, EDwyer
    if not constant_delta:
        raise NotImplementedError('Nonconstant delta is not consistently implemented')
    if not np.isfinite(Katm) or Katm <= 0 or not np.isfinite(Lcloud) or Lcloud <= 0:
        raise ValueError('Katm and Lcloud must be finite and positive')
    if not manual_feedback_parameters and Katm != 1:
        raise NotImplementedError('Fitted feedback is supported only at Katm=1; use the manual density-scaled model')
    Et = 2.8e5*Katm
    alphai = Et/35  # electron-ion pairs per metre; 35 eV per pair
    mui = 3e-4/Katm
    if manual_feedback_parameters:
        if reload_Dwyer2007:
            raise ValueError('Manual parameters and Dwyer reload are mutually exclusive')
        Ef = Et + UR/(200/Katm)
        delta = 0.01
    elif reload_Dwyer2007:
        data = Path(__file__).with_name('Dwyer2007_Fig3_L_Egamma1.txt')
        tmp = np.loadtxt(data).T
        LDwyer, EDwyer = tmp[0], tmp[1]*1e6
        slope, Ef = np.polyfit(1/LDwyer, EDwyer, 1)
        delta = np.exp(-slope/UR)
    else:
        Ef, delta = 326140.5, 0.0038274
    # R=(E-Et)/UR; A is attenuation per metre; gamma=delta*exp((R-A)*L).
    A = (Ef-Et)/UR
    if not np.isfinite(A) or A <= 0 or not 0 < delta < 1:
        raise ValueError('Feedback requires positive attenuation and 0 < delta < 1')
    Es = Ef + np.log(1/delta)*UR/Lcloud

def get_vthermal_ion(E):
    # eta_ion = vthermal_ion/l_ion_att is the ion attachment rate
    # Thermal speed plus ion drift is a model approximation.
    vthermal_ion = np.sqrt(phys.Boltzmann*Tcloud/(Mi*phys.m_u)) # for E=0
    return vthermal_ion + mui*E

#%% Debugging plots
def plot_debugging():
    if reload_Dwyer2007:
        plt.figure(10)
        plt.clf()
        plt.plot(LDwyer, Ef + UR*np.log(1/delta)/LDwyer)
        plt.loglog(LDwyer, EDwyer, 'x-')

#%% The cloud information, derived from the typical charge relaxation rate tau_cloud
def init3_cloud():
    global E0, sigma_cloud, J0, Jf, cloud_ion_source, NR_cloud, Nion_cloud
    E0 = Et*E0Et # Aim for E0 in the relaxed state
    sigma_cloud = phys.epsilon_0/tau_cloud
    NR_cloud = get_NRext(E0) # background NR -- it is >0 if E0 is close to Et
    sigma_cloud_NR = NR_cloud*phys.c*phys.e/E0
    Nprod_cloud = NR_cloud*phys.c*alphai
    Nsec_cloud = Nprod_cloud/nua_fun(E0)
    sigma_cloud_sec = Nsec_cloud*phys.e*mue_fun(E0)
    sigma_cloud_ion = sigma_cloud - sigma_cloud_sec - sigma_cloud_NR
    assert sigma_cloud_ion >=0
    # Solve 2*Nion_cloud*phys.e*mui = sigma_cloud_ion
    Nion_cloud = sigma_cloud_ion/(phys.e*mui)/2 # background positive ion density
    # Some mysterious source that sustains the cloud ions
    cloud_ion_source = Nion_cloud*get_vthermal_ion(E0)/l_ion_att - Nprod_cloud # subtract source due to NR
    # Current needed to sustain the field
    Jf = sigma_cloud*Ef
    J0 = sigma_cloud*E0

#%% The source of change of the field
# Choose which source: field or current (or both)
external_field = (KEext != 0)
external_current = (KJext != 0)

# The functional shapes for them
def get_ext_source_lightcurve(tn, external_source_shape):
    "Somewhat arbitrary; adjusted to be zero at the start"
    if external_source_shape=='sigmoid':
        shape = sigmoid(tn)
    elif external_source_shape=='gaussian':
        shape = np.exp(-tn**2/2)
    elif external_source_shape=='wideramp':
        shape = (np.sqrt(tn**2 + 1) + tn)/2
    elif external_source_shape=='ramp':
        tn = np.asarray(tn)
        shape = tn.copy()
        shape[tn<0] = np.log(1+np.exp(tn[tn<0]))
        shape[tn>=0] = tn[tn>=0] + np.log(np.exp(-tn[tn>=0])+1)
        shape = shape[()]
    elif external_source_shape=='twosigmoids':
        s = 1
        taun = 1/10
        shape1 = sigmoid(tn+s)
        shape2 = sigmoid((tn-s)/taun)
        shape = (shape1 + 10*shape2)/11
    else:
        raise ValueError('unknown shape '+str(external_source_shape))
    return shape

def Jext(t):
    # coerced to zero initial value
    lightcurve = get_ext_source_lightcurve(t/tau, Jext_shape) - get_ext_source_lightcurve(T1/tau, Jext_shape)
    return KJext*Jf*lightcurve
def Eext(t):
    # coerced to zero initial value
    lightcurve = get_ext_source_lightcurve(t/tau, Eext_shape) - get_ext_source_lightcurve(T1/tau, Eext_shape)
    return KEext*Ef*lightcurve

def init():
    "Call this after changing any parameters; source switches are derived."
    global external_field, external_current
    external_field = (KEext != 0)
    external_current = (KJext != 0)
    init1_prefs()
    init2_particles()
    if debugging_plots:
        plot_debugging()
    init3_cloud()

def state_quantities(t, u):
    """Shared RHS/output physics for scalar or sampled [Ecloud, NR, Nion].

    Jtot is signed modeled conduction, excluding J0 and Jext. At zero field,
    retain the original positive polarity and 0.1 V/m magnitude floor.
    Jion means background Ohmic current when use_ions=False. JR and Jsec
    report physical components even when minimal_model omits them from Jtot.
    """
    Ecloud, NR, Nion = u
    E = Ecloud + Eext(t) if external_field else Ecloud
    sign = np.where(np.asarray(E) < 0, -1., 1.)
    Emag = np.maximum(np.abs(E), 0.1)
    NRext = get_NRext(Emag)
    if not do_feedback and instantaneous_NR:
        NR = NRext
    Nprod = NR*phys.c*alphai
    Nsec = Nprod/nua_fun(Emag)
    JR = sign*NR*phys.c*phys.e
    Jsec = sign*Nsec*phys.e*vd_fun(Emag)
    Jion = sign*(2*Nion*phys.e*mui*Emag if use_ions else sigma_cloud*Emag)
    Jtot = Jion if minimal_model else JR + Jsec + Jion
    return dict(E=E, NR=NR, Nion=Nion, Nsec=Nsec, JR=JR, Jsec=Jsec,
                Jion=Jion, Jtot=Jtot, Emag=Emag, NRext=NRext, Nprod=Nprod)


#%% The time derivative
def fgf(t, u):
    "Evolution of E, NR and Nion"
    state = state_quantities(t, u)
    E, NR, Nion = state['Emag'], state['NR'], state['Nion']
    NRext, Nprod = state['NRext'], state['Nprod']
    J = J0 - state['Jtot']
    if external_current:
        J += Jext(t)
    if do_feedback or not instantaneous_NR:
        # The NR evolution, due to explosive feedback
        # Feedback
        tround = 2*Lcloud/phys.c
        if do_feedback:
            loggamma = (E-Es)*Lcloud/UR # logarithm of the feedback factor gamma
            # Correction to get a correct steady-state NR = NRext/(1-gamma) for gamma<1
            if loggamma < 0:
                gamma_minus_1 = np.expm1(loggamma)
            else:
                gamma_minus_1 = loggamma # logarithmic growth per feedback round trip
        else: # not instantaneous_NR -- do the same as with feedback but take gamma=0
            gamma_minus_1 = -1
        NRdot = (NR*gamma_minus_1 + NRext)/tround
    else: # no feedback and take a shortcut in the calculations, i.e. instantaneous NR calculation
        NRdot = 0
    # The evolution of ion density
    if use_ions:
        if minimal_model:
            # Neglect ion attachment and background source of ions
            Niondot = Nprod
        else:
            eta_ion = get_vthermal_ion(E)/l_ion_att # modification due to field is small and does not matter
            Niondot = Nprod - Nion*eta_ion + cloud_ion_source
            # Negative ions are produced at exactly the same rate, because Nprod = Nsec*nu_a
    else:
        Niondot = 0
    return [J/phys.epsilon_0, NRdot, Niondot]

#%% The long part -- solution of the model
num_time_points = 100000
DEFAULT_RTOL = 1e-9
DEFAULT_ATOL = (1e-7, 1e-14, 1e-3)  # Ecloud (V/m), NR and Nion (m^-3)


def _validate_states(y, atol):
    """Check finiteness and density admissibility without modifying the states."""
    if not np.all(np.isfinite(y)):
        raise RuntimeError('Integration returned nonfinite states')
    atol = np.broadcast_to(atol, (3,))
    if np.any(y[1:] < -10*atol[1:, None]):
        raise RuntimeError('Integration returned materially negative density')
    if np.any(y[1:] < 0):
        warnings.warn('Small negative density within solver absolute tolerance; values are not clipped',
                      RuntimeWarning, stacklevel=2)


def validate_solution(sol):
    """Reject incomplete/nonfinite results and material negative densities."""
    if not sol.success or sol.status != 0:
        raise RuntimeError(f'Integration incomplete: {sol.message}')
    start, end = getattr(sol, 'fgf_t_span', (T1, T2))
    t, y = np.asarray(sol.t), np.asarray(sol.y)
    if (t.ndim != 1 or t.size < 2 or y.shape != (3, t.size)
            or not np.all(np.isfinite(t))
            or not np.all(np.diff(t) > 0)):
        raise RuntimeError('Integration returned invalid time/state arrays')
    time_slack = 16*np.finfo(float).eps*max(abs(start), abs(end), end-start)
    if abs(t[0]-start) > time_slack or abs(t[-1]-end) > time_slack:
        raise RuntimeError(f'Integration does not cover [{start}, {end}]')
    _validate_states(y, getattr(sol, 'fgf_atol', DEFAULT_ATOL))


def solve_odes(*, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL, max_step=np.inf):
    """Integrate with explicit component tolerances, independently of output spacing."""
    time0 = time.time()
    t = np.linspace(T1, T2, num_time_points)
    sol = solve_ivp(fgf, [T1, T2], [E0, NR_cloud, Nion_cloud], t_eval=t,
                    dense_output=True, method=ode_method, rtol=rtol, atol=atol,
                    max_step=max_step)
    sol.fgf_t_span = (T1, T2)
    sol.fgf_atol = np.broadcast_to(atol, (3,)).copy()
    validate_solution(sol)
    return sol, time.time()-time0


def sol2values(sol, *, t=None):
    """Postprocess returned samples, or explicitly resample within a complete solution.

    Physical module parameters must still be those used for integration.
    Changing num_time_points does not implicitly resample an existing result.
    """
    validate_solution(sol)
    if t is None:
        t, y = sol.t, sol.y
    else:
        t = np.asarray(t, dtype=float)
        if (t.ndim != 1 or t.size < 2 or not np.all(np.isfinite(t))
                or not np.all(np.diff(t) > 0) or t[0] < sol.t[0] or t[-1] > sol.t[-1]):
            raise ValueError('Resampling times must increase within the integrated interval')
        if sol.sol is None:
            raise ValueError('Resampling requires dense output')
        y = sol.sol(t)
        _validate_states(y, getattr(sol, 'fgf_atol', DEFAULT_ATOL))
    state = state_quantities(t, y)
    return dict(t=t, **{key: state[key] for key in
                ('E', 'NR', 'Nion', 'Nsec', 'JR', 'Jion', 'Jsec', 'Jtot')})

#%% Plot the solution
def plot(values, fig=None, fignum=None, tunitname='ms'):
    "Relies on a lot of global variables responsible for plotting preferences and setups"
    if tunitname=='ms':
        tunit = 1e-3 # ms
    elif tunitname=='s':
        tunit = 1
    else:
        raise Exception('Unknown t unit')
    t = values['t']; E = values['E']; NR = values['NR']; Nion = values['Nion']; Nsec = values['Nsec']
    tplot = t/tunit
    # Prepare the title of the figure
    the_title = r'$L_\mathrm{cloud} =' + latex_float(Lcloud/1e3) + r'$ km; '
    the_title += r'$\tau ='+ latex_float(tau/1e-3) + '$ ms; '
    if external_field:
        the_title += r'$k_E = ' + latex_float(KEext) + r'$; '
    if external_current:
        the_title += r'$k_J = ' + latex_float(KJext) + r'$; '
    if Katm != 1:
        the_title += r'$K_\mathrm{atm} = ' + latex_float(Katm) + r'$; '
    if minimal_model:
        the_title += 'Minimal model'
    else:
        if not do_feedback:
            the_title += 'No feedback; '
        if instantaneous_NR:
            the_title += 'Instantaneous $N_R$; '
    # Prepare the figure
    if fig is None:
        fig = plt.figure(fignum)
    fig.clear()
    if plot_only_NR:
        if plot_currents:
            fig.set_size_inches(9, 2.5)
        else:
            fig.set_size_inches(9, 3)
        ax1 = fig.subplots(1, 1, sharex=True)
    else:
        fig.set_size_inches(9, 10)
        if plot_currents:
            ax1, ax2, ax3, ax5, ax4 = fig.subplots(5, 1, sharex=True)
        else:
            ax1, ax2, ax3, ax4 = fig.subplots(4, 1, sharex=True)
    fix_margins(fig)
    xlims = np.array([T1, T2])/tunit
    # Plot NR
    if plot_logNR:
        ax1.semilogy(tplot, NR)
    else:
        ax1.plot(tplot, NR)
    ax1.set_title(the_title)
    ax1.set_ylabel('$N_R$, m$^{-3}$')
    ax1.set_xlim(*xlims)
    if plot_only_NR:
        ax1.set_xlabel('$t$, ' + tunitname)
        return fig
    # Plot electric field
    Eunit = 1e6; Eunitname = 'MV/m'
    ax2.plot(tplot, E/Eunit, 'k')
    ax2.plot(tplot, -E/Eunit, 'k--')
    # The thresholds
    line, = ax2.plot(xlims, Es*np.ones((2,))/Eunit); line.set_label('$E_s$')
    line, = ax2.plot(xlims, Ef*np.ones((2,))/Eunit); line.set_label('$E_f$')
    line, = ax2.plot(xlims, Et*np.ones((2,))/Eunit); line.set_label('$E_t$')
    ax2.set_ylabel('$E$, '+Eunitname)
    if plot_label_threshold_fields:
        ax2.legend()
    ax2.set_ylim((Et-Eslack)/Eunit, (Es+Eslack)/Eunit)

    line, = ax3.semilogy(tplot, Nion); line.set_label('$N_p=N_n$')

    if plot_Nsec:
        line, = ax3.semilogy(tplot, Nsec); line.set_label('$N_e$')
        ax3.set_ylabel('$N_i$, $N_e$, m$^{-3}$')
        ax3.legend()
    else:
        ax3.set_ylabel('$N_i$, m$^{-3}$')

    if plot_currents:
        JR = values['JR']; Jion = values['Jion']; Jsec = values['Jsec']; Jtot = values['Jtot']
        line, = ax5.plot(tplot, Jion); line.set_label('$J_i$')
        line, = ax5.plot(tplot, Jsec); line.set_label('$J_e$')
        line, = ax5.plot(tplot, JR); line.set_label('$J_R$')
        line, = ax5.plot(tplot, Jtot, color='k'); line.set_label('$J_{tot}$')
        if all(np.all(current > 0) for current in (JR, Jsec, Jion, Jtot)):
            ax5.set_yscale('log')
        else:
            ax5.set_yscale('symlog', linthresh=max(float(np.max(np.abs(Jtot)))*1e-4, 1e-30))
        ax5.set_ylabel('$J$, A/m$^2$')
        ax5.legend(loc='upper right')

    if external_field:
        line, = ax4.plot(tplot, Eext(t)/Ef/KEext)
        line.set_label('$E_{ext}/(k_E E_f)$')
    if external_current:
        line, = ax4.plot(tplot, Jext(t)/Jf/KJext)
        line.set_label('$J_{ext}/(k_J J_f)$')
    ax4.legend()
    ax4.set_ylabel('External sources')
    ax4.set_xlabel('$t$, ' + tunitname)
    return fig

def init_and_solve(**solver_options):
    "Initialize, integrate and return postprocessed values."
    init()
    sol, sol_time = solve_odes(**solver_options)
    print('It took', neat_float(sol_time), 'seconds to calculate')
    values = sol2values(sol)
    return values

if __name__ == '__main__':
    values = init_and_solve()
    fig = plot(values)
    if do_save_figures:
        save_figure(fig, ['png'])


# Additional calculations

#%% Gamma ray calculations
# Some constants
mc2 = phys.m_e*phys.c**2
rclass0 = phys.physical_constants['classical electron radius'][0]
def get_volume(E):
    """Legacy effective volume (m^3); its subthreshold cutoff is unresolved physics."""
    E = np.asarray(E, dtype=float)
    if not np.all(np.isfinite(E)):
        raise ValueError('Production field must be finite')
    if np.any(E < Et):
        warnings.warn('Legacy production volume sets emission to zero below Et, including reversed fields; '
                      'feedback spatial normalization remains unresolved', RuntimeWarning, stacklevel=2)
    R = (E-Et)/UR
    length = np.zeros(E.shape)
    length[E == Et] = Lcloud
    above = E > Et
    length[above] = np.minimum(1/R[above], Lcloud)
    return (length*Lcloud**2)[()]

def get_Fgrays(NR, E, minTGFeV=3e5):
    """Gamma-ray production
    Source photons/s above minTGFeV (eV), using 7.3 MeV electrons; no transport."""
    NR, E = np.broadcast_arrays(np.asarray(NR), np.asarray(E))
    if not np.isfinite(minTGFeV) or minTGFeV <= 0:
        raise ValueError('Gamma threshold must be finite and positive')
    kmin = minTGFeV*phys.e/mc2 # dimensionless (in units of mc2) minimum source gamma energy
    kmax = UR*phys.e/mc2 # the typical RRE energy in mc2 units
    if kmin>=kmax: return np.zeros(NR.shape)[()]
    pmc = np.sqrt((1+kmax)**2-1) # RRE momentum in mc units
    kb = np.linspace(kmin, kmax, 1001)
    # Production by 1 RRE per unit length
    alpha_grays = Katm*N0*np.sum(bremsstrahlung_3BN(pmc, ave(kb))*np.diff(kb))
    # Must be around alpha_grays = 0.01 for sea level
    Fgrays = alpha_grays*phys.c*NR*get_volume(E) # total production per unit time
    return Fgrays

#%% Radio emissions
def get_Erad(t, Jtot, D=1e4, theta=np.pi/2):
    "Calculate the radio emission, also filtered due to the cloud size"
    t, Jtot = np.asarray(t, dtype=float), np.asarray(Jtot, dtype=float)
    if not np.isfinite(D) or D <= 0 or not np.isfinite(theta):
        raise ValueError('Distance must be finite and positive; theta must be finite')
    if (t.ndim != 1 or t.size < 3 or Jtot.shape != t.shape
            or not np.all(np.isfinite(t)) or not np.all(np.isfinite(Jtot))):
        raise ValueError('Radio calculation requires matching finite 1D arrays of length >= 3')
    steps = np.diff(t)
    dt = np.mean(steps)
    if np.any(steps <= 0) or not np.allclose(steps, dt, rtol=1e-6, atol=0):
        raise ValueError('Radio FFT requires increasing uniformly spaced times')
    # Total current moment (in A*m)
    M = Jtot*Lcloud**3
    dMdt = np.hstack((0, ave(np.diff(M)/np.diff(t)), 0)) # to center it at moments of t
    # Radiated field is E = dMdt * sin(theta) * mu0 / (4*pi*D), where D is the distance; mu0/(4*pi) = 1e-7
    # For theta = pi/2 (max)
    Erad = 1e-7*dMdt/D*np.sin(theta)
    N = len(t)
    T = N*dt
    # Filtering (with the scale of cloud, sigma_x = Lcloud/2)
    sigma_t = Lcloud/2/phys.c
    sigma_f = 1/(2*np.pi*sigma_t)
    df = 1/T
    tmp = np.fft.rfft(Erad)
    f = np.arange(len(tmp))*df
    # The filtered value
    Efil = np.fft.irfft( np.exp(-f**2/(2*sigma_f**2))*tmp, len(Erad))
    return Erad, Efil

#%% Optical emissions
def electron_range(En):
    "Eq. (3.5) in Lehtinen thesis, originially from Rees [1989, p. 40]. En is in eV, answer in meters"
    return (8.93e19 + 1.11e21*(En/1e3)**1.67)/(N0*Katm)

def vfun(En):
    "En is in eV"
    gamma = 1+En*phys.e/mc2
    return np.sqrt(1-1/gamma**2)*phys.c

def electron_thermalization_time(En):
    "En is in eV"
    return electron_range(En)/vfun(En)

kappa = 2*np.pi*rclass0**2*mc2*phys.c**2 # Equation (2.3) in Lehtinen thesis, except division by v^2
def sigma_i_general(En0, Ensec, Zm, Ei):
    """Ionization cross-section (differential).
    Energies are in eV.
    The result is in m^2 eV^{-1} per molecule
    Eq. (3.10) in Lehtinen thesis, expression analogous to Rees [1989, p. 43]"""
    return kappa*Zm/vfun(En0)**2/((Ei + Ensec)*phys.e)**2*phys.e

def sigma_i(En0, Ensec):
    "Average for air"
    return 0.8*sigma_i_general(En0, Ensec, 14, 13) + 0.2*sigma_i_general(En0, Ensec, 16, 17.4)

def sigma_itot_general(En0, Zm, Ei, I):
    "Integrated ionization approximation (m^2); unreliable for low primary energies."
    # I = 15.6 for N2, 12.2 for O2
    return kappa*Zm/vfun(En0)**2*(1/Ei-1/(Ei+En0-I))/phys.e

def get_337(E, NR):
    r"""Approximate source photons/s for Second Positive N2 (nominal 337 nm).
    C^3\Pi_u -> B^3\Pi_g
    """
    Nm = N0*Katm
    NN2 = Nm*0.8 # scaled
    # How many suprathermal electrons?
    sigmax, dEn, En = (3.8e-21, 10, 16) # for C band of N2 (2nd positive, 337 nm), from Lehtinen thesis, p. 45
    # sigma_i is per air molecule: use Nm here and NN2 only for excitation.
    Nsupra = electron_thermalization_time(En)*(sigma_i(UR, En)*dEn)*phys.c*Nm*NR
    # Steady-state excited C-state density nC (m^-3).
    alphaC = 0.2*3e-16 + 0.8*0.13e-16 # quenching per molecule of air
    AC = 2.38e7 # inverse lifetime of level C
    tauC = 1/(AC + Nm*alphaC)
    nC = sigmax*vfun(En)*Nsupra*NN2*tauC
    emission_rate_2PN2 = nC*AC*get_volume(E)
    return emission_rate_2PN2 # photons per second
