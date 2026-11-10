#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 15:08:55 2018

Units used are either:
    SI(e.g. J for energy and kg*m/s for momentum)
    or dimensionless (i.e., k = E_phot/mc2 = p_phot/mc and pmc=p/mc).

@author: nle003
"""

#%%
import pickle
import numpy as np
import scipy.constants as phys
import matplotlib.pyplot as plt
from model_helpers import ave

do_exec = __name__ == '__main__'
if do_exec:
    save_figures = False

#%% Bethe stuff
mc2 = phys.m_e*phys.c**2 # m_e*clight**2, in J
mc2eV = mc2/phys.e       # in eV
mc2keV = mc2eV/1e3       # in keV
mc = phys.m_e*phys.c     # in kg*m/s
e2 = phys.e**2/(4*np.pi*phys.epsilon_0) # in J*m
r0 = e2/mc2              # classical radius, in m
Zm = 14.5                # average Z of an air molecule
Za = Zm/2                # average Z of an air atom
N0 = phys.physical_constants['Loschmidt constant (273.15 K, 101.325 kPa)'][0] # Loschmidt constant 2.687e25
nu0 = 2*np.pi*N0*r0**2*phys.c # typical RREA collision frequency

# Important notation: pmc=p/(mc), a dimensionless value

def pmc_from_keV(En):
    "For an electron"
    gamma = 1+En/mc2keV
    return np.sqrt(gamma**2-1)
def keV_from_pmc(pmc):
    gamma = np.sqrt(1+pmc**2)
    return (gamma-1)*mc2keV
def v_from_pmc(pmc):
    return pmc/np.sqrt(1+pmc**2)*phys.c
def v_from_keV(En):
    gamma = 1+En/mc2keV
    return np.sqrt(1-1/gamma**2)*phys.c

EneV_limit = 1000 # below which formulas are not correct
gam_limit = 1+EneV_limit*phys.e/mc2
pmc_limit = np.sqrt(gam_limit**2-1)
def FD(pmc):
    """Dynamic friction as a function of momentum. Argument is dimensionless p/mc.
    See Lehtinen dissertation, equation (2.2)"""
    assert (np.asarray(pmc)>=pmc_limit).all()
    gam = np.sqrt(1+pmc**2)
    b = pmc/gam
    Eionmc2 = 80.5*phys.e/mc2
    res = np.log(b**2*(gam-1)*gam**2/(Eionmc2**2)) - (1+2/gam-1/gam**2)*np.log(2) \
        + (gam-1)**2/8/gam**2 + 1/gam**2
    return nu0*Zm*mc*res/b**2

def nu_FD(pmc):
    "Typical energy loss frequency"
    return FD(pmc)/pmc/mc

def nu_m(pmc):
    "Momentum loss rate: Equation (2.13) in Lehtinen dissertation divided by 2"
    gam = np.sqrt(1+pmc**2)
    res = nu0*Zm**2*gam/pmc**3*np.log(164.7*pmc/Zm**(1/3))
    return res

def mu_ave(eEpnu):
    """<μ>, where μ == cos θ, and θ is the elastic scattering angle (of an electron on a nucleus).
    Based on equation (2.25) in Lehtinen dissertation.
    The argument is e*E/(p*nu_m)."""
    a = 2*np.asarray(eEpnu, dtype=float)
    if not np.all(np.isfinite(a) & (a >= 0)):
        raise ValueError('eEpnu must be finite and nonnegative')
    result = np.empty_like(a)
    small = a < 1e-2
    x = a[small]
    result[small] = x*(1/3 - x*x/45 + 2*x**4/945 - x**6/4725)
    result[~small] = 1/np.tanh(a[~small]) - 1/a[~small]
    return result[()]

#%% Determine the break-even field as a function of pmc
eEs = 10**np.linspace(5,8,1000)*phys.e # Table of energies in J, from 10^5 to 10^8 eV
@np.vectorize
def get_Feff(pmc):
    "Solve equation eE*M(p,eE)=FD(p) for eE, where M = <μ>"
    eEpnu = eEs/(pmc*mc*nu_m(pmc))
    F = FD(pmc)/mu_ave(eEpnu)
    # Solve func(eEs,p) == eEs-FD(p)/M(p,eEs) == 0
    return np.interp(0,eEs-F,eEs) #  interp(0, func(eEs,p), eEs)

#%% Bremsstrahlung cross-sections
# See Koch and Motz [1959]
def bremsstrahlung_3BN(pmc, k):
    """Differential cross-section dchi/dk in m^2 per air molecule per unit k.
    The dimensionless variables are:
        pmc = p_e/(mc) -- initial electron momentum;
        k = En_phot/(mc2) = p_phot/(mc) -- photon energy (or momentum) (can be table)
    Approximations: Born, Nonscreened"""
    pmc, k = np.broadcast_arrays(np.asarray(pmc, dtype=float), np.asarray(k, dtype=float))
    if not np.all(np.isfinite(pmc) & (pmc >= 0) & np.isfinite(k) & (k > 0)):
        raise ValueError('Require finite pmc >= 0 and photon energy k > 0')
    result = np.zeros(pmc.shape)
    gamma = np.sqrt(1+pmc**2)
    valid = k < gamma-1-phys.e/mc2  # retain the original 1 eV endpoint slack
    pmc, k, gamma = pmc[valid], k[valid], gamma[valid]
    chi0 = 2*Za*(Za+1)*r0**2/137  # per air molecule, m^2 per unit k
    gamma1 = gamma-k # final electron gamma-factor
    gg1 = gamma*gamma1
    pmc1 = np.sqrt(gamma1**2-1)
    pp1 = pmc*pmc1
    L = 2*np.log((gg1 + pp1-1)/k)
    z = 2*np.log(gamma+pmc)
    z1 = 2*np.log(gamma1+pmc1)
    chi1 = (4/3)-2*gg1*(pmc**2+pmc1**2)/pp1**2 + z*gamma1/pmc**3 + z1*gamma/pmc1**3 - z*z1/pp1
    chipart = z*(gg1+pmc**2)/pmc**3 - z1*(gg1+pmc1**2)/pmc1**3 + 2*k*gg1/pp1**2
    chi2 = (8/3)*gg1/pp1 + k**2*(gg1**2 + pp1**2)/pp1**3 + k*chipart/2/pp1
    result[valid] = chi0*pmc1/pmc*(chi1+L*chi2)/k
    return result[()]

def brems_radpat(pmc, mu):
    "The radiation pattern, see equation (3.12) in Lehtinen dissertation, normalized int_{-1}^1 dmu = 1"
    gamma = np.sqrt(1+pmc**2)
    beta = pmc/gamma
    tmp = 1-beta*mu
    return (3./8.)*(1 + ((mu-beta)/tmp)**2)/(gamma*tmp)**2

mub = np.linspace(-1,0,1001) # backward only
muc = ave(mub)
dmu = np.diff(mub)
@np.vectorize
def brems_backwardfrac(pmc):
    "Fraction going backward"
    return np.sum(brems_radpat(pmc,muc)*dmu)

@np.vectorize
def seed_photons(pmc, kmin):
    "Inverse production length of a photon with minimum energy of kmin"
    kmax = np.sqrt(1+pmc**2)-1
    if kmin>kmax:
        return 0. # np.vectorize infers its output dtype from the first value.
    else:
        kb = np.linspace(kmin, kmax, 1001)
        return N0*np.sum(bremsstrahlung_3BN(pmc, ave(kb))*np.diff(kb))

if do_exec:
    # 1 MeV electrons, 80 keV photons
    EnekeV = 1000
    EnphkeV = np.linspace(0,EnekeV,1000)
    EnphkeVc = ave(EnphkeV)
    kb = EnphkeV*1000*phys.e/mc2 # dimensionless photon energy
    kc = ave(kb)
    dk = np.diff(kb)
    pmc = pmc_from_keV(EnekeV)
    v = pmc*phys.c/np.sqrt(1+pmc**2)
    chi = bremsstrahlung_3BN(pmc,kc) # per molecule per unit k
    tmp = N0*v*np.cumsum(chi*dk); rate = tmp[-1]-tmp
    plt.figure(600)
    plt.clf()
    nu_eff = get_Feff(pmc)/pmc/mc
    plt.plot(EnphkeVc,rate/nu_eff)
    EnphkeV0 = 100
    ii = np.searchsorted(EnphkeVc,EnphkeV0)
    print('Probability of a',EnekeV,'keV electron to produce',EnphkeV0,'keV photon',
          'before slowing down is',rate[ii]/nu_eff)

#%% Photon cross sections
def photoeffect(k,Z):
    "(13.17) in Heitler"
    g=k+1;
    p=np.sqrt(g**2-1);
    return 4*np.pi*r0**2*Z**5*phys.alpha**4/k**5*p**3*(
        4./3.+g*(g-2)/(g+1)*(1.-1./(2.*g*p)*np.log((g+p)/(g-p))) );

def inv_lambda_absorption(k):
    sig = 2*photoeffect(k,7)*.79 + 2*photoeffect(k,8)*.2 # + photoeffect(k,18)*.01
    # The Argon binding energy (3 keV) is too high compared to N and O (~.5 keV)
    return N0*sig;

#%% Runaway electron stuff

def nu_bakhov(E):
    "A fit to Bakhov et al [2000] runaway frequency"
    # This is very small, I think we can safely exclude it from consideration
    return 10**(11-6*np.exp(-(E/1e6-25)/12))

def vdrift(pmc, E):
    "Energetic electron drift velocity"
    eEpnu = phys.e*E/(pmc*mc*nu_m(pmc))
    mu = mu_ave(eEpnu)
    return  mu*phys.c*pmc/np.sqrt(1+pmc**2)


#%%#################################################################################################
#%% DERIVED STUFF, MAYBE MOVE TO ANOTHER FILE

#%% Plot
Ens = 10**np.linspace(3,7.6,1000)*phys.e # energies in J, from 1 keV to ~40 MeV
pmcs = np.sqrt((1+Ens/mc2)**2-1)

if do_exec:
    # Plot the bremsstrahlung cross section
    plt.figure(1000)
    brems_tmp = bremsstrahlung_3BN(pmcs[:,None], Ens[None,:]/mc2)
    logEn = np.log10(Ens/phys.e)
    plt.pcolormesh(logEn, logEn, np.log10(brems_tmp))
    plt.colorbar()

# Reusable table values
pnus = pmcs*mc*nu_m(pmcs)
Feffs = get_Feff(pmcs) # effective friction force in N (runaway threshold for given pmc)
irreath = np.argmin(Feffs)
Erreath = Feffs[irreath]/phys.e # The runaway threshold field

def get_En_th(E):
    "Solve equation eE*M(p,eE)=FD(p) for p and return energy in J"
    return np.interp(phys.e*E, Feffs[irreath-1::-1], Ens[irreath-1::-1], left=np.inf)

def sigma_re(En, E):
    "Runaway ionization rate for primary of energy En (avalanche rate nu_rrea = sigma_re(inf)*v*Nm)"
    En, Enth = np.broadcast_arrays(np.asarray(En, dtype=float), get_En_th(E))
    if np.any(np.isnan(En) | (En < 0)):
        raise ValueError('Primary energy must be nonnegative (J); +inf gives the high-energy limit')
    res = np.zeros(En.shape)
    active = En/2 > Enth
    gamma = En[active]/mc2 + 1
    beta2 = 1 - 1/gamma**2
    res[active] = 2*np.pi*Zm*r0**2*mc2*(1/Enth[active] - 2/En[active])/beta2
    return res[()]

ks = Ens/mc2 # use the same table
ksc = ave(ks)
dk = np.diff(ks)

@np.vectorize
def sigma_brems_cumulative(En, kmin=1e3*phys.e/mc2):
    """This is exactly the same as seed_photons, except the form of the first argument.
    Could be a bit faster because not so many values of k are used for summation.
    En is the incident electron energy in J
    kmin is the lower photon energy integration limit in dimensionless units"""
    gamma = En/mc2 + 1
    pmc = np.sqrt(gamma**2-1)
    imax = np.searchsorted(ksc, (En-phys.e)/mc2) # add 1 eV to avoid errors at boundaries
    imin = np.searchsorted(ksc, kmin+(phys.e/mc2)) # same
    if imax<=imin:
        return 0.
    assert ksc[imin]>kmin
    assert ksc[imax-1]<En/mc2
    return np.sum(bremsstrahlung_3BN(pmc, ksc[imin:imax])*dk[imin:imax])

#%%
if do_exec:
    plt.figure(1001)
    plt.clf()
    kmin = 300*1e3*phys.e/mc2
    tmp1 = seed_photons(pmcs, kmin)
    tmp2 = N0*sigma_brems_cumulative(Ens, kmin)
    plt.plot(Ens/phys.e, tmp1)
    plt.plot(Ens/phys.e, tmp2)
    plt.xlabel(r'$\mathcal{E}$, eV')
    plt.ylabel('Photons(>300keV)/m')
#%%

# Convert the number of runaway electrons to the number of photons above kmin
# See also: seed_photons (defined above)
EnR = 7.3e6*phys.e # RREA typical energy
def NR_to_gamma(En_th, kmin=300*1e3*phys.e/mc2):
    "Number of gamma photons produced by 1 RRE per metre"
    ii = np.searchsorted(Ens, En_th)
    Entmp = ave(Ens)[ii:]
    dEntmp = np.diff(Ens)[ii:]
    res = np.sum(sigma_brems_cumulative(Entmp, kmin=kmin)*np.exp(-(Entmp-En_th)/EnR)*dEntmp)/EnR
    return res*N0

#%%
def plot_gamma_yield():
    EEth = np.arange(1.1,20.01,.1)
    # This calculation may take a while
    brems_yield = np.array([NR_to_gamma(get_En_th(E)) for E in Erreath*EEth])
    plt.figure(1002)
    plt.clf()
    plt.plot(EEth, brems_yield)
    plt.ylim(0, 2*np.max(brems_yield))
    plt.xlabel(r'$E/E_{th}$')
    plt.ylabel(r'RREA brems yield (>300keV photons), m$^{-1}$')

if do_exec:
    plot_gamma_yield()

#%%

def nu_rrea(E):
    "RREA rate, approximate -- assuming primary has infinite energy"
    # This is ionization rate at the threshold energy, as per Lehtinen dissertation
    return nu0*Zm*mc2/get_En_th(E)

def nu_rrea_Dwyer(E):
    "Legacy sea-level RREA growth fit (s^-1); physical validity remains unresolved."
    ilam = (E-2.76e5)/7.3e6
    return phys.c*ilam*.9 # drift is v=0.9c

FDs = FD(pmcs)
def get_pFD(E):
    "Solve equation eE=FD(p) for p and return p/mc. See also get_En_th(E)"
    ii=np.argmin(FDs)
    return np.interp(E, FDs[ii-1::-1]/phys.e, pmcs[ii-1::-1])

def save_fig_in_pdf_and_pkl(fig,fname):
    fig.savefig(fname+'.pdf')
    with open(fname+'.pkl','wb') as f:
        pickle.dump(fig,f)

#%%
if do_exec:
    fig=plt.figure(30)
    plt.clf()
    line,=plt.loglog(Ens/phys.e,FDs/phys.e)
    line.set_label(r'dynamic friction $F_D$')
    line,=plt.loglog(Ens/phys.e, pnus/phys.e)
    line.set_label(r'momentum loss $\nu_m p$')
    line,=plt.loglog(Ens/phys.e,Feffs/phys.e)
    line.set_label(r'effective friction $F_{eff}$')
    plt.xlabel(r'$\mathcal{E}$, eV')
    plt.ylabel(r'Force, V/m')
    plt.grid(True,which='both')
    plt.legend()
    save_fig_in_pdf_and_pkl(fig,'frictions')

#%%
if do_exec:
    # Friction ratios
    fig=plt.figure(33)
    plt.clf()
    line,=plt.loglog(Ens/phys.e, pnus/FDs)
    line.set_label(r'$\nu_m p/F_D$')
    line,=plt.loglog(Ens/phys.e, Feffs/FDs)
    line.set_label(r'$F_{eff}/F_D$')
    plt.xlabel(r'$\mathcal{E}$, eV')
    plt.ylabel(r'Force ratios, dimensionless')
    plt.grid(True,which='both')
    plt.legend()
    save_fig_in_pdf_and_pkl(fig,'friction_ratios')


#%% Mean scattering cosine
if do_exec:
    plt.figure(31)
    plt.clf()
    # The average cosine for the given p
    M = mu_ave(Feffs/pnus)
    plt.semilogx(Feffs/phys.e,M)
    plt.grid(True,which='both')

#%% Approximate low-energy runaway boundary
# A different runaway boundary -- almost two times different
Fles = np.sqrt(3*FDs*pnus/5)
if do_exec:
    plt.figure(30)
    line,=plt.plot(Ens/phys.e,Fles/phys.e)
    line.set_label(r'low-energy result (inexact)')
    plt.legend()

#%% Probability of runaway
def get_prob_escape(E0,do_approx=False):
    "Exact; pmc_table has to be dense enough to tolerate integration"
    dpmc = np.diff(pmcs)
    pmcc = ave(pmcs)
    if do_approx:
        # This one is actually more complicated
        ii = np.where(Fles/phys.e<E0)[0][0]
        pn = pmcc/pmcs[ii]
        diff_prob = np.exp(-(5/4)/pn**4)/pn**5
    else:
        Dcs = (phys.e*E0)**2/(3*nu_m(pmcc))
        tmp = np.hstack((0,np.cumsum(FD(pmcc)/Dcs*dpmc*mc)))
        tmp -= tmp[-1] # so that the exponent is not huge
        diff_prob = np.exp(ave(tmp))/(pmcc**2*Dcs) # expression to be integrated
    tmp = np.hstack((0,np.cumsum(diff_prob*dpmc)))
    return tmp/tmp[-1]

# Only the standalone demonstration uses this probability table.
if do_exec:
    prob2d = np.array([get_prob_escape(E0) for E0 in eEs/phys.e])

#%%
if do_exec:
    E0 = 1e6 # E=5 Et, like in Figure 2.6 in Lehtinen dissertation
    ii = np.where(Fles/phys.e<E0)[0][0]
    jj = np.where(Feffs/phys.e<E0)[0][0]
    plt.figure(32)
    plt.clf()
    prob_escape_approx = get_prob_escape(E0,do_approx=True)
    line,=plt.semilogx(Ens/phys.e,prob_escape_approx)
    line.set_label('approximate')
    plt.semilogx(Ens[ii]/phys.e,prob_escape_approx[ii],'o')
    plt.semilogx(Ens[jj]/phys.e,prob_escape_approx[jj],'x')
    plt.xlabel(r'$\mathcal{E}$, eV')
    plt.ylabel(r'Probability of runaway')
    pmc1,pmc2 = np.interp(np.array([0.1,0.9]),prob_escape_approx,pmcs)
    plt.title(f'E={E0:g} V/m: runaway boundary is p/(mc) from {pmc1:g} to {pmc2:g}')
    plt.grid(True)
    prob_escape_exact = get_prob_escape(E0,do_approx=False)
    line,=plt.semilogx(Ens/phys.e,prob_escape_exact)
    line.set_label('exact')
    plt.semilogx(Ens[ii]/phys.e,prob_escape_exact[ii],'o')
    plt.semilogx(Ens[jj]/phys.e,prob_escape_exact[jj],'x')
    plt.legend()
    if save_figures:
        plt.savefig('prob_escape.pdf')

#%%
if do_exec:
    plt.figure(34)
    plt.pcolor(np.log10(Ens[::10]/phys.e), np.log10(eEs[::10]/phys.e), prob2d[::10,::10])
    plt.plot(np.log10(Ens/phys.e), np.log10(Fles/phys.e), 'r')
    plt.plot(np.log10(Ens/phys.e), np.log10(Feffs/phys.e), 'g')
