#ifndef RREA_CHARGED_FINAL_STATES_H
#define RREA_CHARGED_FINAL_STATES_H

// The discrete channels of a tracked lepton: their phase-space gates, their
// frozen competing rates, and their final states -- hard Moller, hard Bhabha
// and bremsstrahlung, one named module each.  Every final state owns the
// COMPLETE event --
// the sampled kinematics, the parent's post-event (K, direction), the spawned
// secondary, the material sources and the group's conservation closure -- and
// is used by both the live electron/positron loops and the newborn residual
// advance, so a secondary can never be given different physics from the parent
// that made it.
//
// RNG identity stays with the caller: rng_key(channel) maps a channel id to
// the caller's key, which is the particle's own stream at the substep's
// interaction index in the live loops and the PARENT stream at the packed
// transient index in the newborn advance.  The physics is the same draw on
// whichever stream the caller owns.
//
// Pinned by rrea_charged_interleave_smoke (the kinematics primitives these
// call), rrea_secondary_group_smoke (the group/conservation contracts they
// emit) and rrea_warpx_transport_apply (the assembled step).

#include "RreaChargedInterleave.H"
#include "RreaMaterialPath.H"
#include "RreaParticleInteraction.H"
#include "RreaRng.H"
#include "RreaTransportLog.H"

#include "rrea/RreaDebugOptions.H"
#include "rrea/RreaInteractionPhysics.H"


#include <AMReX_REAL.H>

#include <array>
#include <atomic>

namespace rrea::warpx {

// Phase-space gate of one competing discrete channel, in the species' hazard
// order -- e-: 0 hard Moller, 1 tracked brems; e+: 0 hard Bhabha, 1 tracked
// brems, 2 in-flight annihilation.  A channel is open only where its own
// kinematics allow it AND the table carries a strictly positive rate there:
//   hard knock-on   K > 2 T_cut (Moller, indistinguishable pair) or K > T_cut
//                   (Bhabha, distinguishable);
//   tracked brems   K > E_gamma,cut -- the channel is DEFINED by emitting a
//                   photon at or above the transport cutoff, which a parent
//                   at or below that cutoff cannot do.  The runtime photon
//                   cutoff is a free parameter the table does not encode, so
//                   this gate, not the table, owns it -- for BOTH species;
//   annihilation    always open (e+ only), at any energy down to at rest.
// One owner for the live electron, live positron and newborn loops: a gate
// drifting at any one of them would let a secondary select a channel its own
// parent could not, or select a channel whose defining product cannot exist.
[[nodiscard]] RREA_HOST_DEVICE inline bool RreaChargedChannelOpenAt(
    rrea::RreaInteractionTablesView const& tables,
    bool is_positron,
    amrex::Real hard_secondary_threshold_eV,
    amrex::Real photon_cutoff_eV,
    int channel,
    amrex::Real energy_eV,
    bool hard_moller_discrete_enabled = true)
{
    if (is_positron) {
        auto const coefficients =
            tables.PositronTransportCoefficients(energy_eV, amrex::Real(1.0));
        switch (channel) {
        case 0:
            return RreaHardBhabhaChannelOpen(
                       energy_eV, hard_secondary_threshold_eV)
                && coefficients.hard_bhabha_inverse_m > amrex::Real(0.0);
        case 1:
            return energy_eV > photon_cutoff_eV
                && coefficients.tracked_brems_inverse_m > amrex::Real(0.0);
        case 2:
            return coefficients.annihilation_inverse_m > amrex::Real(0.0);
        default:
            return false;
        }
    }
    auto const coefficients =
        tables.ElectronTransportCoefficients(energy_eV, amrex::Real(1.0));
    switch (channel) {
    case 0:
        return hard_moller_discrete_enabled
            && RreaHardMollerChannelOpen(
                energy_eV, hard_secondary_threshold_eV)
            && coefficients.hard_moller_inverse_m > amrex::Real(0.0);
    case 1:
        return RreaElectronHardBremsChannelOpen(energy_eV, photon_cutoff_eV)
            && coefficients.tracked_brems_inverse_m > amrex::Real(0.0);
    default:
        return false;
    }
}

// Frozen per-channel STP inverse lengths of the competing discrete hazards at
// one energy, in inverse metres per unit density ratio, in the same channel
// order as the gate above ([2] is identically zero for an electron).  Each
// entry is exactly zero unless its gate is open, and the gate SHORT-CIRCUITS,
// so the competing-risk inversion can never select a closed channel and a
// closed channel never queries a table outside the domain where it is defined.
// Multiply by a segment's density path to get its optical depths: every rate
// is exactly linear in the density ratio, so one lookup per refreeze serves
// the whole chord.
[[nodiscard]] RREA_HOST_DEVICE inline std::array<amrex::Real, 3> RreaChargedCompetingStpRates(
    rrea::RreaInteractionTablesView const& tables,
    bool is_positron,
    amrex::Real hard_secondary_threshold_eV,
    amrex::Real photon_cutoff_eV,
    amrex::Real frozen_energy_eV,
    bool hard_moller_discrete_enabled = true)
{
    if (is_positron) {
        return {
            RreaFrozenStpRate(RreaChannelInverseLength(
                RreaHardBhabhaChannelOpen(
                    frozen_energy_eV, hard_secondary_threshold_eV),
                [&] {
                    return tables.PositronHardBhabhaMeanFreePath(
                        frozen_energy_eV, amrex::Real(1.0));
                })),
            RreaFrozenStpRate(RreaChannelInverseLength(
                frozen_energy_eV > photon_cutoff_eV,
                [&] {
                    return tables.PositronBremsstrahlungMeanFreePath(
                        frozen_energy_eV, amrex::Real(1.0));
                })),
            RreaFrozenStpRate(RreaChannelInverseLength(true, [&] {
                return tables.PositronAnnihilationMeanFreePath(
                    frozen_energy_eV, amrex::Real(1.0));
            }))};
    }
    return {
        RreaFrozenStpRate(RreaChannelInverseLength(
            hard_moller_discrete_enabled
                && RreaHardMollerChannelOpen(
                    frozen_energy_eV, hard_secondary_threshold_eV),
            [&] {
                return tables.ElectronHardMollerMeanFreePath(
                    frozen_energy_eV, amrex::Real(1.0));
            })),
        RreaFrozenStpRate(RreaChannelInverseLength(
            RreaElectronHardBremsChannelOpen(
                frozen_energy_eV, photon_cutoff_eV),
            [&] {
                return tables.ElectronBremsstrahlungMeanFreePath(
                    frozen_energy_eV, amrex::Real(1.0));
            })),
        amrex::Real(0.0)};
}

// Moller e-e- hard knock-on: the delta-ray branch that multiplies the RREA
// cascade.  Moller (1932) dsigma/dT for a free electron at rest, sampled
// through the schema-6 CDF; the two outgoing electrons are indistinguishable,
// so by convention the SECONDARY is the lower-energy one and T <= K/2.  The
// channel's own phase-space gate (RreaHardMollerChannelOpen: K > 2 T_cut)
// makes that clamp and the production threshold consistent, so the secondary
// is always at or above T_cut -- and ReadParameters refuses a configuration
// with T_cut <= E_cut, so it is always a TRACKED electron.  Lab angles come
// from RreaSampleMollerLabDirections (cos^2(theta) = T(K+2mc^2)/(K(T+2mc^2))
// for each outgoing electron, placed on opposite azimuths).
template <class Effects, typename RngKeyFn>
RREA_HOST_DEVICE inline void RreaApplyHardMollerFinalState(
    Effects& fx,
    rrea::RreaInteractionTablesView const& tables,
    RngKeyFn&& rng_key,
    RreaInteractionSite const& site,
    amrex::Real particle_weight,
    amrex::Real secondary_threshold_eV,
    amrex::Real& kinetic_energy_eV,
    amrex::Real& dir_x,
    amrex::Real& dir_y,
    amrex::Real& dir_z)
{
    amrex::Real const primary_before_eV = kinetic_energy_eV;
    amrex::Real secondary_energy_eV = tables.SampleMollerSecondaryEnergy(
        primary_before_eV,
        RreaRng::Uniform01(rng_key(RreaRngChannel::ElectronHardMollerEnergy)));
    auto const& sensitivity = tables.DebugOptions();
    if (sensitivity.enable) {
        secondary_energy_eV = secondary_threshold_eV
            + sensitivity.hard_moller_secondary_excess_energy
                * rrea::kinetic::max(
                    amrex::Real(0.0),
                    secondary_energy_eV - secondary_threshold_eV);
    }
    secondary_energy_eV = rrea::kinetic::max(
        secondary_threshold_eV,
        rrea::kinetic::min(
            secondary_energy_eV, amrex::Real(0.5) * primary_before_eV));
    amrex::Real parent_dir_x = dir_x;
    amrex::Real parent_dir_y = dir_y;
    amrex::Real parent_dir_z = dir_z;
    amrex::Real secondary_dir_x = dir_x;
    amrex::Real secondary_dir_y = dir_y;
    amrex::Real secondary_dir_z = dir_z;
    RreaSampleMollerLabDirections(
        dir_x, dir_y, dir_z,
        primary_before_eV,
        secondary_energy_eV,
        RreaRng::Uniform01(rng_key(RreaRngChannel::ElectronHardMollerAzimuth)),
        parent_dir_x, parent_dir_y, parent_dir_z,
        secondary_dir_x, secondary_dir_y, secondary_dir_z);
    kinetic_energy_eV = rrea::kinetic::max(
        amrex::Real(0.0), primary_before_eV - secondary_energy_eV);
    dir_x = parent_dir_x;
    dir_y = parent_dir_y;
    dir_z = parent_dir_z;
    fx.BeginGroup(RreaSecondaryProcess::HardMoller);
    fx.tally.RecordHardMollerEvent(particle_weight, secondary_energy_eV);
    fx.AddSecondary(RreaSecondarySpecies::Electron, Spawn{
        site.x_m, site.y_m, site.z_m,
        secondary_dir_x, secondary_dir_y, secondary_dir_z,
        particle_weight, secondary_energy_eV, site.elapsed_step_fraction});
    fx.AccumulateIonizationEvent(
        0, site.r_m, site.z_m, particle_weight,
        secondary_energy_eV, secondary_energy_eV);
    fx.tally.RecordTransportIonization(
        particle_weight, secondary_energy_eV, false, true);
    fx.SetInteractionConservationClosure(
        particle_weight, primary_before_eV, kinetic_energy_eV);
    fx.EndGroup();
}

// Bhabha e+e- hard knock-on of a bound atomic electron.  Bhabha (1936)
// dsigma/dT: the annihilation diagram interferes with the scattering one, so
// the spectrum differs from Moller and the two outgoing leptons are
// DISTINGUISHABLE -- there is no T <= K/2 identity restriction and the
// secondary is clamped only by the parent's kinetic energy.  Sampled through
// the schema-6 positron CDF; a sampled secondary below the hard-secondary
// production threshold discards THIS event only (the remaining channels of the
// competing hazard stay eligible), which is why the parent state is left
// untouched on that branch.  Outgoing lab angles reuse the same two-body
// sampler as Moller.  Charge bookkeeping differs from Moller: the target
// electron becomes a tracked secondary while the atom it left keeps a positive
// ion, so the source is a positive ion ONLY.
template <class Effects, typename RngKeyFn>
RREA_HOST_DEVICE inline void RreaApplyBhabhaFinalState(
    Effects& fx,
    rrea::RreaInteractionTablesView const& tables,
    RngKeyFn&& rng_key,
    RreaInteractionSite const& site,
    amrex::Real particle_weight,
    amrex::Real secondary_threshold_eV,
    amrex::Real& kinetic_energy_eV,
    amrex::Real& dir_x,
    amrex::Real& dir_y,
    amrex::Real& dir_z)
{
    amrex::Real const primary_before_eV = kinetic_energy_eV;
    amrex::Real const secondary_energy_eV = rrea::kinetic::min(
        tables.SamplePositronBhabhaSecondaryEnergy(
            primary_before_eV,
            RreaRng::Uniform01(rng_key(RreaRngChannel::PositronBhabhaEnergy))),
        rrea::kinetic::max(amrex::Real(0.0), primary_before_eV));
    if (!(secondary_energy_eV > amrex::Real(0.0))
        || (secondary_threshold_eV > amrex::Real(0.0)
            && secondary_energy_eV < secondary_threshold_eV)) {
        return;
    }
    amrex::Real parent_dir_x = dir_x;
    amrex::Real parent_dir_y = dir_y;
    amrex::Real parent_dir_z = dir_z;
    amrex::Real secondary_dir_x = dir_x;
    amrex::Real secondary_dir_y = dir_y;
    amrex::Real secondary_dir_z = dir_z;
    RreaSampleMollerLabDirections(
        dir_x, dir_y, dir_z,
        primary_before_eV,
        secondary_energy_eV,
        RreaRng::Uniform01(rng_key(RreaRngChannel::PositronBhabhaAzimuth)),
        parent_dir_x, parent_dir_y, parent_dir_z,
        secondary_dir_x, secondary_dir_y, secondary_dir_z);
    kinetic_energy_eV = rrea::kinetic::max(
        amrex::Real(0.0), primary_before_eV - secondary_energy_eV);
    dir_x = parent_dir_x;
    dir_y = parent_dir_y;
    dir_z = parent_dir_z;
    fx.BeginGroup(RreaSecondaryProcess::HardBhabha);
    fx.AddSecondary(RreaSecondarySpecies::Electron, Spawn{
        site.x_m, site.y_m, site.z_m,
        secondary_dir_x, secondary_dir_y, secondary_dir_z,
        particle_weight, secondary_energy_eV, site.elapsed_step_fraction});
    fx.AccumulatePositiveIonSourceOnly(
        0, site.r_m, site.z_m, particle_weight, secondary_energy_eV);
    fx.tally.RecordTransportIonization(
        particle_weight, secondary_energy_eV, false, true);
    fx.tally.RecordPositronBhabhaEvent(
        particle_weight, secondary_energy_eV, true);
    fx.SetInteractionConservationClosure(
        particle_weight, primary_before_eV, kinetic_energy_eV);
    fx.EndGroup();
}

// The tracked brems CDFs interpolate their group tips exactly to the primary
// energy, so a sampled photon above it is roundoff and the clamp inside
// RreaClassifyBremsEnergyDisposition absorbs it silently.  A residual above
// 1e-6 relative means a broken table: warn once (from any OpenMP worker) so it
// is visible rather than silently absorbed.
RREA_HOST_DEVICE inline void RreaWarnOnBremsPhotonOvershoot(
    amrex::Real sampled_photon_energy_eV,
    amrex::Real parent_kinetic_energy_eV)
{
    if (!(sampled_photon_energy_eV
          > parent_kinetic_energy_eV
              * (amrex::Real(1.0) + amrex::Real(1.0e-6)))) {
        return;
    }
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    printf("RREA brems warning: photon overshoot >1e-6; clamped as on CPU\n");
#else
    static std::atomic<bool> warned_brems_overshoot{false};
    if (!warned_brems_overshoot.exchange(true)) {
        std::fputs(
            "RREA brems: sampled photon energy exceeds the parent kinetic "
            "energy by > 1e-6 (clamped); check the brems photon-energy CDF\n", stderr);
    }
#endif
}

// Bremsstrahlung on air nuclei and atomic electrons, e- and e+ selected by
// charge_sign.  Seltzer-Berger photon-energy DCS through the schema-6 CDF --
// with a separately measured positron CDF, because the electron SB shape
// overestimates the e+ hard tip -- and a shared bounded ModifiedTsai polar
// distribution, independent of photon energy. The charged direction follows from momentum recoil
// p' = p - k (RreaApplyPhotonRecoilToChargedDirection), and
// RreaClassifyBremsEnergyDisposition is the single owner of the quantum's
// fate: at or above the transport cutoff it is tracked, below it thermalizes
// locally, and the parent loses exactly the emitted energy either way.
// deposit_local(energy_eV) is the caller's local ion-pair sink at this site.
template <class Effects, typename RngKeyFn, typename DepositLocal>
RREA_HOST_DEVICE inline void RreaApplyBremsFinalState(
    Effects& fx,
    rrea::RreaInteractionTablesView const& tables,
    RngKeyFn&& rng_key,
    DepositLocal&& deposit_local,
    RreaInteractionSite const& site,
    amrex::Real particle_weight,
    amrex::Real charge_sign,
    amrex::Real photon_cutoff_eV,
    amrex::Real& kinetic_energy_eV,
    amrex::Real& dir_x,
    amrex::Real& dir_y,
    amrex::Real& dir_z)
{
    bool const is_positron = charge_sign > amrex::Real(0.0);
    amrex::Real const parent_before_eV = kinetic_energy_eV;
    amrex::Real const energy_u = RreaRng::Uniform01(rng_key(
        is_positron ? RreaRngChannel::PositronBremsPhotonEnergy
                    : RreaRngChannel::BremsPhotonEnergy));
    amrex::Real const sampled_photon_energy_eV = is_positron
        ? tables.SamplePositronBremsstrahlungPhotonEnergy(
            parent_before_eV, energy_u)
        : tables.SampleBremsstrahlungPhotonEnergy(parent_before_eV, energy_u);
    RreaWarnOnBremsPhotonOvershoot(
        sampled_photon_energy_eV, parent_before_eV);
    auto const disposition = RreaClassifyBremsEnergyDisposition(
        parent_before_eV, sampled_photon_energy_eV, photon_cutoff_eV);
    amrex::Real const photon_energy_eV = disposition.photon_energy_eV;
    amrex::Real photon_dir_x = dir_x;
    amrex::Real photon_dir_y = dir_y;
    amrex::Real photon_dir_z = dir_z;
    RreaSampleConeDirection(
        dir_x, dir_y, dir_z,
        tables.SampleBremsstrahlungCosTheta(
            parent_before_eV,
            RreaRng::Uniform01(rng_key(
                is_positron ? RreaRngChannel::PositronBremsPhotonAngle
                            : RreaRngChannel::BremsPhotonAngle))),
        RreaRng::Uniform01(rng_key(
            is_positron ? RreaRngChannel::PositronBremsPhotonAzimuth
                        : RreaRngChannel::BremsPhotonAzimuth)),
        photon_dir_x, photon_dir_y, photon_dir_z);
    RreaApplyPhotonRecoilToChargedDirection(
        parent_before_eV, photon_energy_eV,
        photon_dir_x, photon_dir_y, photon_dir_z,
        dir_x, dir_y, dir_z);
    kinetic_energy_eV = disposition.parent_energy_after_eV;
    fx.BeginGroup(RreaSecondaryProcess::Bremsstrahlung);
    if (is_positron) {
        fx.tally.RecordPositronBremsPhoton(particle_weight, photon_energy_eV);
    } else {
        fx.tally.RecordBremsPhoton(particle_weight, photon_energy_eV);
    }
    if (disposition.commit_photon) {
        fx.AddSecondary(RreaSecondarySpecies::Photon, Spawn{
            site.x_m, site.y_m, site.z_m,
            photon_dir_x, photon_dir_y, photon_dir_z,
            particle_weight, photon_energy_eV, site.elapsed_step_fraction});
    } else if (disposition.local_deposit_energy_eV > amrex::Real(0.0)) {
        deposit_local(disposition.local_deposit_energy_eV);
    }
    fx.SetInteractionConservationClosure(
        particle_weight, parent_before_eV, kinetic_energy_eV);
    fx.EndGroup();
}

}  // namespace rrea::warpx

#endif
