#ifndef RREA_PHOTON_FINAL_STATES_H
#define RREA_PHOTON_FINAL_STATES_H

// Discrete final states of a tracked photon: Compton, photoelectric absorption
// and pair production (nuclear and triplet), one named module each, used by
// the one photon history (AdvancePhotonHistory) for live and newborn photons.  Coherent
// (Rayleigh) scattering is deliberately EXCLUDED from this transport; near-cut
// angular transport and feedback must be read with that scope in mind.
//
// Each module books the whole event, and it books the photon's energy flow
// through the RreaComptonPhotonEnergyFlow / RreaPhotoelectricPhotonEnergyFlow /
// RreaPairPhotonEnergyFlow splits in RreaParticleInteraction.H.  Those splits
// are the one owner of the booked photon flow (the photon_to_charged /
// photon_local_deposit ledger columns, closed only as a reduced-CSV
// diagnostic); the transaction validator separately checks each group's
// charge and energy closure from its particles, sources and closure record.
//
// RNG identity stays with the caller: rng_key(channel) -> key, on the photon's
// own stream when live and on its root ancestor's stream at the packed
// transient index (its own lineage) when newborn.  deposit_local(energy_eV) is the caller's
// local ion-pair sink at this site.
//
// Pinned by rrea_secondary_group_smoke (group/conservation contracts),
// rrea_scattering_sampler_smoke (the samplers these call) and
// rrea_warpx_transport_apply (the assembled step).

#include "RreaChargedInterleave.H"
#include "RreaParticleInteraction.H"
#include "RreaRng.H"
#include "RreaTransportLog.H"

#include "rrea/RreaInteractionPhysics.H"


#include <AMReX_REAL.H>

#include <algorithm>
#include <iomanip>
#include <sstream>

namespace rrea::warpx {

// One photon-freed electron (Compton recoil or photoelectron).  At or above
// the charged cutoff it is a tracked secondary with its primary ionization
// event.  Below it, the primary ionization that freed it is one pair and its
// kinetic energy thermalizes into E/W_air cascade pairs through
// deposit_local: the low-energy fluid consumes mesh carriers at face value,
// so without the cascade the yield would be short by up to E_cutoff/W_air.
template <class Effects, typename DepositLocal>
RREA_HOST_DEVICE inline void RreaEmitPhotonFreedElectron(
    Effects& fx,
    DepositLocal&& deposit_local,
    RreaInteractionSite const& site,
    amrex::Real particle_weight,
    amrex::Real energy_eV,
    bool tracked,
    amrex::Real dir_x,
    amrex::Real dir_y,
    amrex::Real dir_z)
{
    fx.AccumulateIonizationEvent(site.r_m, site.z_m, particle_weight, energy_eV);
    if (tracked) {
        fx.AddSecondary(RreaSecondarySpecies::Electron, Spawn{
            site.x_m, site.y_m, site.z_m, dir_x, dir_y, dir_z,
            particle_weight, energy_eV, site.elapsed_step_fraction});
    } else {
        deposit_local(energy_eV);
    }
    fx.tally.RecordTransportIonization(
        particle_weight, amrex::Real(0.0), !tracked, tracked);
}

// Compton scattering off a quasi-free atomic electron.  Klein-Nishina (1929)
// dsigma/dOmega through the schema-6 final-state sampler: the scattered photon
// takes E' = E / (1 + (E/mc^2)(1 - cos theta)) and the recoil electron the
// remainder, its direction fixed by exact momentum conservation
// p_e = k - k'.  The scattered photon's fate (tracked above the photon cutoff,
// thermalized below it) and the recoil's (tracked above the charged cutoff,
// otherwise a local ion-pair cascade) are the SAME two comparisons
// RreaComptonPhotonEnergyFlow partitions.
//
// Returns true when the scattered photon is still above the photon cutoff --
// the caller then owns what that survivor does with the rest of its chord.  On
// false the group has already removed the parent and deposited the residual
// energy locally.  photon_energy_eV and the direction are updated in place.
template <class Effects, typename RngKeyFn, typename DepositLocal>
[[nodiscard]] RREA_HOST_DEVICE inline bool RreaApplyComptonFinalState(
    Effects& fx,
    rrea::RreaInteractionTablesView const& tables,
    RngKeyFn&& rng_key,
    DepositLocal&& deposit_local,
    RreaInteractionSite const& site,
    amrex::Real particle_weight,
    amrex::Real low_energy_cutoff_eV,
    amrex::Real photon_cutoff_eV,
    amrex::Real& photon_energy_eV,
    amrex::Real& dir_x,
    amrex::Real& dir_y,
    amrex::Real& dir_z)
{
    amrex::Real const incident_eV = photon_energy_eV;
    auto const state = tables.SampleComptonKleinNishinaFinalState(
        incident_eV,
        RreaRng::Uniform01(rng_key(RreaRngChannel::ComptonRecoil)),
        RreaRng::Uniform01(rng_key(RreaRngChannel::ComptonScatter)));
    amrex::Real scattered_dir_x;
    amrex::Real scattered_dir_y;
    amrex::Real scattered_dir_z;
    RreaSampleConeDirection(
        dir_x, dir_y, dir_z, state.cos_theta,
        RreaRng::Uniform01(rng_key(RreaRngChannel::ComptonScatterAzimuth)),
        scattered_dir_x, scattered_dir_y, scattered_dir_z);
    amrex::Real recoil_dir_x = incident_eV * dir_x
        - state.scattered_photon_energy_eV * scattered_dir_x;
    amrex::Real recoil_dir_y = incident_eV * dir_y
        - state.scattered_photon_energy_eV * scattered_dir_y;
    amrex::Real recoil_dir_z = incident_eV * dir_z
        - state.scattered_photon_energy_eV * scattered_dir_z;
    RreaNormalizeDirection(recoil_dir_x, recoil_dir_y, recoil_dir_z);
    bool const recoil_tracked =
        state.recoil_electron_energy_eV >= low_energy_cutoff_eV;
    fx.BeginGroup(
        RreaSecondaryProcess::Compton,
        state.scattered_photon_energy_eV < photon_cutoff_eV);
    RreaEmitPhotonFreedElectron(
        fx, deposit_local, site, particle_weight,
        state.recoil_electron_energy_eV, recoil_tracked,
        recoil_dir_x, recoil_dir_y, recoil_dir_z);
    auto const flow = RreaComptonPhotonEnergyFlow(
        state.recoil_electron_energy_eV,
        state.scattered_photon_energy_eV,
        low_energy_cutoff_eV,
        photon_cutoff_eV);
    fx.RecordComptonInteraction(
        particle_weight, incident_eV,
        state.recoil_electron_energy_eV,
        state.scattered_photon_energy_eV,
        flow.charged_kinetic_rest_mass_eV,
        flow.local_recoil_binding_subcutoff_eV);
    photon_energy_eV = state.scattered_photon_energy_eV;
    if (photon_energy_eV < photon_cutoff_eV) {
        // The same coupled draw supplies the recoil and the outgoing photon;
        // once the photon falls below the transport cutoff its residual energy
        // is local.
        deposit_local(photon_energy_eV);
        fx.SetInteractionConservationClosure(
            particle_weight, incident_eV, amrex::Real(0.0));
        fx.EndGroup();
        return false;
    }
    fx.SetInteractionConservationClosure(
        particle_weight, incident_eV, photon_energy_eV);
    fx.EndGroup();
    dir_x = scattered_dir_x;
    dir_y = scattered_dir_y;
    dir_z = scattered_dir_z;
    return true;
}

// Photoelectric absorption: the photon is consumed whole, one bound electron
// leaves with E_gamma - E_binding and the atom keeps the binding energy.  The
// subshell and its binding energy come from the schema-6 final state; the
// photoelectron's polar angle is the forward-boosted Sauter-Gavrila
// distribution (analytic inverse CDF on one draw) about the incident
// direction.  The full binding energy is deposited locally: the N and O K
// edges are below 1 keV and the argon K edge (3.2 keV) is far below the photon
// cutoff, so fluorescence (at most a few keV, millimetre range) is deliberately
// not modelled and treating the binding energy as local thermalization is
// exact at the grid scale.  Depositing binding (rather than the tabulated
// relaxation) is also what closes the ledger when a subshell row carries
// relaxation < binding.  Terminal: the parent photon is always removed.
template <class Effects, typename RngKeyFn, typename DepositLocal>
RREA_HOST_DEVICE inline void RreaApplyPhotoelectricFinalState(
    Effects& fx,
    rrea::RreaInteractionTablesView const& tables,
    RngKeyFn&& rng_key,
    DepositLocal&& deposit_local,
    RreaInteractionSite const& site,
    amrex::Real particle_weight,
    amrex::Real low_energy_cutoff_eV,
    amrex::Real photon_energy_eV,
    amrex::Real dir_x,
    amrex::Real dir_y,
    amrex::Real dir_z)
{
    auto const state = tables.SamplePhotoelectricFinalState(
        photon_energy_eV,
        RreaRng::Uniform01(rng_key(RreaRngChannel::Photoelectric)));
    amrex::Real const electron_energy_eV = rrea::kinetic::min(
        state.photoelectron_energy_eV, photon_energy_eV);
    bool const photoelectron_tracked =
        electron_energy_eV >= low_energy_cutoff_eV;
    fx.BeginGroup(RreaSecondaryProcess::Photoelectric, true);
    amrex::Real electron_dir_x = dir_x;
    amrex::Real electron_dir_y = dir_y;
    amrex::Real electron_dir_z = dir_z;
    if (photoelectron_tracked) {
        RreaSampleConeDirection(
            dir_x, dir_y, dir_z,
            tables.SamplePhotoelectronCosTheta(
                electron_energy_eV,
                RreaRng::Uniform01(
                    rng_key(RreaRngChannel::PhotoelectricAngle))),
            RreaRng::Uniform01(rng_key(RreaRngChannel::PhotoelectricAzimuth)),
            electron_dir_x, electron_dir_y, electron_dir_z);
    }
    RreaEmitPhotonFreedElectron(
        fx, deposit_local, site, particle_weight,
        electron_energy_eV, photoelectron_tracked,
        electron_dir_x, electron_dir_y, electron_dir_z);
    if (state.binding_energy_eV > amrex::Real(0.0)) {
        deposit_local(state.binding_energy_eV);
    }
    auto const flow = RreaPhotoelectricPhotonEnergyFlow(
        electron_energy_eV, state.binding_energy_eV, low_energy_cutoff_eV);
    fx.RecordPhotoelectricInteraction(
        particle_weight, photon_energy_eV, electron_energy_eV,
        state.binding_energy_eV,
        flow.charged_kinetic_rest_mass_eV,
        flow.local_recoil_binding_subcutoff_eV);
    fx.SetInteractionConservationClosure(
        particle_weight, photon_energy_eV, amrex::Real(0.0));
    fx.EndGroup();
}

// Pair production, nuclear (Bethe-Heitler in the screened nuclear field) and
// triplet (conversion in the field of an atomic electron, which recoils and
// leaves a positive ion).  The schema-6 final state supplies the pair
// invariant mass, the pair opening geometry about the photon direction, the
// triplet recoil, and whatever residual energy the kinematics leave local.
// Nuclear conversion is charge-neutral and creates NO air positive ion; a
// below-cutoff pair electron still joins the low-energy mesh even at exactly
// zero kinetic energy, because dropping the zero endpoint breaks samplewise
// charge closure at threshold.  A below-cutoff positron is never dropped: it
// thermalizes and annihilates through the caller's
// annihilate_subcutoff_positron(dir_x, dir_y, dir_z, kinetic_eV), the same
// policy transported positrons use.  Terminal: the parent photon is always
// removed.
template <class Effects, typename RngKeyFn, typename DepositLocal, typename AnnihilatePositron>
RREA_HOST_DEVICE inline void RreaApplyPairFinalState(
    Effects& fx,
    rrea::RreaInteractionTablesView const& tables,
    RngKeyFn&& rng_key,
    DepositLocal&& deposit_local,
    AnnihilatePositron&& annihilate_subcutoff_positron,
    RreaInteractionSite const& site,
    amrex::Real particle_weight,
    amrex::Real low_energy_cutoff_eV,
    amrex::Real photon_energy_eV,
    RreaPhotonChannel channel,
    amrex::Real dir_x,
    amrex::Real dir_y,
    amrex::Real dir_z)
{
    auto const pair = tables.SamplePairFinalState(
        photon_energy_eV, channel,
        RreaRng::Uniform01(rng_key(RreaRngChannel::PairInvariant)),
        RreaRng::Uniform01(rng_key(RreaRngChannel::PairAngle)),
        RreaRng::Uniform01(rng_key(RreaRngChannel::PairAzimuth)));
    if (!pair.valid) {
        rrea::KineticFail("RREA pair-production sampler returned no valid final state");
    }
    if (channel == RreaPhotonChannel::PairTriplet
        && (!pair.has_recoil_electron || !pair.creates_target_ion)) {
        rrea::KineticFail(
            "RREA triplet final state omitted the recoil electron or "
            "target-ion source; refusing a non-conserving event");
    }
    fx.BeginGroup(
        pair.channel == RreaPhotonChannel::PairTriplet
            ? RreaSecondaryProcess::PairTriplet
            : RreaSecondaryProcess::PairNuclear,
        true);
    amrex::Real electron_dir_x;
    amrex::Real electron_dir_y;
    amrex::Real electron_dir_z;
    RreaRotateLocalDirectionToAxis(
        dir_x, dir_y, dir_z,
        pair.electron_dir_x, pair.electron_dir_y, pair.electron_dir_z,
        electron_dir_x, electron_dir_y, electron_dir_z);
    amrex::Real positron_dir_x;
    amrex::Real positron_dir_y;
    amrex::Real positron_dir_z;
    RreaRotateLocalDirectionToAxis(
        dir_x, dir_y, dir_z,
        pair.positron_dir_x, pair.positron_dir_y, pair.positron_dir_z,
        positron_dir_x, positron_dir_y, positron_dir_z);
    if (pair.electron_energy_eV >= low_energy_cutoff_eV) {
        fx.AddSecondary(RreaSecondarySpecies::Electron, Spawn{
            site.x_m, site.y_m, site.z_m,
            electron_dir_x, electron_dir_y, electron_dir_z,
            particle_weight, pair.electron_energy_eV,
            site.elapsed_step_fraction});
    } else {
        // Mesh diagnostic energy is zero, matching the demotion convention;
        // the kinetic energy thermalizes into E/W_air cascade pairs.
        fx.AccumulateLowEnergyElectron(site.r_m, site.z_m, particle_weight);
        deposit_local(pair.electron_energy_eV);
    }
    if (pair.positron_energy_eV >= low_energy_cutoff_eV) {
        fx.AddSecondary(RreaSecondarySpecies::Positron, Spawn{
            site.x_m, site.y_m, site.z_m,
            positron_dir_x, positron_dir_y, positron_dir_z,
            particle_weight, pair.positron_energy_eV,
            site.elapsed_step_fraction});
    } else {
        annihilate_subcutoff_positron(
            positron_dir_x, positron_dir_y, positron_dir_z,
            pair.positron_energy_eV);
        fx.tally.RecordPositronTransport(
            particle_weight, pair.positron_energy_eV, true);
    }
    if (pair.has_recoil_electron) {
        amrex::Real recoil_dir_x;
        amrex::Real recoil_dir_y;
        amrex::Real recoil_dir_z;
        RreaRotateLocalDirectionToAxis(
            dir_x, dir_y, dir_z,
            pair.recoil_dir_x, pair.recoil_dir_y, pair.recoil_dir_z,
            recoil_dir_x, recoil_dir_y, recoil_dir_z);
        if (pair.recoil_electron_energy_eV >= low_energy_cutoff_eV) {
            fx.AddSecondary(RreaSecondarySpecies::Electron, Spawn{
                site.x_m, site.y_m, site.z_m,
                recoil_dir_x, recoil_dir_y, recoil_dir_z,
                particle_weight, pair.recoil_electron_energy_eV,
                site.elapsed_step_fraction});
        }
        // Conversion in the electron field ionizes the target, leaving one
        // local positive ion (diagnostic energy zero).  A sub-cutoff recoil
        // has no kinetic macroparticle, so it also thermalizes.
        fx.AccumulateIonizationEvent(
            site.r_m, site.z_m, particle_weight, pair.recoil_electron_energy_eV);
        if (pair.recoil_electron_energy_eV < low_energy_cutoff_eV) {
            deposit_local(pair.recoil_electron_energy_eV);
        }
    }
    if (pair.local_deposit_energy_eV > amrex::Real(0.0)) {
        deposit_local(pair.local_deposit_energy_eV);
    }
    auto const flow = RreaPairPhotonEnergyFlow(
        pair.electron_energy_eV, pair.positron_energy_eV,
        pair.recoil_electron_energy_eV, pair.local_deposit_energy_eV,
        low_energy_cutoff_eV, pair.has_recoil_electron);
    fx.RecordPairProduction(
        particle_weight, photon_energy_eV,
        pair.electron_energy_eV, pair.positron_energy_eV,
        pair.recoil_electron_energy_eV,
        flow.charged_kinetic_rest_mass_eV,
        flow.local_recoil_binding_subcutoff_eV,
        pair.channel == RreaPhotonChannel::PairTriplet);
    fx.SetInteractionConservationClosure(
        particle_weight, photon_energy_eV, amrex::Real(0.0));
    fx.EndGroup();
}

}  // namespace rrea::warpx

#endif
