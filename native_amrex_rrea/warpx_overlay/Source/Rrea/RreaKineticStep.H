#ifndef RREA_KINETIC_STEP_H
#define RREA_KINETIC_STEP_H
#include "RreaChargedFinalStates.H"
#include "RreaChargedInterleave.H"
#include "RreaMaterialPath.H"
#include "RreaParticleInteraction.H"
#include "RreaPathFieldWork.H"
#include "RreaPhotonFinalStates.H"
#include "RreaRng.H"
#include "RreaTransportLog.H"
#include "RreaKineticEnvironment.H"
#include "rrea/RreaInteractionPhysics.H"
#include "rrea/RreaPlaneFlux.H"
#include "rrea/RreaLorentzPush.H"
#include "rrea/RreaSeedSchedule.H"

namespace rrea::warpx {

// Substeps needed to keep `amount` at or below `per_substep` in each one, never
// fewer than one.
RREA_HOST_DEVICE inline std::uint64_t substep_driver_count(amrex::Real amount, amrex::Real per_substep)
{
    return static_cast<std::uint64_t>(
        rrea::kinetic::max(amrex::Real(1.0), std::ceil(amount / per_substep)));
}

// Cartesian point plus cylindrical radius at a full-path fraction of one chord.
RREA_HOST_DEVICE inline void point_and_radius_at_fraction(
    RreaTransportPathSegment const& path,
    amrex::Real path_fraction,
    amrex::Real& event_x,
    amrex::Real& event_y,
    amrex::Real& event_z,
    amrex::Real& event_r)
{
    path.PointAtFullPathFraction(path_fraction, event_x, event_y, event_z);
    event_r = std::hypot(event_x, event_y);
}

RREA_HOST_DEVICE inline std::uint64_t selected_subcycles(
    amrex::Real max_tau,
    RreaParticleInteractionConfig const& config,
    bool& cap_violation)
{
    amrex::Real const target = config.max_optical_depth_per_substep;
    std::uint64_t const requested = static_cast<std::uint64_t>(
        rrea::kinetic::max(amrex::Real(1.0), std::ceil(max_tau / target)));
    std::uint64_t const cap = config.max_subcycles;
    cap_violation = requested > cap;
    return rrea::kinetic::min(requested, cap);
}

// Substeps small enough that no uniform slice of the chord carries more than
// the configured competing optical depth.  total_stp_per_m is the summed
// frozen inverse length of every channel in the competition, so the worst
// slice is that times the slice's density path -- one geometric measurement
// per candidate count, independent of how many channels compete.
RREA_HOST_DEVICE inline std::uint64_t selected_material_subcycles(
    amrex::Real max_total_tau,
    MaterialPathInterval const& material,
    RreaTransportPathSegment const& path,
    RreaParticleInteractionConfig const& config,
    bool& cap_violation,
    amrex::Real total_stp_per_m)
{
    std::uint64_t subcycles = selected_subcycles(
        max_total_tau, config, cap_violation);
    amrex::Real const target = config.max_optical_depth_per_substep;
    std::uint64_t const cap = config.max_subcycles;
    for (;;) {
        amrex::Real const maximum = total_stp_per_m
            * max_uniform_subcycle_density_path_m(material, path, subcycles);
        amrex::Real const tolerance = amrex::Real(64.0)
            * std::numeric_limits<amrex::Real>::epsilon()
            * rrea::kinetic::max(amrex::Real(1.0), target);
        if (maximum <= target + tolerance) {
            return subcycles;
        }
        if (subcycles >= cap) {
            cap_violation = true;
            return subcycles;
        }
        rrea::RreaKineticWide const estimate = std::ceil(
            static_cast<rrea::RreaKineticWide>(subcycles)
            * static_cast<rrea::RreaKineticWide>(maximum / target));
        std::uint64_t next = subcycles + 1U;
        if (estimate >= static_cast<rrea::RreaKineticWide>(cap)) {
            next = cap;
        } else if (estimate > static_cast<rrea::RreaKineticWide>(next)) {
            next = static_cast<std::uint64_t>(estimate);
        }
        subcycles = rrea::kinetic::min(next, cap);
    }
}

// Elastic COLLISION rate at STP, 1/lambda_el(E), in inverse metres.  The
// Goudsmit-Saunderson path variable is n = s/lambda_el = this times the
// segment's density path.
RREA_HOST_DEVICE inline amrex::Real elastic_inverse_length_stp(
    rrea::RreaInteractionTablesView const& tables,
    bool is_positron,
    amrex::Real kinetic_energy_eV)
{
    amrex::Real const mfp = tables.ElasticMeanFreePath(
        is_positron, kinetic_energy_eV, amrex::Real(1.0));
    return RreaFrozenStpRate(
        mfp > amrex::Real(0.0) ? amrex::Real(1.0) / mfp : amrex::Real(0.0));
}

// Coleman & Dwyer plane-crossing observable of one electron drag segment:
// every nonperiodic crossing of a C&D plane is deferred with the energy at
// the crossing (the segment's frozen realised drag integrated to it, capped at
// the segment's actual loss) and whether |E| there exceeds the local
// ensemble drag (RreaLocalRunawayFieldMagnitudeGate).
template<class Effects>
RREA_HOST_DEVICE inline void defer_cd_electron_plane_crossings(
    Effects& effects,
    KineticEnvironment const& coupling,
    rrea::RreaInteractionTablesView const& tables,
    RreaTransportPathSegment const& path,
    MaterialPathInterval const& material,
    amrex::Real begin_fraction,
    amrex::Real end_fraction,
    amrex::Real particle_weight,
    amrex::Real kinetic_energy_before_segment_eV,
    amrex::Real stopping_energy_eV,
    amrex::Real segment_drag_loss_eV,
    bool hard_moller_discrete_enabled)
{
    if (!coupling.CdPlaneDiagnosticsEnabled() || !(end_fraction > begin_fraction)) {
        return;
    }
    amrex::Real x_m = amrex::Real(0.0);
    amrex::Real y_m = amrex::Real(0.0);
    amrex::Real begin_z = amrex::Real(0.0);
    amrex::Real end_z = amrex::Real(0.0);
    path.PointAtFullPathFraction(begin_fraction, x_m, y_m, begin_z);
    path.PointAtFullPathFraction(end_fraction, x_m, y_m, end_z);
    amrex::Real const fixed_energy_stp_eV_per_m =
        tables.ChargedRealizedDrag(false, stopping_energy_eV, amrex::Real(1.0));

    bool const valid = RreaForEachNonperiodicPlaneCrossing(
        begin_z,
        end_z,
        coupling.CdPlaneZ(),
        [&](std::size_t plane_index,
            amrex::Real,
            amrex::Real segment_fraction,
            int direction) {
            amrex::Real const crossing_fraction = begin_fraction
                + segment_fraction * (end_fraction - begin_fraction);
            amrex::Real z_m = amrex::Real(0.0);
            path.PointAtFullPathFraction(crossing_fraction, x_m, y_m, z_m);
            amrex::Real const radial_position_m = std::hypot(x_m, y_m);
            amrex::Real const raw_loss_to_crossing_eV =
                fixed_energy_stp_eV_per_m
                * density_path_m(
                    material, path, begin_fraction, crossing_fraction);
            amrex::Real const kinetic_energy_at_crossing_eV = rrea::kinetic::max(
                amrex::Real(0.0),
                kinetic_energy_before_segment_eV
                    - rrea::kinetic::min(segment_drag_loss_eV, raw_loss_to_crossing_eV));

            bool local_runaway = false;
            if (kinetic_energy_at_crossing_eV >= coupling.LowEnergyCutoffEv()) {
                amrex::Real const density_ratio =
                    coupling.TransportDensityRatioAtZ(z_m);
                RreaFieldSample const field =
                    coupling.GatherFieldRZ(radial_position_m, z_m);
                local_runaway = RreaLocalRunawayFieldMagnitudeGate(
                    field.er_v_per_m,
                    field.ez_v_per_m,
                    tables.ElectronEnsembleDrag(
                        kinetic_energy_at_crossing_eV,
                        density_ratio,
                        hard_moller_discrete_enabled));
            }
            effects.DeferCdElectronPlaneCrossing(
                plane_index,
                direction,
                particle_weight,
                radial_position_m,
                local_runaway,
                kinetic_energy_at_crossing_eV >= amrex::Real(1.0e6));
        });
    if (!valid) {
        rrea::KineticFail("RREA C&D plane crossing enumeration received invalid geometry");
    }
}

// RNG channels for one Goudsmit-Saunderson elastic stream: live electrons
// 89-93, live positrons 94-98, newborns 112-121.
struct GsScatterChannels {
    RreaRngChannel few_count;
    RreaRngChannel few_scatter;
    RreaRngChannel few_azimuth;
    RreaRngChannel angle;
    RreaRngChannel azimuth;
};

// Goudsmit-Saunderson elastic scatter over one sub-interval -- the one
// application unit of every charged history. Below the few-collision threshold the
// deflections are drawn one collision at a time from a Poisson count; above it
// a single GS angle covers the whole interval.  The selection rate stays frozen
// at the caller's energy.  RNG identity is a caller policy: gs_key maps
// (channel, inner draw index) to the key -- a live parent packs
// (substep << 8 | inner) on its own stream, a newborn packs its transient
// lineage on the root stream with its Newborn* channels.
template <typename KeyFn>
RREA_HOST_DEVICE void apply_gs_elastic_scatter(
    MaterialPathInterval const& material,
    RreaTransportPathSegment const& path,
    amrex::Real begin_fraction,
    amrex::Real end_fraction,
    rrea::RreaInteractionTablesView const& tables,
    bool is_positron,
    GsScatterChannels const& channels,
    KeyFn&& gs_key,
    amrex::Real kinetic_energy_eV,
    amrex::Real& dir_x,
    amrex::Real& dir_y,
    amrex::Real& dir_z)
{
    amrex::Real const gs_n = elastic_inverse_length_stp(
            tables, is_positron, kinetic_energy_eV)
        * density_path_m(material, path, begin_fraction, end_fraction);
    if (!(gs_n > amrex::Real(0.0))) {
        return;
    }
    auto deflect = [&](amrex::Real cos_t, amrex::Real u_phi) {
        amrex::Real nx, ny, nz;
        RreaSampleConeDirection(dir_x, dir_y, dir_z, cos_t, u_phi, nx, ny, nz);
        dir_x = nx; dir_y = ny; dir_z = nz;
    };
    if (gs_n < tables.GsFewCollisionThreshold()) {
        int const poisson_cap = static_cast<int>(std::ceil(
            amrex::Real(3.0) * tables.GsFewCollisionThreshold()));
        if (poisson_cap >= 255) {
            rrea::KineticFail(
                "RREA GS few-collision cap cannot pack into the "
                "8-bit interaction-index field");
        }
        int const few_count = RreaPoissonSampleCapped(
            gs_n,
            RreaRng::Uniform01(gs_key(channels.few_count, 0)),
            poisson_cap);
        for (int k = 0; k < few_count; ++k) {
            amrex::Real const cos_t = tables.SampleSingleElasticCosTheta(
                is_positron, kinetic_energy_eV,
                RreaRng::Uniform01(gs_key(
                    channels.few_scatter, static_cast<std::uint64_t>(k + 1))));
            deflect(cos_t, RreaRng::Uniform01(gs_key(
                channels.few_azimuth, static_cast<std::uint64_t>(k + 1))));
        }
    } else {
        amrex::Real const cos_t = tables.SampleGsCosTheta(
            is_positron, kinetic_energy_eV, gs_n,
            RreaRng::Uniform01(gs_key(channels.angle, 0)));
        deflect(cos_t, RreaRng::Uniform01(gs_key(channels.azimuth, 0)));
    }
}

// PHASE MAP (grep the named waypoints; line numbers rot).  Every particle
// history is ONE KineticStep method acting on a ParticleHistory; adapters
// only move state between it and WarpX (live) or the transient payload
// (newborn), so live and newborn particles cannot drift apart.
//   AdvanceChargedHistory -- e-/e+: substep count (optical-depth, GS and
//       impulse/drag drivers), then per substep chord rebuild, half-kick,
//       RreaRunChargedEventInterleave (drag, Moller/Bhabha, brems, in-flight
//       annihilation), GS, half-kick, commit; terminals via
//       RreaResolveChargedTransportTerminal.
//   AdvancePhotonHistory  -- exponential hazard per chord, Compton
//       continuation, photoelectric / pair absorption, clipped escape.
//   Charged / Photon      -- live adapters (RreaSpeciesTransport.cpp).
//   AdvanceNewborn        -- transient adapter (RreaSecondaryCommit.cpp):
//       the residual (1-b)dt on the root's RNG stream.
// The physics they call are single modules: RreaPlanContinuousDragSegment,
// apply_gs_elastic_scatter, RreaApplyChargedFieldHalfKick, the final-state
// headers, spawn_positron_annihilation, deposit_local_energy_as_ion_pairs.

// One tile's SoA pointers for the live adapters (photons leave the
// previous-momentum pointers null).
struct ChargedSoAView {
    amrex::ParticleReal* r;
    amrex::ParticleReal* z;
    amrex::ParticleReal* w;
    amrex::ParticleReal* ux;
    amrex::ParticleReal* uy;
    amrex::ParticleReal* uz;
    amrex::ParticleReal* theta;
    amrex::ParticleReal* previous_r;
    amrex::ParticleReal* previous_z;
    amrex::ParticleReal* previous_theta;
    amrex::ParticleReal* previous_ux;
    amrex::ParticleReal* previous_uy;
    amrex::ParticleReal* previous_uz;
    amrex::ParticleReal* electron_counted = nullptr;
};

// One particle's state through one transport history, live or newborn.
// energy_eV is kinetic for e-/e+ and total for a photon, as
// RreaSecondaryParticle carries it.
struct ParticleHistory {
    RreaSecondarySpecies species = RreaSecondarySpecies::Electron;
    bool newborn = false;
    amrex::Real weight = amrex::Real(0.0);
    amrex::Real x_m = amrex::Real(0.0);
    amrex::Real y_m = amrex::Real(0.0);
    amrex::Real z_m = amrex::Real(0.0);
    amrex::Real dir_x = amrex::Real(0.0);
    amrex::Real dir_y = amrex::Real(0.0);
    amrex::Real dir_z = amrex::Real(1.0);
    amrex::Real energy_eV = amrex::Real(0.0);
    // Elapsed outer-step fraction at birth; a live parent (0) is owed the
    // whole step.
    amrex::Real birth_elapsed = amrex::Real(0.0);
    // RNG identity: a live parent draws on its own key, a newborn on its
    // root's key within its transient lineage.
    std::uint64_t rng_particle_id = 0;
    std::uint64_t transient_lineage = 0;
    RreaSecondarySpecies root_species = RreaSecondarySpecies::Electron;
    bool rrea_owns_escape = false;  // out: RREA, not WarpX, removes the escaper
    bool scattered = false;         // out: a photon left its straight line
    bool electron_counted = false;
};
enum class HistoryEnd { Survive, Escape, Consumed };

// Scalar geometry copied from AMReX at the step boundary. No host Geometry
// object or WarpX singleton is dereferenced by a particle kernel.
struct KineticGeometry {
    amrex::Real lo[2]{0,0},hi[2]{0,0};
    RREA_HOST_DEVICE amrex::Real ProbLo(int d) const {return lo[d];}
    RREA_HOST_DEVICE amrex::Real ProbHi(int d) const {return hi[d];}
};
struct KineticStep {
    rrea::RreaInteractionTablesView tables;
    KineticEnvironment coupling;
    RreaParticleInteractionConfig m_config;
    int step=0;
    amrex::Real time_s=0,dt_s=0;
    KineticGeometry geom;
    RREA_HOST_DEVICE bool TransientEnergyIsCertified(RreaSecondaryParticle const& p) const {

            // Contains is false for a NaN energy; the loader owns the ranges.
            return (p.species == RreaSecondarySpecies::Photon
                        ? tables.PhotonCertifiedEnergyRangeEv()
                        : tables.ChargedCertifiedEnergyRangeEv())
                .Contains(p.kinetic_or_photon_energy_eV);

    }
        // Local energy deposit -> ion pairs, N = w E_local / W_air
        // (RreaIonPairWeightFromLocalEnergy owns the relation and its
        // citation).  Writes only through the caller-supplied side-effect log
        // so it can run on an OpenMP worker thread; the log is replayed in
        // deterministic chunk order afterwards.
        template<class Effects>
        RREA_HOST_DEVICE void deposit_local_energy_as_ion_pairs(
                Effects& fx,
                amrex::Real r_m,
                amrex::Real z_m,
                amrex::Real macro_weight,
                amrex::Real local_energy_eV) const {
            if (macro_weight <= amrex::Real(0.0) || local_energy_eV <= amrex::Real(0.0)) {
                return;
            }
            fx.AddLocalConservationEnergy(macro_weight, local_energy_eV);
            amrex::Real const pair_weight = RreaIonPairWeightFromLocalEnergy(
                macro_weight, local_energy_eV, tables.WAirEvPerIonPair());
            if (pair_weight <= amrex::Real(0.0)) {
                return;
            }
            // Each created pair is booked with a loss of exactly W_air, so the
            // ledger records macro_weight*local_energy.
            fx.AccumulateIonizationEvent(r_m, z_m, pair_weight, amrex::Real(0.0));
            fx.tally.RecordTransportIonization(
                pair_weight, tables.WAirEvPerIonPair(), true, false);
        }

        template<class Effects>
        RREA_HOST_DEVICE void spawn_positron_annihilation(
            Effects& fx,
            amrex::Real x_m,
            amrex::Real y_m,
            amrex::Real z_m,
            amrex::Real r_m,
            amrex::Real z_source_m,
            amrex::Real axis_x,
            amrex::Real axis_y,
            amrex::Real axis_z,
            amrex::Real event_weight,
            amrex::Real positron_kinetic_energy_eV,
            bool at_rest_or_below_cutoff,
            std::uint64_t rng_particle_id,
            std::uint64_t interaction_index,
            amrex::Real birth_elapsed_step_fraction = amrex::Real(1.0)) const {
            if (event_weight <= amrex::Real(0.0)) {
                return;
            }
            RreaNormalizeDirection(axis_x, axis_y, axis_z);

            // Below-cutoff positrons first thermalize residual kinetic energy
            // locally, then
            // annihilate at rest.  In-flight annihilation keeps the positron
            // kinetic energy in the two-photon final state.
            amrex::Real const annihilation_kinetic_eV = at_rest_or_below_cutoff
                ? amrex::Real(0.0)
                : rrea::kinetic::max(positron_kinetic_energy_eV, amrex::Real(0.0));
            auto const annihilation_state = tables.SamplePositronAnnihilationFinalState(
                annihilation_kinetic_eV,
                RreaRng::Uniform01(RreaRngKey{
                    coupling.RngSeed(),
                    rng_particle_id,
                    static_cast<std::uint64_t>(rrea::kinetic::max(step, 0)),
                    interaction_index,
                    static_cast<std::uint64_t>(RreaRngChannel::PositronAnnihilationPolar)}),
                RreaRng::Uniform01(RreaRngKey{
                    coupling.RngSeed(),
                    rng_particle_id,
                    static_cast<std::uint64_t>(rrea::kinetic::max(step, 0)),
                    interaction_index,
                    static_cast<std::uint64_t>(RreaRngChannel::PositronAnnihilationAzimuth)}));
            amrex::Real photon1_dir_x;
            amrex::Real photon1_dir_y;
            amrex::Real photon1_dir_z;
            RreaSampleConeDirection(
                axis_x,
                axis_y,
                axis_z,
                annihilation_state.photon1_cos_theta,
                annihilation_state.photon1_azimuth_u,
                photon1_dir_x,
                photon1_dir_y,
                photon1_dir_z);
            amrex::Real photon1_energy_eV = rrea::kinetic::max(
                annihilation_state.photon1_energy_eV,
                amrex::Real(0.0));
            amrex::Real photon2_energy_eV = rrea::kinetic::max(
                annihilation_state.photon2_energy_eV,
                amrex::Real(0.0));
            amrex::Real photon2_dir_x = -photon1_dir_x;
            amrex::Real photon2_dir_y = -photon1_dir_y;
            amrex::Real photon2_dir_z = -photon1_dir_z;
            if (!annihilation_state.at_rest && photon2_energy_eV > amrex::Real(0.0)) {
                amrex::Real const p_eV = std::sqrt(rrea::kinetic::max(
                    amrex::Real(0.0),
                    annihilation_kinetic_eV
                        * (annihilation_kinetic_eV
                           + amrex::Real(2.0) * me_c2_eV)));
                photon2_dir_x = (p_eV * axis_x - photon1_energy_eV * photon1_dir_x)
                    / photon2_energy_eV;
                photon2_dir_y = (p_eV * axis_y - photon1_energy_eV * photon1_dir_y)
                    / photon2_energy_eV;
                photon2_dir_z = (p_eV * axis_z - photon1_energy_eV * photon1_dir_z)
                    / photon2_energy_eV;
                RreaNormalizeDirection(photon2_dir_x, photon2_dir_y, photon2_dir_z);
            }

            // Keep annihilation photon local deposition separate from prior
            // sub-cutoff thermalization.  The residual positron kinetic energy is
            // deposited once by the thermalization block below; this accumulator
            // is only for annihilation photon energy that falls below the photon cutoff.
            amrex::Real local_deposit_energy_eV = amrex::Real(0.0);
            if (photon1_energy_eV >= coupling.PhotonCutoffEv()) {
                fx.AddSecondary(RreaSecondarySpecies::Photon, Spawn{
                    x_m, y_m, z_m,
                    photon1_dir_x, photon1_dir_y, photon1_dir_z,
                    event_weight, photon1_energy_eV,
                    birth_elapsed_step_fraction});
            } else {
                local_deposit_energy_eV += photon1_energy_eV;
            }
            if (photon2_energy_eV >= coupling.PhotonCutoffEv()) {
                fx.AddSecondary(RreaSecondarySpecies::Photon, Spawn{
                    x_m, y_m, z_m,
                    photon2_dir_x, photon2_dir_y, photon2_dir_z,
                    event_weight, photon2_energy_eV,
                    birth_elapsed_step_fraction});
            } else {
                local_deposit_energy_eV += photon2_energy_eV;
            }

            // If a positron has already fallen below the kinetic cutoff, thermalize
            // its residual kinetic energy locally before annihilating at rest.  This
            // creates ordinary ion pairs and is recorded separately from the
            // annihilation-photon budget.
            if (at_rest_or_below_cutoff && positron_kinetic_energy_eV > amrex::Real(0.0)) {
                deposit_local_energy_as_ion_pairs(
                    fx,
                    r_m,
                    z_source_m,
                    event_weight,
                    positron_kinetic_energy_eV);
            }
            if (local_deposit_energy_eV > amrex::Real(0.0)) {
                deposit_local_energy_as_ion_pairs(
                    fx,
                    r_m,
                    z_source_m,
                    event_weight,
                    local_deposit_energy_eV);
            }

            // The target electron annihilated by the positron is removed from a
            // neutral air molecule/atom, leaving one local positive ion.  Do
            // not create a low-energy electron source for that annihilated target electron.
            fx.AccumulatePositiveIonSourceOnly(r_m, z_source_m, event_weight);
            fx.RecordPositronAnnihilation(
                event_weight,
                local_deposit_energy_eV,
                photon1_energy_eV,
                photon2_energy_eV,
                at_rest_or_below_cutoff);
        }

    // THE charged history: one e-/e+ through its time budget in this step --
    // a live parent's whole dt or a newborn's residual (1-b)dt.  The substep
    // count comes from the whole-budget driver chord; each substep then
    // rebuilds its chord from the running state and runs, in
    // RreaRunChargedSubstepInterleave order, the leading field half-kick,
    // the discrete hazards interleaved with continuous drag
    // (RreaRunChargedEventInterleave, RreaPlanContinuousDragSegment), GS
    // elastic scatter, the trailing half-kick and the commit.  Terminal order
    // is RreaResolveChargedTransportTerminal: an observed exit outranks a
    // stop, which outranks the end-state policy; each terminal books exactly
    // one remove-parent group.  Species enters only as the charge sign, the
    // channel set, the C&D plane crossings and the stop disposition
    // (electron demotion vs positron at-rest annihilation); origin enters only
    // as the time budget, the RNG namespace, the birth state policy and the
    // Created flag on the net chord.
    template<class Effects>
    RREA_HOST_DEVICE HistoryEnd AdvanceChargedHistory(
        Effects& fx, ParticleHistory& h) const
    {
        bool const pos = h.species == RreaSecondarySpecies::Positron;
        amrex::Real const charge_sign = pos ? amrex::Real(1.0) : amrex::Real(-1.0);
        amrex::Real const w = h.weight;
        amrex::Real& K = h.energy_eV;
        amrex::Real const cutoff_eV = coupling.LowEnergyCutoffEv();
        amrex::Real const hard_threshold_eV = coupling.HardMollerSecondaryThresholdEv();
        amrex::Real const photon_cutoff_eV = coupling.PhotonCutoffEv();
        bool const hard_moller_enabled = !m_config.debug_disable_hard_moller_secondaries;
        std::uint64_t const created = h.newborn ? kRreaSegmentChannelCreated : 0U;
        RreaNormalizeDirection(h.dir_x, h.dir_y, h.dir_z);
        // Net-chord start: every terminal books its charge segment from here.
        amrex::Real const start_r = std::hypot(h.x_m, h.y_m);
        amrex::Real const start_z = h.z_m;
        amrex::Real const residual = RreaNewbornResidualFraction(h.birth_elapsed);
        amrex::Real const budget_s = dt_s * residual;
        if (!pos && coupling.electron_creation_enabled
            && rrea::ElectronFirstPassage(h.electron_counted, K, K) >= 0)
            fx.AccumulateElectronCreation(start_r, start_z, w, !h.newborn);

        auto rng_key = [&](std::uint64_t index, RreaRngChannel channel) {
            return RreaRngKey{coupling.RngSeed(), h.rng_particle_id,
                static_cast<std::uint64_t>(rrea::kinetic::max(step, 0)), index,
                static_cast<std::uint64_t>(channel)};
        };
        // A newborn draws on its root's stream within its own lineage.
        auto transient_index = [&](std::uint64_t segment, std::uint64_t ordinal,
                                   std::uint64_t inner) {
            auto const packed = RreaTryPackTransientRngIndex(
                true, h.root_species, h.transient_lineage, segment, ordinal, inner);
            if (!packed.valid) {
                rrea::KineticFail("RREA newborn charged RNG index overflow; the "
                                  "uncommitted transaction is rejected");
            }
            return packed.value;
        };
        // Identical local state gives the identical disposition: the gate
        // reads the field AND the density at the point it judges.
        auto state_policy_at = [&](amrex::Real x, amrex::Real y, amrex::Real z) {
            if (pos) {
                return RreaElectronStatePolicyDecision{K < cutoff_eV, false};
            }
            RreaFieldSample const field = coupling.GatherFieldRZ(std::hypot(x, y), z);
            return RreaEvaluateElectronStatePolicy(
                K, cutoff_eV, hard_threshold_eV,
                std::hypot(field.er_v_per_m, field.ez_v_per_m),
                tables.ElectronEnsembleDrag(
                    rrea::kinetic::max(K, cutoff_eV),
                    coupling.TransportDensityRatioAtZ(z),
                    hard_moller_enabled));
        };

        std::uint64_t substeps = 1;
        std::uint64_t completed = 0;
        bool exited = false;
        bool stopped = false;
        bool consumed = false;
        amrex::Real run_x = h.x_m;
        amrex::Real run_y = h.y_m;
        amrex::Real run_z = h.z_m;
        amrex::Real stop_x = run_x;
        amrex::Real stop_y = run_y;
        amrex::Real stop_z = run_z;
        amrex::Real stop_elapsed = amrex::Real(1.0);
        RreaTransportPathSegment path;
        MaterialPathInterval material;
        // Microchord fractions to OUTER-step time: a secondary's birth
        // fraction must come through here or AdvanceNewborn flies it a
        // multiple of the residual it is owed.
        auto elapsed_at = [&](amrex::Real f) {
            return RreaNewbornElapsedStepFraction(
                h.birth_elapsed, residual, completed, substeps, f);
        };
        // A stop at or past the exit root of an exiting chord, or at a point
        // outside the half-open domain, is an escape: the particle leaves
        // before it can thermalize.  The stop's point and TIME are recorded
        // now, before completed moves past the microchord it happened on.
        auto mark_stopped_at = [&](amrex::Real f) {
            amrex::Real x = amrex::Real(0.0);
            amrex::Real y = amrex::Real(0.0);
            amrex::Real z = amrex::Real(0.0);
            path.PointAtFullPathFraction(f, x, y, z);
            if (RreaChargedStopIsEscape(path.exits_domain, f, path.in_domain_end_fraction)
                || !RreaCartesianPositionInHalfOpenRzDomain(
                    x, y, z, geom.ProbLo(0), geom.ProbHi(0),
                    geom.ProbLo(1), geom.ProbHi(1))) {
                exited = true;
                return;
            }
            stop_x = x;
            stop_y = y;
            stop_z = z;
            stop_elapsed = elapsed_at(f);
            stopped = true;
        };

        // Entry: a live parent below the cutoff, or a newborn the full state
        // policy rejects at birth, stops where it stands (on a zero-length
        // chord, so a start outside the half-open domain escapes instead).
        bool const stop_at_entry = h.newborn
            ? state_policy_at(h.x_m, h.y_m, h.z_m).Demote()
            : K < cutoff_eV;
        if (stop_at_entry) {
            amrex::Real const theta = std::atan2(h.y_m, h.x_m);
            path = ClipRzTransportPath(
                start_r, theta, start_z, start_r, theta, start_z,
                geom.ProbLo(0), geom.ProbHi(0), geom.ProbLo(1), geom.ProbHi(1));
            mark_stopped_at(amrex::Real(0.0));
        } else {
            // Substep count from the whole-budget driver chord: the competing
            // optical depth per substep (fatal under strict guards), GS
            // collision batching, and the field-impulse + drag turnover per
            // substep.  Budgets are integrated over the driver's clipped
            // in-domain span and divided by it, because the loop flies the
            // whole budget wherever the driver clipped.
            auto const driver = RreaBuildRunningChargedChord(
                RreaRunningChargedState{
                    h.x_m, h.y_m, h.z_m, h.dir_x, h.dir_y, h.dir_z, K},
                budget_s,
                geom.ProbLo(0), geom.ProbHi(0), geom.ProbLo(1), geom.ProbHi(1));
            if (driver.valid && driver.path.in_domain_length_m > amrex::Real(0.0)) {
                auto const& dp = driver.path;
                auto const di = build_material_path_interval(coupling, dp);
                amrex::Real const span = rrea::kinetic::max(
                    dp.in_domain_end_fraction, amrex::Real(1.0e-6));
                amrex::Real const density_path = density_path_m(
                    di, dp, dp.in_domain_begin_fraction, dp.in_domain_end_fraction) / span;
                auto const stp = RreaChargedCompetingStpRates(
                    tables, pos, hard_threshold_eV, photon_cutoff_eV, K,
                    hard_moller_enabled);
                amrex::Real const total_stp = stp[0] + stp[1] + stp[2];
                bool violation = false;
                substeps = selected_material_subcycles(
                    total_stp * density_path, di, dp, m_config, violation, total_stp);
                // GS and impulse drivers clamp silently at the cap: benign
                // refinement, not an optical-depth violation.
                substeps = rrea::kinetic::min(
                    rrea::kinetic::max({substeps,
                        substep_driver_count(
                            rrea::kinetic::min(
                                elastic_inverse_length_stp(tables, pos, K) * density_path,
                                m_config.transport_gs_total_collision_cap),
                            rrea::kinetic::max(
                                m_config.transport_max_gs_collisions_per_substep,
                                amrex::Real(1.0))),
                        substep_driver_count(
                            std::abs(path_field_work_and_average(
                                coupling, dp, charge_sign,
                                amrex::Real(0.0), amrex::Real(1.0)).work_eV) / span
                                + tables.ChargedRealizedDrag(pos, K, amrex::Real(1.0))
                                    * density_path,
                            rrea::kinetic::max(
                                m_config.transport_max_field_impulse_fraction * K,
                                rrea::kinetic::max(
                                    m_config.transport_field_impulse_floor_eV,
                                    amrex::Real(1.0))))}),
                    m_config.max_subcycles);
                fx.tally.RecordTransportSubcycles(substeps);
                if (violation) {
                    rrea::KineticFail("RREA strict transport guard: optical-depth "
                                      "subcycle cap exceeded");
                }
            }

            // Continuous drag over one event-free segment: the realised drag
            // at the frozen energy, its collision share as W_air ion pairs,
            // the hard-Moller mean rate for the ledger partition (electrons;
            // Bhabha stays per-event), and the C&D plane crossings.
            amrex::Real positron_loss_eV = amrex::Real(0.0);
            PathFieldSubRange field{};
            auto apply_drag = [&](amrex::Real b, amrex::Real e, amrex::Real frozen_eV)
                -> RreaContinuousSegmentLimit {
                amrex::Real const K_before = K;
                auto const plan = RreaPlanContinuousDragSegment(
                    b, e, K, frozen_eV, cutoff_eV,
                    [&](amrex::Real pb, amrex::Real pe) {
                        return density_path_m(material, path, pb, pe);
                    },
                    [&] {
                        amrex::Real const stp_density = amrex::Real(1.0);
                        return RreaFrozenStoppingPowers{
                            RreaFrozenStpRate(pos
                                ? tables.PositronCollisionLoss(frozen_eV, stp_density)
                                : tables.ElectronCollisionLoss(frozen_eV, stp_density)),
                            RreaFrozenStpRate(
                                tables.ChargedRealizedDrag(pos, frozen_eV, stp_density))};
                    });
                if (!plan.limit_valid) {
                    rrea::KineticFail("RREA charged continuous stopping could not "
                                      "locate a finite monotone cutoff point");
                }
                if (!plan.applied) {
                    return RreaContinuousSegmentLimit{b, false, true};
                }
                fx.tally.RecordContinuousSoftBrems(w, plan.soft_radiative_loss_eV);
                if (pos) {
                    positron_loss_eV += plan.drag_loss_eV;
                }
                amrex::Real const w_air_eV = tables.WAirEvPerIonPair();
                RreaPartitionContinuousIonPairs(
                    material.begin_fraction, material.end_fraction,
                    b, plan.effective_end_fraction, plan.collision_loss_eV, w, w_air_eV,
                    [&](amrex::Real midpoint, amrex::Real pair_weight) {
                        amrex::Real x = amrex::Real(0.0);
                        amrex::Real y = amrex::Real(0.0);
                        amrex::Real z = amrex::Real(0.0);
                        path.PointAtFullPathFraction(midpoint, x, y, z);
                        fx.AccumulateIonizationEvent(
                            std::hypot(x, y), z, pair_weight, amrex::Real(0.0));
                        fx.tally.RecordTransportIonization(
                            pair_weight, w_air_eV, true, false);
                    });
                if (!pos) {
                    if (coupling.electron_creation_enabled && !h.electron_counted) {
                        double const f = rrea::ElectronFirstPassageInDragSegment(
                            h.electron_counted, K_before, plan.drag_loss_eV, field.work_eV,
                            b, plan.effective_end_fraction,
                            path.in_domain_begin_fraction, path.in_domain_end_fraction);
                        if (f >= 0) {
                            amrex::Real x, y, z;
                            path.PointAtFullPathFraction(
                                b + f*(plan.effective_end_fraction-b), x, y, z);
                            if (RreaCartesianPositionInHalfOpenRzDomain(x,y,z,
                                geom.ProbLo(0),geom.ProbHi(0),geom.ProbLo(1),geom.ProbHi(1)))
                                fx.AccumulateElectronCreation(std::hypot(x,y), z, w);
                        }
                    }
                    defer_cd_electron_plane_crossings(
                        fx, coupling, tables, path, material,
                        b, plan.effective_end_fraction, w, K_before, frozen_eV,
                        plan.drag_loss_eV, hard_moller_enabled);
                }
                K = plan.reaches_cutoff
                    ? cutoff_eV
                    : rrea::kinetic::max(amrex::Real(0.0), K - plan.drag_loss_eV);
                return RreaContinuousSegmentLimit{
                    plan.effective_end_fraction, plan.reaches_cutoff, true};
            };
            using C = RreaRngChannel;
            GsScatterChannels const gs_channels = h.newborn
                ? (pos ? GsScatterChannels{C::NewbornPositronGsFewCount,
                             C::NewbornPositronGsFewScatter, C::NewbornPositronGsFewAzimuth,
                             C::NewbornPositronGsAngle, C::NewbornPositronGsAzimuth}
                       : GsScatterChannels{C::NewbornElectronGsFewCount,
                             C::NewbornElectronGsFewScatter, C::NewbornElectronGsFewAzimuth,
                             C::NewbornElectronGsAngle, C::NewbornElectronGsAzimuth})
                : (pos ? GsScatterChannels{C::PositronGsFewCount, C::PositronGsFewScatter,
                             C::PositronGsFewAzimuth, C::PositronGsAngle, C::PositronGsAzimuth}
                       : GsScatterChannels{C::ElectronGsFewCount, C::ElectronGsFewScatter,
                             C::ElectronGsFewAzimuth, C::ElectronGsAngle, C::ElectronGsAzimuth});
            // Discrete hazards of one substep: the SAME gated rate bundle and
            // final states for a live parent and a newborn.  The depth and
            // channel draws of each event ordinal precede its final state.
            amrex::Real first_frozen_eV = K;
            auto run_events = [&](std::uint64_t substep, amrex::Real begin, amrex::Real end) {
                amrex::Real ex = amrex::Real(0.0);
                amrex::Real ey = amrex::Real(0.0);
                amrex::Real ez = amrex::Real(0.0);
                amrex::Real er = amrex::Real(0.0);
                amrex::Real event_elapsed = amrex::Real(1.0);
                std::uint64_t index = 0;
                amrex::Real total_stp = amrex::Real(0.0);
                RreaRunChargedEventInterleave(
                    begin, end, K, first_frozen_eV, cutoff_eV, pos ? 3 : 2,
                    kRreaTransientEventCapacity,
                    [&](amrex::Real frozen_eV, amrex::Real tb, amrex::Real te) {
                        auto const stp = RreaChargedCompetingStpRates(
                            tables, pos, hard_threshold_eV, photon_cutoff_eV,
                            frozen_eV, hard_moller_enabled);
                        total_stp = stp[0] + stp[1] + stp[2];
                        amrex::Real const segment_path = density_path_m(material, path, tb, te);
                        std::array<amrex::Real, 3> const taus{
                            stp[0] * segment_path, stp[1] * segment_path,
                            stp[2] * segment_path};
                        return taus;
                    },
                    [&](std::uint64_t ordinal) {
                        C depth = C::NewbornCompetingOpticalDepth;
                        C share = C::NewbornCompetingChannel;
                        if (h.newborn) {
                            index = transient_index(substep, ordinal, 0U);
                        } else {
                            auto const live = RreaTryLiveHazardRngIndex(
                                h.species, substep, ordinal);
                            if (!live.valid) {
                                rrea::KineticFail("RREA charged repeated-hazard RNG index "
                                                  "overflow; the uncommitted transaction "
                                                  "is rejected");
                            }
                            index = live.value;
                            bool const first = ordinal == 0U;
                            depth = pos ? (first ? C::PositronCompetingOpticalDepth
                                                 : C::PositronContinuationOpticalDepth)
                                        : (first ? C::ElectronCompetingOpticalDepth
                                                 : C::ElectronContinuationOpticalDepth);
                            share = pos ? (first ? C::PositronCompetingChannel
                                                 : C::PositronContinuationChannel)
                                        : (first ? C::ElectronCompetingChannel
                                                 : C::ElectronContinuationChannel);
                        }
                        return std::pair<amrex::Real, amrex::Real>{
                            RreaRng::Uniform01(rng_key(index, depth)),
                            RreaRng::Uniform01(rng_key(index, share))};
                    },
                    [&](amrex::Real depth, amrex::Real ib, amrex::Real ie) {
                        return fraction_at_density_path_m(
                            material, path, ib, ie, depth / total_stp);
                    },
                    apply_drag,
                    mark_stopped_at,
                    [&](amrex::Real f) {
                        point_and_radius_at_fraction(path, f, ex, ey, ez, er);
                        event_elapsed = elapsed_at(f);
                    },
                    [&](int channel, amrex::Real energy_eV) {
                        return RreaChargedChannelStpRate(
                            tables, pos, hard_threshold_eV, photon_cutoff_eV,
                            channel, energy_eV, hard_moller_enabled) > amrex::Real(0.0);
                    },
                    [&](int channel, amrex::Real) {
                        RreaInteractionSite const site{ex, ey, ez, er, event_elapsed};
                        auto key = [&](RreaRngChannel c) { return rng_key(index, c); };
                        if (channel == 0 && pos) {
                            RreaApplyBhabhaFinalState(
                                fx, tables, key, site, w, hard_threshold_eV,
                                K, h.dir_x, h.dir_y, h.dir_z);
                        } else if (channel == 0) {
                            RreaApplyHardMollerFinalState(
                                fx, tables, key, site, w, hard_threshold_eV,
                                K, h.dir_x, h.dir_y, h.dir_z);
                        } else if (channel == 1) {
                            RreaApplyBremsFinalState(
                                fx, tables, key,
                                [&](amrex::Real local_eV) {
                                    deposit_local_energy_as_ion_pairs(fx, er, ez, w, local_eV);
                                },
                                site, w, charge_sign, photon_cutoff_eV,
                                K, h.dir_x, h.dir_y, h.dir_z);
                        } else {
                            // In-flight annihilation consumes the positron at
                            // the sampled point.
                            fx.BeginGroup(RreaSecondaryProcess::PositronAnnihilation, true);
                            spawn_positron_annihilation(
                                fx, ex, ey, ez, er, ez, h.dir_x, h.dir_y, h.dir_z,
                                w, K, false, h.rng_particle_id, index, event_elapsed);
                            fx.SetInteractionConservationClosure(w, K, amrex::Real(0.0));
                            fx.AccumulateChargedSegment(
                                start_r, start_z, er, ez, charge_sign * w,
                                created | kRreaSegmentChannelAnnihilated);
                            fx.EndGroup();
                            consumed = true;
                            return false;
                        }
                        return true;
                    });
            };
            amrex::Real const sub_dt_s = budget_s / static_cast<amrex::Real>(substeps);
            amrex::Real chord_dt_s = amrex::Real(0.0);
            auto half_kick = [&] {
                static_cast<void>(RreaApplyChargedFieldHalfKick(
                    field, amrex::Real(0.5) * chord_dt_s, charge_sign,
                    K, h.dir_x, h.dir_y, h.dir_z));
            };
            RreaRunChargedSubstepInterleave(
                substeps, K, cutoff_eV, completed, exited,
                [&](std::uint64_t, amrex::Real& begin, amrex::Real& end) {
                    // The chord the particle actually flies this substep, from
                    // the state it is actually in, so deflection bends the path
                    // and the field work projects on the real direction.
                    auto chord = RreaBuildRunningChargedChord(
                        RreaRunningChargedState{
                            run_x, run_y, run_z, h.dir_x, h.dir_y, h.dir_z, K},
                        sub_dt_s,
                        geom.ProbLo(0), geom.ProbHi(0), geom.ProbLo(1), geom.ProbHi(1));
                    if (!chord.valid) {
                        return false;
                    }
                    path = std::move(chord.path);
                    if (!(path.full_length_m > amrex::Real(0.0))
                        || !(path.in_domain_length_m > amrex::Real(0.0))) {
                        exited = path.exits_domain;
                        return false;
                    }
                    material = build_material_path_interval(coupling, path);
                    begin = path.in_domain_begin_fraction;
                    end = path.in_domain_end_fraction;
                    field = path_field_work_and_average(coupling, path, charge_sign, begin, end);
                    // Time ACTUALLY spent on this chord: the kick's rotation
                    // scales with dt, so a clipped chord must not be
                    // over-rotated by 1/(end-begin) -- energy right, direction
                    // wrong, invisible to every conservation ledger.
                    chord_dt_s = sub_dt_s * (end - begin);
                    first_frozen_eV = RreaChargedMidpointFreezeEnergy(
                        K, field.work_eV,
                        tables.ChargedRealizedDrag(pos, K, amrex::Real(1.0))
                            * density_path_m(material, path, begin, end),
                        cutoff_eV);
                    half_kick();
                    if (K < cutoff_eV) {
                        mark_stopped_at(begin);
                        return false;
                    }
                    return true;
                },
                run_events,
                [&] { return stopped || consumed; },
                [&](std::uint64_t substep, amrex::Real begin, amrex::Real end) {
                    if (K >= cutoff_eV) {
                        apply_gs_elastic_scatter(
                            material, path, begin, end, tables, pos, gs_channels,
                            [&](RreaRngChannel channel, std::uint64_t inner) {
                                return rng_key(h.newborn
                                    ? transient_index(substep, 0U, inner)
                                    : (substep << 8) | inner, channel);
                            },
                            K, h.dir_x, h.dir_y, h.dir_z);
                    }
                    half_kick();
                    if (K < cutoff_eV) {
                        mark_stopped_at(end);
                        return true;
                    }
                    return false;
                },
                // Commit: the next chord starts from here.  A refused endpoint
                // lies outside the half-open domain and becomes an escape.
                [&](amrex::Real end) {
                    return RreaAdvanceRunningChargedPosition(
                        path, end, geom.ProbLo(0), geom.ProbHi(0),
                        geom.ProbLo(1), geom.ProbHi(1), run_x, run_y, run_z);
                },
                [&] { return path.exits_domain; });
            if (pos) {
                fx.tally.RecordPositronTransport(w, positron_loss_eV, false);
            }
        }
        if (consumed) {
            return HistoryEnd::Consumed;
        }

        auto const end_policy = (exited || stopped)
            ? RreaElectronStatePolicyDecision{}
            : state_policy_at(run_x, run_y, run_z);
        auto const terminal = RreaResolveChargedTransportTerminal(
            exited, stopped, pos, end_policy.below_transport_cutoff,
            end_policy.low_energy_nonrunaway);
        if (terminal == RreaChargedTerminalDisposition::Escape) {
            // One atomic deposit: the net prefix to the final microchord's
            // start, that microchord to the exact face (including a refused
            // half-open endpoint), and the outward flux that removes the
            // valid part of the CIC cloud.  The clipped policy makes RREA own
            // the removal: WarpX never sees the raw overshoot.
            auto const escape = RreaClippedTransportEscapePolicyForPath(
                path, std::hypot(path.end_x_m, path.end_y_m), path.end_z_m);
            h.rrea_owns_escape = escape.invalidate_parent_in_rrea;
            fx.BeginGroup(RreaSecondaryProcess::ParticleEscape, true,
                          escape.ParentStillValidAtPreflight());
            fx.SetChargedEscapeConservationClosure(w, K);
            fx.AccumulateChargedEscape(
                start_r, start_z, RreaChargedBoundaryCrossingForPath(path),
                charge_sign * w, created | kRreaSegmentChannelEscaped);
            fx.EndGroup();
            return HistoryEnd::Escape;
        }
        if (terminal != RreaChargedTerminalDisposition::Survive) {
            // Where the particle actually is: the recorded stop, or the
            // committed endpoint when the end-state policy fires there.
            amrex::Real const end_x = stopped ? stop_x : run_x;
            amrex::Real const end_y = stopped ? stop_y : run_y;
            amrex::Real const end_z = stopped ? stop_z : run_z;
            amrex::Real const end_r = std::hypot(end_x, end_y);
            std::uint64_t channels = created;
            if (pos) {
                // At-rest RNG segment: 0 before any transport, one past the
                // last substep after it.
                std::uint64_t const segment = stop_at_entry ? 0U : substeps;
                fx.BeginGroup(RreaSecondaryProcess::PositronAnnihilation, true);
                spawn_positron_annihilation(
                    fx, end_x, end_y, end_z, end_r, end_z, h.dir_x, h.dir_y, h.dir_z,
                    w, K, true, h.rng_particle_id,
                    h.newborn ? transient_index(segment, 0U, 0U) : segment,
                    stopped ? stop_elapsed : amrex::Real(1.0));
                fx.tally.RecordPositronTransport(w, K, true);
                channels |= kRreaSegmentChannelAnnihilated;
            } else {
                // The demoted electron joins the low-energy mesh (energy 0,
                // so the mesh diagnostic is not double-counted) and its
                // residual kinetic energy thermalizes into W_air ion pairs.
                fx.BeginGroup(RreaSecondaryProcess::LowEnergyDemotion, true);
                fx.AccumulateLowEnergyElectron(end_r, end_z, w);
                deposit_local_energy_as_ion_pairs(fx, end_r, end_z, w, K);
                channels |= kRreaSegmentChannelDemoted;
            }
            fx.SetInteractionConservationClosure(w, K, amrex::Real(0.0));
            fx.AccumulateChargedSegment(
                start_r, start_z, end_r, end_z, charge_sign * w, channels);
            fx.EndGroup();
            return HistoryEnd::Consumed;
        }
        // Survivor: the committed point, which the adapter hands back.
        h.x_m = run_x;
        h.y_m = run_y;
        h.z_m = run_z;
        fx.AccumulateChargedSegment(
            start_r, start_z, std::hypot(run_x, run_y), run_z,
            charge_sign * w, created);
        return HistoryEnd::Survive;
    }

    // THE photon history: one photon through its time budget (a live
    // parent's dt or a newborn's residual) along a straight chord at c.  Each
    // pass samples one exact exponential optical depth over the remaining
    // in-domain chord; the midpoint density path misplaces the interaction
    // by at most h^2/8H = 9 um on a 0.75 m chord.  A Compton survivor goes on
    // from the interaction point along its new direction for the unused
    // length, clipped afresh (the pre-scatter in-domain remainder would
    // shorten inward-scattered photons merely because the unscattered path
    // would have escaped); photoelectric and pair absorption end it.  A chord
    // that runs out without an event escapes by the clipped endpoint policy
    // or survives at its end.  Coherent Rayleigh scattering is excluded.
    template<class Effects>
    RREA_HOST_DEVICE HistoryEnd AdvancePhotonHistory(
        Effects& fx, ParticleHistory& h) const
    {
        amrex::Real const w = h.weight;
        amrex::Real& E = h.energy_eV;
        RreaNormalizeDirection(h.dir_x, h.dir_y, h.dir_z);
        if (!(w > amrex::Real(0.0)) || E < coupling.PhotonCutoffEv()) {
            // A photon below the production cutoff does not propagate: its
            // energy thermalizes where it stands.
            fx.BeginGroup(RreaSecondaryProcess::LowEnergyDemotion, true);
            deposit_local_energy_as_ion_pairs(
                fx, std::hypot(h.x_m, h.y_m), h.z_m,
                rrea::kinetic::max(w, amrex::Real(0.0)), E);
            if (w > amrex::Real(0.0)) {
                fx.SetInteractionConservationClosure(w, E, amrex::Real(0.0));
            } else {
                fx.SetExplicitZeroConservationClosure();
            }
            fx.EndGroup();
            return HistoryEnd::Consumed;
        }
        auto chord_from = [&](amrex::Real x, amrex::Real y, amrex::Real z,
                              amrex::Real length_m) {
            amrex::Real const end_x = x + length_m * h.dir_x;
            amrex::Real const end_y = y + length_m * h.dir_y;
            amrex::Real const end_z = z + length_m * h.dir_z;
            return ClipRzTransportPath(
                std::hypot(x, y), std::atan2(y, x), z,
                std::hypot(end_x, end_y), std::atan2(end_y, end_x), end_z,
                geom.ProbLo(0), geom.ProbHi(0), geom.ProbLo(1), geom.ProbHi(1));
        };
        RreaTransportPathSegment path = chord_from(
            h.x_m, h.y_m, h.z_m,
            c_m_per_s * dt_s * RreaNewbornResidualFraction(h.birth_elapsed));
        amrex::Real elapsed_start = h.birth_elapsed;
        for (std::uint64_t ordinal = 0;; ++ordinal) {
            auto const index = h.newborn
                ? RreaTryPackTransientRngIndex(
                    true, h.root_species, h.transient_lineage, 0U, ordinal, 0U)
                : RreaTryLiveHazardRngIndex(RreaSecondarySpecies::Photon, 0U, ordinal);
            if (!index.valid) {
                rrea::KineticFail("RREA photon repeated-hazard RNG index overflow; "
                                  "the uncommitted transaction is rejected");
            }
            auto rng_key = [&](RreaRngChannel channel) {
                return RreaRngKey{coupling.RngSeed(), h.rng_particle_id,
                    static_cast<std::uint64_t>(rrea::kinetic::max(step, 0)),
                    index.value, static_cast<std::uint64_t>(channel)};
            };
            bool const first = ordinal == 0U;
            auto const material = build_material_path_interval(coupling, path);
            amrex::Real const mfp = tables.PhotonFeedbackMeanFreePath(E, amrex::Real(1.0));
            amrex::Real const tau = density_path_m(
                material, path, path.in_domain_begin_fraction,
                path.in_domain_end_fraction) / mfp;
            fx.tally.RecordTransportSubcycles(1U);
            RreaRemainingSegmentHazard const hazard = RreaResolveRemainingSegmentHazard(
                {tau, amrex::Real(0.0), amrex::Real(0.0)}, 1,
                path.in_domain_begin_fraction, path.in_domain_end_fraction,
                RreaRng::Uniform01(rng_key(h.newborn
                    ? RreaRngChannel::NewbornCompetingOpticalDepth
                    : first ? RreaRngChannel::PhotonInteraction
                            : RreaRngChannel::PhotonContinuationOpticalDepth)),
                amrex::Real(0.0),
                [&](amrex::Real depth) {
                    return fraction_at_density_path_m(
                        material, path, path.in_domain_begin_fraction,
                        path.in_domain_end_fraction, depth * mfp);
                });
            if (!hazard.selection.event) {
                auto const escape = RreaClippedTransportEscapePolicyForPath(
                    path, std::hypot(path.end_x_m, path.end_y_m), path.end_z_m);
                if (escape.escapes) {
                    h.rrea_owns_escape = escape.invalidate_parent_in_rrea;
                    fx.BeginGroup(RreaSecondaryProcess::PhotonEscape, true,
                                  escape.ParentStillValidAtPreflight());
                    fx.tally.RecordPhotonEscape(w, E);
                    fx.AddEscapedConservationEnergy(w, E);
                    fx.SetInteractionConservationClosure(w, E, amrex::Real(0.0));
                    fx.EndGroup();
                    return HistoryEnd::Escape;
                }
                path.PointAtFullPathFraction(
                    path.in_domain_end_fraction, h.x_m, h.y_m, h.z_m);
                return HistoryEnd::Survive;
            }
            amrex::Real const distance_m = path.full_length_m
                * (hazard.event_fraction - path.in_domain_begin_fraction);
            amrex::Real ex = amrex::Real(0.0);
            amrex::Real ey = amrex::Real(0.0);
            amrex::Real ez = amrex::Real(0.0);
            amrex::Real er = amrex::Real(0.0);
            point_and_radius_at_fraction(path, hazard.event_fraction, ex, ey, ez, er);
            RreaInteractionSite const site{ex, ey, ez, er, rrea::kinetic::min(
                amrex::Real(1.0),
                elapsed_start + distance_m / rrea::kinetic::max(
                    c_m_per_s * dt_s, std::numeric_limits<amrex::Real>::min()))};
            auto deposit_local = [&](amrex::Real local_eV) {
                deposit_local_energy_as_ion_pairs(fx, er, ez, w, local_eV);
            };
            RreaPhotonChannel const channel = tables.SamplePhotonChannel(
                E, RreaRng::Uniform01(rng_key(h.newborn
                    ? RreaRngChannel::NewbornCompetingChannel
                    : first ? RreaRngChannel::PhotonChannel
                            : RreaRngChannel::PhotonContinuationChannel)));
            if (channel == RreaPhotonChannel::Compton) {
                if (!RreaApplyComptonFinalState(
                        fx, tables, rng_key, deposit_local, site, w,
                        coupling.LowEnergyCutoffEv(), coupling.PhotonCutoffEv(),
                        E, h.dir_x, h.dir_y, h.dir_z)) {
                    return HistoryEnd::Consumed;
                }
                h.scattered = true;
                path = chord_from(ex, ey, ez, rrea::kinetic::max(
                    amrex::Real(0.0), path.full_length_m - distance_m));
                elapsed_start = site.elapsed_step_fraction;
                continue;
            }
            if (channel == RreaPhotonChannel::Photoelectric) {
                RreaApplyPhotoelectricFinalState(
                    fx, tables, rng_key, deposit_local, site, w,
                    coupling.LowEnergyCutoffEv(), E, h.dir_x, h.dir_y, h.dir_z);
                return HistoryEnd::Consumed;
            }
            RreaApplyPairFinalState(
                fx, tables, rng_key, deposit_local,
                [&](amrex::Real px, amrex::Real py, amrex::Real pz,
                    amrex::Real positron_eV) {
                    spawn_positron_annihilation(
                        fx, ex, ey, ez, er, ez, px, py, pz, w, positron_eV, true,
                        h.rng_particle_id, index.value, site.elapsed_step_fraction);
                },
                site, w, coupling.LowEnergyCutoffEv(), E, channel,
                h.dir_x, h.dir_y, h.dir_z);
            return HistoryEnd::Consumed;
        }
    }

    // Live e-/e+ adapter.  RREA owns the whole trajectory, so WarpX's kick
    // and push are discarded: the history restarts from the captured
    // pre-push point and momentum.  A survivor hands WarpX the point it
    // actually reached -- z exactly as the plane-crossing scan ended, since
    // step n+1 re-captures prev_z from it and the signed-flux rule only
    // composes for segments chained end to end; theta with r, or the
    // Cartesian identity rotates against the direction.
    template<class Effects, class Id>
    RREA_HOST_DEVICE void Charged(
        Effects& fx, ChargedSoAView const& soa, long ip, Id particle_id,
        int particle_cpu, bool is_positron) const
    {
        if (!particle_id.is_valid()) {
            return;
        }
        ParticleHistory h;
        h.species = is_positron ? RreaSecondarySpecies::Positron
                                : RreaSecondarySpecies::Electron;
        h.rng_particle_id = rrea::kinetic::stable_particle_key(
            static_cast<amrex::Long>(particle_id), particle_cpu);
        fx.SetParentContext(h.rng_particle_id, h.species);
        h.weight = soa.w[ip];
        if (h.weight <= amrex::Real(0.0)) {
            // A zero-weight parent carries nothing; close its ledger explicitly.
            fx.BeginGroup(RreaSecondaryProcess::LowEnergyDemotion, true);
            fx.SetExplicitZeroConservationClosure();
            fx.EndGroup();
            particle_id.make_invalid();
            fx.invalidated = true;
            return;
        }
        if (soa.previous_r[ip] > geom.ProbHi(0) + amrex::Real(1.0e-9) * geom.ProbHi(0)) {
            rrea::KineticFail("RREA charged particle survived outside the domain");
        }
        amrex::Real const r = soa.previous_r[ip];
        amrex::Real const theta = soa.previous_theta[ip];
        h.x_m = r * std::cos(theta);
        h.y_m = r * std::sin(theta);
        h.z_m = soa.previous_z[ip];
        h.dir_x = soa.previous_ux[ip];
        h.dir_y = soa.previous_uy[ip];
        h.dir_z = soa.previous_uz[ip];
        h.energy_eV = RreaKineticEnergyEvFromProperVelocity(h.dir_x, h.dir_y, h.dir_z);
        h.electron_counted = soa.electron_counted && soa.electron_counted[ip] != 0;
        HistoryEnd const end = AdvanceChargedHistory(fx, h);
        if (soa.electron_counted) soa.electron_counted[ip] = h.electron_counted;
        if (end != HistoryEnd::Consumed) {
            amrex::Real const u = rrea::RreaElectronProperVelocityFromKineticEv(h.energy_eV);
            soa.ux[ip] = static_cast<amrex::ParticleReal>(u * h.dir_x);
            soa.uy[ip] = static_cast<amrex::ParticleReal>(u * h.dir_y);
            soa.uz[ip] = static_cast<amrex::ParticleReal>(u * h.dir_z);
        }
        if (end == HistoryEnd::Survive) {
            soa.r[ip] = static_cast<amrex::ParticleReal>(std::hypot(h.x_m, h.y_m));
            soa.theta[ip] = static_cast<amrex::ParticleReal>(std::atan2(h.y_m, h.x_m));
            soa.z[ip] = static_cast<amrex::ParticleReal>(h.z_m);
            fx.repositioned = true;
        } else if (end == HistoryEnd::Consumed || h.rrea_owns_escape) {
            particle_id.make_invalid();
            fx.invalidated = true;
        }
    }

    // Live photon adapter.  The history's chord starts at the captured
    // pre-push point along the photon's (push-invariant) direction, the same
    // straight line WarpX pushed at c.  A survivor is handed back the end the
    // history's escape policy judged; only a Compton scatter moves it off
    // WarpX's line, possibly across a tile, hence the Redistribute flag.
    template<class Effects, class Id>
    RREA_HOST_DEVICE void Photon(
        Effects& fx, ChargedSoAView const& soa, long ip, Id particle_id,
        int particle_cpu) const
    {
        if (!particle_id.is_valid()) {
            return;
        }
        ParticleHistory h;
        h.species = RreaSecondarySpecies::Photon;
        h.rng_particle_id = rrea::kinetic::stable_particle_key(
            static_cast<amrex::Long>(particle_id), particle_cpu);
        fx.SetParentContext(h.rng_particle_id, h.species);
        amrex::Real const r = soa.previous_r[ip];
        amrex::Real const theta = soa.previous_theta[ip];
        h.z_m = soa.previous_z[ip];
        if (r > geom.ProbHi(0) + amrex::Real(1.0e-9) * geom.ProbHi(0)
            || h.z_m < geom.ProbLo(1) - amrex::Real(1.0e-9)
            || h.z_m > geom.ProbHi(1) + amrex::Real(1.0e-9) * geom.ProbHi(1)) {
            rrea::KineticFail("RREA photon survived outside the domain");
        }
        h.weight = soa.w[ip];
        h.x_m = r * std::cos(theta);
        h.y_m = r * std::sin(theta);
        h.dir_x = soa.ux[ip];
        h.dir_y = soa.uy[ip];
        h.dir_z = soa.uz[ip];
        h.energy_eV = std::sqrt(h.dir_x * h.dir_x + h.dir_y * h.dir_y + h.dir_z * h.dir_z);
        HistoryEnd const end = AdvancePhotonHistory(fx, h);
        if (end == HistoryEnd::Survive) {
            soa.r[ip] = static_cast<amrex::ParticleReal>(std::hypot(h.x_m, h.y_m));
            soa.theta[ip] = static_cast<amrex::ParticleReal>(std::atan2(h.y_m, h.x_m));
            soa.z[ip] = static_cast<amrex::ParticleReal>(h.z_m);
            if (h.scattered) {
                soa.ux[ip] = static_cast<amrex::ParticleReal>(h.energy_eV * h.dir_x);
                soa.uy[ip] = static_cast<amrex::ParticleReal>(h.energy_eV * h.dir_y);
                soa.uz[ip] = static_cast<amrex::ParticleReal>(h.energy_eV * h.dir_z);
                fx.repositioned = true;
            }
        } else if (end == HistoryEnd::Consumed || h.rrea_owns_escape) {
            particle_id.make_invalid();
            fx.invalidated = true;
        }
    }

    // Transient adapter: a secondary born at elapsed fraction b of this step
    // flies its residual (1-b)dt through the same history as a live particle,
    // on its root's RNG stream, and a survivor joins the population at the
    // end of the step.
    template<class Effects>
    RREA_HOST_DEVICE bool AdvanceNewborn(RreaSecondaryParticle& p, Effects& fx) const
    {
        fx.SetParentContext(p.root_rng_particle_id, p.species, /*transient=*/true);
        if (!RreaCartesianPositionInHalfOpenRzDomain(
                p.x_m, p.y_m, p.z_m, geom.ProbLo(0), geom.ProbHi(0),
                geom.ProbLo(1), geom.ProbHi(1))) {
            rrea::KineticFail("RREA transient was born outside the half-open "
                              "domain; the uncommitted transaction is rejected");
        }
        ParticleHistory h{p.species, true, p.weight, p.x_m, p.y_m, p.z_m,
            p.dir_x, p.dir_y, p.dir_z, p.kinetic_or_photon_energy_eV,
            p.birth_elapsed_step_fraction, p.root_rng_particle_id,
            p.transient_lineage, p.root_species};
        h.electron_counted = p.electron_counted;
        HistoryEnd const end = p.species == RreaSecondarySpecies::Photon
            ? AdvancePhotonHistory(fx, h)
            : AdvanceChargedHistory(fx, h);
        if (end != HistoryEnd::Survive) {
            return false;
        }
        p.x_m = h.x_m;
        p.y_m = h.y_m;
        p.z_m = h.z_m;
        p.dir_x = h.dir_x;
        p.dir_y = h.dir_y;
        p.dir_z = h.dir_z;
        p.kinetic_or_photon_energy_eV = h.energy_eV;
        p.birth_elapsed_step_fraction = amrex::Real(1.0);
        p.electron_counted = h.electron_counted;
        return true;
    }
};
static_assert(std::is_trivially_copyable_v<KineticStep>,
              "Kinetic kernel context must not own host containers");
static_assert(std::is_trivially_copyable_v<ChargedSoAView>);
} // namespace rrea::warpx
#endif
