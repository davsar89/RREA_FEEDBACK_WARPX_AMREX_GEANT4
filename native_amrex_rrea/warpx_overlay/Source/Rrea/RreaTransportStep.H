#ifndef RREA_TRANSPORT_STEP_H
#define RREA_TRANSPORT_STEP_H

// One RREA transport step: the per-step context every species phase shares,
// and the primitives they share with it.  The phases themselves are four
// translation units (RreaElectronTransport.cpp, RreaPhotonTransport.cpp,
// RreaPositronTransport.cpp, RreaSecondaryCommit.cpp), each defining one
// TransportStep method; RreaParticleInteraction.cpp keeps only the hook that
// builds the context and runs the four phases in order.
//
// What lives here is exactly what more than one phase needs: the captured
// pre-push component indices, the substep-count sizing rules, the frozen
// elastic/drag rates, the Goudsmit-Saunderson scatter application, the
// charged prologue/escape/writeback of the electron and positron loops, and
// the local-energy ion-pair deposit.  Per-species POLICY stays in the phase
// that owns it: the electron and positron loops remain separate.
//
// Pinned by rrea_charged_interleave_smoke (the kinematics primitives),
// rrea_secondary_group_smoke (the deposit and group contracts) and
// rrea_warpx_transport_apply (the assembled step).

#include "RreaChargedFinalStates.H"
#include "RreaChargedInterleave.H"
#include "RreaMaterialPath.H"
#include "RreaParticleInteraction.H"
#include "RreaPathFieldWork.H"
#include "RreaPhotonFinalStates.H"
#include "RreaRng.H"
#include "RreaTransportLog.H"
#include "RreaWarpXCoupling.H"

#include "rrea/RreaDebugOptions.H"
#include "rrea/RreaInteractionTables.H"
#include "rrea/RreaLorentzPush.H"
#include "rrea/RreaSeedSchedule.H"

#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "WarpX.H"

#include <AMReX.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iterator>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rrea::warpx {

inline constexpr amrex::Real kElectronRestEnergyEv = kRreaElectronRestEnergyEv;
inline constexpr amrex::Real kSpeedOfLightMPerS = kRreaSpeedOfLightMPerS;
// Charge-to-mass factor for the proper-velocity impulse: c^2 divided by the
// rest energy in eV, so E [V/m] times dt [s] yields a proper-velocity
// increment directly.  Sign is supplied by the caller's charge_sign.
inline constexpr amrex::Real kChargeOverMassCPerKg =
    kSpeedOfLightMPerS * kSpeedOfLightMPerS / kElectronRestEnergyEv;

struct PreviousRzComponentIndices {
    int r = -1;
    int z = -1;
    int theta = -1;
    int ux = -1;
    int uy = -1;
    int uz = -1;
};

inline PreviousRzComponentIndices require_previous_rz_components(
    WarpXParticleContainer& particles,
    std::string const& species_name)
{
    for (char const* component : {
             "rrea_prev_r", "rrea_prev_z", "rrea_prev_theta",
             "rrea_prev_ux", "rrea_prev_uy", "rrea_prev_uz"}) {
        if (!particles.HasRealComp(component)) {
            amrex::Abort(
                "RREA pre-boundary transport requires runtime component '"
                + std::string(component) + "' on species '" + species_name
                + "'; CaptureRreaPositionsBeforePush must run before WarpX::OneStep");
        }
    }
    return PreviousRzComponentIndices{
        particles.GetRealCompIndex("rrea_prev_r"),
        particles.GetRealCompIndex("rrea_prev_z"),
        particles.GetRealCompIndex("rrea_prev_theta"),
        particles.GetRealCompIndex("rrea_prev_ux"),
        particles.GetRealCompIndex("rrea_prev_uy"),
        particles.GetRealCompIndex("rrea_prev_uz")};
}

// Substeps needed to keep `amount` at or below `per_substep` in each one, never
// fewer than one. Shared by the electron, positron, and newborn charged paths.
inline std::uint64_t substep_driver_count(amrex::Real amount, amrex::Real per_substep)
{
    return static_cast<std::uint64_t>(
        amrex::max(amrex::Real(1.0), std::ceil(amount / per_substep)));
}

// Cartesian point plus cylindrical radius at a full-path fraction of one chord.
inline void point_and_radius_at_fraction(
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

inline std::uint64_t selected_subcycles(
    amrex::Real max_tau,
    RreaParticleInteractionConfig const& config,
    bool& cap_violation)
{
    if (!RreaTransportSubcycleControlsValid(
            config, /*require_strict_guards=*/false)) {
        amrex::Abort(
            "RREA transport received invalid optical-depth subcycle controls");
    }
    amrex::Real const target = config.max_optical_depth_per_substep;
    std::uint64_t const requested = static_cast<std::uint64_t>(
        std::max(amrex::Real(1.0), std::ceil(max_tau / target)));
    std::uint64_t const cap = config.max_subcycles;
    cap_violation = requested > cap;
    return std::min(requested, cap);
}

// Substeps small enough that no uniform slice of the chord carries more than
// the configured competing optical depth.  total_stp_per_m is the summed
// frozen inverse length of every channel in the competition, so the worst
// slice is that times the slice's density path -- one geometric measurement
// per candidate count, independent of how many channels compete.
inline std::uint64_t selected_material_subcycles(
    amrex::Real max_total_tau,
    std::vector<MaterialPathInterval> const& intervals,
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
            * max_uniform_subcycle_density_path_m(intervals, path, subcycles);
        amrex::Real const tolerance = amrex::Real(64.0)
            * std::numeric_limits<amrex::Real>::epsilon()
            * amrex::max(amrex::Real(1.0), target);
        if (maximum <= target + tolerance) {
            return subcycles;
        }
        if (subcycles >= cap) {
            cap_violation = true;
            return subcycles;
        }
        long double const estimate = std::ceil(
            static_cast<long double>(subcycles)
            * static_cast<long double>(maximum / target));
        std::uint64_t next = subcycles + 1U;
        if (estimate >= static_cast<long double>(cap)) {
            next = cap;
        } else if (estimate > static_cast<long double>(next)) {
            next = static_cast<std::uint64_t>(estimate);
        }
        subcycles = std::min(next, cap);
    }
}

inline void abort_if_strict_guard_failed(
    RreaParticleInteractionConfig const& config,
    bool optical_depth_violation)
{
    if (config.strict_transport_guards && optical_depth_violation) {
        amrex::Abort("RREA strict transport guard: optical-depth subcycle cap exceeded");
    }
}


// Elastic COLLISION rate at STP, 1/lambda_el(E), in inverse metres.  The
// Goudsmit-Saunderson path variable is n = s/lambda_el = this times the
// segment's density path.
inline amrex::Real elastic_inverse_length_stp(
    rrea::RreaInteractionTables const& tables,
    bool is_positron,
    amrex::Real kinetic_energy_eV)
{
    amrex::Real const mfp = tables.ElasticMeanFreePath(
        is_positron, kinetic_energy_eV, amrex::Real(1.0));
    return RreaFrozenStpRate(
        mfp > amrex::Real(0.0) ? amrex::Real(1.0) / mfp : amrex::Real(0.0));
}

inline amrex::Real electron_realized_continuous_loss_ev_per_m(
    rrea::RreaInteractionTables const& tables,
    amrex::Real kinetic_energy_eV,
    amrex::Real density_ratio)
{
    if (!(kinetic_energy_eV > amrex::Real(0.0))
        || !(density_ratio > amrex::Real(0.0))) {
        return amrex::Real(0.0);
    }
    // Energy actually removed on an event-free path: restricted collision
    // loss plus sub-cut radiative loss.  Expected discrete events do not
    // remove energy unless they occur.
    return tables.ElectronCollisionLoss(kinetic_energy_eV, density_ratio)
        + tables.ElectronSoftRadiativeDrag(kinetic_energy_eV, density_ratio);
}

inline amrex::Real electron_ensemble_mean_drag_ev_per_m(
    rrea::RreaInteractionTables const& tables,
    amrex::Real kinetic_energy_eV,
    amrex::Real density_ratio,
    bool hard_moller_discrete_enabled)
{
    if (!(kinetic_energy_eV > amrex::Real(0.0))
        || !(density_ratio > amrex::Real(0.0))) {
        return amrex::Real(0.0);
    }
    // Ensemble force balance includes the mean losses of every discrete
    // channel in addition to the event-free continuous loss.
    amrex::Real const discrete_delta_ray_drag_eV_per_m =
        hard_moller_discrete_enabled
        ? tables.ElectronHardMollerEnergyLoss(kinetic_energy_eV, density_ratio)
        : amrex::Real(0.0);
    return electron_realized_continuous_loss_ev_per_m(
               tables, kinetic_energy_eV, density_ratio)
        + discrete_delta_ray_drag_eV_per_m
        + tables.ElectronTrackedBremsEnergyLoss(
            kinetic_energy_eV, density_ratio);
}

inline void defer_cd_electron_plane_crossings(
    TransportSideEffects& effects,
    RreaWarpXCoupling const& coupling,
    rrea::RreaInteractionTables const& tables,
    RreaTransportPathSegment const& path,
    std::vector<MaterialPathInterval> const& material_intervals,
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
    amrex::Real begin_x = amrex::Real(0.0);
    amrex::Real begin_y = amrex::Real(0.0);
    amrex::Real begin_z = amrex::Real(0.0);
    amrex::Real end_x = amrex::Real(0.0);
    amrex::Real end_y = amrex::Real(0.0);
    amrex::Real end_z = amrex::Real(0.0);
    path.PointAtFullPathFraction(
        begin_fraction, begin_x, begin_y, begin_z);
    path.PointAtFullPathFraction(end_fraction, end_x, end_y, end_z);
    amrex::ignore_unused(begin_x, begin_y, end_x, end_y);
    amrex::Real const fixed_energy_stp_eV_per_m =
        electron_realized_continuous_loss_ev_per_m(
            tables, stopping_energy_eV, amrex::Real(1.0));

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
            amrex::Real x_m = amrex::Real(0.0);
            amrex::Real y_m = amrex::Real(0.0);
            amrex::Real z_m = amrex::Real(0.0);
            path.PointAtFullPathFraction(crossing_fraction, x_m, y_m, z_m);
            amrex::Real const radial_position_m = std::hypot(x_m, y_m);
            amrex::Real const raw_loss_to_crossing_eV =
                fixed_energy_stp_eV_per_m
                * density_path_m(
                    material_intervals, path, begin_fraction, crossing_fraction);
            amrex::Real const kinetic_energy_at_crossing_eV = amrex::max(
                amrex::Real(0.0),
                kinetic_energy_before_segment_eV
                    - amrex::min(segment_drag_loss_eV, raw_loss_to_crossing_eV));

            bool local_runaway = false;
            if (kinetic_energy_at_crossing_eV >= coupling.LowEnergyCutoffEv()) {
                amrex::Real const density_ratio = positive_or(
                    coupling.TransportDensityRatioAtZ(z_m), amrex::Real(1.0));
                RreaFieldSample const field =
                    coupling.GatherFieldRZ(0, radial_position_m, z_m);
                amrex::Real const continuous_drag_eV_per_m =
                    electron_ensemble_mean_drag_ev_per_m(
                        tables,
                        kinetic_energy_at_crossing_eV,
                        density_ratio,
                        hard_moller_discrete_enabled);
                local_runaway = RreaLocalRunawayFieldMagnitudeGate(
                    field.er_v_per_m,
                    field.ez_v_per_m,
                    continuous_drag_eV_per_m);
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
        amrex::Abort("RREA C&D plane crossing enumeration received invalid geometry");
    }
}

class ScopedMultipleMFIters {
public:
    ScopedMultipleMFIters()
        : m_previous(amrex::MFIter::allowMultipleMFIters(1))
    {}

    ~ScopedMultipleMFIters()
    {
        amrex::MFIter::allowMultipleMFIters(m_previous);
    }

    ScopedMultipleMFIters(ScopedMultipleMFIters const&) = delete;
    ScopedMultipleMFIters& operator=(ScopedMultipleMFIters const&) = delete;

private:
    int m_previous = 0;
};

// RNG channels for one species' Goudsmit-Saunderson elastic draws: electrons
// use channels 89-93, positrons 94-98. Both pack the interaction index as
// (substep << 8 | k).
struct GsScatterChannels {
    RreaRngChannel few_count;
    RreaRngChannel few_scatter;
    RreaRngChannel few_azimuth;
    RreaRngChannel angle;
    RreaRngChannel azimuth;
};

// Goudsmit-Saunderson elastic scatter over one sub-interval -- the shared
// application unit for the electron, positron and newborn paths. Below the few-collision threshold the
// deflections are drawn one collision at a time from a Poisson count; above it
// a single GS angle covers the whole interval.  The selection rate stays frozen
// at the caller's energy.  RNG identity is a caller policy: gs_key maps
// (channel, inner draw index) to the key -- live loops pack
// (substep << 8 | inner) on the particle's own stream, the newborn loop packs
// its transient lineage on the parent stream with its Newborn* channels.
template <typename KeyFn>
void apply_gs_elastic_scatter(
    std::vector<MaterialPathInterval> const& material_intervals,
    RreaTransportPathSegment const& path,
    amrex::Real begin_fraction,
    amrex::Real end_fraction,
    rrea::RreaInteractionTables const& tables,
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
        * density_path_m(material_intervals, path, begin_fraction, end_fraction);
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
            amrex::Abort(
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

// PHASE MAP (grep the named waypoints; line numbers rot).  Each phase is a
// TransportStep method; the per-step shared state is its members.
//   1. ELECTRON loop      -- ApplyElectronTransport: lambdas
//                            apply_electron_continuous_segment,
//                            apply_electron_gs_scatter, finish_interleaved_substep;
//                            hazards/events via RreaRunChargedEventInterleave.
//   2. PHOTON loop        -- ApplyPhotonTransport: hazard integral,
//                            SamplePhotonChannel, then the photoelectric /
//                            Compton / pair booking branches.
//   3. POSITRON loop      -- ApplyPositronTransport:
//                            apply_positron_continuous_segment,
//                            apply_positron_gs_scatter, Bhabha branches,
//                            annihilation in flight (Heitler final state from
//                            the tables) and at rest.
//   4-6. TRANSACTION, NEWBORN loop, COMMIT tail --
//                            DrainAndCommitSecondaries: collective validation
//                            of every uncommitted secondary group
//                            (charge/energy closure); residual advance of this
//                            step's secondaries on the parent RNG stream
//                            (apply_newborn_drag, apply_newborn_gs), failing
//                            closed like the live loops; terminal writes back
//                            to WarpX and the escape ledger.
// The drag and GS physics of the electron/positron/newborn phases are the
// SAME modules (RreaPlanContinuousDragSegment, apply_gs_elastic_scatter);
// the loops keep only per-species policy.

// One tile's charged SoA pointers, shared by the electron and positron loops
// so the per-particle prologue/escape/writeback helpers of TransportStep see
// exactly the view both species loops do.
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
};

// Everything the shared charged prologue reconstructs for one particle.
struct RunningChargedSetup {
    // Zero-weight parent: the ledger is already booked; the caller
    // invalidates the id and continues.
    bool skip = false;
    std::uint64_t transaction_parent_key = 0;
    amrex::Real particle_weight = amrex::Real(0.0);
    amrex::Real dir_x = amrex::Real(0.0);
    amrex::Real dir_y = amrex::Real(0.0);
    amrex::Real dir_z = amrex::Real(1.0);
    amrex::Real kinetic_energy_eV = amrex::Real(0.0);
    amrex::Real run_x = amrex::Real(0.0);
    amrex::Real run_y = amrex::Real(0.0);
    amrex::Real run_z = amrex::Real(0.0);
    amrex::Real x_particle = amrex::Real(0.0);
    amrex::Real y_particle = amrex::Real(0.0);
    amrex::Real z_particle = amrex::Real(0.0);
    amrex::Real r_particle = amrex::Real(0.0);
    rrea::RreaTransportPathSegment driver_path;
    RreaEndpointEscapePolicy charged_parent_escape;
    amrex::Real ds_m = amrex::Real(0.0);
    std::vector<MaterialPathInterval> driver_intervals;
    amrex::Real driver_span = amrex::Real(0.0);
    amrex::Real field_work_eV = amrex::Real(0.0);
};

// Per-step context shared by the transport phases.
struct TransportStep {
    WarpX& warpx;
    RreaWarpXCoupling& coupling;
    rrea::RreaInteractionTables const& tables;
    RreaParticleInteractionConfig const& m_config;
    int step;
    amrex::Real time_s;
    amrex::Real dt_s;
    amrex::Geometry const& geom;
    bool periodic_z;
    amrex::Real density_ratio;

        // All material/fluid/ledger mutations remain logged until the global
        // projected post-removal population has passed the collective ceiling
        // preflight.  This makes the batch fail closed without committing the
        // non-particle portion of any correlated interaction group first.
        TransportSideEffects pending_side_effects;

        bool invalidated_particles = false;
        bool invalidated_photons = false;
        bool repositioned_photons = false;
        bool invalidated_positrons = false;
        // Set once the charged loops own their trajectories and write a position
        // back.  Until then these stay false and the post-loop branch below is a
        // no-op for both charged species.
        bool repositioned_electrons = false;
        bool repositioned_positrons = false;

        // Local energy deposit -> ion pairs, N = w E_local / W_air
        // (RreaIonPairWeightFromLocalEnergy owns the relation and its
        // citation).  Writes only through the caller-supplied side-effect log
        // so it can run on an OpenMP worker thread; the log is replayed in
        // deterministic chunk order afterwards.
        void deposit_local_energy_as_ion_pairs(
                TransportSideEffects& fx,
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
            // Each created pair is booked with a diagnostic loss of exactly
            // W_air, so the mesh energy ledger records macro_weight*local_energy
            // rather than pair_weight*E.
            amrex::Real const diagnostic_loss_eV = tables.WAirEvPerIonPair();
            fx.AccumulateIonizationEvent(
                0,
                r_m,
                z_m,
                pair_weight,
                amrex::Real(0.0),
                diagnostic_loss_eV);
            fx.tally.RecordTransportIonization(
                pair_weight,
                diagnostic_loss_eV,
                true,
                false);
        }

        void spawn_positron_annihilation(
            TransportSideEffects& fx,
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
            static constexpr amrex::Real electron_rest_energy_eV = kRreaElectronRestEnergyEv;
            RreaNormalizeDirection(axis_x, axis_y, axis_z);

            // Below-cutoff positrons first thermalize residual kinetic energy
            // locally, then
            // annihilate at rest.  In-flight annihilation keeps the positron
            // kinetic energy in the two-photon final state.
            amrex::Real const annihilation_kinetic_eV = at_rest_or_below_cutoff
                ? amrex::Real(0.0)
                : amrex::max(positron_kinetic_energy_eV, amrex::Real(0.0));
            auto const annihilation_state = tables.SamplePositronAnnihilationFinalState(
                annihilation_kinetic_eV,
                RreaRng::Uniform01(RreaRngKey{
                    coupling.RngSeed(),
                    rng_particle_id,
                    static_cast<std::uint64_t>(std::max(step, 0)),
                    interaction_index,
                    static_cast<std::uint64_t>(RreaRngChannel::PositronAnnihilationPolar)}),
                RreaRng::Uniform01(RreaRngKey{
                    coupling.RngSeed(),
                    rng_particle_id,
                    static_cast<std::uint64_t>(std::max(step, 0)),
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
            amrex::Real photon1_energy_eV = amrex::max(
                annihilation_state.photon1_energy_eV,
                amrex::Real(0.0));
            amrex::Real photon2_energy_eV = amrex::max(
                annihilation_state.photon2_energy_eV,
                amrex::Real(0.0));
            amrex::Real photon2_dir_x = -photon1_dir_x;
            amrex::Real photon2_dir_y = -photon1_dir_y;
            amrex::Real photon2_dir_z = -photon1_dir_z;
            if (!annihilation_state.at_rest && photon2_energy_eV > amrex::Real(0.0)) {
                amrex::Real const p_eV = std::sqrt(amrex::max(
                    amrex::Real(0.0),
                    annihilation_kinetic_eV
                        * (annihilation_kinetic_eV
                           + amrex::Real(2.0) * electron_rest_energy_eV)));
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
            fx.AccumulatePositiveIonSourceOnly(
                0,
                r_m,
                z_source_m,
                event_weight,
                amrex::Real(0.0));
            fx.RecordPositronAnnihilation(
                event_weight,
                local_deposit_energy_eV,
                photon1_energy_eV,
                photon2_energy_eV,
                at_rest_or_below_cutoff);
        }


        // Shared electron/positron per-particle prologue of the charged
        // loops: parent ledger/RNG context, the fail-closed zero-weight
        // guard, reconstruction of the running state from the captured
        // pre-push components: transport owns the
        // complete energy/direction sequence; the WarpX kick is discarded
        // from energetics and re-applied per substep as a Strang-split
        // field impulse -- line-integral work as the sole energy authority,
        // the average field vector for direction rotation -- so field work
        // cannot be double-counted, and the driver chord
        // (geometry too -- speed(K_pre)*dt along the
        // particle's own momentum, never WarpX's post-kick displacement)
        // with its clipped escape policy, material intervals and the
        // whole-chord field-work driver estimate.  charge_sign is -1 for
        // electrons and +1 for positrons.  A survivor stored outside the
        // domain aborts for either species: that state is corrupt.
        RunningChargedSetup BeginRunningChargedParticle(
            TransportSideEffects& fx,
            ChargedSoAView const& soa,
            long ip,
            amrex::Long particle_id_value,
            int particle_cpu,
            RreaSecondarySpecies species,
            amrex::Real charge_sign,
            char const* species_label) const
        {
            RunningChargedSetup s;
            s.transaction_parent_key =
                static_cast<std::uint64_t>(std::llabs(particle_id_value))
                ^ (static_cast<std::uint64_t>(particle_cpu) << 40);
            fx.SetParentContext(s.transaction_parent_key, species);
            s.particle_weight = soa.w[ip];
            if (s.particle_weight <= amrex::Real(0.0)) {
                fx.BeginGroup(RreaSecondaryProcess::LowEnergyDemotion, true);
                fx.SetExplicitZeroConservationClosure();
                fx.EndGroup();
                s.skip = true;
                return s;
            }

            if (soa.previous_r[ip] > geom.ProbHi(0)
                    + amrex::Real(1.0e-9) * geom.ProbHi(0)) {
                std::ostringstream survivor;
                survivor << std::setprecision(17)
                    << "RREA " << species_label
                    << " survived outside the domain: prev_r="
                    << soa.previous_r[ip]
                    << " prev_z=" << soa.previous_z[ip]
                    << " r=" << soa.r[ip] << " z=" << soa.z[ip]
                    << " w=" << soa.w[ip]
                    << " ux=" << soa.ux[ip] << " uy=" << soa.uy[ip]
                    << " uz=" << soa.uz[ip]
                    << " id=" << particle_id_value
                    << " cpu=" << particle_cpu
                    << " step=" << step;
                amrex::Abort(survivor.str());
            }
            s.dir_x = soa.previous_ux[ip];
            s.dir_y = soa.previous_uy[ip];
            s.dir_z = soa.previous_uz[ip];
            s.kinetic_energy_eV = RreaKineticEnergyEvFromProperVelocity(
                s.dir_x, s.dir_y, s.dir_z);
            RreaNormalizeDirection(s.dir_x, s.dir_y, s.dir_z);

            amrex::Real const seed_r_m =
                static_cast<amrex::Real>(soa.previous_r[ip]);
            amrex::Real const seed_theta =
                static_cast<amrex::Real>(soa.previous_theta[ip]);
            s.run_x = seed_r_m * std::cos(seed_theta);
            s.run_y = seed_r_m * std::sin(seed_theta);
            s.run_z = static_cast<amrex::Real>(soa.previous_z[ip]);
            // Pre-transport branches book at the pre-push point: they fire
            // before RREA transports anything, and RREA discards WarpX's
            // kick, so by RREA's own model the particle is still here.
            s.x_particle = s.run_x;
            s.y_particle = s.run_y;
            s.z_particle = s.run_z;
            s.r_particle = seed_r_m;

            auto const driver_chord = RreaBuildRunningChargedChord(
                RreaRunningChargedState{
                    s.run_x, s.run_y, s.run_z, s.dir_x, s.dir_y, s.dir_z,
                    s.kinetic_energy_eV},
                dt_s,
                dt_s,
                geom.ProbLo(0),
                geom.ProbHi(0),
                geom.ProbLo(1),
                geom.ProbHi(1),
                periodic_z);
            s.driver_path = driver_chord.path;
            // Re-evaluated on whichever microchord actually leaves.  The
            // clipped policy is mandatory once a clipped in-domain position
            // is written back: WarpX's boundary pass never sees the raw
            // overshoot, so without it an escaper stays alive as a
            // full-energy runaway while its energy is also booked escaped.
            s.charged_parent_escape = RreaClippedTransportEscapePolicyForPath(
                s.driver_path,
                std::hypot(s.driver_path.end_x_m, s.driver_path.end_y_m),
                s.driver_path.end_z_m);
            s.ds_m = driver_chord.valid ? s.driver_path.in_domain_length_m
                                        : amrex::Real(0.0);
            s.driver_intervals = build_material_path_intervals(
                coupling,
                s.driver_path,
                density_ratio);
            // Budgets are integrated over the clipped in-domain span while
            // the loop transports the whole dt; rescale or the boundary
            // population overshoots every cap by 1/span.
            s.driver_span = amrex::max(
                s.driver_path.in_domain_end_fraction, amrex::Real(1.0e-6));
            s.field_work_eV = path_field_work_and_average(
                    coupling, s.driver_path, charge_sign,
                    amrex::Real(0.0), amrex::Real(1.0))
                .work_eV / s.driver_span;
            return s;
        }

        // Shared electron/positron escape closure: the running (K, dir) IS
        // the final state under the substep engine, so hand WarpX the
        // reconstructed momentum; then one atomic deposit owns the net
        // prefix, the actual final microchord (including a refused
        // half-open endpoint), and the outward flux that removes the valid
        // part of the CIC cloud through the physical boundary face.
        // signed_mesh_weight carries the species charge sign.
        void CommitChargedEscape(
            TransportSideEffects& fx,
            ChargedSoAView const& soa,
            long ip,
            amrex::Real particle_weight,
            amrex::Real kinetic_energy_eV,
            amrex::Real dir_x,
            amrex::Real dir_y,
            amrex::Real dir_z,
            amrex::Real r_particle,
            amrex::Real z_particle,
            rrea::RreaTransportPathSegment const& transport_path,
            RreaEndpointEscapePolicy const& charged_parent_escape,
            amrex::Real signed_mesh_weight) const
        {
            amrex::Real const escape_u_mag =
                rrea::RreaElectronProperVelocityFromKineticEv(kinetic_energy_eV);
            soa.ux[ip] = static_cast<amrex::ParticleReal>(
                escape_u_mag * dir_x);
            soa.uy[ip] = static_cast<amrex::ParticleReal>(
                escape_u_mag * dir_y);
            soa.uz[ip] = static_cast<amrex::ParticleReal>(
                escape_u_mag * dir_z);
            fx.BeginGroup(
                RreaSecondaryProcess::ParticleEscape,
                /*remove_parent=*/true,
                charged_parent_escape.ParentStillValidAtPreflight());
            fx.SetChargedEscapeConservationClosure(
                particle_weight, kinetic_energy_eV);
            fx.AccumulateChargedEscape(
                0, r_particle, z_particle,
                RreaChargedBoundaryCrossingForPath(transport_path),
                signed_mesh_weight, kRreaSegmentChannelEscaped);
            fx.EndGroup();
        }

        // Shared survivor writeback: RREA owns the trajectory, so hand
        // WarpX the point the particle actually reached rather than its
        // straight push along the pre-push momentum.  z is written from
        // run_z, which is exactly the z the plane-crossing scan ended at:
        // the signed-flux half-open rule only composes for segments chained
        // end to end, and step n+1 re-captures prev_z from this stored
        // value, so any gap here double-counts every C&D plane inside it --
        // and M is read off the cumulative net.  theta must be written with
        // r or the Cartesian identity rotates against (dir_x, dir_y).
        void CommitSurvivorChargedWriteback(
            TransportSideEffects& fx,
            ChargedSoAView const& soa,
            long ip,
            amrex::Real kinetic_energy_eV,
            amrex::Real dir_x,
            amrex::Real dir_y,
            amrex::Real dir_z,
            amrex::Real run_x,
            amrex::Real run_y,
            amrex::Real run_z,
            amrex::Real r_particle,
            amrex::Real z_particle,
            amrex::Real endpoint_r_m,
            amrex::Real signed_mesh_weight) const
        {
            amrex::Real const u_mag =
                rrea::RreaElectronProperVelocityFromKineticEv(kinetic_energy_eV);
            soa.ux[ip] = static_cast<amrex::ParticleReal>(u_mag * dir_x);
            soa.uy[ip] = static_cast<amrex::ParticleReal>(u_mag * dir_y);
            soa.uz[ip] = static_cast<amrex::ParticleReal>(u_mag * dir_z);
            soa.r[ip] = static_cast<amrex::ParticleReal>(endpoint_r_m);
            soa.theta[ip] = static_cast<amrex::ParticleReal>(
                std::atan2(run_y, run_x));
            soa.z[ip] = static_cast<amrex::ParticleReal>(run_z);
            // Survivor net chord: captured start to the committed endpoint
            // the SoA writeback above just handed WarpX.
            fx.AccumulateChargedSegment(
                0, r_particle, z_particle, endpoint_r_m, run_z,
                signed_mesh_weight, 0);
            fx.repositioned = true;
        }

        // Truncate a newborn's endpoint just inside the half-open domain:
        // WarpX's next push re-books any escape through the certified path.
        // Pulls the fraction back geometrically until the point qualifies;
        // false (a pathological chord) makes the caller abandon the advance.
        [[nodiscard]] bool NewbornFinalPoint(
            rrea::RreaTransportPathSegment const& path,
            amrex::Real fraction,
            amrex::Real& out_x,
            amrex::Real& out_y,
            amrex::Real& out_z) const;

    void ApplyElectronTransport();
    void ApplyPhotonTransport();
    void ApplyPositronTransport();
    // Residual advance of one secondary through the (1 - birth fraction) of
    // the step it was born into, on the parent's RNG streams; true = the
    // newborn survives to become an insertion payload.  Defined in
    // RreaNewbornAdvance.cpp; called from DrainAndCommitSecondaries.
    [[nodiscard]] bool AdvanceChargedNewborn(
        RreaSecondaryParticle& p, TransportSideEffects& newborn_fx) const;
    [[nodiscard]] bool AdvancePhotonNewborn(
        RreaSecondaryParticle& p, TransportSideEffects& newborn_fx) const;
    void DrainAndCommitSecondaries();
};

}  // namespace rrea::warpx

#endif
