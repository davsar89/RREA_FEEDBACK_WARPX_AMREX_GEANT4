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

inline constexpr amrex::Real kElectronRestEnergyEv = kRreaElectronRestEnergyEv;
inline constexpr amrex::Real kSpeedOfLightMPerS = kRreaSpeedOfLightMPerS;
// Charge-to-mass factor for the proper-velocity impulse: c^2 divided by the
// rest energy in eV, so E [V/m] times dt [s] yields a proper-velocity
// increment directly.  Sign is supplied by the caller's charge_sign.
inline constexpr amrex::Real kChargeOverMassCPerKg =
    kSpeedOfLightMPerS * kSpeedOfLightMPerS / kElectronRestEnergyEv;

// Substeps needed to keep `amount` at or below `per_substep` in each one, never
// fewer than one. Shared by the electron, positron, and newborn charged paths.
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
    if (!RreaTransportSubcycleControlsValid(
            config, /*require_strict_guards=*/false)) {
        rrea::KineticFail(
            "RREA transport received invalid optical-depth subcycle controls");
    }
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
    rrea::KineticVector<MaterialPathInterval> const& intervals,
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

RREA_HOST_DEVICE inline void abort_if_strict_guard_failed(
    RreaParticleInteractionConfig const& config,
    bool optical_depth_violation)
{
    if (config.strict_transport_guards && optical_depth_violation) {
        rrea::KineticFail("RREA strict transport guard: optical-depth subcycle cap exceeded");
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

RREA_HOST_DEVICE inline amrex::Real electron_realized_continuous_loss_ev_per_m(
    rrea::RreaInteractionTablesView const& tables,
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

RREA_HOST_DEVICE inline amrex::Real electron_ensemble_mean_drag_ev_per_m(
    rrea::RreaInteractionTablesView const& tables,
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

template<class Effects>
RREA_HOST_DEVICE inline void defer_cd_electron_plane_crossings(
    Effects& effects,
    KineticEnvironment const& coupling,
    rrea::RreaInteractionTablesView const& tables,
    RreaTransportPathSegment const& path,
    rrea::KineticVector<MaterialPathInterval> const& material_intervals,
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
    rrea::kinetic::ignore_unused(begin_x, begin_y, end_x, end_y);
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
            amrex::Real const kinetic_energy_at_crossing_eV = rrea::kinetic::max(
                amrex::Real(0.0),
                kinetic_energy_before_segment_eV
                    - rrea::kinetic::min(segment_drag_loss_eV, raw_loss_to_crossing_eV));

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
        rrea::KineticFail("RREA C&D plane crossing enumeration received invalid geometry");
    }
}

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
RREA_HOST_DEVICE void apply_gs_elastic_scatter(
    rrea::KineticVector<MaterialPathInterval> const& material_intervals,
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
    rrea::KineticVector<MaterialPathInterval> driver_intervals;
    amrex::Real driver_span = amrex::Real(0.0);
    amrex::Real field_work_eV = amrex::Real(0.0);
};

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
    bool periodic_z=false;
    amrex::Real density_ratio=1;
    RREA_HOST_DEVICE bool TransientEnergyIsCertified(RreaSecondaryParticle const& p) const {

            auto in_range = [](amrex::Real energy_eV,
                               rrea::RreaCertifiedEnergyRange const& range) {
                return std::isfinite(static_cast<double>(energy_eV))
                    && std::isfinite(static_cast<double>(range.min_eV))
                    && std::isfinite(static_cast<double>(range.max_eV))
                    && range.min_eV > amrex::Real(0.0)
                    && range.max_eV >= range.min_eV
                    && range.Contains(energy_eV);
            };
            switch (p.species) {
            case RreaSecondarySpecies::Electron:
                return in_range(
                    p.kinetic_or_photon_energy_eV,
                    tables.ElectronCertifiedEnergyRangeEv());
            case RreaSecondarySpecies::Positron:
                return in_range(
                    p.kinetic_or_photon_energy_eV,
                    tables.PositronCertifiedEnergyRangeEv());
            case RreaSecondarySpecies::Photon:
                return in_range(
                    p.kinetic_or_photon_energy_eV,
                    tables.PhotonCertifiedEnergyRangeEv());
            }
            return false;

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
            static constexpr amrex::Real electron_rest_energy_eV = kRreaElectronRestEnergyEv;
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
        template<class Effects>
        RREA_HOST_DEVICE RunningChargedSetup BeginRunningChargedParticle(
            Effects& fx,
            ChargedSoAView const& soa,
            long ip,
            amrex::Long particle_id_value,
            int particle_cpu,
            RreaSecondarySpecies species,
            amrex::Real charge_sign,
            char const* /*species_label*/) const
        {
            RunningChargedSetup s;
            s.transaction_parent_key =
                static_cast<std::uint64_t>(rrea::kinetic::abs_id(particle_id_value))
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
                rrea::KineticFail(" survived outside the domain: prev_r=");
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
            s.driver_span = rrea::kinetic::max(
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
        template<class Effects>
        RREA_HOST_DEVICE void CommitChargedEscape(
            Effects& fx,
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
        template<class Effects>
        RREA_HOST_DEVICE void CommitSurvivorChargedWriteback(
            Effects& fx,
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

RREA_HOST_DEVICE bool NewbornFinalPoint(
    rrea::RreaTransportPathSegment const& path,
    amrex::Real fraction,
    amrex::Real& out_x,
    amrex::Real& out_y,
    amrex::Real& out_z) const
{
    amrex::Real const span = fraction - path.in_domain_begin_fraction;
    amrex::Real pullback = amrex::Real(1.0e-9);
    for (int attempt = 0; attempt < 5; ++attempt) {
        amrex::Real const trial = attempt == 0
            ? fraction
            : rrea::kinetic::max(
                path.in_domain_begin_fraction,
                fraction - pullback * rrea::kinetic::max(span, amrex::Real(1.0e-12)));
        path.PointAtFullPathFraction(trial, out_x, out_y, out_z);
        if (RreaCartesianPositionInHalfOpenRzDomain(
                out_x, out_y, path.WrapZ(out_z),
                geom.ProbLo(0), geom.ProbHi(0),
                geom.ProbLo(1), geom.ProbHi(1))) {
            out_z = path.WrapZ(out_z);
            return true;
        }
        if (attempt > 0) {
            pullback *= amrex::Real(100.0);
        }
    }
    return false;
}

template<class Effects>
RREA_HOST_DEVICE bool AdvanceChargedNewborn(
    RreaSecondaryParticle& p,
    Effects& newborn_fx) const
{
    bool const is_positron = p.species == RreaSecondarySpecies::Positron;
    newborn_fx.SetParentContext(
        p.root_rng_particle_id, p.species, /*transient=*/true);
    amrex::Real const charge_sign =
        is_positron ? amrex::Real(1.0) : amrex::Real(-1.0);
    // Net-chord start for every residual leg below: the birth point, captured
    // before the survivor path overwrites the payload.
    amrex::Real const birth_r_m = std::hypot(p.x_m, p.y_m);
    amrex::Real const birth_z_m = p.z_m;
    amrex::Real const residual_fraction =
        RreaNewbornResidualFraction(p.birth_elapsed_step_fraction);
    amrex::Real kinetic_energy_eV = p.kinetic_or_photon_energy_eV;
    if (!RreaCartesianPositionInHalfOpenRzDomain(
            p.x_m, p.y_m, p.z_m,
            geom.ProbLo(0), geom.ProbHi(0),
            geom.ProbLo(1), geom.ProbHi(1))) {
        rrea::KineticFail(
            "RREA charged transient was born outside the half-open "
            "domain; the uncommitted transaction is rejected");
    }
    amrex::Real dir_x = p.dir_x;
    amrex::Real dir_y = p.dir_y;
    amrex::Real dir_z = p.dir_z;
    RreaNormalizeDirection(dir_x, dir_y, dir_z);
    bool const hard_moller_discrete_enabled =
        !m_config.debug_disable_hard_moller_secondaries;
    amrex::Real const hard_secondary_threshold_eV =
        coupling.HardMollerSecondaryThresholdEv();
    auto electron_state_policy_at = [&](amrex::Real x_m,
                                        amrex::Real y_m,
                                        amrex::Real z_m,
                                        amrex::Real energy_eV) {
        amrex::Real const r_m = std::hypot(x_m, y_m);
        RreaFieldSample const field = coupling.GatherFieldRZ(0, r_m, z_m);
        amrex::Real const local_density_ratio = positive_or(
            coupling.TransportDensityRatioAtZ(z_m), amrex::Real(1.0));
        return RreaEvaluateElectronStatePolicy(
            energy_eV,
            coupling.LowEnergyCutoffEv(),
            hard_secondary_threshold_eV,
            std::hypot(field.er_v_per_m, field.ez_v_per_m),
            electron_ensemble_mean_drag_ev_per_m(
                tables,
                rrea::kinetic::max(energy_eV, coupling.LowEnergyCutoffEv()),
                local_density_ratio,
                hard_moller_discrete_enabled));
    };
    auto const birth_state_policy = is_positron
        ? RreaElectronStatePolicyDecision{
            kinetic_energy_eV < coupling.LowEnergyCutoffEv(), false}
        : electron_state_policy_at(p.x_m, p.y_m, p.z_m, kinetic_energy_eV);
    if (birth_state_policy.Demote()) {
        amrex::Real const r_birth = std::hypot(p.x_m, p.y_m);
        if (is_positron) {
            auto const packed = RreaTryPackTransientRngIndex(
                true, p.root_species, p.transient_lineage, 0U, 0U, 0U);
            if (!packed.valid) {
                rrea::KineticFail(
                    "RREA newborn positron birth RNG index overflow");
            }
            newborn_fx.BeginGroup(
                RreaSecondaryProcess::PositronAnnihilation, true);
            spawn_positron_annihilation(
                newborn_fx,
                p.x_m, p.y_m, p.z_m, r_birth, p.z_m,
                dir_x, dir_y, dir_z,
                p.weight, kinetic_energy_eV, true,
                p.root_rng_particle_id, packed.value,
                p.birth_elapsed_step_fraction);
            newborn_fx.tally.RecordPositronTransport(
                p.weight, kinetic_energy_eV, true);
        } else {
            newborn_fx.BeginGroup(
                RreaSecondaryProcess::LowEnergyDemotion, true);
            newborn_fx.AccumulateLowEnergyElectron(
                0, r_birth, p.z_m, p.weight, amrex::Real(0.0));
            deposit_local_energy_as_ion_pairs(
                newborn_fx, r_birth, p.z_m, p.weight, kinetic_energy_eV);
        }
        newborn_fx.SetInteractionConservationClosure(
            p.weight, kinetic_energy_eV, amrex::Real(0.0));
        newborn_fx.EndGroup();
        return false;
    }
    if (!(residual_fraction > amrex::Real(0.0))) {
        return true;
    }
    amrex::Real const birth_u_mag =
        rrea::RreaElectronProperVelocityFromKineticEv(kinetic_energy_eV);
    amrex::Real const birth_gamma = std::sqrt(
        amrex::Real(1.0)
        + birth_u_mag * birth_u_mag
            / (kSpeedOfLightMPerS * kSpeedOfLightMPerS));
    amrex::Real const residual_dt_s = dt_s * residual_fraction;
    amrex::Real const chord_m = (birth_u_mag / birth_gamma) * residual_dt_s;
    if (!(chord_m > amrex::Real(0.0))) {
        return kinetic_energy_eV >= coupling.LowEnergyCutoffEv();
    }
    amrex::Real const end_x = p.x_m + chord_m * dir_x;
    amrex::Real const end_y = p.y_m + chord_m * dir_y;
    amrex::Real const end_z = p.z_m + chord_m * dir_z;
    // Provisional whole-residual chord, used ONLY to size the substep count.
    // The transported chords are rebuilt per substep from the running
    // direction below, so this one is never integrated along.
    RreaTransportPathSegment const driver_path = ClipRzTransportPath(
        std::hypot(p.x_m, p.y_m), std::atan2(p.y_m, p.x_m), p.z_m,
        std::hypot(end_x, end_y), std::atan2(end_y, end_x), end_z,
        geom.ProbLo(0), geom.ProbHi(0),
        geom.ProbLo(1), geom.ProbHi(1),
        periodic_z);
    if (!(driver_path.full_length_m > amrex::Real(0.0))
        || !(driver_path.in_domain_length_m > amrex::Real(0.0))) {
        if (driver_path.exits_domain) {
            newborn_fx.BeginGroup(
                RreaSecondaryProcess::ParticleEscape, true);
            newborn_fx.SetChargedEscapeConservationClosure(
                p.weight, kinetic_energy_eV);
            newborn_fx.AccumulateChargedEscape(
                0, birth_r_m, birth_z_m,
                RreaChargedBoundaryCrossingForPath(driver_path),
                charge_sign * p.weight,
                kRreaSegmentChannelCreated | kRreaSegmentChannelEscaped);
            newborn_fx.EndGroup();
            return false;
        }
        return true;
    }
    auto const driver_intervals = build_material_path_intervals(
        coupling, driver_path, density_ratio);
    if (driver_intervals.empty()) {
        rrea::KineticFail(
            "RREA charged transient path has positive in-domain "
            "length but no material intervals");
    }
    // The chord actually transported along.  Declared here rather than in the
    // substep loop because the policies below close over it; the loop
    // reassigns it per microchord, which is what makes deflection bend the
    // subsequent path instead of only the momentum.  Seeded with the driver
    // chord so the post-loop final point is well defined even if the first
    // microchord degenerates.
    RreaTransportPathSegment path = driver_path;
    rrea::KineticVector<MaterialPathInterval> material_intervals = driver_intervals;
    auto const& newborn_certified_range = is_positron
        ? tables.PositronCertifiedEnergyRangeEv()
        : tables.ElectronCertifiedEnergyRangeEv();
    auto require_certified_energy = [&](char const* /*what*/) {
        if (!std::isfinite(static_cast<double>(kinetic_energy_eV))
            || !newborn_certified_range.Contains(kinetic_energy_eV)) {
            rrea::KineticFail(
                "RREA newborn charged left the certified transport-energy range; the "
                  "uncommitted transaction is rejected");
        }
    };
    // Substep drivers remain GS batching plus impulse/drag granularity.
    // Repeated discrete hazards are sampled on every remaining substep
    // segment and need no optical-depth cap.
    amrex::Real const driver_density_path_m = density_path_m(
        driver_intervals, driver_path,
        driver_path.in_domain_begin_fraction,
        driver_path.in_domain_end_fraction);
    amrex::Real const gs_n_total =
        elastic_inverse_length_stp(tables, is_positron, kinetic_energy_eV)
        * driver_density_path_m;
    amrex::Real const drag_estimate_eV =
        (is_positron
         ? tables.PositronCollisionLoss(kinetic_energy_eV, amrex::Real(1.0))
         : electron_realized_continuous_loss_ev_per_m(
             tables, kinetic_energy_eV, amrex::Real(1.0)))
        * driver_density_path_m;
    amrex::Real const whole_work_eV = path_field_work_and_average(
        coupling, driver_path, charge_sign,
        amrex::Real(0.0), amrex::Real(1.0)).work_eV;
    amrex::Real const impulse_scale_eV = rrea::kinetic::max(
        m_config.transport_max_field_impulse_fraction * kinetic_energy_eV,
        rrea::kinetic::max(
            m_config.transport_field_impulse_floor_eV, amrex::Real(1.0)));
    // The three budgets above are integrated over the driver chord's IN-DOMAIN
    // span, but the substep loop transports the whole residual regardless of
    // where that chord clipped: each microchord is speed*sub_dt long and
    // sub_dt*substeps == residual_dt always.  Without this rescale a driver
    // clipped to a fraction phi under-sizes every budget by phi, and the
    // realised per-substep impulse and GS batch overshoot their configured caps
    // by 1/phi -- exactly for the boundary population that dominates escape.
    amrex::Real const driver_span = rrea::kinetic::max(
        driver_path.in_domain_end_fraction, amrex::Real(1.0e-6));
    std::uint64_t const substeps = rrea::kinetic::min(
        rrea::kinetic::max(
            substep_driver_count(
                rrea::kinetic::min(
                    gs_n_total / driver_span,
                    m_config.transport_gs_total_collision_cap),
                rrea::kinetic::max(
                    m_config.transport_max_gs_collisions_per_substep,
                    amrex::Real(1.0))),
            substep_driver_count(
                (std::abs(whole_work_eV) + drag_estimate_eV) / driver_span,
                impulse_scale_eV)),
        m_config.max_subcycles);
    if (substeps >= kRreaTransientSegmentCapacity) {
        rrea::KineticFail(
            "RREA newborn charged substep count is outside its "
            "14-bit transient RNG budget");
    }
    amrex::Real positron_total_loss_eV = amrex::Real(0.0);
    auto apply_field_half_kick = [&](
        PathFieldSubRange const& field_range, amrex::Real half_dt_s) {
        static_cast<void>(RreaApplyChargedFieldHalfKick(
            field_range, half_dt_s, charge_sign,
            kinetic_energy_eV, dir_x, dir_y, dir_z));
    };
    // Same drag module as the live loops (RreaPlanContinuousDragSegment); the
    // newborn POLICY serves both species behind is_positron -- electron:
    // partition record with the hard-Moller mean rate; positron: soft record
    // plus the at-rest budget accumulator -- and the ion-pair sink clamps to
    // the half-open domain (a newborn chord may graze a face).
    auto apply_newborn_drag = [&](
        amrex::Real begin_fraction,
        amrex::Real end_fraction,
        amrex::Real stopping_energy_eV) -> RreaContinuousSegmentLimit {
        amrex::Real const kinetic_energy_before_segment_eV = kinetic_energy_eV;
        auto density_path = [&](amrex::Real b, amrex::Real e) {
            return density_path_m(material_intervals, path, b, e);
        };
        auto const plan = RreaPlanContinuousDragSegment(
            begin_fraction, end_fraction, kinetic_energy_eV,
            stopping_energy_eV, coupling.LowEnergyCutoffEv(),
            density_path,
            [&] {
                return RreaFrozenStoppingPowers{
                    RreaFrozenStpRate(is_positron
                        ? tables.PositronCollisionLoss(
                            stopping_energy_eV, amrex::Real(1.0))
                        : tables.ElectronCollisionLoss(
                            stopping_energy_eV, amrex::Real(1.0))),
                    RreaFrozenStpRate(is_positron
                        ? tables.PositronSoftRadiativeDrag(
                            stopping_energy_eV, amrex::Real(1.0))
                        : tables.ElectronSoftRadiativeDrag(
                            stopping_energy_eV, amrex::Real(1.0))),
                    (!is_positron && hard_moller_discrete_enabled
                     && RreaHardMollerChannelOpen(
                         stopping_energy_eV, hard_secondary_threshold_eV))
                    ? RreaFrozenStpRate(tables.ElectronHardMollerEnergyLoss(
                        stopping_energy_eV, amrex::Real(1.0)))
                    : amrex::Real(0.0)};
            });
        if (!plan.limit_valid) {
            rrea::KineticFail(
                "RREA newborn continuous stopping could not locate a "
                "finite monotone cutoff point");
        }
        if (!plan.applied) {
            return RreaContinuousSegmentLimit{begin_fraction, false, true};
        }
        newborn_fx.tally.RecordContinuousSoftBrems(
            p.weight, plan.soft_radiative_loss_eV);
        if (is_positron) {
            positron_total_loss_eV += plan.drag_loss_eV;
        } else {
            newborn_fx.tally.RecordContinuousLossPartition(
                p.weight,
                plan.drag_loss_eV,
                plan.collision_loss_eV,
                plan.hard_transfer_loss_eV);
        }
        {
            amrex::Real const diagnostic_loss_eV = tables.WAirEvPerIonPair();
            RreaPartitionContinuousIonPairs(
                material_intervals,
                path.full_length_m,
                begin_fraction,
                plan.effective_end_fraction,
                plan.collision_loss_eV,
                plan.raw_collision_loss_eV,
                p.weight,
                diagnostic_loss_eV,
                plan.collision_stp_eV_per_m,
                [&](amrex::Real midpoint_fraction,
                    amrex::Real pair_weight) {
                    amrex::Real source_x = amrex::Real(0.0);
                    amrex::Real source_y = amrex::Real(0.0);
                    amrex::Real source_z = amrex::Real(0.0);
                    path.PointAtFullPathFraction(
                        midpoint_fraction, source_x, source_y, source_z);
                    amrex::Real source_r = std::hypot(source_x, source_y);
                    amrex::Real source_wrapped_z = path.WrapZ(source_z);
                    if (!RreaRzPositionInHalfOpenDomain(
                            source_r, source_wrapped_z,
                            geom.ProbLo(0), geom.ProbHi(0),
                            geom.ProbLo(1), geom.ProbHi(1))) {
                        // Roundoff on a clipped face: fall back to the
                        // (validated in-domain) birth position rather than
                        // forming an out-of-domain source; the complete work
                        // graph is collectively validated before replay.
                        source_r = std::hypot(p.x_m, p.y_m);
                        source_wrapped_z = p.z_m;
                    }
                    newborn_fx.AccumulateIonizationEvent(
                        0, source_r, source_wrapped_z, pair_weight,
                        is_positron
                            ? amrex::Real(0.0)
                            : amrex::Real(0.5) * coupling.LowEnergyCutoffEv(),
                        diagnostic_loss_eV);
                    newborn_fx.tally.RecordTransportIonization(
                        pair_weight, diagnostic_loss_eV, true, false);
                });
        }
        if (!is_positron) {
            defer_cd_electron_plane_crossings(
                newborn_fx, coupling, tables, path, material_intervals,
                begin_fraction, plan.effective_end_fraction,
                p.weight, kinetic_energy_before_segment_eV,
                stopping_energy_eV, plan.drag_loss_eV,
                hard_moller_discrete_enabled);
        }
        kinetic_energy_eV = rrea::kinetic::max(
            amrex::Real(0.0), kinetic_energy_eV - plan.drag_loss_eV);
        if (plan.reaches_cutoff) {
            kinetic_energy_eV = coupling.LowEnergyCutoffEv();
        }
        return RreaContinuousSegmentLimit{
            plan.effective_end_fraction, plan.reaches_cutoff, true};
    };
    auto apply_newborn_gs = [&](
        amrex::Real begin_fraction,
        amrex::Real end_fraction,
        std::uint64_t substep_idx) {
        // Same GS application unit as the live loops; only the RNG identity
        // differs (Newborn* channels on the PARENT stream, packed transient
        // lineage instead of substep<<8).
        GsScatterChannels const channels{
            is_positron ? RreaRngChannel::NewbornPositronGsFewCount
                        : RreaRngChannel::NewbornElectronGsFewCount,
            is_positron ? RreaRngChannel::NewbornPositronGsFewScatter
                        : RreaRngChannel::NewbornElectronGsFewScatter,
            is_positron ? RreaRngChannel::NewbornPositronGsFewAzimuth
                        : RreaRngChannel::NewbornElectronGsFewAzimuth,
            is_positron ? RreaRngChannel::NewbornPositronGsAngle
                        : RreaRngChannel::NewbornElectronGsAngle,
            is_positron ? RreaRngChannel::NewbornPositronGsAzimuth
                        : RreaRngChannel::NewbornElectronGsAzimuth};
        auto gs_key = [&](RreaRngChannel ch, std::uint64_t inner) {
            auto const packed = RreaTryPackTransientRngIndex(
                true, p.root_species, p.transient_lineage,
                substep_idx, 0U, inner);
            if (!packed.valid) {
                rrea::KineticFail(
                    "RREA newborn GS RNG index overflow; the "
                    "uncommitted transaction is rejected");
            }
            return RreaRngKey{
                coupling.RngSeed(),
                p.root_rng_particle_id,
                static_cast<std::uint64_t>(rrea::kinetic::max(step, 0)),
                packed.value,
                static_cast<std::uint64_t>(ch)};
        };
        apply_gs_elastic_scatter(
            material_intervals, path, begin_fraction, end_fraction, tables,
            is_positron, channels, gs_key,
            kinetic_energy_eV, dir_x, dir_y, dir_z);
    };
    // Each substep rebuilds its chord from the current direction so elastic
    // deflection, recoil and field work follow the same path.
    amrex::Real run_x = p.x_m;
    amrex::Real run_y = p.y_m;
    amrex::Real run_z = p.z_m;
    amrex::Real const sub_dt_s =
        residual_dt_s / static_cast<amrex::Real>(substeps);
    amrex::Real final_fraction = amrex::Real(0.0);
    bool terminal_interaction = false;
    bool terminal_at_path_point = false;
    bool exited_domain = false;
    std::uint64_t completed_substeps = 0;
    auto elapsed_fraction_at = [&](amrex::Real path_fraction) {
        return RreaNewbornElapsedStepFraction(
            p.birth_elapsed_step_fraction, residual_fraction,
            completed_substeps, substeps, path_fraction);
    };
    auto mark_newborn_stopped_at = [&](amrex::Real path_fraction) {
        final_fraction = path_fraction;
        terminal_at_path_point = true;
    };
    PathFieldSubRange substep_field{};
    amrex::Real substep_dt_s = amrex::Real(0.0);
    RreaRunChargedSubstepInterleave(
        substeps,
        kinetic_energy_eV,
        coupling.LowEnergyCutoffEv(),
        completed_substeps,
        exited_domain,
        [&](std::uint64_t,
            amrex::Real& subcycle_begin,
            amrex::Real& subcycle_end) {
            // The chord the particle will actually follow over this substep,
            // from the state it is actually in.  The header primitive owns the
            // speed law, the direction normalisation and the endpoint clip, so
            // the three species cannot drift apart on any of them.
            auto const chord = RreaBuildRunningChargedChord(
                RreaRunningChargedState{
                    run_x, run_y, run_z, dir_x, dir_y, dir_z,
                    kinetic_energy_eV},
                sub_dt_s,
                residual_dt_s,
                geom.ProbLo(0), geom.ProbHi(0),
                geom.ProbLo(1), geom.ProbHi(1),
                periodic_z);
            if (!chord.valid) {
                return false;
            }
            path = chord.path;
            if (!(path.full_length_m > amrex::Real(0.0))
                || !(path.in_domain_length_m > amrex::Real(0.0))) {
                exited_domain = path.exits_domain;
                // Resolve a degenerate chord at its origin.
                final_fraction = amrex::Real(0.0);
                return false;
            }
            material_intervals = build_material_path_intervals(
                coupling, path, density_ratio);
            if (material_intervals.empty()) {
                rrea::KineticFail(
                    "RREA charged transient microchord has positive "
                    "in-domain length but no material intervals");
            }
            subcycle_begin = path.in_domain_begin_fraction;
            subcycle_end = path.in_domain_end_fraction;
            final_fraction = subcycle_end;
            substep_field = path_field_work_and_average(
                coupling, path, charge_sign, subcycle_begin, subcycle_end);
            // Time ACTUALLY spent on this chord.  The field work is already
            // integrated only over [begin, end], but the half-kick's ROTATION
            // scales with dt, so handing it the whole substep would over-rotate
            // a clipped microchord by 1/(end-begin) -- energy right, direction
            // wrong, which no conservation ledger would catch.
            substep_dt_s = sub_dt_s * (subcycle_end - subcycle_begin);
            apply_field_half_kick(
                substep_field, amrex::Real(0.5) * substep_dt_s);
            if (kinetic_energy_eV < coupling.LowEnergyCutoffEv()) {
                final_fraction = subcycle_begin;
                terminal_at_path_point = true;
                return false;
            }
            require_certified_energy("field impulse");
            return true;
        },
        [&](std::uint64_t subcycle,
            amrex::Real subcycle_begin,
            amrex::Real subcycle_end) {
            // Event-loop policies: the newborn's physics content.  The
            // substep_interleaved_v3 sequencing -- refreeze, exact-depth drag,
            // explicit null event on a drag-closed channel -- lives in
            // RreaRunChargedEventInterleave, exactly as for the live loops.
            amrex::Real event_x = amrex::Real(0.0);
            amrex::Real event_y = amrex::Real(0.0);
            amrex::Real event_z = amrex::Real(0.0);
            amrex::Real event_r = amrex::Real(0.0);
            amrex::Real event_elapsed = amrex::Real(1.0);
            std::uint64_t packed_rng_index = 0;
            amrex::Real total_stp = amrex::Real(0.0);
            RreaRunChargedEventInterleave(
                subcycle_begin,
                subcycle_end,
                kinetic_energy_eV,
                coupling.LowEnergyCutoffEv(),
                is_positron ? 3 : 2,
                kRreaTransientEventCapacity,
                is_positron ? "newborn positron" : "newborn electron",
                [&](amrex::Real frozen_energy_eV,
                    amrex::Real tau_begin_fraction,
                    amrex::Real tau_end_fraction) {
                    // The newborn residual pass runs the SAME channels through
                    // the SAME gated rate bundle as the live loops, so a
                    // newborn cannot acquire physics its parent did not have.
                    auto const stp = RreaChargedCompetingStpRates(
                        tables, is_positron, hard_secondary_threshold_eV,
                        coupling.PhotonCutoffEv(), frozen_energy_eV,
                        hard_moller_discrete_enabled);
                    total_stp = stp[0] + stp[1] + stp[2];
                    amrex::Real const segment_density_path_m = density_path_m(
                        material_intervals, path,
                        tau_begin_fraction, tau_end_fraction);
                    std::array<amrex::Real, 3> const taus{
                        stp[0] * segment_density_path_m,
                        stp[1] * segment_density_path_m,
                        stp[2] * segment_density_path_m};
                    if (!is_positron) {
                        amrex::Real const total_tau = taus[0] + taus[1];
                        newborn_fx.tally.RecordHardMollerExpectation(
                            p.weight
                            * (total_tau > amrex::Real(0.0)
                               ? probability_from_optical_depth(total_tau)
                                   * taus[0] / total_tau
                               : amrex::Real(0.0)));
                    }
                    return taus;
                },
                [&](std::uint64_t draw_event_ordinal) {
                    auto const packed = RreaTryPackTransientRngIndex(
                        true, p.root_species, p.transient_lineage,
                        subcycle, draw_event_ordinal, 0U);
                    if (!packed.valid) {
                        rrea::KineticFail(
                            "RREA newborn charged repeated-hazard RNG index "
                            "overflow; the uncommitted transaction is rejected");
                    }
                    packed_rng_index = packed.value;
                    return std::pair<amrex::Real, amrex::Real>{
                        RreaRng::Uniform01(RreaRngKey{
                            coupling.RngSeed(), p.root_rng_particle_id,
                            static_cast<std::uint64_t>(rrea::kinetic::max(step, 0)),
                            packed_rng_index,
                            static_cast<std::uint64_t>(
                                RreaRngChannel::NewbornCompetingOpticalDepth)}),
                        RreaRng::Uniform01(RreaRngKey{
                            coupling.RngSeed(), p.root_rng_particle_id,
                            static_cast<std::uint64_t>(rrea::kinetic::max(step, 0)),
                            packed_rng_index,
                            static_cast<std::uint64_t>(
                                RreaRngChannel::NewbornCompetingChannel)})};
                },
                [&](amrex::Real sampled_optical_depth,
                    amrex::Real invert_begin_fraction,
                    amrex::Real invert_end_fraction) {
                    return fraction_at_density_path_m(
                        material_intervals, path,
                        invert_begin_fraction, invert_end_fraction,
                        sampled_optical_depth / total_stp);
                },
                apply_newborn_drag,
                mark_newborn_stopped_at,
                [&](amrex::Real point_event_fraction) {
                    point_and_radius_at_fraction(
                        path, point_event_fraction,
                        event_x, event_y, event_z, event_r);
                    event_z = path.WrapZ(event_z);
                    event_elapsed = elapsed_fraction_at(point_event_fraction);
                },
                [&](int channel, amrex::Real energy_eV) {
                    return RreaChargedChannelOpenAt(
                        tables, is_positron, hard_secondary_threshold_eV,
                        coupling.PhotonCutoffEv(), channel, energy_eV,
                        hard_moller_discrete_enabled);
                },
                [&](int channel, amrex::Real event_fraction) {
                    RreaInteractionSite const site{
                        event_x, event_y, event_z, event_r, event_elapsed};
                    auto rng_key = [&](RreaRngChannel rng_channel) {
                        return RreaRngKey{
                            coupling.RngSeed(), p.root_rng_particle_id,
                            static_cast<std::uint64_t>(rrea::kinetic::max(step, 0)),
                            packed_rng_index,
                            static_cast<std::uint64_t>(rng_channel)};
                    };
                    if (channel == 0) {
                        if (is_positron) {
                            RreaApplyBhabhaFinalState(
                                newborn_fx, tables, rng_key, site, p.weight,
                                hard_secondary_threshold_eV,
                                kinetic_energy_eV, dir_x, dir_y, dir_z);
                        } else {
                            RreaApplyHardMollerFinalState(
                                newborn_fx, tables, rng_key, site, p.weight,
                                hard_secondary_threshold_eV,
                                kinetic_energy_eV, dir_x, dir_y, dir_z);
                        }
                    } else if (channel == 1) {
                        RreaApplyBremsFinalState(
                            newborn_fx, tables, rng_key,
                            [&](amrex::Real local_energy_eV) {
                                deposit_local_energy_as_ion_pairs(
                                    newborn_fx, event_r, event_z,
                                    p.weight, local_energy_eV);
                            },
                            site, p.weight, charge_sign,
                            coupling.PhotonCutoffEv(),
                            kinetic_energy_eV, dir_x, dir_y, dir_z);
                    } else if (channel == 2) {
                        newborn_fx.BeginGroup(
                            RreaSecondaryProcess::PositronAnnihilation, true);
                        spawn_positron_annihilation(
                            newborn_fx,
                            event_x, event_y, event_z, event_r, event_z,
                            dir_x, dir_y, dir_z,
                            p.weight, kinetic_energy_eV, false,
                            p.root_rng_particle_id, packed_rng_index,
                            event_elapsed);
                        newborn_fx.SetInteractionConservationClosure(
                            p.weight, kinetic_energy_eV, amrex::Real(0.0));
                        // Newborn residual leg: birth to the sampled in-flight
                        // annihilation point.
                        newborn_fx.AccumulateChargedSegment(
                            0, birth_r_m, birth_z_m, event_r, event_z,
                            charge_sign * p.weight,
                            kRreaSegmentChannelCreated
                                | kRreaSegmentChannelAnnihilated);
                        newborn_fx.EndGroup();
                        terminal_interaction = true;
                        final_fraction = event_fraction;
                        return false;
                    }
                    return true;
                });
        },
        [&] { return terminal_interaction || terminal_at_path_point; },
        [&](std::uint64_t subcycle,
            amrex::Real subcycle_begin,
            amrex::Real subcycle_end) {
            if (kinetic_energy_eV >= coupling.LowEnergyCutoffEv()) {
                apply_newborn_gs(subcycle_begin, subcycle_end, subcycle);
            }
            apply_field_half_kick(
                substep_field, amrex::Real(0.5) * substep_dt_s);
            if (kinetic_energy_eV < coupling.LowEnergyCutoffEv()) {
                final_fraction = subcycle_end;
                terminal_at_path_point = true;
                return true;
            }
            require_certified_energy("trailing field impulse");
            return false;
        },
        // Commit this microchord: the next one is rebuilt from here, along
        // whatever direction the scatter above left behind.
        [&](amrex::Real subcycle_end) {
            return RreaAdvanceRunningChargedPosition(
                path, subcycle_end,
                geom.ProbLo(0), geom.ProbHi(0),
                geom.ProbLo(1), geom.ProbHi(1),
                run_x, run_y, run_z);
        },
        [&] { return path.exits_domain; });
    if (is_positron && positron_total_loss_eV > amrex::Real(0.0)) {
        newborn_fx.tally.RecordPositronTransport(
            p.weight, positron_total_loss_eV, false);
    }
    if (terminal_interaction) {
        return false;
    }
    amrex::Real final_x = amrex::Real(0.0);
    amrex::Real final_y = amrex::Real(0.0);
    amrex::Real final_z = amrex::Real(0.0);
    if (!NewbornFinalPoint(path, final_fraction, final_x, final_y, final_z)) {
        rrea::KineticFail(
            "RREA newborn charged transport could not place its final "
            "point in the half-open domain; the uncommitted "
            "transaction is rejected");
    }
    amrex::Real const final_r = std::hypot(final_x, final_y);
    amrex::Real const final_elapsed = elapsed_fraction_at(final_fraction);
    auto const final_state_policy = is_positron
        ? RreaElectronStatePolicyDecision{
            kinetic_energy_eV < coupling.LowEnergyCutoffEv(), false}
        : electron_state_policy_at(
            final_x, final_y, final_z, kinetic_energy_eV);
    if (terminal_at_path_point || final_state_policy.Demote()) {
        std::uint64_t channels = kRreaSegmentChannelCreated;
        if (is_positron) {
            auto const packed = RreaTryPackTransientRngIndex(
                true, p.root_species, p.transient_lineage,
                rrea::kinetic::min(substeps, kRreaTransientSegmentCapacity - 1U),
                0U, 0U);
            if (!packed.valid) {
                rrea::KineticFail(
                    "RREA newborn positron terminal RNG index overflow");
            }
            newborn_fx.BeginGroup(
                RreaSecondaryProcess::PositronAnnihilation, true);
            spawn_positron_annihilation(
                newborn_fx,
                final_x, final_y, final_z, final_r, final_z,
                dir_x, dir_y, dir_z,
                p.weight, kinetic_energy_eV, true,
                p.root_rng_particle_id, packed.value,
                final_elapsed);
            newborn_fx.tally.RecordPositronTransport(
                p.weight, kinetic_energy_eV, true);
            channels |= kRreaSegmentChannelAnnihilated;
        } else {
            newborn_fx.BeginGroup(
                RreaSecondaryProcess::LowEnergyDemotion, true);
            newborn_fx.AccumulateLowEnergyElectron(
                0, final_r, final_z, p.weight, amrex::Real(0.0));
            deposit_local_energy_as_ion_pairs(
                newborn_fx, final_r, final_z, p.weight, kinetic_energy_eV);
            channels |= kRreaSegmentChannelDemoted;
        }
        newborn_fx.SetInteractionConservationClosure(
            p.weight, kinetic_energy_eV, amrex::Real(0.0));
        newborn_fx.AccumulateChargedSegment(
            0, birth_r_m, birth_z_m, final_r, final_z,
            charge_sign * p.weight, channels);
        newborn_fx.EndGroup();
        return false;
    }
    // Any microchord that left the domain ends the residual advance; the flag
    // is sticky because only the last chord is still in scope.
    if (exited_domain) {
        newborn_fx.BeginGroup(RreaSecondaryProcess::ParticleEscape, true);
        newborn_fx.SetChargedEscapeConservationClosure(
            p.weight, kinetic_energy_eV);
        newborn_fx.AccumulateChargedEscape(
            0, birth_r_m, birth_z_m,
            RreaChargedBoundaryCrossingForPath(path),
            charge_sign * p.weight,
            kRreaSegmentChannelCreated | kRreaSegmentChannelEscaped);
        newborn_fx.EndGroup();
        return false;
    }
    p.x_m = final_x;
    p.y_m = final_y;
    p.z_m = final_z;
    p.dir_x = dir_x;
    p.dir_y = dir_y;
    p.dir_z = dir_z;
    p.kinetic_or_photon_energy_eV = kinetic_energy_eV;
    p.birth_elapsed_step_fraction = amrex::Real(1.0);
    // Surviving newborn: birth to the committed final point of its residual
    // leg (charge entered the kinetic population mid-step).
    newborn_fx.AccumulateChargedSegment(
        0, birth_r_m, birth_z_m, final_r, final_z,
        charge_sign * p.weight, kRreaSegmentChannelCreated);
    return true;
}

template<class Effects>
RREA_HOST_DEVICE bool AdvancePhotonNewborn(
    RreaSecondaryParticle& p,
    Effects& newborn_fx) const
{
    newborn_fx.SetParentContext(
        p.root_rng_particle_id, RreaSecondarySpecies::Photon,
        /*transient=*/true);
    amrex::Real const residual_fraction =
        RreaNewbornResidualFraction(p.birth_elapsed_step_fraction);
    if (!(residual_fraction > amrex::Real(0.0))) {
        return p.kinetic_or_photon_energy_eV >= coupling.PhotonCutoffEv();
    }
    if (!RreaCartesianPositionInHalfOpenRzDomain(
            p.x_m, p.y_m, p.z_m,
            geom.ProbLo(0), geom.ProbHi(0),
            geom.ProbLo(1), geom.ProbHi(1))) {
        rrea::KineticFail(
            "RREA photon transient was born outside the half-open "
            "domain; the uncommitted transaction is rejected");
    }
    static constexpr amrex::Real c_m_per_s = kRreaSpeedOfLightMPerS;
    amrex::Real dir_x = p.dir_x;
    amrex::Real dir_y = p.dir_y;
    amrex::Real dir_z = p.dir_z;
    RreaNormalizeDirection(dir_x, dir_y, dir_z);
    amrex::Real photon_energy_eV = p.kinetic_or_photon_energy_eV;
    if (photon_energy_eV < coupling.PhotonCutoffEv()) {
        newborn_fx.BeginGroup(
            RreaSecondaryProcess::LowEnergyDemotion, true);
        deposit_local_energy_as_ion_pairs(
            newborn_fx, std::hypot(p.x_m, p.y_m), p.z_m,
            p.weight, photon_energy_eV);
        newborn_fx.SetInteractionConservationClosure(
            p.weight, photon_energy_eV, amrex::Real(0.0));
        newborn_fx.EndGroup();
        return false;
    }
    auto commit_photon_escape = [&]() {
        newborn_fx.BeginGroup(RreaSecondaryProcess::PhotonEscape, true);
        newborn_fx.tally.RecordPhotonEscape(p.weight, photon_energy_eV);
        newborn_fx.AddEscapedConservationEnergy(p.weight, photon_energy_eV);
        newborn_fx.SetInteractionConservationClosure(
            p.weight, photon_energy_eV, amrex::Real(0.0));
        newborn_fx.EndGroup();
    };
    amrex::Real const chord_m = c_m_per_s * dt_s * residual_fraction;
    amrex::Real const end_x = p.x_m + chord_m * dir_x;
    amrex::Real const end_y = p.y_m + chord_m * dir_y;
    amrex::Real const end_z = p.z_m + chord_m * dir_z;
    RreaTransportPathSegment path = ClipRzTransportPath(
        std::hypot(p.x_m, p.y_m), std::atan2(p.y_m, p.x_m), p.z_m,
        std::hypot(end_x, end_y), std::atan2(end_y, end_x), end_z,
        geom.ProbLo(0), geom.ProbHi(0),
        geom.ProbLo(1), geom.ProbHi(1),
        periodic_z);
    if (!(path.full_length_m > amrex::Real(0.0))
        || !(path.in_domain_length_m > amrex::Real(0.0))) {
        if (path.exits_domain) {
            commit_photon_escape();
            return false;
        }
        return true;
    }
    amrex::Real elapsed_path_start = p.birth_elapsed_step_fraction;
    std::uint64_t event_ordinal = 0;
    while (true) {
        auto const packed = RreaTryPackTransientRngIndex(
            true, p.root_species, p.transient_lineage,
            0U, event_ordinal, 0U);
        if (!packed.valid) {
            rrea::KineticFail(
                "RREA newborn photon repeated-hazard RNG index "
                "overflow; the uncommitted transaction is rejected");
        }
        // Same material path -- and the same piecewise density-path inversion,
        // whose interaction point is misplaced by at most h^2/8H = 9 um on a
        // 0.75 m chord -- as the live photon loop.
        auto const photon_intervals = build_material_path_intervals(
            coupling, path, density_ratio);
        amrex::Real const mfp = tables.PhotonFeedbackMeanFreePath(
            photon_energy_eV, amrex::Real(1.0));
        amrex::Real const tau = density_path_m(
            photon_intervals, path,
            path.in_domain_begin_fraction,
            path.in_domain_end_fraction) / mfp;
        RreaRemainingSegmentHazard const hazard =
            RreaResolveRemainingSegmentHazard(
                {tau, amrex::Real(0.0), amrex::Real(0.0)},
                1,
                path.in_domain_begin_fraction,
                path.in_domain_end_fraction,
                RreaRng::Uniform01(RreaRngKey{
                    coupling.RngSeed(), p.root_rng_particle_id,
                    static_cast<std::uint64_t>(rrea::kinetic::max(step, 0)),
                    packed.value,
                    static_cast<std::uint64_t>(
                        RreaRngChannel::NewbornCompetingOpticalDepth)}),
                amrex::Real(0.0),
                [&](amrex::Real sampled_optical_depth) {
                    return fraction_at_density_path_m(
                        photon_intervals, path,
                        path.in_domain_begin_fraction,
                        path.in_domain_end_fraction,
                        sampled_optical_depth * mfp);
                });
        if (!hazard.selection.event) {
            amrex::Real final_x = amrex::Real(0.0);
            amrex::Real final_y = amrex::Real(0.0);
            amrex::Real final_z = amrex::Real(0.0);
            if (!NewbornFinalPoint(
                    path, path.in_domain_end_fraction,
                    final_x, final_y, final_z)) {
                rrea::KineticFail(
                    "RREA newborn photon could not place its final "
                    "point; the uncommitted transaction is rejected");
            }
            if (path.exits_domain) {
                commit_photon_escape();
                return false;
            }
            p.x_m = final_x; p.y_m = final_y; p.z_m = final_z;
            p.dir_x = dir_x; p.dir_y = dir_y; p.dir_z = dir_z;
            p.kinetic_or_photon_energy_eV = photon_energy_eV;
            p.birth_elapsed_step_fraction = amrex::Real(1.0);
            return true;
        }
        amrex::Real const event_fraction = hazard.event_fraction;
        amrex::Real const event_distance = path.full_length_m
            * (event_fraction - path.in_domain_begin_fraction);
        amrex::Real const remaining_m = rrea::kinetic::max(
            amrex::Real(0.0), path.full_length_m - event_distance);
        amrex::Real event_x = amrex::Real(0.0);
        amrex::Real event_y = amrex::Real(0.0);
        amrex::Real event_z = amrex::Real(0.0);
        amrex::Real event_r = amrex::Real(0.0);
        point_and_radius_at_fraction(
            path, event_fraction, event_x, event_y, event_z, event_r);
        event_z = path.WrapZ(event_z);
        amrex::Real const event_elapsed = rrea::kinetic::min(
            amrex::Real(1.0),
            elapsed_path_start
                + event_distance
                    / rrea::kinetic::max(
                        c_m_per_s * dt_s,
                        std::numeric_limits<amrex::Real>::min()));
        RreaInteractionSite const site{
            event_x, event_y, event_z, event_r, event_elapsed};
        auto rng_key = [&](RreaRngChannel rng_channel) {
            return RreaRngKey{
                coupling.RngSeed(), p.root_rng_particle_id,
                static_cast<std::uint64_t>(rrea::kinetic::max(step, 0)),
                packed.value,
                static_cast<std::uint64_t>(rng_channel)};
        };
        auto deposit_local = [&](amrex::Real local_energy_eV) {
            deposit_local_energy_as_ion_pairs(
                newborn_fx, event_r, event_z, p.weight, local_energy_eV);
        };
        RreaPhotonChannel const channel = tables.SamplePhotonChannel(
            photon_energy_eV,
            RreaRng::Uniform01(rng_key(
                RreaRngChannel::NewbornCompetingChannel)));
        if (channel == RreaPhotonChannel::Compton) {
            if (!RreaApplyComptonFinalState(
                    newborn_fx, tables, rng_key, deposit_local, site,
                    p.weight, coupling.LowEnergyCutoffEv(),
                    coupling.PhotonCutoffEv(),
                    photon_energy_eV, dir_x, dir_y, dir_z)) {
                return false;
            }
            if (!(remaining_m > amrex::Real(0.0))) {
                p.x_m = event_x; p.y_m = event_y; p.z_m = event_z;
                p.dir_x = dir_x; p.dir_y = dir_y; p.dir_z = dir_z;
                p.kinetic_or_photon_energy_eV = photon_energy_eV;
                p.birth_elapsed_step_fraction = amrex::Real(1.0);
                return true;
            }
            amrex::Real const target_x = event_x + remaining_m * dir_x;
            amrex::Real const target_y = event_y + remaining_m * dir_y;
            amrex::Real const target_z = event_z + remaining_m * dir_z;
            path = ClipRzTransportPath(
                event_r, std::atan2(event_y, event_x), event_z,
                std::hypot(target_x, target_y),
                std::atan2(target_y, target_x), target_z,
                geom.ProbLo(0), geom.ProbHi(0),
                geom.ProbLo(1), geom.ProbHi(1), periodic_z);
            elapsed_path_start = event_elapsed;
            ++event_ordinal;
            continue;
        }
        if (channel == RreaPhotonChannel::Photoelectric) {
            RreaApplyPhotoelectricFinalState(
                newborn_fx, tables, rng_key, deposit_local, site,
                p.weight, coupling.LowEnergyCutoffEv(),
                photon_energy_eV, dir_x, dir_y, dir_z);
            return false;
        }
        RreaApplyPairFinalState(
            newborn_fx, tables, rng_key, deposit_local,
            [&](amrex::Real positron_dir_x,
                amrex::Real positron_dir_y,
                amrex::Real positron_dir_z,
                amrex::Real positron_kinetic_eV) {
                spawn_positron_annihilation(
                    newborn_fx,
                    event_x, event_y, event_z, event_r, event_z,
                    positron_dir_x, positron_dir_y, positron_dir_z,
                    p.weight, positron_kinetic_eV, true,
                    p.root_rng_particle_id, packed.value,
                    event_elapsed);
            },
            site, p.weight, coupling.LowEnergyCutoffEv(),
            photon_energy_eV, channel, dir_x, dir_y, dir_z);
        return false;
    }
}

    template<class Effects,class Id>
    RREA_HOST_DEVICE void Electron(Effects& fx,ChargedSoAView const& soa,long ip,Id particle_id,int particle_cpu) const {
        do {

                if (!particle_id.is_valid()) {
                    continue;
                }
                auto setup = BeginRunningChargedParticle(
                    fx, soa, ip,
                    static_cast<amrex::Long>(particle_id),
                    particle_cpu,
                    RreaSecondarySpecies::Electron,
                    /*charge_sign=*/amrex::Real(-1.0),
                    "electron");
                if (setup.skip) {
                    particle_id.make_invalid();
                    fx.invalidated = true;
                    continue;
                }
                std::uint64_t const transaction_parent_key =
                    setup.transaction_parent_key;
                amrex::Real const particle_weight = setup.particle_weight;
                amrex::Real dir_x = setup.dir_x;
                amrex::Real dir_y = setup.dir_y;
                amrex::Real dir_z = setup.dir_z;
                amrex::Real kinetic_energy_eV = setup.kinetic_energy_eV;
                amrex::Real run_x = setup.run_x;
                amrex::Real run_y = setup.run_y;
                amrex::Real run_z = setup.run_z;
                amrex::Real const z_particle = setup.z_particle;
                amrex::Real const r_particle = setup.r_particle;
                RreaTransportPathSegment const driver_path = setup.driver_path;
                RreaEndpointEscapePolicy charged_parent_escape =
                    setup.charged_parent_escape;
                amrex::Real const ds_m = setup.ds_m;
                auto const& driver_intervals = setup.driver_intervals;
                amrex::Real const chord_density_path_m = density_path_m(
                    driver_intervals,
                    driver_path,
                    driver_path.in_domain_begin_fraction,
                    driver_path.in_domain_end_fraction);
                // Updated after the loop from the density actually traversed.
                // Every demoted electron's thermalized energy and ion
                // pairs are the avalanche's dominant sink; misplacing them
                // coherently displaces the conductivity field.
                amrex::Real particle_density_ratio =
                    ds_m > amrex::Real(0.0)
                    ? chord_density_path_m / ds_m
                    : positive_or(coupling.TransportDensityRatioAtZ(z_particle), density_ratio);
                amrex::Real transported_density_path_m = amrex::Real(0.0);
                amrex::Real transported_length_m = amrex::Real(0.0);
                amrex::Real const driver_span = setup.driver_span;
                amrex::Real const field_work_eV = setup.field_work_eV;
                // Rebuilt with each running-state chord; intervals from another
                // chord have an empty material-rate intersection.
                RreaTransportPathSegment transport_path = driver_path;
                rrea::KineticVector<MaterialPathInterval> material_intervals =
                    driver_intervals;
                auto commit_charged_escape = [&]() {
                    CommitChargedEscape(
                        fx, soa, ip,
                        particle_weight, kinetic_energy_eV,
                        dir_x, dir_y, dir_z,
                        r_particle, z_particle,
                        transport_path, charged_parent_escape,
                        /*signed_mesh_weight=*/-particle_weight);
                    if (charged_parent_escape.invalidate_parent_in_rrea) {
                        particle_id.make_invalid();
                        fx.invalidated = true;
                    }
                };

                auto const initial_terminal_disposition =
                    RreaResolveChargedTerminalDisposition(
                        /*positron=*/false,
                        charged_parent_escape.escapes,
                        kinetic_energy_eV < coupling.LowEnergyCutoffEv(),
                        /*electron_directional_demotion=*/false);
                if (initial_terminal_disposition
                    == RreaChargedTerminalDisposition::Escape
                    && kinetic_energy_eV < coupling.LowEnergyCutoffEv()) {
                    // The clipped endpoint is on an excluded absorbing face.
                    // Do not create a fluid source there, and do not query a
                    // schema-6 charged table below its certified 1 keV floor.
                    commit_charged_escape();
                    continue;
                }
                if (initial_terminal_disposition
                    == RreaChargedTerminalDisposition::ElectronDemotion) {
                    // The demoted electron itself
                    // joins the low-energy mesh (energy 0 here so the mesh
                    // energy diagnostic is not double-counted), and its
                    // residual kinetic energy thermalizes into W_air ion
                    // pairs like every other local deposit.
                    fx.BeginGroup(RreaSecondaryProcess::LowEnergyDemotion, true);
                    fx.AccumulateLowEnergyElectron(
                        0,
                        r_particle,
                        z_particle,
                        particle_weight,
                        amrex::Real(0.0));
                    deposit_local_energy_as_ion_pairs(
                        fx,
                        r_particle,
                        z_particle,
                        particle_weight,
                        kinetic_energy_eV);
                    fx.SetInteractionConservationClosure(
                        particle_weight, kinetic_energy_eV, amrex::Real(0.0));
                    // Zero-length net chord: the charge leaves the kinetic
                    // population at the captured start point.
                    fx.AccumulateChargedSegment(
                        0, r_particle, z_particle, r_particle, z_particle,
                        -particle_weight, kRreaSegmentChannelDemoted);
                    fx.EndGroup();
                    particle_id.make_invalid();
                    fx.invalidated = true;
                    continue;
                }

                if (ds_m <= amrex::Real(0.0)) {
                    if (charged_parent_escape.escapes) {
                        commit_charged_escape();
                    }
                    continue;
                }
                if (driver_intervals.empty()) {
                    rrea::KineticFail(
                        "RREA charged transport path has positive length but no material segments");
                }
                amrex::Real const hard_moller_threshold_eV =
                    coupling.HardMollerSecondaryThresholdEv();
                bool const hard_moller_discrete_enabled =
                    !m_config.debug_disable_hard_moller_secondaries;
                auto const initial_stp = RreaChargedCompetingStpRates(
                    tables, /*is_positron=*/false, hard_moller_threshold_eV,
                    coupling.PhotonCutoffEv(), kinetic_energy_eV,
                    hard_moller_discrete_enabled);
                amrex::Real const initial_hard_stp = initial_stp[0];
                amrex::Real const initial_brems_stp = initial_stp[1];
                // Driver estimates: integrated over the driver chord's clipped
                // in-domain span, then divided by that span because the loop
                // transports the whole dt.
                amrex::Real const driver_density_path_m =
                    chord_density_path_m / driver_span;
                amrex::Real const initial_hard_tau =
                    initial_hard_stp * driver_density_path_m;
                amrex::Real const initial_brems_tau =
                    initial_brems_stp * driver_density_path_m;
                bool optical_depth_cap_violation = false;
                std::uint64_t subcycles = selected_material_subcycles(
                    initial_hard_tau + initial_brems_tau,
                    driver_intervals,
                    driver_path,
                    m_config,
                    optical_depth_cap_violation,
                    initial_hard_stp + initial_brems_stp);
                {
                    // Two additional deterministic whole-chord
                    // substep drivers beyond the hazard-tau refinement -- GS
                    // collision batching and the field-impulse/drag energy
                    // granularity.  Driver counts clamp silently at the cap:
                    // unlike an optical-depth violation this is benign
                    // refinement and does not weaken that strict guard.
                    amrex::Real const initial_gs_n_total =
                        elastic_inverse_length_stp(
                            tables, false, kinetic_energy_eV)
                        * driver_density_path_m;
                    amrex::Real const drag_estimate_eV =
                        electron_realized_continuous_loss_ev_per_m(
                            tables,
                            rrea::kinetic::max(
                                kinetic_energy_eV, coupling.LowEnergyCutoffEv()),
                            amrex::Real(1.0))
                        * driver_density_path_m;
                    amrex::Real const impulse_scale_eV = rrea::kinetic::max(
                        m_config.transport_max_field_impulse_fraction
                            * kinetic_energy_eV,
                        rrea::kinetic::max(
                            m_config.transport_field_impulse_floor_eV,
                            amrex::Real(1.0)));
                    std::uint64_t const gs_driver = substep_driver_count(
                        rrea::kinetic::min(
                            initial_gs_n_total,
                            m_config.transport_gs_total_collision_cap),
                        rrea::kinetic::max(
                            m_config.transport_max_gs_collisions_per_substep,
                            amrex::Real(1.0)));
                    std::uint64_t const impulse_driver = substep_driver_count(
                        std::abs(field_work_eV) + drag_estimate_eV,
                        impulse_scale_eV);
                    subcycles = std::min<std::uint64_t>(
                        rrea::kinetic::max({subcycles, gs_driver, impulse_driver}),
                        m_config.max_subcycles);
                }
                // Driver estimates as well: uniform slices of a chord the loop no
                // longer transports verbatim.  Reported, not enforced -- and
                // divided by the clipped span like every enforced budget above,
                // so the reported per-substep tau describes the full-dt
                // microchords the loop actually flies rather than the driver's
                // clipped fraction of them.
                amrex::Real const max_slice_density_path_m =
                    max_uniform_subcycle_density_path_m(
                        driver_intervals, driver_path, subcycles) / driver_span;
                amrex::Real const max_hard_tau =
                    initial_hard_stp * max_slice_density_path_m;
                amrex::Real const max_brems_tau =
                    initial_brems_stp * max_slice_density_path_m;
                amrex::Real const max_electron_total_competing_tau =
                    (initial_hard_stp + initial_brems_stp)
                    * max_slice_density_path_m;
                fx.tally.RecordTransportGuardMetrics(
                    max_hard_tau,
                    max_brems_tau,
                    max_electron_total_competing_tau,
                    amrex::Real(0.0),
                    amrex::Real(0.0),
                    subcycles,
                    kinetic_energy_eV,
                    amrex::Real(0.0),
                    amrex::Real(0.0),
                    optical_depth_cap_violation);
                abort_if_strict_guard_failed(m_config, optical_depth_cap_violation);

                // Apply one continuous stopping segment at a frozen incident
                // energy.  The caller splits the chord exactly at a sampled
                // discrete-event depth, so sources before the event precede its
                // atomic group and sources after it use the surviving parent's
                // post-event energy.  Below the tracked cutoff no certified table
                // may be queried; terminal handoff owns that residual energy.
                // One drag module for all charged loops: the plan owns the
                // cutoff-limited integrals and the loss partition
                // (RreaPlanContinuousDragSegment); this caller keeps only the
                // electron POLICY -- restricted collision + soft radiative
                // stopping, the hard-Moller mean rate feeding the demotion
                // force balance, the ledger records, the ion-pair sink, and
                // the C&D plane-crossing hook.
                auto apply_electron_continuous_segment = [&] (
                    amrex::Real begin_fraction,
                    amrex::Real end_fraction,
                    amrex::Real stopping_energy_eV)
                    -> RreaContinuousSegmentLimit {
                    amrex::Real const kinetic_energy_before_segment_eV =
                        kinetic_energy_eV;
                    auto density_path = [&](amrex::Real b, amrex::Real e) {
                        return density_path_m(
                            material_intervals, transport_path, b, e);
                    };
                    auto const plan = RreaPlanContinuousDragSegment(
                        begin_fraction, end_fraction, kinetic_energy_eV,
                        stopping_energy_eV, coupling.LowEnergyCutoffEv(),
                        density_path,
                        [&] {
                            return RreaFrozenStoppingPowers{
                                RreaFrozenStpRate(tables.ElectronCollisionLoss(
                                    stopping_energy_eV, amrex::Real(1.0))),
                                RreaFrozenStpRate(tables.ElectronSoftRadiativeDrag(
                                    stopping_energy_eV, amrex::Real(1.0))),
                                (hard_moller_discrete_enabled
                                 && RreaHardMollerChannelOpen(
                                     stopping_energy_eV, hard_moller_threshold_eV))
                                ? RreaFrozenStpRate(
                                    tables.ElectronHardMollerEnergyLoss(
                                        stopping_energy_eV, amrex::Real(1.0)))
                                : amrex::Real(0.0)};
                        });
                    if (!plan.limit_valid) {
                        rrea::KineticFail(
                            "RREA electron continuous stopping could not locate "
                            "a finite monotone cutoff point");
                    }
                    if (!plan.applied) {
                        return RreaContinuousSegmentLimit{
                            begin_fraction, false, true};
                    }

                    fx.tally.RecordContinuousSoftBrems(
                        particle_weight, plan.soft_radiative_loss_eV);
                    fx.tally.RecordContinuousLossPartition(
                        particle_weight,
                        plan.drag_loss_eV,
                        plan.collision_loss_eV,
                        plan.hard_transfer_loss_eV);

                    {
                        amrex::Real const diagnostic_loss_eV =
                            tables.WAirEvPerIonPair();
                        RreaPartitionContinuousIonPairs(
                            material_intervals,
                            transport_path.full_length_m,
                            begin_fraction,
                            plan.effective_end_fraction,
                            plan.collision_loss_eV,
                            plan.raw_collision_loss_eV,
                            particle_weight,
                            diagnostic_loss_eV,
                            plan.collision_stp_eV_per_m,
                            [&](amrex::Real midpoint_fraction,
                                amrex::Real low_pair_weight) {
                                amrex::Real event_x = amrex::Real(0.0);
                                amrex::Real event_y = amrex::Real(0.0);
                                amrex::Real event_z = amrex::Real(0.0);
                                transport_path.PointAtFullPathFraction(
                                    midpoint_fraction,
                                    event_x, event_y, event_z);
                                fx.AccumulateIonizationEvent(
                                    0,
                                    std::hypot(event_x, event_y),
                                    event_z,
                                    low_pair_weight,
                                    amrex::Real(0.5)
                                        * coupling.LowEnergyCutoffEv(),
                                    diagnostic_loss_eV);
                                fx.tally.RecordTransportIonization(
                                    low_pair_weight,
                                    diagnostic_loss_eV,
                                    true,
                                    false);
                            });
                    }

                    defer_cd_electron_plane_crossings(
                        fx,
                        coupling,
                        tables,
                        transport_path,
                        material_intervals,
                        begin_fraction,
                        plan.effective_end_fraction,
                        particle_weight,
                        kinetic_energy_before_segment_eV,
                        stopping_energy_eV,
                        plan.drag_loss_eV,
                        hard_moller_discrete_enabled);

                    kinetic_energy_eV = rrea::kinetic::max(
                        amrex::Real(0.0),
                        kinetic_energy_eV - plan.drag_loss_eV);
                    if (plan.reaches_cutoff) {
                        kinetic_energy_eV = coupling.LowEnergyCutoffEv();
                    }
                    return RreaContinuousSegmentLimit{
                        plan.effective_end_fraction, plan.reaches_cutoff, true};
                };

                // The parent key already folds the particle's stored birth-rank
                // cpu() into the id, so per-rank id reuse cannot collide streams.
                std::uint64_t const rng_particle_id = transaction_parent_key;
                // Each rebuilt microchord is one complete subcycle; its local
                // fractions must not be sliced a second time.
                bool exited_domain = false;
                std::uint64_t completed_substeps = 0;
                // Fractions are local to one microchord. Every quantity that
                // is really an elapsed-time fraction of the OUTER step -- above all
                // a secondary's birth fraction -- must come through here, or a
                // secondary born late in the step is stamped early and
                // AdvanceChargedNewborn flies it a large multiple of the residual
                // it is owed.  The validator only range-checks [0,1].
                auto elapsed_fraction_at = [&](amrex::Real path_fraction) {
                    amrex::Real const within = rrea::kinetic::min(
                        amrex::Real(1.0),
                        rrea::kinetic::max(amrex::Real(0.0), path_fraction));
                    return rrea::kinetic::min(
                        amrex::Real(1.0),
                        (static_cast<amrex::Real>(completed_substeps) + within)
                            / static_cast<amrex::Real>(subcycles));
                };
                auto position_at_fraction = [&transport_path](
                        amrex::Real path_fraction,
                        amrex::Real& event_x,
                        amrex::Real& event_y,
                        amrex::Real& event_z,
                        amrex::Real& event_r) {
                    point_and_radius_at_fraction(
                        transport_path, path_fraction,
                        event_x, event_y, event_z, event_r);
                };
                // Substep-engine state and helpers. Under
                // substep_interleaved_v3 each uniform sub-interval advances
                // half field impulse -> drag/discrete hazards (existing
                // machinery, unchanged) -> GS elastic -> half field impulse,
                // with a mid-chord termination when the running energy falls
                // below the tracked cutoff strictly inside the chord.
                bool terminated_mid_chord = false;
                amrex::Real mid_chord_r_m = r_particle;
                amrex::Real mid_chord_z_m = z_particle;
                auto mark_electron_stopped_at = [&](amrex::Real path_fraction) {
                    if (RreaChargedStopIsEscape(
                            transport_path.exits_domain,
                            path_fraction,
                            transport_path.in_domain_end_fraction)) {
                        exited_domain = true;
                        return;
                    }
                    amrex::Real stop_x = amrex::Real(0.0);
                    amrex::Real stop_y = amrex::Real(0.0);
                    amrex::Real stop_z = amrex::Real(0.0);
                    amrex::Real stop_r = amrex::Real(0.0);
                    position_at_fraction(
                        path_fraction, stop_x, stop_y, stop_z, stop_r);
                    if (!RreaCartesianPositionInHalfOpenRzDomain(
                            stop_x,
                            stop_y,
                            stop_z,
                            geom.ProbLo(0),
                            geom.ProbHi(0),
                            geom.ProbLo(1),
                            geom.ProbHi(1))) {
                        exited_domain = true;
                        return;
                    }
                    mid_chord_z_m = stop_z;
                    mid_chord_r_m = stop_r;
                    terminated_mid_chord = true;
                };
                auto apply_field_half_kick = [&](
                    PathFieldSubRange const& field_range,
                    amrex::Real half_dt_s) {
                    static_cast<void>(RreaApplyChargedFieldHalfKick(
                        field_range, half_dt_s, amrex::Real(-1.0),
                        kinetic_energy_eV, dir_x, dir_y, dir_z));
                };
                auto apply_electron_gs_scatter = [&](
                    amrex::Real begin_fraction,
                    amrex::Real end_fraction,
                    std::uint64_t substep_idx) {
                    static constexpr GsScatterChannels kElectronGsChannels{
                        RreaRngChannel::ElectronGsFewCount,
                        RreaRngChannel::ElectronGsFewScatter,
                        RreaRngChannel::ElectronGsFewAzimuth,
                        RreaRngChannel::ElectronGsAngle,
                        RreaRngChannel::ElectronGsAzimuth};
                    auto gs_key = [&](RreaRngChannel channel,
                                      std::uint64_t inner) {
                        return RreaRngKey{coupling.RngSeed(), rng_particle_id,
                            static_cast<std::uint64_t>(rrea::kinetic::max(step, 0)),
                            (substep_idx << 8) | inner,
                            static_cast<std::uint64_t>(channel)};
                    };
                    apply_gs_elastic_scatter(
                        material_intervals, transport_path,
                        begin_fraction, end_fraction, tables,
                        /*is_positron=*/false, kElectronGsChannels,
                        gs_key, kinetic_energy_eV, dir_x, dir_y, dir_z);
                };
                auto finish_interleaved_substep = [&](
                    std::uint64_t substep_idx,
                    amrex::Real begin_fraction,
                    amrex::Real end_fraction,
                    PathFieldSubRange const& substep_field,
                    amrex::Real substep_dt_s) -> bool {
                    if (kinetic_energy_eV >= coupling.LowEnergyCutoffEv()) {
                        apply_electron_gs_scatter(
                            begin_fraction, end_fraction, substep_idx);
                    }
                    apply_field_half_kick(
                        substep_field, amrex::Real(0.5) * substep_dt_s);
                    if (kinetic_energy_eV < coupling.LowEnergyCutoffEv()) {
                        // Stopped on this microchord: demote at its recorded
                        // endpoint, which is the electron's actual position.
                        // mark_electron_stopped_at itself turns
                        // a stop at the clip root of an exiting chord into an
                        // escape.
                        mark_electron_stopped_at(end_fraction);
                        return true;
                    }
                    return false;
                };
                amrex::Real const sub_dt_s =
                    dt_s / static_cast<amrex::Real>(subcycles);
                PathFieldSubRange substep_field{};
                amrex::Real substep_dt_s = amrex::Real(0.0);
                RreaRunChargedSubstepInterleave(
                    subcycles,
                    kinetic_energy_eV,
                    coupling.LowEnergyCutoffEv(),
                    completed_substeps,
                    exited_domain,
                    [&](std::uint64_t,
                        amrex::Real& subcycle_begin,
                        amrex::Real& subcycle_end) {
                        // Rebuild from the current state so field work follows any
                        // within-step backscattering.
                        auto const chord = RreaBuildRunningChargedChord(
                            RreaRunningChargedState{
                                run_x, run_y, run_z, dir_x, dir_y, dir_z,
                                kinetic_energy_eV},
                            sub_dt_s,
                            dt_s,
                            geom.ProbLo(0),
                            geom.ProbHi(0),
                            geom.ProbLo(1),
                            geom.ProbHi(1),
                            periodic_z);
                        if (!chord.valid) {
                            return false;
                        }
                        transport_path = chord.path;
                        if (!(transport_path.full_length_m > amrex::Real(0.0))
                            || !(transport_path.in_domain_length_m
                                 > amrex::Real(0.0))) {
                            exited_domain = transport_path.exits_domain;
                            return false;
                        }
                        // Material intervals must match this microchord.
                        material_intervals = build_material_path_intervals(
                            coupling, transport_path, density_ratio);
                        if (material_intervals.empty()) {
                            rrea::KineticFail(
                                "RREA electron microchord has positive in-domain "
                                "length but no material intervals");
                        }
                        // The density the electron actually flies through, for the
                        // post-loop demotion gate.
                        for (auto const& interval : material_intervals) {
                            transported_density_path_m +=
                                interval.length_m * interval.density_ratio;
                            transported_length_m += interval.length_m;
                        }
                        subcycle_begin = transport_path.in_domain_begin_fraction;
                        subcycle_end = transport_path.in_domain_end_fraction;
                        substep_field = path_field_work_and_average(
                            coupling,
                            transport_path,
                            amrex::Real(-1.0),
                            subcycle_begin,
                            subcycle_end);
                        // Time ACTUALLY spent on this chord.  The work integral is
                        // already restricted to [begin, end], but the half-kick's
                        // ROTATION scales with dt, so handing it the whole subcycle
                        // over-rotates a clipped microchord by 1/(end-begin): energy
                        // exactly right, direction saturated to E, and no conservation
                        // ledger can see it.
                        substep_dt_s = sub_dt_s * (subcycle_end - subcycle_begin);
                        apply_field_half_kick(
                            substep_field, amrex::Real(0.5) * substep_dt_s);
                        if (kinetic_energy_eV < coupling.LowEnergyCutoffEv()) {
                            mark_electron_stopped_at(subcycle_begin);
                            return false;
                        }
                        return true;
                    },
                    [&](std::uint64_t subcycle,
                        amrex::Real subcycle_begin,
                        amrex::Real subcycle_end) {
                    // Event-loop policies: the electron's physics content; the
                    // substep_interleaved_v3 sequencing lives in
                    // RreaRunChargedEventInterleave, where its invariants are testable.
                    amrex::Real event_x = amrex::Real(0.0);
                    amrex::Real event_y = amrex::Real(0.0);
                    amrex::Real event_z = amrex::Real(0.0);
                    amrex::Real event_r = amrex::Real(0.0);
                    std::uint64_t interaction_rng_index = 0;
                    // Frozen STP inverse lengths of the two competing channels,
                    // refreshed by the channel_taus policy at every refreeze
                    // and read by the depth inversion that follows it.
                    std::array<amrex::Real, 3> stp{};
                    RreaRunChargedEventInterleave(
                        subcycle_begin,
                        subcycle_end,
                        kinetic_energy_eV,
                        coupling.LowEnergyCutoffEv(),
                        2,
                        kRreaTransientEventCapacity,
                        "electron",
                        [&](amrex::Real frozen_energy_eV,
                            amrex::Real tau_begin_fraction,
                            amrex::Real tau_end_fraction) {
                            stp = RreaChargedCompetingStpRates(
                                tables, /*is_positron=*/false,
                                hard_moller_threshold_eV,
                                coupling.PhotonCutoffEv(),
                                frozen_energy_eV,
                                hard_moller_discrete_enabled);
                            amrex::Real const segment_density_path_m = density_path_m(
                                material_intervals,
                                transport_path,
                                tau_begin_fraction,
                                tau_end_fraction);
                            amrex::Real const hard_tau =
                                stp[0] * segment_density_path_m;
                            amrex::Real const brems_tau =
                                stp[1] * segment_density_path_m;
                            amrex::Real const total_tau = hard_tau + brems_tau;
                            amrex::Real const total_probability =
                                probability_from_optical_depth(total_tau);
                            amrex::Real const hard_probability = total_tau > amrex::Real(0.0)
                                ? total_probability * hard_tau / total_tau
                                : amrex::Real(0.0);
                            fx.tally.RecordHardMollerExpectation(particle_weight * hard_probability);
                            return std::array<amrex::Real, 3>{
                                hard_tau, brems_tau, amrex::Real(0.0)};
                        },
                        [&](std::uint64_t draw_event_ordinal) {
                            auto const live_event_index = RreaTryLiveHazardRngIndex(
                                RreaSecondarySpecies::Electron,
                                subcycle, draw_event_ordinal);
                            if (!live_event_index.valid) {
                                rrea::KineticFail(
                                    "RREA electron repeated-hazard RNG index overflow; "
                                    "the uncommitted transaction is rejected");
                            }
                            interaction_rng_index = live_event_index.value;
                            RreaRngChannel const optical_depth_channel = draw_event_ordinal == 0U
                                ? RreaRngChannel::ElectronCompetingOpticalDepth
                                : RreaRngChannel::ElectronContinuationOpticalDepth;
                            RreaRngChannel const competing_channel = draw_event_ordinal == 0U
                                ? RreaRngChannel::ElectronCompetingChannel
                                : RreaRngChannel::ElectronContinuationChannel;
                            return std::pair<amrex::Real, amrex::Real>{
                                RreaRng::Uniform01(RreaRngKey{
                                            coupling.RngSeed(),
                                            rng_particle_id,
                                            static_cast<std::uint64_t>(rrea::kinetic::max(step, 0)),
                                            interaction_rng_index,
                                            static_cast<std::uint64_t>(optical_depth_channel)}),
                                RreaRng::Uniform01(RreaRngKey{
                                            coupling.RngSeed(),
                                            rng_particle_id,
                                            static_cast<std::uint64_t>(rrea::kinetic::max(step, 0)),
                                            interaction_rng_index,
                                            static_cast<std::uint64_t>(competing_channel)})};
                        },
                        [&](amrex::Real sampled_optical_depth,
                            amrex::Real invert_begin_fraction,
                            amrex::Real invert_end_fraction) {
                            // tau = (sigma_hard + sigma_brems) * L_chi, so the
                            // depth inverts as a density path.
                            return fraction_at_density_path_m(
                                material_intervals,
                                transport_path,
                                invert_begin_fraction,
                                invert_end_fraction,
                                sampled_optical_depth / (stp[0] + stp[1]));
                        },
                        [&](amrex::Real segment_begin_fraction,
                            amrex::Real segment_end_fraction,
                            amrex::Real frozen_energy_eV) {
                            return apply_electron_continuous_segment(
                                segment_begin_fraction,
                                segment_end_fraction,
                                frozen_energy_eV);
                        },
                        [&](amrex::Real path_fraction) {
                            mark_electron_stopped_at(path_fraction);
                        },
                        [&](amrex::Real point_event_fraction) {
                            position_at_fraction(
                                point_event_fraction,
                                event_x,
                                event_y,
                                event_z,
                                event_r);
                        },
                        [&](int channel, amrex::Real channel_energy_eV) {
                            return RreaChargedChannelOpenAt(
                                tables, /*is_positron=*/false,
                                hard_moller_threshold_eV,
                                coupling.PhotonCutoffEv(),
                                channel, channel_energy_eV,
                                hard_moller_discrete_enabled);
                        },
                        [&](int channel, amrex::Real event_fraction) {
                            RreaInteractionSite const site{
                                event_x, event_y, event_z, event_r,
                                elapsed_fraction_at(event_fraction)};
                            auto rng_key = [&](RreaRngChannel rng_channel) {
                                return RreaRngKey{
                                    coupling.RngSeed(),
                                    rng_particle_id,
                                    static_cast<std::uint64_t>(
                                        rrea::kinetic::max(step, 0)),
                                    interaction_rng_index,
                                    static_cast<std::uint64_t>(rng_channel)};
                            };
                            if (channel == 0) {
                                RreaApplyHardMollerFinalState(
                                    fx, tables, rng_key, site, particle_weight,
                                    hard_moller_threshold_eV,
                                    kinetic_energy_eV, dir_x, dir_y, dir_z);
                            } else if (channel == 1) {
                                RreaApplyBremsFinalState(
                                    fx, tables, rng_key,
                                    [&](amrex::Real local_energy_eV) {
                                        deposit_local_energy_as_ion_pairs(
                                            fx, event_r, event_z,
                                            particle_weight, local_energy_eV);
                                    },
                                    site, particle_weight,
                                    /*charge_sign=*/amrex::Real(-1.0),
                                    coupling.PhotonCutoffEv(),
                                    kinetic_energy_eV, dir_x, dir_y, dir_z);
                            }
                            return true;
                        });
                    },
                    [&] { return terminated_mid_chord; },
                    [&](std::uint64_t subcycle,
                        amrex::Real subcycle_begin,
                        amrex::Real subcycle_end) {
                        return finish_interleaved_substep(
                            subcycle, subcycle_begin, subcycle_end,
                            substep_field, substep_dt_s);
                    },
                    // Commit this microchord: the next one is rebuilt from here, along
                    // whatever direction the scattering above left behind.  A refused
                    // endpoint is one the next chord would have been built from while
                    // lying outside the half-open domain, so it becomes an escape.
                    [&](amrex::Real subcycle_end) {
                        return RreaAdvanceRunningChargedPosition(
                            transport_path,
                            subcycle_end,
                            geom.ProbLo(0),
                            geom.ProbHi(0),
                            geom.ProbLo(1),
                            geom.ProbHi(1),
                            run_x,
                            run_y,
                            run_z);
                    },
                    [&] { return transport_path.exits_domain; });
                if (exited_domain) {
                    charged_parent_escape =
                        RreaClippedTransportEscapePolicyForPath(
                            transport_path,
                            std::hypot(
                                transport_path.end_x_m,
                                transport_path.end_y_m),
                            transport_path.end_z_m);
                }
                // Evaluate demotion once after transport, using traversed density
                // and the field at the reached endpoint. A per-substep gate would
                // make the disposition timestep-dependent.
                if (transported_length_m > amrex::Real(0.0)) {
                    particle_density_ratio =
                        transported_density_path_m / transported_length_m;
                }
                // run_* is the last COMMITTED point, which is always in-domain: a
                // refused commit leaves it where it was, so this gather can never
                // leave the field cache's halo.
                amrex::Real const endpoint_r_m = std::hypot(run_x, run_y);
                RreaFieldSample const endpoint_field =
                    coupling.GatherFieldRZ(0, endpoint_r_m, run_z);
                // The gate only sees K <= the hard-Moller threshold (50 keV),
                // where the ensemble mean drag is still above 284 kV/m: the drag
                // curve falls below that only above ~205 keV.
                amrex::Real const threshold_v_per_m =
                    electron_ensemble_mean_drag_ev_per_m(
                        tables,
                        rrea::kinetic::max(kinetic_energy_eV, coupling.LowEnergyCutoffEv()),
                        particle_density_ratio,
                        hard_moller_discrete_enabled);
                auto const state_policy = RreaEvaluateElectronStatePolicy(
                        kinetic_energy_eV,
                        coupling.LowEnergyCutoffEv(),
                        hard_moller_threshold_eV,
                        std::hypot(
                            endpoint_field.er_v_per_m,
                            endpoint_field.ez_v_per_m),
                        threshold_v_per_m);
                // A microchord leaving the domain takes precedence over an
                // in-domain terminal disposition.
                auto const terminal_disposition =
                    RreaResolveChargedTransportTerminal(
                        exited_domain,
                        terminated_mid_chord,
                        RreaChargedTerminalDisposition::ElectronDemotion,
                        /*positron=*/false,
                        state_policy.below_transport_cutoff,
                        state_policy.low_energy_nonrunaway);
                if (terminal_disposition
                    == RreaChargedTerminalDisposition::Escape) {
                    // All in-domain stopping and sampled discrete events have
                    // been booked.  WarpX owns geometric overshoots; RREA owns
                    // exact half-open upper-face invalidation.  Neither may be
                    // demoted at the excluded endpoint.
                    commit_charged_escape();
                    continue;
                }

                // Only below-threshold, locally nonrunaway particles are handed
                // to the mesh; freshly-created above-cutoff knock-on electrons
                // remain in kinetic transport.
                if (terminal_disposition
                    == RreaChargedTerminalDisposition::ElectronDemotion) {
                    // A demoted electron can carry
                    // up to the hard-Moller threshold of kinetic energy;
                    // physically it ionizes ~K/W_air pairs within a sub-metre
                    // range before thermalizing.  Deposit that energy locally
                    // instead of dropping it;
                    // energy 0 in the mesh diagnostic avoids double counting.
                    // A mid-chord termination deposits at the
                    // true in-domain stopping position, not the far endpoint.
                    // Not the pre-push point and not WarpX's straight-line
                    // endpoint: where the electron actually ended up.
                    amrex::Real const demotion_r_m =
                        terminated_mid_chord ? mid_chord_r_m : endpoint_r_m;
                    amrex::Real const demotion_z_m =
                        terminated_mid_chord ? mid_chord_z_m : run_z;
                    fx.BeginGroup(RreaSecondaryProcess::LowEnergyDemotion, true);
                    fx.AccumulateLowEnergyElectron(
                        0,
                        demotion_r_m,
                        demotion_z_m,
                        particle_weight,
                        amrex::Real(0.0));
                    deposit_local_energy_as_ion_pairs(
                        fx,
                        demotion_r_m,
                        demotion_z_m,
                        particle_weight,
                        kinetic_energy_eV);
                    fx.SetInteractionConservationClosure(
                        particle_weight, kinetic_energy_eV, amrex::Real(0.0));
                    // Net chord: captured start to the mid-chord kill point
                    // (or the committed endpoint when the gate fired there).
                    fx.AccumulateChargedSegment(
                        0, r_particle, z_particle, demotion_r_m, demotion_z_m,
                        -particle_weight, kRreaSegmentChannelDemoted);
                    fx.EndGroup();
                    particle_id.make_invalid();
                    fx.invalidated = true;
                    continue;
                }

                CommitSurvivorChargedWriteback(
                    fx, soa, ip,
                    kinetic_energy_eV, dir_x, dir_y, dir_z,
                    run_x, run_y, run_z,
                    r_particle, z_particle,
                    endpoint_r_m,
                    /*signed_mesh_weight=*/-particle_weight);

        } while(false);
    }
    template<class Effects,class Id>
    RREA_HOST_DEVICE void Positron(Effects& fx,ChargedSoAView const& soa,long ip,Id particle_id,int particle_cpu) const {
        do {

                if (!particle_id.is_valid()) {
                    continue;
                }
                    auto setup = BeginRunningChargedParticle(
                        fx, soa, ip,
                        static_cast<amrex::Long>(particle_id),
                        particle_cpu,
                        RreaSecondarySpecies::Positron,
                        /*charge_sign=*/amrex::Real(1.0),
                        "positron");
                    if (setup.skip) {
                        particle_id.make_invalid();
                        fx.invalidated = true;
                        continue;
                    }
                    amrex::Real const particle_weight = setup.particle_weight;
                    amrex::Real dir_x = setup.dir_x;
                    amrex::Real dir_y = setup.dir_y;
                    amrex::Real dir_z = setup.dir_z;
                    amrex::Real kinetic_energy_eV = setup.kinetic_energy_eV;
                    amrex::Real run_x = setup.run_x;
                    amrex::Real run_y = setup.run_y;
                    amrex::Real run_z = setup.run_z;
                    amrex::Real const x_particle = setup.x_particle;
                    amrex::Real const y_particle = setup.y_particle;
                    amrex::Real const z_particle = setup.z_particle;
                    amrex::Real const r_particle = setup.r_particle;
                    RreaTransportPathSegment const driver_path =
                        setup.driver_path;
                    RreaEndpointEscapePolicy charged_parent_escape =
                        setup.charged_parent_escape;
                    amrex::Real const ds_m = setup.ds_m;
                    auto const& driver_intervals = setup.driver_intervals;
                    amrex::Real const driver_span = setup.driver_span;
                    // Positrons have no directional demotion
                    // gate; the whole-chord work integral only sizes the
                    // field-impulse substep driver.
                    amrex::Real const clipped_field_work_eV =
                        setup.field_work_eV;
                    // Rebuilt with each running-state chord; intervals from another
                    // chord have an empty material-rate intersection.
                    RreaTransportPathSegment transport_path = driver_path;
                    rrea::KineticVector<MaterialPathInterval> material_intervals =
                        driver_intervals;
                    auto commit_charged_escape = [&]() {
                        CommitChargedEscape(
                            fx, soa, ip,
                            particle_weight, kinetic_energy_eV,
                            dir_x, dir_y, dir_z,
                            r_particle, z_particle,
                            transport_path, charged_parent_escape,
                            /*signed_mesh_weight=*/particle_weight);
                        if (charged_parent_escape.invalidate_parent_in_rrea) {
                            particle_id.make_invalid();
                            fx.invalidated = true;
                        }
                    };
                    std::uint64_t const rng_particle_id =
                        setup.transaction_parent_key;
                    auto const initial_terminal_disposition =
                        RreaResolveChargedTerminalDisposition(
                            /*positron=*/true,
                            charged_parent_escape.escapes,
                            kinetic_energy_eV < coupling.LowEnergyCutoffEv(),
                            /*electron_directional_demotion=*/false);
                    if (initial_terminal_disposition
                        == RreaChargedTerminalDisposition::Escape
                        && kinetic_energy_eV < coupling.LowEnergyCutoffEv()) {
                        // Boundary ownership precedes endpoint thermal
                        // annihilation.  In particular, never place the two
                        // annihilation photons or residual-energy source on an
                        // excluded upper face.
                        commit_charged_escape();
                        continue;
                    }
                    if (initial_terminal_disposition
                        == RreaChargedTerminalDisposition::PositronAnnihilation) {
                        fx.BeginGroup(RreaSecondaryProcess::PositronAnnihilation, true);
                        spawn_positron_annihilation(
                            fx,
                            x_particle,
                            y_particle,
                            z_particle,
                            r_particle,
                            z_particle,
                            dir_x,
                            dir_y,
                            dir_z,
                            particle_weight,
                            kinetic_energy_eV,
                            true,
                            rng_particle_id,
                            0);
                        fx.tally.RecordPositronTransport(particle_weight, kinetic_energy_eV, true);
                        fx.SetInteractionConservationClosure(
                            particle_weight,
                            kinetic_energy_eV,
                            amrex::Real(0.0));
                        // Zero-length net chord: annihilated at the captured
                        // start before any transport.
                        fx.AccumulateChargedSegment(
                            0, r_particle, z_particle, r_particle, z_particle,
                            particle_weight, kRreaSegmentChannelAnnihilated);
                        fx.EndGroup();
                        particle_id.make_invalid();
                        fx.invalidated = true;
                        continue;
                    }
                    if (ds_m <= amrex::Real(0.0)) {
                        if (charged_parent_escape.escapes) {
                            commit_charged_escape();
                        }
                        continue;
                    }
                    if (driver_intervals.empty()) {
                        rrea::KineticFail(
                            "RREA positron path has positive length but no material segments");
                    }
                    // Annihilation is open at any energy; Bhabha and tracked
                    // brems carry their own phase-space gates, exactly as for
                    // the electron loop (RreaChargedCompetingStpRates).
                    auto const initial_positron_stp =
                        RreaChargedCompetingStpRates(
                            tables, /*is_positron=*/true,
                            coupling.HardMollerSecondaryThresholdEv(),
                            coupling.PhotonCutoffEv(),
                            kinetic_energy_eV);
                    amrex::Real const initial_total_positron_stp =
                        initial_positron_stp[0] + initial_positron_stp[1]
                        + initial_positron_stp[2];
                    // Every budget below is a DRIVER estimate: integrated over the
                    // driver chord's clipped in-domain span, then divided by that
                    // span because the loop transports the whole dt.
                    amrex::Real const driver_density_path_m = density_path_m(
                        driver_intervals,
                        driver_path,
                        driver_path.in_domain_begin_fraction,
                        driver_path.in_domain_end_fraction) / driver_span;
                    amrex::Real const positron_tau =
                        initial_total_positron_stp * driver_density_path_m;
                    bool optical_depth_cap_violation = false;
                    std::uint64_t subcycles = selected_material_subcycles(
                        positron_tau,
                        driver_intervals,
                        driver_path,
                        m_config,
                        optical_depth_cap_violation,
                        initial_total_positron_stp);
                    {
                        // GS batching and field-impulse/drag
                        // substep drivers (mirror of the electron loop; benign
                        // silent clamp at the cap).
                        amrex::Real const initial_gs_n_total =
                            elastic_inverse_length_stp(
                                tables, true, kinetic_energy_eV)
                            * driver_density_path_m;
                        amrex::Real const drag_estimate_eV =
                            tables.PositronCollisionLoss(
                                rrea::kinetic::max(
                                    kinetic_energy_eV, coupling.LowEnergyCutoffEv()),
                                amrex::Real(1.0))
                            * driver_density_path_m;
                        amrex::Real const impulse_scale_eV = rrea::kinetic::max(
                            m_config.transport_max_field_impulse_fraction
                                * kinetic_energy_eV,
                            rrea::kinetic::max(
                                m_config.transport_field_impulse_floor_eV,
                                amrex::Real(1.0)));
                        std::uint64_t const gs_driver = substep_driver_count(
                            rrea::kinetic::min(
                                initial_gs_n_total,
                                m_config.transport_gs_total_collision_cap),
                            rrea::kinetic::max(
                                m_config.transport_max_gs_collisions_per_substep,
                                amrex::Real(1.0)));
                        std::uint64_t const impulse_driver = substep_driver_count(
                            std::abs(clipped_field_work_eV) + drag_estimate_eV,
                            impulse_scale_eV);
                        subcycles = std::min<std::uint64_t>(
                            rrea::kinetic::max({subcycles, gs_driver, impulse_driver}),
                            m_config.max_subcycles);
                    }
                    // A driver estimate too: uniform slices of a chord the loop no
                    // longer transports verbatim.  Reported, not enforced, and
                    // divided by the clipped span like every enforced budget so
                    // it describes the full-dt microchords actually flown.
                    amrex::Real const max_positron_tau =
                        initial_total_positron_stp
                        * max_uniform_subcycle_density_path_m(
                            driver_intervals, driver_path, subcycles) / driver_span;
                    fx.tally.RecordTransportGuardMetrics(
                        amrex::Real(0.0),
                        amrex::Real(0.0),
                        amrex::Real(0.0),
                        amrex::Real(0.0),
                        max_positron_tau,
                        subcycles,
                        amrex::Real(0.0),
                        amrex::Real(0.0),
                        kinetic_energy_eV,
                        optical_depth_cap_violation);
                    abort_if_strict_guard_failed(m_config, optical_depth_cap_violation);
                    amrex::Real total_loss_eV = amrex::Real(0.0);
                    bool annihilated = false;
                    // Each rebuilt microchord is one complete subcycle, so the
                    // transport window is its whole in-domain span.
                    auto position_at_fraction = [&transport_path](
                            amrex::Real path_fraction,
                            amrex::Real& event_x,
                            amrex::Real& event_y,
                            amrex::Real& event_z,
                            amrex::Real& event_r) {
                        point_and_radius_at_fraction(
                            transport_path, path_fraction,
                            event_x, event_y, event_z, event_r);
                    };

                    // Apply and book positron collision/soft-radiative stopping
                    // only on the supplied geometric interval.  Selection rates
                    // remain frozen at the enclosing subcycle's start energy;
                    // after a surviving event this helper is called again with
                    // the post-event energy for the unused interval.
                    // Same drag module as the electron loop
                    // (RreaPlanContinuousDragSegment); the positron POLICY:
                    // collision + soft radiative stopping, hard transfers
                    // stay per-event (Bhabha) so the mean-rate partition is
                    // zero, and the running total feeds the
                    // annihilation-at-rest budget.
                    auto apply_positron_continuous_segment = [&] (
                        amrex::Real begin_fraction,
                        amrex::Real end_fraction,
                        amrex::Real stopping_energy_eV)
                        -> RreaContinuousSegmentLimit {
                        auto density_path = [&](amrex::Real b, amrex::Real e) {
                            return density_path_m(
                                material_intervals, transport_path, b, e);
                        };
                        auto const plan = RreaPlanContinuousDragSegment(
                            begin_fraction, end_fraction, kinetic_energy_eV,
                            stopping_energy_eV, coupling.LowEnergyCutoffEv(),
                            density_path,
                            [&] {
                                return RreaFrozenStoppingPowers{
                                    RreaFrozenStpRate(tables.PositronCollisionLoss(
                                        stopping_energy_eV, amrex::Real(1.0))),
                                    RreaFrozenStpRate(tables.PositronSoftRadiativeDrag(
                                        stopping_energy_eV, amrex::Real(1.0))),
                                    amrex::Real(0.0)};
                            });
                        if (!plan.limit_valid) {
                            rrea::KineticFail(
                                "RREA positron continuous stopping could not "
                                "locate a finite monotone cutoff point");
                        }
                        if (!plan.applied) {
                            return RreaContinuousSegmentLimit{
                                begin_fraction, false, true};
                        }
                        fx.tally.RecordContinuousSoftBrems(
                            particle_weight, plan.soft_radiative_loss_eV);
                        total_loss_eV += plan.drag_loss_eV;
                        kinetic_energy_eV = rrea::kinetic::max(
                            amrex::Real(0.0),
                            kinetic_energy_eV - plan.drag_loss_eV);
                        if (plan.reaches_cutoff) {
                            kinetic_energy_eV = coupling.LowEnergyCutoffEv();
                        }

                        {
                            amrex::Real const diagnostic_loss_eV =
                                tables.WAirEvPerIonPair();
                            RreaPartitionContinuousIonPairs(
                                material_intervals,
                                transport_path.full_length_m,
                                begin_fraction,
                                plan.effective_end_fraction,
                                plan.collision_loss_eV,
                                plan.raw_collision_loss_eV,
                                particle_weight,
                                diagnostic_loss_eV,
                                plan.collision_stp_eV_per_m,
                                [&](amrex::Real midpoint_fraction,
                                    amrex::Real pair_weight) {
                                    amrex::Real source_x = amrex::Real(0.0);
                                    amrex::Real source_y = amrex::Real(0.0);
                                    amrex::Real source_z = amrex::Real(0.0);
                                    amrex::Real source_r = amrex::Real(0.0);
                                    position_at_fraction(
                                        midpoint_fraction,
                                        source_x, source_y, source_z, source_r);
                                    fx.AccumulateIonizationEvent(
                                        0,
                                        source_r,
                                        source_z,
                                        pair_weight,
                                        amrex::Real(0.0),
                                        diagnostic_loss_eV);
                                    fx.tally.RecordTransportIonization(
                                        pair_weight,
                                        diagnostic_loss_eV,
                                        true,
                                        false);
                                });
                        }
                        return RreaContinuousSegmentLimit{
                            plan.effective_end_fraction, plan.reaches_cutoff,
                            true};
                    };
                    // The positron substep engine mirrors the electron loop; a
                    // positron stopping inside the chord annihilates at rest at
                    // the substep position.
                    bool exited_domain = false;
                    std::uint64_t completed_substeps = 0;
                    // Convert microchord-local positions to outer-step elapsed
                    // fractions before recording secondary birth times.
                    auto elapsed_fraction_at = [&](amrex::Real path_fraction) {
                        amrex::Real const within = rrea::kinetic::min(
                            amrex::Real(1.0),
                            rrea::kinetic::max(amrex::Real(0.0), path_fraction));
                        return rrea::kinetic::min(
                            amrex::Real(1.0),
                            (static_cast<amrex::Real>(completed_substeps) + within)
                                / static_cast<amrex::Real>(subcycles));
                    };
                    bool stopped_before_trailing_half = false;
                    amrex::Real stopping_x_m = x_particle;
                    amrex::Real stopping_y_m = y_particle;
                    amrex::Real stopping_z_m = z_particle;
                    amrex::Real stopping_r_m = r_particle;
                    // Stop GEOMETRY and stop TIME are different quantities: the
                    // path fraction locates the stop on the current microchord, the
                    // elapsed fraction says when in the outer step it happened, and
                    // it is the latter the annihilation photons' residual advance
                    // needs.  Evaluated at mark time, before completed_substeps
                    // moves past the microchord the stop happened on.
                    amrex::Real stopping_elapsed_fraction = amrex::Real(1.0);
                    auto mark_positron_stopped_at = [&](amrex::Real path_fraction) {
                        if (RreaChargedStopIsEscape(
                                transport_path.exits_domain,
                                path_fraction,
                                transport_path.in_domain_end_fraction)) {
                            exited_domain = true;
                            return;
                        }
                        amrex::Real stop_x = amrex::Real(0.0);
                        amrex::Real stop_y = amrex::Real(0.0);
                        amrex::Real stop_z = amrex::Real(0.0);
                        amrex::Real stop_r = amrex::Real(0.0);
                        position_at_fraction(
                            path_fraction, stop_x, stop_y, stop_z, stop_r);
                        if (!RreaCartesianPositionInHalfOpenRzDomain(
                                stop_x,
                                stop_y,
                                stop_z,
                                geom.ProbLo(0),
                                geom.ProbHi(0),
                                geom.ProbLo(1),
                                geom.ProbHi(1))) {
                            exited_domain = true;
                            return;
                        }
                        stopping_x_m = stop_x;
                        stopping_y_m = stop_y;
                        stopping_z_m = stop_z;
                        stopping_r_m = stop_r;
                        stopping_elapsed_fraction =
                            elapsed_fraction_at(path_fraction);
                        stopped_before_trailing_half = true;
                    };
                    auto apply_field_half_kick = [&](
                        PathFieldSubRange const& field_range,
                        amrex::Real half_dt_s) {
                        static_cast<void>(RreaApplyChargedFieldHalfKick(
                            field_range, half_dt_s, amrex::Real(1.0),
                            kinetic_energy_eV, dir_x, dir_y, dir_z));
                    };
                    auto apply_positron_gs_scatter = [&](
                        amrex::Real begin_fraction,
                        amrex::Real end_fraction,
                        std::uint64_t substep_idx) {
                        static constexpr GsScatterChannels kPositronGsChannels{
                            RreaRngChannel::PositronGsFewCount,
                            RreaRngChannel::PositronGsFewScatter,
                            RreaRngChannel::PositronGsFewAzimuth,
                            RreaRngChannel::PositronGsAngle,
                            RreaRngChannel::PositronGsAzimuth};
                        auto gs_key = [&](RreaRngChannel channel,
                                          std::uint64_t inner) {
                            return RreaRngKey{
                                coupling.RngSeed(), rng_particle_id,
                                static_cast<std::uint64_t>(rrea::kinetic::max(step, 0)),
                                (substep_idx << 8) | inner,
                                static_cast<std::uint64_t>(channel)};
                        };
                        apply_gs_elastic_scatter(
                            material_intervals, transport_path,
                            begin_fraction, end_fraction, tables,
                            /*is_positron=*/true, kPositronGsChannels,
                            gs_key, kinetic_energy_eV, dir_x, dir_y, dir_z);
                    };
                    auto finish_interleaved_substep = [&](
                        std::uint64_t substep_idx,
                        amrex::Real begin_fraction,
                        amrex::Real end_fraction,
                        PathFieldSubRange const& substep_field,
                        amrex::Real substep_dt_s) -> bool {
                        if (kinetic_energy_eV >= coupling.LowEnergyCutoffEv()) {
                            apply_positron_gs_scatter(
                                begin_fraction, end_fraction, substep_idx);
                        }
                        apply_field_half_kick(
                            substep_field, amrex::Real(0.5) * substep_dt_s);
                        if (kinetic_energy_eV < coupling.LowEnergyCutoffEv()) {
                            // Stopped on this microchord.  Only RECORD where and
                            // when; the single terminal booking after the loop
                            // owns the annihilation.  Booking it here would open a
                            // second remove_parent group for the same parent (a
                            // collective abort once every rank has finished
                            // transport) and would let a stop override an escape.
                            mark_positron_stopped_at(end_fraction);
                            return true;
                        }
                        return false;
                    };
                    amrex::Real const sub_dt_s =
                        dt_s / static_cast<amrex::Real>(subcycles);
                    PathFieldSubRange substep_field{};
                    amrex::Real substep_dt_s = amrex::Real(0.0);
                    RreaRunChargedSubstepInterleave(
                        subcycles,
                        kinetic_energy_eV,
                        coupling.LowEnergyCutoffEv(),
                        completed_substeps,
                        exited_domain,
                        [&](std::uint64_t,
                            amrex::Real& subcycle_begin,
                            amrex::Real& subcycle_end) {
                            // Rebuild the chord the positron will actually fly over this
                            // subcycle, from the state it is actually in, so elastic
                            // deflection and discrete recoil bend the path it then
                            // follows and the field work projects onto the direction it
                            // is really travelling.
                            auto const chord = RreaBuildRunningChargedChord(
                                RreaRunningChargedState{
                                    run_x, run_y, run_z, dir_x, dir_y, dir_z,
                                    kinetic_energy_eV},
                                sub_dt_s,
                                dt_s,
                                geom.ProbLo(0),
                                geom.ProbHi(0),
                                geom.ProbLo(1),
                                geom.ProbHi(1),
                                periodic_z);
                            if (!chord.valid) {
                                return false;
                            }
                            transport_path = chord.path;
                            if (!(transport_path.full_length_m > amrex::Real(0.0))
                                || !(transport_path.in_domain_length_m
                                     > amrex::Real(0.0))) {
                                exited_domain = transport_path.exits_domain;
                                return false;
                            }
                            // Material intervals must match this microchord.
                            material_intervals = build_material_path_intervals(
                                coupling, transport_path, density_ratio);
                            if (material_intervals.empty()) {
                                rrea::KineticFail(
                                    "RREA positron microchord has positive in-domain "
                                    "length but no material intervals");
                            }
                            subcycle_begin = transport_path.in_domain_begin_fraction;
                            subcycle_end = transport_path.in_domain_end_fraction;
                            substep_field = path_field_work_and_average(
                                coupling,
                                transport_path,
                                amrex::Real(1.0),
                                subcycle_begin,
                                subcycle_end);
                            // Time ACTUALLY spent on this chord.  The work integral is
                            // already restricted to [begin, end], but the half-kick's
                            // ROTATION scales with dt, so handing it the whole subcycle
                            // over-rotates a clipped microchord by 1/(end-begin):
                            // energy exactly right, direction saturated to E, and no
                            // conservation ledger can see it.
                            substep_dt_s = sub_dt_s * (subcycle_end - subcycle_begin);
                            apply_field_half_kick(
                                substep_field, amrex::Real(0.5) * substep_dt_s);
                            if (kinetic_energy_eV < coupling.LowEnergyCutoffEv()) {
                                mark_positron_stopped_at(subcycle_begin);
                                return false;
                            }
                            return true;
                        },
                        [&](std::uint64_t subcycle,
                            amrex::Real subcycle_begin,
                            amrex::Real subcycle_end) {
                        // Event-loop policies: the positron's physics content; the shared
                        // substep_interleaved_v3 sequencing lives in
                        // RreaRunChargedEventInterleave.
                        amrex::Real event_x = amrex::Real(0.0);
                        amrex::Real event_y = amrex::Real(0.0);
                        amrex::Real event_z = amrex::Real(0.0);
                        amrex::Real event_r = amrex::Real(0.0);
                        std::uint64_t interaction_rng_index = 0;
                        amrex::Real const bhabha_secondary_threshold_eV =
                            coupling.HardMollerSecondaryThresholdEv();
                        // The hard-Bhabha table represents above-threshold target electron
                        // production and is authoritative immediately above its exact
                        // zero-rate threshold.  The outgoing positron need not remain above
                        // the transport cutoff; its later at-rest annihilation handles that
                        // physical branch.  The Bhabha phase-space gate scopes only that
                        // component of the summed hazard; brems and annihilation remain live
                        // in the low band.
                        // Frozen STP inverse lengths, refreshed by the
                        // channel_taus policy and read by the depth inversion.
                        amrex::Real total_stp = amrex::Real(0.0);
                        RreaRunChargedEventInterleave(
                            subcycle_begin,
                            subcycle_end,
                            kinetic_energy_eV,
                            coupling.LowEnergyCutoffEv(),
                            3,
                            kRreaTransientEventCapacity,
                            "positron",
                            [&](amrex::Real frozen_energy_eV,
                                amrex::Real tau_begin_fraction,
                                amrex::Real tau_end_fraction) {
                                auto const stp = RreaChargedCompetingStpRates(
                                    tables, /*is_positron=*/true,
                                    bhabha_secondary_threshold_eV,
                                    coupling.PhotonCutoffEv(),
                                    frozen_energy_eV);
                                total_stp = stp[0] + stp[1] + stp[2];
                                amrex::Real const segment_density_path_m = density_path_m(
                                    material_intervals, transport_path,
                                    tau_begin_fraction, tau_end_fraction);
                                return std::array<amrex::Real, 3>{
                                    stp[0] * segment_density_path_m,
                                    stp[1] * segment_density_path_m,
                                    stp[2] * segment_density_path_m};
                            },
                            [&](std::uint64_t draw_event_ordinal) {
                                auto const live_event_index = RreaTryLiveHazardRngIndex(
                                    RreaSecondarySpecies::Positron,
                                    subcycle, draw_event_ordinal);
                                if (!live_event_index.valid) {
                                    rrea::KineticFail(
                                        "RREA positron repeated-hazard RNG index overflow; "
                                        "the uncommitted transaction is rejected");
                                }
                                interaction_rng_index = live_event_index.value;
                                RreaRngChannel const optical_depth_channel =
                                    draw_event_ordinal == 0U
                                    ? RreaRngChannel::PositronCompetingOpticalDepth
                                    : RreaRngChannel::PositronContinuationOpticalDepth;
                                RreaRngChannel const competing_channel =
                                    draw_event_ordinal == 0U
                                    ? RreaRngChannel::PositronCompetingChannel
                                    : RreaRngChannel::PositronContinuationChannel;
                                return std::pair<amrex::Real, amrex::Real>{
                                    RreaRng::Uniform01(RreaRngKey{
                                                coupling.RngSeed(),
                                                rng_particle_id,
                                                static_cast<std::uint64_t>(rrea::kinetic::max(step, 0)),
                                                interaction_rng_index,
                                                static_cast<std::uint64_t>(optical_depth_channel)}),
                                    RreaRng::Uniform01(RreaRngKey{
                                                coupling.RngSeed(),
                                                rng_particle_id,
                                                static_cast<std::uint64_t>(rrea::kinetic::max(step, 0)),
                                                interaction_rng_index,
                                                static_cast<std::uint64_t>(competing_channel)})};
                            },
                            [&](amrex::Real sampled_optical_depth,
                                amrex::Real invert_begin_fraction,
                                amrex::Real invert_end_fraction) {
                                return fraction_at_density_path_m(
                                    material_intervals,
                                    transport_path,
                                    invert_begin_fraction,
                                    invert_end_fraction,
                                    sampled_optical_depth / total_stp);
                            },
                            [&](amrex::Real segment_begin_fraction,
                                amrex::Real segment_end_fraction,
                                amrex::Real frozen_energy_eV) {
                                return apply_positron_continuous_segment(
                                    segment_begin_fraction,
                                    segment_end_fraction,
                                    frozen_energy_eV);
                            },
                            [&](amrex::Real path_fraction) {
                                mark_positron_stopped_at(path_fraction);
                            },
                            [&](amrex::Real point_event_fraction) {
                                position_at_fraction(
                                    point_event_fraction,
                                    event_x,
                                    event_y,
                                    event_z,
                                    event_r);
                            },
                            [&](int channel, amrex::Real channel_energy_eV) {
                                return RreaChargedChannelOpenAt(
                                    tables, /*is_positron=*/true,
                                    bhabha_secondary_threshold_eV,
                                    coupling.PhotonCutoffEv(),
                                    channel, channel_energy_eV);
                            },
                            [&](int channel, amrex::Real event_fraction) {
                                RreaInteractionSite const site{
                                    event_x, event_y, event_z, event_r,
                                    elapsed_fraction_at(event_fraction)};
                                auto rng_key = [&](RreaRngChannel rng_channel) {
                                    return RreaRngKey{
                                        coupling.RngSeed(),
                                        rng_particle_id,
                                        static_cast<std::uint64_t>(
                                            rrea::kinetic::max(step, 0)),
                                        interaction_rng_index,
                                        static_cast<std::uint64_t>(rng_channel)};
                                };
                                if (channel == 0) {
                                    // A sampled secondary below the production
                                    // threshold discards only this Bhabha
                                    // event; the other channels stay eligible.
                                    // The outgoing positron need not stay above
                                    // the transport cutoff -- its later at-rest
                                    // annihilation owns that branch.
                                    RreaApplyBhabhaFinalState(
                                        fx, tables, rng_key, site,
                                        particle_weight,
                                        bhabha_secondary_threshold_eV,
                                        kinetic_energy_eV,
                                        dir_x, dir_y, dir_z);
                                } else if (channel == 1) {
                                    RreaApplyBremsFinalState(
                                        fx, tables, rng_key,
                                        [&](amrex::Real local_energy_eV) {
                                            deposit_local_energy_as_ion_pairs(
                                                fx, event_r, event_z,
                                                particle_weight,
                                                local_energy_eV);
                                        },
                                        site, particle_weight,
                                        /*charge_sign=*/amrex::Real(1.0),
                                        coupling.PhotonCutoffEv(),
                                        kinetic_energy_eV,
                                        dir_x, dir_y, dir_z);
                                } else if (channel == 2) {
                                    fx.BeginGroup(
                                        RreaSecondaryProcess::PositronAnnihilation,
                                        true);
                                    spawn_positron_annihilation(
                                        fx,
                                        event_x, event_y, event_z,
                                        event_r, event_z,
                                        dir_x, dir_y, dir_z,
                                        particle_weight,
                                        kinetic_energy_eV,
                                        false,
                                        rng_particle_id,
                                        interaction_rng_index,
                                        site.elapsed_step_fraction);
                                    fx.SetInteractionConservationClosure(
                                        particle_weight,
                                        kinetic_energy_eV,
                                        amrex::Real(0.0));
                                    // Net chord: captured start to the sampled
                                    // in-flight annihilation point.
                                    fx.AccumulateChargedSegment(
                                        0, r_particle, z_particle,
                                        event_r, event_z,
                                        particle_weight,
                                        kRreaSegmentChannelAnnihilated);
                                    fx.EndGroup();
                                    annihilated = true;
                                    return false;
                                }
                                return true;
                            });
                        },
                        [&] { return stopped_before_trailing_half || annihilated; },
                        [&](std::uint64_t subcycle,
                            amrex::Real subcycle_begin,
                            amrex::Real subcycle_end) {
                            return finish_interleaved_substep(
                                subcycle, subcycle_begin, subcycle_end,
                                substep_field, substep_dt_s);
                        },
                        // Commit this microchord: the next one is rebuilt from here, along
                        // whatever direction the scattering above left behind.  A refused
                        // endpoint is one the next chord would have been built from while
                        // lying outside the half-open domain, so it becomes an escape.
                        [&](amrex::Real subcycle_end) {
                            return RreaAdvanceRunningChargedPosition(
                                transport_path,
                                subcycle_end,
                                geom.ProbLo(0),
                                geom.ProbHi(0),
                                geom.ProbLo(1),
                                geom.ProbHi(1),
                                run_x,
                                run_y,
                                run_z);
                        },
                        [&] { return transport_path.exits_domain; });
                    if (exited_domain) {
                        charged_parent_escape =
                            RreaClippedTransportEscapePolicyForPath(
                                transport_path,
                                std::hypot(
                                    transport_path.end_x_m,
                                    transport_path.end_y_m),
                                transport_path.end_z_m);
                    }
                    fx.tally.RecordPositronTransport(particle_weight, total_loss_eV, false);
                    if (annihilated) {
                        particle_id.make_invalid();
                        fx.invalidated = true;
                        continue;
                    }
                    // A microchord leaving the domain takes precedence over an
                    // in-domain stop and annihilation.
                    auto const terminal_disposition =
                        RreaResolveChargedTransportTerminal(
                            exited_domain,
                            stopped_before_trailing_half,
                            RreaChargedTerminalDisposition::PositronAnnihilation,
                            /*positron=*/true,
                            kinetic_energy_eV < coupling.LowEnergyCutoffEv(),
                            /*electron_directional_demotion=*/false);
                    if (terminal_disposition
                        == RreaChargedTerminalDisposition::Escape) {
                        // A sampled in-flight annihilation above has already
                        // won if it occurred inside the chord.  Otherwise the
                        // residual positron reaches the absorbing face before
                        // any endpoint-only at-rest annihilation can run.  The
                        // exact-face/overshoot ownership split is applied by
                        // commit_charged_escape below.
                        commit_charged_escape();
                        continue;
                    }
                    if (terminal_disposition
                        == RreaChargedTerminalDisposition::PositronAnnihilation) {
                        // The non-stopped branch annihilates where the positron
                        // actually ended up, which is the running point -- NOT the
                        // pre-push point and not WarpX's straight-line endpoint.
                        amrex::Real const annihilation_x_m =
                            stopped_before_trailing_half
                            ? stopping_x_m : run_x;
                        amrex::Real const annihilation_y_m =
                            stopped_before_trailing_half
                            ? stopping_y_m : run_y;
                        amrex::Real const annihilation_z_m =
                            stopped_before_trailing_half
                            ? stopping_z_m : run_z;
                        amrex::Real const annihilation_r_m =
                            stopped_before_trailing_half
                            ? stopping_r_m : std::hypot(run_x, run_y);
                        fx.BeginGroup(RreaSecondaryProcess::PositronAnnihilation, true);
                        spawn_positron_annihilation(
                            fx,
                            annihilation_x_m,
                            annihilation_y_m,
                            annihilation_z_m,
                            annihilation_r_m,
                            annihilation_z_m,
                            dir_x,
                            dir_y,
                            dir_z,
                            particle_weight,
                            kinetic_energy_eV,
                            true,
                            rng_particle_id,
                            subcycles,
                            stopped_before_trailing_half
                                ? stopping_elapsed_fraction
                                : amrex::Real(1.0));
                        fx.tally.RecordPositronTransport(particle_weight, kinetic_energy_eV, true);
                        fx.SetInteractionConservationClosure(
                            particle_weight,
                            kinetic_energy_eV,
                            amrex::Real(0.0));
                        // Net chord: captured start to the annihilation point
                        // (mid-chord stop point or the running endpoint).
                        fx.AccumulateChargedSegment(
                            0, r_particle, z_particle,
                            annihilation_r_m, annihilation_z_m,
                            particle_weight, kRreaSegmentChannelAnnihilated);
                        fx.EndGroup();
                        particle_id.make_invalid();
                        fx.invalidated = true;
                        continue;
                    }
                    CommitSurvivorChargedWriteback(
                        fx, soa, ip,
                        kinetic_energy_eV, dir_x, dir_y, dir_z,
                        run_x, run_y, run_z,
                        r_particle, z_particle,
                        std::hypot(run_x, run_y),
                        /*signed_mesh_weight=*/particle_weight);

        } while(false);
    }
    template<class Effects,class Id>
    RREA_HOST_DEVICE void Photon(Effects& fx,ChargedSoAView const& soa,long ip,Id particle_id,int particle_cpu) const {
        auto* const r_data=soa.r;
        auto* const z_data=soa.z;
        auto* const theta_data=soa.theta;
        auto* const ux_data=soa.ux;
        auto* const uy_data=soa.uy;
        auto* const uz_data=soa.uz;
        auto* const w_data=soa.w;
        auto* const previous_r_data=soa.previous_r;
        auto* const previous_z_data=soa.previous_z;
        auto* const previous_theta_data=soa.previous_theta;
        do {

                if (!particle_id.is_valid()) {
                    continue;
                }
                    std::uint64_t transaction_parent_key =
                        static_cast<std::uint64_t>(rrea::kinetic::abs_id(static_cast<amrex::Long>(particle_id)))
                        ^ (static_cast<std::uint64_t>(particle_cpu) << 40);
                    fx.SetParentContext(
                        transaction_parent_key, RreaSecondarySpecies::Photon);
                    amrex::Real const particle_weight = w_data[ip];
                    amrex::Real photon_energy_eV =
                        std::sqrt(
                            ux_data[ip] * ux_data[ip]
                            + uy_data[ip] * uy_data[ip]
                            + uz_data[ip] * uz_data[ip]);
                    amrex::Real dir_x = ux_data[ip];
                    amrex::Real dir_y = uy_data[ip];
                    amrex::Real dir_z = uz_data[ip];
                    RreaNormalizeDirection(dir_x, dir_y, dir_z);
                    // Transport precedes WarpX absorbing-boundary deletion. Use
                    // only the captured chord's in-domain part for optical depth
                    // and deposits.
                    if (previous_r_data[ip] > geom.ProbHi(0)
                            + amrex::Real(1.0e-9) * geom.ProbHi(0)
                        || previous_z_data[ip] < geom.ProbLo(1)
                            - amrex::Real(1.0e-9)
                        || previous_z_data[ip] > geom.ProbHi(1)
                            + amrex::Real(1.0e-9) * geom.ProbHi(1)) {
                        rrea::KineticFail("RREA photon survived outside the domain: prev_r=");
                    }
                    RreaTransportPathSegment transport_path = ClipRzTransportPath(
                        previous_r_data[ip],
                        previous_theta_data[ip],
                        previous_z_data[ip],
                        r_data[ip],
                        theta_data[ip],
                        z_data[ip],
                        geom.ProbLo(0),
                        geom.ProbHi(0),
                        geom.ProbLo(1),
                        geom.ProbHi(1),
                        periodic_z);
                    std::uint64_t photon_event_ordinal = 0;
                    amrex::Real elapsed_path_start_fraction = amrex::Real(0.0);
                    static constexpr amrex::Real light_speed_m_per_s =
                        kRreaSpeedOfLightMPerS;
                    while (true) {
                    if (photon_event_ordinal >= kRreaTransientEventCapacity) {
                        rrea::KineticFail(
                            "RREA photon repeated-hazard event ordinal overflow; "
                            "the uncommitted transaction is rejected");
                    }
                    RreaEndpointEscapePolicy const endpoint_escape =
                        RreaClippedTransportEscapePolicyForPath(
                            transport_path,
                            std::hypot(
                                transport_path.end_x_m,
                                transport_path.end_y_m),
                            transport_path.end_z_m);
                    amrex::Real const ds_m = transport_path.in_domain_length_m;
                    // Same material path the charged loops use.  The
                    // piecewise-mean chi places the interaction point with an
                    // error of at most h^2/8H = 9 um on a 0.75 m chord -- far
                    // inside the cell the deposit lands in.
                    auto const photon_intervals = build_material_path_intervals(
                        coupling, transport_path, density_ratio);
                    amrex::Real const chord_density_path_m = density_path_m(
                        photon_intervals,
                        transport_path,
                        transport_path.in_domain_begin_fraction,
                        transport_path.in_domain_end_fraction);
                    if (particle_weight <= amrex::Real(0.0)
                        || photon_energy_eV < coupling.PhotonCutoffEv()) {
                        fx.BeginGroup(RreaSecondaryProcess::LowEnergyDemotion, true);
                        amrex::Real x_deposit = amrex::Real(0.0);
                        amrex::Real y_deposit = amrex::Real(0.0);
                        amrex::Real z_deposit = amrex::Real(0.0);
                        // A photon already below the production cutoff does not
                        // propagate.  Deposit at its captured path start, never
                        // at a clipped absorbing endpoint that lies on the
                        // excluded upper face.
                        transport_path.PointAtFullPathFraction(
                            transport_path.in_domain_begin_fraction,
                            x_deposit,
                            y_deposit,
                            z_deposit);
                        deposit_local_energy_as_ion_pairs(fx,
                            std::hypot(x_deposit, y_deposit),
                            z_deposit,
                            rrea::kinetic::max(particle_weight, amrex::Real(0.0)),
                            photon_energy_eV);
                        if (particle_weight > amrex::Real(0.0)) {
                            fx.SetInteractionConservationClosure(
                                particle_weight,
                                photon_energy_eV,
                                amrex::Real(0.0));
                        } else {
                            fx.SetExplicitZeroConservationClosure();
                        }
                        fx.EndGroup();
                        particle_id.make_invalid();
                        fx.invalidated = true;
                        break;
                    }

                    // The parent key already folds the particle's stored birth-rank
                    // cpu() into the id, so per-rank id reuse cannot collide streams.
                    std::uint64_t const rng_particle_id = transaction_parent_key;
                    auto commit_no_event_photon_escape = [&]() {
                        amrex::Real endpoint_x = amrex::Real(0.0);
                        amrex::Real endpoint_y = amrex::Real(0.0);
                        amrex::Real endpoint_z = amrex::Real(0.0);
                        transport_path.PointAtFullPathFraction(
                            transport_path.in_domain_end_fraction,
                            endpoint_x, endpoint_y, endpoint_z);
                        r_data[ip] = static_cast<amrex::ParticleReal>(
                            std::hypot(endpoint_x, endpoint_y));
                        theta_data[ip] = static_cast<amrex::ParticleReal>(
                            std::atan2(endpoint_y, endpoint_x));
                        z_data[ip] = static_cast<amrex::ParticleReal>(
                            transport_path.WrapZ(endpoint_z));
                        if (!endpoint_escape.escapes) {
                            return;
                        }
                        fx.BeginGroup(
                            RreaSecondaryProcess::PhotonEscape,
                            /*remove_parent=*/true,
                            endpoint_escape.ParentStillValidAtPreflight());
                        fx.tally.RecordPhotonEscape(particle_weight, photon_energy_eV);
                        fx.AddEscapedConservationEnergy(
                            particle_weight, photon_energy_eV);
                        fx.SetInteractionConservationClosure(
                            particle_weight,
                            photon_energy_eV,
                            amrex::Real(0.0));
                        fx.EndGroup();
                        if (endpoint_escape.invalidate_parent_in_rrea) {
                            // Make the clipped recorded escape
                            // atomic with parent removal before the alive-energy
                            // diagnostic and collective cap preflight.
                            particle_id.make_invalid();
                            fx.invalidated = true;
                        }
                    };
                    if (ds_m <= amrex::Real(0.0)) {
                        commit_no_event_photon_escape();
                        break;
                    }
                    amrex::Real const photon_mfp_stp =
                        tables.PhotonFeedbackMeanFreePath(
                            photon_energy_eV,
                            amrex::Real(1.0));
                    amrex::Real const photon_tau =
                        chord_density_path_m / photon_mfp_stp;
                    // Each pass samples one exact exponential optical depth on
                    // the currently remaining chord.  A surviving Compton
                    // state rebuilds its rates and returns through this loop;
                    // terminal photoelectric/pair states never do.
                    fx.tally.RecordTransportGuardMetrics(
                        amrex::Real(0.0),
                        amrex::Real(0.0),
                        amrex::Real(0.0),
                        photon_tau,
                        amrex::Real(0.0),
                        // One exact exponential optical depth is sampled per pass.
                        1U,
                        amrex::Real(0.0),
                        photon_energy_eV,
                        amrex::Real(0.0),
                        false);
                    // Sample one exponential optical depth over the complete
                    // clipped chord.  This is independent of the diagnostic
                    // subcycle partition and is exact for the step's sampled
                    // density/rate, unlike repeated Bernoulli trials followed by
                    // a uniform-within-substep interaction point.
                    auto const live_event_index = RreaTryLiveHazardRngIndex(
                        RreaSecondarySpecies::Photon,
                        0U, photon_event_ordinal);
                    if (!live_event_index.valid) {
                        rrea::KineticFail(
                            "RREA photon repeated-hazard RNG index overflow; "
                            "the uncommitted transaction is rejected");
                    }
                    std::uint64_t const interaction_rng_index =
                        live_event_index.value;
                    RreaRngChannel const optical_depth_channel =
                        photon_event_ordinal == 0U
                        ? RreaRngChannel::PhotonInteraction
                        : RreaRngChannel::PhotonContinuationOpticalDepth;
                    amrex::Real const interact_draw = RreaRng::Uniform01(RreaRngKey{
                        coupling.RngSeed(),
                        rng_particle_id,
                        static_cast<std::uint64_t>(rrea::kinetic::max(step, 0)),
                        interaction_rng_index,
                        static_cast<std::uint64_t>(optical_depth_channel)});
                    RreaRemainingSegmentHazard const hazard =
                        RreaResolveRemainingSegmentHazard(
                            {photon_tau, amrex::Real(0.0), amrex::Real(0.0)},
                            1,
                            transport_path.in_domain_begin_fraction,
                            transport_path.in_domain_end_fraction,
                            interact_draw,
                            amrex::Real(0.0),
                            [&](amrex::Real sampled_optical_depth) {
                                return fraction_at_density_path_m(
                                    photon_intervals,
                                    transport_path,
                                    transport_path.in_domain_begin_fraction,
                                    transport_path.in_domain_end_fraction,
                                    sampled_optical_depth * photon_mfp_stp);
                            });
                    if (!hazard.selection.event) {
                        commit_no_event_photon_escape();
                        break;
                    }
                    amrex::Real const full_path_fraction =
                        hazard.event_fraction;
                    amrex::Real const interaction_distance_m =
                        transport_path.full_length_m
                        * (full_path_fraction
                            - transport_path.in_domain_begin_fraction);
                    amrex::Real x_particle = amrex::Real(0.0);
                    amrex::Real y_particle = amrex::Real(0.0);
                    amrex::Real z_particle = amrex::Real(0.0);
                    transport_path.PointAtFullPathFraction(
                        full_path_fraction, x_particle, y_particle, z_particle);
                    // A scatter changes whether the photon would have reached
                    // its pre-scatter absorbing-boundary intersection. Preserve the
                    // unused geometric distance from the complete WarpX push,
                    // then clip the new chord independently below.  Using the
                    // pre-scatter in-domain remainder would shorten inward-scattered
                    // photons merely because their unscattered trajectory
                    // would have escaped.
                    amrex::Real const remaining_m = rrea::kinetic::max(
                        amrex::Real(0.0),
                        transport_path.full_length_m - interaction_distance_m);
                    amrex::Real const birth_elapsed_step_fraction = rrea::kinetic::min(
                        amrex::Real(1.0),
                        elapsed_path_start_fraction
                            + interaction_distance_m
                                / rrea::kinetic::max(
                                    light_speed_m_per_s * dt_s,
                                    std::numeric_limits<amrex::Real>::min()));
                    bool repeat_compton_hazard = false;
                    RreaTransportPathSegment next_compton_path;
                    amrex::Real next_compton_elapsed_fraction =
                        birth_elapsed_step_fraction;
                    amrex::Real const r_particle = std::hypot(x_particle, y_particle);
                    RreaRngChannel const photon_channel_rng =
                        photon_event_ordinal == 0U
                        ? RreaRngChannel::PhotonChannel
                        : RreaRngChannel::PhotonContinuationChannel;
                    RreaPhotonChannel const channel =
                        tables.SamplePhotonChannel(
                            photon_energy_eV,
                                RreaRng::Uniform01(RreaRngKey{
                                    coupling.RngSeed(),
                                    rng_particle_id,
                                    static_cast<std::uint64_t>(rrea::kinetic::max(step, 0)),
                                    interaction_rng_index,
                                    static_cast<std::uint64_t>(photon_channel_rng)}));
                    RreaInteractionSite const site{
                        x_particle, y_particle, z_particle, r_particle,
                        birth_elapsed_step_fraction};
                    auto rng_key = [&](RreaRngChannel rng_channel) {
                        return RreaRngKey{
                            coupling.RngSeed(),
                            rng_particle_id,
                            static_cast<std::uint64_t>(rrea::kinetic::max(step, 0)),
                            interaction_rng_index,
                            static_cast<std::uint64_t>(rng_channel)};
                    };
                    auto deposit_local = [&](amrex::Real local_energy_eV) {
                        deposit_local_energy_as_ion_pairs(
                            fx, r_particle, z_particle,
                            particle_weight, local_energy_eV);
                    };
                    if (channel == RreaPhotonChannel::Compton) {
                        if (RreaApplyComptonFinalState(
                                fx, tables, rng_key, deposit_local, site,
                                particle_weight,
                                coupling.LowEnergyCutoffEv(),
                                coupling.PhotonCutoffEv(),
                                photon_energy_eV, dir_x, dir_y, dir_z)) {
                            ux_data[ip] = static_cast<amrex::ParticleReal>(photon_energy_eV * dir_x);
                            uy_data[ip] = static_cast<amrex::ParticleReal>(photon_energy_eV * dir_y);
                            uz_data[ip] = static_cast<amrex::ParticleReal>(photon_energy_eV * dir_z);
                            // The scattered photon interacted mid-flight, so
                            // it continues from the interaction point along the new
                            // direction for the remaining path.  Write back the RZ
                            // (r, theta, z) position; the move can cross a tile/box
                            // boundary, so flag a collective Redistribute after the loop.
                            amrex::Real const x_target =
                                x_particle + dir_x * remaining_m;
                            amrex::Real const y_target =
                                y_particle + dir_y * remaining_m;
                            amrex::Real const z_target =
                                z_particle + dir_z * remaining_m;
                            RreaTransportPathSegment const continuation_path =
                                ClipRzTransportPath(
                                    r_particle,
                                    std::atan2(y_particle, x_particle),
                                    z_particle,
                                    std::hypot(x_target, y_target),
                                    std::atan2(y_target, x_target),
                                    z_target,
                                    geom.ProbLo(0),
                                    geom.ProbHi(0),
                                    geom.ProbLo(1),
                                    geom.ProbHi(1),
                                    periodic_z);
                            amrex::Real x_new = amrex::Real(0.0);
                            amrex::Real y_new = amrex::Real(0.0);
                            amrex::Real z_new = amrex::Real(0.0);
                            continuation_path.PointAtFullPathFraction(
                                continuation_path.in_domain_end_fraction,
                                x_new, y_new, z_new);
                            r_data[ip] = static_cast<amrex::ParticleReal>(
                                std::hypot(x_new, y_new));
                            theta_data[ip] = static_cast<amrex::ParticleReal>(
                                std::atan2(y_new, x_new));
                            z_data[ip] = static_cast<amrex::ParticleReal>(
                                continuation_path.WrapZ(z_new));
                            // The scattered photon remains transiently live
                            // over the complete unused time.  Do not classify
                            // an absorbing-boundary endpoint until the
                            // continuation hazard has been sampled over its
                            // in-domain prefix.
                            next_compton_path = continuation_path;
                            repeat_compton_hazard = remaining_m > amrex::Real(0.0);
                            fx.repositioned = true;
                        } else {
                            particle_id.make_invalid();
                            fx.invalidated = true;
                        }
                    } else if (channel == RreaPhotonChannel::Photoelectric) {
                        RreaApplyPhotoelectricFinalState(
                            fx, tables, rng_key, deposit_local, site,
                            particle_weight, coupling.LowEnergyCutoffEv(),
                            photon_energy_eV, dir_x, dir_y, dir_z);
                        particle_id.make_invalid();
                        fx.invalidated = true;
                    } else if (channel == RreaPhotonChannel::PairNuclear
                               || channel == RreaPhotonChannel::PairTriplet) {
                        RreaApplyPairFinalState(
                            fx, tables, rng_key, deposit_local,
                            [&](amrex::Real positron_dir_x,
                                amrex::Real positron_dir_y,
                                amrex::Real positron_dir_z,
                                amrex::Real positron_kinetic_eV) {
                                spawn_positron_annihilation(
                                    fx,
                                    x_particle, y_particle, z_particle,
                                    r_particle, z_particle,
                                    positron_dir_x,
                                    positron_dir_y,
                                    positron_dir_z,
                                    particle_weight,
                                    positron_kinetic_eV,
                                    true,
                                    rng_particle_id,
                                    interaction_rng_index,
                                    birth_elapsed_step_fraction);
                            },
                            site, particle_weight,
                            coupling.LowEnergyCutoffEv(),
                            photon_energy_eV, channel,
                            dir_x, dir_y, dir_z);
                        particle_id.make_invalid();
                        fx.invalidated = true;
                    }
                    if (repeat_compton_hazard) {
                        transport_path = std::move(next_compton_path);
                        elapsed_path_start_fraction =
                            next_compton_elapsed_fraction;
                        ++photon_event_ordinal;
                        continue;
                    }
                    break;
                    }
                    // Photon terminal/no-event disposition is complete, or a
                    // Compton survivor exhausted the residual outer-step time.
                    continue;

        } while(false);
    }
};
static_assert(std::is_trivially_copyable_v<KineticStep>,
              "Kinetic kernel context must not own host containers");
static_assert(std::is_trivially_copyable_v<ChargedSoAView>);
} // namespace rrea::warpx
#endif
