#ifndef WARPX_RREA_WARPX_COUPLING_H
#define WARPX_RREA_WARPX_COUPLING_H

#include "RreaCheckpointRankState.H"
#include "RreaParticleInteraction.H"
#include "RreaSolvePurpose.H"
#include "RreaManagedSpeciesContract.H"
#include "RreaKineticEnvironment.H"

#include "rrea/RreaAmrexAdvance.H"
#include "rrea/RreaAltitudeProfile.H"
#include "rrea/RreaConstants.H"
#include "rrea/RreaEventAccumulator.H"
#include "rrea/RreaEventLedger.H"
#include "rrea/RreaElectronCreation.H"
#include "rrea/RreaInteractionTables.H"
#include "rrea/RreaPlaneFlux.H"
#include "rrea/RreaSeedSchedule.H"
#include "rrea/RreaSegmentDepositionRZ.H"

#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <cstdint>
#include <cstddef>
#include <array>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

class WarpX;

namespace rrea::warpx {

struct RreaSecondaryParticle;



struct RreaVideoFrameRequest {
    int capture_index = 0;
    int step = 0;
    amrex::Real target_time_s = amrex::Real(0.0);
};

class RreaWarpXCoupling {
public:
    static RreaWarpXCoupling& GetInstance();
    [[nodiscard]] rrea::RreaEventLedger& Ledger() noexcept { return m_ledger; }
    [[nodiscard]] rrea::RreaEventLedger const& Ledger() const noexcept { return m_ledger; }

    bool Enabled() const noexcept;
    [[nodiscard]] KineticEnvironment KineticView() const;
    operator KineticEnvironment() const {return KineticView();}

    void InitFromWarpX(WarpX& warpx);

    void InjectScheduledSeeds(
        WarpX& warpx,
        int step,
        amrex::Real time_s,
        amrex::Real dt_s);

    void CaptureRreaPositionsBeforePush(WarpX& warpx, int step);

    void AfterParticlePushBeforeBoundaries(
        WarpX& warpx,
        int step,
        amrex::Real time_s,
        amrex::Real dt_s);

    void AccumulateIonizationEvent(
        amrex::Real r_m,
        amrex::Real z_m,
        amrex::Real event_weight,
        amrex::Real secondary_energy_eV);
    void AccumulatePositiveIonSourceOnly(
        amrex::Real r_m, amrex::Real z_m, amrex::Real ion_weight);
    void AccumulateLowEnergyElectron(
        amrex::Real r_m, amrex::Real z_m, amrex::Real electron_weight);
    // Replay target for one net-chord segment (signed charge in
    // elementary charges times macro weight; channels are the
    // kRreaSegmentChannel* bits).  Deposits the segment's Jr/Jz index-space
    // flux (no rho) into the per-step scratch grid consumed by Maxwell.
    void AccumulateChargedSegment(
        amrex::Real r0_m,
        amrex::Real z0_m,
        amrex::Real r1_m,
        amrex::Real z1_m,
        amrex::Real charge_e_weight,
        std::uint64_t channels);
    void AccumulateChargedEscape(
        amrex::Real trajectory_start_r_m,
        amrex::Real trajectory_start_z_m,
        amrex::Real final_microchord_start_r_m,
        amrex::Real final_microchord_start_z_m,
        amrex::Real crossing_r_m,
        amrex::Real crossing_z_m,
        amrex::Real charge_e_weight,
        std::uint64_t channels,
        std::uint32_t faces);

    // The ONE AddNParticles batch commit (species policy = container name,
    // ceiling, momentum mapping, counters); CreateLocalSecondaries is the
    // species adapter over it.
    void CommitLocalParticleBatch(
        WarpX& warpx,
        std::string const& species_name,
        amrex::Long max_macros,
        char const* ceiling_context,
        bool photon_momentum,
        std::uint64_t& created_count,
        amrex::Real& created_weight,
        std::vector<amrex::Real> const& x_m,
        std::vector<amrex::Real> const& y_m,
        std::vector<amrex::Real> const& z_m,
        std::vector<amrex::Real> const& ux,
        std::vector<amrex::Real> const& uy,
        std::vector<amrex::Real> const& uz,
        std::vector<amrex::Real> const& weight,
        std::vector<amrex::Real> const& energy_eV,
        int step,
        bool population_ceiling_preflighted,
        std::vector<amrex::Real> const& counted);

    void CreateLocalSecondaries(
        WarpX& warpx,
        RreaSecondarySpecies species,
        std::vector<amrex::Real> const& x_m,
        std::vector<amrex::Real> const& y_m,
        std::vector<amrex::Real> const& z_m,
        std::vector<amrex::Real> const& ux,
        std::vector<amrex::Real> const& uy,
        std::vector<amrex::Real> const& uz,
        std::vector<amrex::Real> const& weight,
        std::vector<amrex::Real> const& energy_eV,
        int step = 0,
        bool population_ceiling_preflighted = false,
        std::vector<amrex::Real> const& counted = {});

    void AccumulateElectronCreation(amrex::Real r, amrex::Real z,
                                   amrex::Real weight, bool injected) {
        m_electron_creation.Add(r, z, weight, injected);
    }

    void PreflightSecondaryBatch(
        WarpX& warpx,
        amrex::Long local_electron_candidates,
        amrex::Long local_photon_candidates,
        amrex::Long local_positron_candidates,
        amrex::Long local_electron_removals,
        amrex::Long local_photon_removals,
        amrex::Long local_positron_removals,
        char const* origin,
        int step) const;
    void PreflightAndApplySecondaryPopulationControl(
        WarpX& warpx,
        std::vector<RreaSecondaryParticle>& candidates,
        std::array<std::vector<std::uint64_t>, 3> const& removed_parent_keys,
        std::array<amrex::Long, 3> const& local_parent_removals,
        char const* origin,
        int step);

    void RecordPhotonLocalDeposit(
        amrex::Real event_weight,
        amrex::Real photon_energy_eV);
    void RecordPositronAnnihilation(
        amrex::Real event_weight,
        amrex::Real local_deposit_energy_eV,
        amrex::Real photon1_energy_eV,
        amrex::Real photon2_energy_eV,
        bool at_rest_or_below_cutoff);

    amrex::Real LowEnergyCutoffEv() const noexcept;
    std::string const& ElectronSpeciesName() const noexcept { return m_seed_species_name; }
    std::string const& PhotonSpeciesName() const noexcept { return m_photon_species_name; }
    std::string const& PositronSpeciesName() const noexcept { return m_positron_species_name; }
    std::uint64_t RngSeed() const noexcept { return m_rng_seed; }
    // OpenMP thread count for the transport particle
    // loops only (0 = follow the global OpenMP setting).  Lets transport run
    // multithreaded while OMP_NUM_THREADS=1 keeps every AMReX kernel serial
    // and deterministic; per-chunk replay fixes the side-effect order.
    int TransportOmpThreads() const noexcept { return m_transport_omp_threads; }
    // Density ratios derived from the absolute profile density; each names the
    // density it is relative to.
    amrex::Real TransportReferenceDensityKgM3() const;
    [[nodiscard]] TransportDensityView TransportDensity() const;
    amrex::Real MobilityDensityRatio(amrex::Real transport_ratio) const;
    amrex::Real ProfileAltitudeMslAtZ(amrex::Real z_m) const noexcept;
    [[nodiscard]] bool CdPlaneDiagnosticsEnabled() const noexcept
    {
        return m_cd_plane_flux.Enabled();
    }
    [[nodiscard]] std::vector<amrex::Real> const& CdPlaneZ() const noexcept
    {
        return m_cd_plane_flux.PlaneZ();
    }
    [[nodiscard]] bool PrepareCdElectronPlaneCrossings(
        std::vector<RreaPlaneFluxCrossing> const& crossings,
        RreaPlaneFluxAccumulator::PreparedUpdate& prepared) const;
    // Called only with a successfully prepared complete batch. The commit is
    // a no-fail swap and therefore remains safe after physical replay starts.
    void CommitPreparedCdElectronPlaneCrossings(
        RreaPlaneFluxAccumulator::PreparedUpdate&& prepared) noexcept;
    void RefreshNativeParticleCounts(WarpX& warpx);

    bool BeforeFieldSolve(
        WarpX& warpx,
        amrex::Real dt_s,
        int step,
        amrex::Real time_s,
        SpaceChargeSolvePurpose purpose);

    void WriteDiagnostics(
        std::string const& dir,
        int step,
        amrex::Real time_s);
    void WriteCheckpoint(std::string const& dir);
    void ReadCheckpoint(std::string const& dir);

private:
    RreaWarpXCoupling() = default;

    void ReadParameters();
    void BuildBindings(WarpX& warpx);
    void RefreshDensityRatioFields();
    void LoadSeedSchedule();
    void ValidateSeedSchedule(WarpX const& warpx) const;
    bool InteractionTablesRequired() const;
    void LoadInteractionTables();
    void LoadPendingCheckpointIfAny();
    void EnsurePreviousPositionComponents(WarpX& warpx);
    void EnforcePopulationCeiling(
        WarpX& warpx,
        std::string const& species_name,
        long ceiling,
        amrex::Long global_candidates,
        char const* origin,
        int step,
        amrex::Long global_removals = 0) const;
    void ApplyAdaptivePopulationControl(WarpX& warpx, int step);
    void AdaptiveResampleOneSpecies(
        WarpX& warpx,
        std::string const& species_name,
        long target);
    // Validation-only end-to-end Maxwell continuity fixture.  It mutates the
    // actual managed WarpX containers and the real fluid/event/current paths;
    // the production driver refuses the enabling flag.
    void InjectActualContinuityFixtureParticles(WarpX& warpx, int step);
    void RunActualContinuityFixtureStep(WarpX& warpx, int step);
    void VerifyActualContinuityFixtureAfterAdvance(WarpX& warpx, int step);
    void DepositParticleChargeToRreaGrid(WarpX& warpx);
    void CopyAdaptersToWarpXFields();
    void WriteMeshFields(std::string const& dir) const;
    void WriteRankState(std::string const& dir) const;
    void ReadRankState(std::string const& dir);
    // Registry values and their fold classes are emitted by ONE visitor per
    // registry, so the capture order and the fold table cannot drift apart;
    // a rank-count fold checks that Replicated rows agree across ranks.
    template <typename Visit>
    void VisitRankIntegerState(Visit&& visit) const;
    template <typename Visit>
    void VisitRankRealState(Visit&& visit) const;
    std::vector<RreaRankStateFoldClass> RankIntegerFoldRules() const;
    std::vector<RreaRankStateFoldClass> RankRealFoldRules() const;
    std::vector<std::uint64_t> CaptureRankIntegerState() const;
    std::vector<amrex::Real> CaptureRankRealState() const;
    // Abort (no opt-in) or record + report a tolerated continuation-epoch
    // deviation.  `description` names the gate and both values.
    void ToleratedRestartEpochDeviationOrAbort(std::string const& description) const;
    bool ArchiveCsvIfHeaderChanged(
        std::string const& path, std::string const& header, int step) const;
    void RestoreRankState(
        std::vector<std::uint64_t> const& integers,
        std::vector<amrex::Real> const& reals);
    void ReadMeshFields(std::string const& dir);
    void WriteMetadata(std::string const& dir) const;
    void ReadMetadata(std::string const& dir);
    void AppendReducedDiagnostics(
        std::string const& dir,
        int step,
        amrex::Real time_s) const;
    void LoadVideoFrameSchedule();
    void MaybeWriteVideoFrame(
        WarpX& warpx,
        int step,
        amrex::Real time_s);
    void MaybeWriteCdPlaneFlux(
        int step,
        amrex::Real time_s,
        bool force);

    bool m_parameters_read = false;
    bool m_initialized = false;
    bool m_enabled = false;
    bool m_previous_position_components_ready = false;
    bool m_restored_from_checkpoint = false;
    bool m_positions_captured_for_step = false;
    bool m_transport_completed_for_step = false;
    int m_positions_captured_step = -1;
    int m_transport_completed_step = -1;
    // Scheduled injection is an explicit external source between Maxwell
    // states.  Re-prime rho^n from an actual WarpX deposit after it occurs,
    // before the newly injected particles move.
    bool m_maxwell_continuity_reprime_required = true;
    bool m_debug_disable_transport = false;
    bool m_debug_disable_checkpoint = false;
    bool m_debug_actual_continuity_fixture = false;
    bool m_actual_continuity_fixture_injected = false;
    int m_actual_continuity_fixture_steps_completed = 0;
    amrex::Real m_profile_active_z_min_m = std::numeric_limits<amrex::Real>::quiet_NaN();
    amrex::Real m_profile_active_z_max_m = std::numeric_limits<amrex::Real>::quiet_NaN();
    amrex::Real m_low_energy_cutoff_eV =
        std::numeric_limits<amrex::Real>::quiet_NaN();
    int m_diag_interval = 0;
    // Mesh diagnostics are inode-heavy: -1 follows diag_interval, 0 disables
    // them while retaining reduced CSV output, and N > 0 sets their cadence.
    int m_mesh_diag_interval = 0;
    bool m_video_diag_enable = false;
    amrex::Real m_electron_creation_bin_m = 0;
    rrea::ElectronCreationMap m_electron_creation;
    int m_video_diag_nr = 0;
    int m_video_diag_nz = 0;
    amrex::Real m_video_diag_r_min_m = -std::numeric_limits<amrex::Real>::infinity();
    amrex::Real m_video_diag_r_max_m = std::numeric_limits<amrex::Real>::infinity();
    amrex::Real m_video_diag_z_min_m = -std::numeric_limits<amrex::Real>::infinity();
    amrex::Real m_video_diag_z_max_m = std::numeric_limits<amrex::Real>::infinity();
    amrex::Real m_video_diag_energy_threshold_eV = amrex::Real(0.0);
    RreaPlaneFluxState m_cd_plane_flux;
    amrex::Real m_cd_plane_core_radius_m = amrex::Real(0.0);
    bool m_cd_plane_stream_initialized = false;
    int m_cd_plane_last_observed_step = -1;
    amrex::Real m_cd_plane_last_observed_time_s =
        -std::numeric_limits<amrex::Real>::infinity();
    // Net-chord current grid padding, shared by its sizing and its consumer.
    static constexpr int kNetChordPadCells = 2;
    std::string m_case_id;
    std::string m_seed_schedule_path;
    std::string m_seed_species_name = "rrea_electrons";
    std::string m_photon_species_name = "rrea_photons";
    std::string m_positron_species_name = "rrea_positrons";
    std::string m_interaction_table_config_path;
    // Direct E/N-dependent mobility/attachment/diffusion table, required
    // exactly when an en_table_* fluid model is selected.
    std::string m_electron_closure_table_path;
    LowEnergyElectronClosure m_electron_closure;
    std::string m_output_dir = "rrea_output";
    std::string m_video_diag_output_dir;
    std::string m_video_diag_frame_schedule_path;
    std::string m_pending_checkpoint_dir;
    // Explicit escalation for an intentional engine, MPI-layout or
    // physics-model transition; both keys are required together.
    // Deviations are recorded (mutable: the gates live in const validators)
    // and emitted into the next checkpoint metadata as provenance.
    bool m_allow_restart_epoch_change = false;
    std::string m_restart_epoch_change_reason;
    mutable std::vector<std::string> m_restart_epoch_deviations;
    bool m_schedule_loaded = false;
    bool m_interaction_tables_loaded = false;
    RreaSeedSchedule m_seed_schedule;
    RreaInteractionTables m_interaction_tables;
    std::size_t m_next_seed_event = 0;
    // The event ledger owns scalar transport counters and their fold methods.
    rrea::RreaEventLedger m_ledger;
    std::uint64_t m_injected_macro_count = 0;
    std::uint64_t m_native_seed_particle_count = 0;
    amrex::Real m_injected_physical_weight = amrex::Real(0.0);
    std::uint64_t m_kinetic_secondary_electron_count = 0;
    std::uint64_t m_photon_created_count = 0;
    amrex::Real m_kinetic_secondary_weight = amrex::Real(0.0);
    amrex::Real m_photon_created_weight = amrex::Real(0.0);
    std::uint64_t m_positron_created_count = 0;
    amrex::Real m_positron_created_weight = amrex::Real(0.0);
    amrex::Real m_initial_photon_alive_energy_eV = amrex::Real(0.0);
    std::uint64_t m_positron_annihilation_count = 0;
    std::uint64_t m_positron_annihilation_at_rest_count = 0;
    std::uint64_t m_positron_annihilation_in_flight_count = 0;
    amrex::Real m_positron_annihilation_weight = amrex::Real(0.0);
    amrex::Real m_positron_annihilation_photon_energy_eV = amrex::Real(0.0);
    amrex::Real m_positron_annihilation_local_deposit_eV = amrex::Real(0.0);
    // Cumulative realized positive-ion creation source.  The z moment lets
    // optical analysis locate creation without differentiating transported
    // ion density.
    amrex::Real m_ion_pair_source_weight = amrex::Real(0.0);
    amrex::Real m_ion_pair_source_z_weighted_m = amrex::Real(0.0);
    std::uint64_t m_rng_seed = 0;
    std::uint64_t m_population_controller_rng_salt = 0;
    std::string m_rng_scheme;
    // See TransportOmpThreads().
    int m_transport_omp_threads = 0;
    amrex::Real m_transport_density_ratio = amrex::Real(1.0);
    mutable rrea::GpuVector<amrex::Real> m_kinetic_plane_z;
    bool m_transport_density_profile_enabled = false;
    amrex::Real m_profile_altitude_msl_at_z0_m = amrex::Real(0.0);
    std::string m_initial_field_mode = "constant";
    // Maxwell TM is the sole RREA field evolution path.
    std::string m_field_profile_path;
    std::string m_density_profile_path;
    RreaAltitudeProfile m_transport_density_profile;
    std::string m_population_ceiling_policy = "strict_abort_v1";
    long m_max_electron_macros = 0;
    long m_max_photon_macros = 0;
    long m_max_positron_macros = 0;
    // adaptive_resample_v1 population control (unbiased per-realization
    // CIC-charge-exact thinning in the secondary transaction plus
    // step-boundary split refill; 0 = disabled for that species).  Strict ceilings above stay as fail-closed backstops.
    long m_population_target_electron_macros = 0;
    long m_population_target_positron_macros = 0;
    // No native policy default: production runners/decks own this value via
    // config/rrea_defaults.json.  Adaptive mode rejects a missing/zero value.
    int m_population_control_interval = 0;
    amrex::Real m_population_split_min_weight = amrex::Real(0.0);
    // Net-chord current: per-step, rank-local scratch, never
    // checkpointed (rebuilt every step, so no mesh-sidecar or rank-state
    // epoch).  The grid is padded by kNetChordPadCells cells beyond the physical domain in
    // +r and both z directions so boundary-adjacent CIC support never clamps
    // (clamping breaks the discrete continuity identity; the axis fold is
    // exact and needs no pad).  rho carries only the escape primitive's
    // endpoint closure for the cloud that leaves the domain.
    rrea::SegmentGridRZ m_netchord_grid;
    int m_netchord_physical_nr = 0;
    int m_netchord_physical_nz = 0;
    int m_netchord_physical_z_begin = 0;
    amrex::Real m_netchord_dt_s = amrex::Real(0.0);
    std::uint64_t m_netchord_segment_count = 0;
    amrex::Real m_netchord_z_displacement_e_m = amrex::Real(0.0);
    amrex::Real m_netchord_created_charge_e = amrex::Real(0.0);
    amrex::Real m_netchord_escaped_charge_e = amrex::Real(0.0);
    amrex::Real m_netchord_demoted_charge_e = amrex::Real(0.0);
    amrex::Real m_netchord_annihilated_charge_e = amrex::Real(0.0);
    void ResetNetChordStep(WarpX& warpx, amrex::Real dt_s);
    // Cumulative adaptive-resample bookkeeping (rank-local; checkpointed).
    std::uint64_t m_resample_killed_count = 0;
    std::uint64_t m_resample_split_count = 0;
    std::uint64_t m_resample_thin_rounds = 0;
    amrex::Real m_resample_killed_weight = amrex::Real(0.0);
    amrex::Real m_resample_boost_weight = amrex::Real(0.0);
    amrex::Real m_resample_split_weight = amrex::Real(0.0);
    amrex::Real m_resample_killed_energy_eV = amrex::Real(0.0);
    amrex::Real m_resample_boost_energy_eV = amrex::Real(0.0);
    std::array<amrex::Long, 2> m_resample_exact_capacity{{0, 0}};
    std::array<amrex::Long, 2> m_resample_geometric_floor{{0, 0}};
    int m_population_control_preapplied_step = -1;
    unsigned m_population_control_preapplied_species = 0;
    std::uint64_t m_video_frame_write_count = 0;

    RreaAmrexAdvanceConfig m_advance_config;
    RreaParticleInteractionConfig m_transport_config;
    std::unique_ptr<RreaAmrexAdvance> m_advance;
    std::unique_ptr<RreaEventAccumulator> m_events;
    std::unique_ptr<RreaParticleInteraction> m_air_interaction;
    // Single level throughout (amr.max_level = 0 is a startup contract), so
    // every field below is level 0.  WarpX's rho_fp is nodal while the RREA
    // solver, the event sources and the fluid are cell-centred, so the
    // cell-centred adapters are unconditional: the engine deposits, solves and
    // gathers on them and CopyAdaptersToWarpXFields interpolates back.
    rrea::RreaAmrexLevelBinding m_level;
    amrex::MultiFab* m_native_er = nullptr;
    amrex::MultiFab* m_native_ez = nullptr;
    std::unique_ptr<amrex::MultiFab> m_rho_adapter;
    std::unique_ptr<amrex::MultiFab> m_er_adapter;
    std::unique_ptr<amrex::MultiFab> m_ez_adapter;
    std::unique_ptr<amrex::MultiFab> m_density_ratio_field;
    // The static z-profile density ratio is
    // refreshed once per (re)binding instead of every step.
    bool m_density_ratio_field_refreshed = false;
    bool m_video_schedule_loaded = false;
    bool m_video_stream_initialized = false;
    std::size_t m_next_video_frame = 0;
    amrex::Vector<RreaVideoFrameRequest> m_video_frames;
};

}  // namespace rrea::warpx

#endif
