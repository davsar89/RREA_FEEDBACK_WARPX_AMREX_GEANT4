#ifndef RREA_INTERACTION_TABLES_H
#define RREA_INTERACTION_TABLES_H

#include "rrea/RreaConstants.H"

#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace rrea {

struct RreaCertifiedEnergyRange {
    amrex::Real min_eV = 0.0;
    amrex::Real max_eV = 0.0;

    [[nodiscard]] bool Contains(amrex::Real energy_eV) const noexcept
    {
        return energy_eV >= min_eV && energy_eV <= max_eV;
    }
};

// Schema-6 process authority.  Inverse lengths are returned at the requested
// density (not at STP), in inverse metres.  Keeping the individual rates
// avoids the interpolation inconsistency caused by separately tabulating a
// total MFP and channel probabilities.
struct RreaPhotonProcessInverseLengths {
    amrex::Real compton_per_m = 0.0;
    amrex::Real photoelectric_per_m = 0.0;
    amrex::Real pair_nuclear_per_m = 0.0;
    amrex::Real pair_triplet_per_m = 0.0;

    [[nodiscard]] amrex::Real TotalPerM() const noexcept
    {
        return compton_per_m + photoelectric_per_m
            + pair_nuclear_per_m + pair_triplet_per_m;
    }
};

struct RreaElectronTransportCoefficients {
    amrex::Real collision_stopping_eV_per_m = 0.0;
    amrex::Real soft_radiative_stopping_eV_per_m = 0.0;
    amrex::Real hard_moller_inverse_m = 0.0;
    amrex::Real tracked_brems_inverse_m = 0.0;
    amrex::Real elastic_transport_inverse_m = 0.0;
};

struct RreaPositronTransportCoefficients {
    amrex::Real collision_stopping_eV_per_m = 0.0;
    amrex::Real soft_radiative_stopping_eV_per_m = 0.0;
    amrex::Real hard_bhabha_inverse_m = 0.0;
    amrex::Real tracked_brems_inverse_m = 0.0;
    amrex::Real elastic_transport_inverse_m = 0.0;
    amrex::Real annihilation_inverse_m = 0.0;
};

// The declared interpolation string, resolved once at load.  The transport
// loops evaluate tables per particle per substep; a string comparison per
// query is not a physics decision that has to be re-taken there.
enum class RreaInterpolationMode { LogLogLinear, LogLogPositiveWithZeroThreshold, LogXLinear, Linear };

struct RreaInteractionTable {
    std::string name;
    std::string path;
    std::string value_column;
    std::string units;
    std::string interpolation = "loglog_linear";
    std::string extrapolation = "forbidden";
    RreaInterpolationMode mode = RreaInterpolationMode::LogLogLinear;
    amrex::Vector<amrex::Real> energy_eV;
    amrex::Vector<amrex::Real> values;
    // log(energy_eV) and log(values) at the grid nodes: the same logarithms
    // Interpolate would take, taken once.  log_values holds 0 where the value
    // is not positive; those nodes only ever feed the linear branch.
    amrex::Vector<amrex::Real> log_energy;
    amrex::Vector<amrex::Real> log_values;

    // Resolves `mode` from `interpolation` and fills the node logarithms.
    // Must be re-run after any in-place edit of `values`.
    void Finalize();
    [[nodiscard]] amrex::Real Interpolate(amrex::Real energy_eV_query) const;
};

// Every table the per-particle transport accessors read, as one id-to-bundle
// -name list so the two can never drift apart.  Load resolves each id to a
// pointer once, and the hot path then indexes an array instead of hashing a
// freshly allocated std::string into a std::map on every query.
#define RREA_HOT_TABLES(X)                                                     \
    X(ElectronCollisionStopping, "electron_collision_stopping_air")            \
    X(ElectronSoftRadiativeStopping, "electron_soft_radiative_stopping_air")   \
    X(ElectronHardMollerInverseLength,                                         \
      "electron_hard_moller_inverse_length_air")                               \
    X(ElectronTrackedBremsInverseLength,                                       \
      "electron_tracked_brems_inverse_length_air")                             \
    X(ElectronElasticTransportInverseLength,                                   \
      "electron_elastic_transport_inverse_length_air")                         \
    X(PositronCollisionStopping, "positron_collision_stopping_air")            \
    X(PositronSoftRadiativeStopping, "positron_soft_radiative_stopping_air")   \
    X(PositronHardBhabhaInverseLength,                                         \
      "positron_hard_bhabha_inverse_length_air")                               \
    X(PositronTrackedBremsInverseLength,                                       \
      "positron_tracked_brems_inverse_length_air")                             \
    X(PositronElasticTransportInverseLength,                                   \
      "positron_elastic_transport_inverse_length_air")                         \
    X(PositronAnnihilationInverseLength,                                       \
      "positron_annihilation_inverse_length_air")                              \
    X(PhotonCompton, "photon_compton_inverse_length_air")                      \
    X(PhotonPhotoelectric, "photon_photoelectric_inverse_length_air")          \
    X(PhotonPairNuclear, "photon_pair_nuclear_inverse_length_air")             \
    X(PhotonPairTriplet, "photon_pair_triplet_inverse_length_air")

enum class RreaTable : int {
#define RREA_TABLE_ENUMERATOR(id, table_name) id,
    RREA_HOT_TABLES(RREA_TABLE_ENUMERATOR)
#undef RREA_TABLE_ENUMERATOR
    Count
};

struct RreaMollerCdfGroup {
    amrex::Real primary_energy_eV = 0.0;
    amrex::Vector<amrex::Real> cdf_u;
    amrex::Vector<amrex::Real> secondary_energy_eV;
};

struct RreaConditionalCdfGroup {
    amrex::Real primary_energy_eV = 0.0;
    amrex::Vector<amrex::Real> cdf_u;
    amrex::Vector<amrex::Real> values;
};

// Fail-closed validation shared by the CSV loader and the focused native
// contract smoke.  Conditional inverse-CDF rows are runtime physics authority,
// so malformed endpoints or sampled values must never be repaired/clamped.
void ValidateRreaConditionalCdfGroups(
    amrex::Vector<RreaConditionalCdfGroup> const& groups,
    std::string const& source_label);

// Goudsmit-Saunderson / Wentzel elastic-scattering tables.
// Per-energy elastic mean free path, effective screening eta (few-collision
// analytic single-scatter branch), and 1/G_1 transport MFP audit anchor.
struct RreaElasticMfpTable {
    amrex::Vector<amrex::Real> energy_eV;
    amrex::Vector<amrex::Real> elastic_mfp_m_stp;
    amrex::Vector<amrex::Real> screening_eta;
    amrex::Vector<amrex::Real> transport_mfp_m_stp;
};

// 2D conditional inverse-CDF cos(theta)(u) over (energy, path n = s/lambda_el).
// Schema 6 uses this common representation for separate option4 GS (low-E)
// and WentzelVI (high-E) branches; they are never blended across the declared
// model transition.
// Stored value is one_minus_cos = 1 - cos, interpolated in log space over
// (log E, log n); index = (ie*n_path + ip)*n_u + iu.
struct RreaGsAngleTable {
    amrex::Vector<amrex::Real> energy_eV;   // sorted grid
    amrex::Vector<amrex::Real> path_n;      // sorted grid (s/lambda_el)
    amrex::Vector<amrex::Real> cdf_u;       // shared CDF nodes (strictly increasing)
    amrex::Vector<amrex::Real> one_minus_cos;  // flattened (energy, path, u)
};

struct RreaComptonFinalState {
    amrex::Real recoil_electron_energy_eV = 0.0;
    amrex::Real scattered_photon_energy_eV = 0.0;
    amrex::Real cos_theta = 1.0;
};

struct RreaPhotoelectricSubshellCdfRow {
    amrex::Real cdf_u = 0.0;
    amrex::Real subshell_probability = 0.0;
    int element_Z = 0;
    std::string element_symbol;
    std::string shell_id;
    amrex::Real binding_energy_eV = 0.0;
    amrex::Real relaxation_energy_eV = 0.0;
};

struct RreaPhotoelectricSubshellCdfGroup {
    amrex::Real primary_energy_eV = 0.0;
    amrex::Vector<RreaPhotoelectricSubshellCdfRow> rows;
};

struct RreaPhotoelectricFinalState {
    amrex::Real photoelectron_energy_eV = 0.0;
    amrex::Real binding_energy_eV = 0.0;
    amrex::Real relaxation_energy_eV = 0.0;
    int element_Z = 0;
    std::string element_symbol;
    std::string shell_id;
};

enum class RreaPhotonChannel {
    Compton,
    Photoelectric,
    PairNuclear,
    PairTriplet,
};

struct RreaPositronAnnihilationFinalState {
    amrex::Real photon1_energy_eV = 0.0;
    amrex::Real photon2_energy_eV = 0.0;
    amrex::Real photon1_cos_theta = 1.0;
    amrex::Real photon1_azimuth_u = 0.0;
    bool valid = false;
    bool at_rest = false;
};

struct RreaPairFinalState {
    bool valid = false;
    RreaPhotonChannel channel = RreaPhotonChannel::PairNuclear;
    amrex::Real electron_energy_eV = 0.0;
    amrex::Real positron_energy_eV = 0.0;
    amrex::Real recoil_electron_energy_eV = 0.0;
    amrex::Real local_deposit_energy_eV = 0.0;
    amrex::Real energy_residual_eV = 0.0;
    amrex::Real electron_dir_x = 0.0;
    amrex::Real electron_dir_y = 0.0;
    amrex::Real electron_dir_z = 1.0;
    amrex::Real positron_dir_x = 0.0;
    amrex::Real positron_dir_y = 0.0;
    amrex::Real positron_dir_z = 1.0;
    amrex::Real recoil_dir_x = 0.0;
    amrex::Real recoil_dir_y = 0.0;
    amrex::Real recoil_dir_z = 1.0;
    bool creates_target_ion = false;
    bool has_recoil_electron = false;
};

struct RreaPairEmpiricalParticle {
    int pdg = 0;
    amrex::Real kinetic_fraction = 0.0;
    amrex::Real dir_x = 0.0;
    amrex::Real dir_y = 0.0;
    amrex::Real dir_z = 1.0;
};

struct RreaPairEmpiricalRecord {
    int target_Z = 0;
    RreaPhotonChannel channel = RreaPhotonChannel::PairNuclear;
    amrex::Real primary_energy_eV = 0.0;
    amrex::Real local_recoil_fraction = 0.0;
    int lepton_count = 0;
    RreaPairEmpiricalParticle particles[3];
};

struct RreaPairEmpiricalBlock {
    RreaPhotonChannel channel = RreaPhotonChannel::PairNuclear;
    amrex::Real primary_energy_eV = 0.0;
    amrex::Vector<RreaPairEmpiricalRecord> training_records;
};

[[nodiscard]] bool RreaPairChannelHasPhaseSpace(
    amrex::Real photon_energy_eV,
    RreaPhotonChannel channel) noexcept;

[[nodiscard]] RreaPairFinalState BuildRreaPairEmpiricalFinalState(
    RreaPairEmpiricalRecord const& record,
    amrex::Real photon_energy_eV,
    RreaPhotonChannel channel,
    amrex::Real draw_pair_azimuth);

class RreaInteractionTables {
public:
    // Only the production schema-6 contract is accepted.
    void Load(std::string const& config_path);

    [[nodiscard]] bool Loaded() const noexcept { return m_loaded; }
    [[nodiscard]] int SchemaVersion() const noexcept { return m_loaded ? 6 : 0; }
    [[nodiscard]] RreaCertifiedEnergyRange ElectronCertifiedEnergyRangeEv() const noexcept
    {
        return m_charged_certified_range;
    }
    [[nodiscard]] RreaCertifiedEnergyRange PositronCertifiedEnergyRangeEv() const noexcept
    {
        return m_charged_certified_range;
    }
    [[nodiscard]] RreaCertifiedEnergyRange PhotonCertifiedEnergyRangeEv() const noexcept
    {
        return m_photon_certified_range;
    }
    [[nodiscard]] std::string const& TableClass() const noexcept { return m_table_class; }
    [[nodiscard]] std::string const& TransportModel() const noexcept { return m_transport_model; }
    [[nodiscard]] amrex::Real GsFewCollisionThreshold() const noexcept { return m_gs_few_collision_threshold; }
    // Minimum tabulated value over the energy window [lo_eV, hi_eV]: grid points
    // inside the window plus the interpolated values at both window edges.
    // Returns false when the table is absent/empty, the window is degenerate,
    // or either edge lies outside the table (no silent clipping). Used by the
    // startup worst-case optical-depth guard.
    [[nodiscard]] bool MinTableValueInRange(
        std::string const& name,
        amrex::Real lo_eV,
        amrex::Real hi_eV,
        amrex::Real& min_value) const noexcept;

    // Native probe API: uses the exact production interpolation implementation
    // and therefore shares its strict, endpoint-inclusive range contract.
    [[nodiscard]] amrex::Real ProbeTableValue(
        std::string const& name, amrex::Real energy_eV) const;
    [[nodiscard]] RreaPhotonProcessInverseLengths PhotonProcessInverseLengths(
        amrex::Real energy_eV, amrex::Real density_ratio) const;
    [[nodiscard]] RreaElectronTransportCoefficients ElectronTransportCoefficients(
        amrex::Real energy_eV, amrex::Real density_ratio) const;
    [[nodiscard]] RreaPositronTransportCoefficients PositronTransportCoefficients(
        amrex::Real energy_eV, amrex::Real density_ratio) const;

    [[nodiscard]] amrex::Real WAirEvPerIonPair() const noexcept
    {
        return static_cast<amrex::Real>(mean_energy_per_ion_pair_air_eV);
    }
    [[nodiscard]] amrex::Real ElectronCollisionLoss(
        amrex::Real energy_eV,
        amrex::Real density_ratio) const;
    [[nodiscard]] amrex::Real ElectronHardMollerMeanFreePath(
        amrex::Real energy_eV,
        amrex::Real density_ratio) const;
    [[nodiscard]] amrex::Real ElectronHardMollerEnergyLoss(
        amrex::Real energy_eV,
        amrex::Real density_ratio) const;
    [[nodiscard]] amrex::Real HardMollerSecondaryThresholdEv() const noexcept
    {
        return m_hard_moller_secondary_threshold_eV;
    }
    // Single owner of the tracked-brems rule: below this a photon is continuous
    // drag, above it an explicit particle.  A flat energy, not a fraction of the
    // parent, so there is one number and nothing restates it.
    [[nodiscard]] amrex::Real BremsAbsoluteCutoffEv() const noexcept
    {
        return m_brems_absolute_cutoff_eV;
    }
    // Air density these tables were generated at.  Every density_ratio argument
    // below means rho(z) divided by THIS, and by nothing else -- the RREA
    // threshold and the reduced ion mobility are quoted at different densities
    // and must not reuse the same ratio.  Load() requires the configuration to
    // declare it.
    [[nodiscard]] amrex::Real ReferenceDensityKgM3() const noexcept
    {
        return m_reference_density_kg_m3;
    }
    [[nodiscard]] amrex::Real ElectronElasticTransportMeanFreePath(
        amrex::Real energy_eV,
        amrex::Real density_ratio) const;
    // --- Elastic scattering (is_positron selects the species table) ---
    [[nodiscard]] amrex::Real ElasticMeanFreePath(
        bool is_positron,
        amrex::Real energy_eV,
        amrex::Real density_ratio) const;
    [[nodiscard]] amrex::Real ScreeningEta(bool is_positron, amrex::Real energy_eV) const;
    // Multiple-scattering cos(theta) draw at path n = s/lambda_el (n >= n_few).
    // Production schema 6 uses option4 GS for E <= transition and its
    // independently tabulated WentzelVI branch for
    // E > transition. Energy queries are strict and never clamped.
    [[nodiscard]] amrex::Real SampleGsCosTheta(
        bool is_positron,
        amrex::Real energy_eV,
        amrex::Real path_n,
        amrex::Real cdf_u) const;
    // Analytic screened-Rutherford single-scatter cos(theta) (few-collision branch).
    [[nodiscard]] amrex::Real SampleSingleElasticCosTheta(
        bool is_positron,
        amrex::Real energy_eV,
        amrex::Real cdf_u) const;
    [[nodiscard]] amrex::Real SamplePositronBremsstrahlungPhotonEnergy(
        amrex::Real primary_energy_eV,
        amrex::Real cdf_u) const;
    [[nodiscard]] amrex::Real SampleMollerSecondaryEnergy(
        amrex::Real primary_energy_eV,
        amrex::Real cdf_u) const;
    [[nodiscard]] amrex::Real ElectronSoftRadiativeDrag(
        amrex::Real energy_eV,
        amrex::Real density_ratio) const;
    [[nodiscard]] amrex::Real ElectronBremsstrahlungMeanFreePath(
        amrex::Real energy_eV,
        amrex::Real density_ratio) const;
    [[nodiscard]] amrex::Real ElectronTrackedBremsEnergyLoss(
        amrex::Real energy_eV,
        amrex::Real density_ratio) const;
    [[nodiscard]] amrex::Real SampleBremsstrahlungPhotonEnergy(
        amrex::Real primary_energy_eV,
        amrex::Real cdf_u) const;
    [[nodiscard]] amrex::Real SampleBremsstrahlungCosTheta(
        amrex::Real primary_energy_eV,
        amrex::Real cdf_u) const;
    [[nodiscard]] amrex::Real PhotonFeedbackMeanFreePath(
        amrex::Real energy_eV,
        amrex::Real density_ratio) const;
    [[nodiscard]] RreaPhotonChannel SamplePhotonChannel(
        amrex::Real energy_eV,
        amrex::Real cdf_u) const;
    [[nodiscard]] RreaComptonFinalState SampleComptonKleinNishinaFinalState(
        amrex::Real primary_energy_eV,
        amrex::Real draw_mu,
        amrex::Real draw_accept) const;
    [[nodiscard]] RreaPhotoelectricFinalState SamplePhotoelectricFinalState(
        amrex::Real photon_energy_eV,
        amrex::Real cdf_u) const;
    [[nodiscard]] amrex::Real PhotoelectricElectronEnergy(amrex::Real photon_energy_eV) const;
    // Sauter-Gavrila photoelectron polar cos(theta) relative to the incident
    // photon direction.
    // PhotoelectricAngle is inverted directly; the independent
    // PhotoelectricAzimuth draw is consumed only by the cone rotation at the
    // call site.
    [[nodiscard]] amrex::Real SamplePhotoelectronCosTheta(
        amrex::Real photoelectron_energy_eV,
        amrex::Real draw_polar) const;
    [[nodiscard]] RreaPairFinalState SamplePairFinalState(
        amrex::Real photon_energy_eV,
        RreaPhotonChannel channel,
        amrex::Real draw_invariant,
        amrex::Real draw_pair_polar,
        amrex::Real draw_pair_azimuth,
        amrex::Real draw_lepton_polar,
        amrex::Real draw_lepton_azimuth) const;
    [[nodiscard]] amrex::Real PositronCollisionLoss(
        amrex::Real energy_eV,
        amrex::Real density_ratio) const;
    [[nodiscard]] amrex::Real PositronSoftRadiativeDrag(
        amrex::Real energy_eV,
        amrex::Real density_ratio) const;
    [[nodiscard]] amrex::Real PositronBremsstrahlungMeanFreePath(
        amrex::Real energy_eV,
        amrex::Real density_ratio) const;
    [[nodiscard]] amrex::Real PositronHardBhabhaMeanFreePath(
        amrex::Real energy_eV,
        amrex::Real density_ratio) const;
    // NOTE: there is deliberately no PositronHardBhabhaEnergyLoss.  The mean
    // discrete-transfer drag exists for the electron only because the
    // local-runaway gate composes restricted + soft radiative + <T>/lambda;
    // positrons have no such gate, and their Bhabha losses are taken
    // discretely per sampled event in the transport loop.
    [[nodiscard]] amrex::Real SamplePositronBhabhaSecondaryEnergy(
        amrex::Real primary_energy_eV,
        amrex::Real cdf_u) const;
    [[nodiscard]] amrex::Real PositronAnnihilationMeanFreePath(
        amrex::Real energy_eV,
        amrex::Real density_ratio) const;
    [[nodiscard]] RreaPositronAnnihilationFinalState SamplePositronAnnihilationFinalState(
        amrex::Real positron_kinetic_energy_eV,
        amrex::Real draw_energy,
        amrex::Real draw_azimuth) const;
    [[nodiscard]] amrex::Real PositronElasticTransportMeanFreePath(
        amrex::Real energy_eV,
        amrex::Real density_ratio) const;

private:
    // Loader/validation/probe path only: it takes a name and may fail.
    [[nodiscard]] RreaInteractionTable const& Require(std::string const& name) const;
    // Hot path: Load already proved every id resolves.
    [[nodiscard]] RreaInteractionTable const& Require(RreaTable id) const noexcept
    {
        return *m_hot[static_cast<std::size_t>(id)];
    }
    void RequireSchema6SamplerEnergy(
        char const* sampler,
        char const* species,
        RreaCertifiedEnergyRange const& certified_range,
        amrex::Real energy_eV,
        bool allow_exact_zero = false) const;

    bool m_loaded = false;
    std::string m_table_class;
    std::string m_transport_model;
    amrex::Real m_hard_moller_secondary_threshold_eV = 0.0;
    amrex::Real m_brems_absolute_cutoff_eV = 0.0;
    amrex::Real m_reference_density_kg_m3 = 0.0;
    // Mean discrete delta-ray energy per hard-Moller event, on the hard-Moller
    // rate table's own energy grid. Filled once by Load().
    amrex::Vector<amrex::Real> m_hard_moller_mean_secondary_eV;
    // One certified window for both charged species: the configuration declares a
    // single charged_energy_range_eV.
    RreaCertifiedEnergyRange m_charged_certified_range;
    RreaCertifiedEnergyRange m_photon_certified_range;
    std::map<std::string, RreaInteractionTable> m_tables;
    // Points into m_tables (std::map nodes are address-stable); rebuilt by Load.
    std::array<RreaInteractionTable const*, static_cast<std::size_t>(RreaTable::Count)> m_hot{};
    amrex::Vector<RreaMollerCdfGroup> m_moller_cdf;
    std::map<std::string, amrex::Vector<RreaConditionalCdfGroup>> m_conditional_cdfs;
    amrex::Vector<RreaPhotoelectricSubshellCdfGroup> m_photoelectric_subshell_cdf;
    // Elastic parameter and conditional-angle tables for both option4 branches.
    RreaElasticMfpTable m_electron_elastic_mfp;
    RreaElasticMfpTable m_positron_elastic_mfp;
    RreaGsAngleTable m_electron_gs_angle;
    RreaGsAngleTable m_positron_gs_angle;
    RreaGsAngleTable m_electron_wentzel_angle;
    RreaGsAngleTable m_positron_wentzel_angle;
    amrex::Vector<RreaPairEmpiricalBlock> m_pair_empirical_blocks;
    amrex::Real m_gs_few_collision_threshold = 10.0;
    amrex::Real m_gs_isotropic_above_n = 1.0e5;
    amrex::Real m_gs_wentzel_transition_energy_eV = 0.0;
};

// ---- Low-energy electron closure (measured-data E/N table) -----------------
//
// Field-dependent reduced FLUX mobility K0(E/N) and reference two-/three-body
// attachment frequencies for the en_table_flux_v1 /
// en_table_attachment_v1 fluid models, loaded directly from one semantic CSV.
// Local coefficients are
// K0/chi and nu2_ref*chi + nu3_ref*chi^2, with chi the density ratio;
// the loader refuses a bundle whose reference density is not Loschmidt.
//
// Range policy is part of the loaded model, never silent: inside the
// valid-range interpolation is piecewise log-linear on the positive
// quantities (exact at nodes); BELOW range the lowest node is held and the
// caller counts it (the production geometry guarantees ~zero-field volume,
// so a strict below-range abort could never run); ABOVE range the caller
// must abort before committing any state. The two-body onset is nonnegative
// linear-in-log(E/N) across a zero bracket and log-linear once positive.
//
// Host-side only by design: every production fluid loop runs on CPU, and
// coefficients are written into cell MultiFabs before any kernel reads them.

enum class RreaClosureRangeStatus { InRange, HeldBelow, AboveRange, Invalid };

struct RreaElectronClosureCoefficients {
    amrex::Real k0_flux_ref_m2_per_vs = amrex::Real(0.0);
    amrex::Real nu2_ref_per_s = amrex::Real(0.0);
    amrex::Real nu3_ref_per_s = amrex::Real(0.0);
    amrex::Real longitudinal_diffusion_ref_m2_per_s = amrex::Real(0.0);
    amrex::Real transverse_diffusion_ref_m2_per_s = amrex::Real(0.0);
};

class LowEnergyElectronClosure {
public:
    LowEnergyElectronClosure() = default;

    // Loads and fully validates one table: exact unit-bearing CSV header,
    // strictly increasing E/N nodes, finite positive coefficients, and the
    // internal derivation identities
    // k2 = W_F*eta/N, nu2_ref = k2*N0, k3 = W_F*eta/N^2,
    // nu3_ref = k3*N0^2, K0 = W_F/(E/N*N0). Throws
    // std::runtime_error on any violation.
    static LowEnergyElectronClosure Load(std::string const& table_path);

    [[nodiscard]] bool Loaded() const noexcept { return !m_en_td.empty(); }
    [[nodiscard]] RreaElectronClosureCoefficients Evaluate(
        double en_td, RreaClosureRangeStatus& status) const noexcept;
    // Strict variant for startup validation and tests: throws on
    // AboveRange/Invalid (a below-range hold is legal and returned).
    [[nodiscard]] RreaElectronClosureCoefficients EvaluateHost(
        double en_td) const;
    [[nodiscard]] double MinEnTd() const noexcept { return m_en_td.front(); }
    [[nodiscard]] double MaxEnTd() const noexcept { return m_en_td.back(); }
    [[nodiscard]] std::string const& ModelId() const noexcept
    {
        return m_model_id;
    }
    [[nodiscard]] std::string const& InterpolationId() const noexcept
    {
        return m_interpolation_id;
    }

private:
    std::vector<double> m_en_td;
    std::vector<double> m_log_en_td;
    std::vector<double> m_k0_flux_ref;
    std::vector<double> m_log_k0_flux_ref;
    std::vector<double> m_nu2_ref;
    std::vector<double> m_nu3_ref;
    std::vector<double> m_log_nu3_ref;
    std::vector<double> m_longitudinal_diffusion_ref;
    std::vector<double> m_log_longitudinal_diffusion_ref;
    std::vector<double> m_transverse_diffusion_ref;
    std::vector<double> m_log_transverse_diffusion_ref;
    std::string m_model_id;
    std::string m_interpolation_id;
};

}  // namespace rrea

#endif
