#ifndef RREA_INTERACTION_TYPES_H
#define RREA_INTERACTION_TYPES_H

#include "rrea/RreaConstants.H"
#include "rrea/RreaKineticSupport.H"

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

    [[nodiscard]] RREA_HOST_DEVICE bool Contains(amrex::Real energy_eV) const noexcept
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

    [[nodiscard]] RREA_HOST_DEVICE amrex::Real TotalPerM() const noexcept
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

[[nodiscard]] RREA_HOST_DEVICE inline bool RreaPairChannelHasPhaseSpace(
    amrex::Real photon_energy_eV,
    RreaPhotonChannel channel) noexcept;

[[nodiscard]] RREA_HOST_DEVICE inline RreaPairFinalState BuildRreaPairEmpiricalFinalState(
    RreaPairEmpiricalRecord const& record,
    amrex::Real photon_energy_eV,
    RreaPhotonChannel channel,
    amrex::Real draw_pair_azimuth);


} // namespace rrea
#endif
