#ifndef RREA_COUPLING_DETAIL_H
#define RREA_COUPLING_DETAIL_H

// Shared file-local helpers and version constants for the checkpoint,
// diagnostics, continuity-fixture and core RreaWarpXCoupling translation units.

#include <AMReX.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>
#include <AMReX_Utility.H>
#include <AMReX_Vector.H>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

class WarpXParticleContainer;

namespace rrea::warpx::detail {

inline char constexpr checkpoint_complete_marker[] = "rrea_checkpoint_v2\n";
// Schema 9 includes realized and weighted E/N diagnostics, closure endpoint
// activity, net-chord current/continuity ledgers, material stiffness/substeps,
// particle effective sample size and exact-CIC capacity/geometric floors.
inline int constexpr checkpoint_reduced_diagnostic_schema_version = 9;

inline std::string lower_copy(std::string value)
{
    std::transform(
        value.begin(),
        value.end(),
        value.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

inline std::string join_path(std::string const& parent, std::string const& child)
{
    if (parent.empty()) {
        return child;
    }
    char const last = parent.back();
    if (last == '/' || last == '\\') {
        return parent + child;
    }
    return parent + "/" + child;
}

inline void collective_create_dir_all(std::string const& dir)
{
    constexpr int permission_flag_rwxrxrx = 0755;
    if (amrex::ParallelDescriptor::IOProcessor()) {
        if (!amrex::UtilCreateDirectory(dir, permission_flag_rwxrxrx)) {
            amrex::CreateDirectoryFailed(dir);
        }
    }
    amrex::ParallelDescriptor::Barrier();
}

inline std::string read_binary_file(std::string const& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        amrex::Abort("RREA checkpoint could not open required file: " + path);
    }
    return std::string(
        (std::istreambuf_iterator<char>(input)),
        std::istreambuf_iterator<char>());
}

inline bool dynamic_regrid_requested()
{
    amrex::ParmParse pp("amr");
    amrex::Vector<int> intervals;
    pp.queryarr("regrid_int", intervals);
    return std::any_of(
        intervals.begin(), intervals.end(), [](int value) { return value >= 0; });
}

// Batched allreduces: one collective per operation instead of one per scalar.
inline void global_vector_reduce(
    std::vector<amrex::Real>& values, bool maximum)
{
    if (values.empty()) {
        return;
    }
    if (values.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        amrex::Abort("RREA vector reduction exceeds AMReX int count limit");
    }
    int const count = static_cast<int>(values.size());
    if (maximum) {
        amrex::ParallelDescriptor::ReduceRealMax(values.data(), count);
    } else {
        amrex::ParallelDescriptor::ReduceRealSum(values.data(), count);
    }
}

inline void global_vector_sum(std::vector<amrex::Real>& values)
{
    global_vector_reduce(values, false);
}

inline void global_vector_max(std::vector<amrex::Real>& values)
{
    global_vector_reduce(values, true);
}

inline amrex::Real global_real_min(amrex::Real value)
{
    amrex::ParallelDescriptor::ReduceRealMin(value);
    return value;
}

inline std::uint64_t global_count_sum(std::uint64_t value)
{
    amrex::Long reduced = static_cast<amrex::Long>(value);
    amrex::ParallelDescriptor::ReduceLongSum(reduced);
    return static_cast<std::uint64_t>(reduced);
}

// Largest live macro weight over the named species, in physical electrons;
// the continuity gate's absolute allowance scales with it.  Defined in
// RreaCouplingDiagnostics.cpp, where the weight sweep it reuses lives.
amrex::Real maximum_represented_charge_e(
    std::array<std::string, 3> const& names);

// Fills the RREA runtime real attributes (previous position/momentum, then
// zeros) for one AddNParticles batch and returns the real-component count that
// call must pass.  Defined in RreaCouplingSources.cpp.
int append_rrea_previous_position_attributes(
    WarpXParticleContainer& species,
    amrex::Vector<amrex::ParticleReal> const& x,
    amrex::Vector<amrex::ParticleReal> const& y,
    amrex::Vector<amrex::ParticleReal> const& z,
    amrex::Vector<amrex::ParticleReal> const& ux,
    amrex::Vector<amrex::ParticleReal> const& uy,
    amrex::Vector<amrex::ParticleReal> const& uz,
    amrex::Vector<amrex::Vector<amrex::ParticleReal>>& attr_real);

// Defined in RreaWarpXCoupling.cpp / RreaCouplingConfig.cpp respectively.
amrex::Real photon_species_energy(std::string const& name);
void validate_rrea_external_field_contract(
    std::string const& electron_species,
    std::string const& photon_species,
    std::string const& positron_species);

}  // namespace rrea::warpx::detail

#endif  // RREA_COUPLING_DETAIL_H
