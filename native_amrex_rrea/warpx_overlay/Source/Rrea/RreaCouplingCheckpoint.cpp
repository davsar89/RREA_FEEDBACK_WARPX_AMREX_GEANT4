// Checkpoint, rank-state and restart I/O of RreaWarpXCoupling. The checkpoint
// reader owns format, dimensions and state consistency.

#include "RreaWarpXCoupling.H"

#include "RreaCheckpointJson.H"
#include "RreaCheckpointRankState.H"

#include "Fields.H"
#include "EmbeddedBoundary/Enabled.H"
#include "Particles/MultiParticleContainer.H"
#include "Fluids/MultiFluidContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Utility.H>
#include <AMReX_VisMF.H>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <vector>

#include "RreaCouplingDetail.H"

namespace rrea::warpx {

using detail::checkpoint_complete_marker;
using detail::collective_create_dir_all;
using detail::join_path;
using detail::read_binary_file;

namespace {

std::uint32_t constexpr checkpoint_rank_state_binary_version = 6U;

int constexpr checkpoint_state_version = 3;

int constexpr checkpoint_em_pic_integrator_version = 3;

int constexpr checkpoint_rank_state_version = 9;

char constexpr checkpoint_field_handoff_version[] =
    "synchronized_order1_event_deposition_v2";

int json_int_value(std::string const& text, std::string const& key)
{
    return static_cast<int>(
        rrea::json_text::number_or(text, key, -1.0, "RREA checkpoint metadata"));
}

void write_binary_file(std::string const& path, std::string const& contents)
{
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        amrex::FileOpenFailed(path);
    }
    output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    output.flush();
    if (!output) {
        amrex::Abort("RREA checkpoint failed writing required file: " + path);
    }
}

}  // namespace

// Each registry visitor is the single definition of
// both the capture order and the fold class of every field, so the two can
// never drift apart.  Classification rule: anything produced by (or after) an
// MPI collective is Replicated -- summing it across a rank-count fold would
// double-count a global; rank-local partial tallies are Sum; rank-local
// running watermarks are Max. A rank-count fold (RreaFoldRankStateRow)
// verifies that Replicated rows are exactly equal across ranks.
template <typename Visit>
void RreaWarpXCoupling::VisitRankIntegerState(Visit&& visit) const
{
    using Fold = RreaRankStateFoldClass;
    // Schedule cursor and seed accounting advance identically on every rank
    // (the seed loop is collective; the particle count is a global reduction).
    visit(static_cast<std::uint64_t>(m_next_seed_event), Fold::Replicated);
    visit(m_injected_macro_count, Fold::Replicated);
    visit(m_native_seed_particle_count, Fold::Replicated);
    visit(m_kinetic_secondary_electron_count, Fold::Sum);
    visit(m_photon_created_count, Fold::Sum);
    visit(m_positron_created_count, Fold::Sum);
    visit(m_positron_annihilation_count, Fold::Sum);
    visit(m_positron_annihilation_at_rest_count, Fold::Sum);
    visit(m_positron_annihilation_in_flight_count, Fold::Sum);
#define RREA_VISIT_LEDGER_SUM(f) visit(m_ledger.f, Fold::Sum);
    RREA_LEDGER_COUNTS(RREA_VISIT_LEDGER_SUM)
    visit(m_ledger.max_transport_subcycles, Fold::Max);
    visit(m_events ? m_events->OffrankGhostDepositCount() : 0, Fold::Sum);
    visit(m_events ? m_events->OffrankClampCount() : 0, Fold::Sum);
    // The frame counter bumps in a collective path, in lockstep everywhere.
    visit(m_video_frame_write_count, Fold::Replicated);
    visit(static_cast<std::uint64_t>(m_next_video_frame), Fold::Replicated);
    // adaptive_resample_v1: appended in order; restarts
    // must continue these cumulative tallies.  Thin rounds loop on a global
    // count, so that one is Replicated.
    visit(m_resample_killed_count, Fold::Sum);
    visit(m_resample_split_count, Fold::Sum);
    visit(m_resample_thin_rounds, Fold::Replicated);
    if (m_cd_plane_flux.Enabled()) {
        visit(m_cd_plane_flux.WriteCount(), Fold::Replicated);
        int const last_write_step = m_cd_plane_flux.LastWriteStep();
        visit(last_write_step < 0
                ? 0U
                : static_cast<std::uint64_t>(last_write_step) + 1U,
            Fold::Replicated);
    }
    for (auto const value : m_resample_exact_capacity) {
        visit(static_cast<std::uint64_t>(value), Fold::Replicated);
    }
    for (auto const value : m_resample_geometric_floor) {
        visit(static_cast<std::uint64_t>(value), Fold::Replicated);
    }
}

std::vector<std::uint64_t> RreaWarpXCoupling::CaptureRankIntegerState() const
{
    std::vector<std::uint64_t> values;
    VisitRankIntegerState(
        [&values](std::uint64_t value, RreaRankStateFoldClass) {
            values.push_back(value);
        });
    return values;
}

std::vector<RreaRankStateFoldClass>
RreaWarpXCoupling::RankIntegerFoldRules() const
{
    std::vector<RreaRankStateFoldClass> rules;
    VisitRankIntegerState(
        [&rules](std::uint64_t, RreaRankStateFoldClass fold) {
            rules.push_back(fold);
        });
    return rules;
}

template <typename Visit>
void RreaWarpXCoupling::VisitRankRealState(Visit&& visit) const
{
    using Fold = RreaRankStateFoldClass;
    // Accumulated from the global seed schedule on every rank.
    visit(m_injected_physical_weight, Fold::Replicated);
    visit(m_kinetic_secondary_weight, Fold::Sum);
    visit(m_photon_created_weight, Fold::Sum);
    visit(m_positron_created_weight, Fold::Sum);
    // ReduceRealSum product, identical everywhere; it enters the photon-ledger
    // closure gate raw, so a Sum fold here would silently corrupt the ledger.
    visit(m_initial_photon_alive_energy_eV, Fold::Replicated);
    visit(m_positron_annihilation_weight, Fold::Sum);
    visit(m_positron_annihilation_photon_energy_eV, Fold::Sum);
    visit(m_positron_annihilation_local_deposit_eV, Fold::Sum);
    visit(m_ion_pair_source_weight, Fold::Sum);
    visit(m_ion_pair_source_z_weighted_m, Fold::Sum);
    RREA_LEDGER_REALS(RREA_VISIT_LEDGER_SUM)
#undef RREA_VISIT_LEDGER_SUM
    // adaptive_resample_v1 cumulative state, kept in registry order.
    visit(m_resample_killed_weight, Fold::Sum);
    visit(m_resample_boost_weight, Fold::Sum);
    visit(m_resample_split_weight, Fold::Sum);
    visit(m_resample_killed_energy_eV, Fold::Sum);
    visit(m_resample_boost_energy_eV, Fold::Sum);
    if (m_cd_plane_flux.Enabled()) {
        visit(m_cd_plane_flux.NextOutputTimeS(), Fold::Replicated);
        visit(m_cd_plane_flux.LastWriteStep() >= 0
                ? m_cd_plane_flux.LastWriteTimeS()
                : amrex::Real(0.0),
            Fold::Replicated);
        // Per-rank partial sums, ReduceRealSum-ed only at CSV-write time.
        for (auto const value : m_cd_plane_flux.Accumulator().Flatten()) {
            visit(value, Fold::Sum);
        }
    }
}

std::vector<amrex::Real> RreaWarpXCoupling::CaptureRankRealState() const
{
    std::vector<amrex::Real> values;
    VisitRankRealState(
        [&values](amrex::Real value, RreaRankStateFoldClass) {
            values.push_back(value);
        });
    return values;
}

std::vector<RreaRankStateFoldClass>
RreaWarpXCoupling::RankRealFoldRules() const
{
    std::vector<RreaRankStateFoldClass> rules;
    VisitRankRealState(
        [&rules](amrex::Real, RreaRankStateFoldClass fold) {
            rules.push_back(fold);
        });
    return rules;
}

void RreaWarpXCoupling::ToleratedRestartEpochDeviationOrAbort(
    std::string const& description) const
{
    if (!m_allow_restart_epoch_change) {
        amrex::Abort(
            description
            + "; an intentional engine, MPI-layout or physics epoch break must "
              "set rrea.allow_restart_epoch_change=1 "
              "with rrea.restart_epoch_change_reason");
    }
    m_restart_epoch_deviations.push_back(description);
    amrex::Print() << "RREA RESTART EPOCH: tolerated deviation -- "
                   << description
                   << " (reason: " << m_restart_epoch_change_reason << ")\n";
}

void RreaWarpXCoupling::RestoreRankState(
    std::vector<std::uint64_t> const& integers,
    std::vector<amrex::Real> const& reals)
{
    std::size_t i = 0;
    auto next_i = [&]() -> std::uint64_t {
        if (i >= integers.size()) {
            amrex::Abort("RREA checkpoint integer registry is truncated");
        }
        return integers[i++];
    };
    m_next_seed_event = static_cast<std::size_t>(next_i());
    m_injected_macro_count = next_i();
    m_native_seed_particle_count = next_i();
    m_kinetic_secondary_electron_count = next_i();
    m_photon_created_count = next_i();
    m_positron_created_count = next_i();
    m_positron_annihilation_count = next_i();
    m_positron_annihilation_at_rest_count = next_i();
    m_positron_annihilation_in_flight_count = next_i();
#define RREA_RESTORE_LEDGER_COUNT(f) m_ledger.f = next_i();
    RREA_LEDGER_COUNTS(RREA_RESTORE_LEDGER_COUNT)
#undef RREA_RESTORE_LEDGER_COUNT
    m_ledger.max_transport_subcycles = next_i();
    std::uint64_t const offrank_ghost = next_i();
    std::uint64_t const offrank_clamp = next_i();
    m_video_frame_write_count = next_i();
    m_next_video_frame = static_cast<std::size_t>(next_i());
    m_resample_killed_count = next_i();
    m_resample_split_count = next_i();
    m_resample_thin_rounds = next_i();
    std::uint64_t cd_plane_write_count = 0;
    int cd_plane_last_write_step = -1;
    if (m_cd_plane_flux.Enabled()) {
        cd_plane_write_count = next_i();
        std::uint64_t const encoded_step = next_i();
        if (encoded_step > 0U
            && encoded_step - 1U
                > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
            amrex::Abort("RREA checkpoint C&D plane write step overflows int");
        }
        cd_plane_last_write_step = encoded_step == 0U
            ? -1
            : static_cast<int>(encoded_step - 1U);
    }
    for (auto& value : m_resample_exact_capacity) {
        std::uint64_t const saved = next_i();
        if (saved > static_cast<std::uint64_t>(
                std::numeric_limits<amrex::Long>::max())) {
            amrex::Abort("RREA checkpoint CIC capacity overflows Long");
        }
        value = static_cast<amrex::Long>(saved);
    }
    for (auto& value : m_resample_geometric_floor) {
        std::uint64_t const saved = next_i();
        if (saved > static_cast<std::uint64_t>(
                std::numeric_limits<amrex::Long>::max())) {
            amrex::Abort("RREA checkpoint CIC geometric floor overflows Long");
        }
        value = static_cast<amrex::Long>(saved);
    }
    if (i != integers.size()) {
        amrex::Abort("RREA checkpoint integer registry size mismatch");
    }
    if (m_events) {
        m_events->RestoreOffrankDepositCounts(offrank_ghost, offrank_clamp);
    }

    std::size_t r = 0;
    auto next_r = [&]() -> amrex::Real {
        if (r >= reals.size()) {
            amrex::Abort("RREA checkpoint real registry is truncated");
        }
        return reals[r++];
    };
    m_injected_physical_weight = next_r();
    m_kinetic_secondary_weight = next_r();
    m_photon_created_weight = next_r();
    m_positron_created_weight = next_r();
    m_initial_photon_alive_energy_eV = next_r();
    m_positron_annihilation_weight = next_r();
    m_positron_annihilation_photon_energy_eV = next_r();
    m_positron_annihilation_local_deposit_eV = next_r();
    m_ion_pair_source_weight = next_r();
    m_ion_pair_source_z_weighted_m = next_r();
#define RREA_RESTORE_LEDGER_REAL(f) m_ledger.f = next_r();
    RREA_LEDGER_REALS(RREA_RESTORE_LEDGER_REAL)
#undef RREA_RESTORE_LEDGER_REAL
    m_resample_killed_weight = next_r();
    m_resample_boost_weight = next_r();
    m_resample_split_weight = next_r();
    m_resample_killed_energy_eV = next_r();
    m_resample_boost_energy_eV = next_r();
    if (m_cd_plane_flux.Enabled()) {
        amrex::Real const next_output_time_s = next_r();
        amrex::Real const last_write_time_s = next_r();
        std::vector<amrex::Real> plane_values(
            m_cd_plane_flux.PlaneZ().size() * RreaPlaneFluxValuesPerPlane);
        for (auto& value : plane_values) {
            value = next_r();
        }
        if (!m_cd_plane_flux.Accumulator().RestoreFlat(plane_values)) {
            amrex::Abort("RREA checkpoint C&D plane counters are invalid");
        }
        try {
            m_cd_plane_flux.RestoreSchedule(
                cd_plane_write_count,
                cd_plane_last_write_step,
                last_write_time_s,
                next_output_time_s);
        } catch (std::invalid_argument const& error) {
            amrex::Abort(
                std::string("RREA checkpoint C&D plane schedule is invalid: ")
                + error.what());
        }
        m_cd_plane_last_observed_step = cd_plane_last_write_step;
        m_cd_plane_last_observed_time_s = cd_plane_last_write_step >= 0
            ? last_write_time_s
            : -std::numeric_limits<amrex::Real>::infinity();
        m_cd_plane_stream_initialized = false;
    }
    if (r != reals.size()) {
        amrex::Abort("RREA checkpoint real registry size mismatch");
    }
}

void RreaWarpXCoupling::WriteRankState(std::string const& dir) const
{
    auto const local_i = CaptureRankIntegerState();
    auto const local_r = CaptureRankRealState();
    int const nprocs = amrex::ParallelDescriptor::NProcs();
    int const root = amrex::ParallelDescriptor::IOProcessorNumber();
    auto const layout = RreaValidateRankStatePayloadLayout(
        static_cast<std::uint64_t>(nprocs),
        static_cast<std::uint64_t>(local_i.size()),
        static_cast<std::uint64_t>(local_r.size()),
        static_cast<std::uint32_t>(sizeof(amrex::Real)));
    if (!layout.valid) {
        amrex::Abort(
            std::string("cannot write RREA rank_state.bin: ") + layout.error);
    }
    std::vector<std::uint64_t> all_i;
    std::vector<amrex::Real> all_r;
    if (amrex::ParallelDescriptor::IOProcessor()) {
        all_i.resize(layout.integer_count);
        all_r.resize(layout.real_count);
    }
    amrex::ParallelDescriptor::Gather(
        local_i.data(), static_cast<int>(local_i.size()),
        all_i.data(), static_cast<int>(local_i.size()), root);
    amrex::ParallelDescriptor::Gather(
        local_r.data(), static_cast<int>(local_r.size()),
        all_r.data(), static_cast<int>(local_r.size()), root);
    if (amrex::ParallelDescriptor::IOProcessor()) {
        std::ofstream output(
            join_path(dir, "rank_state.bin"),
            std::ios::out | std::ios::binary | std::ios::trunc);
        if (!output) {
            amrex::FileOpenFailed(join_path(dir, "rank_state.bin"));
        }
        char const magic[8] = {'R','R','E','A','C','K','2','\0'};
        std::uint32_t const version = checkpoint_rank_state_binary_version;
        std::uint32_t const endian = 0x01020304U;
        std::uint32_t const real_size = sizeof(amrex::Real);
        std::uint32_t const reserved = 0;
        std::uint64_t const np = static_cast<std::uint64_t>(nprocs);
        std::uint64_t const ni = static_cast<std::uint64_t>(local_i.size());
        std::uint64_t const nr = static_cast<std::uint64_t>(local_r.size());
        output.write(magic, sizeof(magic));
        output.write(reinterpret_cast<char const*>(&version), sizeof(version));
        output.write(reinterpret_cast<char const*>(&endian), sizeof(endian));
        output.write(reinterpret_cast<char const*>(&real_size), sizeof(real_size));
        output.write(reinterpret_cast<char const*>(&reserved), sizeof(reserved));
        output.write(reinterpret_cast<char const*>(&np), sizeof(np));
        output.write(reinterpret_cast<char const*>(&ni), sizeof(ni));
        output.write(reinterpret_cast<char const*>(&nr), sizeof(nr));
        output.write(
            reinterpret_cast<char const*>(all_i.data()),
            static_cast<std::streamsize>(layout.integer_bytes));
        output.write(
            reinterpret_cast<char const*>(all_r.data()),
            static_cast<std::streamsize>(layout.real_bytes));
        if (!output) {
            amrex::Abort("failed writing RREA rank_state.bin");
        }
    }
    amrex::ParallelDescriptor::Barrier();
}

void RreaWarpXCoupling::ReadRankState(std::string const& dir)
{
    std::string const path = join_path(dir, "rank_state.bin");
    if (!amrex::FileExists(path)) {
        amrex::Abort("RREA checkpoint v2 is missing rank_state.bin");
    }
    int const nprocs = amrex::ParallelDescriptor::NProcs();
    int const root = amrex::ParallelDescriptor::IOProcessorNumber();
    std::uint64_t saved_nprocs = 0;
    std::uint64_t ni = 0;
    std::uint64_t nr = 0;
    std::vector<std::uint64_t> all_i;
    std::vector<amrex::Real> all_r;
    if (amrex::ParallelDescriptor::IOProcessor()) {
        std::ifstream input(path, std::ios::in | std::ios::binary);
        if (!input) {
            amrex::FileOpenFailed(path);
        }
        char magic[8] = {};
        std::uint32_t version = 0;
        std::uint32_t endian = 0;
        std::uint32_t real_size = 0;
        std::uint32_t reserved = 0;
        input.read(magic, sizeof(magic));
        input.read(reinterpret_cast<char*>(&version), sizeof(version));
        input.read(reinterpret_cast<char*>(&endian), sizeof(endian));
        input.read(reinterpret_cast<char*>(&real_size), sizeof(real_size));
        input.read(reinterpret_cast<char*>(&reserved), sizeof(reserved));
        input.read(reinterpret_cast<char*>(&saved_nprocs), sizeof(saved_nprocs));
        input.read(reinterpret_cast<char*>(&ni), sizeof(ni));
        input.read(reinterpret_cast<char*>(&nr), sizeof(nr));
        if (!input) {
            amrex::Abort("RREA checkpoint rank_state.bin header is truncated");
        }
        char const expected_magic[8] = {'R','R','E','A','C','K','2','\0'};
        if (std::memcmp(magic, expected_magic, sizeof(magic)) != 0
            || version != checkpoint_rank_state_binary_version
            || endian != 0x01020304U
            || real_size != sizeof(amrex::Real)) {
            amrex::Abort("RREA checkpoint rank_state.bin header is incompatible");
        }
        // A rank-count change is a tolerated epoch
        // deviation, not a format incompatibility -- the rows are rank-local
        // and fold deterministically (see RreaFoldRankStateRow).
        if (saved_nprocs != static_cast<std::uint64_t>(nprocs)) {
            ToleratedRestartEpochDeviationOrAbort(
                "RREA checkpoint rank_state.bin was written at "
                + std::to_string(saved_nprocs)
                + " MPI ranks but this run has "
                + std::to_string(nprocs));
        }
        if (reserved != 0U) {
            amrex::Abort(
                "RREA checkpoint rank_state.bin reserved header field is nonzero");
        }
        auto const layout = RreaValidateRankStatePayloadLayout(
            saved_nprocs, ni, nr, real_size);
        if (!layout.valid) {
            amrex::Abort(
                std::string("RREA checkpoint rank_state.bin has unsafe dimensions: ")
                + layout.error);
        }
        all_i.resize(layout.integer_count);
        all_r.resize(layout.real_count);
        input.read(
            reinterpret_cast<char*>(all_i.data()),
            static_cast<std::streamsize>(layout.integer_bytes));
        input.read(
            reinterpret_cast<char*>(all_r.data()),
            static_cast<std::streamsize>(layout.real_bytes));
        if (!input) {
            amrex::Abort("RREA checkpoint rank_state.bin is truncated");
        }
        char trailing = 0;
        input.read(&trailing, 1);
        if (input.gcount() != 0) {
            amrex::Abort("RREA checkpoint rank_state.bin has trailing bytes");
        }
        if (!input.eof()) {
            amrex::Abort("failed checking the end of RREA rank_state.bin");
        }
    }
    amrex::ParallelDescriptor::Bcast(&saved_nprocs, 1, root);
    amrex::ParallelDescriptor::Bcast(&ni, 1, root);
    amrex::ParallelDescriptor::Bcast(&nr, 1, root);
    // The payload keeps the WRITER's rank count; every rank must size and
    // validate against it, not against the current communicator.
    auto const layout = RreaValidateRankStatePayloadLayout(
        saved_nprocs, ni, nr,
        static_cast<std::uint32_t>(sizeof(amrex::Real)));
    if (!layout.valid) {
        amrex::Abort(
            std::string("RREA checkpoint rank_state.bin has unsafe dimensions: ")
            + layout.error);
    }
    if (!amrex::ParallelDescriptor::IOProcessor()) {
        all_i.resize(layout.integer_count);
        all_r.resize(layout.real_count);
    }
    amrex::ParallelDescriptor::Bcast(
        all_i.data(), static_cast<int>(layout.integer_count), root);
    amrex::ParallelDescriptor::Bcast(
        all_r.data(), static_cast<int>(layout.real_count), root);
    std::size_t const rank = static_cast<std::size_t>(amrex::ParallelDescriptor::MyProc());
    std::vector<std::uint64_t> local_i;
    std::vector<amrex::Real> local_r;
    if (saved_nprocs == static_cast<std::uint64_t>(nprocs)) {
        local_i.assign(
            all_i.begin() + static_cast<std::ptrdiff_t>(rank * ni),
            all_i.begin() + static_cast<std::ptrdiff_t>((rank + 1) * ni));
        local_r.assign(
            all_r.begin() + static_cast<std::ptrdiff_t>(rank * nr),
            all_r.begin() + static_cast<std::ptrdiff_t>((rank + 1) * nr));
    } else {
        auto const int_rules = RankIntegerFoldRules();
        auto const real_rules = RankRealFoldRules();
        local_i.resize(static_cast<std::size_t>(ni));
        local_r.resize(static_cast<std::size_t>(nr));
        auto const int_status = RreaFoldRankStateRow(
            all_i.data(), saved_nprocs, static_cast<std::uint64_t>(nprocs),
            static_cast<std::uint64_t>(rank), static_cast<std::size_t>(ni),
            int_rules.data(), int_rules.size(), local_i.data());
        if (!int_status.ok) {
            amrex::Abort(
                std::string("RREA rank-state fold failed (integer index ")
                + std::to_string(int_status.field) + "): " + int_status.error);
        }
        auto const real_status = RreaFoldRankStateRow(
            all_r.data(), saved_nprocs, static_cast<std::uint64_t>(nprocs),
            static_cast<std::uint64_t>(rank), static_cast<std::size_t>(nr),
            real_rules.data(), real_rules.size(), local_r.data());
        if (!real_status.ok) {
            amrex::Abort(
                std::string("RREA rank-state fold failed (real index ")
                + std::to_string(real_status.field) + "): " + real_status.error);
        }
        amrex::Print() << "RREA RESTART EPOCH: rank_state.bin folded from "
                       << saved_nprocs << " to " << nprocs << " ranks\n";
    }
    RestoreRankState(local_i, local_r);
}

void RreaWarpXCoupling::WriteMeshFields(std::string const& dir) const
{
    if (!m_initialized || !m_advance) {
        return;
    }
    collective_create_dir_all(dir);
    auto const& fluid = m_advance->Fluid();
    std::string const level_dir = join_path(dir, "Level_0");
    collective_create_dir_all(level_dir);
    amrex::VisMF::Write(fluid.LowElectronDensity(), join_path(level_dir, "ne_low"));
    amrex::VisMF::Write(fluid.PositiveIonDensity(), join_path(level_dir, "n_pos"));
    amrex::VisMF::Write(fluid.NegativeIonDensity(), join_path(level_dir, "n_neg"));
    amrex::VisMF::Write(fluid.FluidChargeDensity(), join_path(level_dir, "rho_fluid"));
    amrex::VisMF::Write(fluid.Conductivity(), join_path(level_dir, "sigma"));
    amrex::VisMF::Write(
        *m_level.rho_c_per_m3,
        join_path(level_dir, "rho_adapter"));
    m_advance->WriteMaxwellState(dir);
}

// root is the resolved checkpoint directory (LoadPendingCheckpointIfAny).
void RreaWarpXCoupling::ReadMeshFields(std::string const& root)
{
    // AMReX's own BoxArray check inside VisMF::Read is assert-only (compiled
    // out in Release), so a structural mismatch could become a silent misread.
    // This hard guard keeps epoch relayout limited to rank-to-box mapping
    // changes (which VisMF reads across), never a BoxArray change.
    auto checked_read = [](amrex::MultiFab& mf, std::string const& name) {
        amrex::VisMF const header_probe(name);
        if (header_probe.boxArray() != mf.boxArray()
            || header_probe.nComp() != mf.nComp()) {
            amrex::Abort(
                "RREA checkpoint mesh field " + name
                + " does not match the run's BoxArray/nComp");
        }
        amrex::VisMF::Read(mf, name);
    };
    auto& fluid = m_advance->Fluid();
    std::string const level_dir = join_path(root, "Level_0");
    checked_read(fluid.LowElectronDensity(), join_path(level_dir, "ne_low"));
    checked_read(fluid.PositiveIonDensity(), join_path(level_dir, "n_pos"));
    checked_read(fluid.NegativeIonDensity(), join_path(level_dir, "n_neg"));
    checked_read(fluid.FluidChargeDensity(), join_path(level_dir, "rho_fluid"));
    checked_read(fluid.Conductivity(), join_path(level_dir, "sigma"));
    checked_read(*m_rho_adapter, join_path(level_dir, "rho_adapter"));
    // Replicated scattered Er/Ez/Btheta; every rank reads the identical
    // state. Missing/mismatched state aborts inside.
    m_advance->ReadMaxwellState(root);
}

void RreaWarpXCoupling::WriteMetadata(std::string const& dir) const
{
    // metadata.json carries the state compatibility fields ReadMetadata checks,
    // run descriptors, two capture write counts, and optional epoch provenance.
    // Cumulative physics counters live in rank_state.bin (counter_semantics
    // rank_local_v1) and the reduced CSV, never here.
    if (!amrex::ParallelDescriptor::IOProcessor()) {
        return;
    }
    std::ofstream output(join_path(dir, "metadata.json"));
    if (!output) {
        amrex::FileOpenFailed(join_path(dir, "metadata.json"));
    }
    output << std::setprecision(17);
    output << "{\n";
    output << "  \"schema\": \"rrea_checkpoint_v2\",\n";
    output << "  \"state_version\": " << checkpoint_state_version << ",\n";
    output << "  \"em_pic_integrator_version\": "
           << checkpoint_em_pic_integrator_version << ",\n";
    output << "  \"field_handoff_version\": \""
           << checkpoint_field_handoff_version << "\",\n";
    output << "  \"counter_semantics\": \"rank_local_v1\",\n";
    output << "  \"rank_state_version\": "
           << checkpoint_rank_state_version << ",\n";
    output << "  \"writer_nprocs\": " << amrex::ParallelDescriptor::NProcs() << ",\n";
    output << "  \"population_ceiling_policy\": \""
           << json_escape(m_population_ceiling_policy) << "\",\n";
    output << "  \"max_electron_macros\": " << m_max_electron_macros << ",\n";
    output << "  \"max_photon_macros\": " << m_max_photon_macros << ",\n";
    output << "  \"max_positron_macros\": " << m_max_positron_macros << ",\n";
    output << "  \"population_target_electron_macros\": "
           << m_population_target_electron_macros << ",\n";
    output << "  \"population_target_positron_macros\": "
           << m_population_target_positron_macros << ",\n";
    output << "  \"population_control_interval\": "
           << m_population_control_interval << ",\n";
    output << "  \"population_split_min_weight\": "
           << m_population_split_min_weight << ",\n";
    output << "  \"transport_model\": \""
           << json_escape(m_interaction_tables.TransportModel()) << "\",\n";
    output << "  \"case_id\": \"" << json_escape(m_case_id) << "\",\n";
    output << "  \"interaction_table_class\": \""
           << json_escape(m_interaction_tables.TableClass()) << "\",\n";
    output << "  \"rng_scheme\": \"" << json_escape(m_rng_scheme) << "\",\n";
    output << "  \"rng_seed\": " << m_rng_seed << ",\n";
    output << "  \"population_controller_rng_salt\": "
           << m_population_controller_rng_salt << ",\n";
    output << "  \"low_energy_cutoff_eV\": " << m_low_energy_cutoff_eV << ",\n";
    output << "  \"photon_cutoff_eV\": "
           << m_interaction_tables.BremsAbsoluteCutoffEv() << ",\n";
    output << "  \"hard_moller_secondary_threshold_eV\": "
           << m_interaction_tables.HardMollerSecondaryThresholdEv() << ",\n";
    output << "  \"video_frame_write_count\": " << m_video_frame_write_count << ",\n";
    output << "  \"cd_plane_write_count\": "
           << (m_cd_plane_flux.Enabled() ? m_cd_plane_flux.WriteCount() : 0U);
    // Optional epoch metadata stays outside the required-key set for format
    // compatibility.
    if (!m_restart_epoch_change_reason.empty()) {
        output << ",\n  \"restart_epoch_change_reason\": \""
               << json_escape(m_restart_epoch_change_reason) << "\"";
    }
    if (!m_restart_epoch_deviations.empty()) {
        std::string joined;
        for (auto const& deviation : m_restart_epoch_deviations) {
            joined += (joined.empty() ? "" : "; ") + deviation;
        }
        output << ",\n  \"restart_epoch_tolerated_deviations\": \""
               << json_escape(joined) << "\"";
    }
    output << "\n";
    output << "}\n";
}

void RreaWarpXCoupling::ReadMetadata(std::string const& root)
{
    std::string const metadata = join_path(root, "metadata.json");
    if (!amrex::FileExists(metadata)) {
        amrex::Abort("RREA restart requires checkpoint-v2 metadata.json");
    }
    // A missing field reads as empty/-1 and fails the comparisons below.
    std::string const text = read_binary_file(metadata);
    if (json_string_value(text, "schema") != "rrea_checkpoint_v2") {
        amrex::Abort("RREA restart rejects an unknown checkpoint schema");
    }
    int const writer_nprocs = json_int_value(text, "writer_nprocs");
    if (writer_nprocs != amrex::ParallelDescriptor::NProcs()) {
        amrex::Print() << "RREA restart relayout: writer ranks="
                       << writer_nprocs << ", reader ranks="
                       << amrex::ParallelDescriptor::NProcs() << "\n";
    }
    if (json_int_value(text, "state_version")
            != checkpoint_state_version
        || json_int_value(text, "em_pic_integrator_version")
            != checkpoint_em_pic_integrator_version
        || json_int_value(text, "rank_state_version")
            != checkpoint_rank_state_version
        || json_string_value(text, "field_handoff_version")
            != checkpoint_field_handoff_version
        || json_string_value(text, "counter_semantics") != "rank_local_v1") {
        amrex::Abort(
            "RREA checkpoint-v2 state/integrator/layout contract is incompatible");
    }
}

void RreaWarpXCoupling::WriteCheckpoint(std::string const& dir)
{
    ReadParameters();
    if (m_debug_disable_checkpoint) {
        return;
    }
    if (!m_initialized) {
        return;
    }
    if (m_cd_plane_flux.Enabled() && m_cd_plane_last_observed_step >= 0) {
        MaybeWriteCdPlaneFlux(
            m_cd_plane_last_observed_step,
            m_cd_plane_last_observed_time_s,
            /*force=*/true);
    }
    // Refresh the native seed count before checkpoint metadata is written.
    RefreshNativeParticleCounts(WarpX::GetInstance());
    std::string const checkpoint_root = join_path(dir, "rrea_checkpoint");
    collective_create_dir_all(checkpoint_root);
    if (amrex::ParallelDescriptor::IOProcessor()) {
        std::string const stale = join_path(checkpoint_root, "COMPLETE");
        if (amrex::FileExists(stale) && std::remove(stale.c_str()) != 0) {
            amrex::Abort("RREA checkpoint could not remove stale file: " + stale);
        }
    }
    amrex::ParallelDescriptor::Barrier();
    WriteMeshFields(checkpoint_root);
    WriteRankState(checkpoint_root);
    if (m_electron_creation_bin_m > 0) {
        auto global = m_electron_creation.weights;
        amrex::ParallelDescriptor::ReduceRealSum(global.data(), global.size(),
                                               amrex::ParallelDescriptor::IOProcessorNumber());
        if (amrex::ParallelDescriptor::IOProcessor())
            m_electron_creation.Write(join_path(checkpoint_root, "electron_creation.bin"),
                WarpX::GetInstance().gett_new(0), m_profile_altitude_msl_at_z0_m, global);
    }
    WriteMetadata(checkpoint_root);
    amrex::ParallelDescriptor::Barrier();
    if (amrex::ParallelDescriptor::IOProcessor()) {
        // COMPLETE is the atomic publication record.  Use binary output so
        // Windows cannot translate LF to CRLF; readers require these exact
        // bytes and reject directories, links, suffixes, and partial writes.
        write_binary_file(
            join_path(checkpoint_root, "COMPLETE"),
            checkpoint_complete_marker);
    }
    amrex::ParallelDescriptor::Barrier();
}

void RreaWarpXCoupling::ReadCheckpoint(std::string const& dir)
{
    ReadParameters();
    if (m_debug_disable_checkpoint) {
        return;
    }
    m_pending_checkpoint_dir = dir;
    if (m_initialized) {
        LoadPendingCheckpointIfAny();
    }
}

}  // namespace rrea::warpx
