#include "rrea/RreaEventAccumulator.H"

#include "rrea/RreaConstants.H"

#include <AMReX_Array4.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_IntVect.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace rrea {

namespace {

// One scratch component per event field.
constexpr int kEventFieldCount = 2;

struct FoldEventComponent {
    amrex::Array4<amrex::Real const> source;
    amrex::Array4<amrex::Real> target;
    int field_id;
    RREA_HOST_DEVICE void operator()(int i, int j, int k) const noexcept
    { target(i,j,k) += source(i,j,k,field_id); }
};

}  // namespace

RreaEventAccumulator::RreaEventAccumulator(
    amrex::Geometry const& geom,
    amrex::BoxArray const& box_array,
    amrex::DistributionMapping const& distribution_map,
    int n_grow)
    : m_geom(geom)
{
    RequireHybridManagedMemory();
    for (auto& field : m_fields) {
        field.define(box_array, distribution_map, 1, n_grow);
    }
    auto const& reference = m_fields[0];
    m_local_fab_routes.reserve(static_cast<std::size_t>(reference.local_size()));
    m_scratch.reserve(static_cast<std::size_t>(reference.local_size()));
    for (amrex::MFIter mfi(reference); mfi.isValid(); ++mfi) {
        amrex::Box const valid_box = mfi.validbox();
        amrex::Box const grown_box = amrex::grow(valid_box, n_grow);
        m_local_fab_routes.push_back(LocalFabRoute{mfi.index(), valid_box, grown_box});
        m_scratch.emplace_back(grown_box, kEventFieldCount);
    }
    Reset();
}

void RreaEventAccumulator::Reset()
{
    GpuSynchronize();
    for (auto& field : m_fields) {
        field.setVal(0.0);
    }
    for (auto& fab : m_scratch) {
        fab.setVal<amrex::RunOn::Host>(amrex::Real(0.0));
    }
    m_deferred_offbox.clear();
}

void RreaEventAccumulator::FoldScratchIntoFields()
{
    for (int field_id = 0; field_id < kEventFieldCount; ++field_id) {
        amrex::MultiFab& field = m_fields[field_id];
        for (std::size_t box = 0; box < m_local_fab_routes.size(); ++box) {
            auto const source = m_scratch[box].const_array();
            auto const target = field.array(m_local_fab_routes[box].global_index);
            amrex::ParallelFor(m_local_fab_routes[box].grown_box,
                FoldEventComponent{source, target, field_id});
        }
    }
    GpuSynchronize();
    for (auto& fab : m_scratch) {
        fab.setVal<amrex::RunOn::Host>(amrex::Real(0.0));
    }
}

void RreaEventAccumulator::FlushDeferredOffboxDeposits()
{
    auto& deferred = m_deferred_offbox;
    amrex::Long total = static_cast<amrex::Long>(deferred.size());
    amrex::ParallelDescriptor::ReduceLongSum(total);
    if (total == 0) {
        return;
    }

    // Serialize as (field_id, i, j, value) quadruples; the integer members
    // are far below 2^53, so the double encoding is exact.
    std::vector<double> flat;
    flat.reserve(deferred.size() * 4U);
    for (auto const& entry : deferred) {
        flat.push_back(static_cast<double>(entry.field_id));
        flat.push_back(static_cast<double>(entry.i));
        flat.push_back(static_cast<double>(entry.j));
        flat.push_back(entry.value);
    }
    deferred.clear();

    std::vector<double> global;
#ifdef AMREX_USE_MPI
    {
        MPI_Comm comm = amrex::ParallelDescriptor::Communicator();
        int const nprocs = amrex::ParallelDescriptor::NProcs();
        int const local_n = static_cast<int>(flat.size());
        std::vector<int> counts(nprocs, 0);
        MPI_Allgather(&local_n, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);
        std::vector<int> displs(nprocs, 0);
        long global_n = 0;
        for (int rank = 0; rank < nprocs; ++rank) {
            displs[rank] = static_cast<int>(global_n);
            global_n += counts[rank];
        }
        global.resize(static_cast<std::size_t>(global_n));
        MPI_Allgatherv(
            flat.data(), local_n, MPI_DOUBLE,
            global.data(), counts.data(), displs.data(), MPI_DOUBLE, comm);
    }
#else
    global = std::move(flat);
#endif

    // One deterministic global order (field, cell, value) so every rank and
    // every layout performs the identical per-cell summation sequence.
    std::vector<DeferredOffboxDeposit> entries;
    entries.reserve(global.size() / 4U);
    for (std::size_t offset = 0; offset + 3U < global.size(); offset += 4U) {
        entries.push_back(DeferredOffboxDeposit{
            static_cast<int>(global[offset]),
            static_cast<int>(global[offset + 1U]),
            static_cast<int>(global[offset + 2U]),
            global[offset + 3U]});
    }
    std::sort(
        entries.begin(),
        entries.end(),
        [](DeferredOffboxDeposit const& a, DeferredOffboxDeposit const& b) {
            if (a.field_id != b.field_id) { return a.field_id < b.field_id; }
            if (a.i != b.i) { return a.i < b.i; }
            if (a.j != b.j) { return a.j < b.j; }
            return a.value < b.value;
        });
    for (auto const& entry : entries) {
        amrex::MultiFab& field = m_fields[entry.field_id];
        amrex::IntVect const cell(AMREX_D_DECL(entry.i, entry.j, 0));
        for (auto const& route : m_local_fab_routes) {
            if (!route.valid_box.contains(cell)) {
                continue;
            }
            auto const arr = field.array(route.global_index);
            arr(cell[0], cell[1], 0) += entry.value;
            break;
        }
    }
}

void RreaEventAccumulator::NormalizeAndSync(double dt_s)
{
    if (dt_s <= 0.0) {
        throw std::runtime_error("RreaEventAccumulator requires positive dt");
    }
    // Deposits live in the scratch until here; fold them into the fields,
    // then route beyond-ghost deposits to their exact owning cells.  The
    // flush is collective and must run on every rank.
    FoldScratchIntoFields();
    FlushDeferredOffboxDeposits();
    auto const& geom = m_geom;
    auto const dx = geom.CellSizeArray();
    auto const plo = geom.ProbLoArray();
    auto const domain_lo = geom.Domain().smallEnd();
    for (auto& mf : m_fields) {
        // Fold grow-cell deposits (off-rank backtracked interaction points)
        // into the owning valid cells before normalizing.
        mf.SumBoundary(geom.periodicity());
        for (amrex::MFIter mfi(mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const arr = mf.array(mfi);
            amrex::ParallelFor(
                mfi.tilebox(),
                [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                    double const volume = RreaRzShellCellVolume(
                        plo[0], dx[0], dx[1], i, domain_lo[0]);
                    double const denom = volume * dt_s;
                    arr(i, j, k) = (denom > 0.0) ? arr(i, j, k) / denom : 0.0;
                });
        }
        mf.FillBoundary(geom.periodicity());
    }
}

void RreaEventAccumulator::AddToField(
    int field_id,
    double r_m,
    double z_m,
    double value)
{
    if (!std::isfinite(value) || value == 0.0) {
        return;
    }
    auto const& geom = m_geom;
    auto const dx = geom.CellSizeArray();
    auto const plo = geom.ProbLoArray();
    auto const& domain = geom.Domain();
    int const i_containing = domain.smallEnd(0)
        + static_cast<int>(std::floor((r_m - plo[0]) / dx[0]));
    int const j_containing = domain.smallEnd(1)
        + static_cast<int>(std::floor((z_m - plo[1]) / dx[1]));
    amrex::IntVect const containing_cell(
        AMREX_D_DECL(i_containing, j_containing, 0));
    if (!domain.contains(containing_cell)) {
        return;
    }

    // WarpX 26.06 order-1 charge deposition evaluates the linear shape at
    // x_cell = (x-ProbLo)/dx + domain_lo - 1/2 for a cell-centered target.
    // Reproduce that stencil before the common RZ cell-volume normalization.
    // std::floor is intentional: WarpX evaluates its local coordinate after
    // growing the particle tile, so the equivalent global left index is the
    // mathematical floor even for support immediately below the RZ axis.
    auto const cr = rrea::cic_axis(r_m, plo[0], dx[0], domain.smallEnd(0));
    int const i0 = cr.lo;
    double const wr1 = cr.frac;
    double const wr0 = 1.0 - wr1;
    auto const cz = rrea::cic_axis(z_m, plo[1], dx[1], domain.smallEnd(1));
    int const j0 = cz.lo;
    double const wz1 = cz.frac;
    double const wz0 = 1.0 - wz1;

    for (int rz = 0; rz < 2; ++rz) {
        double const wz = rz == 0 ? wz0 : wz1;
        if (wz == 0.0) {
            continue;
        }
        int recipient_j = j0 + rz;
        if (recipient_j < domain.smallEnd(1)
            || recipient_j > domain.bigEnd(1)) {
            // WarpX leaves this support in a physical ghost cell; it is
            // not folded into the valid domain at an absorbing boundary.
            continue;
        }
        for (int rr = 0; rr < 2; ++rr) {
            double const wr = rr == 0 ? wr0 : wr1;
            double const recipient_value = value * wr * wz;
            if (recipient_value == 0.0) {
                continue;
            }
            int recipient_i = i0 + rr;
            if (recipient_i < domain.smallEnd(0)
                || recipient_i > domain.bigEnd(0)) {
                if (recipient_i == domain.smallEnd(0) - 1
                    && plo[0] == 0.0) {
                    // WarpX's RZ inverse-volume pass folds the mode-0 radial
                    // ghost contribution across the axis with positive sign.
                    recipient_i = domain.smallEnd(0);
                } else {
                    continue;
                }
            }
            AddToRecipientCell(field_id, recipient_i, recipient_j, recipient_value);
        }
    }
}

void RreaEventAccumulator::AddToRecipientCell(
    int field_id,
    int i,
    int j,
    double value)
{
    if (!std::isfinite(value) || value == 0.0) {
        return;
    }
    amrex::IntVect const cell(AMREX_D_DECL(i, j, 0));
    auto const& routes = m_local_fab_routes;
    auto& scratch = m_scratch;
    auto& [last, previous] = m_last_valid_route;
    // Valid boxes are disjoint; grown-box precedence must keep its original order.
    if (last < routes.size() && routes[last].valid_box.contains(cell)) {
        scratch[last].array()(cell[0], cell[1], 0, field_id) += value;
        return;
    }

    if (previous < routes.size() && routes[previous].valid_box.contains(cell)) {
        std::swap(last, previous);
        scratch[last].array()(cell[0], cell[1], 0, field_id) += value;
        return;
    }

    for (std::size_t box = 0; box < routes.size(); ++box) {
        if (!routes[box].valid_box.contains(cell)) {
            continue;
        }
        previous = last;
        last = box;
        scratch[box].array()(cell[0], cell[1], 0, field_id) += value;
        return;
    }

    // Backtracked interaction points can land on another rank.  Deposit each
    // order-1 recipient into a reachable grow cell so SumBoundary routes it to
    // the owner; otherwise defer the exact recipient index collectively.
    for (std::size_t box = 0; box < routes.size(); ++box) {
        if (!routes[box].grown_box.contains(cell)) {
            continue;
        }
        scratch[box].array()(cell[0], cell[1], 0, field_id) += value;
        ++m_offrank_ghost_deposit_count;
        return;
    }

    // Beyond even the grown boxes (a deck whose c*dt exceeds n_grow cells): NEVER clamp the deposit
    // into a wrong local cell -- that moves charge/energy and makes results
    // decomposition-dependent.
    // Defer instead; FlushDeferredOffboxDeposits routes it exactly to the
    // owning cell during NormalizeAndSync via one deterministic global order.
    m_deferred_offbox.push_back(DeferredOffboxDeposit{
        field_id, cell[0], cell[1], value});
    ++m_offrank_clamp_count;
}

}  // namespace rrea
