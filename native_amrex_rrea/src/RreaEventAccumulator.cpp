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

// Field ids, in the order FieldById resolves them: low-electron source,
// positive-ion source, direct-negative-ion source, ionization event count,
// energy-loss density.  Also the component count of the deposit scratch.
constexpr int kEventFieldCount = 5;

std::unique_ptr<amrex::MultiFab> make_scalar_field(
    amrex::BoxArray const& ba,
    amrex::DistributionMapping const& dm,
    int n_grow)
{
    return std::make_unique<amrex::MultiFab>(ba, dm, 1, n_grow);
}


struct FoldEventComponent {
    amrex::Array4<amrex::Real const> source;
    amrex::Array4<amrex::Real> target;
    int field_id;
    RREA_HOST_DEVICE void operator()(int i, int j, int k) const noexcept
    { target(i,j,k) += source(i,j,k,field_id); }
};

}  // namespace

RreaEventAccumulator::RreaEventAccumulator(
    amrex::Vector<RreaEventAccumulatorLevelBinding> level_bindings,
    int n_grow)
    : m_levels(std::move(level_bindings))
{
    RequireHybridManagedMemory();
    if (m_levels.empty()) {
        throw std::runtime_error("RreaEventAccumulator requires at least one level");
    }
    m_s_low_electron.reserve(m_levels.size());
    m_s_positive_ion.reserve(m_levels.size());
    m_s_direct_negative_ion.reserve(m_levels.size());
    m_ionization_event_count.reserve(m_levels.size());
    m_energy_loss_density_ev_m3.reserve(m_levels.size());
    m_local_fab_routes.resize(static_cast<int>(m_levels.size()));
    m_scratch.resize(static_cast<int>(m_levels.size()));
    m_deferred_offbox.resize(static_cast<int>(m_levels.size()));
    for (std::size_t lev = 0; lev < m_levels.size(); ++lev) {
        auto const& level = m_levels[lev];
        if (level.box_array.empty()) {
            throw std::runtime_error("RreaEventAccumulator level has empty BoxArray");
        }
        m_s_low_electron.push_back(make_scalar_field(
            level.box_array,
            level.distribution_map,
            n_grow));
        m_s_positive_ion.push_back(make_scalar_field(
            level.box_array,
            level.distribution_map,
            n_grow));
        m_s_direct_negative_ion.push_back(make_scalar_field(
            level.box_array,
            level.distribution_map,
            n_grow));
        m_ionization_event_count.push_back(make_scalar_field(
            level.box_array,
            level.distribution_map,
            n_grow));
        m_energy_loss_density_ev_m3.push_back(make_scalar_field(
            level.box_array,
            level.distribution_map,
            n_grow));
        auto& reference = *m_s_low_electron.back();
        auto same_layout = [&](amrex::MultiFab const& field) {
            return field.boxArray() == reference.boxArray()
                && field.DistributionMap() == reference.DistributionMap()
                && field.nGrow() == reference.nGrow();
        };
        if (!same_layout(*m_s_positive_ion.back())
            || !same_layout(*m_s_direct_negative_ion.back())
            || !same_layout(*m_ionization_event_count.back())
            || !same_layout(*m_energy_loss_density_ev_m3.back())) {
            throw std::runtime_error(
                "RreaEventAccumulator event fields must share one layout");
        }
        auto& routes = m_local_fab_routes[lev];
        auto& scratch = m_scratch[lev];
        routes.reserve(static_cast<std::size_t>(reference.local_size()));
        scratch.reserve(static_cast<std::size_t>(reference.local_size()));
        for (amrex::MFIter mfi(reference); mfi.isValid(); ++mfi) {
            amrex::Box const valid_box = mfi.validbox();
            amrex::Box const grown_box = amrex::grow(valid_box, n_grow);
            routes.push_back(LocalFabRoute{mfi.index(), valid_box, grown_box});
            scratch.emplace_back(grown_box, kEventFieldCount);
        }
    }
    ResetAll();
}

int RreaEventAccumulator::FinestLevel() const noexcept
{
    return static_cast<int>(m_levels.size()) - 1;
}

void RreaEventAccumulator::Reset(int lev)
{
    GpuSynchronize();
    m_s_low_electron.at(lev)->setVal(0.0);
    m_s_positive_ion.at(lev)->setVal(0.0);
    m_s_direct_negative_ion.at(lev)->setVal(0.0);
    m_ionization_event_count.at(lev)->setVal(0.0);
    m_energy_loss_density_ev_m3.at(lev)->setVal(0.0);
    for (auto& fab : m_scratch.at(lev)) {
        fab.setVal<amrex::RunOn::Host>(amrex::Real(0.0));
    }
    m_deferred_offbox.at(lev).clear();
}

void RreaEventAccumulator::ResetAll()
{
    for (int lev = 0; lev <= FinestLevel(); ++lev) {
        Reset(lev);
    }
}

void RreaEventAccumulator::AddLowElectronSource(
    int lev,
    double r_m,
    double z_m,
    double weight)
{
    AddToField(lev, 0, r_m, z_m, weight);
}

void RreaEventAccumulator::AddPositiveIonSource(
    int lev,
    double r_m,
    double z_m,
    double weight)
{
    AddToField(lev, 1, r_m, z_m, weight);
}

void RreaEventAccumulator::AddDirectNegativeIonSource(
    int lev,
    double r_m,
    double z_m,
    double weight)
{
    AddToField(lev, 2, r_m, z_m, weight);
}

void RreaEventAccumulator::AddIonizationDiagnostic(
    int lev,
    double r_m,
    double z_m,
    double weight,
    double energy_loss_eV)
{
    AddToField(lev, 3, r_m, z_m, weight);
    AddToField(
        lev, 4, r_m, z_m,
        std::max(weight, 0.0) * std::max(energy_loss_eV, 0.0));
}

amrex::MultiFab& RreaEventAccumulator::FieldById(int lev, int field_id)
{
    switch (field_id) {
        case 0: return *m_s_low_electron.at(lev);
        case 1: return *m_s_positive_ion.at(lev);
        case 2: return *m_s_direct_negative_ion.at(lev);
        case 3: return *m_ionization_event_count.at(lev);
        case 4: return *m_energy_loss_density_ev_m3.at(lev);
        default: break;
    }
    throw std::runtime_error("RreaEventAccumulator: unknown deferred field id");
}

void RreaEventAccumulator::FoldScratchIntoFields(int lev)
{
    auto const& routes = m_local_fab_routes.at(lev);
    auto& scratch = m_scratch.at(lev);
    for (int field_id = 0; field_id < kEventFieldCount; ++field_id) {
        amrex::MultiFab& field = FieldById(lev, field_id);
        for (std::size_t box = 0; box < routes.size(); ++box) {
            auto const source = scratch[box].const_array();
            auto const target = field.array(routes[box].global_index);
            amrex::ParallelFor(routes[box].grown_box,
                FoldEventComponent{source, target, field_id});
        }
    }
    GpuSynchronize();
    for (auto& fab : scratch) {
        fab.setVal<amrex::RunOn::Host>(amrex::Real(0.0));
    }
}

void RreaEventAccumulator::FlushDeferredOffboxDeposits(int lev)
{
    auto& deferred = m_deferred_offbox.at(lev);
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
        amrex::MultiFab& field = FieldById(lev, entry.field_id);
        amrex::IntVect const cell(AMREX_D_DECL(entry.i, entry.j, 0));
        for (auto const& route : m_local_fab_routes.at(lev)) {
            if (!route.valid_box.contains(cell)) {
                continue;
            }
            auto const arr = field.array(route.global_index);
            arr(cell[0], cell[1], 0) += entry.value;
            break;
        }
    }
}

void RreaEventAccumulator::NormalizeAndSync(
    int lev,
    double dt_s,
    bool normalize_diagnostic_fields,
    bool fill_boundaries)
{
    if (dt_s <= 0.0) {
        throw std::runtime_error("RreaEventAccumulator requires positive dt");
    }
    // Deposits live in the scratch until here; fold them into the fields,
    // then route beyond-ghost deposits to their exact owning cells.  The
    // flush is collective and must run on every rank.
    FoldScratchIntoFields(lev);
    FlushDeferredOffboxDeposits(lev);
    auto const& geom = m_levels.at(lev).geom;
    auto const dx = geom.CellSizeArray();
    auto const plo = geom.ProbLoArray();
    auto const domain_lo = geom.Domain().smallEnd();

    auto normalize_source = [&](amrex::MultiFab& mf, bool divide_by_dt) {
        // Fold grow-cell deposits (off-rank
        // backtracked interaction points) into the owning valid cells before
        // normalizing. When nothing landed in a grow cell this has no numerical
        // effect.
        mf.SumBoundary(geom.periodicity());
        for (amrex::MFIter mfi(mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const bx = mfi.tilebox();
            auto const arr = mf.array(mfi);
            amrex::ParallelFor(
                bx,
#if (AMREX_SPACEDIM >= 2)
                [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                    double const volume = RreaRzShellCellVolume(
                        plo[0], dx[0], dx[1], i, domain_lo[0]);
#else
                [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                    double const volume = dx[0];
#endif
                    double const denom = divide_by_dt ? volume * dt_s : volume;
                    arr(i, j, k) = (denom > 0.0) ? arr(i, j, k) / denom : 0.0;
                });
        }
        if (fill_boundaries) {
            mf.FillBoundary(geom.periodicity());
        }
    };

    normalize_source(*m_s_low_electron.at(lev), true);
    normalize_source(*m_s_positive_ion.at(lev), true);
    normalize_source(*m_s_direct_negative_ion.at(lev), true);
    if (normalize_diagnostic_fields) {
        normalize_source(*m_ionization_event_count.at(lev), false);
        normalize_source(*m_energy_loss_density_ev_m3.at(lev), false);
    }
}

amrex::MultiFab& RreaEventAccumulator::LowElectronSourceRate(int lev)
{
    return *m_s_low_electron.at(lev);
}

amrex::MultiFab& RreaEventAccumulator::PositiveIonSourceRate(int lev)
{
    return *m_s_positive_ion.at(lev);
}

amrex::MultiFab& RreaEventAccumulator::DirectNegativeIonSourceRate(int lev)
{
    return *m_s_direct_negative_ion.at(lev);
}

amrex::MultiFab& RreaEventAccumulator::IonizationEventCount(int lev)
{
    return *m_ionization_event_count.at(lev);
}

amrex::MultiFab& RreaEventAccumulator::EnergyLossDensityEvPerM3(int lev)
{
    return *m_energy_loss_density_ev_m3.at(lev);
}

amrex::MultiFab const& RreaEventAccumulator::LowElectronSourceRate(int lev) const
{
    return *m_s_low_electron.at(lev);
}

amrex::MultiFab const& RreaEventAccumulator::PositiveIonSourceRate(int lev) const
{
    return *m_s_positive_ion.at(lev);
}

amrex::MultiFab const& RreaEventAccumulator::DirectNegativeIonSourceRate(int lev) const
{
    return *m_s_direct_negative_ion.at(lev);
}

amrex::MultiFab const& RreaEventAccumulator::IonizationEventCount(int lev) const
{
    return *m_ionization_event_count.at(lev);
}

amrex::MultiFab const& RreaEventAccumulator::EnergyLossDensityEvPerM3(int lev) const
{
    return *m_energy_loss_density_ev_m3.at(lev);
}

void RreaEventAccumulator::AddToField(
    int lev,
    int field_id,
    double r_m,
    double z_m,
    double value)
{
    if (lev < 0 || lev > FinestLevel() || !std::isfinite(value) || value == 0.0) {
        return;
    }
    auto const& geom = m_levels.at(lev).geom;
    auto const dx = geom.CellSizeArray();
    auto const plo = geom.ProbLoArray();
    auto const phi = geom.ProbHiArray();
    auto const& domain = geom.Domain();

    // Match AMReX particle ownership at a periodic seam: ProbHi is the same
    // physical point as ProbLo, and positions displaced by whole domain
    // lengths are canonicalized before WarpX deposits them.  The per-cell
    // support below still wraps independently when a shape straddles a seam.
    auto canonical_periodic_position = [&](double position, int dim) {
        if (!geom.isPeriodic(dim)) {
            return position;
        }
        double const length = phi[dim] - plo[dim];
        double shifted = std::fmod(position - plo[dim], length);
        if (shifted < 0.0) {
            shifted += length;
        }
        return plo[dim] + shifted;
    };
    double const deposit_r_m = canonical_periodic_position(r_m, 0);
#if (AMREX_SPACEDIM >= 2)
    double const deposit_z_m = canonical_periodic_position(z_m, 1);
#else
    double const deposit_z_m = z_m;
#endif
    int const i_containing = domain.smallEnd(0)
        + static_cast<int>(std::floor((deposit_r_m - plo[0]) / dx[0]));
#if (AMREX_SPACEDIM >= 2)
    int const j_containing = domain.smallEnd(1)
        + static_cast<int>(std::floor((deposit_z_m - plo[1]) / dx[1]));
#else
    int const j_containing = 0;
#endif
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
    double const r_cell = (deposit_r_m - plo[0]) / dx[0]
        + static_cast<double>(domain.smallEnd(0)) - 0.5;
    int const i0 = static_cast<int>(std::floor(r_cell));
    double const wr1 = r_cell - static_cast<double>(i0);
    double const wr0 = 1.0 - wr1;
#if (AMREX_SPACEDIM >= 2)
    double const z_cell = (deposit_z_m - plo[1]) / dx[1]
        + static_cast<double>(domain.smallEnd(1)) - 0.5;
    int const j0 = static_cast<int>(std::floor(z_cell));
    double const wz1 = z_cell - static_cast<double>(j0);
    double const wz0 = 1.0 - wz1;
#else
    int const j0 = 0;
    double const wz0 = 1.0;
    double const wz1 = 0.0;
#endif

    auto wrap_periodic = [&](int index, int dim) {
        int const lo = domain.smallEnd(dim);
        int const length = domain.length(dim);
        int shifted = (index - lo) % length;
        if (shifted < 0) {
            shifted += length;
        }
        return lo + shifted;
    };

    for (int rz = 0; rz < 2; ++rz) {
        double const wz = rz == 0 ? wz0 : wz1;
        if (wz == 0.0) {
            continue;
        }
        int recipient_j = j0 + rz;
#if (AMREX_SPACEDIM >= 2)
        if (recipient_j < domain.smallEnd(1)
            || recipient_j > domain.bigEnd(1)) {
            if (geom.isPeriodic(1)) {
                recipient_j = wrap_periodic(recipient_j, 1);
            } else {
                // WarpX leaves this support in a physical ghost cell; it is
                // not folded into the valid domain at an absorbing boundary.
                continue;
            }
        }
#endif
        for (int rr = 0; rr < 2; ++rr) {
            double const wr = rr == 0 ? wr0 : wr1;
            double const recipient_value = value * wr * wz;
            if (recipient_value == 0.0) {
                continue;
            }
            int recipient_i = i0 + rr;
            if (recipient_i < domain.smallEnd(0)
                || recipient_i > domain.bigEnd(0)) {
                if (geom.isPeriodic(0)) {
                    recipient_i = wrap_periodic(recipient_i, 0);
                } else if (recipient_i == domain.smallEnd(0) - 1
                    && plo[0] == 0.0) {
                    // WarpX's RZ inverse-volume pass folds the mode-0 radial
                    // ghost contribution across the axis with positive sign.
                    recipient_i = domain.smallEnd(0);
                } else {
                    continue;
                }
            }
            AddToRecipientCell(
                lev, field_id, recipient_i, recipient_j, recipient_value);
        }
    }
}

void RreaEventAccumulator::AddToRecipientCell(
    int lev,
    int field_id,
    int i,
    int j,
    double value)
{
    if (!std::isfinite(value) || value == 0.0) {
        return;
    }
    amrex::IntVect const cell(AMREX_D_DECL(i, j, 0));
    auto const& routes = m_local_fab_routes.at(lev);
    auto& scratch = m_scratch.at(lev);

    for (std::size_t box = 0; box < routes.size(); ++box) {
        if (!routes[box].valid_box.contains(cell)) {
            continue;
        }
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

    // Beyond even the grown boxes (single-step chords in certification beams,
    // or a deck whose c*dt exceeds n_grow cells): NEVER clamp the deposit
    // into a wrong local cell -- that moves charge/energy and makes results
    // decomposition-dependent.
    // Defer instead; FlushDeferredOffboxDeposits routes it exactly to the
    // owning cell during NormalizeAndSync via one deterministic global order.
    m_deferred_offbox.at(lev).push_back(DeferredOffboxDeposit{
        field_id, cell[0], cell[1], value});
    ++m_offrank_clamp_count;
}

}  // namespace rrea
