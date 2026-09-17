// Deposit-routing smoke for RreaEventAccumulator. Beyond-ghost deposits are
// deferred and routed to the owning cell in one deterministic global order;
// they are never clamped into a local cell.
//
// The smoke runs at ANY MPI rank count and asserts bitwise the SAME per-cell
// values against an analytic expectation that is independent of the layout:
//  * at 1 rank every deposit lands directly in a local valid box;
//  * at 2+ ranks the same fixed event list (recorded entirely by rank 0)
//    exercises the direct, grow-cell/SumBoundary, and deferred/Allgatherv
//    routing paths, and a dedicated case forces a deep off-rank deposit.
// All deposit values are powers of two so per-cell sums are exact in any
// summation order; the layout independence check is therefore exact equality,
// not a tolerance.
//
// Local serial AMReX builds compile the non-MPI branch of the flush; run this
// smoke against an MPI AMReX build with mpiexec -n 2 / -n 4 to exercise the
// MPI_Allgather/MPI_Allgatherv routing collective itself.

#include "rrea/RreaEventAccumulator.H"
#include "rrea/RreaSmokeRequire.H"

#include <AMReX.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

constexpr int kNGrow = 2;
constexpr double kDt = 0.5;

struct TestDomain {
    amrex::Geometry geom;
    amrex::BoxArray ba;
    amrex::DistributionMapping dm;
};

TestDomain make_domain(int nr, int nz, int nz_boxes, bool periodic_z = false)
{
    amrex::IntVect lo(0);
    amrex::IntVect hi(0);
    hi[0] = nr - 1;
#if (AMREX_SPACEDIM >= 2)
    hi[1] = nz_boxes - 1;
#endif
    amrex::Box covered(lo, hi);
    amrex::IntVect domain_hi(0);
    domain_hi[0] = nr - 1;
#if (AMREX_SPACEDIM >= 2)
    domain_hi[1] = nz - 1;
#endif
    amrex::Box domain(lo, domain_hi);
    amrex::RealBox real_box;
    real_box.setLo(0, 0.0);
    real_box.setHi(0, static_cast<double>(nr));
#if (AMREX_SPACEDIM >= 2)
    real_box.setLo(1, 0.0);
    real_box.setHi(1, static_cast<double>(nz));
#endif
    amrex::Vector<int> periodic(AMREX_SPACEDIM, 0);
#if (AMREX_SPACEDIM >= 2)
    periodic[1] = periodic_z ? 1 : 0;
#else
    amrex::ignore_unused(periodic_z);
#endif
    amrex::Geometry geom(domain, &real_box, 0, periodic.data());
    amrex::BoxArray ba(covered);
    ba.maxSize(8);
    amrex::DistributionMapping dm(ba);
    return {geom, ba, dm};
}

rrea::RreaEventAccumulator make_accumulator(TestDomain const& domain)
{
    amrex::Vector<rrea::RreaEventAccumulatorLevelBinding> bindings;
    bindings.push_back({domain.geom, domain.ba, domain.dm});
    return rrea::RreaEventAccumulator(std::move(bindings), kNGrow);
}

// Unpacks the geometry for the production shell-volume helper so the smoke
// normalizes with the exact expression NormalizeAndSync uses.
double cell_volume(amrex::Geometry const& geom, int i)
{
    auto const dx = geom.CellSizeArray();
#if (AMREX_SPACEDIM >= 2)
    return rrea::RreaRzShellCellVolume(
        geom.ProbLoArray()[0], dx[0], dx[1], i, geom.Domain().smallEnd(0));
#else
    return dx[0];
#endif
}

struct Deposit {
    int field_id;  // 0 low-e, 1 pos-ion, 2 neg-ion, 3 event count, 4 energy loss
    int i;
    int j;
    double value;
};

void apply_deposit(
    rrea::RreaEventAccumulator& events, Deposit const& deposit)
{
    double const r_m = static_cast<double>(deposit.i) + 0.5;
    double const z_m = static_cast<double>(deposit.j) + 0.5;
    switch (deposit.field_id) {
        case 0:
            events.AddLowElectronSource(0, r_m, z_m, deposit.value);
            return;
        case 1:
            events.AddPositiveIonSource(0, r_m, z_m, deposit.value);
            return;
        case 2:
            events.AddDirectNegativeIonSource(0, r_m, z_m, deposit.value);
            return;
        default:
            throw std::runtime_error("unsupported smoke deposit field");
    }
}

void apply_position_deposit(
    rrea::RreaEventAccumulator& events,
    int field_id,
    double r_m,
    double z_m,
    double value)
{
    switch (field_id) {
        case 0: events.AddLowElectronSource(0, r_m, z_m, value); return;
        case 1: events.AddPositiveIonSource(0, r_m, z_m, value); return;
        case 2: events.AddDirectNegativeIonSource(0, r_m, z_m, value); return;
        default: throw std::runtime_error("unsupported position-deposit field");
    }
}

amrex::MultiFab const& field_by_id(
    rrea::RreaEventAccumulator const& events, int field_id)
{
    switch (field_id) {
        case 0: return events.LowElectronSourceRate(0);
        case 1: return events.PositiveIonSourceRate(0);
        case 2: return events.DirectNegativeIonSourceRate(0);
        case 3: return events.IonizationEventCount(0);
        case 4: return events.EnergyLossDensityEvPerM3(0);
        default: break;
    }
    throw std::runtime_error("unknown smoke field id");
}

using rrea::smoke::require;

// Compares every locally-owned valid cell of every field against the exact
// analytic expectation and requires the mismatch count to be globally zero.
void require_exact_fields(
    rrea::RreaEventAccumulator const& events,
    amrex::Geometry const& geom,
    std::vector<Deposit> const& deposits,
    std::string const& label)
{
    std::map<std::tuple<int, int, int>, double> sums;
    for (auto const& deposit : deposits) {
        sums[{deposit.field_id, deposit.i, deposit.j}] += deposit.value;
    }
    amrex::Long mismatches = 0;
    for (int field_id = 0; field_id <= 4; ++field_id) {
        amrex::MultiFab const& field = field_by_id(events, field_id);
        bool const divide_by_dt = field_id < 3;
        for (amrex::MFIter mfi(field); mfi.isValid(); ++mfi) {
            auto const bx = mfi.validbox();
            auto const arr = field.const_array(mfi);
            amrex::LoopOnCpu(bx, [&](int i, int j, int k) noexcept {
                double expected = 0.0;
                auto const found = sums.find({field_id, i, j});
                if (found != sums.end()) {
                    double const volume = cell_volume(geom, i);
                    double const denom = divide_by_dt ? volume * kDt : volume;
                    expected = found->second / denom;
                }
                if (arr(i, j, k) != expected) {
                    ++mismatches;
                }
            });
        }
    }
    amrex::ParallelDescriptor::ReduceLongSum(mismatches);
    require(mismatches == 0, label + ": per-cell values are not exact");
}

amrex::Long global_count(std::uint64_t value)
{
    amrex::Long count = static_cast<amrex::Long>(value);
    amrex::ParallelDescriptor::ReduceLongSum(count);
    return count;
}

// Case 1: rank 0 records one fixed event list across the whole domain.  At a
// single rank every deposit is direct; at 2+ ranks the identical list flows
// through the grow-cell and deferred routes, and the result must be bitwise
// the same analytic field either way.
void run_layout_independence_case()
{
    auto const domain = make_domain(8, 64, 64);
    auto events = make_accumulator(domain);

    std::vector<Deposit> const deposits = {
        {0, 2, 3, 1.0}, {0, 2, 3, 0.5}, {0, 5, 12, 2.0},
        {0, 1, 27, 4.0}, {0, 6, 45, 0.25}, {0, 3, 61, 8.0},
        {1, 4, 8, 1.0}, {1, 4, 9, 0.5}, {1, 0, 33, 2.0},
        {2, 7, 20, 0.25}, {2, 2, 52, 1.0},
    };
    if (amrex::ParallelDescriptor::IOProcessor()) {
        for (auto const& deposit : deposits) {
            apply_deposit(events, deposit);
        }
        // Diagnostic pair: weight 0.5 at (r=3.5, z=18.5) with 8 eV loss.
        events.AddIonizationDiagnostic(0, 3.5, 18.5, 0.5, 8.0);
    }
    std::vector<Deposit> expected = deposits;
    expected.push_back({3, 3, 18, 0.5});
    expected.push_back({4, 3, 18, 4.0});

    events.NormalizeAndSync(0, kDt);
    require_exact_fields(events, domain.geom, expected, "layout independence");

    if (amrex::ParallelDescriptor::NProcs() == 1) {
        require(
            events.OffrankGhostDepositCount() == 0
                && events.OffrankClampCount() == 0,
            "single-rank run must not route any deposit off-box");
    }
}

// Case 2 (2+ ranks): rank A deposits into the deep interior of a box owned by
// another rank, beyond A's grow cells, forcing the deferred/Allgatherv route.
void run_forced_offrank_case()
{
    if (amrex::ParallelDescriptor::NProcs() < 2) {
        return;
    }
    auto const domain = make_domain(8, 64, 64);
    auto events = make_accumulator(domain);

    int const my_rank = amrex::ParallelDescriptor::MyProc();
    int target_box = -1;
    for (int index = 0; index < static_cast<int>(domain.ba.size()); ++index) {
        if (domain.dm[index] != 0) {
            target_box = index;
            break;
        }
    }
    require(target_box >= 0, "distribution map assigns every box to rank 0");
    amrex::Box const box = domain.ba[target_box];
    amrex::IntVect const center = (box.smallEnd() + box.bigEnd()) / 2;
    int const target_i = center[0];
#if (AMREX_SPACEDIM >= 2)
    int const target_j = center[1];
#else
    int const target_j = 0;
#endif
    // Rank 0 owns neither the target box nor (with 8-cell boxes and 2 grow
    // cells) any grown box reaching its center: the deposits must defer.
    std::vector<Deposit> const deposits = {
        {0, target_i, target_j, 2.0},
        {1, target_i, target_j, 0.5},
        {2, target_i, target_j, 4.0},
    };
    if (my_rank == 0) {
        for (auto const& deposit : deposits) {
            apply_deposit(events, deposit);
        }
        events.AddIonizationDiagnostic(
            0,
            static_cast<double>(target_i) + 0.5,
            static_cast<double>(target_j) + 0.5,
            1.0,
            2.0);
    }
    std::vector<Deposit> expected = deposits;
    expected.push_back({3, target_i, target_j, 1.0});
    expected.push_back({4, target_i, target_j, 2.0});

    require(
        global_count(events.OffrankClampCount()) == 5,
        "deep off-rank deposits did not all take the deferred route");
    events.NormalizeAndSync(0, kDt);
    require_exact_fields(events, domain.geom, expected, "forced off-rank routing");
    require(
        global_count(events.OffrankClampCount()) == 5,
        "deferred routing changed the deferred-deposit counter");
}

// Case 3: a BoxArray that does not cover the whole domain (impossible in
// production, where the fields tile the domain) exercises the flush with a
// deposit no rank owns: it must complete collectively without crashing, drop
// the orphan deposit, and Reset must clear any still-deferred entries.
void run_partial_coverage_and_reset_case()
{
    auto const domain = make_domain(8, 64, 32);
    auto events = make_accumulator(domain);

    if (amrex::ParallelDescriptor::IOProcessor()) {
        events.AddLowElectronSource(0, 4.5, 40.5, 1.0);
    }
    require(
        global_count(events.OffrankClampCount()) == 1,
        "uncovered in-domain deposit was not deferred");
    events.NormalizeAndSync(0, kDt);
    require_exact_fields(events, domain.geom, {}, "orphan deposit drop");

    if (amrex::ParallelDescriptor::IOProcessor()) {
        events.AddLowElectronSource(0, 4.5, 40.5, 2.0);
    }
    events.Reset(0);
    events.NormalizeAndSync(0, kDt);
    require_exact_fields(events, domain.geom, {}, "reset clears deferred queue");
}

// Case 4: the event-source stencil must be the cell-centered order-1 shape
// used by WarpX DepositCharge, including RZ axis folding and physical-boundary
// loss.  Dyadic positions/weights keep the expected cell sums exact.
void run_order1_stencil_case()
{
    auto const domain = make_domain(4, 4, 4);
    auto events = make_accumulator(domain);
    if (amrex::ParallelDescriptor::IOProcessor()) {
        apply_position_deposit(events, 0, 1.5, 1.5, 1.0);  // cell center
        apply_position_deposit(events, 0, 2.0, 1.5, 2.0);  // radial face
        apply_position_deposit(events, 0, 1.5, 2.0, 4.0);  // axial face
        apply_position_deposit(events, 0, 2.0, 2.0, 8.0);  // corner
        apply_position_deposit(events, 0, 0.0, 1.5, 16.0); // RZ axis
        apply_position_deposit(events, 0, 2.5, 0.25, 32.0); // absorbing low-z
        apply_position_deposit(events, 0, 2.5, 3.75, 64.0); // absorbing high-z
    }
    std::vector<Deposit> const expected = {
        {0, 1, 1, 1.0},
        {0, 1, 1, 1.0}, {0, 2, 1, 1.0},
        {0, 1, 1, 2.0}, {0, 1, 2, 2.0},
        {0, 1, 1, 2.0}, {0, 2, 1, 2.0},
        {0, 1, 2, 2.0}, {0, 2, 2, 2.0},
        {0, 0, 1, 16.0},
        {0, 2, 0, 24.0},
        {0, 2, 3, 48.0},
    };
    events.NormalizeAndSync(0, kDt);
    require_exact_fields(events, domain.geom, expected, "order-1 stencil");
}

// Case 5: axial order-1 support wraps across a periodic seam instead of being
// clamped or discarded.
void run_periodic_stencil_case()
{
    auto const domain = make_domain(4, 4, 4, true);
    auto events = make_accumulator(domain);
    if (amrex::ParallelDescriptor::IOProcessor()) {
        apply_position_deposit(events, 0, 1.5, 0.25, 4.0);
        apply_position_deposit(events, 0, 2.5, 3.75, 8.0);
        apply_position_deposit(events, 0, 1.5, 4.0, 16.0);
    }
    std::vector<Deposit> const expected = {
        {0, 1, 3, 1.0}, {0, 1, 0, 3.0},
        {0, 2, 3, 6.0}, {0, 2, 0, 2.0},
        {0, 1, 3, 8.0}, {0, 1, 0, 8.0},
    };
    events.NormalizeAndSync(0, kDt);
    require_exact_fields(events, domain.geom, expected, "periodic order-1 seam");
}

// Case 6: colocated opposite source charges must have identical cell weights,
// hence cancel cell-by-cell when the fluid charge density combines them.
void run_colocated_charge_cancellation_case()
{
    auto const domain = make_domain(4, 4, 4);
    auto events = make_accumulator(domain);
    if (amrex::ParallelDescriptor::IOProcessor()) {
        apply_position_deposit(events, 0, 2.0, 2.0, 8.0);
        apply_position_deposit(events, 1, 2.0, 2.0, 8.0);
        apply_position_deposit(events, 2, 2.0, 2.0, 8.0);
    }
    events.NormalizeAndSync(0, kDt);
    amrex::Long mismatches = 0;
    auto const& negative_electron = events.LowElectronSourceRate(0);
    auto const& positive_ion = events.PositiveIonSourceRate(0);
    auto const& negative_ion = events.DirectNegativeIonSourceRate(0);
    for (amrex::MFIter mfi(positive_ion); mfi.isValid(); ++mfi) {
        auto const bx = mfi.validbox();
        auto const pos = positive_ion.const_array(mfi);
        auto const electron = negative_electron.const_array(mfi);
        auto const ion = negative_ion.const_array(mfi);
        amrex::LoopOnCpu(bx, [&](int i, int j, int k) noexcept {
            if (pos(i, j, k) != electron(i, j, k)
                || pos(i, j, k) != ion(i, j, k)) {
                ++mismatches;
            }
        });
    }
    amrex::ParallelDescriptor::ReduceLongSum(mismatches);
    require(mismatches == 0, "colocated charges do not cancel cell-by-cell");
}

}  // namespace

int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    int status = 0;
    {
        try {
            run_layout_independence_case();
            run_forced_offrank_case();
            run_partial_coverage_and_reset_case();
            run_order1_stencil_case();
            run_periodic_stencil_case();
            run_colocated_charge_cancellation_case();
            if (amrex::ParallelDescriptor::IOProcessor()) {
                std::cout
                    << "RREA event accumulator smoke passed on "
                    << amrex::ParallelDescriptor::NProcs() << " rank(s)\n";
            }
        } catch (std::exception const& error) {
            std::cerr << "RREA event accumulator smoke failed: "
                      << error.what() << '\n';
            status = 1;
        }
    }
    amrex::Finalize();
    return status;
}
