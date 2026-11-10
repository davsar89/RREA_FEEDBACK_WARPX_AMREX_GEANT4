#include "rrea/RreaSmokeDomain.H"
#include "rrea/RreaGpuSmoke.H"
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
// Cases 1-5 use small powers of two so sums are exact in any order.
// Case 6 tests an order-sensitive sum on one owning rank.
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

using rrea::smoke::TestDomain;

// The domain spans nz cells; only the first nz_boxes rows carry boxes.
TestDomain make_domain(int nr, int nz, int nz_boxes)
{
    auto const geom = rrea::smoke::MakeRzGeometry(nr, nz, nr, nz);
    amrex::IntVect hi(0);
    hi[0] = nr - 1;
    hi[1] = nz_boxes - 1;
    amrex::BoxArray ba(amrex::Box(amrex::IntVect(0), hi));
    ba.maxSize(8);
    return {geom, ba, amrex::DistributionMapping(ba)};
}

rrea::RreaEventAccumulator make_accumulator(TestDomain const& domain)
{
    return rrea::RreaEventAccumulator(domain.geom, domain.ba, domain.dm, kNGrow);
}

// Unpacks the geometry for the production shell-volume helper so the smoke
// normalizes with the exact expression NormalizeAndSync uses.
double cell_volume(amrex::Geometry const& geom, int i)
{
    auto const dx = geom.CellSizeArray();
    return rrea::RreaRzShellCellVolume(
        geom.ProbLoArray()[0], dx[0], dx[1], i, geom.Domain().smallEnd(0));
}

struct Deposit {
    int field_id;  // 0 low-e, 1 pos-ion
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
            events.AddLowElectronSource(r_m, z_m, deposit.value);
            return;
        case 1:
            events.AddPositiveIonSource(r_m, z_m, deposit.value);
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
        case 0: events.AddLowElectronSource(r_m, z_m, value); return;
        case 1: events.AddPositiveIonSource(r_m, z_m, value); return;
        default: throw std::runtime_error("unsupported position-deposit field");
    }
}

amrex::MultiFab const& field_by_id(
    rrea::RreaEventAccumulator const& events, int field_id)
{
    switch (field_id) {
        case 0: return events.LowElectronSourceRate();
        case 1: return events.PositiveIonSourceRate();
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
    for (int field_id = 0; field_id < 2; ++field_id) {
        amrex::MultiFab const& field = field_by_id(events, field_id);
        for (amrex::MFIter mfi(field); mfi.isValid(); ++mfi) {
            auto const bx = mfi.validbox();
            auto const arr = field.const_array(mfi);
            rrea::GpuSynchronize();
        amrex::LoopOnCpu(bx, [&](int i, int j, int k) noexcept {
                double expected = 0.0;
                auto const found = sums.find({field_id, i, j});
                if (found != sums.end()) {
                    expected = found->second / (cell_volume(geom, i) * kDt);
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
        {1, 7, 20, 0.25}, {1, 2, 52, 1.0},
    };
    if (amrex::ParallelDescriptor::IOProcessor()) {
        for (auto const& deposit : deposits) {
            apply_deposit(events, deposit);
        }
    }
    events.NormalizeAndSync(kDt);
    require_exact_fields(events, domain.geom, deposits, "layout independence");

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
    int const target_j = center[1];
    // Rank 0 owns neither the target box nor (with 8-cell boxes and 2 grow
    // cells) any grown box reaching its center: the deposits must defer.
    std::vector<Deposit> const deposits = {
        {0, target_i, target_j, 2.0},
        {1, target_i, target_j, 0.5},
    };
    if (my_rank == 0) {
        for (auto const& deposit : deposits) {
            apply_deposit(events, deposit);
        }
    }
    require(
        global_count(events.OffrankClampCount()) == 3,
        "deep off-rank deposits did not all take the deferred route");
    events.NormalizeAndSync(kDt);
    require_exact_fields(events, domain.geom, deposits, "forced off-rank routing");
    require(
        global_count(events.OffrankClampCount()) == 3,
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
        events.AddLowElectronSource(4.5, 40.5, 1.0);
    }
    require(
        global_count(events.OffrankClampCount()) == 1,
        "uncovered in-domain deposit was not deferred");
    events.NormalizeAndSync(kDt);
    require_exact_fields(events, domain.geom, {}, "orphan deposit drop");

    if (amrex::ParallelDescriptor::IOProcessor()) {
        events.AddLowElectronSource(4.5, 40.5, 2.0);
    }
    events.Reset();
    events.NormalizeAndSync(kDt);
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
    events.NormalizeAndSync(kDt);
    require_exact_fields(events, domain.geom, expected, "order-1 stencil");
}

// Case 5: colocated opposite source charges must have identical cell weights,
// hence cancel cell-by-cell when the fluid charge density combines them.
void run_colocated_charge_cancellation_case()
{
    auto const domain = make_domain(4, 4, 4);
    auto events = make_accumulator(domain);
    if (amrex::ParallelDescriptor::IOProcessor()) {
        apply_position_deposit(events, 0, 2.0, 2.0, 8.0);
        apply_position_deposit(events, 1, 2.0, 2.0, 8.0);
    }
    events.NormalizeAndSync(kDt);
    amrex::Long mismatches = 0;
    auto const& negative_electron = events.LowElectronSourceRate();
    auto const& positive_ion = events.PositiveIonSourceRate();
    for (amrex::MFIter mfi(positive_ion); mfi.isValid(); ++mfi) {
        auto const bx = mfi.validbox();
        auto const pos = positive_ion.const_array(mfi);
        auto const electron = negative_electron.const_array(mfi);
        rrea::GpuSynchronize();
        amrex::LoopOnCpu(bx, [&](int i, int j, int k) noexcept {
            if (pos(i, j, k) != electron(i, j, k)) {
                ++mismatches;
            }
        });
    }
    amrex::ParallelDescriptor::ReduceLongSum(mismatches);
    require(mismatches == 0, "colocated charges do not cancel cell-by-cell");
}

// Case 6: a warm valid-route hint across reset, overlapping grow cells,
// order-sensitive additions and an emitting rank with no local FABs.
void run_warm_route_and_reset_case()
{
    auto domain = make_domain(8, 32, 32);
    require(domain.ba.size() == 4, "warm-route fixture needs four boxes");
    domain.dm = amrex::DistributionMapping(amrex::Vector<int>(4, 0));
    auto events = make_accumulator(domain);
    std::vector<Deposit> const deposits = {
        {1, 2, 10, 4.0}, {1, 2, 3, 2.0},
        // Alternate cached boxes, then evict both with two more valid boxes.
        {1, 2, 10, 2.0}, {1, 2, 3, 2.0},
        {1, 2, 18, 4.0}, {1, 2, 26, 4.0}, {1, 2, 3, 2.0},
        {0, 2, 7, 0x1p53}, {1, 2, 10, 2.0},
        {0, 2, 7, 1.0}, {0, 2, 7, 1.0},
    };
    std::vector<Deposit> expected = deposits;
    expected.insert(expected.end(), {
        {0, 3, 7, 2.0}, {0, 4, 7, 2.0},
        {0, 3, 8, 2.0}, {0, 4, 8, 2.0},
    });
    for (int repeat = 0; repeat < 2; ++repeat) {
        events.Reset();
        if (amrex::ParallelDescriptor::MyProc() == 0) {
            for (auto const& deposit : deposits) apply_deposit(events, deposit);
            // Four CIC recipients straddle the boxes; finish in box 1.
            apply_position_deposit(events, 0, 4.0, 8.0, 8.0);
        }
        require(global_count(events.OffrankGhostDepositCount()) == 0,
            "local valid recipient incorrectly took a ghost route");
        require(global_count(events.OffrankClampCount()) == 0,
            "local valid recipient incorrectly deferred");
        events.NormalizeAndSync(kDt);
        require_exact_fields(events, domain.geom, expected, "warm route/reset");
    }
    if (amrex::ParallelDescriptor::NProcs() > 1) {
        events.Reset();
        if (amrex::ParallelDescriptor::MyProc() == 1)
            apply_deposit(events, {0, 2, 10, 0.5});
        require(global_count(events.OffrankClampCount()) == 1,
            "empty-rank deposit was not deferred");
        events.NormalizeAndSync(kDt);
        require_exact_fields(events, domain.geom, {{0, 2, 10, 0.5}}, "empty-rank emitter");
    }
}

}  // namespace

int main(int argc, char* argv[])
{
    rrea::smoke::InitializeAmrexSmoke(argc, argv);
    int status = 0;
    {
        try {
            run_layout_independence_case();
            run_forced_offrank_case();
            run_partial_coverage_and_reset_case();
            run_order1_stencil_case();
            run_colocated_charge_cancellation_case();
            run_warm_route_and_reset_case();
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
