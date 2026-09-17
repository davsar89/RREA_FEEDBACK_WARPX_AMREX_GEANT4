#include "rrea/RreaGpuSmoke.H"
#include "rrea/RreaFieldGatherStencil.H"
#include "rrea/RreaSmokeRequire.H"

#include <AMReX.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_Loop.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_REAL.H>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

using rrea::smoke::require;
using rrea::smoke::require_close;

// C&D plane -> bracketing-rows mapping.  Independent references: exact
// hand-stated outcomes at cell centers and midpoints, and edge clamping to
// (edge, edge, 0).  The fractional coordinate itself is the already-pinned
// RreaCellCenteredGatherIndexCoordinate, so the two call sites (field
// gather, plane flux) share one tested mapping.
void exercise_axial_plane_interpolation()
{
    using rrea::RreaAxialPlaneInterpolationFor;
    // Interior plane exactly on a cell center: weight 0, rows (j, j+1).
    auto const on_center = RreaAxialPlaneInterpolationFor(9.0, 7, 22);
    require(on_center.lower == 9 && on_center.upper == 10
                && on_center.upper_weight == 0.0,
        "plane on a cell center must have zero upper weight");
    // Exactly midway: weight one-half, exact in FP.
    auto const midway = RreaAxialPlaneInterpolationFor(9.5, 7, 22);
    require(midway.lower == 9 && midway.upper == 10
                && midway.upper_weight == 0.5,
        "plane midway between centers must split exactly one-half");
    // At and beyond both physical edges: collapse to (edge, edge, 0).
    for (double c : {7.0, 6.25, -3.0}) {
        auto const lo = RreaAxialPlaneInterpolationFor(
            static_cast<amrex::Real>(c), 7, 22);
        require(lo.lower == 7 && lo.upper == 7 && lo.upper_weight == 0.0,
            "plane at/below the low edge must clamp one-sided");
    }
    for (double c : {22.0, 22.75, 400.0}) {
        auto const hi = RreaAxialPlaneInterpolationFor(
            static_cast<amrex::Real>(c), 7, 22);
        require(hi.lower == 22 && hi.upper == 22 && hi.upper_weight == 0.0,
            "plane at/above the high edge must clamp one-sided");
    }
    // Linear-reproduction identity of the (1-w, w) split: interpolating
    // f(j) = a + b*j at the plane must give a + b*coordinate exactly for
    // the exactly-representable coordinate 9.25.  Independent because it
    // is the defining property of linear interpolation, not a replay of
    // the mapping's own arithmetic.
    auto const q = RreaAxialPlaneInterpolationFor(9.25, 7, 22);
    double const a = 3.0;
    double const b = 0.5;
    double const interpolated =
        (1.0 - q.upper_weight) * (a + b * q.lower)
        + q.upper_weight * (a + b * q.upper);
    require(interpolated == a + b * 9.25,
        "(1-w, w) split does not reproduce a linear field at the plane");
}

// Core annulus overlap.  Independent reference: the areas telescope --
// summed over all covering cells they must equal core_r^2 - r_first^2
// EXACTLY when cell edges are exactly representable, because r_hi of cell
// i equals r_lo of cell i+1 by construction.
void exercise_core_annulus_area()
{
    using rrea::RreaCoreAnnulusArea;
    double const plo = 0.0;
    double const dx = 0.5;
    double const core_r = 3.25;
    double sum = 0.0;
    int covered = 0;
    for (int i = 0; i < 32; ++i) {
        double const area = RreaCoreAnnulusArea(i, plo, dx, core_r);
        require(area >= 0.0, "annulus area must be nonnegative");
        if (area > 0.0) {
            ++covered;
        }
        sum += area;
    }
    require(sum == core_r * core_r,
        "annulus areas must telescope exactly to core_radius^2");
    require(covered == 7,
        "cells [0,3.25) at dx=0.5 must contribute exactly 7 annuli");
    // Fully-outside cell and negative-r clamp.
    require(RreaCoreAnnulusArea(20, plo, dx, core_r) == 0.0,
        "cell beyond the core radius must contribute zero");
    require(RreaCoreAnnulusArea(0, -1.0, 0.5, 3.0) == 0.0,
        "cell entirely below r=0 must clamp to zero area");
}

// Stencil build + bilinear corners.  Independent references: hand-stated
// index outcomes at edges (periodic wrap vs one-sided clamp), the exact
// constant-in-z collapse, and the linear-reproduction identity
// f = a + b*i + c*j which bilinear interpolation must reproduce exactly at
// representable fractions -- a defining property, not a replay of the
// formula.
void exercise_gather_stencil_build()
{
    using rrea::RreaBilinearFromCorners;
    using rrea::RreaFieldGatherStencilFor;
    // Interior point: plain floor/fraction, anchor == (i0, j0).
    auto const in = RreaFieldGatherStencilFor(2.25, 9.75, 0, 7, 7, 22, false);
    require(in.stencil.i0 == 2 && in.stencil.i1 == 3
                && in.stencil.j0 == 9 && in.stencil.j1 == 10
                && in.tx == 0.25 && in.tz == 0.75
                && in.stencil.anchor_i == 2 && in.stencil.anchor_j == 9,
        "interior stencil build changed");
    // Nonperiodic axial edge: one-sided collapse j0 == j1 below the low
    // edge, and the anchor stays clamped in-domain.
    auto const low = RreaFieldGatherStencilFor(0.5, 6.25, 0, 7, 7, 22, false);
    require(low.stencil.j0 == 7 && low.stencil.j1 == 7
                && low.stencil.anchor_j == 7,
        "nonperiodic low edge must collapse one-sided");
    // With j0 == j1 the bilinear must become exactly constant in z:
    // (1-tz) + tz == 1 regardless of the out-of-domain fraction.
    require(
        RreaBilinearFromCorners(5.0, 5.0, 9.0, 9.0, 0.5, low.tz) == 7.0,
        "collapsed axial stencil is not constant in z");
    // Periodic axial edge: indices wrap into the one-cell ghost, both
    // below (j0 = axial_lo - 1) and above (j1 = axial_hi + 1); the anchor
    // is still clamped in-domain.
    auto const pl = RreaFieldGatherStencilFor(0.5, 6.75, 0, 7, 7, 22, true);
    require(pl.stencil.j0 == 6 && pl.stencil.j1 == 7
                && pl.stencil.anchor_j == 7,
        "periodic low edge must reach the ghost row, anchor clamped");
    auto const ph = RreaFieldGatherStencilFor(0.5, 22.5, 0, 7, 7, 22, true);
    require(ph.stencil.j0 == 22 && ph.stencil.j1 == 23
                && ph.stencil.anchor_j == 22,
        "periodic high edge must reach the ghost row, anchor clamped");
    // Radial direction is ALWAYS clamped (RZ axis / outer edge), even
    // when z is periodic.
    auto const ax = RreaFieldGatherStencilFor(-0.25, 9.5, 0, 7, 7, 22, true);
    require(ax.stencil.i0 == 0 && ax.stencil.i1 == 0
                && ax.stencil.anchor_i == 0,
        "radial axis must clamp, never wrap");
    // Linear reproduction at exactly-representable fractions: for
    // f(i, j) = 3 - 2*i + 0.5*j the interpolant at (2.25, 9.75) must be
    // 3 - 2*2.25 + 0.5*9.75 exactly.
    auto const f = [](int i, int j) {
        return 3.0 - 2.0 * static_cast<double>(i)
            + 0.5 * static_cast<double>(j);
    };
    require(
        RreaBilinearFromCorners(
            f(2, 9), f(2, 10), f(3, 9), f(3, 10), in.tx, in.tz)
            == 3.0 - 2.0 * 2.25 + 0.5 * 9.75,
        "bilinear does not reproduce a linear field exactly");
}

void exercise_radial_vector_parity()
{
    require(rrea::RreaOddRadialAxisScale(0.0, 0.0, 5.0) == 0.0,
        "odd Er must vanish on axis");
    require(rrea::RreaOddRadialAxisScale(1.25, 0.0, 5.0) == 0.5,
        "odd Er must continue linearly inside the first half-cell");
    require(rrea::RreaOddRadialAxisScale(2.5, 0.0, 5.0) == 1.0,
        "odd Er must reproduce the first cell-centre value");
}

void run_smoke()
{
#if (AMREX_SPACEDIM < 2)
    throw std::runtime_error("RREA RZ field-gather smoke requires AMREX_SPACEDIM >= 2");
#else
    require_close(
        rrea::RreaCellCenteredGatherIndexCoordinate(0.5, 0.0, 1.0, 7),
        7.0,
        "shifted-domain first cell-center coordinate");
    require_close(
        rrea::RreaCellCenteredGatherIndexCoordinate(3.0, 0.0, 1.0, 7),
        9.5,
        "shifted-domain fractional gather coordinate");
    exercise_axial_plane_interpolation();
    exercise_core_annulus_area();
    exercise_gather_stencil_build();
    exercise_radial_vector_parity();
    amrex::IntVect lo(0);
    amrex::IntVect hi(0);
    hi[0] = 0;
    hi[1] = 15;
    amrex::Box const domain(lo, hi);
    amrex::BoxArray boxes(domain);
    boxes.maxSize(8);
    require(boxes.size() == 2, "field-gather smoke did not create two axial boxes");

    int const nprocs = amrex::ParallelDescriptor::NProcs();
    amrex::Vector<int> process_map(boxes.size(), 0);
    for (int index = 0; index < boxes.size(); ++index) {
        process_map[index] = boxes[index].smallEnd(1) == 0
            ? 0
            : std::min(1, nprocs - 1);
    }
    amrex::DistributionMapping const distribution(process_map);

    amrex::RealBox real_box;
    real_box.setLo(0, 0.0);
    real_box.setHi(0, 1.0);
    real_box.setLo(1, 0.0);
    real_box.setHi(1, 16.0);
#if (AMREX_SPACEDIM >= 3)
    real_box.setLo(2, 0.0);
    real_box.setHi(2, 1.0);
#endif
    amrex::Vector<int> periodic(AMREX_SPACEDIM, 0);
    amrex::Geometry const geom(domain, &real_box, 0, periodic.data());

    amrex::MultiFab field(boxes, distribution, 1, 1);
    field.setVal(amrex::Real(-999.0));
    for (amrex::MFIter mfi(field, false); mfi.isValid(); ++mfi) {
        auto const valid = mfi.validbox();
        auto const values = field.array(mfi);
        rrea::GpuSynchronize();
        amrex::LoopOnCpu(
            valid,
            [=](int i, int j, int k) noexcept {
                values(i, j, k) = static_cast<amrex::Real>(j);
            });
    }
    field.FillBoundary(geom.periodicity());

    rrea::RreaFieldGatherStencil2D const seam_stencil{
        0, 0, 7, 8, 0, 7};
    rrea::RreaFieldGatherStencil2D const crossed_anchor_stencil{
        0, 0, 8, 9, 0, 8};
    amrex::Long local_left_checks = 0;
    amrex::Long local_right_checks = 0;
    for (amrex::MFIter mfi(field, false); mfi.isValid(); ++mfi) {
        auto const valid = mfi.validbox();
        auto const fab = mfi.fabbox();
        auto const values = field.const_array(mfi);
        if (valid.smallEnd(1) == 0) {
            ++local_left_checks;
            require(
                rrea::RreaFieldGatherEntryCanServe(
                    valid, fab, seam_stencil, /*allow_ghost_anchor=*/false),
                "left valid box cannot serve its cross-rank one-ghost stencil");
            require(
                !rrea::RreaFieldGatherEntryCanServe(
                    valid, fab, crossed_anchor_stencil,
                    /*allow_ghost_anchor=*/true),
                "one-ghost left FAB incorrectly accepted a two-cell-deep stencil");
            // The ghost-carried corner must feed the interpolant: with
            // f = j the seam value at tz = 0.25 between rows 7 (valid)
            // and 8 (cross-rank ghost) is exactly 7.25.
            require(
                rrea::RreaBilinearFromCorners(
                    values(seam_stencil.i0, seam_stencil.j0, 0),
                    values(seam_stencil.i0, seam_stencil.j1, 0),
                    values(seam_stencil.i1, seam_stencil.j0, 0),
                    values(seam_stencil.i1, seam_stencil.j1, 0),
                    amrex::Real(0.5), amrex::Real(0.25))
                    == amrex::Real(7.25),
                "cross-rank ghost corner does not feed the bilinear seam value");
        } else {
            ++local_right_checks;
            require(
                rrea::RreaFieldGatherEntryCanServe(
                    valid, fab, crossed_anchor_stencil,
                    /*allow_ghost_anchor=*/false),
                "right valid owner cannot serve the crossed-anchor stencil");
        }
    }
    amrex::ParallelDescriptor::ReduceLongSum(local_left_checks);
    amrex::ParallelDescriptor::ReduceLongSum(local_right_checks);
    require(local_left_checks == 1, "left seam FAB was not checked exactly once globally");
    require(local_right_checks == 1, "right seam FAB was not checked exactly once globally");
#endif
}

}  // namespace

int main(int argc, char** argv)
{
    rrea::smoke::InitializeAmrexSmoke(argc, argv);
    int result = 0;
    try {
        run_smoke();
        amrex::Print() << "rrea_field_gather_stencil_smoke passed\n";
    } catch (std::exception const& error) {
        result = 1;
        amrex::AllPrint() << "rrea_field_gather_stencil_smoke failed: "
                          << error.what() << "\n";
    }
    amrex::Finalize();
    return result;
}
