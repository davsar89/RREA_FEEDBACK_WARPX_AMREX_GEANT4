#include "rrea/RreaGpuSmoke.H"
#include "rrea/RreaFieldGatherScalar.H"
#include "rrea/RreaSmokeRequire.H"

#include <AMReX.H>
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
// index outcomes at edges (one-sided clamp), the exact
// constant-in-z collapse, and the linear-reproduction identity
// f = a + b*i + c*j which bilinear interpolation must reproduce exactly at
// representable fractions -- a defining property, not a replay of the
// formula.
void exercise_gather_stencil_build()
{
    using rrea::RreaBilinearFromCorners;
    using rrea::RreaFieldGatherStencilFor;
    // Interior point: plain floor/fraction.
    auto const in = RreaFieldGatherStencilFor(2.25, 9.75, 0, 7, 7, 22);
    require(in.stencil.i0 == 2 && in.stencil.i1 == 3
                && in.stencil.j0 == 9 && in.stencil.j1 == 10
                && in.tx == 0.25 && in.tz == 0.75,
        "interior stencil build changed");
    // Nonperiodic axial edge: one-sided collapse j0 == j1 below the low edge.
    auto const low = RreaFieldGatherStencilFor(0.5, 6.25, 0, 7, 7, 22);
    require(low.stencil.j0 == 7 && low.stencil.j1 == 7,
        "nonperiodic low edge must collapse one-sided");
    // With j0 == j1 the bilinear must become exactly constant in z:
    // (1-tz) + tz == 1 regardless of the out-of-domain fraction.
    require(
        RreaBilinearFromCorners(5.0, 5.0, 9.0, 9.0, 0.5, low.tz) == 7.0,
        "collapsed axial stencil is not constant in z");
    // The radial direction clamps at the RZ axis.
    auto const ax = RreaFieldGatherStencilFor(-0.25, 9.5, 0, 7, 7, 22);
    require(ax.stencil.i0 == 0 && ax.stencil.i1 == 0,
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
