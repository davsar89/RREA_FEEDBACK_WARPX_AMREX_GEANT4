#ifndef RREA_FIELD_GATHER_STENCIL_H
#define RREA_FIELD_GATHER_STENCIL_H

#include <AMReX_Box.H>
#include <AMReX_IntVect.H>
#include <AMReX_REAL.H>

#include <algorithm>
#include <cmath>

namespace rrea {

// Host-side description of the fixed cell-centered bilinear stencil used by
// the RZ material-transport field gather.  Keeping the coverage predicate in
// this small WarpX-independent header lets the native MPI smoke exercise the
// exact valid-box versus ghost-box ownership rule.
struct RreaFieldGatherStencil2D {
    int i0 = 0;
    int i1 = 0;
    int j0 = 0;
    int j1 = 0;
    int anchor_i = 0;
    int anchor_j = 0;
};

inline amrex::Real RreaCellCenteredGatherIndexCoordinate(
    amrex::Real position_m,
    amrex::Real prob_lo_m,
    amrex::Real cell_size_m,
    int domain_index_lo) noexcept
{
    return (position_m - prob_lo_m) / cell_size_m
        + static_cast<amrex::Real>(domain_index_lo) - amrex::Real(0.5);
}

inline amrex::IntVect RreaFieldGatherIndex2D(int i, int j) noexcept
{
    amrex::IntVect index(0);
    index[0] = i;
#if (AMREX_SPACEDIM >= 2)
    index[1] = j;
#else
    (void)j;
#endif
    return index;
}

// Odd-vector continuation from the first cell centre to the RZ axis.
[[nodiscard]] inline amrex::Real RreaOddRadialAxisScale(
    amrex::Real r_m, amrex::Real prob_lo_r_m, amrex::Real dr_m) noexcept
{
    amrex::Real const first_center = prob_lo_r_m + amrex::Real(0.5) * dr_m;
    if (r_m >= first_center) {
        return amrex::Real(1.0);
    }
    return std::max(
        amrex::Real(0.0),
        (r_m - prob_lo_r_m) / (amrex::Real(0.5) * dr_m));
}

inline bool RreaFieldGatherStencilFitsFab(
    amrex::Box const& fab_box,
    RreaFieldGatherStencil2D const& stencil) noexcept
{
    return fab_box.contains(RreaFieldGatherIndex2D(stencil.i0, stencil.j0))
        && fab_box.contains(RreaFieldGatherIndex2D(stencil.i0, stencil.j1))
        && fab_box.contains(RreaFieldGatherIndex2D(stencil.i1, stencil.j0))
        && fab_box.contains(RreaFieldGatherIndex2D(stencil.i1, stencil.j1));
}

inline bool RreaFieldGatherEntryCanServe(
    amrex::Box const& valid_box,
    amrex::Box const& fab_box,
    RreaFieldGatherStencil2D const& stencil,
    bool allow_ghost_anchor) noexcept
{
    if (!RreaFieldGatherStencilFitsFab(fab_box, stencil)) {
        return false;
    }
    return allow_ghost_anchor
        || valid_box.contains(
            RreaFieldGatherIndex2D(stencil.anchor_i, stencil.anchor_j));
}

// Complete stencil build for one cell-centered bilinear gather: floor the
// fractional index coordinates, clamp radially (the RZ axis and outer
// edge are never periodic), and axially either wrap into the one-cell
// FillBoundary ghost (periodic -- the ghost carries the opposite-side
// value) or clamp one-sided (nonperiodic physical edge).  The anchor is
// the nearest in-domain cell in both directions; it also gates the Er
// read upstream, so it lives here to keep the two from diverging.
struct RreaFieldGatherStencilBuild {
    RreaFieldGatherStencil2D stencil;
    amrex::Real tx = amrex::Real(0.0);
    amrex::Real tz = amrex::Real(0.0);
};

[[nodiscard]] inline RreaFieldGatherStencilBuild RreaFieldGatherStencilFor(
    amrex::Real fx, amrex::Real fz,
    int radial_lo, int radial_hi,
    int axial_lo, int axial_hi,
    bool periodic_z) noexcept
{
    int const ir0 = static_cast<int>(std::floor(static_cast<double>(fx)));
    int const iz0 = static_cast<int>(std::floor(static_cast<double>(fz)));
    RreaFieldGatherStencilBuild build;
    build.tx = fx - static_cast<amrex::Real>(ir0);
    build.tz = fz - static_cast<amrex::Real>(iz0);
    build.stencil.i0 = std::max(radial_lo, std::min(radial_hi, ir0));
    build.stencil.i1 = std::max(radial_lo, std::min(radial_hi, ir0 + 1));
    build.stencil.j0 = periodic_z
        ? iz0
        : std::max(axial_lo, std::min(axial_hi, iz0));
    build.stencil.j1 = periodic_z
        ? iz0 + 1
        : std::max(axial_lo, std::min(axial_hi, iz0 + 1));
    build.stencil.anchor_i = std::max(radial_lo, std::min(radial_hi, ir0));
    build.stencil.anchor_j = std::max(axial_lo, std::min(axial_hi, iz0));
    return build;
}

[[nodiscard]] inline amrex::Real RreaBilinearFromCorners(
    amrex::Real f00, amrex::Real f01,
    amrex::Real f10, amrex::Real f11,
    amrex::Real tx, amrex::Real tz) noexcept
{
    return (amrex::Real(1.0) - tx)
               * ((amrex::Real(1.0) - tz) * f00
                  + tz * f01)
           + tx * ((amrex::Real(1.0) - tz) * f10
                   + tz * f11);
}

// Axial plane -> (lower row, upper row, upper weight) for the C&D plane
// diagnostics.  coordinate is the RreaCellCenteredGatherIndexCoordinate
// of the plane; at or beyond a physical edge the plane collapses to that
// edge row with weight 0 (one-sided), and an interior plane splits
// (1-w, w) between the bracketing cell-center rows.
struct RreaAxialPlaneInterpolation {
    int lower = 0;
    int upper = 0;
    amrex::Real upper_weight = amrex::Real(0.0);
};

[[nodiscard]] inline RreaAxialPlaneInterpolation RreaAxialPlaneInterpolationFor(
    amrex::Real coordinate, int axial_lo, int axial_hi) noexcept
{
    if (coordinate <= static_cast<amrex::Real>(axial_lo)) {
        return {axial_lo, axial_lo, amrex::Real(0.0)};
    }
    if (coordinate >= static_cast<amrex::Real>(axial_hi)) {
        return {axial_hi, axial_hi, amrex::Real(0.0)};
    }
    int const lower = static_cast<int>(std::floor(coordinate));
    return {lower, lower + 1, coordinate - static_cast<amrex::Real>(lower)};
}

// Cell-i overlap annulus with the core disc [0, core_radius], pi dropped
// (it cancels from every area-weighted mean).  i_rel counts cells from
// the radial domain edge.  Cell edges shared between neighbours make the
// areas telescope exactly.
[[nodiscard]] inline amrex::Real RreaCoreAnnulusArea(
    int i_rel,
    amrex::Real prob_lo_r_m,
    amrex::Real dx_r_m,
    amrex::Real core_radius_m) noexcept
{
    amrex::Real const r_lo = std::max(
        amrex::Real(0.0),
        prob_lo_r_m + static_cast<amrex::Real>(i_rel) * dx_r_m);
    amrex::Real const r_hi = std::min(
        core_radius_m,
        prob_lo_r_m + static_cast<amrex::Real>(i_rel + 1) * dx_r_m);
    return r_hi > r_lo ? r_hi * r_hi - r_lo * r_lo : amrex::Real(0.0);
}

}  // namespace rrea

#endif
