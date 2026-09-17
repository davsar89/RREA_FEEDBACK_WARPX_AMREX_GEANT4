#ifndef RREA_FIELD_GATHER_STENCIL_H
#define RREA_FIELD_GATHER_STENCIL_H
#include "rrea/RreaFieldGatherScalar.H"
#include <AMReX_Box.H>
#include <AMReX_IntVect.H>
namespace rrea {
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
}
#endif
