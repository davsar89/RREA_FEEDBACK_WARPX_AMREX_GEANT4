#ifndef RREA_FIELD_HANDOFF_H
#define RREA_FIELD_HANDOFF_H

#include <AMReX_Array.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>

namespace rrea {

// Interpolate an authoritative cell-centered scalar onto any WarpX native
// staggering.  Nodal directions average the two bracketing cells.  Periodic
// neighbors come from a filled one-cell source halo; only physical boundaries
// use one-sided clamping.  FillBoundaryAndSync then resolves overlapping valid
// nodes and fills the destination's interior/periodic guards in one operation.
inline void InterpolateCellCenteredToNative(
    amrex::MultiFab const& cell_src,
    amrex::MultiFab& native_dst,
    amrex::Geometry const& geom)
{
    amrex::BoxArray const expected_native =
        amrex::convert(cell_src.boxArray(), native_dst.ixType().toIntVect());
    if (cell_src.nComp() != 1 || native_dst.nComp() != 1
        || native_dst.boxArray() != expected_native
        || cell_src.DistributionMap() != native_dst.DistributionMap()) {
        amrex::Abort(
            "RREA adapter interpolation requires scalar fields with matching grid decomposition");
    }
    amrex::MultiFab src_g(cell_src.boxArray(), cell_src.DistributionMap(), 1, 1);
    src_g.setVal(0.0);
    amrex::MultiFab::Copy(src_g, cell_src, 0, 0, 1, 0);
    src_g.FillBoundary(geom.periodicity());

    auto const ix = native_dst.ixType();
    bool const node_r = ix.nodeCentered(0);
#if (AMREX_SPACEDIM >= 2)
    bool const node_z = ix.nodeCentered(1);
#else
    bool const node_z = false;
#endif
    auto const domain = geom.Domain();
    int const cell_lo_r = domain.smallEnd(0);
    int const cell_hi_r = domain.bigEnd(0);
    bool const periodic_r = geom.isPeriodic(0);
#if (AMREX_SPACEDIM >= 2)
    int const cell_lo_z = domain.smallEnd(1);
    int const cell_hi_z = domain.bigEnd(1);
    bool const periodic_z = geom.isPeriodic(1);
#else
    int const cell_lo_z = 0;
    int const cell_hi_z = 0;
    bool const periodic_z = false;
#endif
    for (amrex::MFIter mfi(native_dst, amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi) {
        auto const bx = mfi.tilebox();
        auto const src_arr = src_g.const_array(mfi);
        auto const dst_arr = native_dst.array(mfi);
        amrex::ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                amrex::Real sum = amrex::Real(0.0);
                int count = 0;
                for (int a = (node_r ? -1 : 0); a <= 0; ++a) {
                    int ii = i + a;
                    if (!periodic_r) {
                        ii = amrex::min(amrex::max(ii, cell_lo_r), cell_hi_r);
                    }
                    for (int b = (node_z ? -1 : 0); b <= 0; ++b) {
                        int jj = j + b;
                        if (!periodic_z) {
                            jj = amrex::min(amrex::max(jj, cell_lo_z), cell_hi_z);
                        }
                        sum += src_arr(ii, jj, k);
                        ++count;
                    }
                }
                dst_arr(i, j, k) = sum / static_cast<amrex::Real>(count);
            });
    }
    native_dst.FillBoundaryAndSync(geom.periodicity());
}

}  // namespace rrea

#endif
