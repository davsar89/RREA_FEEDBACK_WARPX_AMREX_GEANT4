#ifndef RREA_SMOKE_DOMAIN_H
#define RREA_SMOKE_DOMAIN_H

// The AMReX smokes' RZ test domain: cells [offset, offset + n) in r and z
// spanning the physical box [0, r_max] x [0, z_max].

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>

namespace rrea::smoke {

struct TestDomain {
    amrex::Geometry geom;
    amrex::BoxArray ba;
    amrex::DistributionMapping dm;
};

inline amrex::Geometry MakeRzGeometry(
    int nr, int nz, double r_max_m, double z_max_m,
    bool periodic_z = false, int index_offset = 0)
{
    amrex::IntVect const lo(index_offset);
    amrex::IntVect hi(index_offset);
    hi[0] = index_offset + nr - 1;
    hi[1] = index_offset + nz - 1;
    amrex::RealBox real_box;
    real_box.setLo(0, 0.0);
    real_box.setHi(0, r_max_m);
    real_box.setLo(1, 0.0);
    real_box.setHi(1, z_max_m);
    amrex::Vector<int> periodic(AMREX_SPACEDIM, 0);
    periodic[1] = periodic_z ? 1 : 0;
    return amrex::Geometry(amrex::Box(lo, hi), &real_box, 0, periodic.data());
}

// max_box <= 0 keeps the domain as one box.
inline TestDomain MakeRzDomain(
    int nr, int nz, double r_max_m, double z_max_m, int max_box = 8,
    bool periodic_z = false, int index_offset = 0)
{
    auto const geom =
        MakeRzGeometry(nr, nz, r_max_m, z_max_m, periodic_z, index_offset);
    amrex::BoxArray ba(geom.Domain());
    if (max_box > 0) {
        ba.maxSize(max_box);
    }
    return {geom, ba, amrex::DistributionMapping(ba)};
}

} // namespace rrea::smoke

#endif // RREA_SMOKE_DOMAIN_H
