#ifndef RREA_KINETIC_FIELD_VIEW_H
#define RREA_KINETIC_FIELD_VIEW_H
#include "rrea/RreaFieldGatherScalar.H"
#include "rrea/RreaMaxwellTMRZ.H"
namespace rrea {
// Views refer to the SAME authoritative Maxwell arrays as the fluid update.
struct RreaKineticFieldView {
    MaxwellTMRZConstView field;
    MaxwellTMRZConstView ambient;
    double prob_lo_r=0.0, prob_lo_z=0.0;
    int lo[2]{0,0}, hi[2]{0,0};
    bool periodic_z=false;
    RREA_HOST_DEVICE double MaxwellBthetaAt(double r_m, double z_m) const noexcept {

    if (r_m <= 0.0) {
        return 0.0;
    }
    double const fr = rrea::kinetic::clamp(r_m / field.dr, 0.0,
        static_cast<double>(field.nr));
    double const fz = rrea::kinetic::clamp((z_m - field.z_lo) / field.dz, 0.0,
        static_cast<double>(field.nz));
    int const i0 = rrea::kinetic::min(static_cast<int>(std::floor(fr)), field.nr - 1);
    int const j0 = rrea::kinetic::min(static_cast<int>(std::floor(fz)), field.nz - 1);
    int const i1 = i0 + 1;
    int const j1 = j0 + 1;
    double const tr = fr - i0;
    double const tz = fz - j0;
    return (1.0 - tr) * ((1.0 - tz) * field.Bt(i0, j0)
                         + tz * field.Bt(i0, j1))
        + tr * ((1.0 - tz) * field.Bt(i1, j0)
                + tz * field.Bt(i1, j1));
    }
    RREA_HOST_DEVICE void Gather(double r_m, double z_m, double& er, double& ez, double& btheta) const noexcept {

    auto const build = RreaFieldGatherStencilFor(
        RreaCellCenteredGatherIndexCoordinate(
            r_m, prob_lo_r, field.dr, lo[0]),
        RreaCellCenteredGatherIndexCoordinate(
            z_m, prob_lo_z, field.dz, lo[1]),
        lo[0], hi[0],
        lo[1], hi[1],
        periodic_z);
    // The periodic stencil addresses one cell into the FillBoundary ghost the
    // cell fields carry; on the replicated vectors that is the wrap itself.
    auto const wrap_j = [&](int j) {
        return periodic_z ? (j + field.nz) % field.nz : j;
    };
    int const i0 = build.stencil.i0;
    int const i1 = build.stencil.i1;
    int const j0 = wrap_j(build.stencil.j0);
    int const j1 = wrap_j(build.stencil.j1);
    // The identical cell-centred face averages CommitMaxwellTotalField writes.
    auto const cell_er = [&](int i, int j) {
        double const lo = field.er[field.ErIndex(i, j)]
            + ambient.er[field.ErIndex(i, j)];
        double const hi = field.er[field.ErIndex(i + 1, j)]
            + ambient.er[field.ErIndex(i + 1, j)];
        return 0.5 * (lo + hi);
    };
    auto const cell_ez = [&](int i, int j) {
        double const lo = field.ez[field.EzIndex(i, j)]
            + ambient.ez[field.EzIndex(i, j)];
        double const hi = field.ez[field.EzIndex(i, j + 1)]
            + ambient.ez[field.EzIndex(i, j + 1)];
        return 0.5 * (lo + hi);
    };
    er = RreaBilinearFromCorners(
             cell_er(i0, j0), cell_er(i0, j1),
             cell_er(i1, j0), cell_er(i1, j1), build.tx, build.tz)
        * RreaOddRadialAxisScale(r_m, prob_lo_r, field.dr);
    ez = RreaBilinearFromCorners(
        cell_ez(i0, j0), cell_ez(i0, j1),
        cell_ez(i1, j0), cell_ez(i1, j1), build.tx, build.tz);
    btheta = MaxwellBthetaAt(r_m, z_m);
    }
};
}
#endif
