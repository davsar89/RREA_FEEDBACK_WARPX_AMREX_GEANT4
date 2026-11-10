#ifndef RREA_PATH_FIELD_WORK_H
#define RREA_PATH_FIELD_WORK_H

// Field work and mean field along one transport chord -- the single owner of
// "what did the electromagnetic field do to this particle over this part of
// its path" for AdvanceChargedHistory (RreaKineticStep.H).
//
// Governing form.  Over a sub-range [a, b] of a chord of length L, with
// dhat the unit chord direction and q the charge sign in elementary charges,
//
//     W    = q Int_a^b E(r(l), z(l)) . dhat  L dl                      [eV]
//     <F>  = (1/L_ab) Int_a^b F(r(l), z(l)) L dl,   F in {E, B}
//
// integrated over the clipped in-domain span only, so the gather never reaches
// past the boundary.  The scalar
// work is the SOLE energy authority of a substep field impulse; the vector
// averages only rotate the running proper velocity
// (RreaApplyChargedFieldHalfKick).  Both therefore have to come from the same
// nodes, which is why one pass integrates all six components and every node
// gathers the field exactly once.
//
// Quadrature. On a meridional chord within one bilinear gather patch,
// (r(l), z(l)) is affine and the integrand is quadratic, so the two-node
// Gauss-Legendre rule is exact. Stencil/axis transitions and azimuthal chords
// (r = hypot(x, y), with a rotating radial direction) generally break that
// polynomial condition. Their accuracy needs mesh/chord refinement; cubic
// exactness alone is not a bound on the full gathered-path integral.
//
// Pinned by rrea_gauss_legendre2_smoke (the two-node rule's cubic exactness)
// and rrea_charged_interleave_smoke (the half-kick consumer).

#include "RreaChargedInterleave.H"
#include "RreaParticleInteraction.H"
#include "RreaKineticEnvironment.H"

#include "rrea/RreaGaussLegendre2.H"
#include "rrea/RreaTransportPath.H"



#include <AMReX_REAL.H>

#include <array>
#include <cmath>

namespace rrea::warpx {

using PathFieldSubRange = RreaPathFieldSubRange;

RREA_HOST_DEVICE inline PathFieldSubRange path_field_work_and_average(
    KineticEnvironment const& coupling,
    RreaTransportPathSegment const& path,
    amrex::Real charge_sign,
    amrex::Real begin_fraction,
    amrex::Real end_fraction)
{
    PathFieldSubRange result;
    if (!(path.full_length_m > amrex::Real(0.0))) {
        return result;
    }
    begin_fraction = rrea::kinetic::max(path.in_domain_begin_fraction, begin_fraction);
    end_fraction = rrea::kinetic::min(path.in_domain_end_fraction, end_fraction);
    if (!(end_fraction > begin_fraction)) {
        return result;
    }
    amrex::Real const path_dir_x =
        (path.end_x_m - path.start_x_m) / path.full_length_m;
    amrex::Real const path_dir_y =
        (path.end_y_m - path.start_y_m) / path.full_length_m;
    amrex::Real const path_dir_z =
        (path.end_z_m - path.start_z_m) / path.full_length_m;
    auto const integral =
        rrea::RreaGaussLegendre2IntegrateArray<amrex::Real, 6>(
            begin_fraction,
            end_fraction,
            [&](amrex::Real sample_fraction) {
                amrex::Real x = amrex::Real(0.0);
                amrex::Real y = amrex::Real(0.0);
                amrex::Real z = amrex::Real(0.0);
                path.PointAtInDomainFraction(sample_fraction, x, y, z);
                amrex::Real const r = std::hypot(x, y);
                amrex::Real radial_path_dir = amrex::Real(0.0);
                amrex::Real radial_cos = amrex::Real(1.0);
                amrex::Real radial_sin = amrex::Real(0.0);
                if (r > amrex::Real(0.0)) {
                    radial_path_dir = (path_dir_x * x + path_dir_y * y) / r;
                    radial_cos = x / r;
                    radial_sin = y / r;
                }
                RreaFieldSample const field = coupling.GatherFieldRZ(r, z);
                amrex::Real const work_per_m = RreaFieldWorkIncrementEv(
                    charge_sign,
                    field.er_v_per_m,
                    field.ez_v_per_m,
                    radial_path_dir,
                    path_dir_z,
                    amrex::Real(1.0));
                if (!std::isfinite(static_cast<double>(work_per_m))
                    || !std::isfinite(
                        static_cast<double>(field.er_v_per_m))
                    || !std::isfinite(
                        static_cast<double>(field.ez_v_per_m))
                    || !std::isfinite(
                        static_cast<double>(field.btheta_t))) {
                    rrea::KineticFail(
                        "RREA substep field-impulse gather is non-finite");
                }
                return std::array<amrex::Real, 6>{
                    work_per_m,
                    field.er_v_per_m * radial_cos,
                    field.er_v_per_m * radial_sin,
                    field.ez_v_per_m,
                    -field.btheta_t * radial_sin,
                    field.btheta_t * radial_cos};
            });
    amrex::Real const span = end_fraction - begin_fraction;
    result.work_eV = path.full_length_m * integral[0];
    result.avg_ex_v_per_m = integral[1] / span;
    result.avg_ey_v_per_m = integral[2] / span;
    result.avg_ez_v_per_m = integral[3] / span;
    result.avg_bx_t = integral[4] / span;
    result.avg_by_t = integral[5] / span;
    result.length_m = path.full_length_m * span;
    return result;
}

}  // namespace rrea::warpx

#endif
