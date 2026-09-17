#ifndef RREA_PATH_FIELD_WORK_H
#define RREA_PATH_FIELD_WORK_H

// Field work and mean field along one transport chord -- the single owner of
// "what did the electromagnetic field do to this particle over this piece of
// its path" for every species loop in RreaParticleInteraction.cpp.
//
// Governing form.  Over a sub-range [a, b] of a chord of length L, with
// dhat the unit chord direction and q the charge sign in elementary charges,
//
//     W    = q Int_a^b E(r(l), z(l)) . dhat  L dl                      [eV]
//     <F>  = (1/L_ab) Int_a^b F(r(l), z(l)) L dl,   F in {E, B}
//
// integrated separately over each explicit in-domain piece so a domain clip or
// a periodic seam can never smear the gather across the boundary.  The scalar
// work is the SOLE energy authority of a substep field impulse; the vector
// averages only rotate the running proper velocity
// (RreaApplyChargedFieldHalfKick).  Both therefore have to come from the same
// nodes, which is why one pass returns all seven components and every node
// gathers the field exactly once.
//
// Quadrature.  Inside one cell the gather is bilinear in (r, z) -- the
// cell-corner reconstruction of RreaAmrexAdvance::GatherTotalFieldRZ, with the
// odd-radial axis scale keeping E_r . rhat regular on axis -- and (r(l), z(l))
// is affine along a meridional chord, so the integrand is quadratic in l and
// the TWO-node Gauss-Legendre rule is exact there.  Two residual errors
// remain, both far below the impulse driver's own granularity
// (transport_max_field_impulse_fraction):
//   * a microchord straddling a cell face samples a C0 reconstruction, giving
//     a relative work error <= h*Delta / (50 L_E^2), about 2e-5 for a field
//     resolved over L_E ~ 10 Delta, on the ~h/Delta fraction of microchords
//     that straddle a face;
//   * an azimuthal chord makes r(l) depart from affine by at most h^2/(8 r),
//     7e-4 m for h = 0.75 m at r = 100 m, a fraction 1e-4 of a 5 m cell.
//
// Pinned by rrea_gauss_legendre8_smoke (the two-node rule's cubic exactness)
// and rrea_charged_interleave_smoke (the half-kick consumer).

#include "RreaChargedInterleave.H"
#include "RreaParticleInteraction.H"
#include "RreaWarpXCoupling.H"

#include "rrea/RreaGaussLegendre8.H"
#include "rrea/RreaTransportPath.H"

#include <AMReX.H>
#include <AMReX_Algorithm.H>
#include <AMReX_REAL.H>

#include <array>
#include <cmath>

namespace rrea::warpx {

using PathFieldSubRange = RreaPathFieldSubRange;

inline PathFieldSubRange path_field_work_and_average(
    RreaWarpXCoupling const& coupling,
    RreaTransportPathSegment const& path,
    amrex::Real charge_sign,
    amrex::Real begin_fraction,
    amrex::Real end_fraction)
{
    PathFieldSubRange result;
    if (!(path.full_length_m > amrex::Real(0.0))) {
        return result;
    }
    begin_fraction = amrex::max(path.in_domain_begin_fraction, begin_fraction);
    end_fraction = amrex::min(path.in_domain_end_fraction, end_fraction);
    if (!(end_fraction > begin_fraction)) {
        return result;
    }
    amrex::Real const path_dir_x =
        (path.end_x_m - path.start_x_m) / path.full_length_m;
    amrex::Real const path_dir_y =
        (path.end_y_m - path.start_y_m) / path.full_length_m;
    amrex::Real const path_dir_z =
        (path.end_z_m - path.start_z_m) / path.full_length_m;
    for (auto const& piece : path.in_domain_pieces) {
        amrex::Real const piece_begin =
            amrex::max(begin_fraction, piece.begin_fraction);
        amrex::Real const piece_end =
            amrex::min(end_fraction, piece.end_fraction);
        if (!(piece_end > piece_begin)) {
            continue;
        }
        auto const piece_integral =
            rrea::RreaGaussLegendre2IntegrateArray<amrex::Real, 7>(
                piece_begin,
                piece_end,
                [&](amrex::Real sample_fraction) {
                    amrex::Real x = amrex::Real(0.0);
                    amrex::Real y = amrex::Real(0.0);
                    amrex::Real z = amrex::Real(0.0);
                    path.PointAtPieceFraction(piece, sample_fraction, x, y, z);
                    amrex::Real const r = std::hypot(x, y);
                    amrex::Real radial_path_dir = amrex::Real(0.0);
                    amrex::Real radial_cos = amrex::Real(1.0);
                    amrex::Real radial_sin = amrex::Real(0.0);
                    if (r > amrex::Real(0.0)) {
                        radial_path_dir = (path_dir_x * x + path_dir_y * y) / r;
                        radial_cos = x / r;
                        radial_sin = y / r;
                    }
                    RreaFieldSample const field = coupling.GatherFieldRZ(0, r, z);
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
                        amrex::Abort(
                            "RREA substep field-impulse gather is non-finite");
                    }
                    return std::array<amrex::Real, 7>{
                        work_per_m,
                        field.er_v_per_m * radial_cos,
                        field.er_v_per_m * radial_sin,
                        field.ez_v_per_m,
                        -field.btheta_t * radial_sin,
                        field.btheta_t * radial_cos,
                        amrex::Real(0.0)};
                });
        result.work_eV += path.full_length_m * piece_integral[0];
        result.avg_ex_v_per_m += path.full_length_m * piece_integral[1];
        result.avg_ey_v_per_m += path.full_length_m * piece_integral[2];
        result.avg_ez_v_per_m += path.full_length_m * piece_integral[3];
        result.avg_bx_t += path.full_length_m * piece_integral[4];
        result.avg_by_t += path.full_length_m * piece_integral[5];
        result.avg_bz_t += path.full_length_m * piece_integral[6];
        result.length_m += path.full_length_m * (piece_end - piece_begin);
    }
    if (result.length_m > amrex::Real(0.0)) {
        result.avg_ex_v_per_m /= result.length_m;
        result.avg_ey_v_per_m /= result.length_m;
        result.avg_ez_v_per_m /= result.length_m;
        result.avg_bx_t /= result.length_m;
        result.avg_by_t /= result.length_m;
        result.avg_bz_t /= result.length_m;
    }
    return result;
}

}  // namespace rrea::warpx

#endif
