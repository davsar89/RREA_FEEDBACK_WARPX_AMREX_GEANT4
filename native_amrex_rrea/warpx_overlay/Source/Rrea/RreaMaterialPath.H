#ifndef RREA_MATERIAL_PATH_H
#define RREA_MATERIAL_PATH_H

// Material path along one transport chord -- the single owner of the
// "how much air did this particle fly through, and where" question for every
// species loop in RreaParticleInteraction.cpp.
//
// Governing form.  Every tabulated process rate is a cross section per unit
// STP length times the local density ratio chi(l) = rho(z(l)) / rho_ref, so a
// channel's optical depth along a chord is
//
//     tau = Int_a^b sigma_stp(E) chi(l) dl = sigma_stp(E) * L_chi(a,b),
//     L_chi(a,b) = Int_a^b chi(l) dl                                [metres]
//
// with sigma_stp frozen at the segment's energy.  The energy dependence
// therefore factors out completely: ONE table lookup per refreeze multiplies a
// purely geometric density path.  L_chi is built once per chord as a
// piecewise-constant profile -- one interval per in-domain (clipped/periodic)
// piece, carrying that piece's path-mean chi -- so the depth-to-position
// inversion is exact inside an interval and additive across the periodic seam.
//
// Pinned by rrea_transport_path_smoke (chord clipping and pieces) and
// rrea_charged_interleave_smoke (the drag/ion-pair consumers).

#include "RreaParticleInteraction.H"
#include "RreaKineticEnvironment.H"

#include "rrea/RreaTransportPath.H"



#include <AMReX_REAL.H>

#include <cmath>
#include <cstdint>
#include <vector>

namespace rrea::warpx {

RREA_HOST_DEVICE inline amrex::Real positive_or(amrex::Real value, amrex::Real fallback)
{
    return value > amrex::Real(0.0) ? value : fallback;
}

// One in-domain chord piece with the mean density ratio the particle flew
// through on it.
struct MaterialPathInterval {
    amrex::Real begin_fraction = 0.0;
    amrex::Real end_fraction = 0.0;
    amrex::Real length_m = 0.0;
    amrex::Real density_ratio = 1.0;
};

// Int chi dl over one piece, by the MIDPOINT rule.  For an exponential
// atmosphere chi ~ exp(-z/H) the midpoint relative error over a piece of
// chord length h is (1/24)(h/H)^2 = 3.7e-10 at h = 0.75 m, H ~ 8 km -- eleven
// orders below the cross-section accuracy the integral is multiplied by.
// Evaluating each piece separately (rather than the whole chord) is what keeps
// a profile discontinuity from smearing across the axial periodic seam.
RREA_HOST_DEVICE inline amrex::Real integrate_density_piece_m(
    KineticEnvironment const& coupling,
    RreaTransportPathSegment const& path,
    RreaTransportPathPiece const& piece,
    amrex::Real begin_full_path_fraction,
    amrex::Real end_full_path_fraction,
    amrex::Real fallback_density_ratio)
{
    amrex::Real const begin = rrea::kinetic::max(
        piece.begin_fraction, begin_full_path_fraction);
    amrex::Real const end = rrea::kinetic::min(
        piece.end_fraction, end_full_path_fraction);
    if (!(end > begin) || !(path.full_length_m > amrex::Real(0.0))) {
        return amrex::Real(0.0);
    }
    amrex::Real x = amrex::Real(0.0);
    amrex::Real y = amrex::Real(0.0);
    amrex::Real z = amrex::Real(0.0);
    path.PointAtPieceFraction(
        piece, amrex::Real(0.5) * (begin + end), x, y, z);
    rrea::kinetic::ignore_unused(x, y);
    return path.full_length_m * (end - begin)
        * positive_or(
            coupling.TransportDensityRatioAtZ(z), fallback_density_ratio);
}

RREA_HOST_DEVICE inline rrea::KineticVector<MaterialPathInterval> build_material_path_intervals(
    KineticEnvironment const& coupling,
    RreaTransportPathSegment const& path,
    amrex::Real fallback_density_ratio)
{
    rrea::KineticVector<MaterialPathInterval> intervals;
    intervals.reserve(path.in_domain_pieces.size());
    for (auto const& piece : path.in_domain_pieces) {
        if (!(piece.length_m > amrex::Real(0.0))) {
            continue;
        }
        amrex::Real const density_length_m = integrate_density_piece_m(
            coupling,
            path,
            piece,
            piece.begin_fraction,
            piece.end_fraction,
            fallback_density_ratio);
        MaterialPathInterval interval;
        interval.begin_fraction = piece.begin_fraction;
        interval.end_fraction = piece.end_fraction;
        interval.length_m = piece.length_m;
        interval.density_ratio = positive_or(
            density_length_m / piece.length_m,
            fallback_density_ratio);
        intervals.push_back(interval);
    }
    return intervals;
}

// L_chi over [begin, end], clamped to the chord's in-domain span.
RREA_HOST_DEVICE inline amrex::Real density_path_m(
    rrea::KineticVector<MaterialPathInterval> const& intervals,
    RreaTransportPathSegment const& path,
    amrex::Real begin_fraction,
    amrex::Real end_fraction)
{
    begin_fraction = rrea::kinetic::max(path.in_domain_begin_fraction, begin_fraction);
    end_fraction = rrea::kinetic::min(path.in_domain_end_fraction, end_fraction);
    if (!(end_fraction > begin_fraction)) {
        return amrex::Real(0.0);
    }
    amrex::Real integral_m = amrex::Real(0.0);
    for (auto const& interval : intervals) {
        amrex::Real const begin = rrea::kinetic::max(begin_fraction, interval.begin_fraction);
        amrex::Real const end = rrea::kinetic::min(end_fraction, interval.end_fraction);
        if (end > begin) {
            integral_m += path.full_length_m * (end - begin) * interval.density_ratio;
        }
    }
    return integral_m;
}

// Inverse of density_path_m: the chord fraction at which L_chi from
// begin_fraction first reaches target_m.  Exact within an interval because chi
// is constant there; returns end_fraction when the whole span is shorter than
// the target.
RREA_HOST_DEVICE inline amrex::Real fraction_at_density_path_m(
    rrea::KineticVector<MaterialPathInterval> const& intervals,
    RreaTransportPathSegment const& path,
    amrex::Real begin_fraction,
    amrex::Real end_fraction,
    amrex::Real target_m)
{
    begin_fraction = rrea::kinetic::max(path.in_domain_begin_fraction, begin_fraction);
    end_fraction = rrea::kinetic::min(path.in_domain_end_fraction, end_fraction);
    if (!std::isfinite(static_cast<double>(target_m))
        || target_m < amrex::Real(0.0)) {
        rrea::KineticFail("RREA sampled material density path is invalid");
    }
    amrex::Real remaining_m = target_m;
    for (auto const& interval : intervals) {
        amrex::Real const begin = rrea::kinetic::max(begin_fraction, interval.begin_fraction);
        amrex::Real const end = rrea::kinetic::min(end_fraction, interval.end_fraction);
        if (!(end > begin) || !(interval.density_ratio > amrex::Real(0.0))) {
            continue;
        }
        amrex::Real const contribution_m =
            path.full_length_m * (end - begin) * interval.density_ratio;
        if (remaining_m <= contribution_m) {
            return begin
                + remaining_m / (path.full_length_m * interval.density_ratio);
        }
        remaining_m -= contribution_m;
    }
    return end_fraction;
}

// Largest L_chi of any one of `subcycles` uniform slices of the in-domain
// span.  A channel's worst per-substep optical depth is this times its frozen
// sigma_stp, which is why the substep sizing needs no per-channel integral.
RREA_HOST_DEVICE inline amrex::Real max_uniform_subcycle_density_path_m(
    rrea::KineticVector<MaterialPathInterval> const& intervals,
    RreaTransportPathSegment const& path,
    std::uint64_t subcycles)
{
    if (subcycles == 0U) {
        rrea::KineticFail("RREA material subcycle integration received zero subcycles");
    }
    amrex::Real const span = path.in_domain_end_fraction
        - path.in_domain_begin_fraction;
    amrex::Real const denominator = static_cast<amrex::Real>(subcycles);
    amrex::Real maximum_m = amrex::Real(0.0);
    for (std::uint64_t index = 0; index < subcycles; ++index) {
        maximum_m = rrea::kinetic::max(
            maximum_m,
            density_path_m(
                intervals,
                path,
                path.in_domain_begin_fraction
                    + static_cast<amrex::Real>(index) / denominator * span,
                path.in_domain_begin_fraction
                    + static_cast<amrex::Real>(index + 1U) / denominator * span));
    }
    return maximum_m;
}

// Interaction probability of a competing-risk optical depth tau.
RREA_HOST_DEVICE inline amrex::Real probability_from_optical_depth(amrex::Real tau)
{
    if (!(tau > amrex::Real(0.0))) {
        return amrex::Real(0.0);
    }
    return rrea::kinetic::min(amrex::Real(1.0), -std::expm1(-tau));
}

// A process rate frozen at one energy, in inverse metres (or eV/m for a
// stopping power) per unit density ratio.  Constructing one is where a table
// is read; multiplying it by a density path is all the transport loops do
// afterwards, so the finiteness/sign check belongs here and nowhere else.
[[nodiscard]] RREA_HOST_DEVICE inline amrex::Real RreaFrozenStpRate(amrex::Real stp_value)
{
    if (!std::isfinite(static_cast<double>(stp_value))
        || stp_value < amrex::Real(0.0)) {
        rrea::KineticFail("RREA frozen material-path rate is negative or non-finite");
    }
    return stp_value;
}

}  // namespace rrea::warpx

#endif
