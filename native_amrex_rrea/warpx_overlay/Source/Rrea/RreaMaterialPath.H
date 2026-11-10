#ifndef RREA_MATERIAL_PATH_H
#define RREA_MATERIAL_PATH_H

// Material path along one transport chord -- the single owner of the
// "how much air did this particle fly through, and where" question for the
// charged and photon histories (RreaKineticStep.H).
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
// purely geometric density path.  L_chi is built once per chord from the
// in-domain span's midpoint chi, so the depth-to-position inversion is exact.
//
// Pinned by rrea_transport_path_smoke (chord clipping) and
// rrea_charged_interleave_smoke (the drag/ion-pair consumers).

#include "RreaParticleInteraction.H"
#include "RreaKineticEnvironment.H"

#include "rrea/RreaTransportPath.H"



#include <AMReX_REAL.H>

#include <cmath>
#include <cstdint>
#include <vector>

namespace rrea::warpx {

// The chord's in-domain span with the mean density ratio the particle flew
// through on it (length 0 when the chord has no in-domain part).
struct MaterialPathInterval {
    amrex::Real begin_fraction = 0.0;
    amrex::Real end_fraction = 0.0;
    amrex::Real length_m = 0.0;
    amrex::Real density_ratio = 1.0;
};

// Int chi dl over the in-domain span by the MIDPOINT rule.  For an
// exponential atmosphere chi ~ exp(-z/H) the relative error over a chord of
// length h is (1/24)(h/H)^2 = 3.7e-10 at h = 0.75 m, H ~ 8 km -- more than seven
// orders below the percent-level cross-section accuracy.
RREA_HOST_DEVICE inline MaterialPathInterval build_material_path_interval(
    KineticEnvironment const& coupling,
    RreaTransportPathSegment const& path)
{
    MaterialPathInterval interval{path.in_domain_begin_fraction,
        path.in_domain_end_fraction, path.in_domain_length_m};
    if (interval.length_m > amrex::Real(0.0)) {
        amrex::Real x = amrex::Real(0.0);
        amrex::Real y = amrex::Real(0.0);
        amrex::Real z = amrex::Real(0.0);
        path.PointAtInDomainFraction(
            amrex::Real(0.5) * (interval.begin_fraction + interval.end_fraction), x, y, z);
        interval.density_ratio = coupling.TransportDensityRatioAtZ(z);
    }
    return interval;
}

// L_chi over [begin, end], clamped to the chord's in-domain span.
RREA_HOST_DEVICE inline amrex::Real density_path_m(
    MaterialPathInterval const& interval,
    RreaTransportPathSegment const& path,
    amrex::Real begin_fraction,
    amrex::Real end_fraction)
{
    amrex::Real const begin = rrea::kinetic::max(interval.begin_fraction, begin_fraction);
    amrex::Real const end = rrea::kinetic::min(interval.end_fraction, end_fraction);
    return end > begin
        ? path.full_length_m * (end - begin) * interval.density_ratio
        : amrex::Real(0.0);
}

// Inverse of density_path_m: the chord fraction at which L_chi from
// begin_fraction first reaches target_m.  Exact because chi is constant on
// the span; returns the (span-clamped) end when the span is shorter than the
// target.
RREA_HOST_DEVICE inline amrex::Real fraction_at_density_path_m(
    MaterialPathInterval const& interval,
    RreaTransportPathSegment const& path,
    amrex::Real begin_fraction,
    amrex::Real end_fraction,
    amrex::Real target_m)
{
    if (!std::isfinite(static_cast<double>(target_m))
        || target_m < amrex::Real(0.0)) {
        rrea::KineticFail("RREA sampled material density path is invalid");
    }
    amrex::Real const begin = rrea::kinetic::max(interval.begin_fraction, begin_fraction);
    amrex::Real const end = rrea::kinetic::min(interval.end_fraction, end_fraction);
    if (!(end > begin) || !(interval.density_ratio > amrex::Real(0.0))) {
        return end;
    }
    amrex::Real const fraction =
        begin + target_m / (path.full_length_m * interval.density_ratio);
    return fraction <= end ? fraction : end;
}

// L_chi of one of `subcycles` uniform slices of the in-domain span (chi is
// constant on it, so every slice has the same path).  A channel's worst
// per-substep optical depth is this times its frozen sigma_stp.
RREA_HOST_DEVICE inline amrex::Real max_uniform_subcycle_density_path_m(
    MaterialPathInterval const& interval,
    RreaTransportPathSegment const& path,
    std::uint64_t subcycles)
{
    if (subcycles == 0U) {
        rrea::KineticFail("RREA material subcycle integration received zero subcycles");
    }
    return density_path_m(interval, path, interval.begin_fraction, interval.end_fraction)
        / static_cast<amrex::Real>(subcycles);
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
