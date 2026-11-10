#ifndef RREA_TRANSPORT_PATH_H
#define RREA_TRANSPORT_PATH_H

#include <AMReX_REAL.H>

#include "rrea/RreaKineticSupport.H"

namespace rrea {

struct RreaTransportPathSegment {
    amrex::Real start_x_m = 0.0;
    amrex::Real start_y_m = 0.0;
    amrex::Real start_z_m = 0.0;
    amrex::Real end_x_m = 0.0;
    amrex::Real end_y_m = 0.0;
    amrex::Real end_z_m = 0.0;
    amrex::Real full_length_m = 0.0;
    amrex::Real in_domain_begin_fraction = 0.0;
    amrex::Real in_domain_end_fraction = 0.0;
    amrex::Real in_domain_length_m = 0.0;
    bool exits_domain = false;
    amrex::Real domain_r_lo_m = 0.0;
    amrex::Real domain_r_hi_m = 0.0;
    amrex::Real domain_z_lo_m = 0.0;
    amrex::Real domain_z_hi_m = 0.0;

    RREA_HOST_DEVICE void PointAtFullPathFraction(
        amrex::Real fraction,
        amrex::Real& x_m,
        amrex::Real& y_m,
        amrex::Real& z_m) const noexcept;
    RREA_HOST_DEVICE void PointAtInDomainFraction(
        amrex::Real fraction,
        amrex::Real& x_m,
        amrex::Real& y_m,
        amrex::Real& z_m) const noexcept;
    RREA_HOST_DEVICE [[nodiscard]] bool OutsideCertifiedDomain(
        amrex::Real r_m, amrex::Real z_m) const noexcept;
};

// Clip a post-push RZ chord against the cylindrical physical domain.  The
// captured start must be in-domain; radial and axial exits are clipped (the
// coupling accepts only absorbing boundaries).
[[nodiscard]] RREA_HOST_DEVICE inline RreaTransportPathSegment ClipRzTransportPath(
    amrex::Real start_r_m,
    amrex::Real start_theta,
    amrex::Real start_z_m,
    amrex::Real end_r_m,
    amrex::Real end_theta,
    amrex::Real end_z_m,
    amrex::Real domain_r_lo_m,
    amrex::Real domain_r_hi_m,
    amrex::Real domain_z_lo_m,
    amrex::Real domain_z_hi_m);

RREA_HOST_DEVICE inline bool RreaTransportPathSegment::OutsideCertifiedDomain(
    amrex::Real r_m, amrex::Real z_m) const noexcept
{
    if (!std::isfinite(static_cast<double>(r_m))
        || !std::isfinite(static_cast<double>(z_m))) {
        return true;
    }
    if (r_m < domain_r_lo_m || r_m >= domain_r_hi_m) {
        return true;
    }
    return z_m < domain_z_lo_m || z_m >= domain_z_hi_m;
}

RREA_HOST_DEVICE inline void RreaTransportPathSegment::PointAtFullPathFraction(
    amrex::Real fraction,
    amrex::Real& x_m,
    amrex::Real& y_m,
    amrex::Real& z_m) const noexcept
{
    fraction = rrea::kinetic::max(amrex::Real(0.0), rrea::kinetic::min(amrex::Real(1.0), fraction));
    x_m = start_x_m + fraction * (end_x_m - start_x_m);
    y_m = start_y_m + fraction * (end_y_m - start_y_m);
    z_m = start_z_m + fraction * (end_z_m - start_z_m);
}

// A point on the in-domain span, for density and field lookups: the fraction
// is held inside the span and z inside the closed axial interval, so roundoff
// at a clipped exit never looks up a point a few ulps outside the mesh.
RREA_HOST_DEVICE inline void RreaTransportPathSegment::PointAtInDomainFraction(
    amrex::Real fraction,
    amrex::Real& x_m,
    amrex::Real& y_m,
    amrex::Real& z_m) const noexcept
{
    fraction = rrea::kinetic::max(
        in_domain_begin_fraction,
        rrea::kinetic::min(in_domain_end_fraction, fraction));
    x_m = start_x_m + fraction * (end_x_m - start_x_m);
    y_m = start_y_m + fraction * (end_y_m - start_y_m);
    z_m = start_z_m + fraction * (end_z_m - start_z_m);
    z_m = rrea::kinetic::max(domain_z_lo_m, rrea::kinetic::min(domain_z_hi_m, z_m));
}

RREA_HOST_DEVICE inline RreaTransportPathSegment ClipRzTransportPath(
    amrex::Real start_r_m,
    amrex::Real start_theta,
    amrex::Real start_z_m,
    amrex::Real end_r_m,
    amrex::Real end_theta,
    amrex::Real end_z_m,
    amrex::Real domain_r_lo_m,
    amrex::Real domain_r_hi_m,
    amrex::Real domain_z_lo_m,
    amrex::Real domain_z_hi_m)
{
    auto const finite = [](amrex::Real value) {
        return std::isfinite(static_cast<double>(value));
    };
    if (!finite(start_r_m) || !finite(start_theta) || !finite(start_z_m)
        || !finite(end_r_m) || !finite(end_theta) || !finite(end_z_m)
        || !finite(domain_r_lo_m) || !finite(domain_r_hi_m)
        || !finite(domain_z_lo_m) || !finite(domain_z_hi_m)
        || start_r_m < amrex::Real(0.0) || end_r_m < amrex::Real(0.0)
        || domain_r_lo_m < amrex::Real(0.0)
        || !(domain_r_lo_m < domain_r_hi_m)
        || !(domain_z_lo_m < domain_z_hi_m)) {
        rrea::KineticFail("ClipRzTransportPath received invalid geometry or particle coordinates");
    }

    RreaTransportPathSegment result;
    result.domain_r_lo_m = domain_r_lo_m;
    result.domain_r_hi_m = domain_r_hi_m;
    result.domain_z_lo_m = domain_z_lo_m;
    result.domain_z_hi_m = domain_z_hi_m;
    result.start_x_m = start_r_m * std::cos(start_theta);
    result.start_y_m = start_r_m * std::sin(start_theta);
    result.start_z_m = start_z_m;
    result.end_x_m = end_r_m * std::cos(end_theta);
    result.end_y_m = end_r_m * std::sin(end_theta);
    result.end_z_m = end_z_m;
    amrex::Real const dx = result.end_x_m - result.start_x_m;
    amrex::Real const dy = result.end_y_m - result.start_y_m;
    amrex::Real const dz = result.end_z_m - result.start_z_m;
    result.full_length_m = std::sqrt(dx * dx + dy * dy + dz * dz);
    result.in_domain_begin_fraction = amrex::Real(0.0);

    amrex::Real const scale = rrea::kinetic::max(
        amrex::Real(1.0),
        rrea::kinetic::max(domain_r_hi_m,
            rrea::kinetic::max(std::abs(domain_z_lo_m), std::abs(domain_z_hi_m))));
    amrex::Real const tol = amrex::Real(64.0)
        * std::numeric_limits<amrex::Real>::epsilon() * scale;
    if (start_r_m < domain_r_lo_m - tol || start_r_m > domain_r_hi_m + tol
        || start_z_m < domain_z_lo_m - tol || start_z_m > domain_z_hi_m + tol) {
        rrea::KineticFail("RREA previous particle position is outside the physical RZ domain");
    }

    amrex::Real exit_fraction = amrex::Real(1.0);
    auto retain_positive_exit = [&exit_fraction](amrex::Real candidate) {
        if (candidate >= amrex::Real(0.0) && candidate < exit_fraction) {
            exit_fraction = candidate;
        }
    };
    if (dz > amrex::Real(0.0)) {
        retain_positive_exit((domain_z_hi_m - start_z_m) / dz);
    } else if (dz < amrex::Real(0.0)) {
        retain_positive_exit((domain_z_lo_m - start_z_m) / dz);
    }

    amrex::Real const a = dx * dx + dy * dy;
    if (a > amrex::Real(0.0)) {
        amrex::Real const b = amrex::Real(2.0)
            * (result.start_x_m * dx + result.start_y_m * dy);
        // Classify each root from d(r^2)/df = 2 a f + b, which
        // is positive when the chord moves outward, so a crossing of the outer
        // cylinder is an exit iff the derivative is positive there (and of an
        // inner cylinder iff negative).  A grazing tangency (zero derivative)
        // never leaves the domain and is correctly skipped.
        auto retain_cylinder_exit = [&](amrex::Real radius_m, bool outer) {
            amrex::Real const c = result.start_x_m * result.start_x_m
                + result.start_y_m * result.start_y_m - radius_m * radius_m;
            amrex::Real const discriminant = b * b - amrex::Real(4.0) * a * c;
            if (discriminant < amrex::Real(0.0)) {
                return;
            }
            amrex::Real const root = std::sqrt(rrea::kinetic::max(discriminant, amrex::Real(0.0)));
            amrex::Real const inv_2a = amrex::Real(0.5) / a;
            amrex::Real const roots[2] = {
                (-b - root) * inv_2a,
                (-b + root) * inv_2a};
            for (amrex::Real candidate : roots) {
                if (candidate <= amrex::Real(0.0) || candidate > amrex::Real(1.0)) {
                    continue;
                }
                amrex::Real const radial_derivative =
                    amrex::Real(2.0) * a * candidate + b;
                bool const leaves = outer
                    ? radial_derivative > amrex::Real(0.0)
                    : radial_derivative < amrex::Real(0.0);
                if (leaves) {
                    retain_positive_exit(candidate);
                }
            }
        };
        retain_cylinder_exit(domain_r_hi_m, /*outer=*/true);
        if (domain_r_lo_m > amrex::Real(0.0)) {
            retain_cylinder_exit(domain_r_lo_m, /*outer=*/false);
        }
    }

    result.in_domain_end_fraction = rrea::kinetic::max(
        amrex::Real(0.0), rrea::kinetic::min(amrex::Real(1.0), exit_fraction));
    result.in_domain_length_m = result.full_length_m
        * (result.in_domain_end_fraction - result.in_domain_begin_fraction);
    result.exits_domain = result.in_domain_end_fraction
        < amrex::Real(1.0) - amrex::Real(64.0)
            * std::numeric_limits<amrex::Real>::epsilon();

    return result;
}

}  // namespace rrea

#endif
