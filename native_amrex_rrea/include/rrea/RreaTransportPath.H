#ifndef RREA_TRANSPORT_PATH_H
#define RREA_TRANSPORT_PATH_H

#include <AMReX_REAL.H>

#include "rrea/RreaKineticSupport.H"

namespace rrea {

// One contiguous portion of a transport chord in a single periodic image of
// the physical z domain.  Fractions always refer to the unwrapped start/end
// chord.  Splitting at every periodic seam makes density, field, and optical-
// depth integration additive even when a particle crosses the axial period
// more than once in one WarpX push.
struct RreaTransportPathPiece {
    amrex::Real begin_fraction = 0.0;
    amrex::Real end_fraction = 0.0;
    amrex::Real length_m = 0.0;
    // Add this shift to the unwrapped z coordinate on the open interior of
    // this piece to obtain the corresponding physical-domain coordinate.
    amrex::Real wrapped_z_shift_m = 0.0;
};

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
    bool periodic_z = false;
    amrex::Real domain_r_lo_m = 0.0;
    amrex::Real domain_r_hi_m = 0.0;
    amrex::Real domain_z_lo_m = 0.0;
    amrex::Real domain_z_hi_m = 0.0;
    rrea::KineticVector<RreaTransportPathPiece> in_domain_pieces;

    RREA_HOST_DEVICE void PointAtFullPathFraction(
        amrex::Real fraction,
        amrex::Real& x_m,
        amrex::Real& y_m,
        amrex::Real& z_m) const noexcept;
    RREA_HOST_DEVICE [[nodiscard]] amrex::Real WrapZ(amrex::Real z_m) const noexcept;
    RREA_HOST_DEVICE void PointAtPieceFraction(
        RreaTransportPathPiece const& piece,
        amrex::Real fraction,
        amrex::Real& x_m,
        amrex::Real& y_m,
        amrex::Real& z_m) const noexcept;
    RREA_HOST_DEVICE [[nodiscard]] bool OutsideCertifiedDomain(
        amrex::Real r_m, amrex::Real z_m) const noexcept;
};

// Clip a post-push RZ chord against the cylindrical physical domain.  The
// captured start must be in-domain.  Radial exits are clipped; axial exits are
// either clipped (absorbing) or wrapped for density/field lookup (periodic).
// The coupling rejects all other boundary policies before transport.
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
    amrex::Real domain_z_hi_m,
    bool periodic_z = false);

RREA_HOST_DEVICE inline amrex::Real RreaTransportPathSegment::WrapZ(amrex::Real z_m) const noexcept
{
    if (!periodic_z) {
        return z_m;
    }
    amrex::Real const length = domain_z_hi_m - domain_z_lo_m;
    if (!(length > amrex::Real(0.0))) {
        return z_m;
    }
    amrex::Real wrapped = std::fmod(z_m - domain_z_lo_m, length);
    if (wrapped < amrex::Real(0.0)) {
        wrapped += length;
    }
    return domain_z_lo_m + wrapped;
}

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
    return !periodic_z && (z_m < domain_z_lo_m || z_m >= domain_z_hi_m);
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
    z_m = WrapZ(start_z_m + fraction * (end_z_m - start_z_m));
}

RREA_HOST_DEVICE inline void RreaTransportPathSegment::PointAtPieceFraction(
    RreaTransportPathPiece const& piece,
    amrex::Real fraction,
    amrex::Real& x_m,
    amrex::Real& y_m,
    amrex::Real& z_m) const noexcept
{
    fraction = rrea::kinetic::max(
        piece.begin_fraction,
        rrea::kinetic::min(piece.end_fraction, fraction));
    x_m = start_x_m + fraction * (end_x_m - start_x_m);
    y_m = start_y_m + fraction * (end_y_m - start_y_m);
    z_m = start_z_m + fraction * (end_z_m - start_z_m)
        + piece.wrapped_z_shift_m;
    // Roundoff at a seam can put an endpoint a few ulps outside the closed
    // physical interval.  Keep the piece-local representation one-sided;
    // PointAtFullPathFraction remains the canonical half-open wrapped lookup.
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
    amrex::Real domain_z_hi_m,
    bool periodic_z)
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
    result.periodic_z = periodic_z;
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
        rrea::KineticFail("RREA previous particle position is outside the physical RZ ");
    }

    amrex::Real exit_fraction = amrex::Real(1.0);
    auto retain_positive_exit = [&exit_fraction](amrex::Real candidate) {
        if (candidate >= amrex::Real(0.0) && candidate < exit_fraction) {
            exit_fraction = candidate;
        }
    };
    if (!periodic_z) {
        if (dz > amrex::Real(0.0)) {
            retain_positive_exit((domain_z_hi_m - start_z_m) / dz);
        } else if (dz < amrex::Real(0.0)) {
            retain_positive_exit((domain_z_lo_m - start_z_m) / dz);
        }
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

    // Split the retained chord at every periodic z seam.  The radial clip is
    // already reflected in in_domain_end_fraction, so all pieces below are
    // physically in-domain.  Nonperiodic paths have exactly one piece.
    rrea::KineticVector<amrex::Real> breakpoints;
    breakpoints.push_back(result.in_domain_begin_fraction);
    breakpoints.push_back(result.in_domain_end_fraction);
    if (periodic_z && std::abs(dz) > amrex::Real(0.0)
        && result.in_domain_end_fraction > result.in_domain_begin_fraction) {
        rrea::RreaKineticWide const z0 = static_cast<rrea::RreaKineticWide>(start_z_m);
        rrea::RreaKineticWide const z1 = z0 + static_cast<rrea::RreaKineticWide>(dz)
            * static_cast<rrea::RreaKineticWide>(result.in_domain_end_fraction);
        rrea::RreaKineticWide const zlo = static_cast<rrea::RreaKineticWide>(domain_z_lo_m);
        rrea::RreaKineticWide const period = static_cast<rrea::RreaKineticWide>(
            domain_z_hi_m - domain_z_lo_m);
        rrea::RreaKineticWide const low = rrea::kinetic::min(z0, z1);
        rrea::RreaKineticWide const high = rrea::kinetic::max(z0, z1);
        rrea::RreaKineticWide const first_ld = std::floor((low - zlo) / period) + rrea::RreaKineticWide(1.0);
        rrea::RreaKineticWide const last_ld = std::ceil((high - zlo) / period) - rrea::RreaKineticWide(1.0);
        if (first_ld < static_cast<rrea::RreaKineticWide>(std::numeric_limits<long long>::min())
            || first_ld >= -static_cast<rrea::RreaKineticWide>(std::numeric_limits<long long>::min())
            || last_ld < static_cast<rrea::RreaKineticWide>(std::numeric_limits<long long>::min())
            || last_ld >= -static_cast<rrea::RreaKineticWide>(std::numeric_limits<long long>::min())) {
            rrea::KineticFail("periodic RREA path seam index overflows int64");
        }
        long long const first = static_cast<long long>(first_ld);
        long long const last = static_cast<long long>(last_ld);
        if (last >= first) {
            rrea::RreaKineticWide const seam_count_ld = last_ld - first_ld + rrea::RreaKineticWide(1.0);
            // A physical particle step should cross at most a handful of
            // periods.  Bound malformed microscopic-period inputs before they
            // can request an unbounded allocation.
            if (seam_count_ld > rrea::RreaKineticWide(1000000.0)) {
                rrea::KineticFail("periodic RREA path crosses more than 1,000,000 axial seams");
            }
            std::size_t const seam_count = static_cast<std::size_t>(seam_count_ld);
            breakpoints.reserve(static_cast<std::size_t>(seam_count) + 2U);
            for (long long image = first; image <= last; ++image) {
                rrea::RreaKineticWide const seam_z = zlo
                    + static_cast<rrea::RreaKineticWide>(image) * period;
                amrex::Real const fraction = static_cast<amrex::Real>(
                    (seam_z - z0) / static_cast<rrea::RreaKineticWide>(dz));
                amrex::Real const fraction_tol = amrex::Real(128.0)
                    * std::numeric_limits<amrex::Real>::epsilon()
                    * rrea::kinetic::max(amrex::Real(1.0), std::abs(fraction));
                if (fraction > result.in_domain_begin_fraction + fraction_tol
                    && fraction < result.in_domain_end_fraction - fraction_tol) {
                    breakpoints.push_back(fraction);
                }
                if (image == std::numeric_limits<long long>::max()) {
                    break;
                }
            }
        }
    }
    rrea::kinetic::sort(breakpoints.begin(), breakpoints.end());
    breakpoints.erase(
        rrea::kinetic::unique(
            breakpoints.begin(), breakpoints.end(),
            [](amrex::Real lhs, amrex::Real rhs) {
                return std::abs(lhs - rhs) <= amrex::Real(128.0)
                    * std::numeric_limits<amrex::Real>::epsilon()
                    * rrea::kinetic::max(
                        amrex::Real(1.0),
                        rrea::kinetic::max(std::abs(lhs), std::abs(rhs)));
            }),
        breakpoints.end());
    result.in_domain_pieces.clear();
    result.in_domain_pieces.reserve(
        breakpoints.empty() ? 0U : breakpoints.size() - 1U);
    amrex::Real const period_m = domain_z_hi_m - domain_z_lo_m;
    for (std::size_t index = 1; index < breakpoints.size(); ++index) {
        amrex::Real const begin = breakpoints[index - 1U];
        amrex::Real const end = breakpoints[index];
        if (!(end > begin)) {
            continue;
        }
        amrex::Real const midpoint = amrex::Real(0.5) * (begin + end);
        amrex::Real const midpoint_z = start_z_m + midpoint * dz;
        amrex::Real shift = amrex::Real(0.0);
        if (periodic_z) {
            amrex::Real const image = std::floor(
                (midpoint_z - domain_z_lo_m) / period_m);
            shift = -image * period_m;
        }
        result.in_domain_pieces.push_back(RreaTransportPathPiece{
            begin,
            end,
            result.full_length_m * (end - begin),
            shift});
    }
    return result;
}

}  // namespace rrea

#endif
