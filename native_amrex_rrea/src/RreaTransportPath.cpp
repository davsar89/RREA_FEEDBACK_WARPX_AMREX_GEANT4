#include "rrea/RreaTransportPath.H"

#include <AMReX.H>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

namespace rrea {

amrex::Real RreaTransportPathSegment::WrapZ(amrex::Real z_m) const noexcept
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

bool RreaTransportPathSegment::OutsideCertifiedDomain(
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

void RreaTransportPathSegment::PointAtFullPathFraction(
    amrex::Real fraction,
    amrex::Real& x_m,
    amrex::Real& y_m,
    amrex::Real& z_m) const noexcept
{
    fraction = std::max(amrex::Real(0.0), std::min(amrex::Real(1.0), fraction));
    x_m = start_x_m + fraction * (end_x_m - start_x_m);
    y_m = start_y_m + fraction * (end_y_m - start_y_m);
    z_m = WrapZ(start_z_m + fraction * (end_z_m - start_z_m));
}

void RreaTransportPathSegment::PointAtPieceFraction(
    RreaTransportPathPiece const& piece,
    amrex::Real fraction,
    amrex::Real& x_m,
    amrex::Real& y_m,
    amrex::Real& z_m) const noexcept
{
    fraction = std::max(
        piece.begin_fraction,
        std::min(piece.end_fraction, fraction));
    x_m = start_x_m + fraction * (end_x_m - start_x_m);
    y_m = start_y_m + fraction * (end_y_m - start_y_m);
    z_m = start_z_m + fraction * (end_z_m - start_z_m)
        + piece.wrapped_z_shift_m;
    // Roundoff at a seam can put an endpoint a few ulps outside the closed
    // physical interval.  Keep the piece-local representation one-sided;
    // PointAtFullPathFraction remains the canonical half-open wrapped lookup.
    z_m = std::max(domain_z_lo_m, std::min(domain_z_hi_m, z_m));
}

RreaTransportPathSegment ClipRzTransportPath(
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
        amrex::Abort("ClipRzTransportPath received invalid geometry or particle coordinates");
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

    amrex::Real const scale = std::max(
        amrex::Real(1.0),
        std::max(domain_r_hi_m,
            std::max(std::abs(domain_z_lo_m), std::abs(domain_z_hi_m))));
    amrex::Real const tol = amrex::Real(64.0)
        * std::numeric_limits<amrex::Real>::epsilon() * scale;
    if (start_r_m < domain_r_lo_m - tol || start_r_m > domain_r_hi_m + tol
        || start_z_m < domain_z_lo_m - tol || start_z_m > domain_z_hi_m + tol) {
        std::ostringstream detail;
        detail << std::setprecision(17)
               << "RREA previous particle position is outside the physical RZ "
                  "domain; pre-push capture or boundary ordering is invalid: "
                  "start_r_m=" << start_r_m << " start_z_m=" << start_z_m
               << " end_r_m=" << end_r_m << " end_z_m=" << end_z_m
               << " domain_r=[" << domain_r_lo_m << ',' << domain_r_hi_m
               << "] domain_z=[" << domain_z_lo_m << ',' << domain_z_hi_m
               << "] tol=" << tol;
        amrex::Abort(detail.str());
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
            amrex::Real const root = std::sqrt(std::max(discriminant, amrex::Real(0.0)));
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

    result.in_domain_end_fraction = std::max(
        amrex::Real(0.0), std::min(amrex::Real(1.0), exit_fraction));
    result.in_domain_length_m = result.full_length_m
        * (result.in_domain_end_fraction - result.in_domain_begin_fraction);
    result.exits_domain = result.in_domain_end_fraction
        < amrex::Real(1.0) - amrex::Real(64.0)
            * std::numeric_limits<amrex::Real>::epsilon();

    // Split the retained chord at every periodic z seam.  The radial clip is
    // already reflected in in_domain_end_fraction, so all pieces below are
    // physically in-domain.  Nonperiodic paths have exactly one piece.
    std::vector<amrex::Real> breakpoints = {
        result.in_domain_begin_fraction,
        result.in_domain_end_fraction};
    if (periodic_z && std::abs(dz) > amrex::Real(0.0)
        && result.in_domain_end_fraction > result.in_domain_begin_fraction) {
        long double const z0 = static_cast<long double>(start_z_m);
        long double const z1 = z0 + static_cast<long double>(dz)
            * static_cast<long double>(result.in_domain_end_fraction);
        long double const zlo = static_cast<long double>(domain_z_lo_m);
        long double const period = static_cast<long double>(
            domain_z_hi_m - domain_z_lo_m);
        long double const low = std::min(z0, z1);
        long double const high = std::max(z0, z1);
        long double const first_ld = std::floor((low - zlo) / period) + 1.0L;
        long double const last_ld = std::ceil((high - zlo) / period) - 1.0L;
        if (first_ld < static_cast<long double>(std::numeric_limits<long long>::min())
            || first_ld > static_cast<long double>(std::numeric_limits<long long>::max())
            || last_ld < static_cast<long double>(std::numeric_limits<long long>::min())
            || last_ld > static_cast<long double>(std::numeric_limits<long long>::max())) {
            amrex::Abort("periodic RREA path seam index overflows int64");
        }
        long long const first = static_cast<long long>(first_ld);
        long long const last = static_cast<long long>(last_ld);
        if (last >= first) {
            long double const seam_count_ld = last_ld - first_ld + 1.0L;
            // A physical particle step should cross at most a handful of
            // periods.  Bound malformed microscopic-period inputs before they
            // can request an unbounded allocation.
            if (seam_count_ld > 1000000.0L) {
                amrex::Abort("periodic RREA path crosses more than 1,000,000 axial seams");
            }
            std::size_t const seam_count = static_cast<std::size_t>(seam_count_ld);
            breakpoints.reserve(static_cast<std::size_t>(seam_count) + 2U);
            for (long long image = first; image <= last; ++image) {
                long double const seam_z = zlo
                    + static_cast<long double>(image) * period;
                amrex::Real const fraction = static_cast<amrex::Real>(
                    (seam_z - z0) / static_cast<long double>(dz));
                amrex::Real const fraction_tol = amrex::Real(128.0)
                    * std::numeric_limits<amrex::Real>::epsilon()
                    * std::max(amrex::Real(1.0), std::abs(fraction));
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
    std::sort(breakpoints.begin(), breakpoints.end());
    breakpoints.erase(
        std::unique(
            breakpoints.begin(), breakpoints.end(),
            [](amrex::Real lhs, amrex::Real rhs) {
                return std::abs(lhs - rhs) <= amrex::Real(128.0)
                    * std::numeric_limits<amrex::Real>::epsilon()
                    * std::max(
                        amrex::Real(1.0),
                        std::max(std::abs(lhs), std::abs(rhs)));
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
