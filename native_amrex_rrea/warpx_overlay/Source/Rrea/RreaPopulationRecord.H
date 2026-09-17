#ifndef RREA_POPULATION_RECORD_H
#define RREA_POPULATION_RECORD_H
#include "rrea/RreaCicConservingResample.H"
#include <cmath>
#include <limits>
namespace rrea::warpx {
// Cell support and phase are computed from exactly the same coordinates and
// momentum on both backends. No alternate GPU deposition shape is introduced.
struct PopulationRecordBuilder {
    double plo[2]{},phi[2]{},dx[2]{};
    int small[2]{};
    bool periodic[2]{};
    int rank=0;
    RREA_HOST_DEVICE double CanonicalPosition(double value, int dim) const noexcept {
        if (!periodic[dim]) return value;
        double const length = phi[dim] - plo[dim];
        double shifted = std::fmod(value - plo[dim], length);
        if (shifted < 0.0) shifted += length;
        return plo[dim] + shifted;
    }
    RREA_HOST_DEVICE static int PhaseBin(double ux, double uy, double uz, double theta) noexcept {
        double const c = std::cos(theta);
        double const s = std::sin(theta);
        double const ur = ux * c + uy * s;
        double const utheta = -ux * s + uy * c;
        double const magnitude = std::sqrt(ux * ux + uy * uy + uz * uz);
        int exponent = magnitude > 0.0 && std::isfinite(magnitude)
            ? std::ilogb(magnitude) : -1024;
        exponent = rrea::kinetic::max(-1024, rrea::kinetic::min(1023, exponent));
        return (exponent + 1024) * 8
            + (ur >= 0.0 ? 1 : 0)
            + (utheta >= 0.0 ? 2 : 0)
            + (uz >= 0.0 ? 4 : 0);
    }
    RREA_HOST_DEVICE rrea::RreaGlobalCicParticle operator()(double r, double z, double ux, double uy, double uz,
                      double theta, double weight, bool candidate,
                      std::uint64_t high, std::uint64_t low) const noexcept {
        r = CanonicalPosition(r, 0);
        z = CanonicalPosition(z, 1);
        double const r_cell = (r - plo[0]) / dx[0]
            + static_cast<double>(small[0]) - 0.5;
        double const z_cell = (z - plo[1]) / dx[1]
            + static_cast<double>(small[1]) - 0.5;
        double const i0 = std::floor(r_cell);
        double const j0 = std::floor(z_cell);
        if (!std::isfinite(r_cell) || !std::isfinite(z_cell)
            || i0 < std::numeric_limits<int>::min()
            || i0 > std::numeric_limits<int>::max()
            || j0 < std::numeric_limits<int>::min()
            || j0 > std::numeric_limits<int>::max()) {
            rrea::KineticFail("adaptive_resample_v1 found an invalid global CIC support");
        }
        rrea::RreaGlobalCicParticle result;
        result.support_r = static_cast<int>(i0);
        result.support_z = static_cast<int>(j0);
        result.phase = PhaseBin(ux, uy, uz, theta);
        result.source_rank = rank;
        result.stable_id = low;
        result.fraction_r = r_cell - i0;
        result.fraction_z = z_cell - j0;
        result.weight = weight;
        result.candidate = candidate;
        result.stable_id_high = high;
        return result;
    }
};
} // namespace rrea::warpx
#endif
