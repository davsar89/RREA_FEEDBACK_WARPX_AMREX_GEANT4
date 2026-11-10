#ifndef RREA_POPULATION_RECORD_H
#define RREA_POPULATION_RECORD_H
#include "rrea/RreaCicConservingResample.H"
#include "rrea/RreaConstants.H"
#include <cmath>
namespace rrea::warpx {
// Cell support and phase are computed from exactly the same coordinates and
// momentum on both backends. No alternate GPU deposition shape is introduced.
struct PopulationRecordBuilder {
    double plo[2]{},dx[2]{};
    int small[2]{};
    int rank=0;
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
        // Live particles lie inside the domain, so the support index fits.
        if (!std::isfinite(r) || !std::isfinite(z)) {
            rrea::KineticFail("adaptive_resample_v1 found an invalid global CIC support");
        }
        auto const cr = rrea::cic_axis(r, plo[0], dx[0], small[0]);
        auto const cz = rrea::cic_axis(z, plo[1], dx[1], small[1]);
        rrea::RreaGlobalCicParticle result;
        result.support_r = cr.lo;
        result.support_z = cz.lo;
        result.phase = PhaseBin(ux, uy, uz, theta);
        result.source_rank = rank;
        result.stable_id = low;
        result.fraction_r = cr.frac;
        result.fraction_z = cz.frac;
        result.weight = weight;
        result.candidate = candidate;
        result.stable_id_high = high;
        return result;
    }
};
} // namespace rrea::warpx
#endif
