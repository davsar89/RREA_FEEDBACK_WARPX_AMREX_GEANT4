#ifndef RREA_MATERIAL_CONTINUITY_H
#define RREA_MATERIAL_CONTINUITY_H

#include "rrea/RreaMaxwellTMRZ.H"

#include <cmath>
#include <limits>

namespace rrea {

inline constexpr double kMaterialContinuityRelativeTolerance = 1.0e-10;
// Binary64 cancellation floor for the material identity, scaled by the largest
// represented charge quantum. The sqrt(eps) allowance leaves cancellation
// headroom without hiding loss of a resolved macroparticle fraction.
inline const double kMaterialContinuityAbsoluteMacroFraction =
    8.0 * 0x1p-26;  // sqrt(binary64 epsilon)

struct MaterialChargeCell {
    double old_charge, new_charge, particle_charge, fluid_charge;
    double source_e, source_pos, source_neg;
};

struct MaterialContinuitySample {
    double relative, absolute_e, signed_e;
    double old_charge, new_charge, divergence, residual;
};

// Discrete material continuity, in SI:
//   residual = rho_new - rho_old + dt * div_RZ(J).
// The backward-error denominator contains the GROSS charge/current/source
// operands, not their potentially cancelling net sum. This primitive is shared
// by the GPU reduction, its exceptional-cell report, and the native smoke.
RREA_HOST_DEVICE inline MaterialContinuitySample EvaluateMaterialContinuity(
    MaxwellTMRZLayout const& layout, int i, int j, double dt,
    double absolute_allowance_e, MaterialChargeCell const& charge,
    double const* jr, double const* jz) noexcept
{
    double const r_c_dr = (i + 0.5) * layout.dr * layout.dr;
    double const radial_hi = (i + 1) * layout.dr
        * jr[layout.ErIndex(i + 1, j)] / r_c_dr;
    double const radial_lo = i * layout.dr
        * jr[layout.ErIndex(i, j)] / r_c_dr;
    double const axial_hi = jz[layout.EzIndex(i, j + 1)] / layout.dz;
    double const axial_lo = jz[layout.EzIndex(i, j)] / layout.dz;
    double const divergence = radial_hi - radial_lo + axial_hi - axial_lo;
    double const residual = charge.new_charge - charge.old_charge + dt * divergence;
    double const relative_scale = std::abs(charge.particle_charge)
        + std::abs(charge.fluid_charge) + std::abs(charge.old_charge)
        + dt * (std::abs(radial_hi) + std::abs(radial_lo)
                + std::abs(axial_hi) + std::abs(axial_lo))
        + dt * qe * (std::max(charge.source_e, 0.0)
                   + std::max(charge.source_pos, 0.0)
                   + std::max(charge.source_neg, 0.0));
    double const volume = layout.CellVolume(i);
    double const floor = absolute_allowance_e * qe
        / (volume * kMaterialContinuityRelativeTolerance);
    double const denominator = std::max(std::max(relative_scale, floor),
                                        std::numeric_limits<double>::min());
    double relative = std::abs(residual) / denominator;
    double absolute_e = std::abs(residual) * volume / qe;
    if (!std::isfinite(relative) || !std::isfinite(relative_scale)
        || !std::isfinite(absolute_e)) {
        // Max reductions must not silently drop NaNs.
        relative = std::numeric_limits<double>::infinity();
        absolute_e = std::numeric_limits<double>::infinity();
    }
    return {relative, absolute_e, residual * volume / qe,
            charge.old_charge, charge.new_charge, divergence, residual};
}

} // namespace rrea
#endif
