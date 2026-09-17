#ifndef RREA_ELECTRON_CLOSURE_H
#define RREA_ELECTRON_CLOSURE_H

#include "rrea/RreaGpu.H"
#include <cmath>
#include <cstddef>

namespace rrea {

// The dry-air closure's existing piecewise log-linear model, shared by CPU
// and GPU. Positive coefficients interpolate in log(E/N); two-body onset
// uses a nonnegative linear coefficient across a zero bracket. Range policy
// remains explicit: hold below, report above, reject nonfinite/negative E/N.
// The validated CSV loader remains LowEnergyElectronClosure::Load.
enum class RreaClosureRangeStatus { InRange, HeldBelow, AboveRange, Invalid };

template<class Real>
struct ElectronClosureCoefficientsT {
    Real k0_flux_ref_m2_per_vs = Real(0.0);
    Real nu2_ref_per_s = Real(0.0);
    Real nu3_ref_per_s = Real(0.0);
    Real longitudinal_diffusion_ref_m2_per_s = Real(0.0);
    Real transverse_diffusion_ref_m2_per_s = Real(0.0);
};

template<class Real>
struct ElectronClosureViewT {
    std::size_t size = 0;
    double const* m_en_td = nullptr;
    double const* m_log_en_td = nullptr;
    double const* m_k0_flux_ref = nullptr;
    double const* m_log_k0_flux_ref = nullptr;
    double const* m_nu2_ref = nullptr;
    double const* m_nu3_ref = nullptr;
    double const* m_log_nu3_ref = nullptr;
    double const* m_longitudinal_diffusion_ref = nullptr;
    double const* m_log_longitudinal_diffusion_ref = nullptr;
    double const* m_transverse_diffusion_ref = nullptr;
    double const* m_log_transverse_diffusion_ref = nullptr;

    [[nodiscard]] RREA_HOST_DEVICE ElectronClosureCoefficientsT<Real> Evaluate(
        double en_td, RreaClosureRangeStatus& status) const noexcept
    {
        ElectronClosureCoefficientsT<Real> out;
        if (size == 0 || !std::isfinite(en_td) || en_td < 0.0) {
            status = RreaClosureRangeStatus::Invalid;
            return out;
        }
        std::size_t node = size;
        if (en_td <= m_en_td[0]) {
            status = en_td < m_en_td[0]
                ? RreaClosureRangeStatus::HeldBelow
                : RreaClosureRangeStatus::InRange;
            node = 0;
        } else if (en_td >= m_en_td[size - 1]) {
            status = en_td > m_en_td[size - 1]
                ? RreaClosureRangeStatus::AboveRange
                : RreaClosureRangeStatus::InRange;
            node = size - 1U;
        }
        if (node < size) {
            out.k0_flux_ref_m2_per_vs =
                static_cast<Real>(m_k0_flux_ref[node]);
            out.nu2_ref_per_s = static_cast<Real>(m_nu2_ref[node]);
            out.nu3_ref_per_s = static_cast<Real>(m_nu3_ref[node]);
            out.longitudinal_diffusion_ref_m2_per_s = static_cast<Real>(
                m_longitudinal_diffusion_ref[node]);
            out.transverse_diffusion_ref_m2_per_s = static_cast<Real>(
                m_transverse_diffusion_ref[node]);
            return out;
        }
        status = RreaClosureRangeStatus::InRange;
        // upper_bound on the immutable device view (no host STL iterator).
        std::size_t first = 0, last = size;
        while (first < last) {
            auto const middle = first + (last - first) / 2;
            if (m_en_td[middle] <= en_td) { first = middle + 1; }
            else { last = middle; }
        }
        std::size_t const hi = first;
        std::size_t const lo = hi - 1U;
        if (en_td == m_en_td[lo]) {
            // Exact interior node: return the stored value, not exp(log(value)).
            out.k0_flux_ref_m2_per_vs =
                static_cast<Real>(m_k0_flux_ref[lo]);
            out.nu2_ref_per_s = static_cast<Real>(m_nu2_ref[lo]);
            out.nu3_ref_per_s = static_cast<Real>(m_nu3_ref[lo]);
            out.longitudinal_diffusion_ref_m2_per_s = static_cast<Real>(
                m_longitudinal_diffusion_ref[lo]);
            out.transverse_diffusion_ref_m2_per_s = static_cast<Real>(
                m_transverse_diffusion_ref[lo]);
            return out;
        }
        double const f = (std::log(en_td) - m_log_en_td[lo])
            / (m_log_en_td[hi] - m_log_en_td[lo]);
        out.k0_flux_ref_m2_per_vs = static_cast<Real>(std::exp(
            m_log_k0_flux_ref[lo]
            + f * (m_log_k0_flux_ref[hi] - m_log_k0_flux_ref[lo])));
        double const nu2_lo = m_nu2_ref[lo];
        double const nu2_hi = m_nu2_ref[hi];
        out.nu2_ref_per_s = static_cast<Real>(
            nu2_lo > 0.0 && nu2_hi > 0.0
                ? std::exp(std::log(nu2_lo) + f * std::log(nu2_hi / nu2_lo))
                : nu2_lo + f * (nu2_hi - nu2_lo));
        out.nu3_ref_per_s = static_cast<Real>(std::exp(
            m_log_nu3_ref[lo] + f * (m_log_nu3_ref[hi] - m_log_nu3_ref[lo])));
        out.longitudinal_diffusion_ref_m2_per_s = static_cast<Real>(
            std::exp(
                m_log_longitudinal_diffusion_ref[lo]
                + f * (m_log_longitudinal_diffusion_ref[hi]
                       - m_log_longitudinal_diffusion_ref[lo])));
        out.transverse_diffusion_ref_m2_per_s = static_cast<Real>(
            std::exp(
                m_log_transverse_diffusion_ref[lo]
                + f * (m_log_transverse_diffusion_ref[hi]
                       - m_log_transverse_diffusion_ref[lo])));
        return out;
    }
};

} // namespace rrea
#endif
