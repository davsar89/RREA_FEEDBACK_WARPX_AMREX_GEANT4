#ifndef RREA_REDUCED_OBSERVABLES_H
#define RREA_REDUCED_OBSERVABLES_H

// Field-based reduced-CSV observables: the pure MultiFab -> struct
// reductions behind the quoted numbers (field_fraction_*, en_td_accel_*,
// fluid_current_* columns).  Header-inline so the standalone
// rrea_reduced_observables_smoke can pin each against analytic fields
// without linking the WarpX overlay; the coupling's AppendReducedDiagnostics
// is a consumer, not an owner.  Everything here is collective (batched
// allreduces) and deterministic.

#include "rrea/RreaConstants.H"

#include <AMReX_Array4.H>
#include <AMReX_Geometry.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace rrea {

// Index-space origin: ProbLo shifted so index i maps to
// plo + (i + 1/2) dx regardless of the domain's smallEnd.
inline auto cell_index_origin(amrex::Geometry const& geom)
{
    auto origin = geom.ProbLoArray();
    auto const dx = geom.CellSizeArray();
    auto const domain_lo = geom.Domain().smallEnd();
    for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
        origin[direction] -=
            static_cast<amrex::Real>(domain_lo[direction]) * dx[direction];
    }
    return origin;
}

// Physical RZ cell volume for cell index (i, j); rz_shell_volume owns the
// formula.
inline amrex::Real cell_volume_rz(amrex::Geometry const& geom, int i, int j)
{
    amrex::ignore_unused(j);
    auto const dx = geom.CellSizeArray();
    auto const plo = cell_index_origin(geom);
    double const r_inner = plo[0] + static_cast<double>(i) * dx[0];
#if (AMREX_SPACEDIM >= 2)
    double const dz = dx[1];
#else
    double const dz = 1.0;
#endif
    return static_cast<amrex::Real>(
        rz_shell_volume(r_inner, r_inner + dx[0], dz));
}

// Field-fraction statistics of |Ez_realized| / |Ez_imposed| over the
// imposed-field (acceleration) region: how much of the driving field
// survives screening -- mean/min/p10/max, the volume fraction below one
// half, and the region's mean |E| (reduced-CSV field_fraction_* columns).
// p10 is the volume-weighted 10th percentile from a 512-bin linear
// histogram over the realized span, reported as the geometric bin center.
struct AccelerationFieldMetrics {
    amrex::Real mean_fraction = amrex::Real(0.0);
    amrex::Real min_fraction = std::numeric_limits<amrex::Real>::infinity();
    amrex::Real p10_fraction = amrex::Real(0.0);
    amrex::Real max_fraction = amrex::Real(0.0);
    amrex::Real fraction_below_half = amrex::Real(0.0);
    amrex::Real volume_m3 = amrex::Real(0.0);
    amrex::Real mean_abs_e_v_per_m = amrex::Real(0.0);
};

inline AccelerationFieldMetrics acceleration_field_metrics_rz(
    amrex::MultiFab const& ez,
    amrex::Geometry const& geom,
    amrex::Real e0_abs,
    amrex::Real z_min_m,
    amrex::Real z_max_m,
    amrex::Real r_max_m)
{
    AccelerationFieldMetrics metrics;
    if (e0_abs <= amrex::Real(0.0)) {
        return metrics;
    }

    amrex::Real weighted_fraction = amrex::Real(0.0);
    amrex::Real below_half_volume = amrex::Real(0.0);
    auto const dx = geom.CellSizeArray();
    auto const plo = cell_index_origin(geom);
    for (amrex::MFIter mfi(ez, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const bx = mfi.validbox();
        auto const arr = ez.const_array(mfi);
        amrex::LoopOnCpu(bx, [&](int i, int j, int k) noexcept {
#if (AMREX_SPACEDIM >= 2)
            amrex::Real const z =
                plo[1] + (static_cast<amrex::Real>(j) + amrex::Real(0.5)) * dx[1];
#else
            amrex::Real const z = amrex::Real(0.0);
#endif
            if (z < z_min_m || z > z_max_m) {
                return;
            }
            amrex::Real const r =
                plo[0] + (static_cast<amrex::Real>(i) + amrex::Real(0.5)) * dx[0];
            if (r > r_max_m) {
                return;
            }
            amrex::Real const volume = cell_volume_rz(geom, i, j);
            amrex::Real const fraction = std::abs(arr(i, j, k)) / e0_abs;
            weighted_fraction += fraction * volume;
            metrics.volume_m3 += volume;
            metrics.mean_abs_e_v_per_m += std::abs(arr(i, j, k)) * volume;
            metrics.min_fraction = amrex::min(metrics.min_fraction, fraction);
            metrics.max_fraction = amrex::max(metrics.max_fraction, fraction);
            if (fraction < amrex::Real(0.5)) {
                below_half_volume += volume;
            }
        });
    }
    // One array allreduce for the four sums.
    {
        amrex::Real sums[4] = {
            weighted_fraction,
            metrics.volume_m3,
            metrics.mean_abs_e_v_per_m,
            below_half_volume};
        amrex::ParallelDescriptor::ReduceRealSum(sums, 4);
        weighted_fraction = sums[0];
        metrics.volume_m3 = sums[1];
        metrics.mean_abs_e_v_per_m = sums[2];
        below_half_volume = sums[3];
    }
    amrex::ParallelDescriptor::ReduceRealMin(metrics.min_fraction);
    amrex::ParallelDescriptor::ReduceRealMax(metrics.max_fraction);
    if (metrics.volume_m3 > amrex::Real(0.0)) {
        metrics.mean_fraction = weighted_fraction / metrics.volume_m3;
        metrics.fraction_below_half = below_half_volume / metrics.volume_m3;
        metrics.mean_abs_e_v_per_m /= metrics.volume_m3;
    } else {
        metrics.min_fraction = amrex::Real(0.0);
    }
    if (metrics.volume_m3 > amrex::Real(0.0)) {
        if (metrics.max_fraction <= metrics.min_fraction) {
            metrics.p10_fraction = metrics.min_fraction;
        } else {
            static constexpr int bins = 512;
            amrex::Vector<amrex::Real> histogram(bins, amrex::Real(0.0));
            amrex::Real const span = metrics.max_fraction - metrics.min_fraction;
            for (amrex::MFIter mfi(ez, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                auto const bx = mfi.validbox();
                auto const arr = ez.const_array(mfi);
                amrex::LoopOnCpu(bx, [&](int i, int j, int k) noexcept {
#if (AMREX_SPACEDIM >= 2)
                    amrex::Real const z =
                        plo[1] + (static_cast<amrex::Real>(j) + amrex::Real(0.5)) * dx[1];
#else
                    amrex::Real const z = amrex::Real(0.0);
#endif
                    if (z < z_min_m || z > z_max_m) {
                        return;
                    }
                    amrex::Real const r =
                        plo[0] + (static_cast<amrex::Real>(i) + amrex::Real(0.5)) * dx[0];
                    if (r > r_max_m) {
                        return;
                    }
                    amrex::Real const fraction = std::abs(arr(i, j, k)) / e0_abs;
                    int bin = static_cast<int>(
                        (fraction - metrics.min_fraction) / span * static_cast<amrex::Real>(bins));
                    bin = std::max(0, std::min(bins - 1, bin));
                    histogram[bin] += cell_volume_rz(geom, i, j);
                });
            }
            // One array allreduce for the whole
            // histogram instead of 512 sequential scalar allreduces.
            amrex::ParallelDescriptor::ReduceRealSum(
                histogram.data(), static_cast<int>(histogram.size()));
            amrex::Real const target = amrex::Real(0.10) * metrics.volume_m3;
            amrex::Real cumulative = amrex::Real(0.0);
            metrics.p10_fraction = metrics.max_fraction;
            for (int bin = 0; bin < bins; ++bin) {
                cumulative += histogram[bin];
                if (cumulative >= target) {
                    metrics.p10_fraction = metrics.min_fraction
                        + (static_cast<amrex::Real>(bin) + amrex::Real(0.5))
                            * span / static_cast<amrex::Real>(bins);
                    break;
                }
            }
        }
    }
    return metrics;
}

// Weighted mean of |Ez|/E0 over the region (weight clamped non-negative);
// fraction is NaN when the weight integral vanishes -- consumers fail
// CLOSED on a missing weighting, never open to an unweighted field.
struct WeightedFieldMetric {
    amrex::Real fraction = std::numeric_limits<amrex::Real>::quiet_NaN();
    amrex::Real weight_integral = amrex::Real(0.0);
};

inline WeightedFieldMetric weighted_field_fraction_rz(
    amrex::MultiFab const& ez,
    amrex::MultiFab const& weight,
    amrex::Geometry const& geom,
    amrex::Real e0_abs,
    amrex::Real z_min_m,
    amrex::Real z_max_m,
    amrex::Real r_max_m)
{
    WeightedFieldMetric metric;
    if (e0_abs <= amrex::Real(0.0)) {
        return metric;
    }
    amrex::Real weighted_fraction = amrex::Real(0.0);
    auto const dx = geom.CellSizeArray();
    auto const plo = cell_index_origin(geom);
    for (amrex::MFIter mfi(ez, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const bx = mfi.validbox();
        auto const ez_arr = ez.const_array(mfi);
        auto const w_arr = weight.const_array(mfi);
        amrex::LoopOnCpu(bx, [&](int i, int j, int k) noexcept {
#if (AMREX_SPACEDIM >= 2)
            amrex::Real const z =
                plo[1] + (static_cast<amrex::Real>(j) + amrex::Real(0.5)) * dx[1];
#else
            amrex::Real const z = amrex::Real(0.0);
#endif
            amrex::Real const r =
                plo[0] + (static_cast<amrex::Real>(i) + amrex::Real(0.5)) * dx[0];
            if (z < z_min_m || z > z_max_m || r > r_max_m) {
                return;
            }
            amrex::Real const volume = cell_volume_rz(geom, i, j);
            amrex::Real const w = amrex::max(w_arr(i, j, k), amrex::Real(0.0));
            weighted_fraction += w * volume * std::abs(ez_arr(i, j, k)) / e0_abs;
            metric.weight_integral += w * volume;
        });
    }
    // One array allreduce for both sums.
    {
        amrex::Real sums[2] = {weighted_fraction, metric.weight_integral};
        amrex::ParallelDescriptor::ReduceRealSum(sums, 2);
        weighted_fraction = sums[0];
        metric.weight_integral = sums[1];
    }
    if (metric.weight_integral > amrex::Real(0.0)) {
        metric.fraction = weighted_fraction / metric.weight_integral;
    }
    return metric;
}

// Realized reduced-field E/N = |E|*1e21/(chi*N0) distribution over the
// acceleration region. The complete
// |E| = hypot(Er, Ez) and the Loschmidt-relative mobility density ratio, on
// fixed log-spaced bins (160 over 0.03-300 Td) so the percentiles are
// deterministic and reduce in one batched allreduce; percentiles report the
// geometric bin center clamped into the realized span.  Three weight
// channels answer three different questions -- where the VOLUME sits, where
// the existing LOW-ENERGY ELECTRONS sit, and where the next plasma is being
// CREATED -- which separate widely during a collapsing avalanche; no single
// weighting is "the" operating E/N.  The 3/30 Td markers are the de Urquijo
// dry-air certified range: these columns are what decide (and later audit)
// the en_table closure's range policy on the real production state
// (en_td_accel_* reduced-CSV columns).
struct EnTdRegionMetrics {
    static constexpr int bins = 160;
    static constexpr amrex::Real bin_lo_td = amrex::Real(0.03);
    static constexpr amrex::Real bin_hi_td = amrex::Real(300.0);
    amrex::Real min_en_td = std::numeric_limits<amrex::Real>::quiet_NaN();
    amrex::Real max_en_td = std::numeric_limits<amrex::Real>::quiet_NaN();
    // Channels: 0 = volume, 1 = ne_low-weighted, 2 = ionization-source
    // weighted.  Percentiles are NaN when a channel has zero total weight.
    std::array<std::array<amrex::Real, 3>, 3> p10_p50_p90{};
    std::array<amrex::Real, 3> weighted_mean_en_td{};
    amrex::Real vol_frac_below_3td = std::numeric_limits<amrex::Real>::quiet_NaN();
    amrex::Real vol_frac_above_30td = std::numeric_limits<amrex::Real>::quiet_NaN();
    amrex::Real ne_frac_below_3td = std::numeric_limits<amrex::Real>::quiet_NaN();
};

inline EnTdRegionMetrics en_td_region_metrics_rz(
    amrex::MultiFab const& er,
    amrex::MultiFab const& ez,
    amrex::MultiFab const* density_ratio,
    amrex::Real scalar_density_ratio,
    amrex::MultiFab const& ne_low,
    amrex::MultiFab const& s_low,
    amrex::Geometry const& geom,
    amrex::Real z_min_m,
    amrex::Real z_max_m,
    amrex::Real r_max_m)
{
    EnTdRegionMetrics metrics;
    constexpr int bins = EnTdRegionMetrics::bins;
    constexpr int channels = 3;
    double const log_lo =
        std::log(static_cast<double>(EnTdRegionMetrics::bin_lo_td));
    double const inv_log_width = static_cast<double>(bins)
        / (std::log(static_cast<double>(EnTdRegionMetrics::bin_hi_td))
           - log_lo);
    // Layout: channels*bins histogram weights, the below-3/above-30 volume
    // and below-3 ne split, three weighted E/N sums, then -min/max.
    std::vector<amrex::Real> sums(
        static_cast<std::size_t>(channels * bins) + 8,
        amrex::Real(0.0));
    double local_min = std::numeric_limits<double>::max();
    double local_max = 0.0;
    auto const dx = geom.CellSizeArray();
    auto const plo = cell_index_origin(geom);
    for (amrex::MFIter mfi(ez, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const bx = mfi.validbox();
        auto const er_arr = er.const_array(mfi);
        auto const ez_arr = ez.const_array(mfi);
        auto const ne_arr = ne_low.const_array(mfi);
        auto const src_arr = s_low.const_array(mfi);
        auto const den_arr = density_ratio
            ? density_ratio->const_array(mfi)
            : amrex::Array4<amrex::Real const>{};
        bool const has_density = density_ratio != nullptr;
        amrex::LoopOnCpu(bx, [&](int i, int j, int k) noexcept {
#if (AMREX_SPACEDIM >= 2)
            amrex::Real const z =
                plo[1] + (static_cast<amrex::Real>(j) + amrex::Real(0.5)) * dx[1];
#else
            amrex::Real const z = amrex::Real(0.0);
#endif
            amrex::Real const r =
                plo[0] + (static_cast<amrex::Real>(i) + amrex::Real(0.5)) * dx[0];
            if (z < z_min_m || z > z_max_m || r > r_max_m) {
                return;
            }
            double const chi = has_density
                ? static_cast<double>(den_arr(i, j, k))
                : static_cast<double>(scalar_density_ratio);
            if (!(chi > 0.0) || !std::isfinite(chi)) {
                return;
            }
            double const e_mag = std::hypot(
                static_cast<double>(er_arr(i, j, k)),
                static_cast<double>(ez_arr(i, j, k)));
            double const en_td = e_mag * 1.0e21 / (chi * rrea::n_loschmidt_m3);
            if (!std::isfinite(en_td)) {
                return;
            }
            local_min = std::min(local_min, en_td);
            local_max = std::max(local_max, en_td);
            double const volume =
                static_cast<double>(cell_volume_rz(geom, i, j));
            double const weight[channels] = {
                volume,
                std::max(static_cast<double>(ne_arr(i, j, k)), 0.0) * volume,
                std::max(static_cast<double>(src_arr(i, j, k)), 0.0) * volume};
            std::size_t const extras = static_cast<std::size_t>(channels * bins);
            int bin = en_td > 0.0
                ? static_cast<int>((std::log(en_td) - log_lo) * inv_log_width)
                : 0;
            bin = std::max(0, std::min(bins - 1, bin));
            for (int channel = 0; channel < channels; ++channel) {
                sums[static_cast<std::size_t>(channel * bins + bin)] +=
                    static_cast<amrex::Real>(weight[channel]);
                sums[extras + 3 + static_cast<std::size_t>(channel)] +=
                    static_cast<amrex::Real>(weight[channel] * en_td);
            }
            if (en_td < 3.0) {
                sums[extras + 0] += static_cast<amrex::Real>(weight[0]);
                sums[extras + 2] += static_cast<amrex::Real>(weight[1]);
            }
            if (en_td > 30.0) {
                sums[extras + 1] += static_cast<amrex::Real>(weight[0]);
            }
        });
    }
    {
        std::size_t const extras = static_cast<std::size_t>(channels * bins);
        sums[extras + 6] = static_cast<amrex::Real>(-local_min);
        sums[extras + 7] = static_cast<amrex::Real>(local_max);
        // Histogram sums and negated
        // extrema in one array reduce; max(-min) == -global_min.
        amrex::ParallelDescriptor::ReduceRealMax(
            sums.data() + extras + 6, 2);
        amrex::ParallelDescriptor::ReduceRealSum(
            sums.data(), static_cast<int>(extras + 6));
    }
    std::size_t const extras = static_cast<std::size_t>(channels * bins);
    if (static_cast<double>(sums[extras + 7]) <= 0.0
        && -static_cast<double>(sums[extras + 6])
            >= std::numeric_limits<double>::max()) {
        return metrics;  // empty region on every rank
    }
    metrics.min_en_td = -sums[extras + 6];
    metrics.max_en_td = sums[extras + 7];
    for (int channel = 0; channel < channels; ++channel) {
        amrex::Real total = amrex::Real(0.0);
        for (int bin = 0; bin < bins; ++bin) {
            total += sums[static_cast<std::size_t>(channel * bins + bin)];
        }
        if (!(total > amrex::Real(0.0))) {
            metrics.weighted_mean_en_td[static_cast<std::size_t>(channel)] =
                std::numeric_limits<amrex::Real>::quiet_NaN();
            metrics.p10_p50_p90[static_cast<std::size_t>(channel)] = {
                std::numeric_limits<amrex::Real>::quiet_NaN(),
                std::numeric_limits<amrex::Real>::quiet_NaN(),
                std::numeric_limits<amrex::Real>::quiet_NaN()};
            continue;
        }
        metrics.weighted_mean_en_td[static_cast<std::size_t>(channel)] =
            sums[extras + 3 + static_cast<std::size_t>(channel)] / total;
        std::array<amrex::Real, 3> const quantiles = {
            amrex::Real(0.10), amrex::Real(0.50), amrex::Real(0.90)};
        for (std::size_t q = 0; q < quantiles.size(); ++q) {
            amrex::Real const target = quantiles[q] * total;
            amrex::Real cumulative = amrex::Real(0.0);
            amrex::Real value = metrics.max_en_td;
            for (int bin = 0; bin < bins; ++bin) {
                cumulative +=
                    sums[static_cast<std::size_t>(channel * bins + bin)];
                if (cumulative >= target) {
                    value = static_cast<amrex::Real>(std::exp(
                        log_lo
                        + (static_cast<double>(bin) + 0.5) / inv_log_width));
                    break;
                }
            }
            // Clamp the geometric bin center into the exact realized span so
            // the edge bins' catch-all role cannot report an E/N outside it.
            value = std::max(
                metrics.min_en_td, std::min(metrics.max_en_td, value));
            metrics.p10_p50_p90[static_cast<std::size_t>(channel)]
                   [q] = value;
        }
    }
    amrex::Real const vol_total = [&sums]() {
        amrex::Real total = amrex::Real(0.0);
        for (int bin = 0; bin < bins; ++bin) {
            total += sums[static_cast<std::size_t>(bin)];
        }
        return total;
    }();
    amrex::Real const ne_total = [&sums]() {
        amrex::Real total = amrex::Real(0.0);
        for (int bin = 0; bin < bins; ++bin) {
            total += sums[static_cast<std::size_t>(bins + bin)];
        }
        return total;
    }();
    if (vol_total > amrex::Real(0.0)) {
        metrics.vol_frac_below_3td = sums[extras + 0] / vol_total;
        metrics.vol_frac_above_30td = sums[extras + 1] / vol_total;
    }
    if (ne_total > amrex::Real(0.0)) {
        metrics.ne_frac_below_3td = sums[extras + 2] / ne_total;
    }
    return metrics;
}

// sigma|Ez|-weighted conduction-current statistics over the region: mean
// sigma|E| and signed sigma Ez (A/m^2), the axial current proxy
// I = <sigma Ez> V / L_z, the predicted ambient |E| decay rate
// -<sigma|E|>/eps0, and the sigma|Ez|-weighted spatial moments needed to size
// the conduction-current source independently of the energetic beam. Raw sums
// in the loop are finalized to rms after the allreduce.
struct OhmicDischargeMetrics {
    amrex::Real mean_sigma_abs_e_A_per_m2 = amrex::Real(0.0);
    amrex::Real mean_sigma_ez_A_per_m2 = amrex::Real(0.0);
    amrex::Real current_proxy_A = amrex::Real(0.0);
    amrex::Real predicted_abs_e_decay_v_m_s = amrex::Real(0.0);
    amrex::Real volume_m3 = amrex::Real(0.0);
    amrex::Real sigma_e_weight = amrex::Real(0.0);
    amrex::Real sigma_e_r2 = amrex::Real(0.0);
    amrex::Real sigma_e_z = amrex::Real(0.0);
    amrex::Real sigma_e_z2 = amrex::Real(0.0);
    amrex::Real fluid_current_rms_r_m = amrex::Real(0.0);
    amrex::Real fluid_current_rms_z_m = amrex::Real(0.0);
};

inline OhmicDischargeMetrics ohmic_discharge_metrics_rz(
    amrex::MultiFab const& ez,
    amrex::MultiFab const& sigma,
    amrex::Geometry const& geom,
    amrex::Real z_min_m,
    amrex::Real z_max_m,
    amrex::Real r_max_m)
{
    OhmicDischargeMetrics metrics;
    auto const dx = geom.CellSizeArray();
    auto const plo = cell_index_origin(geom);
    amrex::Real z_length = amrex::max(z_max_m - z_min_m, amrex::Real(0.0));
    for (amrex::MFIter mfi(ez, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const bx = mfi.validbox();
        auto const ez_arr = ez.const_array(mfi);
        auto const sigma_arr = sigma.const_array(mfi);
        amrex::LoopOnCpu(bx, [&](int i, int j, int k) noexcept {
#if (AMREX_SPACEDIM >= 2)
            amrex::Real const z =
                plo[1] + (static_cast<amrex::Real>(j) + amrex::Real(0.5)) * dx[1];
#else
            amrex::Real const z = amrex::Real(0.0);
#endif
            amrex::Real const r =
                plo[0] + (static_cast<amrex::Real>(i) + amrex::Real(0.5)) * dx[0];
            if (z < z_min_m || z > z_max_m || r > r_max_m) {
                return;
            }
            amrex::Real const volume = cell_volume_rz(geom, i, j);
            amrex::Real const sigma_value = amrex::max(sigma_arr(i, j, k), amrex::Real(0.0));
            amrex::Real const ez_value = ez_arr(i, j, k);
            metrics.mean_sigma_abs_e_A_per_m2 += sigma_value * std::abs(ez_value) * volume;
            metrics.mean_sigma_ez_A_per_m2 += sigma_value * ez_value * volume;
            metrics.volume_m3 += volume;
            amrex::Real const w = sigma_value * std::abs(ez_value) * volume;
            metrics.sigma_e_weight += w;
            metrics.sigma_e_r2 += w * r * r;
            metrics.sigma_e_z += w * z;
            metrics.sigma_e_z2 += w * z * z;
        });
    }
    // One array allreduce for the seven sums.
    {
        amrex::Real sums[7] = {
            metrics.mean_sigma_abs_e_A_per_m2,
            metrics.mean_sigma_ez_A_per_m2,
            metrics.volume_m3,
            metrics.sigma_e_weight,
            metrics.sigma_e_r2,
            metrics.sigma_e_z,
            metrics.sigma_e_z2};
        amrex::ParallelDescriptor::ReduceRealSum(sums, 7);
        metrics.mean_sigma_abs_e_A_per_m2 = sums[0];
        metrics.mean_sigma_ez_A_per_m2 = sums[1];
        metrics.volume_m3 = sums[2];
        metrics.sigma_e_weight = sums[3];
        metrics.sigma_e_r2 = sums[4];
        metrics.sigma_e_z = sums[5];
        metrics.sigma_e_z2 = sums[6];
    }
    if (metrics.sigma_e_weight > amrex::Real(0.0)) {
        metrics.fluid_current_rms_r_m =
            std::sqrt(metrics.sigma_e_r2 / metrics.sigma_e_weight);
        amrex::Real const mean_z = metrics.sigma_e_z / metrics.sigma_e_weight;
        metrics.fluid_current_rms_z_m = std::sqrt(amrex::max(
            metrics.sigma_e_z2 / metrics.sigma_e_weight - mean_z * mean_z,
            amrex::Real(0.0)));
    }
    if (metrics.volume_m3 > amrex::Real(0.0)) {
        metrics.mean_sigma_abs_e_A_per_m2 /= metrics.volume_m3;
        metrics.mean_sigma_ez_A_per_m2 /= metrics.volume_m3;
        metrics.predicted_abs_e_decay_v_m_s =
            -metrics.mean_sigma_abs_e_A_per_m2 / static_cast<amrex::Real>(rrea::eps0);
        if (z_length > amrex::Real(0.0)) {
            metrics.current_proxy_A =
                metrics.mean_sigma_ez_A_per_m2 * metrics.volume_m3 / z_length;
        }
    }
    return metrics;
}

}  // namespace rrea

#endif
