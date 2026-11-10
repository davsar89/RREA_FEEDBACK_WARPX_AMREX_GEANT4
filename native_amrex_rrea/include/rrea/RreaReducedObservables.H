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
    double const r_inner = cell_index_origin(geom)[0] + static_cast<double>(i) * dx[0];
    return static_cast<amrex::Real>(
        rz_shell_volume(r_inner, r_inner + dx[0], dx[1]));
}

// The one region loop of these reductions: for every valid cell whose centre
// lies in z_min <= z <= z_max, r <= r_max, calls cell(i, j, k, r, z, volume)
// with the physical RZ volume.  box(mfi) runs once per box and returns that
// box's cell callable, so the callers bind their Array4 views once per box.
template <class BoxBody>
void for_each_region_cell_rz(
    amrex::MultiFab const& layout,
    amrex::Geometry const& geom,
    amrex::Real z_min_m,
    amrex::Real z_max_m,
    amrex::Real r_max_m,
    BoxBody&& box)
{
    auto const dx = geom.CellSizeArray();
    auto const plo = cell_index_origin(geom);
    for (amrex::MFIter mfi(layout, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto cell = box(mfi);
        amrex::LoopOnCpu(mfi.tilebox(), [&](int i, int j, int k) noexcept {
            amrex::Real const r =
                plo[0] + (static_cast<amrex::Real>(i) + amrex::Real(0.5)) * dx[0];
            amrex::Real const z =
                plo[1] + (static_cast<amrex::Real>(j) + amrex::Real(0.5)) * dx[1];
            if (z < z_min_m || z > z_max_m || r > r_max_m) {
                return;
            }
            cell(i, j, k, r, z, cell_volume_rz(geom, i, j));
        });
    }
}

// Field-fraction statistics of |Ez_realized| / |E0| (E0 the imposed peak
// magnitude) over the
// imposed-field (acceleration) region: how much of the driving field
// survives screening -- mean/min/p10/max and the volume fraction below one
// half (reduced-CSV field_fraction_* columns).
// p10 is the volume-weighted 10th percentile from a 512-bin linear
// histogram over the realized span, reported as the linear bin center.
struct AccelerationFieldMetrics {
    amrex::Real mean_fraction = amrex::Real(0.0);
    amrex::Real min_fraction = std::numeric_limits<amrex::Real>::infinity();
    amrex::Real p10_fraction = amrex::Real(0.0);
    amrex::Real max_fraction = amrex::Real(0.0);
    amrex::Real fraction_below_half = amrex::Real(0.0);
    amrex::Real volume_m3 = amrex::Real(0.0);
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
    for_each_region_cell_rz(ez, geom, z_min_m, z_max_m, r_max_m,
        [&](amrex::MFIter const& mfi) {
            auto const arr = ez.const_array(mfi);
            return [&, arr](int i, int j, int k, amrex::Real, amrex::Real,
                            amrex::Real volume) {
                amrex::Real const fraction = std::abs(arr(i, j, k)) / e0_abs;
                weighted_fraction += fraction * volume;
                metrics.volume_m3 += volume;
                metrics.min_fraction = amrex::min(metrics.min_fraction, fraction);
                metrics.max_fraction = amrex::max(metrics.max_fraction, fraction);
                if (fraction < amrex::Real(0.5)) {
                    below_half_volume += volume;
                }
            };
        });
    {
        amrex::Real sums[3] = {
            weighted_fraction, metrics.volume_m3, below_half_volume};
        amrex::ParallelDescriptor::ReduceRealSum(sums, 3);
        weighted_fraction = sums[0];
        metrics.volume_m3 = sums[1];
        below_half_volume = sums[2];
    }
    amrex::ParallelDescriptor::ReduceRealMin(metrics.min_fraction);
    amrex::ParallelDescriptor::ReduceRealMax(metrics.max_fraction);
    if (metrics.volume_m3 > amrex::Real(0.0)) {
        metrics.mean_fraction = weighted_fraction / metrics.volume_m3;
        metrics.fraction_below_half = below_half_volume / metrics.volume_m3;
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
            for_each_region_cell_rz(ez, geom, z_min_m, z_max_m, r_max_m,
                [&](amrex::MFIter const& mfi) {
                    auto const arr = ez.const_array(mfi);
                    return [&, arr](int i, int j, int k, amrex::Real, amrex::Real,
                                    amrex::Real volume) {
                        amrex::Real const fraction = std::abs(arr(i, j, k)) / e0_abs;
                        int const bin = static_cast<int>(
                            (fraction - metrics.min_fraction) / span
                            * static_cast<amrex::Real>(bins));
                        histogram[std::max(0, std::min(bins - 1, bin))] += volume;
                    };
                });
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

// Realized reduced-field E/N = |E|*1e21/(chi*N0) distribution over the
// acceleration region. The complete
// |E| = hypot(Er, Ez) and the Loschmidt-relative mobility density ratio, on
// fixed log-spaced bins (160 over 0.03-300 Td) so the percentiles are
// deterministic and reduce in one batched allreduce; percentiles report the
// geometric bin center clamped into the realized span.  Three weight
// channels answer three different questions -- where the VOLUME sits, where
// the existing LOW-ENERGY ELECTRONS sit, and where the next plasma is being
// CREATED -- which separate widely during a collapsing avalanche; no single
// weighting is "the" operating E/N.  The 3/30 Td markers bracket the
// closure's de Urquijo dry-air measurements; below 3 Td it uses the low-field
// anchors (en_td_accel_* reduced-CSV columns).
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
    for_each_region_cell_rz(ez, geom, z_min_m, z_max_m, r_max_m,
        [&](amrex::MFIter const& mfi) {
        auto const er_arr = er.const_array(mfi);
        auto const ez_arr = ez.const_array(mfi);
        auto const ne_arr = ne_low.const_array(mfi);
        auto const src_arr = s_low.const_array(mfi);
        auto const den_arr = density_ratio
            ? density_ratio->const_array(mfi)
            : amrex::Array4<amrex::Real const>{};
        return [&, er_arr, ez_arr, ne_arr, src_arr, den_arr](
                   int i, int j, int k, amrex::Real, amrex::Real, amrex::Real cell_volume) {
            double const chi = density_ratio
                ? static_cast<double>(den_arr(i, j, k))
                : static_cast<double>(scalar_density_ratio);
            if (!(chi > 0.0) || !std::isfinite(chi)) {
                return;
            }
            double const en_td = reduced_field_td(
                std::hypot(static_cast<double>(er_arr(i, j, k)),
                           static_cast<double>(ez_arr(i, j, k))),
                chi);
            if (!std::isfinite(en_td)) {
                return;
            }
            local_min = std::min(local_min, en_td);
            local_max = std::max(local_max, en_td);
            double const volume = static_cast<double>(cell_volume);
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
        };
    });
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

// Extent of the conduction current over the region: the sigma|Ez|-weighted
// rms radius and rms axial spread about the centroid, which size the
// conduction-current source independently of the energetic beam.  (The
// moment itself comes from the Maxwell face current, MaxwellJzMomentProfile.)
struct OhmicDischargeMetrics {
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
    // The sigma|Ez| weight and its r^2, z, z^2 moments.
    amrex::Real sums[4] = {};
    for_each_region_cell_rz(ez, geom, z_min_m, z_max_m, r_max_m,
        [&](amrex::MFIter const& mfi) {
            auto const ez_arr = ez.const_array(mfi);
            auto const sigma_arr = sigma.const_array(mfi);
            return [&, ez_arr, sigma_arr](int i, int j, int k, amrex::Real r,
                                          amrex::Real z, amrex::Real volume) {
                amrex::Real const w = amrex::max(sigma_arr(i, j, k), amrex::Real(0.0))
                    * std::abs(ez_arr(i, j, k)) * volume;
                sums[0] += w;
                sums[1] += w * r * r;
                sums[2] += w * z;
                sums[3] += w * z * z;
            };
        });
    amrex::ParallelDescriptor::ReduceRealSum(sums, 4);
    OhmicDischargeMetrics metrics;
    if (sums[0] > amrex::Real(0.0)) {
        metrics.fluid_current_rms_r_m = std::sqrt(sums[1] / sums[0]);
        amrex::Real const mean_z = sums[2] / sums[0];
        metrics.fluid_current_rms_z_m = std::sqrt(amrex::max(
            sums[3] / sums[0] - mean_z * mean_z, amrex::Real(0.0)));
    }
    return metrics;
}

}  // namespace rrea

#endif
