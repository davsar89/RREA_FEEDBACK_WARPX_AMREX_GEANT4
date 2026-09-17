#ifndef RREA_PLANE_FLUX_H
#define RREA_PLANE_FLUX_H

#include <AMReX_REAL.H>
#include "rrea/RreaKineticSupport.H"

#include <array>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace rrea {

enum class RreaPlaneFluxSelection : std::size_t {
    AllTracked = 0,
    LocalRunaway = 1,
    AtLeastOneMeV = 2,
    Count = 3,
};

struct RreaDirectionalPlaneFlux {
    amrex::Real upward = amrex::Real(0.0);
    amrex::Real downward = amrex::Real(0.0);

    [[nodiscard]] amrex::Real Net() const noexcept
    {
        return upward - downward;
    }
};

struct RreaDirectionalRadialMoments {
    amrex::Real upward_first_m = amrex::Real(0.0);
    amrex::Real downward_first_m = amrex::Real(0.0);
    amrex::Real upward_second_m2 = amrex::Real(0.0);
    amrex::Real downward_second_m2 = amrex::Real(0.0);
};

struct RreaPlaneFluxBin {
    std::array<RreaDirectionalPlaneFlux,
        static_cast<std::size_t>(RreaPlaneFluxSelection::Count)> selection{};
    // Minimal spatial moments for the primary local-runaway selection.  These
    // make the timestep gate compare the radial crossing distribution itself
    // rather than substituting an unrelated final-frame particle moment.
    RreaDirectionalRadialMoments local_runaway_radial{};
};

struct RreaPlaneFluxCrossing {
    std::size_t plane_index = 0;
    int direction = 0;
    amrex::Real weight = amrex::Real(0.0);
    amrex::Real radial_position_m = amrex::Real(0.0);
    bool local_runaway = false;
    bool at_least_one_mev = false;
};

inline constexpr std::size_t RreaPlaneFluxValuesPerPlane =
    static_cast<std::size_t>(RreaPlaneFluxSelection::Count) * 2U + 4U;

template<class Planes>
[[nodiscard]] RREA_HOST_DEVICE inline bool RreaPlaneLocationsAreStrictlyIncreasingFinite(
    Planes const& plane_z_m) noexcept
{
    for (std::size_t i = 0; i < plane_z_m.size(); ++i) {
        if (!std::isfinite(static_cast<double>(plane_z_m[i]))) {
            return false;
        }
        if (i > 0 && !(plane_z_m[i] > plane_z_m[i - 1])) {
            return false;
        }
    }
    return true;
}

// Enumerate every plane crossed by one non-periodic, straight transport
// segment.  The asymmetric half-open convention is deliberate:
//
//   upward   z0 <  plane <= z1
//   downward z1 <  plane <= z0
//
// Exact-plane ownership is on the plane's upper side: an upward arrival is
// counted, while a downward arrival is counted by the following departure.
// Thus an immediate reversal at the plane contributes one crossing of each
// sign and cancels, while continuing through an exact endpoint never counts
// the same directional crossing twice.
// The callback receives (plane index, plane z, segment fraction, direction),
// where direction is +1 for upward and -1 for downward.
template <class Planes, typename Callback>
[[nodiscard]] RREA_HOST_DEVICE inline bool RreaForEachNonperiodicPlaneCrossing(
    amrex::Real z0_m,
    amrex::Real z1_m,
    Planes const& plane_z_m,
    Callback&& callback)
{
    if (!std::isfinite(static_cast<double>(z0_m))
        || !std::isfinite(static_cast<double>(z1_m))
        || !RreaPlaneLocationsAreStrictlyIncreasingFinite(plane_z_m)) {
        return false;
    }
    if (z1_m == z0_m) {
        return true;
    }

    amrex::Real const inverse_span = amrex::Real(1.0) / (z1_m - z0_m);
    // The plane list is strictly increasing (checked above), so the half-open
    // rule (lo, hi] is exactly the index range [upper_bound(lo), upper_bound(hi)):
    // bracket it with two binary searches instead of scanning every plane.
    auto const begin = plane_z_m.begin();
    std::size_t const first = static_cast<std::size_t>(
        rrea::kinetic::upper_bound(begin, plane_z_m.end(), rrea::kinetic::min(z0_m, z1_m)) - begin);
    std::size_t const last = static_cast<std::size_t>(
        rrea::kinetic::upper_bound(begin, plane_z_m.end(), rrea::kinetic::max(z0_m, z1_m)) - begin);
    if (z1_m > z0_m) {
        for (std::size_t i = first; i < last; ++i) {
            callback(i, plane_z_m[i], (plane_z_m[i] - z0_m) * inverse_span, 1);
        }
    } else {
        for (std::size_t reverse = last; reverse > first; --reverse) {
            std::size_t const i = reverse - 1;
            callback(i, plane_z_m[i], (plane_z_m[i] - z0_m) * inverse_span, -1);
        }
    }
    return true;
}

class RreaPlaneFluxAccumulator {
public:
    struct PreparedUpdate {
    private:
        friend class RreaPlaneFluxAccumulator;
        std::vector<RreaPlaneFluxBin> bins;
    };

    explicit RreaPlaneFluxAccumulator(std::size_t plane_count = 0)
        : m_bins(plane_count)
    {
    }

    void Resize(std::size_t plane_count)
    {
        m_bins.assign(plane_count, RreaPlaneFluxBin{});
    }

    [[nodiscard]] std::size_t Size() const noexcept { return m_bins.size(); }

    [[nodiscard]] std::vector<RreaPlaneFluxBin> const& Bins() const noexcept
    {
        return m_bins;
    }

    [[nodiscard]] std::vector<amrex::Real> Flatten() const
    {
        std::vector<amrex::Real> values;
        values.reserve(m_bins.size() * RreaPlaneFluxValuesPerPlane);
        for (auto const& bin : m_bins) {
            for (auto const& directional : bin.selection) {
                values.push_back(directional.upward);
                values.push_back(directional.downward);
            }
            values.push_back(bin.local_runaway_radial.upward_first_m);
            values.push_back(bin.local_runaway_radial.downward_first_m);
            values.push_back(bin.local_runaway_radial.upward_second_m2);
            values.push_back(bin.local_runaway_radial.downward_second_m2);
        }
        return values;
    }

    [[nodiscard]] bool RestoreFlat(
        std::vector<amrex::Real> const& values) noexcept
    {
        if (values.size() != m_bins.size() * RreaPlaneFluxValuesPerPlane) {
            return false;
        }
        for (amrex::Real const value : values) {
            if (!std::isfinite(static_cast<double>(value))
                || value < amrex::Real(0.0)) {
                return false;
            }
        }
        std::size_t cursor = 0;
        for (auto& bin : m_bins) {
            for (auto& directional : bin.selection) {
                amrex::Real const upward = values[cursor++];
                amrex::Real const downward = values[cursor++];
                directional.upward = upward;
                directional.downward = downward;
            }
            bin.local_runaway_radial.upward_first_m = values[cursor++];
            bin.local_runaway_radial.downward_first_m = values[cursor++];
            bin.local_runaway_radial.upward_second_m2 = values[cursor++];
            bin.local_runaway_radial.downward_second_m2 = values[cursor++];
        }
        return true;
    }

    [[nodiscard]] bool AddCrossing(
        std::size_t plane_index,
        int direction,
        amrex::Real weight,
        amrex::Real radial_position_m,
        bool local_runaway,
        bool at_least_one_mev) noexcept
    {
        if (plane_index >= m_bins.size()
            || (direction != 1 && direction != -1)
            || !std::isfinite(static_cast<double>(weight))
            || !(weight >= amrex::Real(0.0))
            || !std::isfinite(static_cast<double>(radial_position_m))
            || !(radial_position_m >= amrex::Real(0.0))) {
            return false;
        }
        std::array<bool,
            static_cast<std::size_t>(RreaPlaneFluxSelection::Count)> const selected{
            true, local_runaway, at_least_one_mev};
        std::array<amrex::Real,
            static_cast<std::size_t>(RreaPlaneFluxSelection::Count)> updated{};
        auto const& bin = m_bins[plane_index];
        for (std::size_t selection = 0; selection < selected.size(); ++selection) {
            if (!selected[selection]) {
                continue;
            }
            auto const& directional = bin.selection[selection];
            amrex::Real const old_value = direction > 0
                ? directional.upward
                : directional.downward;
            updated[selection] = old_value + weight;
            if (!std::isfinite(static_cast<double>(updated[selection]))) {
                return false;
            }
        }
        std::array<amrex::Real, 2> radial_updates{};
        if (local_runaway) {
            auto const& radial = bin.local_runaway_radial;
            amrex::Real const old_first = direction > 0
                ? radial.upward_first_m
                : radial.downward_first_m;
            amrex::Real const old_second = direction > 0
                ? radial.upward_second_m2
                : radial.downward_second_m2;
            radial_updates[0] = old_first + weight * radial_position_m;
            radial_updates[1] = old_second
                + weight * radial_position_m * radial_position_m;
            if (!std::isfinite(static_cast<double>(radial_updates[0]))
                || !std::isfinite(static_cast<double>(radial_updates[1]))) {
                return false;
            }
        }
        auto& mutable_bin = m_bins[plane_index];
        for (std::size_t selection = 0; selection < selected.size(); ++selection) {
            if (!selected[selection]) {
                continue;
            }
            auto& directional = mutable_bin.selection[selection];
            if (direction > 0) {
                directional.upward = updated[selection];
            } else {
                directional.downward = updated[selection];
            }
        }
        if (local_runaway) {
            auto& radial = mutable_bin.local_runaway_radial;
            if (direction > 0) {
                radial.upward_first_m = radial_updates[0];
                radial.upward_second_m2 = radial_updates[1];
            } else {
                radial.downward_first_m = radial_updates[0];
                radial.downward_second_m2 = radial_updates[1];
            }
        }
        return true;
    }

    // Build the complete post-transaction state without mutating the live
    // cumulative counters. This detects invalid payloads and cumulative
    // floating-point overflow across the whole deferred crossing batch.
    [[nodiscard]] bool PrepareCrossings(
        std::vector<RreaPlaneFluxCrossing> const& crossings,
        PreparedUpdate& prepared) const
    {
        RreaPlaneFluxAccumulator candidate(*this);
        for (auto const& crossing : crossings) {
            if (!candidate.AddCrossing(
                    crossing.plane_index,
                    crossing.direction,
                    crossing.weight,
                    crossing.radial_position_m,
                    crossing.local_runaway,
                    crossing.at_least_one_mev)) {
                return false;
            }
        }
        prepared.bins = std::move(candidate.m_bins);
        return true;
    }

    // The fallible work is complete in PrepareCrossings. Swapping a prepared
    // std::vector with the same allocator is noexcept, so transaction replay
    // cannot fail after particle/source/ledger mutation has begun.
    void CommitPrepared(PreparedUpdate&& prepared) noexcept
    {
        m_bins.swap(prepared.bins);
    }

private:
    std::vector<RreaPlaneFluxBin> m_bins;
};

// Rank-local cumulative state for the compact C&D diagnostic.  MPI reduction
// is intentionally performed only by the output layer: keeping local counters
// here makes checkpoint/restart exact with the existing same-rank contract and
// avoids one collective per particle crossing.
class RreaPlaneFluxState {
public:
    void Configure(
        std::vector<amrex::Real> plane_z_m,
        amrex::Real output_interval_s)
    {
        if (plane_z_m.empty()
            || !RreaPlaneLocationsAreStrictlyIncreasingFinite(plane_z_m)
            || !std::isfinite(static_cast<double>(output_interval_s))
            || !(output_interval_s > amrex::Real(0.0))) {
            throw std::invalid_argument(
                "RREA C&D plane diagnostic requires finite increasing planes "
                "and a positive finite output interval");
        }
        m_plane_z_m = std::move(plane_z_m);
        m_accumulator.Resize(m_plane_z_m.size());
        m_output_interval_s = output_interval_s;
        m_next_output_time_s = output_interval_s;
        m_write_count = 0;
        m_last_write_step = -1;
        m_last_write_time_s = -std::numeric_limits<amrex::Real>::infinity();
    }

    [[nodiscard]] bool Enabled() const noexcept { return !m_plane_z_m.empty(); }
    [[nodiscard]] std::vector<amrex::Real> const& PlaneZ() const noexcept
    {
        return m_plane_z_m;
    }
    [[nodiscard]] RreaPlaneFluxAccumulator const& Accumulator() const noexcept
    {
        return m_accumulator;
    }
    [[nodiscard]] RreaPlaneFluxAccumulator& Accumulator() noexcept
    {
        return m_accumulator;
    }
    [[nodiscard]] std::uint64_t WriteCount() const noexcept
    {
        return m_write_count;
    }
    [[nodiscard]] amrex::Real NextOutputTimeS() const noexcept
    {
        return m_next_output_time_s;
    }
    [[nodiscard]] amrex::Real OutputIntervalS() const noexcept
    {
        return m_output_interval_s;
    }
    [[nodiscard]] int LastWriteStep() const noexcept { return m_last_write_step; }
    [[nodiscard]] amrex::Real LastWriteTimeS() const noexcept
    {
        return m_last_write_time_s;
    }

    [[nodiscard]] bool Due(int step, amrex::Real time_s, bool force) const noexcept
    {
        if (!Enabled() || step < 0
            || !std::isfinite(static_cast<double>(time_s))) {
            return false;
        }
        if (step == m_last_write_step && time_s == m_last_write_time_s) {
            return false;
        }
        amrex::Real const scale = std::max(
            {amrex::Real(1.0), std::abs(time_s), std::abs(m_next_output_time_s)});
        amrex::Real const tolerance = amrex::Real(32.0)
            * std::numeric_limits<amrex::Real>::epsilon() * scale;
        return force || time_s + tolerance >= m_next_output_time_s;
    }

    void MarkWritten(int step, amrex::Real time_s)
    {
        if (!Due(step, time_s, true)) {
            throw std::invalid_argument("invalid or duplicate RREA C&D plane write");
        }
        ++m_write_count;
        m_last_write_step = step;
        m_last_write_time_s = time_s;
        amrex::Real const scale = std::max(
            {amrex::Real(1.0), std::abs(time_s), std::abs(m_next_output_time_s)});
        amrex::Real const tolerance = amrex::Real(32.0)
            * std::numeric_limits<amrex::Real>::epsilon() * scale;
        // A forced checkpoint before the next cadence boundary must not move the
        // schedule.  A scheduled write that was accepted within Due's roundoff
        // tolerance must advance past that same tolerance, otherwise a time one
        // ulp below the boundary can cause a duplicate write on the next step.
        if (time_s + tolerance >= m_next_output_time_s) {
            long double const interval = static_cast<long double>(m_output_interval_s);
            long double const next = static_cast<long double>(m_next_output_time_s);
            long double const accepted_time = static_cast<long double>(time_s + tolerance);
            long double const intervals =
                std::floor((accepted_time - next) / interval) + 1.0L;
            m_next_output_time_s = static_cast<amrex::Real>(
                next + std::max(1.0L, intervals) * interval);
        }
    }

    void RestoreSchedule(
        std::uint64_t write_count,
        int last_write_step,
        amrex::Real last_write_time_s,
        amrex::Real next_output_time_s)
    {
        if (!Enabled() || last_write_step < -1
            || !std::isfinite(static_cast<double>(next_output_time_s))
            || !(next_output_time_s > amrex::Real(0.0))
            || (last_write_step >= 0
                && !std::isfinite(static_cast<double>(last_write_time_s)))) {
            throw std::invalid_argument("invalid checkpointed RREA C&D plane schedule");
        }
        m_write_count = write_count;
        m_last_write_step = last_write_step;
        m_last_write_time_s = last_write_step >= 0
            ? last_write_time_s
            : -std::numeric_limits<amrex::Real>::infinity();
        m_next_output_time_s = next_output_time_s;
    }

private:
    std::vector<amrex::Real> m_plane_z_m;
    RreaPlaneFluxAccumulator m_accumulator;
    amrex::Real m_output_interval_s = amrex::Real(0.0);
    amrex::Real m_next_output_time_s = amrex::Real(0.0);
    std::uint64_t m_write_count = 0;
    int m_last_write_step = -1;
    amrex::Real m_last_write_time_s =
        -std::numeric_limits<amrex::Real>::infinity();
};

}  // namespace rrea

#endif
