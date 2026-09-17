#ifndef RREA_CIC_CONSERVING_RESAMPLE_H
#define RREA_CIC_CONSERVING_RESAMPLE_H

// Charge-exact stochastic thinning for order-1 cell-centered CIC particles.
//
// Every particle handed to one call has the same four-cell support.  Its
// deposited cell charges are its weight times
//
//   ((1-fr)(1-fz), fr(1-fz), (1-fr)fz, fr fz).
//
// For at most five live particles at a time, the routine finds a direction d
// in the null space of those four CIC columns.  It then moves the weights to
// one of the two non-negative endpoints w + alpha*d.  The endpoint
// probabilities make E[alpha]=0.  Consequently every realization preserves
// all four deposited cell charges (to the round-off of the production CIC
// arithmetic), while E[w'_i]=w_i for every original particle.  At least one
// weight reaches zero at each step, so this is genuine thinning rather than a
// weight shuffle.
//
// This header deliberately has no AMReX/WarpX dependency.  The production
// coupling owns grouping by common support and supplies its counter-RNG draw;
// the native smoke exercises this exact primitive.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace rrea {

struct RreaCicResampleParticle {
    double fraction_r = 0.0;
    double fraction_z = 0.0;
    double weight = 0.0;
};

struct RreaCicResampleResult {
    std::size_t removed = 0;
    std::size_t draws = 0;
    bool valid = true;
};

namespace cic_resample_detail {

using CicColumn = std::array<long double, 4>;

inline CicColumn column(RreaCicResampleParticle const& particle) noexcept
{
    long double const fr = static_cast<long double>(particle.fraction_r);
    long double const fz = static_cast<long double>(particle.fraction_z);
    return CicColumn{
        (1.0L - fr) * (1.0L - fz),
        fr * (1.0L - fz),
        (1.0L - fr) * fz,
        fr * fz};
}

// Reduced-row-echelon elimination on a 4 x n matrix (n <= 5).  When the
// columns are dependent, direction receives one non-zero null vector.
inline bool null_direction(
    std::array<CicColumn, 5> const& columns,
    std::size_t n,
    std::array<long double, 5>& direction) noexcept
{
    std::array<std::array<long double, 5>, 4> matrix{};
    for (std::size_t c = 0; c < n; ++c) {
        for (std::size_t r = 0; r < 4; ++r) {
            matrix[r][c] = columns[c][r];
        }
    }

    std::array<int, 4> pivot_column{{-1, -1, -1, -1}};
    std::array<bool, 5> is_pivot{{false, false, false, false, false}};
    std::size_t rank = 0;
    long double const tolerance =
        512.0L * std::numeric_limits<long double>::epsilon();
    for (std::size_t c = 0; c < n && rank < 4; ++c) {
        std::size_t pivot = rank;
        long double pivot_abs = 0.0L;
        for (std::size_t r = rank; r < 4; ++r) {
            long double const candidate = std::abs(matrix[r][c]);
            if (candidate > pivot_abs) {
                pivot = r;
                pivot_abs = candidate;
            }
        }
        if (!(pivot_abs > tolerance)) {
            continue;
        }
        if (pivot != rank) {
            std::swap(matrix[pivot], matrix[rank]);
        }
        long double const scale = matrix[rank][c];
        for (std::size_t j = 0; j < n; ++j) {
            matrix[rank][j] /= scale;
        }
        for (std::size_t r = 0; r < 4; ++r) {
            if (r == rank) {
                continue;
            }
            long double const factor = matrix[r][c];
            if (factor == 0.0L) {
                continue;
            }
            for (std::size_t j = 0; j < n; ++j) {
                matrix[r][j] -= factor * matrix[rank][j];
            }
        }
        pivot_column[rank] = static_cast<int>(c);
        is_pivot[c] = true;
        ++rank;
    }
    if (rank == n) {
        return false;
    }

    std::size_t free_column = n;
    for (std::size_t c = n; c-- > 0;) {
        if (!is_pivot[c]) {
            free_column = c;
            break;
        }
    }
    if (free_column == n) {
        return false;
    }

    direction.fill(0.0L);
    direction[free_column] = 1.0L;
    for (std::size_t rr = rank; rr-- > 0;) {
        std::size_t const pivot =
            static_cast<std::size_t>(pivot_column[rr]);
        long double value = 0.0L;
        for (std::size_t c = 0; c < n; ++c) {
            if (c != pivot) {
                value += matrix[rr][c] * direction[c];
            }
        }
        direction[pivot] = -value;
    }

    long double max_abs = 0.0L;
    for (std::size_t c = 0; c < n; ++c) {
        max_abs = std::max(max_abs, std::abs(direction[c]));
    }
    if (!(max_abs > 0.0L) || !std::isfinite(max_abs)) {
        return false;
    }
    bool positive = false;
    bool negative = false;
    for (std::size_t c = 0; c < n; ++c) {
        direction[c] /= max_abs;
        positive = positive || direction[c] > tolerance;
        negative = negative || direction[c] < -tolerance;
    }
    // The CIC columns each sum to one, so a true null direction necessarily
    // has both signs.  Refuse a numerically spurious direction otherwise.
    return positive && negative;
}

inline std::size_t constraint_rank(
    std::vector<RreaCicResampleParticle> const& particles) noexcept
{
    std::array<std::array<long double, 4>, 4> basis{};
    std::size_t rank = 0;
    long double const tolerance =
        512.0L * std::numeric_limits<long double>::epsilon();
    for (auto const& particle : particles) {
        CicColumn value = column(particle);
        for (std::size_t r = 0; r < rank; ++r) {
            std::size_t pivot = 0;
            while (pivot < 4 && std::abs(basis[r][pivot]) <= tolerance) {
                ++pivot;
            }
            if (pivot == 4) {
                continue;
            }
            long double const factor = value[pivot] / basis[r][pivot];
            for (std::size_t j = pivot; j < 4; ++j) {
                value[j] -= factor * basis[r][j];
            }
        }
        std::size_t pivot = 0;
        while (pivot < 4 && std::abs(value[pivot]) <= tolerance) {
            ++pivot;
        }
        if (pivot == 4) {
            continue;
        }
        long double const pivot_value = value[pivot];
        for (std::size_t j = pivot; j < 4; ++j) {
            value[j] /= pivot_value;
        }
        basis[rank] = value;
        ++rank;
        if (rank == 4) {
            break;
        }
    }
    return rank;
}

}  // namespace cic_resample_detail

inline std::size_t RreaCicResampleRemovalCapacity(
    std::vector<RreaCicResampleParticle> const& particles) noexcept
{
    std::size_t live = 0;
    std::vector<RreaCicResampleParticle> positive;
    positive.reserve(particles.size());
    for (auto const& particle : particles) {
        if (particle.weight > 0.0) {
            ++live;
            positive.push_back(particle);
        }
    }
    std::size_t const rank =
        cic_resample_detail::constraint_rank(positive);
    return live > rank ? live - rank : 0;
}

// draw(draw_index) must return a deterministic uniform in (0,1).  Calls are
// made only for successful null-space steps and draw_index begins at zero.
template <typename UniformDraw>
RreaCicResampleResult RreaThinCicWeights(
    std::vector<RreaCicResampleParticle>& particles,
    std::size_t maximum_removals,
    UniformDraw&& draw) noexcept
{
    RreaCicResampleResult result;
    std::vector<std::size_t> active;
    active.reserve(particles.size());
    for (std::size_t i = 0; i < particles.size(); ++i) {
        auto const& particle = particles[i];
        if (!std::isfinite(particle.fraction_r)
            || !std::isfinite(particle.fraction_z)
            || particle.fraction_r < 0.0 || particle.fraction_r > 1.0
            || particle.fraction_z < 0.0 || particle.fraction_z > 1.0
            || !std::isfinite(particle.weight) || particle.weight < 0.0) {
            result.valid = false;
            return result;
        }
        if (particle.weight > 0.0) {
            active.push_back(i);
        }
    }

    while (result.removed < maximum_removals && active.size() >= 2) {
        std::size_t const n = std::min<std::size_t>(5, active.size());
        std::array<cic_resample_detail::CicColumn, 5> columns{};
        for (std::size_t c = 0; c < n; ++c) {
            columns[c] = cic_resample_detail::column(particles[active[c]]);
        }
        std::array<long double, 5> direction{};
        if (!cic_resample_detail::null_direction(columns, n, direction)) {
            if (active.size() > n) {
                // Any five columns in four dimensions are dependent.  Treat
                // failure to recover their mixed-sign null vector as a
                // numerical error rather than rotating forever.
                result.valid = false;
                return result;
            }
            break;
        }

        long double alpha_plus = std::numeric_limits<long double>::infinity();
        long double alpha_minus = std::numeric_limits<long double>::infinity();
        std::size_t hit_plus = n;
        std::size_t hit_minus = n;
        for (std::size_t c = 0; c < n; ++c) {
            long double const weight =
                static_cast<long double>(particles[active[c]].weight);
            if (direction[c] < 0.0L) {
                long double const bound = weight / -direction[c];
                if (bound < alpha_plus) {
                    alpha_plus = bound;
                    hit_plus = c;
                }
            } else if (direction[c] > 0.0L) {
                long double const bound = weight / direction[c];
                if (bound < alpha_minus) {
                    alpha_minus = bound;
                    hit_minus = c;
                }
            }
        }
        if (!std::isfinite(alpha_plus) || !std::isfinite(alpha_minus)
            || !(alpha_plus >= 0.0L) || !(alpha_minus >= 0.0L)
            || hit_plus == n || hit_minus == n) {
            result.valid = false;
            return result;
        }

        double const uniform = static_cast<double>(draw(result.draws));
        if (!std::isfinite(uniform) || !(uniform > 0.0) || !(uniform < 1.0)) {
            result.valid = false;
            return result;
        }
        ++result.draws;
        long double const plus_probability =
            alpha_minus / (alpha_plus + alpha_minus);
        bool const take_plus =
            static_cast<long double>(uniform) < plus_probability;
        long double const alpha = take_plus ? alpha_plus : -alpha_minus;
        std::size_t const guaranteed_zero = take_plus ? hit_plus : hit_minus;

        for (std::size_t c = 0; c < n; ++c) {
            long double updated =
                static_cast<long double>(particles[active[c]].weight)
                + alpha * direction[c];
            // Only the endpoint that defined alpha is analytically zero.
            // Clamping other small positive weights would be an extra vector
            // outside the CIC null space and would accumulate charge error.
            if (c == guaranteed_zero) {
                updated = 0.0L;
            }
            // A roundoff-negative weight is the analytic zero seen through
            // long-double arithmetic: clamp it inside the same slack the stored
            // weight verifier below already accepts (512 ulp x support size,
            // scaled by the weight), and kill the plan only for larger negatives.
            if (updated < 0.0L
                && -updated <= 512.0L * std::numeric_limits<double>::epsilon()
                        * static_cast<long double>(n)
                        * std::max(1.0L, std::abs(static_cast<long double>(
                              particles[active[c]].weight)))) {
                updated = 0.0L;
            }
            if (updated < 0.0L || !std::isfinite(updated)) {
                result.valid = false;
                return result;
            }
            particles[active[c]].weight =
                updated > 0.0L ? static_cast<double>(updated) : 0.0;
        }

        std::size_t before = active.size();
        active.erase(
            std::remove_if(
                active.begin(), active.end(),
                [&](std::size_t i) { return particles[i].weight == 0.0; }),
            active.end());
        std::size_t const removed_now = before - active.size();
        if (removed_now == 0) {
            result.valid = false;
            return result;
        }
        result.removed += removed_now;
    }
    return result;
}

// Global support plan for survivors and still-staged candidates.  The caller
// allgathers these compact records, calls this pure function on every rank,
// preflights all species/physical ledgers, and only then applies the returned
// stored weights.  source_rank and candidate route an update back to live or
// staged storage; neither enters the physics, so the plan is decomposition
// invariant.  Event/source/current ledgers retain the original physical
// weight owned by their transaction and are intentionally absent here.
struct RreaGlobalCicParticle {
    int support_r = 0;
    int support_z = 0;
    int phase = 0;
    int source_rank = 0;
    std::uint64_t stable_id = 0;
    double fraction_r = 0.0;
    double fraction_z = 0.0;
    double weight = 0.0;
    bool candidate = false;
    std::uint64_t stable_id_high = 0;
};

struct RreaGlobalCicUpdate {
    int source_rank = 0;
    std::uint64_t stable_id = 0;
    double stored_weight = 0.0;
    bool candidate = false;
    std::uint64_t stable_id_high = 0;
};

struct RreaGlobalCicSupportPlan {
    int support_r = 0;
    int support_z = 0;
    int owner_rank = 0;
    std::size_t capacity = 0;
    std::size_t geometric_floor = 0;
};

enum class RreaGlobalCicPlanStatus {
    Ready,
    InvalidInput,
    DuplicateStableId,
    InvalidStoredMoment,
    InvalidReduction
};

struct RreaGlobalCicPlan {
    RreaGlobalCicPlanStatus status = RreaGlobalCicPlanStatus::Ready;
    std::size_t capacity = 0;
    std::size_t geometric_floor = 0;
    std::size_t requested = 0;
    std::size_t removed = 0;
    std::vector<RreaGlobalCicSupportPlan> supports;
    std::vector<RreaGlobalCicUpdate> updates;

    [[nodiscard]] bool valid() const noexcept
    {
        return status == RreaGlobalCicPlanStatus::Ready;
    }
};

inline std::uint64_t RreaCicMix64(std::uint64_t value) noexcept
{
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

inline std::uint64_t RreaGlobalCicSupportKey(
    int support_r, int support_z, std::uint64_t stream_key = 0) noexcept
{
    std::uint64_t const cells =
        (static_cast<std::uint64_t>(static_cast<std::uint32_t>(support_r)) << 32)
        | static_cast<std::uint32_t>(support_z);
    return RreaCicMix64(cells ^ RreaCicMix64(stream_key));
}

inline int RreaGlobalCicSupportOwner(
    int support_r, int support_z, int ranks,
    std::uint64_t stream_key = 0) noexcept
{
    return ranks > 0
        ? static_cast<int>(RreaGlobalCicSupportKey(
              support_r, support_z, stream_key) % static_cast<std::uint64_t>(ranks))
        : -1;
}

template <typename StoredReal>
inline bool RreaRoundAndVerifyCicWeights(
    std::vector<RreaCicResampleParticle> const& before,
    std::vector<RreaCicResampleParticle> const& planned,
    std::vector<StoredReal>& stored) noexcept
{
    if (before.size() != planned.size()) {
        return false;
    }
    auto moments = [](auto const& particles) {
        std::array<long double, 4> cells{};
        for (auto const& particle : particles) {
            double const fr = particle.fraction_r;
            double const fz = particle.fraction_z;
            double const w = particle.weight;
            cells[0] += w * (1.0 - fr) * (1.0 - fz);
            cells[1] += w * fr * (1.0 - fz);
            cells[2] += w * (1.0 - fr) * fz;
            cells[3] += w * fr * fz;
        }
        return cells;
    };
    auto const before_cells = moments(before);
    std::vector<RreaCicResampleParticle> rounded = planned;
    stored.resize(planned.size());
    for (std::size_t i = 0; i < planned.size(); ++i) {
        stored[i] = static_cast<StoredReal>(planned[i].weight);
        rounded[i].weight = static_cast<double>(stored[i]);
    }
    auto const after_cells = moments(rounded);
    for (std::size_t cell = 0; cell < 4; ++cell) {
        long double const scale = std::max(1.0L, std::abs(before_cells[cell]));
        long double const tolerance = 512.0L
            * std::numeric_limits<double>::epsilon()
            * static_cast<long double>(before.size()) * scale;
        if (std::abs(after_cells[cell] - before_cells[cell]) > tolerance) {
            stored.clear();
            return false;
        }
    }
    return true;
}

// One deterministic plan over globally gathered survivors + candidates.
// draw(group_id, reduction_index) returns a uniform in (0,1).  All mutations
// stay in local temporaries until every support has passed StoredReal rounding
// and its four deposited moments have been rechecked.
template <typename StoredReal, typename UniformDraw>
RreaGlobalCicPlan RreaPlanGlobalCicThinning(
    std::vector<RreaGlobalCicParticle> const& input,
    std::size_t target_count,
    int ranks,
    std::uint64_t stream_key,
    UniformDraw&& draw) noexcept
{
    RreaGlobalCicPlan plan;
    if (ranks <= 0) {
        plan.status = RreaGlobalCicPlanStatus::InvalidInput;
        return plan;
    }
    auto particles = input;
    for (auto const& particle : particles) {
        if (particle.source_rank < 0 || particle.source_rank >= ranks
            || !std::isfinite(particle.fraction_r)
            || !std::isfinite(particle.fraction_z)
            || particle.fraction_r < 0.0 || particle.fraction_r > 1.0
            || particle.fraction_z < 0.0 || particle.fraction_z > 1.0
            || !std::isfinite(particle.weight) || particle.weight <= 0.0) {
            plan.status = RreaGlobalCicPlanStatus::InvalidInput;
            return plan;
        }
    }
    std::vector<std::array<std::uint64_t, 3>> ids;
    ids.reserve(particles.size());
    for (auto const& particle : particles) {
        ids.push_back({static_cast<std::uint64_t>(particle.candidate),
                       particle.stable_id_high, particle.stable_id});
    }
    std::sort(ids.begin(), ids.end());
    if (std::adjacent_find(ids.begin(), ids.end()) != ids.end()) {
        plan.status = RreaGlobalCicPlanStatus::DuplicateStableId;
        return plan;
    }
    std::sort(
        particles.begin(), particles.end(),
        [](auto const& lhs, auto const& rhs) {
            if (lhs.support_r != rhs.support_r) return lhs.support_r < rhs.support_r;
            if (lhs.support_z != rhs.support_z) return lhs.support_z < rhs.support_z;
            if (lhs.phase != rhs.phase) return lhs.phase < rhs.phase;
            if (lhs.candidate != rhs.candidate) return lhs.candidate < rhs.candidate;
            if (lhs.stable_id_high != rhs.stable_id_high) {
                return lhs.stable_id_high < rhs.stable_id_high;
            }
            return lhs.stable_id < rhs.stable_id;
        });
    struct Group {
        std::size_t begin = 0;
        std::size_t end = 0;
        std::size_t capacity = 0;
        std::uint64_t id = 0;
    };
    std::vector<Group> groups;
    for (std::size_t begin = 0; begin < particles.size();) {
        std::size_t end = begin + 1;
        while (end < particles.size()
               && particles[end].support_r == particles[begin].support_r
               && particles[end].support_z == particles[begin].support_z) {
            ++end;
        }
        std::vector<RreaCicResampleParticle> group;
        group.reserve(end - begin);
        for (std::size_t i = begin; i < end; ++i) {
            group.push_back({particles[i].fraction_r, particles[i].fraction_z,
                             particles[i].weight});
        }
        std::size_t const capacity = RreaCicResampleRemovalCapacity(group);
        std::size_t const floor = end - begin - capacity;
        std::uint64_t const support_key = RreaGlobalCicSupportKey(
            particles[begin].support_r, particles[begin].support_z, stream_key);
        groups.push_back(Group{
            begin, end, capacity,
            RreaCicMix64(support_key
                ^ RreaCicMix64(particles[begin].stable_id_high)
                ^ RreaCicMix64(particles[begin].stable_id)
                ^ static_cast<std::uint64_t>(particles[begin].candidate))});
        plan.capacity += capacity;
        plan.geometric_floor += floor;
        plan.supports.push_back(RreaGlobalCicSupportPlan{
            particles[begin].support_r,
            particles[begin].support_z,
            RreaGlobalCicSupportOwner(
                particles[begin].support_r, particles[begin].support_z,
                ranks, stream_key),
            capacity, floor});
        begin = end;
    }
    plan.requested = std::min(
        input.size() > target_count ? input.size() - target_count : 0U,
        plan.capacity);
    if (plan.requested == 0) {
        return plan;
    }

    std::vector<double> staged_weights(particles.size());
    for (std::size_t i = 0; i < particles.size(); ++i) {
        staged_weights[i] = particles[i].weight;
    }
    for (auto const& descriptor : groups) {
        if (plan.removed >= plan.requested) break;
        std::vector<RreaCicResampleParticle> before;
        before.reserve(descriptor.end - descriptor.begin);
        for (std::size_t i = descriptor.begin; i < descriptor.end; ++i) {
            before.push_back({particles[i].fraction_r, particles[i].fraction_z,
                              particles[i].weight});
        }
        auto planned = before;
        std::size_t const remaining = plan.requested - plan.removed;
        std::size_t const group_request = std::min(descriptor.capacity, remaining);
        if (group_request == 0) continue;
        auto const result = RreaThinCicWeights(
            planned, group_request,
            [&](std::size_t reduction) { return draw(descriptor.id, reduction); });
        if (!result.valid || result.removed == 0) {
            plan.status = RreaGlobalCicPlanStatus::InvalidReduction;
            plan.updates.clear();
            return plan;
        }
        std::vector<StoredReal> stored;
        if (!RreaRoundAndVerifyCicWeights(before, planned, stored)) {
            plan.status = RreaGlobalCicPlanStatus::InvalidStoredMoment;
            plan.updates.clear();
            return plan;
        }
        for (std::size_t i = 0; i < stored.size(); ++i) {
            staged_weights[descriptor.begin + i] = static_cast<double>(stored[i]);
        }
        plan.removed += result.removed;
    }
    plan.updates.reserve(particles.size());
    for (std::size_t i = 0; i < particles.size(); ++i) {
        plan.updates.push_back(RreaGlobalCicUpdate{
            particles[i].source_rank, particles[i].stable_id,
            staged_weights[i], particles[i].candidate,
            particles[i].stable_id_high});
    }
    return plan;
}

}  // namespace rrea

#endif
