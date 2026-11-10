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

#include "rrea/RreaKineticSupport.H"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace rrea {
// CUDA has native binary64 but no extended-precision long double. Name the
// working type explicitly so rank tolerances use the ACTUAL arithmetic, not
// the host long-double epsilon accidentally folded into a device constant.
// Stored weights and the final deposited-moment guard remain binary64.
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__) || defined(RREA_CIC_TEST_BINARY64) || defined(RREA_KINETIC_TEST_BINARY64)
using RreaCicWide=double;
#else
using RreaCicWide=long double;
#endif


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

template<typename Wide> using CicColumn = std::array<Wide, 4>;

template<typename Wide,class Particle>
RREA_HOST_DEVICE inline CicColumn<Wide> column(Particle const& particle) noexcept
{
    Wide const fr = static_cast<Wide>(particle.fraction_r);
    Wide const fz = static_cast<Wide>(particle.fraction_z);
    return CicColumn<Wide>{
        (Wide(1.0) - fr) * (Wide(1.0) - fz),
        fr * (Wide(1.0) - fz),
        (Wide(1.0) - fr) * fz,
        fr * fz};
}

// Reduced-row-echelon elimination on a 4 x n matrix (n <= 5).  When the
// columns are dependent, direction receives one non-zero null vector.
template<typename Wide>
RREA_HOST_DEVICE inline bool null_direction(
    std::array<CicColumn<Wide>, 5> const& columns,
    std::size_t n,
    std::array<Wide, 5>& direction) noexcept
{
    std::array<std::array<Wide, 5>, 4> matrix{};
    for (std::size_t c = 0; c < n; ++c) {
        for (std::size_t r = 0; r < 4; ++r) {
            matrix[r][c] = columns[c][r];
        }
    }

    std::array<int, 4> pivot_column{{-1, -1, -1, -1}};
    std::array<bool, 5> is_pivot{{false, false, false, false, false}};
    std::size_t rank = 0;
    Wide const tolerance =
        Wide(512.0) * std::numeric_limits<Wide>::epsilon();
    for (std::size_t c = 0; c < n && rank < 4; ++c) {
        std::size_t pivot = rank;
        Wide pivot_abs = Wide(0.0);
        for (std::size_t r = rank; r < 4; ++r) {
            Wide const candidate = std::abs(matrix[r][c]);
            if (candidate > pivot_abs) {
                pivot = r;
                pivot_abs = candidate;
            }
        }
        if (!(pivot_abs > tolerance)) {
            continue;
        }
        if (pivot != rank) {
            rrea::kinetic::swap(matrix[pivot], matrix[rank]);
        }
        Wide const scale = matrix[rank][c];
        for (std::size_t j = 0; j < n; ++j) {
            matrix[rank][j] /= scale;
        }
        for (std::size_t r = 0; r < 4; ++r) {
            if (r == rank) {
                continue;
            }
            Wide const factor = matrix[r][c];
            if (factor == Wide(0.0)) {
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

    direction.fill(Wide(0.0));
    direction[free_column] = Wide(1.0);
    for (std::size_t rr = rank; rr-- > 0;) {
        std::size_t const pivot =
            static_cast<std::size_t>(pivot_column[rr]);
        Wide value = Wide(0.0);
        for (std::size_t c = 0; c < n; ++c) {
            if (c != pivot) {
                value += matrix[rr][c] * direction[c];
            }
        }
        direction[pivot] = -value;
    }

    Wide max_abs = Wide(0.0);
    for (std::size_t c = 0; c < n; ++c) {
        max_abs = rrea::kinetic::max(max_abs, std::abs(direction[c]));
    }
    if (!(max_abs > Wide(0.0)) || !std::isfinite(max_abs)) {
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

template<typename Wide,class Particles>
RREA_HOST_DEVICE inline std::size_t constraint_rank(
    Particles const& particles) noexcept
{
    std::array<std::array<Wide, 4>, 4> basis{};
    std::size_t rank = 0;
    Wide const tolerance =
        Wide(512.0) * std::numeric_limits<Wide>::epsilon();
    for (auto const& particle : particles) {
        if(!(particle.weight>0.0))continue;
        CicColumn<Wide> value = column<Wide>(particle);
        for (std::size_t r = 0; r < rank; ++r) {
            std::size_t pivot = 0;
            while (pivot < 4 && std::abs(basis[r][pivot]) <= tolerance) {
                ++pivot;
            }
            if (pivot == 4) {
                continue;
            }
            Wide const factor = value[pivot] / basis[r][pivot];
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
        Wide const pivot_value = value[pivot];
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

template<class Particle>
RREA_HOST_DEVICE inline std::size_t RreaCicResampleRemovalCapacity(rrea::KineticSpan<Particle> particles) noexcept
{
    std::size_t live=0;
    for(auto const& particle:particles)if(particle.weight>0.0)++live;
    auto const rank=cic_resample_detail::constraint_rank<RreaCicWide>(particles);
    return live>rank?live-rank:0;
}

// Host STL containers are adapted before entering device-callable numerics.
// Never instantiate a __host__ __device__ function with std::vector methods.
template<class Particle>
RREA_HOST_DEVICE inline std::size_t RreaCicResampleRemovalCapacity(
    rrea::KineticVector<Particle> const& particles) noexcept {
    return RreaCicResampleRemovalCapacity(rrea::KineticSpan<Particle const>{
        particles.data(),particles.size()});
}
template<class Particle,class Allocator>
inline std::size_t RreaCicResampleRemovalCapacity(
    std::vector<Particle,Allocator> const& particles) noexcept {
    return RreaCicResampleRemovalCapacity(rrea::KineticSpan<Particle const>{
        particles.data(),particles.size()});
}

// draw(draw_index) must return a deterministic uniform in (0,1).  Calls are
// made only for successful null-space steps and draw_index begins at zero.
template <class Particle, typename UniformDraw>
RREA_HOST_DEVICE RreaCicResampleResult RreaThinCicWeights(
    rrea::KineticSpan<Particle> particles,
    std::size_t maximum_removals,
    UniformDraw&& draw) noexcept
{
    RreaCicResampleResult result;
    std::array<std::size_t,5> active{};
    std::size_t live=0, window=0, cursor=0;
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
            ++live;
        }
    }

    while (result.removed < maximum_removals && live >= 2) {
        while(window<5 && cursor<particles.size()) {
            if(particles[cursor].weight>0.0)active[window++]=cursor;
            ++cursor;
        }
        std::size_t const n=window;
        std::array<cic_resample_detail::CicColumn<RreaCicWide>, 5> columns{};
        for (std::size_t c = 0; c < n; ++c) {
            columns[c] = cic_resample_detail::column<RreaCicWide>(particles[active[c]]);
        }
        std::array<RreaCicWide, 5> direction{};
        if (!cic_resample_detail::null_direction(columns, n, direction)) {
            if (live > n) {
                // Any five columns in four dimensions are dependent.  Treat
                // failure to recover their mixed-sign null vector as a
                // numerical error rather than rotating forever.
                result.valid = false;
                return result;
            }
            break;
        }

        RreaCicWide alpha_plus = std::numeric_limits<RreaCicWide>::infinity();
        RreaCicWide alpha_minus = std::numeric_limits<RreaCicWide>::infinity();
        std::size_t hit_plus = n;
        std::size_t hit_minus = n;
        for (std::size_t c = 0; c < n; ++c) {
            RreaCicWide const weight =
                static_cast<RreaCicWide>(particles[active[c]].weight);
            if (direction[c] < RreaCicWide(0.0)) {
                RreaCicWide const bound = weight / -direction[c];
                if (bound < alpha_plus) {
                    alpha_plus = bound;
                    hit_plus = c;
                }
            } else if (direction[c] > RreaCicWide(0.0)) {
                RreaCicWide const bound = weight / direction[c];
                if (bound < alpha_minus) {
                    alpha_minus = bound;
                    hit_minus = c;
                }
            }
        }
        if (!std::isfinite(alpha_plus) || !std::isfinite(alpha_minus)
            || !(alpha_plus >= RreaCicWide(0.0)) || !(alpha_minus >= RreaCicWide(0.0))
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
        RreaCicWide const plus_probability =
            alpha_minus / (alpha_plus + alpha_minus);
        bool const take_plus =
            static_cast<RreaCicWide>(uniform) < plus_probability;
        RreaCicWide const alpha = take_plus ? alpha_plus : -alpha_minus;
        std::size_t const guaranteed_zero = take_plus ? hit_plus : hit_minus;

        for (std::size_t c = 0; c < n; ++c) {
            RreaCicWide updated =
                static_cast<RreaCicWide>(particles[active[c]].weight)
                + alpha * direction[c];
            // Only the endpoint that defined alpha is analytically zero.
            // Clamping other small positive weights would be an extra vector
            // outside the CIC null space and would accumulate charge error.
            if (c == guaranteed_zero) {
                updated = RreaCicWide(0.0);
            }
            // A roundoff-negative weight is the analytic zero seen through
            // long-double arithmetic: clamp it inside the same slack the stored
            // weight verifier below already accepts (512 ulp x support size,
            // scaled by the weight), and kill the plan only for larger negatives.
            if (updated < RreaCicWide(0.0)
                && -updated <= RreaCicWide(512.0) * std::numeric_limits<double>::epsilon()
                        * static_cast<RreaCicWide>(n)
                        * rrea::kinetic::max(RreaCicWide(1.0), std::abs(static_cast<RreaCicWide>(
                              particles[active[c]].weight)))) {
                updated = RreaCicWide(0.0);
            }
            if (updated < RreaCicWide(0.0) || !std::isfinite(updated)) {
                result.valid = false;
                return result;
            }
            particles[active[c]].weight =
                updated > RreaCicWide(0.0) ? static_cast<double>(updated) : 0.0;
        }

        std::size_t keep=0;
        for(std::size_t i=0;i<window;++i)
            if(particles[active[i]].weight!=0.0)active[keep++]=active[i];
        std::size_t const removed_now=window-keep;
        window=keep;
        live-=removed_now;
        if (removed_now == 0) {
            result.valid = false;
            return result;
        }
        result.removed += removed_now;
    }
    return result;
}

template<class Particle,typename UniformDraw>
RREA_HOST_DEVICE RreaCicResampleResult RreaThinCicWeights(
    rrea::KineticVector<Particle>& particles,std::size_t request,UniformDraw&& draw) noexcept {
    return RreaThinCicWeights(rrea::KineticSpan<Particle>{particles.data(),particles.size()},
        request,static_cast<UniformDraw&&>(draw));
}
template<class Particle,class Allocator,typename UniformDraw>
RreaCicResampleResult RreaThinCicWeights(
    std::vector<Particle,Allocator>& particles,std::size_t request,UniformDraw&& draw) noexcept {
    return RreaThinCicWeights(rrea::KineticSpan<Particle>{particles.data(),particles.size()},
        request,static_cast<UniformDraw&&>(draw));
}

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

enum class RreaGlobalCicPlanStatus {
    Ready,
    InvalidInput,
    InvalidStoredMoment,
    InvalidReduction
};

template<template<class...>class Sequence>
struct BasicGlobalCicPlan {
    RreaGlobalCicPlanStatus status = RreaGlobalCicPlanStatus::Ready;
    std::size_t capacity = 0;
    std::size_t geometric_floor = 0;
    std::size_t requested = 0;
    std::size_t removed = 0;
    Sequence<RreaGlobalCicUpdate> updates;

    [[nodiscard]] RREA_HOST_DEVICE bool valid() const noexcept
    {
        return status == RreaGlobalCicPlanStatus::Ready;
    }
};
using RreaGlobalCicPlan=BasicGlobalCicPlan<std::vector>;
using RreaDeviceGlobalCicPlan=BasicGlobalCicPlan<rrea::KineticVector>;


RREA_HOST_DEVICE inline std::uint64_t RreaGlobalCicSupportKey(
    int support_r, int support_z, std::uint64_t stream_key = 0) noexcept
{
    std::uint64_t const cells =
        (static_cast<std::uint64_t>(static_cast<std::uint32_t>(support_r)) << 32)
        | static_cast<std::uint32_t>(support_z);
    return rrea::splitmix64(cells ^ rrea::splitmix64(stream_key));
}

RREA_HOST_DEVICE inline int RreaGlobalCicSupportOwner(
    int support_r, int support_z, int ranks,
    std::uint64_t stream_key = 0) noexcept
{
    return ranks > 0
        ? static_cast<int>(RreaGlobalCicSupportKey(
              support_r, support_z, stream_key) % static_cast<std::uint64_t>(ranks))
        : -1;
}

template <typename StoredReal>
RREA_HOST_DEVICE inline bool RreaRoundAndVerifyCicWeights(
    rrea::KineticVector<RreaCicResampleParticle> const& before,
    rrea::KineticVector<RreaCicResampleParticle> const& planned,
    rrea::KineticVector<StoredReal>& stored) noexcept
{
    if (before.size() != planned.size()) {
        return false;
    }
    auto moments = [](auto const& particles) {
        std::array<RreaCicWide, 4> cells{};
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
    rrea::KineticVector<RreaCicResampleParticle> rounded = planned;
    stored.resize(planned.size());
    for (std::size_t i = 0; i < planned.size(); ++i) {
        stored[i] = static_cast<StoredReal>(planned[i].weight);
        if(!std::isfinite(stored[i]) || stored[i]<StoredReal(0)) {
            stored.clear();return false;
        }
        rounded[i].weight = static_cast<double>(stored[i]);
    }
    auto const after_cells = moments(rounded);
    for (std::size_t cell = 0; cell < 4; ++cell) {
        RreaCicWide const scale = rrea::kinetic::max(RreaCicWide(1.0), std::abs(before_cells[cell]));
        RreaCicWide const tolerance = RreaCicWide(512.0)
            * std::numeric_limits<double>::epsilon()
            * static_cast<RreaCicWide>(before.size()) * scale;
        if (!(std::abs(after_cells[cell] - before_cells[cell]) <= tolerance)) {
            stored.clear();
            return false;
        }
    }
    return true;
}

template<typename StoredReal>
inline bool RreaRoundAndVerifyCicWeights(
    std::vector<RreaCicResampleParticle> const& before,
    std::vector<RreaCicResampleParticle> const& planned,
    std::vector<StoredReal>& stored) noexcept {
    rrea::KineticVector<RreaCicResampleParticle> a(before.begin(),before.end());
    rrea::KineticVector<RreaCicResampleParticle> b(planned.begin(),planned.end());
    rrea::KineticVector<StoredReal> out;
    bool const valid=RreaRoundAndVerifyCicWeights(a,b,out);
    stored.assign(out.begin(),out.end());
    return valid;
}

// One support's thinning plan.  Every input particle shares one four-cell
// support: the owner rank groups survivors and staged candidates by support
// (RreaCouplingPopulation) and routes the updates back by source_rank, which
// never enters the physics.  Ordering by (phase, candidate, stable id) makes
// the plan independent of the MPI decomposition.  draw(group_id,
// reduction_index) returns a uniform in (0,1).  All mutations stay in local
// temporaries until the StoredReal rounding and the four deposited moments
// have been rechecked.
template <typename StoredReal,typename UniformDraw>
RREA_HOST_DEVICE RreaDeviceGlobalCicPlan RreaPlanGlobalCicThinning(
    rrea::KineticVector<RreaGlobalCicParticle> const& input,
    std::size_t target_count,
    std::uint64_t stream_key,
    UniformDraw&& draw) noexcept
{
    RreaDeviceGlobalCicPlan plan;
    for (auto const& particle : input) {
        if (!std::isfinite(particle.fraction_r)
            || !std::isfinite(particle.fraction_z)
            || particle.fraction_r < 0.0 || particle.fraction_r > 1.0
            || particle.fraction_z < 0.0 || particle.fraction_z > 1.0
            || !std::isfinite(particle.weight) || particle.weight <= 0.0) {
            plan.status = RreaGlobalCicPlanStatus::InvalidInput;
            return plan;
        }
    }
    auto particles = input;
    rrea::kinetic::sort(
        particles.begin(), particles.end(),
        [](auto const& lhs, auto const& rhs) {
            if (lhs.phase != rhs.phase) return lhs.phase < rhs.phase;
            if (lhs.candidate != rhs.candidate) return lhs.candidate < rhs.candidate;
            if (lhs.stable_id_high != rhs.stable_id_high) {
                return lhs.stable_id_high < rhs.stable_id_high;
            }
            return lhs.stable_id < rhs.stable_id;
        });
    rrea::KineticVector<RreaCicResampleParticle> before;
    before.reserve(particles.size());
    for (auto const& particle : particles) {
        before.push_back({particle.fraction_r, particle.fraction_z, particle.weight});
    }
    plan.capacity = RreaCicResampleRemovalCapacity(before);
    plan.geometric_floor = particles.size() - plan.capacity;
    plan.requested = rrea::kinetic::min(
        particles.size() > target_count ? particles.size() - target_count : 0U,
        plan.capacity);
    if (plan.requested == 0) {
        return plan;
    }
    auto const& first = particles[0];
    std::uint64_t const group_id = rrea::splitmix64(
        RreaGlobalCicSupportKey(first.support_r, first.support_z, stream_key)
        ^ rrea::splitmix64(first.stable_id_high)
        ^ rrea::splitmix64(first.stable_id)
        ^ static_cast<std::uint64_t>(first.candidate));
    auto planned = before;
    auto const result = RreaThinCicWeights(
        planned, plan.requested,
        [&](std::size_t reduction) { return draw(group_id, reduction); });
    if (!result.valid || result.removed == 0) {
        plan.status = RreaGlobalCicPlanStatus::InvalidReduction;
        return plan;
    }
    rrea::KineticVector<StoredReal> stored;
    if (!RreaRoundAndVerifyCicWeights(before, planned, stored)) {
        plan.status = RreaGlobalCicPlanStatus::InvalidStoredMoment;
        return plan;
    }
    plan.removed = result.removed;
    plan.updates.reserve(particles.size());
    for (std::size_t i = 0; i < particles.size(); ++i) {
        plan.updates.push_back(RreaGlobalCicUpdate{
            particles[i].source_rank, particles[i].stable_id,
            static_cast<double>(stored[i]), particles[i].candidate,
            particles[i].stable_id_high});
    }
    return plan;
}

// Same equations and host long-double working precision; only the outer
// application-owned containers are std::vector.
template<typename StoredReal,typename UniformDraw>
RreaGlobalCicPlan RreaPlanGlobalCicThinning(
    std::vector<RreaGlobalCicParticle> const& input,std::size_t target_count,
    std::uint64_t stream_key,UniformDraw&& draw) noexcept {
    rrea::KineticVector<RreaGlobalCicParticle> particles(input.begin(),input.end());
    auto const numeric=RreaPlanGlobalCicThinning<StoredReal>(particles,target_count,
        stream_key,static_cast<UniformDraw&&>(draw));
    RreaGlobalCicPlan result;
    result.status=numeric.status;result.capacity=numeric.capacity;
    result.geometric_floor=numeric.geometric_floor;result.requested=numeric.requested;
    result.removed=numeric.removed;
    if(!numeric.updates.empty())result.updates.assign(numeric.updates.begin(),numeric.updates.end());
    return result;
}

}  // namespace rrea

#endif
