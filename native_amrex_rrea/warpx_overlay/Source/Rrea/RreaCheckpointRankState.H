#ifndef RREA_CHECKPOINT_RANK_STATE_H_
#define RREA_CHECKPOINT_RANK_STATE_H_

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <ios>
#include <limits>
#include <type_traits>

namespace rrea::warpx {

// How one rank-local registry field maps onto a different rank count. The
// classification rule is semantic, not syntactic:
// anything produced by (or after) an MPI collective is Replicated -- identical
// on every rank, so summing it across a fold would double-count a global.
// Rank-local partial accumulators whose only consumer is a global sum are Sum;
// rank-local running watermarks consumed via a global max are Max.  Both fold
// identities are zero, so Sum and Max fields must be non-negative by
// construction (every registered one is a count, weight, energy, tau, or
// fraction).
enum class RreaRankStateFoldClass : std::uint8_t {
    Replicated,
    Sum,
    Max,
};

struct RreaRankStateFoldStatus
{
    bool ok = false;
    char const* error = "unknown rank-state fold error";
    // First offending registry index when !ok (valid only for per-field errors).
    std::size_t field = 0;
};

// Bit-level equality is the right notion for Replicated fields: they are the
// byproduct of one deterministic collective, so even -0.0 vs 0.0 or a NaN
// payload difference is evidence of misclassification, not of tolerance.
template <typename T>
inline bool RreaFoldBitsEqual(T const& a, T const& b)
{
    return std::memcmp(&a, &b, sizeof(T)) == 0;
}

// Map saved_nprocs stored rows onto current_nprocs ranks and emit the row for
// rank my_rank.  Row r contributes to rank r % current_nprocs, which is the
// identity permutation when the counts match, so a same-count fold reproduces
// the stored slice exactly. Replicated fields are verified bit-identical
// across every stored row first -- the misclassification tripwire -- and a Sum
// that wraps (integral) or leaves the finite range (floating) is an error, so
// a wrong fold can never restore silently.
template <typename T>
inline RreaRankStateFoldStatus RreaFoldRankStateRow(
    T const* all_rows,
    std::uint64_t saved_nprocs,
    std::uint64_t current_nprocs,
    std::uint64_t my_rank,
    std::size_t fields_per_rank,
    RreaRankStateFoldClass const* rules,
    std::size_t rule_count,
    T* out_row)
{
    RreaRankStateFoldStatus status;
    if (all_rows == nullptr || rules == nullptr || out_row == nullptr) {
        status.error = "rank-state fold was given a null buffer";
        return status;
    }
    if (saved_nprocs == 0 || current_nprocs == 0 || my_rank >= current_nprocs) {
        status.error = "rank-state fold rank shape is invalid";
        return status;
    }
    if (rule_count != fields_per_rank || fields_per_rank == 0) {
        status.error =
            "rank-state fold rules do not match the registry width";
        return status;
    }
    std::size_t const np = static_cast<std::size_t>(saved_nprocs);
    for (std::size_t f = 0; f < fields_per_rank; ++f) {
        if (rules[f] != RreaRankStateFoldClass::Replicated) {
            continue;
        }
        for (std::size_t r = 1; r < np; ++r) {
            if (!RreaFoldBitsEqual(all_rows[r * fields_per_rank + f],
                                   all_rows[f])) {
                status.error =
                    "rank-state field classified Replicated differs between "
                    "stored ranks";
                status.field = f;
                return status;
            }
        }
    }
    for (std::size_t f = 0; f < fields_per_rank; ++f) {
        T value{};
        bool ok = true;
        switch (rules[f]) {
        case RreaRankStateFoldClass::Replicated:
            value = all_rows[f];
            break;
        case RreaRankStateFoldClass::Sum:
        case RreaRankStateFoldClass::Max:
            for (std::size_t r = static_cast<std::size_t>(my_rank); r < np;
                 r += static_cast<std::size_t>(current_nprocs)) {
                T const term = all_rows[r * fields_per_rank + f];
                if (rules[f] == RreaRankStateFoldClass::Max) {
                    value = value < term ? term : value;
                } else if constexpr (std::is_integral_v<T>) {
                    if (value > std::numeric_limits<T>::max() - term) {
                        ok = false;
                        break;
                    }
                    value = static_cast<T>(value + term);
                } else {
                    value += term;
                }
            }
            if constexpr (!std::is_integral_v<T>) {
                ok = ok && std::isfinite(static_cast<double>(value));
            }
            break;
        }
        if (!ok) {
            status.error = "rank-state fold left the representable range";
            status.field = f;
            return status;
        }
        out_row[f] = value;
    }
    status.ok = true;
    status.error = "";
    return status;
}

struct RreaRankStatePayloadLayout
{
    bool valid = false;
    char const* error = "unknown rank-state layout error";
    std::size_t integer_count = 0;
    std::size_t real_count = 0;
    std::size_t integer_bytes = 0;
    std::size_t real_bytes = 0;
    std::size_t payload_bytes = 0;
};

// Dimensions of one rank_state.bin payload: one fixed-size registry row per
// rank, integers then reals.  Zero dimensions are never a valid checkpoint,
// and every collective count must fit an MPI int.
inline RreaRankStatePayloadLayout RreaValidateRankStatePayloadLayout(
    std::uint64_t nprocs,
    std::uint64_t integer_fields_per_rank,
    std::uint64_t real_fields_per_rank,
    std::uint32_t real_size)
{
    RreaRankStatePayloadLayout result;
    auto const mpi_max = static_cast<std::uint64_t>(std::numeric_limits<int>::max());
    if (nprocs == 0 || integer_fields_per_rank == 0 || real_fields_per_rank == 0) {
        result.error = "rank count and per-rank field counts must be nonzero";
    } else if (real_size != 4U && real_size != 8U) {
        result.error = "unsupported Real element size";
    } else if (nprocs > mpi_max / integer_fields_per_rank
               || nprocs > mpi_max / real_fields_per_rank) {
        result.error = "rank-state collective element count does not fit int";
    } else {
        result.integer_count = static_cast<std::size_t>(nprocs * integer_fields_per_rank);
        result.real_count = static_cast<std::size_t>(nprocs * real_fields_per_rank);
        result.integer_bytes = result.integer_count * sizeof(std::uint64_t);
        result.real_bytes = result.real_count * static_cast<std::size_t>(real_size);
        result.payload_bytes = result.integer_bytes + result.real_bytes;
        result.valid = true;
        result.error = "";
    }
    return result;
}

}  // namespace rrea::warpx

#endif
