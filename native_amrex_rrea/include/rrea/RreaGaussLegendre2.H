#pragma once

#include <array>
#include "rrea/RreaGpu.H"
#include <cstddef>
#include <utility>

namespace rrea {

namespace detail {
// Two-point Gauss-Legendre node 1/sqrt(3) on [-1, 1]; both weights are 1.
inline constexpr long double kRreaGl2Node =
    0.577350269189625764509148780501957L;
}  // namespace detail

// Deterministic two-point Gauss-Legendre integration of an N-component
// integrand, EXACT for any cubic. Field-work applicability and approximation
// limits belong to the consumer in RreaPathFieldWork.H.
// Samples are evaluated in the fixed order (-, +).
template <typename Real, std::size_t N, typename Function>
RREA_HOST_DEVICE std::array<Real, N> RreaGaussLegendre2IntegrateArray(
    Real begin,
    Real end,
    Function&& function)
{
    std::array<Real, N> integral{};
    if (!(end > begin)) {
        return integral;
    }
    Real const midpoint = Real(0.5) * (begin + end);
    Real const half_span = Real(0.5) * (end - begin);
    Real const offset = half_span * static_cast<Real>(detail::kRreaGl2Node);
    std::array<Real, N> const below = function(midpoint - offset);
    std::array<Real, N> const above = function(midpoint + offset);
    for (std::size_t c = 0; c < N; ++c) {
        integral[c] = half_span * (below[c] + above[c]);
    }
    return integral;
}

}  // namespace rrea
