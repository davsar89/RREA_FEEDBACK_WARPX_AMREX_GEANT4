#pragma once

#include <array>
#include <cstddef>
#include <utility>

namespace rrea {

namespace detail {
// Shared eight-point Gauss-Legendre abscissae/weights on [-1, 1].  Both
// integrators below MUST sample in the identical fixed order so scalar and
// array integrations of the same integrand agree bit-for-bit.
inline constexpr std::array<long double, 4> kRreaGl8Nodes = {{
    0.183434642495649804939476142360184L,
    0.525532409916328985817739049189246L,
    0.796666477413626739591553936475831L,
    0.960289856497536231683560868569473L}};
inline constexpr std::array<long double, 4> kRreaGl8Weights = {{
    0.362683783378361982965150449277195L,
    0.313706645877887287337962201986601L,
    0.222381034453374470544355994426240L,
    0.101228536290376259152531354309962L}};
// Two-point node 1/sqrt(3); both weights are 1.
inline constexpr long double kRreaGl2Node =
    0.577350269189625764509148780501957L;
}  // namespace detail

// Deterministic two-point Gauss-Legendre integration of an N-component
// integrand, EXACT for any cubic.  The charged field impulse uses it per
// in-domain chord piece, where the gathered field is bilinear in (r, z) and
// the chord is affine, so the integrand is quadratic in the path fraction.
// Same fixed (-, +) sample order as the eight-point rules above.
template <typename Real, std::size_t N, typename Function>
std::array<Real, N> RreaGaussLegendre2IntegrateArray(
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

// Deterministic eight-point Gauss-Legendre integration on [begin, end].
// Samples are evaluated in the fixed order (-x0,+x0,...,-x3,+x3); do not
// reorder or parallelize this loop because transport reproducibility relies on
// an identical field-gather and floating-point accumulation order.
template <typename Real, typename Function>
Real RreaGaussLegendre8Integrate(
    Real begin,
    Real end,
    Function&& function)
{
    auto const& nodes = detail::kRreaGl8Nodes;
    auto const& weights = detail::kRreaGl8Weights;
    if (!(end > begin)) {
        return Real(0);
    }
    Real const midpoint = Real(0.5) * (begin + end);
    Real const half_span = Real(0.5) * (end - begin);
    Real weighted_sum = Real(0);
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        Real const offset = half_span * static_cast<Real>(nodes[i]);
        Real const weight = static_cast<Real>(weights[i]);
        weighted_sum += weight * static_cast<Real>(function(midpoint - offset));
        weighted_sum += weight * static_cast<Real>(function(midpoint + offset));
    }
    return half_span * weighted_sum;
}

// N-component variant used to integrate the charged substep
// field impulse (scalar work plus the vector field average) in ONE pass so
// each Gauss-Legendre node gathers the field exactly once.  Same fixed node
// order and accumulation order as the scalar integrator above; component 0 of
// an all-scalar integrand is bit-identical with RreaGaussLegendre8Integrate.
template <typename Real, std::size_t N, typename Function>
std::array<Real, N> RreaGaussLegendre8IntegrateArray(
    Real begin,
    Real end,
    Function&& function)
{
    auto const& nodes = detail::kRreaGl8Nodes;
    auto const& weights = detail::kRreaGl8Weights;
    std::array<Real, N> integral{};
    if (!(end > begin)) {
        return integral;
    }
    Real const midpoint = Real(0.5) * (begin + end);
    Real const half_span = Real(0.5) * (end - begin);
    std::array<Real, N> weighted_sum{};
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        Real const offset = half_span * static_cast<Real>(nodes[i]);
        Real const weight = static_cast<Real>(weights[i]);
        std::array<Real, N> const below = function(midpoint - offset);
        for (std::size_t c = 0; c < N; ++c) {
            weighted_sum[c] += weight * below[c];
        }
        std::array<Real, N> const above = function(midpoint + offset);
        for (std::size_t c = 0; c < N; ++c) {
            weighted_sum[c] += weight * above[c];
        }
    }
    for (std::size_t c = 0; c < N; ++c) {
        integral[c] = half_span * weighted_sum[c];
    }
    return integral;
}

}  // namespace rrea
