#ifndef RREA_SMOKE_REQUIRE_H
#define RREA_SMOKE_REQUIRE_H

// Shared fail-fast assertion helpers for the native smokes.  One tolerance
// convention lives here instead of ~15 per-file copies.  Comparisons are
// written NaN-rejecting (`!(diff <= tol)`), so a NaN never passes a
// closeness check.  Count-and-continue smokes keep their local counters on
// purpose; this header only owns the throwing idiom.

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace rrea::smoke {

inline void require(bool condition, std::string const& message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

inline std::string mismatch_detail(
    std::string const& label, double actual, double expected)
{
    std::ostringstream detail;
    detail.precision(17);
    detail << label << " mismatch: actual=" << actual
           << " expected=" << expected;
    return detail.str();
}

// Epsilon-scaled closeness on scale max(1, |actual|, |expected|); the factor
// is in units of machine epsilon (default 64, the tightest convention the
// smokes use).
inline void require_close(
    double actual,
    double expected,
    std::string const& label,
    double epsilon_factor = 64.0)
{
    double const scale =
        std::max({1.0, std::abs(actual), std::abs(expected)});
    double const tolerance =
        epsilon_factor * std::numeric_limits<double>::epsilon() * scale;
    if (!(std::abs(actual - expected) <= tolerance)) {
        throw std::runtime_error(mismatch_detail(label, actual, expected));
    }
}

// Relative closeness on scale max(|expected|, 1).
inline void require_close_rel(
    double actual,
    double expected,
    double relative_tolerance,
    std::string const& label)
{
    double const scale = std::max(std::abs(expected), 1.0);
    if (!(std::abs(actual - expected) <= relative_tolerance * scale)) {
        throw std::runtime_error(mismatch_detail(label, actual, expected));
    }
}

template <typename Function>
inline void require_throws(Function&& function, std::string const& message)
{
    bool threw = false;
    try {
        std::forward<Function>(function)();
    } catch (std::exception const&) {
        threw = true;
    }
    require(threw, message);
}

} // namespace rrea::smoke

#endif // RREA_SMOKE_REQUIRE_H
