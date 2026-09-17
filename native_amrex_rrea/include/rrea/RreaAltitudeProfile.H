#ifndef RREA_ALTITUDE_PROFILE_H
#define RREA_ALTITUDE_PROFILE_H

#include <AMReX_REAL.H>
#include <AMReX_Vector.H>
#include "rrea/RreaKineticSupport.H"

#include <string>

namespace rrea {


// Immutable numeric profile; storage is owned by the host loader, not a kernel.
struct RreaAltitudeProfileView {
    KineticSpan<amrex::Real const> altitudes;
    KineticSpan<amrex::Real const> values;
    [[nodiscard]] RREA_HOST_DEVICE amrex::Real Interpolate(amrex::Real altitude_m) const {
        if (altitudes.empty() || altitudes.size()!=values.size()
            || !std::isfinite(altitude_m)
            || altitude_m<altitudes.front() || altitude_m>altitudes.back()) {
            KineticFail("RREA altitude outside loaded profile range");
        }
    auto const upper = rrea::kinetic::upper_bound(altitudes.begin(), altitudes.end(), altitude_m);
    if (upper == altitudes.begin()) {
        return values.front();
    }
    if (upper == altitudes.end()) {
        return values.back();
    }
    std::size_t const hi = static_cast<std::size_t>(upper - altitudes.begin());
    std::size_t const lo = hi - 1U;
    amrex::Real const span = altitudes[hi] - altitudes[lo];
    amrex::Real const t = span > amrex::Real(0.0)
        ? (altitude_m - altitudes[lo]) / span
        : amrex::Real(0.0);
    return (amrex::Real(1.0) - t) * values[lo] + t * values[hi];
    }
};

class RreaAltitudeProfile {
public:
    // require_positive rejects nonpositive scaled values at ingestion; density
    // profiles use it because zero or negative neutral density is invalid.
    void LoadCsv(
        std::string const& path,
        std::string const& altitude_column,
        std::string const& value_column,
        amrex::Real altitude_scale,
        amrex::Real value_scale,
        bool require_positive = false);

    [[nodiscard]] RreaAltitudeProfileView View() const noexcept {
        return {{m_altitude_m.data(), m_altitude_m.size()}, {m_value.data(), m_value.size()}};
    }
    [[nodiscard]] bool Loaded() const noexcept;
    [[nodiscard]] amrex::Real Interpolate(amrex::Real altitude_m) const;
    [[nodiscard]] amrex::Real MaxAbsValue() const noexcept;
    [[nodiscard]] amrex::Real MinAltitudeM() const noexcept;
    [[nodiscard]] amrex::Real MaxAltitudeM() const noexcept;
    [[nodiscard]] std::string const& Path() const noexcept;
    [[nodiscard]] std::string const& AltitudeColumn() const noexcept;
    [[nodiscard]] std::string const& ValueColumn() const noexcept;

private:
    std::string m_path;
    std::string m_altitude_column;
    std::string m_value_column;
    GpuVector<amrex::Real> m_altitude_m;
    GpuVector<amrex::Real> m_value;
    amrex::Real m_max_abs_value = amrex::Real(0.0);
};

}  // namespace rrea

#endif
