#ifndef RREA_ALTITUDE_PROFILE_H
#define RREA_ALTITUDE_PROFILE_H

#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <string>

namespace rrea {

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
    amrex::Vector<amrex::Real> m_altitude_m;
    amrex::Vector<amrex::Real> m_value;
    amrex::Real m_max_abs_value = amrex::Real(0.0);
};

}  // namespace rrea

#endif
