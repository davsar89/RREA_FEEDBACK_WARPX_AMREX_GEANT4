#ifndef RREA_LORENTZ_PUSH_H
#define RREA_LORENTZ_PUSH_H

#include "rrea/RreaConstants.H"

#include <AMReX_REAL.H>

#include <cmath>

namespace rrea {

struct RreaProperVelocity3 {
    amrex::Real x = 0.0;
    amrex::Real y = 0.0;
    amrex::Real z = 0.0;
};

// Relativistic Boris magnetic rotation. q_over_m_dt already carries the
// charge sign and the duration of this kick. Magnetic work is identically
// zero; the returned proper velocity has the input magnitude to round-off.
[[nodiscard]] inline RreaProperVelocity3 RreaBorisMagneticRotate(
    RreaProperVelocity3 const& u,
    amrex::Real q_over_m_dt,
    amrex::Real bx_t,
    amrex::Real by_t,
    amrex::Real bz_t) noexcept
{
    amrex::Real const gamma = std::sqrt(
        amrex::Real(1.0) + (u.x * u.x + u.y * u.y + u.z * u.z)
            / (amrex::Real(c_m_per_s) * amrex::Real(c_m_per_s)));
    amrex::Real const tx = q_over_m_dt * bx_t / (amrex::Real(2.0) * gamma);
    amrex::Real const ty = q_over_m_dt * by_t / (amrex::Real(2.0) * gamma);
    amrex::Real const tz = q_over_m_dt * bz_t / (amrex::Real(2.0) * gamma);
    amrex::Real const t2 = tx * tx + ty * ty + tz * tz;
    amrex::Real const sx = amrex::Real(2.0) * tx / (amrex::Real(1.0) + t2);
    amrex::Real const sy = amrex::Real(2.0) * ty / (amrex::Real(1.0) + t2);
    amrex::Real const sz = amrex::Real(2.0) * tz / (amrex::Real(1.0) + t2);
    RreaProperVelocity3 const prime{
        u.x + u.y * tz - u.z * ty,
        u.y + u.z * tx - u.x * tz,
        u.z + u.x * ty - u.y * tx};
    return {
        u.x + prime.y * sz - prime.z * sy,
        u.y + prime.z * sx - prime.x * sz,
        u.z + prime.x * sy - prime.y * sx};
}

}  // namespace rrea

#endif
