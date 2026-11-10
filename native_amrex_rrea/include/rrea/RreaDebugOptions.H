#pragma once

#include <AMReX_REAL.H>

namespace rrea {

struct RreaDebugTransportSensitivityOptions {
    bool enable = false;
    amrex::Real electron_collision_loss = amrex::Real(1.0);
    amrex::Real hard_moller_mfp = amrex::Real(1.0);
    amrex::Real electron_scattering_theta = amrex::Real(1.0);
    amrex::Real electron_wentzel_mfp = amrex::Real(1.0);
    amrex::Real hard_moller_secondary_excess_energy = amrex::Real(1.0);
    // Sensitivity scale for soft radiative drag, shared by both charges.
    amrex::Real soft_radiative_drag = amrex::Real(1.0);
};

[[nodiscard]] RreaDebugTransportSensitivityOptions const&
DebugTransportSensitivityOptions();

} // namespace rrea
