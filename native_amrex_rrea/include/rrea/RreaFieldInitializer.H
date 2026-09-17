#ifndef RREA_FIELD_INITIALIZER_H
#define RREA_FIELD_INITIALIZER_H

#include "rrea/RreaAltitudeProfile.H"

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

namespace rrea {

struct FieldInitializerConfig {
    double ez_acceleration_v_per_m = 0.0;
    double acceleration_z_max_m = 0.0;
    bool use_altitude_profile = false;
    double profile_altitude_msl_at_z0_m = 0.0;
    double profile_phi_integration_step_m = 1.0;
    // Radial taper of the background field: full strength for r <= start,
    // quintic-smoothstep to zero at r >= end.  Applied at the potential
    // level (phi_bg(r,z) = phi_ref + f(r)*(phi_z(z) - phi_ref)) so the
    // background stays curl-free and L(phi_bg) == rho_ext keeps holding.
    // Taper is active iff 0 <= start < end; the -1 defaults disable it.
    double taper_r_start_m = -1.0;
    double taper_r_end_m = -1.0;
    RreaAltitudeProfile ez_altitude_profile;

    bool TaperActive() const noexcept
    {
        return taper_r_start_m >= 0.0 && taper_r_end_m > taper_r_start_m;
    }
};

class RreaFieldInitializer {
public:
    // The analytic axial ambient at a point, including the radial potential
    // taper: the single source for diagnostics that compare a solved field
    // with its imposed profile.  The field the run evolves is the compatible
    // face gradient of InitializePhi's potential (RreaAmrexAdvance).
    [[nodiscard]] static amrex::Real EvaluateBackgroundEz(
        amrex::Real r_m,
        amrex::Real z_m,
        FieldInitializerConfig const& config);

    static void InitializePhi(
        amrex::MultiFab& phi,
        amrex::Geometry const& geom,
        FieldInitializerConfig const& config);
};

}  // namespace rrea

#endif
