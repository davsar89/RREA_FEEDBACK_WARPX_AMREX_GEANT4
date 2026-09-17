#include "rrea/RreaDebugOptions.H"

#include <AMReX.H>
#include <AMReX_ParmParse.H>

#include <cmath>
#include <sstream>

namespace rrea {

namespace {

void require_positive_finite(amrex::Real value, char const* name)
{
    if (!std::isfinite(static_cast<double>(value)) || value <= amrex::Real(0.0)) {
        std::ostringstream message;
        message << "rrea." << name << " must be finite and positive";
        amrex::Abort(message.str());
    }
}

} // namespace

RreaDebugTransportSensitivityOptions const& DebugTransportSensitivityOptions()
{
    static RreaDebugTransportSensitivityOptions const options = []() {
        RreaDebugTransportSensitivityOptions value;
        amrex::ParmParse pp("rrea");
        pp.query("debug_transport_sensitivity_enable", value.enable);
        pp.query(
            "debug_scale_electron_collision_loss",
            value.electron_collision_loss);
        pp.query("debug_scale_hard_moller_mfp", value.hard_moller_mfp);
        pp.query(
            "debug_scale_electron_scattering_theta",
            value.electron_scattering_theta);
        pp.query("debug_scale_electron_wentzel_mfp", value.electron_wentzel_mfp);
        pp.query(
            "debug_scale_hard_moller_secondary_excess_energy",
            value.hard_moller_secondary_excess_energy);
        pp.query(
            "debug_scale_soft_radiative_drag",
            value.soft_radiative_drag);
        require_positive_finite(
            value.electron_collision_loss,
            "debug_scale_electron_collision_loss");
        require_positive_finite(
            value.hard_moller_mfp,
            "debug_scale_hard_moller_mfp");
        require_positive_finite(
            value.electron_scattering_theta,
            "debug_scale_electron_scattering_theta");
        require_positive_finite(
            value.electron_wentzel_mfp,
            "debug_scale_electron_wentzel_mfp");
        require_positive_finite(
            value.hard_moller_secondary_excess_energy,
            "debug_scale_hard_moller_secondary_excess_energy");
        require_positive_finite(
            value.soft_radiative_drag,
            "debug_scale_soft_radiative_drag");
        return value;
    }();
    return options;
}

} // namespace rrea
