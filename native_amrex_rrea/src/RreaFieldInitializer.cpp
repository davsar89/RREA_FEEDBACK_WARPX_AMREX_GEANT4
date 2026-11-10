#include "rrea/RreaFieldInitializer.H"

#include "rrea/RreaConstants.H"

#include <AMReX_Array4.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_IntVect.H>
#include <AMReX_MFIter.H>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace rrea {

namespace {

double ez_from_z(double z, FieldInitializerConfig const& config)
{
    if (!config.use_altitude_profile) {
        return (z <= config.acceleration_z_max_m)
            ? config.ez_acceleration_v_per_m
            : 0.0;
    }
    if (!config.ez_altitude_profile.Loaded()) {
        throw std::runtime_error("Altitude-profile field initialization requested without a loaded profile");
    }
    double const altitude_m = config.profile_altitude_msl_at_z0_m + z;
    if (altitude_m < static_cast<double>(config.ez_altitude_profile.MinAltitudeM())
        || altitude_m > static_cast<double>(config.ez_altitude_profile.MaxAltitudeM())) {
        return 0.0;
    }
    double const normalized =
        config.ez_altitude_profile.Interpolate(altitude_m)
        / config.ez_altitude_profile.MaxAbsValue();
    return -std::abs(config.ez_acceleration_v_per_m) * normalized;
}

double background_phi_from_z(double z, FieldInitializerConfig const& config)
{
    if (config.use_altitude_profile) {
        double const z_hi = std::max(z, 0.0);
        double const step_m = std::max(config.profile_phi_integration_step_m, 0.1);
        int const steps = std::max(1, static_cast<int>(std::ceil(z_hi / step_m)));
        double phi = 0.0;
        double previous_z = 0.0;
        double previous_ez = ez_from_z(previous_z, config);
        for (int step = 1; step <= steps; ++step) {
            double const current_z = z_hi * static_cast<double>(step)
                / static_cast<double>(steps);
            double const current_ez = ez_from_z(current_z, config);
            phi += -0.5 * (previous_ez + current_ez) * (current_z - previous_z);
            previous_z = current_z;
            previous_ez = current_ez;
        }
        return phi;
    }
    double const clamped_z = amrex::min(amrex::max(z, 0.0), config.acceleration_z_max_m);
    return -config.ez_acceleration_v_per_m * clamped_z;
}

int constexpr axial_dir = 1;

// Radial taper weight: 1 in the core (r <= start), quintic smoothstep down
// to 0 at r >= end.  C2 at both ends, so the fringe field f'(r) vanishes at
// the core edge and at the wall and the implied background charge stays
// smooth across the taper annulus.
double taper_f(double r, FieldInitializerConfig const& config)
{
    if (!config.TaperActive() || r <= config.taper_r_start_m) {
        return 1.0;
    }
    if (r >= config.taper_r_end_m) {
        return 0.0;
    }
    double const t = (r - config.taper_r_start_m)
        / (config.taper_r_end_m - config.taper_r_start_m);
    return 1.0 - t * t * t * (t * (6.0 * t - 15.0) + 10.0);
}

// Gauge reference for the tapered background: phi_bg(r,z) relaxes toward
// phi_ref as f(r) -> 0, and Er = -f'(r)*(phi_z(z)-phi_ref), so the midrange
// of phi_z over the domain minimizes and symmetrizes the fringe field.  The
// wall column (f=0) is pinned at the constant phi_ref: a distant grounded
// shell instead of a wall carrying the full ambient potential drop.
double taper_phi_reference(
    amrex::Geometry const& geom,
    FieldInitializerConfig const& config)
{
    int constexpr samples = 257;
    double const z_lo = geom.ProbLo(axial_dir);
    double const z_hi = geom.ProbHi(axial_dir);
    double phi_min = std::numeric_limits<double>::infinity();
    double phi_max = -std::numeric_limits<double>::infinity();
    for (int sample = 0; sample < samples; ++sample) {
        double const z = z_lo
            + (z_hi - z_lo) * static_cast<double>(sample)
                / static_cast<double>(samples - 1);
        double const phi = background_phi_from_z(z, config);
        phi_min = std::min(phi_min, phi);
        phi_max = std::max(phi_max, phi);
    }
    return 0.5 * (phi_min + phi_max);
}

void validate_taper(
    amrex::Geometry const& geom,
    FieldInitializerConfig const& config)
{
    bool const configured =
        config.taper_r_start_m >= 0.0 || config.taper_r_end_m >= 0.0;
    if (!configured) {
        return;
    }
    if (!config.TaperActive()) {
        throw std::runtime_error(
            "rrea.field_taper requires 0 <= field_taper_r_start_m < "
            "field_taper_r_end_m (set both, or neither)");
    }
    double const r_hi = geom.ProbHi(0);
    double const tolerance = 64.0
        * std::numeric_limits<double>::epsilon() * std::max(1.0, r_hi);
    if (config.taper_r_end_m > r_hi + tolerance) {
        std::ostringstream message;
        message << "radial field taper end " << config.taper_r_end_m
                << " m exceeds the domain radius " << r_hi << " m";
        throw std::runtime_error(message.str());
    }
}

}  // namespace

amrex::Real RreaFieldInitializer::EvaluateBackgroundEz(
    amrex::Real r_m,
    amrex::Real z_m,
    FieldInitializerConfig const& config)
{
    return static_cast<amrex::Real>(
        taper_f(static_cast<double>(r_m), config)
        * ez_from_z(static_cast<double>(z_m), config));
}

void RreaFieldInitializer::InitializePhi(
    amrex::MultiFab& phi,
    amrex::Geometry const& geom,
    FieldInitializerConfig const& config)
{
    validate_taper(geom, config);
    phi.setVal(0.0);
    auto const dx = geom.CellSizeArray();
    auto const plo = geom.ProbLoArray();
    auto const phi_domain = geom.Domain();

    int constexpr z_dir = axial_dir;
    int const zlo = phi_domain.smallEnd(z_dir);
    int const zhi = phi_domain.bigEnd(z_dir);
    double const z_face_lo = geom.ProbLo(z_dir);
    double const z_face_hi = geom.ProbHi(z_dir);
    double const phi_ref =
        config.TaperActive() ? taper_phi_reference(geom, config) : 0.0;
    int const rlo = phi_domain.smallEnd(0);
    int const rhi = phi_domain.bigEnd(0);
    double const r_face_lo = geom.ProbLo(0);
    double const r_face_hi = geom.ProbHi(0);

    // Integrate each axial profile point ONCE, rather than once per radial
    // cell. Only the compact profiles are built on the host; their tensor
    // product fills the mesh on the GPU.
    int const profile_zlo = zlo - phi.nGrowVect()[z_dir];
    int const profile_zhi = zhi + phi.nGrowVect()[z_dir];
    GpuVector<double> axial_phi(profile_zhi - profile_zlo + 1);
    for (int iz = profile_zlo; iz <= profile_zhi; ++iz) {
        double z;
        if (iz < zlo) {
            z = z_face_lo;
        } else if (iz > zhi) {
            z = z_face_hi;
        } else {
            z = plo[z_dir] + (iz - zlo + 0.5) * dx[z_dir];
        }
        axial_phi[iz - profile_zlo] = background_phi_from_z(z, config);
    }
    int const profile_rlo = rlo - phi.nGrowVect()[0];
    int const profile_rhi = rhi + phi.nGrowVect()[0];
    GpuVector<double> radial_taper(profile_rhi - profile_rlo + 1);
    for (int i = profile_rlo; i <= profile_rhi; ++i) {
        double const radius = i < rlo ? r_face_lo : i > rhi ? r_face_hi
            : plo[0] + (i - rlo + 0.5) * dx[0];
        radial_taper[i - profile_rlo] = taper_f(radius, config);
    }
    auto const* const taper = radial_taper.data();
    auto const* const profile = axial_phi.data();
    for (amrex::MFIter mfi(phi, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        amrex::Box const bx = mfi.growntilebox(phi.nGrowVect());
        auto const arr = phi.array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            arr(i, j, k) = phi_ref + taper[i - profile_rlo]
                * (profile[j - profile_zlo] - phi_ref);
        });
    }
    GpuSynchronize();  // local profile storage must outlive every fill kernel
}

}  // namespace rrea
