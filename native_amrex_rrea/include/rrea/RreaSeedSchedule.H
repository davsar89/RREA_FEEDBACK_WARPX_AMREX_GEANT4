#ifndef RREA_SEED_SCHEDULE_H
#define RREA_SEED_SCHEDULE_H

#include <AMReX_REAL.H>

#include "rrea/RreaConstants.H"
#include <AMReX_Vector.H>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>

namespace rrea {

// The schedule's u-triple is dual-interpretation, dispatched on magnitude:
// strictly below this threshold it is a dimensionless DIRECTION HINT
// (unit-ish or gamma*beta; the largest plausible gamma*beta at 1 GeV is
// ~2e3) whose magnitude is meaningless and is rescaled from
// kinetic_energy_eV; at or above it, the triple is already electron
// PROPER VELOCITY in m/s (1e6 m/s ~ gamma*beta 3.3e-3 ~ 2.8 eV, far
// below the certified electron floor), passed through unchanged and
// cross-checked against kinetic_energy_eV by ValidateProduction.  That
// leaves ~2.5 decades of dead band on each side of the cliff.  The
// validator and the injection hook MUST share this constant, or a
// schedule validates under one reading and injects under the other.
constexpr amrex::Real RreaSeedDirectionHintMaxMPerS = amrex::Real(1.0e6);

// Pinned by u(rest/4) == 3c/4 and the validator's inverse algebra.
[[nodiscard]] inline amrex::Real RreaElectronProperVelocityFromKineticEv(
    amrex::Real kinetic_energy_eV) noexcept
{
    constexpr amrex::Real electron_rest_energy_eV = amrex::Real(rrea::me_c2_eV);
    constexpr amrex::Real c_m_per_s = amrex::Real(rrea::c_m_per_s);
    if (kinetic_energy_eV <= amrex::Real(0.0)) {
        return amrex::Real(0.0);
    }
    amrex::Real const gamma =
        amrex::Real(1.0) + kinetic_energy_eV / electron_rest_energy_eV;
    return c_m_per_s
        * std::sqrt(std::max(
            gamma * gamma - amrex::Real(1.0), amrex::Real(0.0)));
}

struct RreaSeedMomentum {
    amrex::Real ux = 0.0;
    amrex::Real uy = 0.0;
    amrex::Real uz = 0.0;
};

// The magnitude dispatch itself: rescale a direction hint to the proper
// velocity implied by kinetic_energy_eV, pass a proper-velocity triple
// (or a degenerate hint: zero direction, nonpositive energy) through
// unchanged.
[[nodiscard]] inline RreaSeedMomentum RreaNormalizeSeedMomentum(
    amrex::Real ux,
    amrex::Real uy,
    amrex::Real uz,
    amrex::Real kinetic_energy_eV) noexcept
{
    amrex::Real const u_norm = std::sqrt(ux * ux + uy * uy + uz * uz);
    if (kinetic_energy_eV > amrex::Real(0.0)
        && u_norm > amrex::Real(0.0)
        && u_norm < RreaSeedDirectionHintMaxMPerS) {
        amrex::Real const u_mag =
            RreaElectronProperVelocityFromKineticEv(kinetic_energy_eV);
        return {u_mag * ux / u_norm, u_mag * uy / u_norm,
                u_mag * uz / u_norm};
    }
    return {ux, uy, uz};
}

// Injection-window classification for one schedule event against the step
// window [begin, end): before begin the event is stale (missed -- skip
// and advance the cursor), at or beyond end it belongs to a later step,
// otherwise it injects this step.
enum class RreaSeedWindowClass { SkipStale, Inject, BeyondWindow };

[[nodiscard]] inline RreaSeedWindowClass RreaSeedEventWindowClass(
    amrex::Real event_time_s,
    amrex::Real window_begin_s,
    amrex::Real window_end_s) noexcept
{
    if (event_time_s < window_begin_s) {
        return RreaSeedWindowClass::SkipStale;
    }
    if (event_time_s >= window_end_s) {
        return RreaSeedWindowClass::BeyondWindow;
    }
    return RreaSeedWindowClass::Inject;
}

// Seed species ids carried by the optional species_id CSV column (absent
// means electron).  Positrons share the electron rest mass, so the proper
// velocity and momentum primitives above apply to both.
constexpr int RreaSeedSpeciesElectron = 1;
constexpr int RreaSeedSpeciesPositron = 2;

struct RreaSeedEvent {
    std::string case_id;
    int macro_index = -1;
    int species_id = RreaSeedSpeciesElectron;
    amrex::Real time_s = 0.0;
    amrex::Real x_m = 0.0;
    amrex::Real y_m = 0.0;
    amrex::Real z_m = 0.0;
    amrex::Real r_m = 0.0;
    amrex::Real kinetic_energy_eV = 0.0;
    amrex::Real ux = 0.0;
    amrex::Real uy = 0.0;
    amrex::Real uz = 0.0;
    amrex::Real weight_real_electrons = 0.0;
};

class RreaSeedSchedule {
public:
    void Load(std::string const& path, std::string const& case_id);
    void ValidateProduction(
        amrex::Real electron_min_energy_eV,
        amrex::Real electron_max_energy_eV,
        amrex::Real radial_min_m,
        amrex::Real radial_max_m,
        amrex::Real axial_min_m,
        amrex::Real axial_max_m) const;

    [[nodiscard]] bool Empty() const noexcept;
    [[nodiscard]] std::size_t Size() const noexcept;
    [[nodiscard]] std::string const& Path() const noexcept;
    [[nodiscard]] RreaSeedEvent const& Event(std::size_t index) const;

private:
    std::string m_path;
    std::string m_case_id;
    amrex::Vector<RreaSeedEvent> m_events;
};

}  // namespace rrea

#endif
