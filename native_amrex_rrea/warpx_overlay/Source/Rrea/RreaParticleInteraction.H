#ifndef WARPX_RREA_PARTICLE_INTERACTION_H
#define WARPX_RREA_PARTICLE_INTERACTION_H

#include <AMReX_REAL.H>

#include "rrea/RreaConstants.H"
#include "rrea/RreaSegmentDepositionRZ.H"
#include "rrea/RreaTransportPath.H"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

class WarpX;

namespace rrea {
class RreaInteractionTables;
}

namespace rrea::warpx {

class RreaWarpXCoupling;

enum class RreaSecondarySpecies : int {
    Electron,
    Positron,
    Photon,
};

enum class RreaSecondaryProcess : int {
    ContinuousTransport,
    Seed,
    HardMoller,
    HardBhabha,
    Bremsstrahlung,
    Compton,
    Photoelectric,
    PairNuclear,
    PairTriplet,
    PositronAnnihilation,
    PhotonEscape,
    ParticleEscape,
    LowEnergyDemotion,
};

// Terminal decisions at the clipped end of a charged-particle chord.  An
// absorbing-boundary crossing owns the parent before any endpoint-only fluid
// handoff or at-rest annihilation can create a source at the half-open upper
// boundary.  Material interactions sampled strictly inside the chord remain
// authoritative and are resolved before this endpoint decision is queried.
enum class RreaChargedTerminalDisposition : int {
    Survive,
    Escape,
    ElectronDemotion,
    PositronAnnihilation,
};

struct RreaContinuousSegmentLimit {
    amrex::Real effective_end_fraction = amrex::Real(0.0);
    bool reaches_cutoff = false;
    bool valid = false;
};

// Authoritative spatial state for charged transport reconstructed from the
// captured pre-push state.  WarpX's charged endpoint is deliberately not kept:
// field impulses, elastic scattering, and discrete recoil can all change the
// direction before the outer step ends.
struct RreaRunningChargedState {
    amrex::Real x_m = amrex::Real(0.0);
    amrex::Real y_m = amrex::Real(0.0);
    amrex::Real z_m = amrex::Real(0.0);
    amrex::Real dir_x = amrex::Real(0.0);
    amrex::Real dir_y = amrex::Real(0.0);
    amrex::Real dir_z = amrex::Real(1.0);
    amrex::Real kinetic_energy_eV = amrex::Real(0.0);
};

struct RreaRunningChargedChord {
    rrea::RreaTransportPathSegment path;
    amrex::Real requested_dt_s = amrex::Real(0.0);
    amrex::Real requested_outer_step_fraction = amrex::Real(0.0);
    bool valid = false;
};

constexpr amrex::Real kRreaElectronRestEnergyEv = amrex::Real(rrea::me_c2_eV);
constexpr amrex::Real kRreaSpeedOfLightMPerS = amrex::Real(rrea::c_m_per_s);

[[nodiscard]] inline amrex::Real RreaChargedSpeedFromKineticEnergy(
    amrex::Real kinetic_energy_eV) noexcept
{
    if (!std::isfinite(static_cast<double>(kinetic_energy_eV))
        || kinetic_energy_eV < amrex::Real(0.0)) {
        return std::numeric_limits<amrex::Real>::quiet_NaN();
    }
    amrex::Real const gamma =
        amrex::Real(1.0) + kinetic_energy_eV / kRreaElectronRestEnergyEv;
    amrex::Real const beta_squared = std::max(
        amrex::Real(0.0),
        amrex::Real(1.0) - amrex::Real(1.0) / (gamma * gamma));
    return kRreaSpeedOfLightMPerS * std::sqrt(beta_squared);
}

// Build one numerical charged chord from the current state.  The chord is a
// proposal only: callers must resolve boundary/event/cutoff ordering before
// committing any position, source, crossing, or ledger side effect.
[[nodiscard]] inline RreaRunningChargedChord RreaBuildRunningChargedChord(
    RreaRunningChargedState const& state,
    amrex::Real requested_dt_s,
    amrex::Real outer_dt_s,
    amrex::Real domain_r_lo_m,
    amrex::Real domain_r_hi_m,
    amrex::Real domain_z_lo_m,
    amrex::Real domain_z_hi_m,
    bool periodic_z = false)
{
    RreaRunningChargedChord result;
    result.requested_dt_s = requested_dt_s;
    if (!std::isfinite(static_cast<double>(requested_dt_s))
        || !std::isfinite(static_cast<double>(outer_dt_s))
        || !(requested_dt_s >= amrex::Real(0.0))
        || !(outer_dt_s > amrex::Real(0.0))) {
        return result;
    }
    amrex::Real const speed_m_per_s =
        RreaChargedSpeedFromKineticEnergy(state.kinetic_energy_eV);
    amrex::Real const direction_norm = std::sqrt(
        state.dir_x * state.dir_x
        + state.dir_y * state.dir_y
        + state.dir_z * state.dir_z);
    if (!std::isfinite(static_cast<double>(speed_m_per_s))
        || !std::isfinite(static_cast<double>(direction_norm))
        || !(direction_norm > amrex::Real(0.0))) {
        return result;
    }
    amrex::Real const scale =
        speed_m_per_s * requested_dt_s / direction_norm;
    amrex::Real const end_x_m = state.x_m + scale * state.dir_x;
    amrex::Real const end_y_m = state.y_m + scale * state.dir_y;
    amrex::Real const end_z_m = state.z_m + scale * state.dir_z;
    result.path = rrea::ClipRzTransportPath(
        std::hypot(state.x_m, state.y_m),
        std::atan2(state.y_m, state.x_m),
        state.z_m,
        std::hypot(end_x_m, end_y_m),
        std::atan2(end_y_m, end_x_m),
        end_z_m,
        domain_r_lo_m,
        domain_r_hi_m,
        domain_z_lo_m,
        domain_z_hi_m,
        periodic_z);
    result.requested_outer_step_fraction = requested_dt_s / outer_dt_s;
    result.valid = std::isfinite(
            static_cast<double>(result.path.full_length_m))
        && result.path.in_domain_begin_fraction == amrex::Real(0.0);
    return result;
}

// The newborn time budget, extracted header-inline so the smoke asserts the
// PRODUCTION mapping rather than modelling it with its own arithmetic.
//
// A secondary born at elapsed fraction b of the outer step is owed exactly
// the residual (1 - b) of that step -- no time is created or destroyed at
// birth -- and its own elapsed clock advances linearly through that residual
// as its substeps complete.  Both ends clamp: birth can never be negative
// time, and the clock never passes the end of the step.
[[nodiscard]] inline amrex::Real RreaNewbornResidualFraction(
    amrex::Real birth_elapsed_step_fraction) noexcept
{
    return std::min(
        amrex::Real(1.0),
        std::max(
            amrex::Real(0.0),
            amrex::Real(1.0) - birth_elapsed_step_fraction));
}

[[nodiscard]] inline amrex::Real RreaNewbornElapsedStepFraction(
    amrex::Real birth_elapsed_step_fraction,
    amrex::Real residual_fraction,
    std::uint64_t completed_substeps,
    std::uint64_t substeps,
    amrex::Real within_substep_fraction) noexcept
{
    amrex::Real const within = std::min(
        amrex::Real(1.0),
        std::max(amrex::Real(0.0), within_substep_fraction));
    amrex::Real const consumed =
        (static_cast<amrex::Real>(completed_substeps) + within)
        / static_cast<amrex::Real>(substeps);
    return std::min(
        amrex::Real(1.0),
        birth_elapsed_step_fraction + residual_fraction * consumed);
}

// Locate the point where frozen-energy continuous stopping first reaches the
// tracked-particle cutoff. integrated_loss_to(f) returns the non-negative
// cumulative loss from begin_fraction through f. Sharing this limiter keeps
// live and residual transport from booking drag or crossings past the stop.
template <typename IntegratedLossTo>
[[nodiscard]] inline RreaContinuousSegmentLimit
RreaLimitContinuousSegmentAtCutoff(
    amrex::Real begin_fraction,
    amrex::Real end_fraction,
    amrex::Real kinetic_energy_eV,
    amrex::Real cutoff_energy_eV,
    IntegratedLossTo&& integrated_loss_to)
{
    RreaContinuousSegmentLimit result{begin_fraction, false, false};
    if (!std::isfinite(static_cast<double>(begin_fraction))
        || !std::isfinite(static_cast<double>(end_fraction))
        || !std::isfinite(static_cast<double>(kinetic_energy_eV))
        || !std::isfinite(static_cast<double>(cutoff_energy_eV))
        || !(end_fraction >= begin_fraction)
        || !(kinetic_energy_eV >= cutoff_energy_eV)
        || !(cutoff_energy_eV >= amrex::Real(0.0))) {
        return result;
    }
    if (end_fraction == begin_fraction) {
        result.valid = true;
        result.reaches_cutoff = kinetic_energy_eV == cutoff_energy_eV;
        return result;
    }

    amrex::Real const loss_to_cutoff_eV =
        kinetic_energy_eV - cutoff_energy_eV;
    amrex::Real const requested_loss_eV = integrated_loss_to(end_fraction);
    if (!std::isfinite(static_cast<double>(requested_loss_eV))
        || requested_loss_eV < amrex::Real(0.0)) {
        return result;
    }
    result.valid = true;
    if (requested_loss_eV < loss_to_cutoff_eV) {
        result.effective_end_fraction = end_fraction;
        return result;
    }
    result.reaches_cutoff = true;
    if (loss_to_cutoff_eV == amrex::Real(0.0)) {
        return result;
    }

    amrex::Real lo = begin_fraction;
    amrex::Real hi = end_fraction;
    for (int iteration = 0; iteration < 56; ++iteration) {
        amrex::Real const mid = amrex::Real(0.5) * (lo + hi);
        amrex::Real const loss_eV = integrated_loss_to(mid);
        if (!std::isfinite(static_cast<double>(loss_eV))
            || loss_eV < amrex::Real(0.0)) {
            result.valid = false;
            result.reaches_cutoff = false;
            result.effective_end_fraction = begin_fraction;
            return result;
        }
        if (loss_eV < loss_to_cutoff_eV) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    result.effective_end_fraction = amrex::Real(0.5) * (lo + hi);
    return result;
}

[[nodiscard]] inline RreaChargedTerminalDisposition
RreaResolveChargedTerminalDisposition(
    bool positron,
    bool exits_domain,
    bool below_transport_cutoff,
    bool electron_directional_demotion) noexcept
{
    if (exits_domain) {
        return RreaChargedTerminalDisposition::Escape;
    }
    if (positron) {
        return below_transport_cutoff
            ? RreaChargedTerminalDisposition::PositronAnnihilation
            : RreaChargedTerminalDisposition::Survive;
    }
    return (below_transport_cutoff || electron_directional_demotion)
        ? RreaChargedTerminalDisposition::ElectronDemotion
        : RreaChargedTerminalDisposition::Survive;
}

// Composition of the transport loop's terminal decision: an OBSERVED domain
// exit (a microchord actually left, or the endpoint was refused) outranks a
// recorded mid-chord stop, which outranks the endpoint-state resolution.
// Ordering these any other way books stop-time sources -- two 511 keV
// photons, ion pairs -- for a particle that already left the domain.
[[nodiscard]] inline RreaChargedTerminalDisposition
RreaResolveChargedTransportTerminal(
    bool exited_domain,
    bool stopped_mid_chord,
    RreaChargedTerminalDisposition stop_disposition,
    bool positron,
    bool below_transport_cutoff,
    bool electron_directional_demotion) noexcept
{
    if (exited_domain) {
        return RreaChargedTerminalDisposition::Escape;
    }
    if (stopped_mid_chord) {
        return stop_disposition;
    }
    return RreaResolveChargedTerminalDisposition(
        positron,
        /*exits_domain=*/false,
        below_transport_cutoff,
        electron_directional_demotion);
}

// RREA source/secondary positions use the same half-open physical-domain
// convention as AMReX: lower faces belong to the domain and upper faces do
// not.  Keeping this as a pure helper gives the transaction smoke test exact
// endpoint coverage independent of a WarpX geometry instance.
[[nodiscard]] inline bool RreaRzPositionInHalfOpenDomain(
    amrex::Real r_m,
    amrex::Real z_m,
    amrex::Real r_lo_m,
    amrex::Real r_hi_m,
    amrex::Real z_lo_m,
    amrex::Real z_hi_m) noexcept
{
    return std::isfinite(static_cast<double>(r_m))
        && std::isfinite(static_cast<double>(z_m))
        && std::isfinite(static_cast<double>(r_lo_m))
        && std::isfinite(static_cast<double>(r_hi_m))
        && std::isfinite(static_cast<double>(z_lo_m))
        && std::isfinite(static_cast<double>(z_hi_m))
        && r_hi_m > r_lo_m
        && z_hi_m > z_lo_m
        && r_m >= r_lo_m && r_m < r_hi_m
        && z_m >= z_lo_m && z_m < z_hi_m;
}

[[nodiscard]] inline bool RreaCartesianPositionInHalfOpenRzDomain(
    amrex::Real x_m,
    amrex::Real y_m,
    amrex::Real z_m,
    amrex::Real r_lo_m,
    amrex::Real r_hi_m,
    amrex::Real z_lo_m,
    amrex::Real z_hi_m) noexcept
{
    if (!std::isfinite(static_cast<double>(x_m))
        || !std::isfinite(static_cast<double>(y_m))) {
        return false;
    }
    return RreaRzPositionInHalfOpenDomain(
        std::hypot(x_m, y_m), z_m,
        r_lo_m, r_hi_m, z_lo_m, z_hi_m);
}

// Move the running position to a point on the microchord just transported, so
// the next chord is rebuilt from where the particle actually is.
//
// The endpoint is validated before it is kept, because the next chord
// construction starts FROM it.  `exits_domain` uses a 64-eps slack and a strict
// root test, so a chord grazing the outer face can report no exit while landing
// exactly on r_hi, which the half-open domain excludes.  Committing that point
// would gather fields and deposit an entire substep outside the domain,
// silently.  Returning false instead lets the caller turn it into an escape.
//
// Periodic z is wrapped only after the point has been evaluated in its
// unwrapped image, which keeps field and material integration additive.
[[nodiscard]] inline bool RreaAdvanceRunningChargedPosition(
    rrea::RreaTransportPathSegment const& path,
    amrex::Real full_path_fraction,
    amrex::Real r_lo_m,
    amrex::Real r_hi_m,
    amrex::Real z_lo_m,
    amrex::Real z_hi_m,
    amrex::Real& run_x_m,
    amrex::Real& run_y_m,
    amrex::Real& run_z_m) noexcept
{
    amrex::Real x_m = amrex::Real(0.0);
    amrex::Real y_m = amrex::Real(0.0);
    amrex::Real z_m = amrex::Real(0.0);
    path.PointAtFullPathFraction(full_path_fraction, x_m, y_m, z_m);
    amrex::Real const wrapped_z_m = path.WrapZ(z_m);
    if (!RreaCartesianPositionInHalfOpenRzDomain(
            x_m, y_m, wrapped_z_m, r_lo_m, r_hi_m, z_lo_m, z_hi_m)) {
        return false;
    }
    run_x_m = x_m;
    run_y_m = y_m;
    run_z_m = wrapped_z_m;
    return true;
}

struct RreaEndpointEscapePolicy {
    bool escapes = false;
    // A geometric overshoot whose final endpoint is strictly outside is still
    // present in WarpX's post-push particle tile and is removed by its boundary
    // pass.  WarpX's endpoint-only strict comparisons cannot remove an exact
    // excluded upper-face endpoint or a chord that crossed an absorber and
    // re-entered, so RREA owns those invalidations before cap preflight.
    bool invalidate_parent_in_rrea = false;

    [[nodiscard]] bool ParentStillValidAtPreflight() const noexcept
    {
        return escapes && !invalidate_parent_in_rrea;
    }
};

[[nodiscard]] inline RreaEndpointEscapePolicy RreaEndpointEscapePolicyForPath(
    rrea::RreaTransportPathSegment const& path,
    amrex::Real endpoint_r_m,
    amrex::Real endpoint_z_m) noexcept
{
    // Callers pass the raw post-push endpoint. Periodic z is wrapped only for
    // the certified-domain classification; it is never a WarpX absorbing
    // overshoot.  Non-finite endpoints fail closed under RREA ownership.
    if (!std::isfinite(static_cast<double>(endpoint_r_m))
        || !std::isfinite(static_cast<double>(endpoint_z_m))) {
        return RreaEndpointEscapePolicy{/*escapes=*/true,
                                        /*invalidate_parent_in_rrea=*/true};
    }
    amrex::Real const certified_z_m = path.WrapZ(endpoint_z_m);
    bool const endpoint_outside_certified_domain =
        path.OutsideCertifiedDomain(endpoint_r_m, certified_z_m);
    bool const endpoint_strictly_outside_for_warpx =
        endpoint_r_m < path.domain_r_lo_m
        || endpoint_r_m > path.domain_r_hi_m
        || (!path.periodic_z
            && (endpoint_z_m < path.domain_z_lo_m
                || endpoint_z_m > path.domain_z_hi_m));
    bool const escapes = path.exits_domain
        || endpoint_outside_certified_domain;
    return RreaEndpointEscapePolicy{
        escapes,
        escapes && !endpoint_strictly_outside_for_warpx};
}

[[nodiscard]] inline RreaEndpointEscapePolicy
RreaClippedTransportEscapePolicyForPath(
    rrea::RreaTransportPathSegment const& path,
    amrex::Real endpoint_r_m,
    amrex::Real endpoint_z_m) noexcept
{
    auto policy = RreaEndpointEscapePolicyForPath(
        path, endpoint_r_m, endpoint_z_m);
    // Native transport writes the clipped endpoint before WarpX's absorbing
    // boundary pass; RREA owns every recorded escape.
    if (policy.escapes) {
        policy.invalidate_parent_in_rrea = true;
    }
    return policy;
}

struct RreaChargedBoundaryCrossing {
    bool valid = false;
    amrex::Real final_microchord_start_r_m = 0.0;
    amrex::Real final_microchord_start_z_m = 0.0;
    amrex::Real crossing_r_m = 0.0;
    amrex::Real crossing_z_m = 0.0;
    std::uint32_t faces = 0U;
};

// Recover the exact absorbing-face point from the final transport microchord.
// The running position can remain at the preceding interior point when the
// half-open upper face refuses a commit, so it is not an escape endpoint.
// Face classification uses the chord's outward direction; this distinguishes
// a radial exit grazing along z_lo from a true two-face corner crossing.
[[nodiscard]] inline RreaChargedBoundaryCrossing
RreaChargedBoundaryCrossingForPath(
    rrea::RreaTransportPathSegment const& path) noexcept
{
    RreaChargedBoundaryCrossing result;
    amrex::Real crossing_x = amrex::Real(0.0);
    amrex::Real crossing_y = amrex::Real(0.0);
    path.PointAtFullPathFraction(
        path.in_domain_end_fraction,
        crossing_x,
        crossing_y,
        result.crossing_z_m);
    result.crossing_z_m = path.WrapZ(result.crossing_z_m);
    result.crossing_r_m = std::hypot(crossing_x, crossing_y);
    result.final_microchord_start_r_m = std::hypot(
        path.start_x_m, path.start_y_m);
    result.final_microchord_start_z_m = path.WrapZ(path.start_z_m);

    amrex::Real const scale = std::max(
        amrex::Real(1.0),
        std::max(
            path.domain_r_hi_m,
            std::max(
                std::abs(path.domain_z_lo_m),
                std::abs(path.domain_z_hi_m))));
    amrex::Real const tol = amrex::Real(512.0)
        * std::numeric_limits<amrex::Real>::epsilon() * scale;
    bool const on_r_hi = std::abs(
        result.crossing_r_m - path.domain_r_hi_m) <= tol;
    bool const on_z_lo = !path.periodic_z && std::abs(
        result.crossing_z_m - path.domain_z_lo_m) <= tol;
    bool const on_z_hi = !path.periodic_z && std::abs(
        result.crossing_z_m - path.domain_z_hi_m) <= tol;
    amrex::Real const dx = path.end_x_m - path.start_x_m;
    amrex::Real const dy = path.end_y_m - path.start_y_m;
    amrex::Real const dz = path.end_z_m - path.start_z_m;
    amrex::Real const outward_r_numerator = crossing_x * dx + crossing_y * dy;

    if (on_r_hi && outward_r_numerator > -tol * scale) {
        result.faces |= rrea::kEscapeFaceRHi;
        result.crossing_r_m = path.domain_r_hi_m;
    }
    if (on_z_lo && dz < amrex::Real(0.0)) {
        result.faces |= rrea::kEscapeFaceZLo;
        result.crossing_z_m = path.domain_z_lo_m;
    }
    if (on_z_hi && dz > amrex::Real(0.0)) {
        result.faces |= rrea::kEscapeFaceZHi;
        result.crossing_z_m = path.domain_z_hi_m;
    }

    // A zero-length/exact-face invalidation has no directional derivative.
    // Only excluded upper faces qualify for that fallback; z_lo is owned by
    // the half-open domain and therefore requires an actual outward chord.
    if (result.faces == 0U) {
        if (on_r_hi) {
            result.faces |= rrea::kEscapeFaceRHi;
            result.crossing_r_m = path.domain_r_hi_m;
        }
        if (on_z_hi) {
            result.faces |= rrea::kEscapeFaceZHi;
            result.crossing_z_m = path.domain_z_hi_m;
        }
    }
    result.valid = result.faces != 0U
        && std::isfinite(static_cast<double>(
            result.final_microchord_start_r_m))
        && std::isfinite(static_cast<double>(
            result.final_microchord_start_z_m))
        && std::isfinite(static_cast<double>(result.crossing_r_m))
        && std::isfinite(static_cast<double>(result.crossing_z_m));
    return result;
}

enum class RreaMaterialSourceKind : int {
    IonizationEvent,
    LowEnergyElectron,
    PositiveIon,
};

// Ion pairs made by energy that never leaves the interaction point -- a
// sub-cutoff recoil, a shell binding energy, a thermalized residual kinetic
// energy, a sub-cutoff annihilation photon:
//
//     N_pairs = w * E_local / W_air
//
// with W_air the mean energy expended per ion pair in air (ICRU Report 31;
// the runtime value is the table bundle's WAirEvPerIonPair()).  Every created
// pair is then booked with a diagnostic loss of exactly W_air, so the mesh
// energy ledger and the interaction's conservation closure both see w*E_local
// once.  The floor makes a corrupt or zero W_air fail as an under-count
// instead of a division by zero; the loader rejects such a bundle first.
[[nodiscard]] inline amrex::Real RreaIonPairWeightFromLocalEnergy(
    amrex::Real macro_weight,
    amrex::Real local_energy_eV,
    amrex::Real w_air_ev_per_pair) noexcept
{
    if (macro_weight <= amrex::Real(0.0)
        || local_energy_eV <= amrex::Real(0.0)) {
        return amrex::Real(0.0);
    }
    return macro_weight * local_energy_eV
        / std::max(w_air_ev_per_pair, amrex::Real(1.0));
}

// Deferred coupling calls are part of the same transaction as the correlated
// particle payload.  Keeping the typed operation and its complete arguments in
// the group prevents a population-ceiling abort from leaving fluid/event/ledger
// state behind.
// Only side effects whose target is a grid cell, a coupling counter, or the
// transaction validator's expected-flow contract are replayed as ordered
// operations.  Scalar ledger sums are accumulated in place instead (see the
// per-chunk tally in RreaTransportLog.H).
enum class RreaLedgerOpKind : int {
    AccumulateIonizationEvent,
    AccumulateLowEnergyElectron,
    AccumulatePositiveIonSourceOnly,
    RecordComptonInteraction,
    RecordPhotoelectricInteraction,
    RecordPairProduction,
    RecordPositronAnnihilation,
    AccumulateChargedSegment,
    AccumulateChargedEscape,
};

// AccumulateChargedSegment channel bits (op.u0): the terminal disposition of
// the net chord, so the per-step charge balance can name where PIC charge
// left or entered the kinetic population.  Survivors carry no bit.
inline constexpr std::uint64_t kRreaSegmentChannelEscaped = 1;
inline constexpr std::uint64_t kRreaSegmentChannelDemoted = 2;
inline constexpr std::uint64_t kRreaSegmentChannelAnnihilated = 4;
inline constexpr std::uint64_t kRreaSegmentChannelCreated = 8;

struct RreaLedgerEntry {
    RreaLedgerOpKind kind = RreaLedgerOpKind::AccumulateIonizationEvent;
    int lev = 0;
    std::array<amrex::Real, 7> a{};
    amrex::Real electron_total_competing_tau = 0.0;
    std::uint64_t u0 = 0;
    std::uint64_t u1 = 0;
    bool b0 = false;
    bool b1 = false;
};

struct RreaSecondaryParticle {
    RreaSecondarySpecies species = RreaSecondarySpecies::Electron;
    amrex::Real x_m = 0.0;
    amrex::Real y_m = 0.0;
    amrex::Real z_m = 0.0;
    amrex::Real dir_x = 0.0;
    amrex::Real dir_y = 0.0;
    amrex::Real dir_z = 1.0;
    amrex::Real weight = 0.0;
    amrex::Real kinetic_or_photon_energy_eV = 0.0;
    // Elapsed outer-step fraction at creation.  A sampled geometric fraction
    // on a fixed WarpX chord is converted to this time coordinate before the
    // payload is formed.  The transient FIFO advances the remaining
    // (1-f)*dt, including discrete hazards, before any survivor is committed.
    // Conservation groups retain these immutable birth values.
    amrex::Real birth_elapsed_step_fraction = 1.0;
    // RNG lineage is rooted at the original tracked AMReX particle.  Every
    // transient descendant keeps the root key/species and receives a distinct
    // 14-bit lineage.  Segment, event ordinal and inner draw are packed by the
    // checked helper below; overflow rejects the uncommitted transaction.
    std::uint64_t root_rng_particle_id = 0;
    std::uint64_t transient_lineage = 0;
    RreaSecondarySpecies root_species = RreaSecondarySpecies::Electron;
};

struct RreaTransientRngIndex {
    bool valid = false;
    std::uint64_t value = 0;
};

inline constexpr std::uint64_t kRreaTransientLineageCapacity =
    std::uint64_t{1} << 14;
inline constexpr std::uint64_t kRreaTransientSegmentCapacity =
    std::uint64_t{1} << 14;
inline constexpr std::uint64_t kRreaTransientEventCapacity =
    std::uint64_t{1} << 13;
inline constexpr std::uint64_t kRreaTransientInnerCapacity =
    std::uint64_t{1} << 8;
inline constexpr std::uint64_t kRreaLiveContinuationLineage = 1U;

// Collision-free 52-bit interaction-index layout for repeated hazards and the
// pre-commit transient work graph:
//   bit 51       newborn marker
//   bits 49..50  original tracked-root species
//   bits 35..48  14-bit transient lineage
//   bits 21..34  14-bit segment/substep
//   bits  8..20  13-bit event ordinal
//   bits  0.. 7  8-bit inner draw
// Live-parent continuations reserve lineage 1 (lineage 0 is not used there),
// so an electron continuation cannot alias another packed subcycle index on
// the final-state channels. The newborn marker already
// separates transient lineages from every live-parent index.
[[nodiscard]] inline RreaTransientRngIndex RreaTryPackTransientRngIndex(
    bool newborn,
    RreaSecondarySpecies root_species,
    std::uint64_t transient_lineage,
    std::uint64_t segment,
    std::uint64_t event_ordinal,
    std::uint64_t inner_draw) noexcept
{
    auto const species = static_cast<std::uint64_t>(
        static_cast<int>(root_species));
    if (species >= 3U || transient_lineage >= kRreaTransientLineageCapacity
        || segment >= kRreaTransientSegmentCapacity
        || event_ordinal >= kRreaTransientEventCapacity
        || inner_draw >= kRreaTransientInnerCapacity) {
        return {};
    }
    return {true,
        (static_cast<std::uint64_t>(newborn) << 51)
        | (species << 49)
        | (transient_lineage << 35)
        | (segment << 21)
        | (event_ordinal << 8)
        | inner_draw};
}

// The first event uses the segment index; later live-parent hazards use the
// disjoint packed namespace.
[[nodiscard]] inline RreaTransientRngIndex RreaTryLiveHazardRngIndex(
    RreaSecondarySpecies root_species,
    std::uint64_t segment,
    std::uint64_t event_ordinal) noexcept
{
    auto const species = static_cast<std::uint64_t>(
        static_cast<int>(root_species));
    if (species >= 3U) {
        return {};
    }
    if (event_ordinal == 0U) {
        if (segment >= kRreaTransientSegmentCapacity) {
            return {};
        }
        return {true, segment};
    }
    return RreaTryPackTransientRngIndex(
        false, root_species, kRreaLiveContinuationLineage,
        segment, event_ordinal, 0U);
}

struct RreaMaterialSource {
    RreaMaterialSourceKind kind = RreaMaterialSourceKind::IonizationEvent;
    int level = 0;
    amrex::Real r_m = 0.0;
    amrex::Real z_m = 0.0;
    amrex::Real event_weight = 0.0;
    amrex::Real secondary_energy_eV = 0.0;
    amrex::Real energy_loss_eV = 0.0;
};

// Per-interaction conservation contract.  Parent energies are unweighted
// kinetic (charged particles) or total photon energies; particle and material
// payloads carry their own macro weights.  Local and escaped energies are
// already macro-weighted.  The validator below reconstructs all remaining
// charge and energy terms from the transaction payload, including electron
// rest mass and target electrons removed by ionization/annihilation.
enum class RreaConservationClosureMode : int {
    Missing,
    Detailed,
    ExplicitZero,
};

struct RreaInteractionConservationClosure {
    RreaConservationClosureMode mode = RreaConservationClosureMode::Missing;
    amrex::Real parent_weight = 0.0;
    amrex::Real parent_energy_before_eV = 0.0;
    amrex::Real parent_energy_after_eV = 0.0;
    bool parent_survives = false;
    amrex::Real local_recoil_binding_subcutoff_energy_eV = 0.0;
    amrex::Real escaped_energy_eV = 0.0;
    amrex::Real escaped_charge_e = 0.0;
};

// The material-source record is the validation authority while the paired
// ledger operation preserves deterministic replay order.  Require an exact
// one-to-one copy so neither representation can be silently omitted or
// redirected before commit.
[[nodiscard]] inline bool RreaMaterialSourceMatchesLedgerEntry(
    RreaMaterialSource const& source,
    RreaLedgerEntry const& op) noexcept
{
    if (source.level != op.lev
        || source.r_m != op.a[0]
        || source.z_m != op.a[1]
        || source.event_weight != op.a[2]) {
        return false;
    }
    switch (source.kind) {
    case RreaMaterialSourceKind::IonizationEvent:
        return op.kind == RreaLedgerOpKind::AccumulateIonizationEvent
            && source.secondary_energy_eV == op.a[3]
            && source.energy_loss_eV == op.a[4];
    case RreaMaterialSourceKind::LowEnergyElectron:
        return op.kind == RreaLedgerOpKind::AccumulateLowEnergyElectron
            && source.energy_loss_eV == op.a[3];
    case RreaMaterialSourceKind::PositiveIon:
        return op.kind == RreaLedgerOpKind::AccumulatePositiveIonSourceOnly
            && source.energy_loss_eV == op.a[3];
    }
    return false;
}

// One physical interaction is committed as a unit.  Population-control code
// preflights all particles in this group before applying any material source,
// parent removal, diagnostic, or particle insertion.
struct RreaSecondaryGroup {
    std::uint64_t parent_key = 0;
    RreaSecondaryProcess process = RreaSecondaryProcess::Seed;
    bool remove_parent = false;
    // A transient parent exists only in the pre-commit FIFO.  Its terminal
    // removal participates in the interaction conservation closure but must
    // never decrement a native WarpX population or collide with removal of the
    // original tracked root.
    bool parent_is_transient = false;
    // True only for a parent that remains valid in the native container until
    // WarpX applies an absorbing boundary after this hook.  Parents invalidated
    // while assembling a terminal interaction are already absent from the
    // valid-particle count used by preflight.
    bool parent_still_valid_at_preflight = false;
    RreaSecondarySpecies parent_species = RreaSecondarySpecies::Electron;
    std::vector<RreaSecondaryParticle> particles;
    std::vector<RreaMaterialSource> material_sources;
    std::vector<RreaLedgerEntry> ledger_entries;
    RreaInteractionConservationClosure conservation;

    [[nodiscard]] std::uint64_t ParticleCount(
        RreaSecondarySpecies species) const noexcept
    {
        return static_cast<std::uint64_t>(std::count_if(
            particles.begin(), particles.end(),
            [species](RreaSecondaryParticle const& particle) {
                return particle.species == species;
            }));
    }
};

[[nodiscard]] inline bool RreaSecondaryGroupRemovesNativeParent(
    RreaSecondaryGroup const& group) noexcept
{
    return group.remove_parent && !group.parent_is_transient;
}

enum class RreaTrackedChargedSecondaryDisposition : int {
    Track,
    DemoteToFluid,
    Reject,
};

// Photon transport is certified to 100 GeV while charged transport ends at
// 10 GeV.  Classify every charged photon product before the deferred batch is
// committed: exact charged endpoints are trackable, a finite nonnegative value
// below the floor belongs to the low-energy fluid, and an invalid/above-range
// value makes the complete collective transaction fail closed.
[[nodiscard]] inline RreaTrackedChargedSecondaryDisposition
RreaClassifyTrackedChargedSecondaryEnergy(
    amrex::Real kinetic_energy_eV,
    amrex::Real certified_min_eV,
    amrex::Real certified_max_eV) noexcept
{
    bool const valid_range = std::isfinite(static_cast<double>(certified_min_eV))
        && std::isfinite(static_cast<double>(certified_max_eV))
        && certified_min_eV >= amrex::Real(0.0)
        && certified_max_eV >= certified_min_eV;
    if (!valid_range
        || !std::isfinite(static_cast<double>(kinetic_energy_eV))
        || kinetic_energy_eV < amrex::Real(0.0)
        || kinetic_energy_eV > certified_max_eV) {
        return RreaTrackedChargedSecondaryDisposition::Reject;
    }
    return kinetic_energy_eV < certified_min_eV
        ? RreaTrackedChargedSecondaryDisposition::DemoteToFluid
        : RreaTrackedChargedSecondaryDisposition::Track;
}

[[nodiscard]] inline bool RreaProcessHasPhotonParent(
    RreaSecondaryProcess process) noexcept
{
    return process == RreaSecondaryProcess::Compton
        || process == RreaSecondaryProcess::Photoelectric
        || process == RreaSecondaryProcess::PairNuclear
        || process == RreaSecondaryProcess::PairTriplet;
}

[[nodiscard]] inline std::uint64_t
RreaCountInvalidTrackedPhotonChargedSecondaries(
    RreaSecondaryGroup const& group,
    amrex::Real electron_min_eV,
    amrex::Real electron_max_eV,
    amrex::Real positron_min_eV,
    amrex::Real positron_max_eV) noexcept
{
    if (!RreaProcessHasPhotonParent(group.process)) {
        return 0U;
    }
    std::uint64_t invalid = 0U;
    for (auto const& particle : group.particles) {
        if (particle.species == RreaSecondarySpecies::Electron) {
            invalid += RreaClassifyTrackedChargedSecondaryEnergy(
                particle.kinetic_or_photon_energy_eV,
                electron_min_eV,
                electron_max_eV)
                != RreaTrackedChargedSecondaryDisposition::Track;
        } else if (particle.species == RreaSecondarySpecies::Positron) {
            invalid += RreaClassifyTrackedChargedSecondaryEnergy(
                particle.kinetic_or_photon_energy_eV,
                positron_min_eV,
                positron_max_eV)
                != RreaTrackedChargedSecondaryDisposition::Track;
        }
    }
    return invalid;
}

struct RreaInteractionConservationResult {
    bool required = false;
    bool present = false;
    bool finite = false;
    bool valid = false;
    amrex::Real charge_residual_e = 0.0;
    amrex::Real charge_tolerance_e = 0.0;
    amrex::Real energy_residual_eV = 0.0;
    amrex::Real energy_tolerance_eV = 0.0;
};

[[nodiscard]] inline bool RreaProcessRequiresConservationClosure(
    RreaSecondaryProcess process) noexcept
{
    switch (process) {
    case RreaSecondaryProcess::HardMoller:
    case RreaSecondaryProcess::HardBhabha:
    case RreaSecondaryProcess::Bremsstrahlung:
    case RreaSecondaryProcess::Compton:
    case RreaSecondaryProcess::Photoelectric:
    case RreaSecondaryProcess::PairNuclear:
    case RreaSecondaryProcess::PairTriplet:
    case RreaSecondaryProcess::PositronAnnihilation:
    case RreaSecondaryProcess::PhotonEscape:
    case RreaSecondaryProcess::ParticleEscape:
    case RreaSecondaryProcess::LowEnergyDemotion:
        return true;
    case RreaSecondaryProcess::ContinuousTransport:
    case RreaSecondaryProcess::Seed:
        return false;
    }
    return true;
}

[[nodiscard]] inline amrex::Real RreaSpeciesChargeE(
    RreaSecondarySpecies species) noexcept
{
    switch (species) {
    case RreaSecondarySpecies::Electron: return amrex::Real(-1.0);
    case RreaSecondarySpecies::Positron: return amrex::Real(1.0);
    case RreaSecondarySpecies::Photon: return amrex::Real(0.0);
    }
    return std::numeric_limits<amrex::Real>::quiet_NaN();
}

[[nodiscard]] inline amrex::Real RreaSpeciesRestEnergyEv(
    RreaSecondarySpecies species) noexcept
{
    static constexpr amrex::Real electron_rest_energy_eV = kRreaElectronRestEnergyEv;
    return species == RreaSecondarySpecies::Photon
        ? amrex::Real(0.0)
        : electron_rest_energy_eV;
}

// Where and when one discrete interaction happened, in the form every final
// state needs it: the Cartesian point a secondary is born at, the cylindrical
// (r, z) a material source is deposited at, and the elapsed fraction of the
// outer step at which the event occurred -- the residual a newborn is owed.
struct RreaInteractionSite {
    amrex::Real x_m = amrex::Real(0.0);
    amrex::Real y_m = amrex::Real(0.0);
    amrex::Real z_m = amrex::Real(0.0);
    amrex::Real r_m = amrex::Real(0.0);
    amrex::Real elapsed_step_fraction = amrex::Real(1.0);
};

// Photon-energy bookkeeping is a flow ledger, not an interaction-energy
// proxy.  Energy that remains in an explicitly transported charged particle
// (including newly-created pair rest mass) belongs to the charged-transfer
// flow; recoil, binding, and kinetic energy handed to the low-energy material
// model belongs to the local-deposit flow.  Keep the partition as a pure
// helper so production call sites and endpoint fixtures use the same exact
// cutoff comparisons.
struct RreaPhotonEnergyFlowSplit {
    amrex::Real charged_kinetic_rest_mass_eV = amrex::Real(0.0);
    amrex::Real local_recoil_binding_subcutoff_eV = amrex::Real(0.0);

    [[nodiscard]] amrex::Real TotalEv() const noexcept
    {
        return charged_kinetic_rest_mass_eV
            + local_recoil_binding_subcutoff_eV;
    }

    [[nodiscard]] bool Valid() const noexcept
    {
        return std::isfinite(static_cast<double>(
                   charged_kinetic_rest_mass_eV))
            && std::isfinite(static_cast<double>(
                   local_recoil_binding_subcutoff_eV))
            && charged_kinetic_rest_mass_eV >= amrex::Real(0.0)
            && local_recoil_binding_subcutoff_eV >= amrex::Real(0.0);
    }
};

struct RreaPhotonLedgerFlows {
    amrex::Real initial_alive_eV = amrex::Real(0.0);
    amrex::Real external_injected_eV = amrex::Real(0.0);
    amrex::Real brems_created_eV = amrex::Real(0.0);
    amrex::Real annihilation_created_eV = amrex::Real(0.0);
    amrex::Real alive_eV = amrex::Real(0.0);
    amrex::Real escaped_eV = amrex::Real(0.0);
    amrex::Real charged_transfer_eV = amrex::Real(0.0);
    amrex::Real local_deposit_eV = amrex::Real(0.0);

    [[nodiscard]] amrex::Real SourceEv() const noexcept
    {
        return initial_alive_eV + external_injected_eV
            + brems_created_eV + annihilation_created_eV;
    }

    [[nodiscard]] amrex::Real SinkEv() const noexcept
    {
        return alive_eV + escaped_eV
            + charged_transfer_eV + local_deposit_eV;
    }

    [[nodiscard]] amrex::Real ResidualEv() const noexcept
    {
        return SourceEv() - SinkEv();
    }

    [[nodiscard]] amrex::Real RelativeResidual() const noexcept
    {
        return std::abs(ResidualEv())
            / std::max({amrex::Real(1.0), std::abs(SourceEv()),
                        std::abs(SinkEv())});
    }
};

[[nodiscard]] inline RreaPhotonEnergyFlowSplit
RreaComptonPhotonEnergyFlow(
    amrex::Real recoil_electron_energy_eV,
    amrex::Real scattered_photon_energy_eV,
    amrex::Real low_energy_cutoff_eV,
    amrex::Real photon_cutoff_eV) noexcept
{
    RreaPhotonEnergyFlowSplit split;
    amrex::Real const recoil_eV =
        std::max(recoil_electron_energy_eV, amrex::Real(0.0));
    amrex::Real const scattered_eV =
        std::max(scattered_photon_energy_eV, amrex::Real(0.0));
    if (recoil_eV >= low_energy_cutoff_eV) {
        split.charged_kinetic_rest_mass_eV = recoil_eV;
    } else {
        split.local_recoil_binding_subcutoff_eV = recoil_eV;
    }
    if (scattered_eV < photon_cutoff_eV) {
        split.local_recoil_binding_subcutoff_eV += scattered_eV;
    }
    return split;
}

[[nodiscard]] inline RreaPhotonEnergyFlowSplit
RreaPhotoelectricPhotonEnergyFlow(
    amrex::Real photoelectron_energy_eV,
    amrex::Real binding_energy_eV,
    amrex::Real low_energy_cutoff_eV) noexcept
{
    RreaPhotonEnergyFlowSplit split;
    amrex::Real const electron_eV =
        std::max(photoelectron_energy_eV, amrex::Real(0.0));
    amrex::Real const binding_eV =
        std::max(binding_energy_eV, amrex::Real(0.0));
    split.local_recoil_binding_subcutoff_eV = binding_eV;
    if (electron_eV >= low_energy_cutoff_eV) {
        split.charged_kinetic_rest_mass_eV = electron_eV;
    } else {
        split.local_recoil_binding_subcutoff_eV += electron_eV;
    }
    return split;
}

[[nodiscard]] inline RreaPhotonEnergyFlowSplit
RreaPairPhotonEnergyFlow(
    amrex::Real electron_energy_eV,
    amrex::Real positron_energy_eV,
    amrex::Real recoil_electron_energy_eV,
    amrex::Real final_state_local_deposit_energy_eV,
    amrex::Real low_energy_cutoff_eV,
    bool has_recoil_electron) noexcept
{
    RreaPhotonEnergyFlowSplit split;
    // Pair creation transfers two newly-created lepton rest masses even when a
    // sub-cutoff electron is handed to the fluid or a sub-cutoff positron
    // immediately annihilates.  The triplet recoil electron already existed in
    // the neutral target, so only its kinetic energy is partitioned here.
    split.charged_kinetic_rest_mass_eV = amrex::Real(2.0)
        * RreaSpeciesRestEnergyEv(RreaSecondarySpecies::Electron);
    split.local_recoil_binding_subcutoff_eV =
        std::max(final_state_local_deposit_energy_eV, amrex::Real(0.0));

    auto partition_kinetic = [&](amrex::Real kinetic_energy_eV) {
        amrex::Real const energy_eV =
            std::max(kinetic_energy_eV, amrex::Real(0.0));
        if (energy_eV >= low_energy_cutoff_eV) {
            split.charged_kinetic_rest_mass_eV += energy_eV;
        } else {
            split.local_recoil_binding_subcutoff_eV += energy_eV;
        }
    };
    partition_kinetic(electron_energy_eV);
    partition_kinetic(positron_energy_eV);
    if (has_recoil_electron) {
        partition_kinetic(recoil_electron_energy_eV);
    }
    return split;
}

// Validate one group without mutating runtime state.  Charge is measured in
// elementary-charge macro-weight units and energy in macro-weighted eV.  Each
// IonizationEvent or PositiveIon source consumes one target electron from
// neutral matter; a sub-cutoff IonizationEvent also emits that electron into
// the low-energy fluid.  This makes both target-electron rest mass and the
// low-electron/positive-ion charge pair explicit in the same atomic check.
[[nodiscard]] inline RreaInteractionConservationResult
RreaValidateSecondaryGroupConservation(
    RreaSecondaryGroup const& group,
    amrex::Real low_energy_cutoff_eV) noexcept
{
    RreaInteractionConservationResult result;
    result.required = RreaProcessRequiresConservationClosure(group.process);
    if (!result.required) {
        result.present = true;
        result.finite = true;
        result.valid = true;
        return result;
    }

    auto const& closure = group.conservation;
    result.present = closure.mode != RreaConservationClosureMode::Missing;
    if (!result.present) {
        return result;
    }
    if (closure.mode == RreaConservationClosureMode::ExplicitZero) {
        bool const payload_free = group.particles.empty()
            && group.material_sources.empty();
        bool const exact_zero = group.remove_parent
            && !group.parent_still_valid_at_preflight
            && !closure.parent_survives
            && std::isfinite(static_cast<double>(closure.parent_weight))
            && std::isfinite(static_cast<double>(closure.parent_energy_before_eV))
            && std::isfinite(static_cast<double>(closure.parent_energy_after_eV))
            && std::isfinite(static_cast<double>(
                closure.local_recoil_binding_subcutoff_energy_eV))
            && std::isfinite(static_cast<double>(closure.escaped_energy_eV))
            && std::isfinite(static_cast<double>(closure.escaped_charge_e))
            && closure.parent_weight == amrex::Real(0.0)
            && closure.parent_energy_before_eV == amrex::Real(0.0)
            && closure.parent_energy_after_eV == amrex::Real(0.0)
            && closure.local_recoil_binding_subcutoff_energy_eV
                == amrex::Real(0.0)
            && closure.escaped_energy_eV == amrex::Real(0.0)
            && closure.escaped_charge_e == amrex::Real(0.0);
        result.finite = payload_free && exact_zero;
        result.valid = result.finite;
        return result;
    }
    if (closure.mode != RreaConservationClosureMode::Detailed) {
        return result;
    }

    bool finite = std::isfinite(static_cast<double>(low_energy_cutoff_eV))
        && low_energy_cutoff_eV > amrex::Real(0.0)
        && std::isfinite(static_cast<double>(closure.parent_weight))
        && std::isfinite(static_cast<double>(closure.parent_energy_before_eV))
        && std::isfinite(static_cast<double>(closure.parent_energy_after_eV))
        && std::isfinite(static_cast<double>(
            closure.local_recoil_binding_subcutoff_energy_eV))
        && std::isfinite(static_cast<double>(closure.escaped_energy_eV))
        && std::isfinite(static_cast<double>(closure.escaped_charge_e))
        && closure.parent_weight > amrex::Real(0.0)
        && closure.parent_energy_before_eV >= amrex::Real(0.0)
        && closure.parent_energy_after_eV >= amrex::Real(0.0)
        && closure.local_recoil_binding_subcutoff_energy_eV >= amrex::Real(0.0)
        && closure.escaped_energy_eV >= amrex::Real(0.0)
        && closure.parent_survives == !group.remove_parent;

    amrex::Real const parent_charge_e =
        RreaSpeciesChargeE(group.parent_species) * closure.parent_weight;
    amrex::Real const parent_rest_eV =
        RreaSpeciesRestEnergyEv(group.parent_species);
    amrex::Real input_charge_e = parent_charge_e;
    amrex::Real output_charge_e = closure.parent_survives
        ? parent_charge_e
        : amrex::Real(0.0);
    output_charge_e += closure.escaped_charge_e;
    amrex::Real input_energy_eV = closure.parent_weight
        * (closure.parent_energy_before_eV + parent_rest_eV);
    amrex::Real output_energy_eV = closure.parent_survives
        ? closure.parent_weight
            * (closure.parent_energy_after_eV + parent_rest_eV)
        : amrex::Real(0.0);
    output_energy_eV += closure.local_recoil_binding_subcutoff_energy_eV
        + closure.escaped_energy_eV;
    amrex::Real charge_throughput_e = std::abs(input_charge_e)
        + std::abs(output_charge_e);

    for (auto const& particle : group.particles) {
        finite = finite
            && std::isfinite(static_cast<double>(particle.weight))
            && std::isfinite(static_cast<double>(
                particle.kinetic_or_photon_energy_eV))
            && particle.weight > amrex::Real(0.0)
            && particle.kinetic_or_photon_energy_eV >= amrex::Real(0.0);
        amrex::Real const particle_charge_e =
            RreaSpeciesChargeE(particle.species) * particle.weight;
        output_charge_e += particle_charge_e;
        charge_throughput_e += std::abs(particle_charge_e);
        output_energy_eV += particle.weight
            * (particle.kinetic_or_photon_energy_eV
                + RreaSpeciesRestEnergyEv(particle.species));
    }

    static constexpr amrex::Real electron_rest_energy_eV = kRreaElectronRestEnergyEv;
    for (auto const& source : group.material_sources) {
        finite = finite
            && std::isfinite(static_cast<double>(source.event_weight))
            && std::isfinite(static_cast<double>(source.secondary_energy_eV))
            && source.event_weight > amrex::Real(0.0);
        switch (source.kind) {
        case RreaMaterialSourceKind::IonizationEvent:
            // The neutral target contributes its electron rest energy on the
            // input side and the residual ion contributes +e on output.
            input_energy_eV += source.event_weight * electron_rest_energy_eV;
            output_charge_e += source.event_weight;
            charge_throughput_e += source.event_weight;
            if (source.secondary_energy_eV < low_energy_cutoff_eV) {
                output_charge_e -= source.event_weight;
                charge_throughput_e += source.event_weight;
                output_energy_eV += source.event_weight * electron_rest_energy_eV;
            }
            break;
        case RreaMaterialSourceKind::PositiveIon:
            input_energy_eV += source.event_weight * electron_rest_energy_eV;
            output_charge_e += source.event_weight;
            charge_throughput_e += source.event_weight;
            break;
        case RreaMaterialSourceKind::LowEnergyElectron:
            output_charge_e -= source.event_weight;
            charge_throughput_e += source.event_weight;
            output_energy_eV += source.event_weight * electron_rest_energy_eV;
            break;
        }
    }

    result.finite = finite
        && std::isfinite(static_cast<double>(input_charge_e))
        && std::isfinite(static_cast<double>(output_charge_e))
        && std::isfinite(static_cast<double>(input_energy_eV))
        && std::isfinite(static_cast<double>(output_energy_eV));
    if (!result.finite) {
        return result;
    }
    result.charge_residual_e = output_charge_e - input_charge_e;
    result.energy_residual_eV = output_energy_eV - input_energy_eV;
    amrex::Real constexpr eps64 = std::numeric_limits<double>::epsilon();
    result.charge_tolerance_e = amrex::Real(64.0) * eps64
        * std::max(amrex::Real(1.0), charge_throughput_e);
    result.energy_tolerance_eV = amrex::Real(64.0) * eps64
        * std::max({amrex::Real(1.0), std::abs(input_energy_eV),
                    std::abs(output_energy_eV)});
    result.valid = std::abs(result.charge_residual_e)
            <= result.charge_tolerance_e
        && std::abs(result.energy_residual_eV)
            <= result.energy_tolerance_eV;
    return result;
}

// Pure helper shared by the runtime ceiling preflight and its focused smoke
// test.  A zero ceiling is handled by the caller as "disabled".  General
// production ceilings are inclusive; C&D certification sets reject_equality
// because its statistical contract invalidates any ceiling hit.
[[nodiscard]] inline bool RreaStrictPopulationProjection(
    std::uint64_t current,
    std::uint64_t removals,
    std::uint64_t insertions,
    std::uint64_t ceiling,
    std::uint64_t& projected,
    bool reject_equality = false) noexcept
{
    if (removals > current) {
        return false;
    }
    std::uint64_t const survivors = current - removals;
    if (insertions > std::numeric_limits<std::uint64_t>::max() - survivors) {
        projected = std::numeric_limits<std::uint64_t>::max();
        return false;
    }
    projected = survivors + insertions;
    return reject_equality ? projected < ceiling : projected <= ceiling;
}

// State-based low-energy demotion is independent of the stochastic particle
// direction, so reevaluation is idempotent and timestep robust.
[[nodiscard]] inline bool RreaLowEnergyNonrunawayFieldMagnitudeGate(
    amrex::Real kinetic_energy_eV,
    amrex::Real hard_moller_threshold_eV,
    amrex::Real field_magnitude_v_per_m,
    amrex::Real runaway_or_drag_threshold_v_per_m) noexcept
{
    return hard_moller_threshold_eV > amrex::Real(0.0)
        && kinetic_energy_eV <= hard_moller_threshold_eV
        && field_magnitude_v_per_m <= runaway_or_drag_threshold_v_per_m;
}

struct RreaElectronStatePolicyDecision {
    bool below_transport_cutoff = false;
    bool low_energy_nonrunaway = false;

    [[nodiscard]] bool Demote() const noexcept
    {
        return below_transport_cutoff || low_energy_nonrunaway;
    }
};

// One state-policy kernel for live and newborn electrons.  Identical local
// state must produce the same kinetic/fluid disposition regardless of origin.
[[nodiscard]] inline RreaElectronStatePolicyDecision
RreaEvaluateElectronStatePolicy(
    amrex::Real kinetic_energy_eV,
    amrex::Real transport_cutoff_eV,
    amrex::Real hard_moller_threshold_eV,
    amrex::Real field_magnitude_v_per_m,
    amrex::Real ensemble_mean_drag_v_per_m) noexcept
{
    return RreaElectronStatePolicyDecision{
        kinetic_energy_eV < transport_cutoff_eV,
        RreaLowEnergyNonrunawayFieldMagnitudeGate(
            kinetic_energy_eV,
            hard_moller_threshold_eV,
            field_magnitude_v_per_m,
            ensemble_mean_drag_v_per_m)};
}

// A signed plane flux can cancel recrossings only when its particle selection
// is independent of crossing direction.  Classify a tracked electron as
// locally capable of runaway acceleration from |E| versus the same continuous
// drag used by transport; projecting E onto the instantaneous chord would
// count the upward crossing but discard the corresponding downward recrossing.
[[nodiscard]] inline bool RreaLocalRunawayFieldMagnitudeGate(
    amrex::Real er_v_per_m,
    amrex::Real ez_v_per_m,
    amrex::Real continuous_drag_eV_per_m) noexcept
{
    amrex::Real const field_magnitude_v_per_m =
        std::hypot(er_v_per_m, ez_v_per_m);
    return std::isfinite(static_cast<double>(field_magnitude_v_per_m))
        && std::isfinite(static_cast<double>(continuous_drag_eV_per_m))
        && continuous_drag_eV_per_m >= amrex::Real(0.0)
        && field_magnitude_v_per_m > continuous_drag_eV_per_m;
}

// Exact hard-secondary phase-space gates.  The rate tables carry an explicit
// zero at the threshold and become authoritative immediately above it; do not
// add numerical guard bands or a surviving-parent cutoff here.
[[nodiscard]] inline bool RreaHardMollerChannelOpen(
    amrex::Real primary_energy_eV,
    amrex::Real hard_secondary_threshold_eV) noexcept
{
    return hard_secondary_threshold_eV > amrex::Real(0.0)
        && primary_energy_eV
            > amrex::Real(2.0) * hard_secondary_threshold_eV;
}

[[nodiscard]] inline bool RreaHardBhabhaChannelOpen(
    amrex::Real primary_energy_eV,
    amrex::Real hard_secondary_threshold_eV) noexcept
{
    return !(hard_secondary_threshold_eV > amrex::Real(0.0)
        && primary_energy_eV <= hard_secondary_threshold_eV);
}

struct RreaCompetingHazardSelection {
    bool valid = false;
    bool event = false;
    int channel = -1;
    amrex::Real sampled_optical_depth = amrex::Real(0.0);
    amrex::Real total_optical_depth = amrex::Real(0.0);
};

struct RreaChargedEventDepthSegments {
    bool valid = false;
    amrex::Real before_begin_fraction = amrex::Real(0.0);
    amrex::Real before_end_fraction = amrex::Real(0.0);
    amrex::Real after_begin_fraction = amrex::Real(0.0);
    amrex::Real after_end_fraction = amrex::Real(0.0);
};

// Exact geometric operator split around one sampled charged event.  Both
// pieces share the event fraction, leaving neither a gap nor an overlap.
[[nodiscard]] inline RreaChargedEventDepthSegments
RreaSplitChargedSegmentAtEvent(
    amrex::Real begin_fraction,
    amrex::Real event_fraction,
    amrex::Real end_fraction) noexcept
{
    RreaChargedEventDepthSegments result;
    if (!std::isfinite(static_cast<double>(begin_fraction))
        || !std::isfinite(static_cast<double>(event_fraction))
        || !std::isfinite(static_cast<double>(end_fraction))
        || !(end_fraction > begin_fraction)
        || event_fraction < begin_fraction
        || event_fraction > end_fraction) {
        return result;
    }
    result.valid = true;
    result.before_begin_fraction = begin_fraction;
    result.before_end_fraction = event_fraction;
    result.after_begin_fraction = event_fraction;
    result.after_end_fraction = end_fraction;
    return result;
}

// Select one earliest discrete event from up to three competing integrated
// hazards.  The first draw is a true exponential optical depth against the
// total rate; the second selects the winning channel by its hazard share.  A
// caller may then invert sampled_optical_depth against the summed path rate.
[[nodiscard]] inline RreaCompetingHazardSelection RreaSelectCompetingHazard(
    std::array<amrex::Real, 3> const& channel_optical_depths,
    int channel_count,
    amrex::Real optical_depth_draw,
    amrex::Real channel_draw) noexcept
{
    RreaCompetingHazardSelection result;
    if (channel_count < 1 || channel_count > 3
        || !std::isfinite(static_cast<double>(optical_depth_draw))
        || !std::isfinite(static_cast<double>(channel_draw))) {
        return result;
    }
    for (int index = 0; index < channel_count; ++index) {
        amrex::Real const tau = channel_optical_depths[static_cast<std::size_t>(index)];
        if (!std::isfinite(static_cast<double>(tau)) || tau < amrex::Real(0.0)) {
            return result;
        }
        result.total_optical_depth += tau;
    }
    result.valid = std::isfinite(static_cast<double>(result.total_optical_depth));
    if (!result.valid) {
        return result;
    }
    optical_depth_draw = std::max(
        amrex::Real(0.0), std::min(amrex::Real(1.0), optical_depth_draw));
    result.sampled_optical_depth = -std::log(std::max(
        amrex::Real(1.0e-30), amrex::Real(1.0) - optical_depth_draw));
    if (!(result.sampled_optical_depth < result.total_optical_depth)) {
        return result;
    }
    result.event = true;
    channel_draw = std::max(
        amrex::Real(0.0), std::min(amrex::Real(1.0), channel_draw));
    amrex::Real const target = channel_draw * result.total_optical_depth;
    amrex::Real cumulative = amrex::Real(0.0);
    int last_open_channel = -1;
    for (int index = 0; index < channel_count; ++index) {
        amrex::Real const tau = channel_optical_depths[static_cast<std::size_t>(index)];
        if (tau > amrex::Real(0.0)) {
            last_open_channel = index;
        }
        cumulative += tau;
        if (target < cumulative) {
            result.channel = index;
            return result;
        }
    }
    // channel_draw can round to exactly one; select the final nonzero channel.
    result.channel = last_open_channel;
    result.valid = result.channel >= 0;
    result.event = result.valid;
    return result;
}

struct RreaRemainingSegmentHazard {
    RreaCompetingHazardSelection selection;
    RreaChargedEventDepthSegments segments;
    amrex::Real event_fraction = amrex::Real(0.0);
};

// Shared, side-effect-free core of every live/newborn remaining-segment
// hazard loop.  Species policies still own rate construction, RNG keys,
// continuous transport, final states, ledgers, and continuation decisions.
// This helper only selects one exponential competing hazard and resolves its
// optical depth to the exact geometric event split.
template <typename OpticalDepthToFraction>
[[nodiscard]] inline RreaRemainingSegmentHazard
RreaResolveRemainingSegmentHazard(
    std::array<amrex::Real, 3> const& channel_optical_depths,
    int channel_count,
    amrex::Real begin_fraction,
    amrex::Real end_fraction,
    amrex::Real optical_depth_draw,
    amrex::Real channel_draw,
    OpticalDepthToFraction&& optical_depth_to_fraction)
{
    RreaRemainingSegmentHazard result;
    result.selection = RreaSelectCompetingHazard(
        channel_optical_depths,
        channel_count,
        optical_depth_draw,
        channel_draw);
    if (!result.selection.valid || !result.selection.event) {
        return result;
    }
    result.event_fraction =
        optical_depth_to_fraction(result.selection.sampled_optical_depth);
    result.segments = RreaSplitChargedSegmentAtEvent(
        begin_fraction, result.event_fraction, end_fraction);
    return result;
}

[[nodiscard]] inline amrex::Real RreaKineticEnergyEvFromProperVelocity(
    amrex::Real ux,
    amrex::Real uy,
    amrex::Real uz) noexcept
{
    constexpr amrex::Real rest_energy_eV = kRreaElectronRestEnergyEv;
    constexpr amrex::Real c_m_per_s = kRreaSpeedOfLightMPerS;
    amrex::Real const u2 = ux * ux + uy * uy + uz * uz;
    amrex::Real const gamma = std::sqrt(
        amrex::Real(1.0) + u2 / (c_m_per_s * c_m_per_s));
    return u2 * rest_energy_eV
        / ((c_m_per_s * c_m_per_s) * (gamma + amrex::Real(1.0)));
}

// Work in eV for one piece of an RZ chord.  charge_sign is -1 for electrons
// and +1 for positrons; volts times elementary-charge units map directly to eV.
[[nodiscard]] inline amrex::Real RreaFieldWorkIncrementEv(
    amrex::Real charge_sign,
    amrex::Real er_v_per_m,
    amrex::Real ez_v_per_m,
    amrex::Real radial_path_direction,
    amrex::Real z_path_direction,
    amrex::Real length_m) noexcept
{
    return charge_sign
        * (er_v_per_m * radial_path_direction
            + ez_v_per_m * z_path_direction)
        * length_m;
}

// A tracked brems final-state CDF is authoritative through its physical hard
// tip.  Clamp only roundoff beyond the parent's kinetic energy; a sub-cutoff
// charged survivor is handled by the normal demotion/annihilation path.
[[nodiscard]] inline amrex::Real RreaClampTrackedBremsPhotonEnergy(
    amrex::Real sampled_photon_energy_eV,
    amrex::Real parent_kinetic_energy_eV) noexcept
{
    return std::min(
        std::max(sampled_photon_energy_eV, amrex::Real(0.0)),
        std::max(parent_kinetic_energy_eV, amrex::Real(0.0)));
}

[[nodiscard]] inline bool RreaElectronHardBremsChannelOpen(
    amrex::Real parent_kinetic_energy_eV,
    amrex::Real photon_cutoff_eV) noexcept
{
    return std::isfinite(static_cast<double>(parent_kinetic_energy_eV))
        && std::isfinite(static_cast<double>(photon_cutoff_eV))
        && parent_kinetic_energy_eV > photon_cutoff_eV;
}

struct RreaBremsEnergyDisposition {
    amrex::Real photon_energy_eV = 0.0;
    amrex::Real parent_energy_after_eV = 0.0;
    amrex::Real local_deposit_energy_eV = 0.0;
    bool commit_photon = false;
};

[[nodiscard]] inline RreaBremsEnergyDisposition
RreaClassifyBremsEnergyDisposition(
    amrex::Real parent_kinetic_energy_eV,
    amrex::Real sampled_photon_energy_eV,
    amrex::Real photon_cutoff_eV) noexcept
{
    RreaBremsEnergyDisposition result;
    result.photon_energy_eV = RreaClampTrackedBremsPhotonEnergy(
        sampled_photon_energy_eV, parent_kinetic_energy_eV);
    result.parent_energy_after_eV = std::max(
        amrex::Real(0.0),
        std::max(parent_kinetic_energy_eV, amrex::Real(0.0))
            - result.photon_energy_eV);
    bool const tracked_photon =
        std::isfinite(static_cast<double>(photon_cutoff_eV))
        && photon_cutoff_eV >= amrex::Real(0.0)
        && result.photon_energy_eV >= photon_cutoff_eV;
    result.commit_photon = tracked_photon;
    if (!tracked_photon) {
        result.local_deposit_energy_eV = result.photon_energy_eV;
    }
    return result;
}

struct RreaParticleInteractionConfig {
    amrex::Real density_ratio = 1.0;
    amrex::Real max_optical_depth_per_substep = 0.05;
    std::uint64_t max_subcycles = 256;
    bool strict_transport_guards = true;
    bool debug_disable_hard_moller_secondaries = false;
    // Transport owns the interleaved field, drag, discrete-hazard, and GS
    // sequence; newborns advance through the residual time of their birth step.
    // Substep drivers beyond the hazard-tau refinement: cap GS collisions
    // interleaved per substep (the sampler is effectively isotropic past the
    // total cap, so finer interleaving adds nothing), and cap the field
    // impulse + drag energy turnover per substep at a fraction of the current
    // kinetic energy with an absolute floor.  Driver counts clamp silently at
    // max_subcycles: unlike an optical-depth violation this is benign
    // refinement, so strict-guard semantics are unchanged.
    amrex::Real transport_max_gs_collisions_per_substep = 32.0;
    amrex::Real transport_gs_total_collision_cap = 1024.0;
    amrex::Real transport_max_field_impulse_fraction = 0.1;
    amrex::Real transport_field_impulse_floor_eV = 1.0e4;
    std::uint64_t rng_seed = 0;
};

[[nodiscard]] inline bool RreaTransportSubcycleControlsValid(
    RreaParticleInteractionConfig const& config,
    bool require_strict_guards = true) noexcept
{
    return std::isfinite(static_cast<double>(
               config.max_optical_depth_per_substep))
        && config.max_optical_depth_per_substep > amrex::Real(0.0)
        && config.max_subcycles > 0U
        && (!require_strict_guards || config.strict_transport_guards);
}

class RreaParticleInteraction {
public:
    explicit RreaParticleInteraction(RreaParticleInteractionConfig config);

    void Apply(
        WarpX& warpx,
        RreaWarpXCoupling& coupling,
        rrea::RreaInteractionTables const& tables,
        int step,
        amrex::Real time_s,
        amrex::Real dt_s);

private:
    RreaParticleInteractionConfig m_config;
};

}  // namespace rrea::warpx

#endif
