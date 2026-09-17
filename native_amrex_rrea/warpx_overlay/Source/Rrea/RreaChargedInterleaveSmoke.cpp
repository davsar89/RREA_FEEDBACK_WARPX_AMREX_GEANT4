// Behavioural tests of the charged-interleave primitives in
// RreaChargedInterleave.H -- the engine's own functions, not re-derived copies
// of the production kinematics.

#include "RreaChargedInterleave.H"
#include "rrea/RreaSmokeRequire.H"
#include "RreaParticleInteraction.H"
#include "rrea/RreaLorentzPush.H"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr double kMeC2eV = 510998.95000;  // the engine's own constant

using rrea::smoke::require;

double dot(
    double ax, double ay, double az, double bx, double by, double bz)
{
    return ax * bx + ay * by + az * bz;
}

void exercise_unit_cycle_and_normalize()
{
    using rrea::warpx::RreaNormalizeDirection;
    using rrea::warpx::RreaUnitCycle;

    require(RreaUnitCycle(-0.25) == 0.75, "unit cycle of -0.25");
    require(RreaUnitCycle(1.25) == 0.25, "unit cycle of 1.25");
    require(RreaUnitCycle(1.0) == 0.0, "unit cycle of exactly 1");
    require(RreaUnitCycle(0.0) == 0.0, "unit cycle of exactly 0");

    amrex::Real ux = 0.0, uy = 0.0, uz = 0.0;
    RreaNormalizeDirection(ux, uy, uz);
    require(ux == 0.0 && uy == 0.0 && uz == 1.0,
        "degenerate zero vector must normalize to +z exactly");
    ux = 3.0; uy = 0.0; uz = 4.0;
    RreaNormalizeDirection(ux, uy, uz);
    rrea::smoke::require_close_rel(ux, 0.6, 1.0e-15, "3-4-5 normalization x");
    rrea::smoke::require_close_rel(uz, 0.8, 1.0e-15, "3-4-5 normalization z");
}

void exercise_magnetic_rotation()
{
    rrea::RreaProperVelocity3 const initial{1.0e8, 2.0e7, -3.0e7};
    auto const rotated = rrea::RreaBorisMagneticRotate(
        initial, -1.75882001076e11 * 2.5e-9, 0.0, 18.0e-6, 0.0);
    double const before = initial.x * initial.x + initial.y * initial.y
        + initial.z * initial.z;
    double const after = rotated.x * rotated.x + rotated.y * rotated.y
        + rotated.z * rotated.z;
    rrea::smoke::require_close_rel(after, before, 2.0e-15,
        "magnetic Boris rotation changed kinetic energy");
}

void exercise_cone_direction()
{
    using rrea::warpx::RreaSampleConeDirection;

    // Both reference branches of the transverse basis: a z-dominant axis
    // (|dir_z| > 0.9 flips the reference vector) and a transverse axis.
    struct Axis { double x, y, z; };
    for (Axis const axis : {Axis{0.1, 0.2, 0.97}, Axis{0.9, 0.1, 0.3}}) {
        for (double cos_theta : {-0.8, -0.2, 0.3, 0.7, 0.999}) {
            for (double azimuth_u : {0.0, 0.31, 0.77}) {
                amrex::Real ox = 0.0, oy = 0.0, oz = 0.0;
                RreaSampleConeDirection(
                    axis.x, axis.y, axis.z, cos_theta, azimuth_u, ox, oy, oz);
                rrea::smoke::require_close_rel(
                    dot(ox, oy, oz, ox, oy, oz), 1.0, 1.0e-15,
                    "cone direction is unit norm");
                double nx = axis.x, ny = axis.y, nz = axis.z;
                double const n = std::sqrt(dot(nx, ny, nz, nx, ny, nz));
                rrea::smoke::require_close_rel(
                    dot(ox, oy, oz, nx / n, ny / n, nz / n), cos_theta, 1.0e-14,
                    "cone direction reproduces the requested polar cosine");
            }
        }
    }
    // Out-of-range cosines clamp to the axis (or its antipode) exactly.
    amrex::Real ox = 0.0, oy = 0.0, oz = 0.0;
    RreaSampleConeDirection(0.0, 0.0, 1.0, 1.5, 0.4, ox, oy, oz);
    rrea::smoke::require_close_rel(oz, 1.0, 1.0e-15, "cos_theta above 1 clamps onto the axis");
    RreaSampleConeDirection(0.0, 0.0, 1.0, -2.0, 0.4, ox, oy, oz);
    rrea::smoke::require_close_rel(oz, -1.0, 1.0e-15, "cos_theta below -1 clamps to the antipode");
}

void exercise_moller_cosines()
{
    using rrea::warpx::RreaMollerPrimaryCosThetaLab;
    using rrea::warpx::RreaMollerSecondaryCosThetaLab;

    // Closed form cos^2 = eps (T + 2m) / (T (eps + 2m)) at spread points.
    for (double t_eV : {2.0e5, 1.0e6, 7.2e6, 5.0e7, 1.0e9}) {
        for (double fraction : {0.05, 0.25, 0.5}) {
            double const eps = fraction * t_eV;
            double const expected = std::sqrt(
                eps * (t_eV + 2.0 * kMeC2eV) / (t_eV * (eps + 2.0 * kMeC2eV)));
            rrea::smoke::require_close_rel(
                RreaMollerSecondaryCosThetaLab(t_eV, eps), expected, 1.0e-14,
                "secondary lab cosine matches the two-body closed form");
        }
    }

    // At T = 2 Tcut with an even split, the knock-on must leave the parent
    // axis decisively (closed form: cos = 0.7379...).
    require(RreaMollerSecondaryCosThetaLab(2.0e5, 1.0e5) < 0.80,
        "threshold knock-on must not be field-aligned");

    // Limits: a vanishing transfer is perpendicular, a total transfer is
    // forward, and an even split gives equal cosines for both leptons.
    require(RreaMollerSecondaryCosThetaLab(1.0e6, 1.0e-6) < 1.0e-3,
        "vanishing transfer must leave perpendicular");
    require(RreaMollerSecondaryCosThetaLab(1.0e6, 1.0e6) == 1.0,
        "total transfer must be exactly forward");
    rrea::smoke::require_close_rel(
        RreaMollerSecondaryCosThetaLab(1.0e6, 5.0e5),
        RreaMollerPrimaryCosThetaLab(1.0e6, 5.0e5),
        1.0e-15,
        "even split gives symmetric lepton cosines");
    require(RreaMollerPrimaryCosThetaLab(1.0e6, 1.0e6) == 1.0,
        "fully-drained parent takes the guard branch exactly");
}

void exercise_moller_lab_directions()
{
    using rrea::warpx::RreaMollerPrimaryCosThetaLab;
    using rrea::warpx::RreaMollerSecondaryCosThetaLab;
    using rrea::warpx::RreaSampleMollerLabDirections;

    double const ix = 0.267261241912424, iy = 0.534522483824849,
                 iz = 0.801783725737273;  // normalized (1,2,3)
    // Bhabha reuses this sampler with transfers past T/2 (secondary up to T);
    // sweep both regimes.
    for (double t_eV : {3.0e5, 2.0e6, 3.0e7}) {
        for (double fraction : {0.1, 0.5, 0.8, 0.97}) {
            for (double azimuth_u : {0.05, 0.6}) {
                double const eps = fraction * t_eV;
                amrex::Real px = 0.0, py = 0.0, pz = 0.0;
                amrex::Real sx = 0.0, sy = 0.0, sz = 0.0;
                RreaSampleMollerLabDirections(
                    ix, iy, iz, t_eV, eps, azimuth_u, px, py, pz, sx, sy, sz);
                rrea::smoke::require_close_rel(dot(px, py, pz, px, py, pz), 1.0, 1.0e-15,
                    "scattered parent is unit norm");
                rrea::smoke::require_close_rel(dot(sx, sy, sz, sx, sy, sz), 1.0, 1.0e-15,
                    "secondary is unit norm");
                rrea::smoke::require_close_rel(
                    dot(sx, sy, sz, ix, iy, iz),
                    double(RreaMollerSecondaryCosThetaLab(t_eV, eps)),
                    1.0e-14,
                    "secondary sits on its own two-body cone");
                rrea::smoke::require_close_rel(
                    dot(px, py, pz, ix, iy, iz),
                    double(RreaMollerPrimaryCosThetaLab(t_eV, eps)),
                    1.0e-14,
                    "scattered parent sits on its own two-body cone");
                // Opposite azimuths: the two transverse components must be
                // exactly anti-parallel (the UnitCycle(u + 0.5) convention).
                double const cs = dot(sx, sy, sz, ix, iy, iz);
                double const cp = dot(px, py, pz, ix, iy, iz);
                double tsx = sx - cs * ix, tsy = sy - cs * iy, tsz = sz - cs * iz;
                double tpx = px - cp * ix, tpy = py - cp * iy, tpz = pz - cp * iz;
                double const ns = std::sqrt(dot(tsx, tsy, tsz, tsx, tsy, tsz));
                double const np = std::sqrt(dot(tpx, tpy, tpz, tpx, tpy, tpz));
                if (ns > 1.0e-12 && np > 1.0e-12) {
                    rrea::smoke::require_close_rel(
                        dot(tsx / ns, tsy / ns, tsz / ns,
                            tpx / np, tpy / np, tpz / np),
                        -1.0, 1.0e-13,
                        "pair transverse directions are anti-parallel");
                }
            }
        }
    }
}

void exercise_photon_recoil_and_rotation()
{
    using rrea::warpx::RreaApplyPhotonRecoilToChargedDirection;
    using rrea::warpx::RreaRotateLocalDirectionToAxis;

    // Forward photon leaves the direction unchanged; a transverse photon
    // deflects opposite to its own transverse momentum.
    amrex::Real cx = 0.0, cy = 0.0, cz = 1.0;
    RreaApplyPhotonRecoilToChargedDirection(
        1.0e6, 2.0e5, 0.0, 0.0, 1.0, cx, cy, cz);
    rrea::smoke::require_close_rel(cz, 1.0, 1.0e-15, "forward emission keeps the parent forward");
    cx = 0.0; cy = 0.0; cz = 1.0;
    RreaApplyPhotonRecoilToChargedDirection(
        1.0e6, 2.0e5, 1.0, 0.0, 0.0, cx, cy, cz);
    require(cx < 0.0 && cz > 0.0,
        "transverse emission recoils the parent the opposite way");
    double const p_initial =
        std::sqrt(1.0e6 * (1.0e6 + 2.0 * kMeC2eV));
    rrea::smoke::require_close_rel(
        cx, -2.0e5 / std::sqrt(2.0e5 * 2.0e5 + p_initial * p_initial),
        1.0e-14, "recoil deflection follows exact momentum subtraction");

    // Rotation: the local +z axis maps onto the axis itself; a local
    // transverse unit vector stays perpendicular to the axis.
    amrex::Real rx = 0.0, ry = 0.0, rz = 0.0;
    RreaRotateLocalDirectionToAxis(0.6, 0.0, 0.8, 0.0, 0.0, 1.0, rx, ry, rz);
    rrea::smoke::require_close_rel(rx, 0.6, 1.0e-15, "local +z maps to the axis (x)");
    rrea::smoke::require_close_rel(rz, 0.8, 1.0e-15, "local +z maps to the axis (z)");
    RreaRotateLocalDirectionToAxis(0.6, 0.0, 0.8, 1.0, 0.0, 0.0, rx, ry, rz);
    rrea::smoke::require_close_rel(dot(rx, ry, rz, 0.6, 0.0, 0.8), 0.0, 1.0e-15,
        "local transverse stays perpendicular to the axis");
}

void exercise_ion_pair_partition()
{
    using rrea::warpx::RreaPartitionContinuousIonPairs;

    struct Interval {
        amrex::Real begin_fraction, end_fraction, density_ratio;
    };
    Interval const intervals[] = {
        {0.0, 0.4, 1.0},   // clamped to [0.25, 0.4] by the segment
        {0.4, 1.0, 0.5},   // clamped to [0.4, 0.85]
        {0.9, 0.95, 2.0},  // outside the segment once clamped: must be skipped
    };
    amrex::Real const stopping_stp = amrex::Real(1.0e5);
    double emitted_fraction[2] = {0.0, 0.0};
    double emitted_weight[2] = {0.0, 0.0};
    int emits = 0;
    RreaPartitionContinuousIonPairs(
        intervals, 2.0, 0.25, 0.85, 6.0e4, 7.5e4, 100.0, 34.0, stopping_stp,
        [&](amrex::Real midpoint_fraction, amrex::Real pair_weight) {
            require(emits < 2, "partition emitted more intervals than overlap");
            emitted_fraction[emits] = midpoint_fraction;
            emitted_weight[emits] = pair_weight;
            ++emits;
        });
    require(emits == 2, "partition must emit exactly the overlapping intervals");
    // Analytic proration: raw losses 2*0.15*1e5 = 3e4 and 2*0.45*5e4 = 4.5e4
    // against the caller total 7.5e4, scaled onto the clamped segment loss
    // 6e4, then divided by W_air = 34 eV per pair at weight 100.
    rrea::smoke::require_close_rel(emitted_fraction[0], 0.325, 1.0e-15, "first interval midpoint");
    rrea::smoke::require_close_rel(emitted_fraction[1], 0.625, 1.0e-15, "second interval midpoint");
    rrea::smoke::require_close_rel(emitted_weight[0], 100.0 * 2.4e4 / 34.0, 1.0e-12,
        "first interval prorated pair weight");
    rrea::smoke::require_close_rel(emitted_weight[1], 100.0 * 3.6e4 / 34.0, 1.0e-12,
        "second interval prorated pair weight");
    rrea::smoke::require_close_rel(
        (emitted_weight[0] + emitted_weight[1]) * 34.0 / 100.0, 6.0e4, 1.0e-12,
        "proration must conserve the clamped segment collision loss");

    // A closed segment emits nothing; a sub-unity W_air divides by the 1.0
    // floor rather than amplifying the pair weight.
    RreaPartitionContinuousIonPairs(
        intervals, 2.0, 0.25, 0.85, 0.0, 7.5e4, 100.0, 34.0, stopping_stp,
        [&](amrex::Real, amrex::Real) {
            require(false, "zero segment loss must not emit ion pairs");
        });
    double floored_weight = 0.0;
    RreaPartitionContinuousIonPairs(
        intervals, 2.0, 0.25, 0.4, 3.0e4, 3.0e4, 100.0, 0.5, stopping_stp,
        [&](amrex::Real, amrex::Real pair_weight) {
            floored_weight = pair_weight;
        });
    rrea::smoke::require_close_rel(floored_weight, 100.0 * 3.0e4 / 1.0, 1.0e-12,
        "sub-unity W_air must divide by the 1.0 floor");
}

void exercise_decision_helpers()
{
    using rrea::warpx::RreaChargedChannelRevalidation;
    using rrea::warpx::RreaChargedStopIsEscape;
    using rrea::warpx::RreaChargedTerminalDisposition;
    using rrea::warpx::RreaClassifyChargedChannelRevalidation;
    using rrea::warpx::RreaResolveChargedTransportTerminal;

    // Revalidation: an open channel proceeds regardless of the frozen state;
    // drag-closed becomes the explicit null event; closed-at-selection is the
    // fail-closed abort arm (asserted on the classifier, not a death test).
    require(RreaClassifyChargedChannelRevalidation(true, true)
            == RreaChargedChannelRevalidation::Proceed,
        "open channel must proceed");
    require(RreaClassifyChargedChannelRevalidation(true, false)
            == RreaChargedChannelRevalidation::Proceed,
        "open-at-event must proceed whatever the frozen state");
    require(RreaClassifyChargedChannelRevalidation(false, true)
            == RreaChargedChannelRevalidation::NullEvent,
        "drag-closed channel must become an explicit null event, "
        "never a silent resample or conversion");
    require(RreaClassifyChargedChannelRevalidation(false, false)
            == RreaChargedChannelRevalidation::Invalid,
        "closed-at-selection must fail closed");

    // Stop-vs-escape: a stop exactly AT the clip root of an exiting chord is
    // an escape (the >= boundary), an interior stop on an exiting chord is a
    // real stop, and a non-exiting chord never escapes this way.
    require(RreaChargedStopIsEscape(true, 0.8, 0.8),
        "a stop exactly at the clip root must escape");
    require(RreaChargedStopIsEscape(true, 0.80000001, 0.8),
        "a stop past the clip root must escape");
    require(!RreaChargedStopIsEscape(true, 0.79, 0.8),
        "an interior stop on an exiting chord is a real stop");
    require(!RreaChargedStopIsEscape(false, 0.9, 0.8),
        "a non-exiting chord never converts a stop to an escape");

    // Terminal composition: an observed exit outranks a recorded stop for
    // BOTH species -- the swapped ordering emits two 511 keV photons and ion
    // pairs in-domain for a particle that already left.
    require(RreaResolveChargedTransportTerminal(
                true, true,
                RreaChargedTerminalDisposition::ElectronDemotion,
                false, true, true)
            == RreaChargedTerminalDisposition::Escape,
        "electron: escape must outrank a recorded mid-chord stop");
    require(RreaResolveChargedTransportTerminal(
                true, true,
                RreaChargedTerminalDisposition::PositronAnnihilation,
                true, true, false)
            == RreaChargedTerminalDisposition::Escape,
        "positron: escape must outrank a recorded mid-chord stop");
    require(RreaResolveChargedTransportTerminal(
                false, true,
                RreaChargedTerminalDisposition::PositronAnnihilation,
                true, false, false)
            == RreaChargedTerminalDisposition::PositronAnnihilation,
        "in-domain mid-chord stop takes the species stop disposition");
    require(RreaResolveChargedTransportTerminal(
                false, false,
                RreaChargedTerminalDisposition::ElectronDemotion,
                true, true, false)
            == RreaChargedTerminalDisposition::PositronAnnihilation,
        "neither exit nor stop delegates to the endpoint resolution");
    require(RreaResolveChargedTransportTerminal(
                false, false,
                RreaChargedTerminalDisposition::ElectronDemotion,
                false, false, false)
            == RreaChargedTerminalDisposition::Survive,
        "an above-cutoff in-domain electron survives");
}

void exercise_channel_inverse_length()
{
    using rrea::warpx::RreaChannelInverseLength;

    // 4 m at 2 MeV and unit density, scaling as 1/density like a real table.
    int calls = 0;
    auto mfp_at = [&calls](double energy_eV, double density_ratio) {
        ++calls;
        return 4.0 * (energy_eV / 1.0e6) / density_ratio;
    };

    rrea::smoke::require_close_rel(
        RreaChannelInverseLength(true, [&] { return mfp_at(2.0e6, 1.0); }),
        1.0 / 8.0, 1.0e-14, "open channel inverts the mean free path");
    rrea::smoke::require_close_rel(
        RreaChannelInverseLength(true, [&] { return mfp_at(2.0e6, 0.25); }),
        0.25 / 8.0, 1.0e-14, "inverse length scales with density ratio");

    // The gate is EXACT: a closed channel contributes nothing, so the hazard
    // inversion can never select it.  A small residual would let it fire.
    int const before = calls;
    require(RreaChannelInverseLength(false, [&] { return mfp_at(2.0e6, 1.0); })
                == 0.0,
        "closed channel is exactly zero");
    // ...and it SHORT-CIRCUITS: a closed channel must not query the table at
    // all, because the channel gate is what keeps the lookup inside the
    // energy range where that channel is defined.
    require(calls == before,
        "closed channel must not evaluate the mean free path");

    // A non-positive mean free path is a table hole, not an infinite rate.
    require(RreaChannelInverseLength(true, [] { return 0.0; }) == 0.0,
        "non-positive mean free path yields zero, never a division");
    require(RreaChannelInverseLength(true, [] { return -1.0; }) == 0.0,
        "negative mean free path yields zero");
}

void exercise_charged_field_half_kick()
{
    using rrea::warpx::RreaApplyChargedFieldHalfKick;
    using rrea::warpx::RreaPathFieldSubRange;

    // Energy comes STRICTLY from the line-integral work, never from the
    // vector impulse: a range carrying a transverse field but zero work must
    // turn the direction and leave the energy untouched.
    RreaPathFieldSubRange turn_only;
    turn_only.work_eV = 0.0;
    turn_only.avg_ey_v_per_m = 5.0e6;
    double energy = 3.0e6, dx = 0.0, dy = 0.0, dz = 1.0;
    double const work = RreaApplyChargedFieldHalfKick(
        turn_only, 1.0e-9, -1.0, energy, dx, dy, dz);
    require(work == 0.0, "zero-work range does no work");
    rrea::smoke::require_close_rel(energy, 3.0e6, 1.0e-14,
        "the vector impulse must not change the energy");
    rrea::smoke::require_close_rel(dx * dx + dy * dy + dz * dz, 1.0, 1.0e-14,
        "direction stays a unit vector");
    require(dy != 0.0, "a transverse field must turn the direction");

    // Half the range work lands on the energy, with the sign of the work.
    RreaPathFieldSubRange gain;
    gain.work_eV = 4.0e5;
    double e2 = 1.0e6, ax = 0.0, ay = 0.0, az = 1.0;
    rrea::smoke::require_close_rel(
        RreaApplyChargedFieldHalfKick(gain, 1.0e-12, -1.0, e2, ax, ay, az),
        2.0e5, 1.0e-14, "returns half the range work");
    rrea::smoke::require_close_rel(e2, 1.2e6, 1.0e-14,
        "half the work is added to the kinetic energy");

    // Energy is floored at zero rather than going negative.
    RreaPathFieldSubRange drain;
    drain.work_eV = -1.0e7;
    double e3 = 1.0e5, bx = 0.0, by = 0.0, bz = 1.0;
    (void)RreaApplyChargedFieldHalfKick(drain, 1.0e-12, -1.0, e3, bx, by, bz);
    require(e3 == 0.0, "kinetic energy is floored at zero, never negative");

    // Charge sign reverses the impulse: a positron turns the other way.
    RreaPathFieldSubRange turn;
    turn.avg_ey_v_per_m = 5.0e6;
    double ee = 3.0e6, ex_ = 0.0, ey_ = 0.0, ez_ = 1.0;
    double pe = 3.0e6, px = 0.0, py = 0.0, pz = 1.0;
    (void)RreaApplyChargedFieldHalfKick(turn, 1.0e-9, -1.0, ee, ex_, ey_, ez_);
    (void)RreaApplyChargedFieldHalfKick(turn, 1.0e-9, +1.0, pe, px, py, pz);
    rrea::smoke::require_close_rel(ey_, -py, 1.0e-14,
        "opposite charge sign turns the opposite way");
}

void exercise_poisson_sample()
{
    using rrea::warpx::RreaPoissonSampleCapped;

    require(RreaPoissonSampleCapped(0.0, 0.7, 12) == 0,
        "nonpositive mean must return zero draws");
    require(RreaPoissonSampleCapped(3.0, 1.0 - 1.0e-16, 9) == 9,
        "u -> 1 saturates at the cap");
    double sum = 0.0;
    int const n = 10000;
    for (int i = 0; i < n; ++i) {
        sum += RreaPoissonSampleCapped(3.0, (i + 0.5) / n, 30);
    }
    rrea::smoke::require_close_rel(sum / n, 3.0, 1.0e-2,
        "inverse-transform Poisson mean over a uniform grid");
}

// ---- The event-interleave core against a recording analytic medium --------
//
// Rate sigma(E) = E/1e6 per metre and stopping S(E) = 0.5*E per metre over a
// unit chord: both DEPEND on the frozen energy, so a refreeze bug cannot hide.
// Draws are scripted (u = -expm1(-tau) places an event at an exact optical
// depth); every policy records what the core handed it.

struct InterleaveRecorder {
    amrex::Real kinetic_eV;
    amrex::Real cutoff_eV;
    amrex::Real open_above_eV;   // channel_open_at: E must exceed this
    std::vector<std::pair<double, double>> draws;
    std::vector<double> event_losses_eV;  // subtracted per final_state call
    bool consume_parent_on_event = false;

    std::vector<double> frozen_energies;
    std::vector<double> taus;
    struct Segment { double begin, end, frozen; };
    std::vector<Segment> segments;
    std::vector<std::uint64_t> draw_ordinals;
    std::vector<double> open_check_energies;
    std::vector<double> final_state_fractions;
    std::vector<double> stops;
    std::size_t draw_cursor = 0;
    std::size_t final_state_calls = 0;

    void run()
    {
        struct FakeLimit {
            bool reaches_cutoff;
            amrex::Real effective_end_fraction;
        };
        rrea::warpx::RreaRunChargedEventInterleave(
            amrex::Real(0.0),
            amrex::Real(1.0),
            kinetic_eV,
            cutoff_eV,
            /*channel_count=*/2,
            /*max_event_ordinal=*/64,
            "smoke",
            [&](amrex::Real frozen_eV, amrex::Real begin, amrex::Real end) {
                frozen_energies.push_back(double(frozen_eV));
                double const tau = double(end - begin) * double(frozen_eV) / 1.0e6;
                taus.push_back(tau);
                return std::array<amrex::Real, 3>{
                    amrex::Real(tau), amrex::Real(0.0), amrex::Real(0.0)};
            },
            [&](std::uint64_t ordinal) {
                draw_ordinals.push_back(ordinal);
                require(draw_cursor < draws.size(),
                    "interleave asked for more draws than the script holds");
                auto const draw = draws[draw_cursor++];
                return std::pair<amrex::Real, amrex::Real>{
                    amrex::Real(draw.first), amrex::Real(draw.second)};
            },
            [&](amrex::Real tau, amrex::Real begin, amrex::Real) {
                return begin
                    + tau * amrex::Real(1.0e6) / kinetic_eV;
            },
            [&](amrex::Real begin, amrex::Real end, amrex::Real frozen_eV) {
                segments.push_back({double(begin), double(end), double(frozen_eV)});
                amrex::Real const loss =
                    amrex::Real(0.5) * (end - begin) * frozen_eV;
                kinetic_eV -= loss;
                if (kinetic_eV < cutoff_eV) {
                    return FakeLimit{
                        true, amrex::Real(0.5) * (begin + end)};
                }
                return FakeLimit{false, end};
            },
            [&](amrex::Real fraction) { stops.push_back(double(fraction)); },
            [&](amrex::Real) {},
            [&](int, amrex::Real energy_eV) {
                open_check_energies.push_back(double(energy_eV));
                return double(energy_eV) > double(open_above_eV);
            },
            [&](int channel, amrex::Real event_fraction) {
                require(channel == 0, "single-rate fixture must select channel 0");
                final_state_fractions.push_back(double(event_fraction));
                require(final_state_calls < event_losses_eV.size(),
                    "final_state called more often than scripted");
                kinetic_eV -= amrex::Real(event_losses_eV[final_state_calls]);
                ++final_state_calls;
                return !consume_parent_on_event;
            });
    }
};

double u_for_tau(double tau) { return -std::expm1(-tau); }

void exercise_event_interleave()
{
    // Refreeze at survivor energy + loss to the exact sampled depth.  One
    // event at fraction 0.25 losing 1e5 eV: iteration 2's rates MUST be
    // frozen at 7.75e5 eV (tau 0.58125). Hoisting the refreeze out of the
    // loop freezes 1e6 and yields tau 0.75;
    // passing subcycle_end instead of the sampled depth books 5e5 not 1.25e5.
    {
        InterleaveRecorder rec;
        rec.kinetic_eV = 1.0e6;
        rec.cutoff_eV = 1.0e3;
        rec.open_above_eV = 0.0;
        rec.draws = {{u_for_tau(0.25), 0.1}, {u_for_tau(10.0), 0.1}};
        rec.event_losses_eV = {1.0e5};
        rec.run();
        require(rec.frozen_energies.size() == 2,
            "expected exactly two hazard iterations");
        rrea::smoke::require_close_rel(rec.frozen_energies[0], 1.0e6, 1.0e-15,
            "first iteration freezes the entry energy");
        rrea::smoke::require_close_rel(rec.frozen_energies[1], 7.75e5, 1.0e-12,
            "second iteration must refreeze at the survivor energy");
        rrea::smoke::require_close_rel(rec.taus[1], 0.58125, 1.0e-12,
            "survivor-energy tau over the remaining segment");
        require(rec.segments.size() == 2, "one pre-event and one tail segment");
        rrea::smoke::require_close_rel(rec.segments[0].begin, 0.0, 1.0e-15,
            "pre-event segment starts at the running begin");
        rrea::smoke::require_close_rel(rec.segments[0].end, 0.25, 1.0e-12,
            "continuous loss runs to the EXACT sampled depth");
        rrea::smoke::require_close_rel(rec.segments[0].frozen, 1.0e6, 1.0e-15,
            "pre-event segment is frozen at the iteration energy");
        rrea::smoke::require_close_rel(rec.open_check_energies.at(0), 8.75e5, 1.0e-12,
            "revalidation sees the post-segment survivor energy");
        require(rec.final_state_calls == 1 && rec.stops.empty(),
            "exactly one committed event and no stop");
        rrea::smoke::require_close_rel(rec.segments[1].begin, 0.25, 1.0e-12,
            "tail segment continues from the event fraction");
        rrea::smoke::require_close_rel(rec.segments[1].frozen, 7.75e5, 1.0e-12,
            "tail segment is frozen at the survivor energy");
    }

    // Post-event loss at the survivor energy: a 5e5 eV event at 0.25 leaves
    // 3.75e5 eV; the second event's pre-segment (0.25 -> 0.5) must book
    // 0.5 * 0.25 * 3.75e5 = 46875 eV from THAT energy, not the entry energy.
    {
        InterleaveRecorder rec;
        rec.kinetic_eV = 1.0e6;
        rec.cutoff_eV = 1.0e3;
        rec.open_above_eV = 0.0;
        rec.draws = {
            {u_for_tau(0.25), 0.1},
            {u_for_tau(0.09375), 0.1},
            {u_for_tau(10.0), 0.1},
        };
        rec.event_losses_eV = {5.0e5, 1.0e4};
        rec.run();
        require(rec.final_state_calls == 2, "two committed events expected");
        rrea::smoke::require_close_rel(rec.segments[1].begin, 0.25, 1.0e-12,
            "second pre-event segment starts at event one");
        rrea::smoke::require_close_rel(rec.segments[1].end, 0.5, 1.0e-11,
            "second event lands at its scripted depth");
        rrea::smoke::require_close_rel(rec.segments[1].frozen, 3.75e5, 1.0e-12,
            "post-event segment is frozen at the survivor energy");
        double const booked =
            0.5 * (rec.segments[1].end - rec.segments[1].begin)
            * rec.segments[1].frozen;
        rrea::smoke::require_close_rel(booked, 46875.0, 1.0e-11,
            "second segment books the survivor-energy loss");
    }

    // Revalidation: the channel opens only above 8e5 eV.  Drag to the sampled
    // depth (0.6) leaves 6.3e5 eV -- closed at the event, open at the frozen
    // 9e5 eV -- so the event must become an explicit null event: no final
    // state, a NEW rng ordinal, and the loop continues from the event
    // fraction with rates refrozen at 6.3e5 eV (tau 0.252).  Resampling the
    // same ordinal, converting the channel, or breaking out all fail here.
    {
        InterleaveRecorder rec;
        rec.kinetic_eV = 9.0e5;
        rec.cutoff_eV = 1.0e3;
        rec.open_above_eV = 8.0e5;
        rec.draws = {{u_for_tau(0.54), 0.1}, {u_for_tau(10.0), 0.1}};
        rec.event_losses_eV = {};
        rec.run();
        require(rec.final_state_calls == 0,
            "a drag-closed channel must not assemble a final state");
        require(rec.draw_ordinals.size() == 2
                && rec.draw_ordinals[0] == 0 && rec.draw_ordinals[1] == 1,
            "the null event must advance to a NEW rng ordinal");
        rrea::smoke::require_close_rel(rec.taus[1], 0.252, 1.0e-11,
            "continuation refreezes at the dragged energy past the null event");
        require(rec.open_check_energies.size() == 2,
            "revalidation checks the event energy then the frozen energy");
        rrea::smoke::require_close_rel(rec.open_check_energies[1], 9.0e5, 1.0e-15,
            "the frozen-energy side of the null-event classification");
    }

    // Parent consumed: a final state returning false (positron annihilation)
    // ends the loop immediately -- no tail segment, no stop, no extra draws.
    {
        InterleaveRecorder rec;
        rec.kinetic_eV = 1.0e6;
        rec.cutoff_eV = 1.0e3;
        rec.open_above_eV = 0.0;
        rec.draws = {{u_for_tau(0.25), 0.1}};
        rec.event_losses_eV = {1.0e5};
        rec.consume_parent_on_event = true;
        rec.run();
        require(rec.final_state_calls == 1 && rec.segments.size() == 1
                && rec.stops.empty() && rec.draw_ordinals.size() == 1,
            "a consumed parent must end the interleave immediately");
    }

    // Cutoff stop: when the no-event tail segment reaches the cutoff, the
    // core must record the stop at the limiter's fraction and end.
    {
        InterleaveRecorder rec;
        rec.kinetic_eV = 1.5e3;
        rec.cutoff_eV = 1.0e3;
        rec.open_above_eV = 0.0;
        rec.draws = {{u_for_tau(10.0), 0.1}};
        rec.event_losses_eV = {};
        rec.run();
        require(rec.stops.size() == 1, "cutoff must record exactly one stop");
        rrea::smoke::require_close_rel(rec.stops[0], 0.5, 1.0e-15,
            "the stop uses the limiter's effective fraction");
        require(rec.final_state_calls == 0, "no event below the hazard depth");
    }
}

// ---- The substep skeleton: ordering, sticky exit, committed counting ------

struct SubstepTrace {
    std::vector<std::string> tags;
    std::uint64_t completed = 0;
    bool exited = false;
    amrex::Real kinetic_eV = 1.0e6;

    void run(
        std::uint64_t subcycles,
        bool build_ok_all,
        std::uint64_t refuse_advance_at,   // subcycle index, or huge to never
        std::uint64_t exit_path_at,        // subcycle index, or huge to never
        bool stop_after_events)
    {
        rrea::warpx::RreaRunChargedSubstepInterleave(
            subcycles,
            kinetic_eV,
            amrex::Real(1.0e3),
            completed,
            exited,
            [&](std::uint64_t subcycle, amrex::Real& begin, amrex::Real& end) {
                tags.push_back("build" + std::to_string(subcycle));
                begin = amrex::Real(0.0);
                end = amrex::Real(1.0);
                return build_ok_all;
            },
            [&](std::uint64_t subcycle, amrex::Real, amrex::Real) {
                tags.push_back("events" + std::to_string(subcycle));
            },
            [&] { return stop_after_events; },
            [&](std::uint64_t subcycle, amrex::Real, amrex::Real) {
                tags.push_back("finish" + std::to_string(subcycle));
                return false;
            },
            [&, refuse_advance_at](amrex::Real) {
                std::uint64_t const subcycle = completed;
                tags.push_back("advance" + std::to_string(subcycle));
                return subcycle != refuse_advance_at;
            },
            [&, exit_path_at] { return completed == exit_path_at; });
    }
};

void exercise_substep_interleave()
{
    std::uint64_t const never = 1000;

    // Full committed run: build -> events -> finish -> advance, once per
    // substep and in that order; scattering (finish) NEVER precedes an event.
    {
        SubstepTrace trace;
        trace.run(3, true, never, never, false);
        std::vector<std::string> const expected = {
            "build0", "events0", "finish0", "advance0",
            "build1", "events1", "finish1", "advance1",
            "build2", "events2", "finish2", "advance2",
        };
        require(trace.tags == expected,
            "substep skeleton must run build/events/finish/advance in order");
        require(trace.completed == 3 && !trace.exited,
            "a fully committed run counts every substep and never exits");
    }

    // Sticky exit on a refused endpoint: the second advance refuses, so the
    // loop ends with completed_substeps == 1 (NOT 2 -- elapsed clocks depend
    // on counting only fully committed microchords) and no third build.
    {
        SubstepTrace trace;
        trace.run(3, true, /*refuse_advance_at=*/1, never, false);
        require(trace.exited, "a refused endpoint must set exited_domain");
        require(trace.completed == 1,
            "a refused substep must not count as completed");
        require(trace.tags.back() == "advance1"
                && trace.tags.size() == 8,
            "no substep may run after the refusal");
    }

    // An exiting committed chord is equally sticky.
    {
        SubstepTrace trace;
        trace.run(3, true, never, /*exit_path_at=*/0, false);
        require(trace.exited && trace.completed == 0,
            "an exiting chord must set exited_domain before counting");
    }

    // A stop after the events skips scattering AND the commit: a stopped
    // parent must never be scattered or advanced.
    {
        SubstepTrace trace;
        trace.run(3, true, never, never, /*stop_after_events=*/true);
        require(trace.tags == std::vector<std::string>{"build0", "events0"},
            "a stopped parent skips finish and advance entirely");
        require(trace.completed == 0 && !trace.exited,
            "a stopped substep is not committed");
    }

    // A failed build ends the loop before any event runs; a sub-cutoff parent
    // never even builds.
    {
        SubstepTrace trace;
        trace.run(3, false, never, never, false);
        require(trace.tags == std::vector<std::string>{"build0"},
            "a failed build must end the loop immediately");
    }
    {
        SubstepTrace trace;
        trace.kinetic_eV = amrex::Real(0.5e3);
        trace.run(3, true, never, never, false);
        require(trace.tags.empty() && trace.completed == 0,
            "a sub-cutoff parent must not build a chord");
    }
}

void exercise_exact_charged_boundary_crossings()
{
    using rrea::ClipRzTransportPath;
    using rrea::kEscapeFaceRHi;
    using rrea::kEscapeFaceZHi;
    using rrea::kEscapeFaceZLo;
    using rrea::warpx::RreaAdvanceRunningChargedPosition;
    using rrea::warpx::RreaChargedBoundaryCrossingForPath;

    auto path = [](double r0, double z0, double r1, double z1) {
        return ClipRzTransportPath(
            r0, 0.0, z0, r1, 0.0, z1,
            0.0, 8.0, 0.0, 10.0, false);
    };
    auto check_crossing = [&path](
        char const* name,
        double r0, double z0, double r1, double z1,
        double crossing_r, double crossing_z,
        std::uint32_t faces) {
        auto const clipped = path(r0, z0, r1, z1);
        auto const crossing = RreaChargedBoundaryCrossingForPath(clipped);
        require(crossing.valid, std::string(name) + " crossing is invalid");
        require(
            crossing.faces == faces,
            std::string(name) + " classified the wrong absorbing face(s)");
        rrea::smoke::require_close_rel(
            crossing.final_microchord_start_r_m, r0, 2.0e-14,
            std::string(name) + " lost final-microchord radial start");
        rrea::smoke::require_close_rel(
            crossing.final_microchord_start_z_m, z0, 2.0e-14,
            std::string(name) + " lost final-microchord axial start");
        rrea::smoke::require_close_rel(
            crossing.crossing_r_m, crossing_r, 2.0e-14,
            std::string(name) + " recovered wrong crossing radius");
        rrea::smoke::require_close_rel(
            crossing.crossing_z_m, crossing_z, 2.0e-14,
            std::string(name) + " recovered wrong crossing z");
        return clipped;
    };

    check_crossing(
        "r_hi", 7.0, 4.0, 9.0, 5.0,
        8.0, 4.5, kEscapeFaceRHi);
    check_crossing(
        "z_lo", 3.0, 1.0, 3.0, -1.0,
        3.0, 0.0, kEscapeFaceZLo);
    check_crossing(
        "z_hi", 3.0, 9.0, 3.0, 11.0,
        3.0, 10.0, kEscapeFaceZHi);
    check_crossing(
        "r_hi-z_hi corner", 7.0, 9.0, 9.0, 11.0,
        8.0, 10.0, kEscapeFaceRHi | kEscapeFaceZHi);
    check_crossing(
        "r_hi-z_lo corner", 7.0, 1.0, 9.0, -1.0,
        8.0, 0.0, kEscapeFaceRHi | kEscapeFaceZLo);
    check_crossing(
        "radial grazing on z_lo", 7.0, 0.0, 9.0, 0.0,
        8.0, 0.0, kEscapeFaceRHi);

    // A clipped path whose endpoint lands exactly on an excluded upper face
    // need not set exits_domain (candidate fraction is exactly one).  The
    // half-open commit refuses it, yet the crossing recovery must still snap
    // it to the face so the atomic escape closure cannot omit the microchord.
    auto const exact_upper = check_crossing(
        "exact refused r_hi", 7.0, 4.0, 8.0, 4.0,
        8.0, 4.0, kEscapeFaceRHi);
    require(!exact_upper.exits_domain,
        "exact upper-face fixture unexpectedly reports a geometric overshoot");
    amrex::Real run_x = 7.0;
    amrex::Real run_y = 0.0;
    amrex::Real run_z = 4.0;
    require(
        !RreaAdvanceRunningChargedPosition(
            exact_upper, exact_upper.in_domain_end_fraction,
            0.0, 8.0, 0.0, 10.0, run_x, run_y, run_z),
        "exact excluded upper face was not refused");
    require(run_x == 7.0 && run_y == 0.0 && run_z == 4.0,
        "refused upper-face commit moved the running point");

    check_crossing(
        "exact refused z_hi", 3.0, 9.0, 3.0, 10.0,
        3.0, 10.0, kEscapeFaceZHi);

    // A charged newborn may be created on the included low-z face.  Its first
    // outward residual chord has an exit fraction of exactly zero; crossing
    // recovery must retain the birth point as both microchord start and exit.
    auto const newborn_low = check_crossing(
        "newborn on z_lo", 3.0, 0.0, 3.0, -1.0,
        3.0, 0.0, kEscapeFaceZLo);
    require(newborn_low.exits_domain
            && newborn_low.in_domain_end_fraction == 0.0,
        "newborn low-face outward chord lost its zero-fraction exit");
}

}  // namespace

int main()
{
    try {
        exercise_unit_cycle_and_normalize();
        exercise_magnetic_rotation();
        exercise_cone_direction();
        exercise_moller_cosines();
        exercise_moller_lab_directions();
        exercise_photon_recoil_and_rotation();
        exercise_ion_pair_partition();
        exercise_decision_helpers();
        exercise_event_interleave();
        exercise_substep_interleave();
        exercise_exact_charged_boundary_crossings();
        exercise_poisson_sample();
        exercise_channel_inverse_length();
        exercise_charged_field_half_kick();
        std::cout << "rrea_charged_interleave_smoke passed\n";
        return 0;
    } catch (std::exception const& exc) {
        std::cerr << "rrea_charged_interleave_smoke failed: " << exc.what()
                  << "\n";
        return 1;
    }
}
