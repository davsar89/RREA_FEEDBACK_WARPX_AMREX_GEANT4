#ifndef RREA_CHARGED_INTERLEAVE_H
#define RREA_CHARGED_INTERLEAVE_H

// Pure kinematics and geometry primitives of the charged-transport
// interleave, shared by the production loops in RreaParticleInteraction.cpp
// and by rrea_charged_interleave_smoke.  Header-RREA_HOST_DEVICE inline on purpose: a separate
// translation unit would put the loops behind a call boundary and risk
// inlining/FMA divergence between the tested and the shipped code.

#include "RreaParticleInteraction.H"
#include "rrea/RreaLorentzPush.H"
#include "rrea/RreaSeedSchedule.H"



#include <AMReX_REAL.H>

#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <utility>

namespace rrea::warpx {

// Classification of the frozen-rate channel revalidation at a sampled event
// depth: open at the event -> proceed; closed
// at the event but open at the frozen substep-start energy -> explicit
// unbiased null event (continuous drag legitimately carried the particle
// across the channel-opening boundary); closed at both -> the selection
// itself used a closed channel and the caller must fail closed.
enum class RreaChargedChannelRevalidation { Proceed, NullEvent, Invalid };

[[nodiscard]] RREA_HOST_DEVICE inline RreaChargedChannelRevalidation
RreaClassifyChargedChannelRevalidation(
    bool open_at_event, bool open_at_frozen_energy) noexcept
{
    if (open_at_event) {
        return RreaChargedChannelRevalidation::Proceed;
    }
    return open_at_frozen_energy
        ? RreaChargedChannelRevalidation::NullEvent
        : RreaChargedChannelRevalidation::Invalid;
}

// A stop recorded at/past the clip root of an EXITING chord is an escape, not
// a stop: the particle leaves before it can thermalize, and recording the
// stop would book its sources on the excluded half-open face (which the
// collective transaction validator turns into an abort after every rank has
// finished transport).
// Per-channel hazard rate, in inverse metres: what one discrete channel
// contributes to the competing-risk optical depth.
//
//     1/lambda = 1 / mfp   channel open and mfp > 0
//              = 0         otherwise
//
// The gate is exact rather than a small rate, so the hazard inversion can
// never select a closed channel; and it SHORT-CIRCUITS, so a closed channel
// never queries a table outside the domain where that channel is defined.
// The mean free path therefore arrives as a nullary callable, not a value.
//
// One owner.  This construction was written out at seven sites across the
// electron, positron and newborn phases; a gate or an inversion drifting at
// any one of them would silently have given a particle's secondaries
// different physics from the particle itself.  Deliberately takes no state
// of its own: the callers hold mutable per-substep frozen energies and gate
// flags, and capturing those here by value would freeze them at their
// pre-substep values.
template <typename MeanFreePath>
[[nodiscard]] RREA_HOST_DEVICE inline amrex::Real RreaChannelInverseLength(
    bool channel_open, MeanFreePath&& mean_free_path_m)
{
    if (!channel_open) {
        return amrex::Real(0.0);
    }
    amrex::Real const mfp = mean_free_path_m();
    return mfp > amrex::Real(0.0) ? amrex::Real(1.0) / mfp : amrex::Real(0.0);
}

// Path-averaged field over one sub-interval of a transport chord: the work
// the parallel field did along it, and the mean E and B the impulse uses.
struct RreaPathFieldSubRange {
    amrex::Real work_eV = 0.0;
    amrex::Real avg_ex_v_per_m = 0.0;
    amrex::Real avg_ey_v_per_m = 0.0;
    amrex::Real avg_ez_v_per_m = 0.0;
    amrex::Real avg_bx_t = 0.0;
    amrex::Real avg_by_t = 0.0;
    amrex::Real avg_bz_t = 0.0;
    amrex::Real length_m = 0.0;
};

// One half-step field impulse (Strang), shared by the electron, positron and
// newborn charged paths.  Energy comes STRICTLY from the line-integral work:
// the vector impulse only rotates the direction -- physically turning a
// backward-scattered lepton around -- after which the magnitude is
// renormalized to the running kinetic energy, so direction and energy can
// never disagree.  Returns the half-step work in eV for the callers that keep
// a field-work ledger.
[[nodiscard]] RREA_HOST_DEVICE inline amrex::Real RreaApplyChargedFieldHalfKick(
    RreaPathFieldSubRange const& field_range,
    amrex::Real half_dt_s,
    amrex::Real charge_sign,
    amrex::Real& kinetic_energy_eV,
    amrex::Real& dir_x,
    amrex::Real& dir_y,
    amrex::Real& dir_z)
{
    amrex::Real const u_mag =
        rrea::RreaElectronProperVelocityFromKineticEv(kinetic_energy_eV);
    amrex::Real const impulse_scale =
        charge_sign
        * (amrex::Real(kRreaSpeedOfLightMPerS) * amrex::Real(kRreaSpeedOfLightMPerS)
           / amrex::Real(kRreaElectronRestEnergyEv))
        * half_dt_s;
    amrex::Real const ux_minus =
        u_mag * dir_x + impulse_scale * field_range.avg_ex_v_per_m;
    amrex::Real const uy_minus =
        u_mag * dir_y + impulse_scale * field_range.avg_ey_v_per_m;
    amrex::Real const uz_minus =
        u_mag * dir_z + impulse_scale * field_range.avg_ez_v_per_m;
    auto const rotated = rrea::RreaBorisMagneticRotate(
        {ux_minus, uy_minus, uz_minus}, impulse_scale,
        field_range.avg_bx_t, field_range.avg_by_t, field_range.avg_bz_t);
    amrex::Real const ux = rotated.x;
    amrex::Real const uy = rotated.y;
    amrex::Real const uz = rotated.z;
    amrex::Real const norm = std::sqrt(ux * ux + uy * uy + uz * uz);
    if (norm > amrex::Real(0.0) && std::isfinite(static_cast<double>(norm))) {
        dir_x = ux / norm;
        dir_y = uy / norm;
        dir_z = uz / norm;
    }
    amrex::Real const half_work_eV = amrex::Real(0.5) * field_range.work_eV;
    kinetic_energy_eV =
        rrea::kinetic::max(amrex::Real(0.0), kinetic_energy_eV + half_work_eV);
    return half_work_eV;
}

[[nodiscard]] RREA_HOST_DEVICE inline bool RreaChargedStopIsEscape(
    bool path_exits_domain,
    amrex::Real path_fraction,
    amrex::Real in_domain_end_fraction) noexcept
{
    return path_exits_domain && path_fraction >= in_domain_end_fraction;
}

// Prorates one continuous-drag segment's collision loss over the chord's
// material intervals and converts each share into low-energy ion pairs at
// the interval midpoint.  This arithmetic was byte-duplicated in the
// electron, positron, and newborn loops; only the position handling and the
// booked secondary energy differ, and those stay with the caller via emit.
// Intervals need begin_fraction / end_fraction / density_ratio members.
// collision_stp_eV_per_m is the restricted collision stopping power at the
// segment's frozen energy per unit density ratio; it cancels out of the
// prorating and is carried only so an interval's raw share stays a share.
template <typename Intervals, typename EmitIonPairs>
RREA_HOST_DEVICE inline void RreaPartitionContinuousIonPairs(
    Intervals const& material_intervals,
    amrex::Real full_length_m,
    amrex::Real begin_fraction,
    amrex::Real end_fraction,
    amrex::Real segment_collision_loss_eV,
    amrex::Real raw_collision_loss_eV,
    amrex::Real particle_weight,
    amrex::Real w_air_eV_per_pair,
    amrex::Real collision_stp_eV_per_m,
    EmitIonPairs&& emit)
{
    if (!(segment_collision_loss_eV > amrex::Real(0.0))
        || !(raw_collision_loss_eV > amrex::Real(0.0))) {
        return;
    }
    for (auto const& interval : material_intervals) {
        amrex::Real const begin = rrea::kinetic::max(
            begin_fraction, interval.begin_fraction);
        amrex::Real const end = rrea::kinetic::min(
            end_fraction, interval.end_fraction);
        if (!(end > begin)) {
            continue;
        }
        amrex::Real const raw_interval_loss_eV =
            full_length_m * (end - begin)
            * interval.density_ratio * collision_stp_eV_per_m;
        amrex::Real const interval_loss_eV =
            segment_collision_loss_eV
            * raw_interval_loss_eV / raw_collision_loss_eV;
        amrex::Real const pair_weight = particle_weight
            * interval_loss_eV
            / rrea::kinetic::max(w_air_eV_per_pair, amrex::Real(1.0));
        if (!(pair_weight > amrex::Real(0.0))) {
            continue;
        }
        emit(amrex::Real(0.5) * (begin + end), pair_weight);
    }
}

// Continuous drag over one event-free chord segment -- the one owner for the
// electron, positron and newborn loops. Governing law:
//
//     dE = (L_col,restricted + L_soft,rad)|_E * Int_segment chi(l) dl,
//
// capped so the tracked energy never crosses the charged cutoff: the
// limiter shortens the segment at the crossing and the plan promotes the
// loss to exactly E - E_cut (terminal handoff owns the residual).  Ion
// pairs are dE_col / W_air, prorated by RreaPartitionContinuousIonPairs.
// Rates are exactly linear in the density ratio, so the species policy hands
// over three STP stopping powers frozen at stopping_energy_eV (eV per metre
// per unit density ratio) and density_path(b, e) supplies Int chi dl: this
// module reads no table, and one lookup per refreeze replaces one per material
// interval per integral.  frozen_stopping() is nullary and called ONLY after
// the guard passes, so a sub-cutoff stopping energy never reaches a table.
// Tracked hard-transfer and hard-brems losses stay DISCRETE: hard_transfer
// feeds only the ledger's mean-rate partition (leave it zero where the policy
// is per-event, e.g. positron Bhabha).  Callers own the ledger
// records, the ion-pair emit sink, and the energy update, all from the
// returned plan; on !limit_valid the caller must fail closed.
struct RreaFrozenStoppingPowers {
    amrex::Real collision_eV_per_m = amrex::Real(0.0);
    amrex::Real soft_radiative_eV_per_m = amrex::Real(0.0);
    amrex::Real hard_transfer_eV_per_m = amrex::Real(0.0);
};

struct RreaContinuousDragSegmentPlan {
    amrex::Real effective_end_fraction = amrex::Real(0.0);
    bool applied = false;      // guard passed and a segment was planned
    bool limit_valid = true;   // false: no finite monotone cutoff point
    bool reaches_cutoff = false;
    amrex::Real drag_loss_eV = amrex::Real(0.0);
    amrex::Real collision_loss_eV = amrex::Real(0.0);
    amrex::Real soft_radiative_loss_eV = amrex::Real(0.0);
    amrex::Real hard_transfer_loss_eV = amrex::Real(0.0);
    amrex::Real raw_collision_loss_eV = amrex::Real(0.0);
    // The frozen collision stopping power the plan used, for the ion-pair
    // prorating the caller runs next.
    amrex::Real collision_stp_eV_per_m = amrex::Real(0.0);
};

template <typename DensityPath, typename FrozenStopping>
[[nodiscard]] RREA_HOST_DEVICE inline RreaContinuousDragSegmentPlan
RreaPlanContinuousDragSegment(
    amrex::Real begin_fraction,
    amrex::Real end_fraction,
    amrex::Real kinetic_energy_eV,
    amrex::Real stopping_energy_eV,
    amrex::Real low_energy_cutoff_eV,
    DensityPath&& density_path,
    FrozenStopping&& frozen_stopping)
{
    RreaContinuousDragSegmentPlan plan;
    plan.effective_end_fraction = begin_fraction;
    if (!(end_fraction > begin_fraction)
        || !(kinetic_energy_eV > amrex::Real(0.0))
        || stopping_energy_eV < low_energy_cutoff_eV) {
        return plan;
    }
    RreaFrozenStoppingPowers const stp = frozen_stopping();
    amrex::Real const collision_stp_eV_per_m = stp.collision_eV_per_m;
    plan.collision_stp_eV_per_m = collision_stp_eV_per_m;
    amrex::Real const total_stp_eV_per_m =
        collision_stp_eV_per_m + stp.soft_radiative_eV_per_m;
    auto const limit = RreaLimitContinuousSegmentAtCutoff(
        begin_fraction, end_fraction, kinetic_energy_eV,
        low_energy_cutoff_eV,
        [&](amrex::Real requested_end_fraction) {
            return total_stp_eV_per_m
                * density_path(begin_fraction, requested_end_fraction);
        });
    if (!limit.valid) {
        plan.limit_valid = false;
        return plan;
    }
    plan.applied = true;
    plan.reaches_cutoff = limit.reaches_cutoff;
    plan.effective_end_fraction = limit.effective_end_fraction;
    amrex::Real const path_m =
        density_path(begin_fraction, plan.effective_end_fraction);
    plan.raw_collision_loss_eV = collision_stp_eV_per_m * path_m;
    amrex::Real const raw_soft_eV = stp.soft_radiative_eV_per_m * path_m;
    amrex::Real const available_eV = rrea::kinetic::max(
        amrex::Real(0.0), kinetic_energy_eV - low_energy_cutoff_eV);
    plan.drag_loss_eV = limit.reaches_cutoff
        ? available_eV
        : rrea::kinetic::min(plan.raw_collision_loss_eV + raw_soft_eV, available_eV);
    plan.collision_loss_eV = rrea::kinetic::min(
        plan.raw_collision_loss_eV, plan.drag_loss_eV);
    plan.soft_radiative_loss_eV = rrea::kinetic::max(
        amrex::Real(0.0), plan.drag_loss_eV - plan.collision_loss_eV);
    plan.hard_transfer_loss_eV = rrea::kinetic::min(
        stp.hard_transfer_eV_per_m * path_m, available_eV);
    return plan;
}

RREA_HOST_DEVICE inline void RreaNormalizeDirection(
    amrex::Real& ux, amrex::Real& uy, amrex::Real& uz)
{
    amrex::Real const norm = std::sqrt(ux * ux + uy * uy + uz * uz);
    if (norm <= amrex::Real(0.0)) {
        ux = amrex::Real(0.0);
        uy = amrex::Real(0.0);
        uz = amrex::Real(1.0);
        return;
    }
    ux /= norm;
    uy /= norm;
    uz /= norm;
}

RREA_HOST_DEVICE inline void RreaTransverseBasis(
    amrex::Real dir_x,
    amrex::Real dir_y,
    amrex::Real dir_z,
    amrex::Real& basis1_x,
    amrex::Real& basis1_y,
    amrex::Real& basis1_z,
    amrex::Real& basis2_x,
    amrex::Real& basis2_y,
    amrex::Real& basis2_z)
{
    RreaNormalizeDirection(dir_x, dir_y, dir_z);
    amrex::Real ref_x = amrex::Real(0.0);
    amrex::Real ref_y = amrex::Real(0.0);
    amrex::Real ref_z = amrex::Real(1.0);
    if (std::abs(dir_z) > amrex::Real(0.9)) {
        ref_x = amrex::Real(1.0);
        ref_z = amrex::Real(0.0);
    }

    basis1_x = ref_y * dir_z - ref_z * dir_y;
    basis1_y = ref_z * dir_x - ref_x * dir_z;
    basis1_z = ref_x * dir_y - ref_y * dir_x;
    RreaNormalizeDirection(basis1_x, basis1_y, basis1_z);

    basis2_x = dir_y * basis1_z - dir_z * basis1_y;
    basis2_y = dir_z * basis1_x - dir_x * basis1_z;
    basis2_z = dir_x * basis1_y - dir_y * basis1_x;
    RreaNormalizeDirection(basis2_x, basis2_y, basis2_z);
}

RREA_HOST_DEVICE inline void RreaSampleConeDirection(
    amrex::Real axis_x,
    amrex::Real axis_y,
    amrex::Real axis_z,
    amrex::Real cos_theta,
    amrex::Real azimuth_u,
    amrex::Real& out_x,
    amrex::Real& out_y,
    amrex::Real& out_z)
{
    static constexpr amrex::Real two_pi = 6.283185307179586476925286766559;
    RreaNormalizeDirection(axis_x, axis_y, axis_z);
    amrex::Real b1x, b1y, b1z, b2x, b2y, b2z;
    RreaTransverseBasis(axis_x, axis_y, axis_z, b1x, b1y, b1z, b2x, b2y, b2z);
    cos_theta = rrea::kinetic::min(amrex::Real(1.0), rrea::kinetic::max(amrex::Real(-1.0), cos_theta));
    amrex::Real const sin_theta =
        std::sqrt(rrea::kinetic::max(amrex::Real(0.0), amrex::Real(1.0) - cos_theta * cos_theta));
    amrex::Real const phi = two_pi * azimuth_u;
    amrex::Real const cphi = std::cos(phi);
    amrex::Real const sphi = std::sin(phi);
    out_x = axis_x * cos_theta + sin_theta * (cphi * b1x + sphi * b2x);
    out_y = axis_y * cos_theta + sin_theta * (cphi * b1y + sphi * b2y);
    out_z = axis_z * cos_theta + sin_theta * (cphi * b1z + sphi * b2z);
    RreaNormalizeDirection(out_x, out_y, out_z);
}

// Inverse-transform Poisson sample from a single uniform, capped at k_max
// (few-collision branch; mean < GsFewCollisionThreshold so
// exp(-mean) is safe).  Callers set k_max = ceil(3*threshold) so P(K > k_max) ~
// 1e-6 no matter how the transport configuration configures the threshold -- no silent
// truncation.  Consumes exactly one uniform regardless of k_max.
RREA_HOST_DEVICE inline int RreaPoissonSampleCapped(amrex::Real mean, amrex::Real u, int k_max)
{
    if (mean <= amrex::Real(0.0)) return 0;
    amrex::Real p = std::exp(-mean);
    amrex::Real cum = p;
    int k = 0;
    while (u > cum && k < k_max) {
        ++k;
        p *= mean / static_cast<amrex::Real>(k);
        cum += p;
    }
    return k;
}

RREA_HOST_DEVICE inline amrex::Real RreaUnitCycle(amrex::Real u)
{
    u -= std::floor(u);
    return u < amrex::Real(0.0) ? u + amrex::Real(1.0) : u;
}

RREA_HOST_DEVICE inline amrex::Real RreaMollerSecondaryCosThetaLab(
    amrex::Real primary_kinetic_eV,
    amrex::Real secondary_kinetic_eV)
{
    static constexpr amrex::Real me_c2_eV = kRreaElectronRestEnergyEv;
    primary_kinetic_eV = rrea::kinetic::max(primary_kinetic_eV, amrex::Real(0.0));
    secondary_kinetic_eV = rrea::kinetic::max(secondary_kinetic_eV, amrex::Real(0.0));
    if (primary_kinetic_eV <= amrex::Real(0.0) || secondary_kinetic_eV <= amrex::Real(0.0)) {
        return amrex::Real(1.0);
    }
    // Free-electron Moller two-body lab kinematics for an initially resting
    // target electron.
    amrex::Real const value = secondary_kinetic_eV
        * (primary_kinetic_eV + amrex::Real(2.0) * me_c2_eV)
        / (primary_kinetic_eV * (secondary_kinetic_eV + amrex::Real(2.0) * me_c2_eV));
    return std::sqrt(rrea::kinetic::min(amrex::Real(1.0), rrea::kinetic::max(amrex::Real(0.0), value)));
}

RREA_HOST_DEVICE inline amrex::Real RreaMollerPrimaryCosThetaLab(
    amrex::Real primary_kinetic_eV,
    amrex::Real secondary_kinetic_eV)
{
    static constexpr amrex::Real me_c2_eV = kRreaElectronRestEnergyEv;
    amrex::Real const scattered_kinetic_eV =
        rrea::kinetic::max(amrex::Real(0.0), primary_kinetic_eV - secondary_kinetic_eV);
    if (primary_kinetic_eV <= amrex::Real(0.0) || scattered_kinetic_eV <= amrex::Real(0.0)) {
        return amrex::Real(1.0);
    }
    amrex::Real const value = scattered_kinetic_eV
        * (primary_kinetic_eV + amrex::Real(2.0) * me_c2_eV)
        / (primary_kinetic_eV * (scattered_kinetic_eV + amrex::Real(2.0) * me_c2_eV));
    return std::sqrt(rrea::kinetic::min(amrex::Real(1.0), rrea::kinetic::max(amrex::Real(0.0), value)));
}

// Shared Moller AND Bhabha outgoing-direction sampler: the two lab angles are
// placed on opposite azimuths (the 0.5 cycle offset), each from its own
// two-body cosine.  Transverse momentum balance between the pair is a
// modelling choice deliberately NOT enforced here.
RREA_HOST_DEVICE inline void RreaSampleMollerLabDirections(
    amrex::Real parent_dir_x,
    amrex::Real parent_dir_y,
    amrex::Real parent_dir_z,
    amrex::Real primary_kinetic_eV,
    amrex::Real secondary_kinetic_eV,
    amrex::Real azimuth_u,
    amrex::Real& scattered_parent_dir_x,
    amrex::Real& scattered_parent_dir_y,
    amrex::Real& scattered_parent_dir_z,
    amrex::Real& secondary_dir_x,
    amrex::Real& secondary_dir_y,
    amrex::Real& secondary_dir_z)
{
    RreaNormalizeDirection(parent_dir_x, parent_dir_y, parent_dir_z);
    azimuth_u = RreaUnitCycle(azimuth_u);
    RreaSampleConeDirection(
        parent_dir_x,
        parent_dir_y,
        parent_dir_z,
        RreaMollerSecondaryCosThetaLab(primary_kinetic_eV, secondary_kinetic_eV),
        azimuth_u,
        secondary_dir_x,
        secondary_dir_y,
        secondary_dir_z);
    RreaSampleConeDirection(
        parent_dir_x,
        parent_dir_y,
        parent_dir_z,
        RreaMollerPrimaryCosThetaLab(primary_kinetic_eV, secondary_kinetic_eV),
        RreaUnitCycle(azimuth_u + amrex::Real(0.5)),
        scattered_parent_dir_x,
        scattered_parent_dir_y,
        scattered_parent_dir_z);
}

RREA_HOST_DEVICE inline void RreaApplyPhotonRecoilToChargedDirection(
    amrex::Real initial_kinetic_eV,
    amrex::Real photon_energy_eV,
    amrex::Real photon_dir_x,
    amrex::Real photon_dir_y,
    amrex::Real photon_dir_z,
    amrex::Real& charged_dir_x,
    amrex::Real& charged_dir_y,
    amrex::Real& charged_dir_z)
{
    static constexpr amrex::Real me_c2_eV = kRreaElectronRestEnergyEv;
    photon_energy_eV = rrea::kinetic::max(photon_energy_eV, amrex::Real(0.0));
    if (initial_kinetic_eV <= amrex::Real(0.0) || photon_energy_eV <= amrex::Real(0.0)) {
        return;
    }
    RreaNormalizeDirection(charged_dir_x, charged_dir_y, charged_dir_z);
    RreaNormalizeDirection(photon_dir_x, photon_dir_y, photon_dir_z);
    amrex::Real const p_initial =
        std::sqrt(rrea::kinetic::max(
            amrex::Real(0.0),
            initial_kinetic_eV * (initial_kinetic_eV + amrex::Real(2.0) * me_c2_eV)));
    amrex::Real const px = p_initial * charged_dir_x - photon_energy_eV * photon_dir_x;
    amrex::Real const py = p_initial * charged_dir_y - photon_energy_eV * photon_dir_y;
    amrex::Real const pz = p_initial * charged_dir_z - photon_energy_eV * photon_dir_z;
    amrex::Real const norm = std::sqrt(px * px + py * py + pz * pz);
    if (norm > amrex::Real(0.0)) {
        charged_dir_x = px / norm;
        charged_dir_y = py / norm;
        charged_dir_z = pz / norm;
        RreaNormalizeDirection(charged_dir_x, charged_dir_y, charged_dir_z);
    }
}

RREA_HOST_DEVICE inline void RreaRotateLocalDirectionToAxis(
    amrex::Real axis_x,
    amrex::Real axis_y,
    amrex::Real axis_z,
    amrex::Real local_x,
    amrex::Real local_y,
    amrex::Real local_z,
    amrex::Real& out_x,
    amrex::Real& out_y,
    amrex::Real& out_z)
{
    RreaNormalizeDirection(axis_x, axis_y, axis_z);
    RreaNormalizeDirection(local_x, local_y, local_z);
    amrex::Real b1x, b1y, b1z, b2x, b2y, b2z;
    RreaTransverseBasis(axis_x, axis_y, axis_z, b1x, b1y, b1z, b2x, b2y, b2z);
    out_x = local_x * b1x + local_y * b2x + local_z * axis_x;
    out_y = local_x * b1y + local_y * b2y + local_z * axis_y;
    out_z = local_x * b1z + local_y * b2z + local_z * axis_z;
    RreaNormalizeDirection(out_x, out_y, out_z);
}

// The repeated-hazard event loop of one charged substep: the
// substep_interleaved_v3 SEQUENCING, with every physics decision delegated to
// policy callables supplied by the species loop.  The invariants this core
// owns -- and that its smoke can therefore fail on -- are exactly the
// dt-collapse class: hazards refrozen at the survivor energy each iteration,
// continuous loss applied to the exact sampled event depth, the post-event
// segment frozen at the survivor energy, and a drag-closed channel rejected
// as an explicit null event rather than resampled or converted.
//
// Policies:
//   channel_taus(frozen_eV, begin, end) -> std::array<amrex::Real, 3>
//       per-channel integrated optical depths at the frozen energy (any
//       per-attempt ledger bookkeeping belongs here, at this sequence point);
//   hazard_draws(event_ordinal) -> std::pair<amrex::Real, amrex::Real>
//       the optical-depth and channel-share uniforms (RNG indexing and its
//       overflow abort belong to the policy);
//   depth_to_fraction(tau, begin, end) -> amrex::Real
//       inversion of the sampled optical depth against the frozen total rate;
//   continuous_segment(begin, end, frozen_eV)
//       applies drag over the segment, returns the cutoff limiter;
//   mark_stopped_at(fraction)         records a mid-chord stop (or escape);
//   event_point_at(fraction)          resolves the event position;
//   channel_open_at(channel, energy_eV) -> bool
//       phase-space revalidation predicate;
//   final_state(channel, event_fraction) -> bool
//       samples and books the discrete event; false means the parent was
//       consumed (positron annihilation) and the loop must end.
template <
    typename ChannelTaus,
    typename HazardDraws,
    typename DepthToFraction,
    typename ContinuousSegment,
    typename MarkStoppedAt,
    typename EventPointAt,
    typename ChannelOpenAt,
    typename FinalState>
RREA_HOST_DEVICE inline void RreaRunChargedEventInterleave(
    amrex::Real subcycle_begin,
    amrex::Real subcycle_end,
    amrex::Real const& kinetic_energy_eV,
    amrex::Real cutoff_energy_eV,
    int channel_count,
    std::uint64_t max_event_ordinal,
    char const* /*species_label*/,
    ChannelTaus&& channel_taus,
    HazardDraws&& hazard_draws,
    DepthToFraction&& depth_to_fraction,
    ContinuousSegment&& continuous_segment,
    MarkStoppedAt&& mark_stopped_at,
    EventPointAt&& event_point_at,
    ChannelOpenAt&& channel_open_at,
    FinalState&& final_state)
{
    amrex::Real remaining_begin_fraction = subcycle_begin;
    std::uint64_t event_ordinal = 0;
    while (remaining_begin_fraction < subcycle_end
           && kinetic_energy_eV >= cutoff_energy_eV) {
        if (event_ordinal >= max_event_ordinal) {
            rrea::KineticFail("RREA kinetic " " repeated-hazard event ordinal overflow; "
                  "the uncommitted transaction is rejected");
        }
        // Refreeze: rates and the sampled exponential depth use the SURVIVOR
        // energy of the previous event, re-taken every iteration. Hoisting
        // this out of the loop incorrectly reuses a pre-event rate.
        amrex::Real const frozen_energy_eV = kinetic_energy_eV;
        std::array<amrex::Real, 3> const taus = channel_taus(
            frozen_energy_eV, remaining_begin_fraction, subcycle_end);
        std::pair<amrex::Real, amrex::Real> const draws =
            hazard_draws(event_ordinal);
        RreaRemainingSegmentHazard const hazard =
            RreaResolveRemainingSegmentHazard(
                taus,
                channel_count,
                remaining_begin_fraction,
                subcycle_end,
                draws.first,
                draws.second,
                [&](amrex::Real sampled_optical_depth) {
                    return depth_to_fraction(
                        sampled_optical_depth,
                        remaining_begin_fraction,
                        subcycle_end);
                });
        RreaCompetingHazardSelection const& selection = hazard.selection;
        if (!selection.valid) {
            rrea::KineticFail("RREA kinetic " " competing-hazard selection is invalid");
        }
        if (!selection.event) {
            auto const segment_limit = continuous_segment(
                remaining_begin_fraction, subcycle_end, frozen_energy_eV);
            if (segment_limit.reaches_cutoff) {
                mark_stopped_at(segment_limit.effective_end_fraction);
                break;
            }
            break;
        }
        amrex::Real const event_fraction = hazard.event_fraction;
        auto const& event_segments = hazard.segments;
        if (!event_segments.valid) {
            rrea::KineticFail("RREA kinetic " " sampled event depth is outside its subcycle");
        }
        event_point_at(event_fraction);
        // Continuous loss to the EXACT sampled depth, at the frozen energy.
        auto const segment_limit = continuous_segment(
            event_segments.before_begin_fraction,
            event_segments.before_end_fraction,
            frozen_energy_eV);
        if (segment_limit.reaches_cutoff) {
            mark_stopped_at(segment_limit.effective_end_fraction);
            break;
        }
        bool const open_at_event =
            kinetic_energy_eV >= cutoff_energy_eV
            && channel_open_at(selection.channel, kinetic_energy_eV);
        if (!open_at_event) {
            bool const open_at_frozen =
                frozen_energy_eV >= cutoff_energy_eV
                && channel_open_at(selection.channel, frozen_energy_eV);
            if (RreaClassifyChargedChannelRevalidation(
                    open_at_event, open_at_frozen)
                == RreaChargedChannelRevalidation::Invalid) {
                rrea::KineticFail("RREA kinetic " " selected channel closed before its sampled event depth");
            }
            remaining_begin_fraction = event_fraction;
            ++event_ordinal;
            continue;
        }
        if (!final_state(selection.channel, event_fraction)) {
            return;  // parent consumed: nothing further may run this substep
        }
        if (kinetic_energy_eV <= cutoff_energy_eV) {
            mark_stopped_at(event_fraction);
            break;
        }
        remaining_begin_fraction = event_fraction;
        ++event_ordinal;
    }
}

// The outer substep loop of one charged macro-step: rebuild-the-chord,
// half-kick, event interleave, scatter-and-trailing-kick, commit -- in that
// order, once per substep.  The invariants this skeleton owns: scattering and
// the trailing half-kick run exactly once per substep and only AFTER every
// discrete event; a stopped or consumed parent skips them entirely; a refused
// endpoint or an exiting chord sets exited_domain stickily and ends the loop;
// and completed_substeps counts a microchord only once it is fully committed,
// so elapsed clocks stay (completed + within)/subcycles on every early exit.
//
// Policies:
//   build_substep(subcycle, begin&, end&) -> bool
//       chord rebuild, material intervals, field sub-range and leading
//       half-kick, and the post-kick cutoff stop; false ends the loop (the
//       policy sets exited_domain itself for a zero-length exiting chord);
//   run_events(subcycle, begin, end)      the event interleave;
//   stopped_mid_chord() -> bool           species stop/consumed flags;
//   finish_substep(subcycle, begin, end) -> bool
//       scattering + trailing half-kick; true ends the loop;
//   advance_position(end) -> bool         commit; false = refused endpoint;
//   path_exits_domain() -> bool           the committed chord left the domain.
template <
    typename BuildSubstep,
    typename RunEvents,
    typename StoppedMidChord,
    typename FinishSubstep,
    typename AdvancePosition,
    typename PathExitsDomain>
RREA_HOST_DEVICE inline void RreaRunChargedSubstepInterleave(
    std::uint64_t subcycles,
    amrex::Real const& kinetic_energy_eV,
    amrex::Real cutoff_energy_eV,
    std::uint64_t& completed_substeps,
    bool& exited_domain,
    BuildSubstep&& build_substep,
    RunEvents&& run_events,
    StoppedMidChord&& stopped_mid_chord,
    FinishSubstep&& finish_substep,
    AdvancePosition&& advance_position,
    PathExitsDomain&& path_exits_domain)
{
    for (std::uint64_t subcycle = 0; subcycle < subcycles; ++subcycle) {
        if (kinetic_energy_eV < cutoff_energy_eV) {
            break;
        }
        amrex::Real subcycle_begin = amrex::Real(0.0);
        amrex::Real subcycle_end = amrex::Real(0.0);
        if (!build_substep(subcycle, subcycle_begin, subcycle_end)) {
            break;
        }
        run_events(subcycle, subcycle_begin, subcycle_end);
        if (stopped_mid_chord()) {
            break;
        }
        if (finish_substep(subcycle, subcycle_begin, subcycle_end)) {
            break;
        }
        if (!advance_position(subcycle_end)) {
            exited_domain = true;
            break;
        }
        if (path_exits_domain()) {
            exited_domain = true;
            break;
        }
        completed_substeps = subcycle + 1U;
    }
}

}  // namespace rrea::warpx

#endif
