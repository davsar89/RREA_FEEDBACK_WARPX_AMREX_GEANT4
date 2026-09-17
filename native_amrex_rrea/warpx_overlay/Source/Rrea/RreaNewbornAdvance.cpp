#include "RreaTransportStep.H"

// Newborn residual advance.  Every secondary created at parent-chord elapsed
// fraction f < 1 is flown here through the remaining (1 - f) dt of the step it
// was born into, so no time is created or destroyed at birth.  It runs the SAME
// modules as the live loops -- RreaRunChargedSubstepInterleave /
// RreaRunChargedEventInterleave for the sequencing, RreaPlanContinuousDragSegment
// for drag, apply_gs_elastic_scatter for elastic deflection, and the shared
// charged/photon final states -- with only three POLICY differences, all of
// which are identity rather than physics: the residual dt in place of the whole
// step, RreaNewbornElapsedStepFraction in place of the live elapsed clock, and
// the PARENT's RNG stream indexed by the packed transient lineage in place of
// the particle's own stream.
//
// Pinned by rrea_secondary_group_smoke (the newborn residual budget and the
// group contracts) and rrea_warpx_transport_apply (the assembled step).

namespace rrea::warpx {

bool TransportStep::NewbornFinalPoint(
    rrea::RreaTransportPathSegment const& path,
    amrex::Real fraction,
    amrex::Real& out_x,
    amrex::Real& out_y,
    amrex::Real& out_z) const
{
    amrex::Real const span = fraction - path.in_domain_begin_fraction;
    amrex::Real pullback = amrex::Real(1.0e-9);
    for (int attempt = 0; attempt < 5; ++attempt) {
        amrex::Real const trial = attempt == 0
            ? fraction
            : amrex::max(
                path.in_domain_begin_fraction,
                fraction - pullback * amrex::max(span, amrex::Real(1.0e-12)));
        path.PointAtFullPathFraction(trial, out_x, out_y, out_z);
        if (RreaCartesianPositionInHalfOpenRzDomain(
                out_x, out_y, path.WrapZ(out_z),
                geom.ProbLo(0), geom.ProbHi(0),
                geom.ProbLo(1), geom.ProbHi(1))) {
            out_z = path.WrapZ(out_z);
            return true;
        }
        if (attempt > 0) {
            pullback *= amrex::Real(100.0);
        }
    }
    return false;
}

bool TransportStep::AdvanceChargedNewborn(
    RreaSecondaryParticle& p,
    TransportSideEffects& newborn_fx) const
{
    bool const is_positron = p.species == RreaSecondarySpecies::Positron;
    newborn_fx.SetParentContext(
        p.root_rng_particle_id, p.species, /*transient=*/true);
    amrex::Real const charge_sign =
        is_positron ? amrex::Real(1.0) : amrex::Real(-1.0);
    // Net-chord start for every residual leg below: the birth point, captured
    // before the survivor path overwrites the payload.
    amrex::Real const birth_r_m = std::hypot(p.x_m, p.y_m);
    amrex::Real const birth_z_m = p.z_m;
    amrex::Real const residual_fraction =
        RreaNewbornResidualFraction(p.birth_elapsed_step_fraction);
    amrex::Real kinetic_energy_eV = p.kinetic_or_photon_energy_eV;
    if (!RreaCartesianPositionInHalfOpenRzDomain(
            p.x_m, p.y_m, p.z_m,
            geom.ProbLo(0), geom.ProbHi(0),
            geom.ProbLo(1), geom.ProbHi(1))) {
        amrex::Abort(
            "RREA charged transient was born outside the half-open "
            "domain; the uncommitted transaction is rejected");
    }
    amrex::Real dir_x = p.dir_x;
    amrex::Real dir_y = p.dir_y;
    amrex::Real dir_z = p.dir_z;
    RreaNormalizeDirection(dir_x, dir_y, dir_z);
    bool const hard_moller_discrete_enabled =
        !m_config.debug_disable_hard_moller_secondaries;
    amrex::Real const hard_secondary_threshold_eV =
        coupling.HardMollerSecondaryThresholdEv();
    auto electron_state_policy_at = [&](amrex::Real x_m,
                                        amrex::Real y_m,
                                        amrex::Real z_m,
                                        amrex::Real energy_eV) {
        amrex::Real const r_m = std::hypot(x_m, y_m);
        RreaFieldSample const field = coupling.GatherFieldRZ(0, r_m, z_m);
        amrex::Real const local_density_ratio = positive_or(
            coupling.TransportDensityRatioAtZ(z_m), amrex::Real(1.0));
        return RreaEvaluateElectronStatePolicy(
            energy_eV,
            coupling.LowEnergyCutoffEv(),
            hard_secondary_threshold_eV,
            std::hypot(field.er_v_per_m, field.ez_v_per_m),
            electron_ensemble_mean_drag_ev_per_m(
                tables,
                amrex::max(energy_eV, coupling.LowEnergyCutoffEv()),
                local_density_ratio,
                hard_moller_discrete_enabled));
    };
    auto const birth_state_policy = is_positron
        ? RreaElectronStatePolicyDecision{
            kinetic_energy_eV < coupling.LowEnergyCutoffEv(), false}
        : electron_state_policy_at(p.x_m, p.y_m, p.z_m, kinetic_energy_eV);
    if (birth_state_policy.Demote()) {
        amrex::Real const r_birth = std::hypot(p.x_m, p.y_m);
        if (is_positron) {
            auto const packed = RreaTryPackTransientRngIndex(
                true, p.root_species, p.transient_lineage, 0U, 0U, 0U);
            if (!packed.valid) {
                amrex::Abort(
                    "RREA newborn positron birth RNG index overflow");
            }
            newborn_fx.BeginGroup(
                RreaSecondaryProcess::PositronAnnihilation, true);
            spawn_positron_annihilation(
                newborn_fx,
                p.x_m, p.y_m, p.z_m, r_birth, p.z_m,
                dir_x, dir_y, dir_z,
                p.weight, kinetic_energy_eV, true,
                p.root_rng_particle_id, packed.value,
                p.birth_elapsed_step_fraction);
            newborn_fx.tally.RecordPositronTransport(
                p.weight, kinetic_energy_eV, true);
        } else {
            newborn_fx.BeginGroup(
                RreaSecondaryProcess::LowEnergyDemotion, true);
            newborn_fx.AccumulateLowEnergyElectron(
                0, r_birth, p.z_m, p.weight, amrex::Real(0.0));
            deposit_local_energy_as_ion_pairs(
                newborn_fx, r_birth, p.z_m, p.weight, kinetic_energy_eV);
        }
        newborn_fx.SetInteractionConservationClosure(
            p.weight, kinetic_energy_eV, amrex::Real(0.0));
        newborn_fx.EndGroup();
        return false;
    }
    if (!(residual_fraction > amrex::Real(0.0))) {
        return true;
    }
    amrex::Real const birth_u_mag =
        rrea::RreaElectronProperVelocityFromKineticEv(kinetic_energy_eV);
    amrex::Real const birth_gamma = std::sqrt(
        amrex::Real(1.0)
        + birth_u_mag * birth_u_mag
            / (kSpeedOfLightMPerS * kSpeedOfLightMPerS));
    amrex::Real const residual_dt_s = dt_s * residual_fraction;
    amrex::Real const chord_m = (birth_u_mag / birth_gamma) * residual_dt_s;
    if (!(chord_m > amrex::Real(0.0))) {
        return kinetic_energy_eV >= coupling.LowEnergyCutoffEv();
    }
    amrex::Real const end_x = p.x_m + chord_m * dir_x;
    amrex::Real const end_y = p.y_m + chord_m * dir_y;
    amrex::Real const end_z = p.z_m + chord_m * dir_z;
    // Provisional whole-residual chord, used ONLY to size the substep count.
    // The transported chords are rebuilt per substep from the running
    // direction below, so this one is never integrated along.
    RreaTransportPathSegment const driver_path = ClipRzTransportPath(
        std::hypot(p.x_m, p.y_m), std::atan2(p.y_m, p.x_m), p.z_m,
        std::hypot(end_x, end_y), std::atan2(end_y, end_x), end_z,
        geom.ProbLo(0), geom.ProbHi(0),
        geom.ProbLo(1), geom.ProbHi(1),
        periodic_z);
    if (!(driver_path.full_length_m > amrex::Real(0.0))
        || !(driver_path.in_domain_length_m > amrex::Real(0.0))) {
        if (driver_path.exits_domain) {
            newborn_fx.BeginGroup(
                RreaSecondaryProcess::ParticleEscape, true);
            newborn_fx.SetChargedEscapeConservationClosure(
                p.weight, kinetic_energy_eV);
            newborn_fx.AccumulateChargedEscape(
                0, birth_r_m, birth_z_m,
                RreaChargedBoundaryCrossingForPath(driver_path),
                charge_sign * p.weight,
                kRreaSegmentChannelCreated | kRreaSegmentChannelEscaped);
            newborn_fx.EndGroup();
            return false;
        }
        return true;
    }
    auto const driver_intervals = build_material_path_intervals(
        coupling, driver_path, density_ratio);
    if (driver_intervals.empty()) {
        amrex::Abort(
            "RREA charged transient path has positive in-domain "
            "length but no material intervals");
    }
    // The chord actually transported along.  Declared here rather than in the
    // substep loop because the policies below close over it; the loop
    // reassigns it per microchord, which is what makes deflection bend the
    // subsequent path instead of only the momentum.  Seeded with the driver
    // chord so the post-loop final point is well defined even if the first
    // microchord degenerates.
    RreaTransportPathSegment path = driver_path;
    std::vector<MaterialPathInterval> material_intervals = driver_intervals;
    auto const& newborn_certified_range = is_positron
        ? tables.PositronCertifiedEnergyRangeEv()
        : tables.ElectronCertifiedEnergyRangeEv();
    auto require_certified_energy = [&](char const* what) {
        if (!std::isfinite(static_cast<double>(kinetic_energy_eV))
            || !newborn_certified_range.Contains(kinetic_energy_eV)) {
            amrex::Abort(
                std::string("RREA newborn charged ") + what
                + " left the certified transport-energy range; the "
                  "uncommitted transaction is rejected");
        }
    };
    // Substep drivers remain GS batching plus impulse/drag granularity.
    // Repeated discrete hazards are sampled on every remaining substep
    // segment and need no optical-depth cap.
    amrex::Real const driver_density_path_m = density_path_m(
        driver_intervals, driver_path,
        driver_path.in_domain_begin_fraction,
        driver_path.in_domain_end_fraction);
    amrex::Real const gs_n_total =
        elastic_inverse_length_stp(tables, is_positron, kinetic_energy_eV)
        * driver_density_path_m;
    amrex::Real const drag_estimate_eV =
        (is_positron
         ? tables.PositronCollisionLoss(kinetic_energy_eV, amrex::Real(1.0))
         : electron_realized_continuous_loss_ev_per_m(
             tables, kinetic_energy_eV, amrex::Real(1.0)))
        * driver_density_path_m;
    amrex::Real const whole_work_eV = path_field_work_and_average(
        coupling, driver_path, charge_sign,
        amrex::Real(0.0), amrex::Real(1.0)).work_eV;
    amrex::Real const impulse_scale_eV = amrex::max(
        m_config.transport_max_field_impulse_fraction * kinetic_energy_eV,
        amrex::max(
            m_config.transport_field_impulse_floor_eV, amrex::Real(1.0)));
    // The three budgets above are integrated over the driver chord's IN-DOMAIN
    // span, but the substep loop transports the whole residual regardless of
    // where that chord clipped: each microchord is speed*sub_dt long and
    // sub_dt*substeps == residual_dt always.  Without this rescale a driver
    // clipped to a fraction phi under-sizes every budget by phi, and the
    // realised per-substep impulse and GS batch overshoot their configured caps
    // by 1/phi -- exactly for the boundary population that dominates escape.
    amrex::Real const driver_span = amrex::max(
        driver_path.in_domain_end_fraction, amrex::Real(1.0e-6));
    std::uint64_t const substeps = std::min(
        std::max(
            substep_driver_count(
                amrex::min(
                    gs_n_total / driver_span,
                    m_config.transport_gs_total_collision_cap),
                amrex::max(
                    m_config.transport_max_gs_collisions_per_substep,
                    amrex::Real(1.0))),
            substep_driver_count(
                (std::abs(whole_work_eV) + drag_estimate_eV) / driver_span,
                impulse_scale_eV)),
        m_config.max_subcycles);
    if (substeps >= kRreaTransientSegmentCapacity) {
        amrex::Abort(
            "RREA newborn charged substep count is outside its "
            "14-bit transient RNG budget");
    }
    amrex::Real positron_total_loss_eV = amrex::Real(0.0);
    auto apply_field_half_kick = [&](
        PathFieldSubRange const& field_range, amrex::Real half_dt_s) {
        static_cast<void>(RreaApplyChargedFieldHalfKick(
            field_range, half_dt_s, charge_sign,
            kinetic_energy_eV, dir_x, dir_y, dir_z));
    };
    // Same drag module as the live loops (RreaPlanContinuousDragSegment); the
    // newborn POLICY serves both species behind is_positron -- electron:
    // partition record with the hard-Moller mean rate; positron: soft record
    // plus the at-rest budget accumulator -- and the ion-pair sink clamps to
    // the half-open domain (a newborn chord may graze a face).
    auto apply_newborn_drag = [&](
        amrex::Real begin_fraction,
        amrex::Real end_fraction,
        amrex::Real stopping_energy_eV) -> RreaContinuousSegmentLimit {
        amrex::Real const kinetic_energy_before_segment_eV = kinetic_energy_eV;
        auto density_path = [&](amrex::Real b, amrex::Real e) {
            return density_path_m(material_intervals, path, b, e);
        };
        auto const plan = RreaPlanContinuousDragSegment(
            begin_fraction, end_fraction, kinetic_energy_eV,
            stopping_energy_eV, coupling.LowEnergyCutoffEv(),
            density_path,
            [&] {
                return RreaFrozenStoppingPowers{
                    RreaFrozenStpRate(is_positron
                        ? tables.PositronCollisionLoss(
                            stopping_energy_eV, amrex::Real(1.0))
                        : tables.ElectronCollisionLoss(
                            stopping_energy_eV, amrex::Real(1.0))),
                    RreaFrozenStpRate(is_positron
                        ? tables.PositronSoftRadiativeDrag(
                            stopping_energy_eV, amrex::Real(1.0))
                        : tables.ElectronSoftRadiativeDrag(
                            stopping_energy_eV, amrex::Real(1.0))),
                    (!is_positron && hard_moller_discrete_enabled
                     && RreaHardMollerChannelOpen(
                         stopping_energy_eV, hard_secondary_threshold_eV))
                    ? RreaFrozenStpRate(tables.ElectronHardMollerEnergyLoss(
                        stopping_energy_eV, amrex::Real(1.0)))
                    : amrex::Real(0.0)};
            });
        if (!plan.limit_valid) {
            amrex::Abort(
                "RREA newborn continuous stopping could not locate a "
                "finite monotone cutoff point");
        }
        if (!plan.applied) {
            return RreaContinuousSegmentLimit{begin_fraction, false, true};
        }
        newborn_fx.tally.RecordContinuousSoftBrems(
            p.weight, plan.soft_radiative_loss_eV);
        if (is_positron) {
            positron_total_loss_eV += plan.drag_loss_eV;
        } else {
            newborn_fx.tally.RecordContinuousLossPartition(
                p.weight,
                plan.drag_loss_eV,
                plan.collision_loss_eV,
                plan.hard_transfer_loss_eV);
        }
        {
            amrex::Real const diagnostic_loss_eV = tables.WAirEvPerIonPair();
            RreaPartitionContinuousIonPairs(
                material_intervals,
                path.full_length_m,
                begin_fraction,
                plan.effective_end_fraction,
                plan.collision_loss_eV,
                plan.raw_collision_loss_eV,
                p.weight,
                diagnostic_loss_eV,
                plan.collision_stp_eV_per_m,
                [&](amrex::Real midpoint_fraction,
                    amrex::Real pair_weight) {
                    amrex::Real source_x = amrex::Real(0.0);
                    amrex::Real source_y = amrex::Real(0.0);
                    amrex::Real source_z = amrex::Real(0.0);
                    path.PointAtFullPathFraction(
                        midpoint_fraction, source_x, source_y, source_z);
                    amrex::Real source_r = std::hypot(source_x, source_y);
                    amrex::Real source_wrapped_z = path.WrapZ(source_z);
                    if (!RreaRzPositionInHalfOpenDomain(
                            source_r, source_wrapped_z,
                            geom.ProbLo(0), geom.ProbHi(0),
                            geom.ProbLo(1), geom.ProbHi(1))) {
                        // Roundoff on a clipped face: fall back to the
                        // (validated in-domain) birth position rather than
                        // forming an out-of-domain source; the complete work
                        // graph is collectively validated before replay.
                        source_r = std::hypot(p.x_m, p.y_m);
                        source_wrapped_z = p.z_m;
                    }
                    newborn_fx.AccumulateIonizationEvent(
                        0, source_r, source_wrapped_z, pair_weight,
                        is_positron
                            ? amrex::Real(0.0)
                            : amrex::Real(0.5) * coupling.LowEnergyCutoffEv(),
                        diagnostic_loss_eV);
                    newborn_fx.tally.RecordTransportIonization(
                        pair_weight, diagnostic_loss_eV, true, false);
                });
        }
        if (!is_positron) {
            defer_cd_electron_plane_crossings(
                newborn_fx, coupling, tables, path, material_intervals,
                begin_fraction, plan.effective_end_fraction,
                p.weight, kinetic_energy_before_segment_eV,
                stopping_energy_eV, plan.drag_loss_eV,
                hard_moller_discrete_enabled);
        }
        kinetic_energy_eV = amrex::max(
            amrex::Real(0.0), kinetic_energy_eV - plan.drag_loss_eV);
        if (plan.reaches_cutoff) {
            kinetic_energy_eV = coupling.LowEnergyCutoffEv();
        }
        return RreaContinuousSegmentLimit{
            plan.effective_end_fraction, plan.reaches_cutoff, true};
    };
    auto apply_newborn_gs = [&](
        amrex::Real begin_fraction,
        amrex::Real end_fraction,
        std::uint64_t substep_idx) {
        // Same GS application unit as the live loops; only the RNG identity
        // differs (Newborn* channels on the PARENT stream, packed transient
        // lineage instead of substep<<8).
        GsScatterChannels const channels{
            is_positron ? RreaRngChannel::NewbornPositronGsFewCount
                        : RreaRngChannel::NewbornElectronGsFewCount,
            is_positron ? RreaRngChannel::NewbornPositronGsFewScatter
                        : RreaRngChannel::NewbornElectronGsFewScatter,
            is_positron ? RreaRngChannel::NewbornPositronGsFewAzimuth
                        : RreaRngChannel::NewbornElectronGsFewAzimuth,
            is_positron ? RreaRngChannel::NewbornPositronGsAngle
                        : RreaRngChannel::NewbornElectronGsAngle,
            is_positron ? RreaRngChannel::NewbornPositronGsAzimuth
                        : RreaRngChannel::NewbornElectronGsAzimuth};
        auto gs_key = [&](RreaRngChannel ch, std::uint64_t inner) {
            auto const packed = RreaTryPackTransientRngIndex(
                true, p.root_species, p.transient_lineage,
                substep_idx, 0U, inner);
            if (!packed.valid) {
                amrex::Abort(
                    "RREA newborn GS RNG index overflow; the "
                    "uncommitted transaction is rejected");
            }
            return RreaRngKey{
                coupling.RngSeed(),
                p.root_rng_particle_id,
                static_cast<std::uint64_t>(std::max(step, 0)),
                packed.value,
                static_cast<std::uint64_t>(ch)};
        };
        apply_gs_elastic_scatter(
            material_intervals, path, begin_fraction, end_fraction, tables,
            is_positron, channels, gs_key,
            kinetic_energy_eV, dir_x, dir_y, dir_z);
    };
    // Each substep rebuilds its chord from the current direction so elastic
    // deflection, recoil and field work follow the same path.
    amrex::Real run_x = p.x_m;
    amrex::Real run_y = p.y_m;
    amrex::Real run_z = p.z_m;
    amrex::Real const sub_dt_s =
        residual_dt_s / static_cast<amrex::Real>(substeps);
    amrex::Real final_fraction = amrex::Real(0.0);
    bool terminal_interaction = false;
    bool terminal_at_path_point = false;
    bool exited_domain = false;
    std::uint64_t completed_substeps = 0;
    auto elapsed_fraction_at = [&](amrex::Real path_fraction) {
        return RreaNewbornElapsedStepFraction(
            p.birth_elapsed_step_fraction, residual_fraction,
            completed_substeps, substeps, path_fraction);
    };
    auto mark_newborn_stopped_at = [&](amrex::Real path_fraction) {
        final_fraction = path_fraction;
        terminal_at_path_point = true;
    };
    PathFieldSubRange substep_field{};
    amrex::Real substep_dt_s = amrex::Real(0.0);
    RreaRunChargedSubstepInterleave(
        substeps,
        kinetic_energy_eV,
        coupling.LowEnergyCutoffEv(),
        completed_substeps,
        exited_domain,
        [&](std::uint64_t,
            amrex::Real& subcycle_begin,
            amrex::Real& subcycle_end) {
            // The chord the particle will actually follow over this substep,
            // from the state it is actually in.  The header primitive owns the
            // speed law, the direction normalisation and the endpoint clip, so
            // the three species cannot drift apart on any of them.
            auto const chord = RreaBuildRunningChargedChord(
                RreaRunningChargedState{
                    run_x, run_y, run_z, dir_x, dir_y, dir_z,
                    kinetic_energy_eV},
                sub_dt_s,
                residual_dt_s,
                geom.ProbLo(0), geom.ProbHi(0),
                geom.ProbLo(1), geom.ProbHi(1),
                periodic_z);
            if (!chord.valid) {
                return false;
            }
            path = chord.path;
            if (!(path.full_length_m > amrex::Real(0.0))
                || !(path.in_domain_length_m > amrex::Real(0.0))) {
                exited_domain = path.exits_domain;
                // Resolve a degenerate chord at its origin.
                final_fraction = amrex::Real(0.0);
                return false;
            }
            material_intervals = build_material_path_intervals(
                coupling, path, density_ratio);
            if (material_intervals.empty()) {
                amrex::Abort(
                    "RREA charged transient microchord has positive "
                    "in-domain length but no material intervals");
            }
            subcycle_begin = path.in_domain_begin_fraction;
            subcycle_end = path.in_domain_end_fraction;
            final_fraction = subcycle_end;
            substep_field = path_field_work_and_average(
                coupling, path, charge_sign, subcycle_begin, subcycle_end);
            // Time ACTUALLY spent on this chord.  The field work is already
            // integrated only over [begin, end], but the half-kick's ROTATION
            // scales with dt, so handing it the whole substep would over-rotate
            // a clipped microchord by 1/(end-begin) -- energy right, direction
            // wrong, which no conservation ledger would catch.
            substep_dt_s = sub_dt_s * (subcycle_end - subcycle_begin);
            apply_field_half_kick(
                substep_field, amrex::Real(0.5) * substep_dt_s);
            if (kinetic_energy_eV < coupling.LowEnergyCutoffEv()) {
                final_fraction = subcycle_begin;
                terminal_at_path_point = true;
                return false;
            }
            require_certified_energy("field impulse");
            return true;
        },
        [&](std::uint64_t subcycle,
            amrex::Real subcycle_begin,
            amrex::Real subcycle_end) {
            // Event-loop policies: the newborn's physics content.  The
            // substep_interleaved_v3 sequencing -- refreeze, exact-depth drag,
            // explicit null event on a drag-closed channel -- lives in
            // RreaRunChargedEventInterleave, exactly as for the live loops.
            amrex::Real event_x = amrex::Real(0.0);
            amrex::Real event_y = amrex::Real(0.0);
            amrex::Real event_z = amrex::Real(0.0);
            amrex::Real event_r = amrex::Real(0.0);
            amrex::Real event_elapsed = amrex::Real(1.0);
            std::uint64_t packed_rng_index = 0;
            amrex::Real total_stp = amrex::Real(0.0);
            RreaRunChargedEventInterleave(
                subcycle_begin,
                subcycle_end,
                kinetic_energy_eV,
                coupling.LowEnergyCutoffEv(),
                is_positron ? 3 : 2,
                kRreaTransientEventCapacity,
                is_positron ? "newborn positron" : "newborn electron",
                [&](amrex::Real frozen_energy_eV,
                    amrex::Real tau_begin_fraction,
                    amrex::Real tau_end_fraction) {
                    // The newborn residual pass runs the SAME channels through
                    // the SAME gated rate bundle as the live loops, so a
                    // newborn cannot acquire physics its parent did not have.
                    auto const stp = RreaChargedCompetingStpRates(
                        tables, is_positron, hard_secondary_threshold_eV,
                        coupling.PhotonCutoffEv(), frozen_energy_eV,
                        hard_moller_discrete_enabled);
                    total_stp = stp[0] + stp[1] + stp[2];
                    amrex::Real const segment_density_path_m = density_path_m(
                        material_intervals, path,
                        tau_begin_fraction, tau_end_fraction);
                    std::array<amrex::Real, 3> const taus{
                        stp[0] * segment_density_path_m,
                        stp[1] * segment_density_path_m,
                        stp[2] * segment_density_path_m};
                    if (!is_positron) {
                        amrex::Real const total_tau = taus[0] + taus[1];
                        newborn_fx.tally.RecordHardMollerExpectation(
                            p.weight
                            * (total_tau > amrex::Real(0.0)
                               ? probability_from_optical_depth(total_tau)
                                   * taus[0] / total_tau
                               : amrex::Real(0.0)));
                    }
                    return taus;
                },
                [&](std::uint64_t draw_event_ordinal) {
                    auto const packed = RreaTryPackTransientRngIndex(
                        true, p.root_species, p.transient_lineage,
                        subcycle, draw_event_ordinal, 0U);
                    if (!packed.valid) {
                        amrex::Abort(
                            "RREA newborn charged repeated-hazard RNG index "
                            "overflow; the uncommitted transaction is rejected");
                    }
                    packed_rng_index = packed.value;
                    return std::pair<amrex::Real, amrex::Real>{
                        RreaRng::Uniform01(RreaRngKey{
                            coupling.RngSeed(), p.root_rng_particle_id,
                            static_cast<std::uint64_t>(std::max(step, 0)),
                            packed_rng_index,
                            static_cast<std::uint64_t>(
                                RreaRngChannel::NewbornCompetingOpticalDepth)}),
                        RreaRng::Uniform01(RreaRngKey{
                            coupling.RngSeed(), p.root_rng_particle_id,
                            static_cast<std::uint64_t>(std::max(step, 0)),
                            packed_rng_index,
                            static_cast<std::uint64_t>(
                                RreaRngChannel::NewbornCompetingChannel)})};
                },
                [&](amrex::Real sampled_optical_depth,
                    amrex::Real invert_begin_fraction,
                    amrex::Real invert_end_fraction) {
                    return fraction_at_density_path_m(
                        material_intervals, path,
                        invert_begin_fraction, invert_end_fraction,
                        sampled_optical_depth / total_stp);
                },
                apply_newborn_drag,
                mark_newborn_stopped_at,
                [&](amrex::Real point_event_fraction) {
                    point_and_radius_at_fraction(
                        path, point_event_fraction,
                        event_x, event_y, event_z, event_r);
                    event_z = path.WrapZ(event_z);
                    event_elapsed = elapsed_fraction_at(point_event_fraction);
                },
                [&](int channel, amrex::Real energy_eV) {
                    return RreaChargedChannelOpenAt(
                        tables, is_positron, hard_secondary_threshold_eV,
                        coupling.PhotonCutoffEv(), channel, energy_eV,
                        hard_moller_discrete_enabled);
                },
                [&](int channel, amrex::Real event_fraction) {
                    RreaInteractionSite const site{
                        event_x, event_y, event_z, event_r, event_elapsed};
                    auto rng_key = [&](RreaRngChannel rng_channel) {
                        return RreaRngKey{
                            coupling.RngSeed(), p.root_rng_particle_id,
                            static_cast<std::uint64_t>(std::max(step, 0)),
                            packed_rng_index,
                            static_cast<std::uint64_t>(rng_channel)};
                    };
                    if (channel == 0) {
                        if (is_positron) {
                            RreaApplyBhabhaFinalState(
                                newborn_fx, tables, rng_key, site, p.weight,
                                hard_secondary_threshold_eV,
                                kinetic_energy_eV, dir_x, dir_y, dir_z);
                        } else {
                            RreaApplyHardMollerFinalState(
                                newborn_fx, tables, rng_key, site, p.weight,
                                hard_secondary_threshold_eV,
                                kinetic_energy_eV, dir_x, dir_y, dir_z);
                        }
                    } else if (channel == 1) {
                        RreaApplyBremsFinalState(
                            newborn_fx, tables, rng_key,
                            [&](amrex::Real local_energy_eV) {
                                deposit_local_energy_as_ion_pairs(
                                    newborn_fx, event_r, event_z,
                                    p.weight, local_energy_eV);
                            },
                            site, p.weight, charge_sign,
                            coupling.PhotonCutoffEv(),
                            kinetic_energy_eV, dir_x, dir_y, dir_z);
                    } else if (channel == 2) {
                        newborn_fx.BeginGroup(
                            RreaSecondaryProcess::PositronAnnihilation, true);
                        spawn_positron_annihilation(
                            newborn_fx,
                            event_x, event_y, event_z, event_r, event_z,
                            dir_x, dir_y, dir_z,
                            p.weight, kinetic_energy_eV, false,
                            p.root_rng_particle_id, packed_rng_index,
                            event_elapsed);
                        newborn_fx.SetInteractionConservationClosure(
                            p.weight, kinetic_energy_eV, amrex::Real(0.0));
                        // Newborn residual leg: birth to the sampled in-flight
                        // annihilation point.
                        newborn_fx.AccumulateChargedSegment(
                            0, birth_r_m, birth_z_m, event_r, event_z,
                            charge_sign * p.weight,
                            kRreaSegmentChannelCreated
                                | kRreaSegmentChannelAnnihilated);
                        newborn_fx.EndGroup();
                        terminal_interaction = true;
                        final_fraction = event_fraction;
                        return false;
                    }
                    return true;
                });
        },
        [&] { return terminal_interaction || terminal_at_path_point; },
        [&](std::uint64_t subcycle,
            amrex::Real subcycle_begin,
            amrex::Real subcycle_end) {
            if (kinetic_energy_eV >= coupling.LowEnergyCutoffEv()) {
                apply_newborn_gs(subcycle_begin, subcycle_end, subcycle);
            }
            apply_field_half_kick(
                substep_field, amrex::Real(0.5) * substep_dt_s);
            if (kinetic_energy_eV < coupling.LowEnergyCutoffEv()) {
                final_fraction = subcycle_end;
                terminal_at_path_point = true;
                return true;
            }
            require_certified_energy("trailing field impulse");
            return false;
        },
        // Commit this microchord: the next one is rebuilt from here, along
        // whatever direction the scatter above left behind.
        [&](amrex::Real subcycle_end) {
            return RreaAdvanceRunningChargedPosition(
                path, subcycle_end,
                geom.ProbLo(0), geom.ProbHi(0),
                geom.ProbLo(1), geom.ProbHi(1),
                run_x, run_y, run_z);
        },
        [&] { return path.exits_domain; });
    if (is_positron && positron_total_loss_eV > amrex::Real(0.0)) {
        newborn_fx.tally.RecordPositronTransport(
            p.weight, positron_total_loss_eV, false);
    }
    if (terminal_interaction) {
        return false;
    }
    amrex::Real final_x = amrex::Real(0.0);
    amrex::Real final_y = amrex::Real(0.0);
    amrex::Real final_z = amrex::Real(0.0);
    if (!NewbornFinalPoint(path, final_fraction, final_x, final_y, final_z)) {
        amrex::Abort(
            "RREA newborn charged transport could not place its final "
            "point in the half-open domain; the uncommitted "
            "transaction is rejected");
    }
    amrex::Real const final_r = std::hypot(final_x, final_y);
    amrex::Real const final_elapsed = elapsed_fraction_at(final_fraction);
    auto const final_state_policy = is_positron
        ? RreaElectronStatePolicyDecision{
            kinetic_energy_eV < coupling.LowEnergyCutoffEv(), false}
        : electron_state_policy_at(
            final_x, final_y, final_z, kinetic_energy_eV);
    if (terminal_at_path_point || final_state_policy.Demote()) {
        std::uint64_t channels = kRreaSegmentChannelCreated;
        if (is_positron) {
            auto const packed = RreaTryPackTransientRngIndex(
                true, p.root_species, p.transient_lineage,
                std::min(substeps, kRreaTransientSegmentCapacity - 1U),
                0U, 0U);
            if (!packed.valid) {
                amrex::Abort(
                    "RREA newborn positron terminal RNG index overflow");
            }
            newborn_fx.BeginGroup(
                RreaSecondaryProcess::PositronAnnihilation, true);
            spawn_positron_annihilation(
                newborn_fx,
                final_x, final_y, final_z, final_r, final_z,
                dir_x, dir_y, dir_z,
                p.weight, kinetic_energy_eV, true,
                p.root_rng_particle_id, packed.value,
                final_elapsed);
            newborn_fx.tally.RecordPositronTransport(
                p.weight, kinetic_energy_eV, true);
            channels |= kRreaSegmentChannelAnnihilated;
        } else {
            newborn_fx.BeginGroup(
                RreaSecondaryProcess::LowEnergyDemotion, true);
            newborn_fx.AccumulateLowEnergyElectron(
                0, final_r, final_z, p.weight, amrex::Real(0.0));
            deposit_local_energy_as_ion_pairs(
                newborn_fx, final_r, final_z, p.weight, kinetic_energy_eV);
            channels |= kRreaSegmentChannelDemoted;
        }
        newborn_fx.SetInteractionConservationClosure(
            p.weight, kinetic_energy_eV, amrex::Real(0.0));
        newborn_fx.AccumulateChargedSegment(
            0, birth_r_m, birth_z_m, final_r, final_z,
            charge_sign * p.weight, channels);
        newborn_fx.EndGroup();
        return false;
    }
    // Any microchord that left the domain ends the residual advance; the flag
    // is sticky because only the last chord is still in scope.
    if (exited_domain) {
        newborn_fx.BeginGroup(RreaSecondaryProcess::ParticleEscape, true);
        newborn_fx.SetChargedEscapeConservationClosure(
            p.weight, kinetic_energy_eV);
        newborn_fx.AccumulateChargedEscape(
            0, birth_r_m, birth_z_m,
            RreaChargedBoundaryCrossingForPath(path),
            charge_sign * p.weight,
            kRreaSegmentChannelCreated | kRreaSegmentChannelEscaped);
        newborn_fx.EndGroup();
        return false;
    }
    p.x_m = final_x;
    p.y_m = final_y;
    p.z_m = final_z;
    p.dir_x = dir_x;
    p.dir_y = dir_y;
    p.dir_z = dir_z;
    p.kinetic_or_photon_energy_eV = kinetic_energy_eV;
    p.birth_elapsed_step_fraction = amrex::Real(1.0);
    // Surviving newborn: birth to the committed final point of its residual
    // leg (charge entered the kinetic population mid-step).
    newborn_fx.AccumulateChargedSegment(
        0, birth_r_m, birth_z_m, final_r, final_z,
        charge_sign * p.weight, kRreaSegmentChannelCreated);
    return true;
}

bool TransportStep::AdvancePhotonNewborn(
    RreaSecondaryParticle& p,
    TransportSideEffects& newborn_fx) const
{
    newborn_fx.SetParentContext(
        p.root_rng_particle_id, RreaSecondarySpecies::Photon,
        /*transient=*/true);
    amrex::Real const residual_fraction =
        RreaNewbornResidualFraction(p.birth_elapsed_step_fraction);
    if (!(residual_fraction > amrex::Real(0.0))) {
        return p.kinetic_or_photon_energy_eV >= coupling.PhotonCutoffEv();
    }
    if (!RreaCartesianPositionInHalfOpenRzDomain(
            p.x_m, p.y_m, p.z_m,
            geom.ProbLo(0), geom.ProbHi(0),
            geom.ProbLo(1), geom.ProbHi(1))) {
        amrex::Abort(
            "RREA photon transient was born outside the half-open "
            "domain; the uncommitted transaction is rejected");
    }
    static constexpr amrex::Real c_m_per_s = kRreaSpeedOfLightMPerS;
    amrex::Real dir_x = p.dir_x;
    amrex::Real dir_y = p.dir_y;
    amrex::Real dir_z = p.dir_z;
    RreaNormalizeDirection(dir_x, dir_y, dir_z);
    amrex::Real photon_energy_eV = p.kinetic_or_photon_energy_eV;
    if (photon_energy_eV < coupling.PhotonCutoffEv()) {
        newborn_fx.BeginGroup(
            RreaSecondaryProcess::LowEnergyDemotion, true);
        deposit_local_energy_as_ion_pairs(
            newborn_fx, std::hypot(p.x_m, p.y_m), p.z_m,
            p.weight, photon_energy_eV);
        newborn_fx.SetInteractionConservationClosure(
            p.weight, photon_energy_eV, amrex::Real(0.0));
        newborn_fx.EndGroup();
        return false;
    }
    auto commit_photon_escape = [&]() {
        newborn_fx.BeginGroup(RreaSecondaryProcess::PhotonEscape, true);
        newborn_fx.tally.RecordPhotonEscape(p.weight, photon_energy_eV);
        newborn_fx.AddEscapedConservationEnergy(p.weight, photon_energy_eV);
        newborn_fx.SetInteractionConservationClosure(
            p.weight, photon_energy_eV, amrex::Real(0.0));
        newborn_fx.EndGroup();
    };
    amrex::Real const chord_m = c_m_per_s * dt_s * residual_fraction;
    amrex::Real const end_x = p.x_m + chord_m * dir_x;
    amrex::Real const end_y = p.y_m + chord_m * dir_y;
    amrex::Real const end_z = p.z_m + chord_m * dir_z;
    RreaTransportPathSegment path = ClipRzTransportPath(
        std::hypot(p.x_m, p.y_m), std::atan2(p.y_m, p.x_m), p.z_m,
        std::hypot(end_x, end_y), std::atan2(end_y, end_x), end_z,
        geom.ProbLo(0), geom.ProbHi(0),
        geom.ProbLo(1), geom.ProbHi(1),
        periodic_z);
    if (!(path.full_length_m > amrex::Real(0.0))
        || !(path.in_domain_length_m > amrex::Real(0.0))) {
        if (path.exits_domain) {
            commit_photon_escape();
            return false;
        }
        return true;
    }
    amrex::Real elapsed_path_start = p.birth_elapsed_step_fraction;
    std::uint64_t event_ordinal = 0;
    while (true) {
        auto const packed = RreaTryPackTransientRngIndex(
            true, p.root_species, p.transient_lineage,
            0U, event_ordinal, 0U);
        if (!packed.valid) {
            amrex::Abort(
                "RREA newborn photon repeated-hazard RNG index "
                "overflow; the uncommitted transaction is rejected");
        }
        // Same material path -- and the same piecewise density-path inversion,
        // whose interaction point is misplaced by at most h^2/8H = 9 um on a
        // 0.75 m chord -- as the live photon loop.
        auto const photon_intervals = build_material_path_intervals(
            coupling, path, density_ratio);
        amrex::Real const mfp = tables.PhotonFeedbackMeanFreePath(
            photon_energy_eV, amrex::Real(1.0));
        amrex::Real const tau = density_path_m(
            photon_intervals, path,
            path.in_domain_begin_fraction,
            path.in_domain_end_fraction) / mfp;
        RreaRemainingSegmentHazard const hazard =
            RreaResolveRemainingSegmentHazard(
                {tau, amrex::Real(0.0), amrex::Real(0.0)},
                1,
                path.in_domain_begin_fraction,
                path.in_domain_end_fraction,
                RreaRng::Uniform01(RreaRngKey{
                    coupling.RngSeed(), p.root_rng_particle_id,
                    static_cast<std::uint64_t>(std::max(step, 0)),
                    packed.value,
                    static_cast<std::uint64_t>(
                        RreaRngChannel::NewbornCompetingOpticalDepth)}),
                amrex::Real(0.0),
                [&](amrex::Real sampled_optical_depth) {
                    return fraction_at_density_path_m(
                        photon_intervals, path,
                        path.in_domain_begin_fraction,
                        path.in_domain_end_fraction,
                        sampled_optical_depth * mfp);
                });
        if (!hazard.selection.event) {
            amrex::Real final_x = amrex::Real(0.0);
            amrex::Real final_y = amrex::Real(0.0);
            amrex::Real final_z = amrex::Real(0.0);
            if (!NewbornFinalPoint(
                    path, path.in_domain_end_fraction,
                    final_x, final_y, final_z)) {
                amrex::Abort(
                    "RREA newborn photon could not place its final "
                    "point; the uncommitted transaction is rejected");
            }
            if (path.exits_domain) {
                commit_photon_escape();
                return false;
            }
            p.x_m = final_x; p.y_m = final_y; p.z_m = final_z;
            p.dir_x = dir_x; p.dir_y = dir_y; p.dir_z = dir_z;
            p.kinetic_or_photon_energy_eV = photon_energy_eV;
            p.birth_elapsed_step_fraction = amrex::Real(1.0);
            return true;
        }
        amrex::Real const event_fraction = hazard.event_fraction;
        amrex::Real const event_distance = path.full_length_m
            * (event_fraction - path.in_domain_begin_fraction);
        amrex::Real const remaining_m = amrex::max(
            amrex::Real(0.0), path.full_length_m - event_distance);
        amrex::Real event_x = amrex::Real(0.0);
        amrex::Real event_y = amrex::Real(0.0);
        amrex::Real event_z = amrex::Real(0.0);
        amrex::Real event_r = amrex::Real(0.0);
        point_and_radius_at_fraction(
            path, event_fraction, event_x, event_y, event_z, event_r);
        event_z = path.WrapZ(event_z);
        amrex::Real const event_elapsed = amrex::min(
            amrex::Real(1.0),
            elapsed_path_start
                + event_distance
                    / amrex::max(
                        c_m_per_s * dt_s,
                        std::numeric_limits<amrex::Real>::min()));
        RreaInteractionSite const site{
            event_x, event_y, event_z, event_r, event_elapsed};
        auto rng_key = [&](RreaRngChannel rng_channel) {
            return RreaRngKey{
                coupling.RngSeed(), p.root_rng_particle_id,
                static_cast<std::uint64_t>(std::max(step, 0)),
                packed.value,
                static_cast<std::uint64_t>(rng_channel)};
        };
        auto deposit_local = [&](amrex::Real local_energy_eV) {
            deposit_local_energy_as_ion_pairs(
                newborn_fx, event_r, event_z, p.weight, local_energy_eV);
        };
        RreaPhotonChannel const channel = tables.SamplePhotonChannel(
            photon_energy_eV,
            RreaRng::Uniform01(rng_key(
                RreaRngChannel::NewbornCompetingChannel)));
        if (channel == RreaPhotonChannel::Compton) {
            if (!RreaApplyComptonFinalState(
                    newborn_fx, tables, rng_key, deposit_local, site,
                    p.weight, coupling.LowEnergyCutoffEv(),
                    coupling.PhotonCutoffEv(),
                    photon_energy_eV, dir_x, dir_y, dir_z)) {
                return false;
            }
            if (!(remaining_m > amrex::Real(0.0))) {
                p.x_m = event_x; p.y_m = event_y; p.z_m = event_z;
                p.dir_x = dir_x; p.dir_y = dir_y; p.dir_z = dir_z;
                p.kinetic_or_photon_energy_eV = photon_energy_eV;
                p.birth_elapsed_step_fraction = amrex::Real(1.0);
                return true;
            }
            amrex::Real const target_x = event_x + remaining_m * dir_x;
            amrex::Real const target_y = event_y + remaining_m * dir_y;
            amrex::Real const target_z = event_z + remaining_m * dir_z;
            path = ClipRzTransportPath(
                event_r, std::atan2(event_y, event_x), event_z,
                std::hypot(target_x, target_y),
                std::atan2(target_y, target_x), target_z,
                geom.ProbLo(0), geom.ProbHi(0),
                geom.ProbLo(1), geom.ProbHi(1), periodic_z);
            elapsed_path_start = event_elapsed;
            ++event_ordinal;
            continue;
        }
        if (channel == RreaPhotonChannel::Photoelectric) {
            RreaApplyPhotoelectricFinalState(
                newborn_fx, tables, rng_key, deposit_local, site,
                p.weight, coupling.LowEnergyCutoffEv(),
                photon_energy_eV, dir_x, dir_y, dir_z);
            return false;
        }
        RreaApplyPairFinalState(
            newborn_fx, tables, rng_key, deposit_local,
            [&](amrex::Real positron_dir_x,
                amrex::Real positron_dir_y,
                amrex::Real positron_dir_z,
                amrex::Real positron_kinetic_eV) {
                spawn_positron_annihilation(
                    newborn_fx,
                    event_x, event_y, event_z, event_r, event_z,
                    positron_dir_x, positron_dir_y, positron_dir_z,
                    p.weight, positron_kinetic_eV, true,
                    p.root_rng_particle_id, packed.value,
                    event_elapsed);
            },
            site, p.weight, coupling.LowEnergyCutoffEv(),
            photon_energy_eV, channel, dir_x, dir_y, dir_z);
        return false;
    }
}

}  // namespace rrea::warpx
