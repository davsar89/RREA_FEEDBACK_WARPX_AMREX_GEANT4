#include "RreaTransportStep.H"

#ifdef AMREX_USE_OMP
#include <omp.h>
#endif

namespace rrea::warpx {

void TransportStep::ApplyPositronTransport()
{
#if defined(WARPX_DIM_RZ)
            auto& positrons =
                warpx.GetPartContainer().GetParticleContainerFromName(coupling.PositronSpeciesName());
            PreviousRzComponentIndices const positron_previous =
                require_previous_rz_components(positrons, coupling.PositronSpeciesName());
            // Use the same deterministic per-chunk replay as the electron and
            // photon paths when scoped transport threading is enabled.
            for (WarpXParIter pti(positrons, 0); pti.isValid(); ++pti) {
                auto& tile = pti.GetParticleTile();
                auto& attribs = pti.GetAttribs();
                auto* const r_data = attribs[PIdx::r].dataPtr();
                auto* const z_data = attribs[PIdx::z].dataPtr();
                auto* const w_data = attribs[PIdx::w].dataPtr();
                auto* const ux_data = attribs[PIdx::ux].dataPtr();
                auto* const uy_data = attribs[PIdx::uy].dataPtr();
                auto* const uz_data = attribs[PIdx::uz].dataPtr();
                auto* const theta_data = attribs[PIdx::theta].dataPtr();
                auto* const previous_r_data = pti.GetAttribs(positron_previous.r).dataPtr();
                auto* const previous_z_data = pti.GetAttribs(positron_previous.z).dataPtr();
                auto* const previous_theta_data = pti.GetAttribs(positron_previous.theta).dataPtr();
                auto* const previous_ux_data = pti.GetAttribs(positron_previous.ux).dataPtr();
                auto* const previous_uy_data = pti.GetAttribs(positron_previous.uy).dataPtr();
                auto* const previous_uz_data = pti.GetAttribs(positron_previous.uz).dataPtr();
            ChargedSoAView const soa{
                r_data, z_data, w_data, ux_data, uy_data, uz_data,
                theta_data,
                previous_r_data, previous_z_data, previous_theta_data,
                previous_ux_data, previous_uy_data, previous_uz_data};
                long const np = pti.numParticles();
                constexpr long transport_chunk_size = 64;
                long const n_chunks =
                    (np + transport_chunk_size - 1) / transport_chunk_size;
                std::vector<TransportSideEffects> chunk_fx(
                    static_cast<std::size_t>(std::max<long>(n_chunks, 0)),
                    TransportSideEffects{coupling.LowEnergyCutoffEv()});
#ifdef AMREX_USE_OMP
                int const transport_num_threads =
                    (coupling.TransportOmpThreads() > 0)
                        ? coupling.TransportOmpThreads()
                        : omp_get_max_threads();
#pragma omp parallel for schedule(dynamic, 1) num_threads(transport_num_threads) if(transport_num_threads > 1 && n_chunks > 1)
#endif
                for (long chunk = 0; chunk < n_chunks; ++chunk) {
                TransportSideEffects& fx = chunk_fx[static_cast<std::size_t>(chunk)];
                long const ip_begin = chunk * transport_chunk_size;
                long const ip_end = std::min(np, ip_begin + transport_chunk_size);
                for (long ip = ip_begin; ip < ip_end; ++ip) {
                auto particle_id = tile.id(static_cast<int>(ip));
                if (!particle_id.is_valid()) {
                    continue;
                }
                    auto setup = BeginRunningChargedParticle(
                        fx, soa, ip,
                        static_cast<amrex::Long>(particle_id),
                        tile.cpu(static_cast<int>(ip)),
                        RreaSecondarySpecies::Positron,
                        /*charge_sign=*/amrex::Real(1.0),
                        "positron");
                    if (setup.skip) {
                        particle_id.make_invalid();
                        fx.invalidated = true;
                        continue;
                    }
                    amrex::Real const particle_weight = setup.particle_weight;
                    amrex::Real dir_x = setup.dir_x;
                    amrex::Real dir_y = setup.dir_y;
                    amrex::Real dir_z = setup.dir_z;
                    amrex::Real kinetic_energy_eV = setup.kinetic_energy_eV;
                    amrex::Real run_x = setup.run_x;
                    amrex::Real run_y = setup.run_y;
                    amrex::Real run_z = setup.run_z;
                    amrex::Real const x_particle = setup.x_particle;
                    amrex::Real const y_particle = setup.y_particle;
                    amrex::Real const z_particle = setup.z_particle;
                    amrex::Real const r_particle = setup.r_particle;
                    RreaTransportPathSegment const driver_path =
                        setup.driver_path;
                    RreaEndpointEscapePolicy charged_parent_escape =
                        setup.charged_parent_escape;
                    amrex::Real const ds_m = setup.ds_m;
                    auto const& driver_intervals = setup.driver_intervals;
                    amrex::Real const driver_span = setup.driver_span;
                    // Positrons have no directional demotion
                    // gate; the whole-chord work integral only sizes the
                    // field-impulse substep driver.
                    amrex::Real const clipped_field_work_eV =
                        setup.field_work_eV;
                    // Rebuilt with each running-state chord; intervals from another
                    // chord have an empty material-rate intersection.
                    RreaTransportPathSegment transport_path = driver_path;
                    std::vector<MaterialPathInterval> material_intervals =
                        driver_intervals;
                    auto commit_charged_escape = [&]() {
                        CommitChargedEscape(
                            fx, soa, ip,
                            particle_weight, kinetic_energy_eV,
                            dir_x, dir_y, dir_z,
                            r_particle, z_particle,
                            transport_path, charged_parent_escape,
                            /*signed_mesh_weight=*/particle_weight);
                        if (charged_parent_escape.invalidate_parent_in_rrea) {
                            particle_id.make_invalid();
                            fx.invalidated = true;
                        }
                    };
                    std::uint64_t const rng_particle_id =
                        setup.transaction_parent_key;
                    auto const initial_terminal_disposition =
                        RreaResolveChargedTerminalDisposition(
                            /*positron=*/true,
                            charged_parent_escape.escapes,
                            kinetic_energy_eV < coupling.LowEnergyCutoffEv(),
                            /*electron_directional_demotion=*/false);
                    if (initial_terminal_disposition
                        == RreaChargedTerminalDisposition::Escape
                        && kinetic_energy_eV < coupling.LowEnergyCutoffEv()) {
                        // Boundary ownership precedes endpoint thermal
                        // annihilation.  In particular, never place the two
                        // annihilation photons or residual-energy source on an
                        // excluded upper face.
                        commit_charged_escape();
                        continue;
                    }
                    if (initial_terminal_disposition
                        == RreaChargedTerminalDisposition::PositronAnnihilation) {
                        fx.BeginGroup(RreaSecondaryProcess::PositronAnnihilation, true);
                        spawn_positron_annihilation(
                            fx,
                            x_particle,
                            y_particle,
                            z_particle,
                            r_particle,
                            z_particle,
                            dir_x,
                            dir_y,
                            dir_z,
                            particle_weight,
                            kinetic_energy_eV,
                            true,
                            rng_particle_id,
                            0);
                        fx.tally.RecordPositronTransport(particle_weight, kinetic_energy_eV, true);
                        fx.SetInteractionConservationClosure(
                            particle_weight,
                            kinetic_energy_eV,
                            amrex::Real(0.0));
                        // Zero-length net chord: annihilated at the captured
                        // start before any transport.
                        fx.AccumulateChargedSegment(
                            0, r_particle, z_particle, r_particle, z_particle,
                            particle_weight, kRreaSegmentChannelAnnihilated);
                        fx.EndGroup();
                        particle_id.make_invalid();
                        fx.invalidated = true;
                        continue;
                    }
                    if (ds_m <= amrex::Real(0.0)) {
                        if (charged_parent_escape.escapes) {
                            commit_charged_escape();
                        }
                        continue;
                    }
                    if (driver_intervals.empty()) {
                        amrex::Abort(
                            "RREA positron path has positive length but no material segments");
                    }
                    // Annihilation is open at any energy; Bhabha and tracked
                    // brems carry their own phase-space gates, exactly as for
                    // the electron loop (RreaChargedCompetingStpRates).
                    auto const initial_positron_stp =
                        RreaChargedCompetingStpRates(
                            tables, /*is_positron=*/true,
                            coupling.HardMollerSecondaryThresholdEv(),
                            coupling.PhotonCutoffEv(),
                            kinetic_energy_eV);
                    amrex::Real const initial_total_positron_stp =
                        initial_positron_stp[0] + initial_positron_stp[1]
                        + initial_positron_stp[2];
                    // Every budget below is a DRIVER estimate: integrated over the
                    // driver chord's clipped in-domain span, then divided by that
                    // span because the loop transports the whole dt.
                    amrex::Real const driver_density_path_m = density_path_m(
                        driver_intervals,
                        driver_path,
                        driver_path.in_domain_begin_fraction,
                        driver_path.in_domain_end_fraction) / driver_span;
                    amrex::Real const positron_tau =
                        initial_total_positron_stp * driver_density_path_m;
                    bool optical_depth_cap_violation = false;
                    std::uint64_t subcycles = selected_material_subcycles(
                        positron_tau,
                        driver_intervals,
                        driver_path,
                        m_config,
                        optical_depth_cap_violation,
                        initial_total_positron_stp);
                    {
                        // GS batching and field-impulse/drag
                        // substep drivers (mirror of the electron loop; benign
                        // silent clamp at the cap).
                        amrex::Real const initial_gs_n_total =
                            elastic_inverse_length_stp(
                                tables, true, kinetic_energy_eV)
                            * driver_density_path_m;
                        amrex::Real const drag_estimate_eV =
                            tables.PositronCollisionLoss(
                                amrex::max(
                                    kinetic_energy_eV, coupling.LowEnergyCutoffEv()),
                                amrex::Real(1.0))
                            * driver_density_path_m;
                        amrex::Real const impulse_scale_eV = amrex::max(
                            m_config.transport_max_field_impulse_fraction
                                * kinetic_energy_eV,
                            amrex::max(
                                m_config.transport_field_impulse_floor_eV,
                                amrex::Real(1.0)));
                        std::uint64_t const gs_driver = substep_driver_count(
                            amrex::min(
                                initial_gs_n_total,
                                m_config.transport_gs_total_collision_cap),
                            amrex::max(
                                m_config.transport_max_gs_collisions_per_substep,
                                amrex::Real(1.0)));
                        std::uint64_t const impulse_driver = substep_driver_count(
                            std::abs(clipped_field_work_eV) + drag_estimate_eV,
                            impulse_scale_eV);
                        subcycles = std::min<std::uint64_t>(
                            std::max({subcycles, gs_driver, impulse_driver}),
                            m_config.max_subcycles);
                    }
                    // A driver estimate too: uniform slices of a chord the loop no
                    // longer transports verbatim.  Reported, not enforced, and
                    // divided by the clipped span like every enforced budget so
                    // it describes the full-dt microchords actually flown.
                    amrex::Real const max_positron_tau =
                        initial_total_positron_stp
                        * max_uniform_subcycle_density_path_m(
                            driver_intervals, driver_path, subcycles) / driver_span;
                    fx.tally.RecordTransportGuardMetrics(
                        amrex::Real(0.0),
                        amrex::Real(0.0),
                        amrex::Real(0.0),
                        amrex::Real(0.0),
                        max_positron_tau,
                        subcycles,
                        amrex::Real(0.0),
                        amrex::Real(0.0),
                        kinetic_energy_eV,
                        optical_depth_cap_violation);
                    abort_if_strict_guard_failed(m_config, optical_depth_cap_violation);
                    amrex::Real total_loss_eV = amrex::Real(0.0);
                    bool annihilated = false;
                    // Each rebuilt microchord is one complete subcycle, so the
                    // transport window is its whole in-domain span.
                    auto position_at_fraction = [&transport_path](
                            amrex::Real path_fraction,
                            amrex::Real& event_x,
                            amrex::Real& event_y,
                            amrex::Real& event_z,
                            amrex::Real& event_r) {
                        point_and_radius_at_fraction(
                            transport_path, path_fraction,
                            event_x, event_y, event_z, event_r);
                    };

                    // Apply and book positron collision/soft-radiative stopping
                    // only on the supplied geometric interval.  Selection rates
                    // remain frozen at the enclosing subcycle's start energy;
                    // after a surviving event this helper is called again with
                    // the post-event energy for the unused interval.
                    // Same drag module as the electron loop
                    // (RreaPlanContinuousDragSegment); the positron POLICY:
                    // collision + soft radiative stopping, hard transfers
                    // stay per-event (Bhabha) so the mean-rate partition is
                    // zero, and the running total feeds the
                    // annihilation-at-rest budget.
                    auto apply_positron_continuous_segment = [&] (
                        amrex::Real begin_fraction,
                        amrex::Real end_fraction,
                        amrex::Real stopping_energy_eV)
                        -> RreaContinuousSegmentLimit {
                        auto density_path = [&](amrex::Real b, amrex::Real e) {
                            return density_path_m(
                                material_intervals, transport_path, b, e);
                        };
                        auto const plan = RreaPlanContinuousDragSegment(
                            begin_fraction, end_fraction, kinetic_energy_eV,
                            stopping_energy_eV, coupling.LowEnergyCutoffEv(),
                            density_path,
                            [&] {
                                return RreaFrozenStoppingPowers{
                                    RreaFrozenStpRate(tables.PositronCollisionLoss(
                                        stopping_energy_eV, amrex::Real(1.0))),
                                    RreaFrozenStpRate(tables.PositronSoftRadiativeDrag(
                                        stopping_energy_eV, amrex::Real(1.0))),
                                    amrex::Real(0.0)};
                            });
                        if (!plan.limit_valid) {
                            amrex::Abort(
                                "RREA positron continuous stopping could not "
                                "locate a finite monotone cutoff point");
                        }
                        if (!plan.applied) {
                            return RreaContinuousSegmentLimit{
                                begin_fraction, false, true};
                        }
                        fx.tally.RecordContinuousSoftBrems(
                            particle_weight, plan.soft_radiative_loss_eV);
                        total_loss_eV += plan.drag_loss_eV;
                        kinetic_energy_eV = amrex::max(
                            amrex::Real(0.0),
                            kinetic_energy_eV - plan.drag_loss_eV);
                        if (plan.reaches_cutoff) {
                            kinetic_energy_eV = coupling.LowEnergyCutoffEv();
                        }

                        {
                            amrex::Real const diagnostic_loss_eV =
                                tables.WAirEvPerIonPair();
                            RreaPartitionContinuousIonPairs(
                                material_intervals,
                                transport_path.full_length_m,
                                begin_fraction,
                                plan.effective_end_fraction,
                                plan.collision_loss_eV,
                                plan.raw_collision_loss_eV,
                                particle_weight,
                                diagnostic_loss_eV,
                                plan.collision_stp_eV_per_m,
                                [&](amrex::Real midpoint_fraction,
                                    amrex::Real pair_weight) {
                                    amrex::Real source_x = amrex::Real(0.0);
                                    amrex::Real source_y = amrex::Real(0.0);
                                    amrex::Real source_z = amrex::Real(0.0);
                                    amrex::Real source_r = amrex::Real(0.0);
                                    position_at_fraction(
                                        midpoint_fraction,
                                        source_x, source_y, source_z, source_r);
                                    fx.AccumulateIonizationEvent(
                                        0,
                                        source_r,
                                        source_z,
                                        pair_weight,
                                        amrex::Real(0.0),
                                        diagnostic_loss_eV);
                                    fx.tally.RecordTransportIonization(
                                        pair_weight,
                                        diagnostic_loss_eV,
                                        true,
                                        false);
                                });
                        }
                        return RreaContinuousSegmentLimit{
                            plan.effective_end_fraction, plan.reaches_cutoff,
                            true};
                    };
                    // The positron substep engine mirrors the electron loop; a
                    // positron stopping inside the chord annihilates at rest at
                    // the substep position.
                    bool exited_domain = false;
                    std::uint64_t completed_substeps = 0;
                    // Convert microchord-local positions to outer-step elapsed
                    // fractions before recording secondary birth times.
                    auto elapsed_fraction_at = [&](amrex::Real path_fraction) {
                        amrex::Real const within = amrex::min(
                            amrex::Real(1.0),
                            amrex::max(amrex::Real(0.0), path_fraction));
                        return amrex::min(
                            amrex::Real(1.0),
                            (static_cast<amrex::Real>(completed_substeps) + within)
                                / static_cast<amrex::Real>(subcycles));
                    };
                    bool stopped_before_trailing_half = false;
                    amrex::Real stopping_x_m = x_particle;
                    amrex::Real stopping_y_m = y_particle;
                    amrex::Real stopping_z_m = z_particle;
                    amrex::Real stopping_r_m = r_particle;
                    // Stop GEOMETRY and stop TIME are different quantities: the
                    // path fraction locates the stop on the current microchord, the
                    // elapsed fraction says when in the outer step it happened, and
                    // it is the latter the annihilation photons' residual advance
                    // needs.  Evaluated at mark time, before completed_substeps
                    // moves past the microchord the stop happened on.
                    amrex::Real stopping_elapsed_fraction = amrex::Real(1.0);
                    auto mark_positron_stopped_at = [&](amrex::Real path_fraction) {
                        if (RreaChargedStopIsEscape(
                                transport_path.exits_domain,
                                path_fraction,
                                transport_path.in_domain_end_fraction)) {
                            exited_domain = true;
                            return;
                        }
                        amrex::Real stop_x = amrex::Real(0.0);
                        amrex::Real stop_y = amrex::Real(0.0);
                        amrex::Real stop_z = amrex::Real(0.0);
                        amrex::Real stop_r = amrex::Real(0.0);
                        position_at_fraction(
                            path_fraction, stop_x, stop_y, stop_z, stop_r);
                        if (!RreaCartesianPositionInHalfOpenRzDomain(
                                stop_x,
                                stop_y,
                                stop_z,
                                geom.ProbLo(0),
                                geom.ProbHi(0),
                                geom.ProbLo(1),
                                geom.ProbHi(1))) {
                            exited_domain = true;
                            return;
                        }
                        stopping_x_m = stop_x;
                        stopping_y_m = stop_y;
                        stopping_z_m = stop_z;
                        stopping_r_m = stop_r;
                        stopping_elapsed_fraction =
                            elapsed_fraction_at(path_fraction);
                        stopped_before_trailing_half = true;
                    };
                    auto apply_field_half_kick = [&](
                        PathFieldSubRange const& field_range,
                        amrex::Real half_dt_s) {
                        static_cast<void>(RreaApplyChargedFieldHalfKick(
                            field_range, half_dt_s, amrex::Real(1.0),
                            kinetic_energy_eV, dir_x, dir_y, dir_z));
                    };
                    auto apply_positron_gs_scatter = [&](
                        amrex::Real begin_fraction,
                        amrex::Real end_fraction,
                        std::uint64_t substep_idx) {
                        static constexpr GsScatterChannels kPositronGsChannels{
                            RreaRngChannel::PositronGsFewCount,
                            RreaRngChannel::PositronGsFewScatter,
                            RreaRngChannel::PositronGsFewAzimuth,
                            RreaRngChannel::PositronGsAngle,
                            RreaRngChannel::PositronGsAzimuth};
                        auto gs_key = [&](RreaRngChannel channel,
                                          std::uint64_t inner) {
                            return RreaRngKey{
                                coupling.RngSeed(), rng_particle_id,
                                static_cast<std::uint64_t>(std::max(step, 0)),
                                (substep_idx << 8) | inner,
                                static_cast<std::uint64_t>(channel)};
                        };
                        apply_gs_elastic_scatter(
                            material_intervals, transport_path,
                            begin_fraction, end_fraction, tables,
                            /*is_positron=*/true, kPositronGsChannels,
                            gs_key, kinetic_energy_eV, dir_x, dir_y, dir_z);
                    };
                    auto finish_interleaved_substep = [&](
                        std::uint64_t substep_idx,
                        amrex::Real begin_fraction,
                        amrex::Real end_fraction,
                        PathFieldSubRange const& substep_field,
                        amrex::Real substep_dt_s) -> bool {
                        if (kinetic_energy_eV >= coupling.LowEnergyCutoffEv()) {
                            apply_positron_gs_scatter(
                                begin_fraction, end_fraction, substep_idx);
                        }
                        apply_field_half_kick(
                            substep_field, amrex::Real(0.5) * substep_dt_s);
                        if (kinetic_energy_eV < coupling.LowEnergyCutoffEv()) {
                            // Stopped on this microchord.  Only RECORD where and
                            // when; the single terminal booking after the loop
                            // owns the annihilation.  Booking it here would open a
                            // second remove_parent group for the same parent (a
                            // collective abort once every rank has finished
                            // transport) and would let a stop override an escape.
                            mark_positron_stopped_at(end_fraction);
                            return true;
                        }
                        return false;
                    };
                    amrex::Real const sub_dt_s =
                        dt_s / static_cast<amrex::Real>(subcycles);
                    PathFieldSubRange substep_field{};
                    amrex::Real substep_dt_s = amrex::Real(0.0);
                    RreaRunChargedSubstepInterleave(
                        subcycles,
                        kinetic_energy_eV,
                        coupling.LowEnergyCutoffEv(),
                        completed_substeps,
                        exited_domain,
                        [&](std::uint64_t,
                            amrex::Real& subcycle_begin,
                            amrex::Real& subcycle_end) {
                            // Rebuild the chord the positron will actually fly over this
                            // subcycle, from the state it is actually in, so elastic
                            // deflection and discrete recoil bend the path it then
                            // follows and the field work projects onto the direction it
                            // is really travelling.
                            auto const chord = RreaBuildRunningChargedChord(
                                RreaRunningChargedState{
                                    run_x, run_y, run_z, dir_x, dir_y, dir_z,
                                    kinetic_energy_eV},
                                sub_dt_s,
                                dt_s,
                                geom.ProbLo(0),
                                geom.ProbHi(0),
                                geom.ProbLo(1),
                                geom.ProbHi(1),
                                periodic_z);
                            if (!chord.valid) {
                                return false;
                            }
                            transport_path = chord.path;
                            if (!(transport_path.full_length_m > amrex::Real(0.0))
                                || !(transport_path.in_domain_length_m
                                     > amrex::Real(0.0))) {
                                exited_domain = transport_path.exits_domain;
                                return false;
                            }
                            // Material intervals must match this microchord.
                            material_intervals = build_material_path_intervals(
                                coupling, transport_path, density_ratio);
                            if (material_intervals.empty()) {
                                amrex::Abort(
                                    "RREA positron microchord has positive in-domain "
                                    "length but no material intervals");
                            }
                            subcycle_begin = transport_path.in_domain_begin_fraction;
                            subcycle_end = transport_path.in_domain_end_fraction;
                            substep_field = path_field_work_and_average(
                                coupling,
                                transport_path,
                                amrex::Real(1.0),
                                subcycle_begin,
                                subcycle_end);
                            // Time ACTUALLY spent on this chord.  The work integral is
                            // already restricted to [begin, end], but the half-kick's
                            // ROTATION scales with dt, so handing it the whole subcycle
                            // over-rotates a clipped microchord by 1/(end-begin):
                            // energy exactly right, direction saturated to E, and no
                            // conservation ledger can see it.
                            substep_dt_s = sub_dt_s * (subcycle_end - subcycle_begin);
                            apply_field_half_kick(
                                substep_field, amrex::Real(0.5) * substep_dt_s);
                            if (kinetic_energy_eV < coupling.LowEnergyCutoffEv()) {
                                mark_positron_stopped_at(subcycle_begin);
                                return false;
                            }
                            return true;
                        },
                        [&](std::uint64_t subcycle,
                            amrex::Real subcycle_begin,
                            amrex::Real subcycle_end) {
                        // Event-loop policies: the positron's physics content; the shared
                        // substep_interleaved_v3 sequencing lives in
                        // RreaRunChargedEventInterleave.
                        amrex::Real event_x = amrex::Real(0.0);
                        amrex::Real event_y = amrex::Real(0.0);
                        amrex::Real event_z = amrex::Real(0.0);
                        amrex::Real event_r = amrex::Real(0.0);
                        std::uint64_t interaction_rng_index = 0;
                        amrex::Real const bhabha_secondary_threshold_eV =
                            coupling.HardMollerSecondaryThresholdEv();
                        // The hard-Bhabha table represents above-threshold target electron
                        // production and is authoritative immediately above its exact
                        // zero-rate threshold.  The outgoing positron need not remain above
                        // the transport cutoff; its later at-rest annihilation handles that
                        // physical branch.  The Bhabha phase-space gate scopes only that
                        // component of the summed hazard; brems and annihilation remain live
                        // in the low band.
                        // Frozen STP inverse lengths, refreshed by the
                        // channel_taus policy and read by the depth inversion.
                        amrex::Real total_stp = amrex::Real(0.0);
                        RreaRunChargedEventInterleave(
                            subcycle_begin,
                            subcycle_end,
                            kinetic_energy_eV,
                            coupling.LowEnergyCutoffEv(),
                            3,
                            kRreaTransientEventCapacity,
                            "positron",
                            [&](amrex::Real frozen_energy_eV,
                                amrex::Real tau_begin_fraction,
                                amrex::Real tau_end_fraction) {
                                auto const stp = RreaChargedCompetingStpRates(
                                    tables, /*is_positron=*/true,
                                    bhabha_secondary_threshold_eV,
                                    coupling.PhotonCutoffEv(),
                                    frozen_energy_eV);
                                total_stp = stp[0] + stp[1] + stp[2];
                                amrex::Real const segment_density_path_m = density_path_m(
                                    material_intervals, transport_path,
                                    tau_begin_fraction, tau_end_fraction);
                                return std::array<amrex::Real, 3>{
                                    stp[0] * segment_density_path_m,
                                    stp[1] * segment_density_path_m,
                                    stp[2] * segment_density_path_m};
                            },
                            [&](std::uint64_t draw_event_ordinal) {
                                auto const live_event_index = RreaTryLiveHazardRngIndex(
                                    RreaSecondarySpecies::Positron,
                                    subcycle, draw_event_ordinal);
                                if (!live_event_index.valid) {
                                    amrex::Abort(
                                        "RREA positron repeated-hazard RNG index overflow; "
                                        "the uncommitted transaction is rejected");
                                }
                                interaction_rng_index = live_event_index.value;
                                RreaRngChannel const optical_depth_channel =
                                    draw_event_ordinal == 0U
                                    ? RreaRngChannel::PositronCompetingOpticalDepth
                                    : RreaRngChannel::PositronContinuationOpticalDepth;
                                RreaRngChannel const competing_channel =
                                    draw_event_ordinal == 0U
                                    ? RreaRngChannel::PositronCompetingChannel
                                    : RreaRngChannel::PositronContinuationChannel;
                                return std::pair<amrex::Real, amrex::Real>{
                                    RreaRng::Uniform01(RreaRngKey{
                                                coupling.RngSeed(),
                                                rng_particle_id,
                                                static_cast<std::uint64_t>(std::max(step, 0)),
                                                interaction_rng_index,
                                                static_cast<std::uint64_t>(optical_depth_channel)}),
                                    RreaRng::Uniform01(RreaRngKey{
                                                coupling.RngSeed(),
                                                rng_particle_id,
                                                static_cast<std::uint64_t>(std::max(step, 0)),
                                                interaction_rng_index,
                                                static_cast<std::uint64_t>(competing_channel)})};
                            },
                            [&](amrex::Real sampled_optical_depth,
                                amrex::Real invert_begin_fraction,
                                amrex::Real invert_end_fraction) {
                                return fraction_at_density_path_m(
                                    material_intervals,
                                    transport_path,
                                    invert_begin_fraction,
                                    invert_end_fraction,
                                    sampled_optical_depth / total_stp);
                            },
                            [&](amrex::Real segment_begin_fraction,
                                amrex::Real segment_end_fraction,
                                amrex::Real frozen_energy_eV) {
                                return apply_positron_continuous_segment(
                                    segment_begin_fraction,
                                    segment_end_fraction,
                                    frozen_energy_eV);
                            },
                            [&](amrex::Real path_fraction) {
                                mark_positron_stopped_at(path_fraction);
                            },
                            [&](amrex::Real point_event_fraction) {
                                position_at_fraction(
                                    point_event_fraction,
                                    event_x,
                                    event_y,
                                    event_z,
                                    event_r);
                            },
                            [&](int channel, amrex::Real channel_energy_eV) {
                                return RreaChargedChannelOpenAt(
                                    tables, /*is_positron=*/true,
                                    bhabha_secondary_threshold_eV,
                                    coupling.PhotonCutoffEv(),
                                    channel, channel_energy_eV);
                            },
                            [&](int channel, amrex::Real event_fraction) {
                                RreaInteractionSite const site{
                                    event_x, event_y, event_z, event_r,
                                    elapsed_fraction_at(event_fraction)};
                                auto rng_key = [&](RreaRngChannel rng_channel) {
                                    return RreaRngKey{
                                        coupling.RngSeed(),
                                        rng_particle_id,
                                        static_cast<std::uint64_t>(
                                            std::max(step, 0)),
                                        interaction_rng_index,
                                        static_cast<std::uint64_t>(rng_channel)};
                                };
                                if (channel == 0) {
                                    // A sampled secondary below the production
                                    // threshold discards only this Bhabha
                                    // event; the other channels stay eligible.
                                    // The outgoing positron need not stay above
                                    // the transport cutoff -- its later at-rest
                                    // annihilation owns that branch.
                                    RreaApplyBhabhaFinalState(
                                        fx, tables, rng_key, site,
                                        particle_weight,
                                        bhabha_secondary_threshold_eV,
                                        kinetic_energy_eV,
                                        dir_x, dir_y, dir_z);
                                } else if (channel == 1) {
                                    RreaApplyBremsFinalState(
                                        fx, tables, rng_key,
                                        [&](amrex::Real local_energy_eV) {
                                            deposit_local_energy_as_ion_pairs(
                                                fx, event_r, event_z,
                                                particle_weight,
                                                local_energy_eV);
                                        },
                                        site, particle_weight,
                                        /*charge_sign=*/amrex::Real(1.0),
                                        coupling.PhotonCutoffEv(),
                                        kinetic_energy_eV,
                                        dir_x, dir_y, dir_z);
                                } else if (channel == 2) {
                                    fx.BeginGroup(
                                        RreaSecondaryProcess::PositronAnnihilation,
                                        true);
                                    spawn_positron_annihilation(
                                        fx,
                                        event_x, event_y, event_z,
                                        event_r, event_z,
                                        dir_x, dir_y, dir_z,
                                        particle_weight,
                                        kinetic_energy_eV,
                                        false,
                                        rng_particle_id,
                                        interaction_rng_index,
                                        site.elapsed_step_fraction);
                                    fx.SetInteractionConservationClosure(
                                        particle_weight,
                                        kinetic_energy_eV,
                                        amrex::Real(0.0));
                                    // Net chord: captured start to the sampled
                                    // in-flight annihilation point.
                                    fx.AccumulateChargedSegment(
                                        0, r_particle, z_particle,
                                        event_r, event_z,
                                        particle_weight,
                                        kRreaSegmentChannelAnnihilated);
                                    fx.EndGroup();
                                    annihilated = true;
                                    return false;
                                }
                                return true;
                            });
                        },
                        [&] { return stopped_before_trailing_half || annihilated; },
                        [&](std::uint64_t subcycle,
                            amrex::Real subcycle_begin,
                            amrex::Real subcycle_end) {
                            return finish_interleaved_substep(
                                subcycle, subcycle_begin, subcycle_end,
                                substep_field, substep_dt_s);
                        },
                        // Commit this microchord: the next one is rebuilt from here, along
                        // whatever direction the scattering above left behind.  A refused
                        // endpoint is one the next chord would have been built from while
                        // lying outside the half-open domain, so it becomes an escape.
                        [&](amrex::Real subcycle_end) {
                            return RreaAdvanceRunningChargedPosition(
                                transport_path,
                                subcycle_end,
                                geom.ProbLo(0),
                                geom.ProbHi(0),
                                geom.ProbLo(1),
                                geom.ProbHi(1),
                                run_x,
                                run_y,
                                run_z);
                        },
                        [&] { return transport_path.exits_domain; });
                    if (exited_domain) {
                        charged_parent_escape =
                            RreaClippedTransportEscapePolicyForPath(
                                transport_path,
                                std::hypot(
                                    transport_path.end_x_m,
                                    transport_path.end_y_m),
                                transport_path.end_z_m);
                    }
                    fx.tally.RecordPositronTransport(particle_weight, total_loss_eV, false);
                    if (annihilated) {
                        particle_id.make_invalid();
                        fx.invalidated = true;
                        continue;
                    }
                    // A microchord leaving the domain takes precedence over an
                    // in-domain stop and annihilation.
                    auto const terminal_disposition =
                        RreaResolveChargedTransportTerminal(
                            exited_domain,
                            stopped_before_trailing_half,
                            RreaChargedTerminalDisposition::PositronAnnihilation,
                            /*positron=*/true,
                            kinetic_energy_eV < coupling.LowEnergyCutoffEv(),
                            /*electron_directional_demotion=*/false);
                    if (terminal_disposition
                        == RreaChargedTerminalDisposition::Escape) {
                        // A sampled in-flight annihilation above has already
                        // won if it occurred inside the chord.  Otherwise the
                        // residual positron reaches the absorbing face before
                        // any endpoint-only at-rest annihilation can run.  The
                        // exact-face/overshoot ownership split is applied by
                        // commit_charged_escape below.
                        commit_charged_escape();
                        continue;
                    }
                    if (terminal_disposition
                        == RreaChargedTerminalDisposition::PositronAnnihilation) {
                        // The non-stopped branch annihilates where the positron
                        // actually ended up, which is the running point -- NOT the
                        // pre-push point and not WarpX's straight-line endpoint.
                        amrex::Real const annihilation_x_m =
                            stopped_before_trailing_half
                            ? stopping_x_m : run_x;
                        amrex::Real const annihilation_y_m =
                            stopped_before_trailing_half
                            ? stopping_y_m : run_y;
                        amrex::Real const annihilation_z_m =
                            stopped_before_trailing_half
                            ? stopping_z_m : run_z;
                        amrex::Real const annihilation_r_m =
                            stopped_before_trailing_half
                            ? stopping_r_m : std::hypot(run_x, run_y);
                        fx.BeginGroup(RreaSecondaryProcess::PositronAnnihilation, true);
                        spawn_positron_annihilation(
                            fx,
                            annihilation_x_m,
                            annihilation_y_m,
                            annihilation_z_m,
                            annihilation_r_m,
                            annihilation_z_m,
                            dir_x,
                            dir_y,
                            dir_z,
                            particle_weight,
                            kinetic_energy_eV,
                            true,
                            rng_particle_id,
                            subcycles,
                            stopped_before_trailing_half
                                ? stopping_elapsed_fraction
                                : amrex::Real(1.0));
                        fx.tally.RecordPositronTransport(particle_weight, kinetic_energy_eV, true);
                        fx.SetInteractionConservationClosure(
                            particle_weight,
                            kinetic_energy_eV,
                            amrex::Real(0.0));
                        // Net chord: captured start to the annihilation point
                        // (mid-chord stop point or the running endpoint).
                        fx.AccumulateChargedSegment(
                            0, r_particle, z_particle,
                            annihilation_r_m, annihilation_z_m,
                            particle_weight, kRreaSegmentChannelAnnihilated);
                        fx.EndGroup();
                        particle_id.make_invalid();
                        fx.invalidated = true;
                        continue;
                    }
                    CommitSurvivorChargedWriteback(
                        fx, soa, ip,
                        kinetic_energy_eV, dir_x, dir_y, dir_z,
                        run_x, run_y, run_z,
                        r_particle, z_particle,
                        std::hypot(run_x, run_y),
                        /*signed_mesh_weight=*/particle_weight);
                }
                }  // chunk loop

                for (auto& chunk_effects : chunk_fx) {
                    pending_side_effects.AppendGroupsFrom(chunk_effects);
                    invalidated_positrons =
                        invalidated_positrons || chunk_effects.invalidated;
                    repositioned_positrons =
                        repositioned_positrons || chunk_effects.repositioned;
                }
            }
#else
        amrex::Abort("RREA MC/PIC positron transport currently requires WarpX RZ");
#endif
}

}  // namespace rrea::warpx
