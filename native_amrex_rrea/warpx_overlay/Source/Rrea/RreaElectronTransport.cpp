#include "RreaTransportStep.H"

#ifdef AMREX_USE_OMP
#include <omp.h>
#endif

namespace rrea::warpx {

void TransportStep::ApplyElectronTransport()
{
        auto& electrons =
            warpx.GetPartContainer().GetParticleContainerFromName(coupling.ElectronSpeciesName());
#if defined(WARPX_DIM_RZ)
        PreviousRzComponentIndices const electron_previous =
            require_previous_rz_components(electrons, coupling.ElectronSpeciesName());
#endif
#if defined(WARPX_DIM_RZ)
        for (WarpXParIter pti(electrons, 0); pti.isValid(); ++pti) {
            auto& tile = pti.GetParticleTile();
            auto& attribs = pti.GetAttribs();
            auto* const r_data = attribs[PIdx::r].dataPtr();
            auto* const z_data = attribs[PIdx::z].dataPtr();
            auto* const w_data = attribs[PIdx::w].dataPtr();
            auto* const ux_data = attribs[PIdx::ux].dataPtr();
            auto* const uy_data = attribs[PIdx::uy].dataPtr();
            auto* const uz_data = attribs[PIdx::uz].dataPtr();
            auto* const theta_data = attribs[PIdx::theta].dataPtr();
            auto* const previous_r_data = pti.GetAttribs(electron_previous.r).dataPtr();
            auto* const previous_z_data = pti.GetAttribs(electron_previous.z).dataPtr();
            auto* const previous_theta_data = pti.GetAttribs(electron_previous.theta).dataPtr();
            auto* const previous_ux_data = pti.GetAttribs(electron_previous.ux).dataPtr();
            auto* const previous_uy_data = pti.GetAttribs(electron_previous.uy).dataPtr();
            auto* const previous_uz_data = pti.GetAttribs(electron_previous.uz).dataPtr();
            ChargedSoAView const soa{
                r_data, z_data, w_data, ux_data, uy_data, uz_data,
                theta_data,
                previous_r_data, previous_z_data, previous_theta_data,
                previous_ux_data, previous_uy_data, previous_uz_data};
            long const np = pti.numParticles();
            // Every execution mode logs per-chunk side effects.  They are
            // replayed deterministically only after the collective cap check.
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
                    RreaSecondarySpecies::Electron,
                    /*charge_sign=*/amrex::Real(-1.0),
                    "electron");
                if (setup.skip) {
                    particle_id.make_invalid();
                    fx.invalidated = true;
                    continue;
                }
                std::uint64_t const transaction_parent_key =
                    setup.transaction_parent_key;
                amrex::Real const particle_weight = setup.particle_weight;
                amrex::Real dir_x = setup.dir_x;
                amrex::Real dir_y = setup.dir_y;
                amrex::Real dir_z = setup.dir_z;
                amrex::Real kinetic_energy_eV = setup.kinetic_energy_eV;
                amrex::Real run_x = setup.run_x;
                amrex::Real run_y = setup.run_y;
                amrex::Real run_z = setup.run_z;
                amrex::Real const z_particle = setup.z_particle;
                amrex::Real const r_particle = setup.r_particle;
                RreaTransportPathSegment const driver_path = setup.driver_path;
                RreaEndpointEscapePolicy charged_parent_escape =
                    setup.charged_parent_escape;
                amrex::Real const ds_m = setup.ds_m;
                auto const& driver_intervals = setup.driver_intervals;
                amrex::Real const chord_density_path_m = density_path_m(
                    driver_intervals,
                    driver_path,
                    driver_path.in_domain_begin_fraction,
                    driver_path.in_domain_end_fraction);
                // Updated after the loop from the density actually traversed.
                // Every demoted electron's thermalized energy and ion
                // pairs are the avalanche's dominant sink; misplacing them
                // coherently displaces the conductivity field.
                amrex::Real particle_density_ratio =
                    ds_m > amrex::Real(0.0)
                    ? chord_density_path_m / ds_m
                    : positive_or(coupling.TransportDensityRatioAtZ(z_particle), density_ratio);
                amrex::Real transported_density_path_m = amrex::Real(0.0);
                amrex::Real transported_length_m = amrex::Real(0.0);
                amrex::Real const driver_span = setup.driver_span;
                amrex::Real const field_work_eV = setup.field_work_eV;
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
                        /*signed_mesh_weight=*/-particle_weight);
                    if (charged_parent_escape.invalidate_parent_in_rrea) {
                        particle_id.make_invalid();
                        fx.invalidated = true;
                    }
                };

                auto const initial_terminal_disposition =
                    RreaResolveChargedTerminalDisposition(
                        /*positron=*/false,
                        charged_parent_escape.escapes,
                        kinetic_energy_eV < coupling.LowEnergyCutoffEv(),
                        /*electron_directional_demotion=*/false);
                if (initial_terminal_disposition
                    == RreaChargedTerminalDisposition::Escape
                    && kinetic_energy_eV < coupling.LowEnergyCutoffEv()) {
                    // The clipped endpoint is on an excluded absorbing face.
                    // Do not create a fluid source there, and do not query a
                    // schema-6 charged table below its certified 1 keV floor.
                    commit_charged_escape();
                    continue;
                }
                if (initial_terminal_disposition
                    == RreaChargedTerminalDisposition::ElectronDemotion) {
                    // The demoted electron itself
                    // joins the low-energy mesh (energy 0 here so the mesh
                    // energy diagnostic is not double-counted), and its
                    // residual kinetic energy thermalizes into W_air ion
                    // pairs like every other local deposit.
                    fx.BeginGroup(RreaSecondaryProcess::LowEnergyDemotion, true);
                    fx.AccumulateLowEnergyElectron(
                        0,
                        r_particle,
                        z_particle,
                        particle_weight,
                        amrex::Real(0.0));
                    deposit_local_energy_as_ion_pairs(
                        fx,
                        r_particle,
                        z_particle,
                        particle_weight,
                        kinetic_energy_eV);
                    fx.SetInteractionConservationClosure(
                        particle_weight, kinetic_energy_eV, amrex::Real(0.0));
                    // Zero-length net chord: the charge leaves the kinetic
                    // population at the captured start point.
                    fx.AccumulateChargedSegment(
                        0, r_particle, z_particle, r_particle, z_particle,
                        -particle_weight, kRreaSegmentChannelDemoted);
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
                        "RREA charged transport path has positive length but no material segments");
                }
                amrex::Real const hard_moller_threshold_eV =
                    coupling.HardMollerSecondaryThresholdEv();
                bool const hard_moller_discrete_enabled =
                    !m_config.debug_disable_hard_moller_secondaries;
                auto const initial_stp = RreaChargedCompetingStpRates(
                    tables, /*is_positron=*/false, hard_moller_threshold_eV,
                    coupling.PhotonCutoffEv(), kinetic_energy_eV,
                    hard_moller_discrete_enabled);
                amrex::Real const initial_hard_stp = initial_stp[0];
                amrex::Real const initial_brems_stp = initial_stp[1];
                // Driver estimates: integrated over the driver chord's clipped
                // in-domain span, then divided by that span because the loop
                // transports the whole dt.
                amrex::Real const driver_density_path_m =
                    chord_density_path_m / driver_span;
                amrex::Real const initial_hard_tau =
                    initial_hard_stp * driver_density_path_m;
                amrex::Real const initial_brems_tau =
                    initial_brems_stp * driver_density_path_m;
                bool optical_depth_cap_violation = false;
                std::uint64_t subcycles = selected_material_subcycles(
                    initial_hard_tau + initial_brems_tau,
                    driver_intervals,
                    driver_path,
                    m_config,
                    optical_depth_cap_violation,
                    initial_hard_stp + initial_brems_stp);
                {
                    // Two additional deterministic whole-chord
                    // substep drivers beyond the hazard-tau refinement -- GS
                    // collision batching and the field-impulse/drag energy
                    // granularity.  Driver counts clamp silently at the cap:
                    // unlike an optical-depth violation this is benign
                    // refinement and does not weaken that strict guard.
                    amrex::Real const initial_gs_n_total =
                        elastic_inverse_length_stp(
                            tables, false, kinetic_energy_eV)
                        * driver_density_path_m;
                    amrex::Real const drag_estimate_eV =
                        electron_realized_continuous_loss_ev_per_m(
                            tables,
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
                        std::abs(field_work_eV) + drag_estimate_eV,
                        impulse_scale_eV);
                    subcycles = std::min<std::uint64_t>(
                        std::max({subcycles, gs_driver, impulse_driver}),
                        m_config.max_subcycles);
                }
                // Driver estimates as well: uniform slices of a chord the loop no
                // longer transports verbatim.  Reported, not enforced -- and
                // divided by the clipped span like every enforced budget above,
                // so the reported per-substep tau describes the full-dt
                // microchords the loop actually flies rather than the driver's
                // clipped fraction of them.
                amrex::Real const max_slice_density_path_m =
                    max_uniform_subcycle_density_path_m(
                        driver_intervals, driver_path, subcycles) / driver_span;
                amrex::Real const max_hard_tau =
                    initial_hard_stp * max_slice_density_path_m;
                amrex::Real const max_brems_tau =
                    initial_brems_stp * max_slice_density_path_m;
                amrex::Real const max_electron_total_competing_tau =
                    (initial_hard_stp + initial_brems_stp)
                    * max_slice_density_path_m;
                fx.tally.RecordTransportGuardMetrics(
                    max_hard_tau,
                    max_brems_tau,
                    max_electron_total_competing_tau,
                    amrex::Real(0.0),
                    amrex::Real(0.0),
                    subcycles,
                    kinetic_energy_eV,
                    amrex::Real(0.0),
                    amrex::Real(0.0),
                    optical_depth_cap_violation);
                abort_if_strict_guard_failed(m_config, optical_depth_cap_violation);

                // Apply one continuous stopping segment at a frozen incident
                // energy.  The caller splits the chord exactly at a sampled
                // discrete-event depth, so sources before the event precede its
                // atomic group and sources after it use the surviving parent's
                // post-event energy.  Below the tracked cutoff no certified table
                // may be queried; terminal handoff owns that residual energy.
                // One drag module for all charged loops: the plan owns the
                // cutoff-limited integrals and the loss partition
                // (RreaPlanContinuousDragSegment); this caller keeps only the
                // electron POLICY -- restricted collision + soft radiative
                // stopping, the hard-Moller mean rate feeding the demotion
                // force balance, the ledger records, the ion-pair sink, and
                // the C&D plane-crossing hook.
                auto apply_electron_continuous_segment = [&] (
                    amrex::Real begin_fraction,
                    amrex::Real end_fraction,
                    amrex::Real stopping_energy_eV)
                    -> RreaContinuousSegmentLimit {
                    amrex::Real const kinetic_energy_before_segment_eV =
                        kinetic_energy_eV;
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
                                RreaFrozenStpRate(tables.ElectronCollisionLoss(
                                    stopping_energy_eV, amrex::Real(1.0))),
                                RreaFrozenStpRate(tables.ElectronSoftRadiativeDrag(
                                    stopping_energy_eV, amrex::Real(1.0))),
                                (hard_moller_discrete_enabled
                                 && RreaHardMollerChannelOpen(
                                     stopping_energy_eV, hard_moller_threshold_eV))
                                ? RreaFrozenStpRate(
                                    tables.ElectronHardMollerEnergyLoss(
                                        stopping_energy_eV, amrex::Real(1.0)))
                                : amrex::Real(0.0)};
                        });
                    if (!plan.limit_valid) {
                        amrex::Abort(
                            "RREA electron continuous stopping could not locate "
                            "a finite monotone cutoff point");
                    }
                    if (!plan.applied) {
                        return RreaContinuousSegmentLimit{
                            begin_fraction, false, true};
                    }

                    fx.tally.RecordContinuousSoftBrems(
                        particle_weight, plan.soft_radiative_loss_eV);
                    fx.tally.RecordContinuousLossPartition(
                        particle_weight,
                        plan.drag_loss_eV,
                        plan.collision_loss_eV,
                        plan.hard_transfer_loss_eV);

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
                                amrex::Real low_pair_weight) {
                                amrex::Real event_x = amrex::Real(0.0);
                                amrex::Real event_y = amrex::Real(0.0);
                                amrex::Real event_z = amrex::Real(0.0);
                                transport_path.PointAtFullPathFraction(
                                    midpoint_fraction,
                                    event_x, event_y, event_z);
                                fx.AccumulateIonizationEvent(
                                    0,
                                    std::hypot(event_x, event_y),
                                    event_z,
                                    low_pair_weight,
                                    amrex::Real(0.5)
                                        * coupling.LowEnergyCutoffEv(),
                                    diagnostic_loss_eV);
                                fx.tally.RecordTransportIonization(
                                    low_pair_weight,
                                    diagnostic_loss_eV,
                                    true,
                                    false);
                            });
                    }

                    defer_cd_electron_plane_crossings(
                        fx,
                        coupling,
                        tables,
                        transport_path,
                        material_intervals,
                        begin_fraction,
                        plan.effective_end_fraction,
                        particle_weight,
                        kinetic_energy_before_segment_eV,
                        stopping_energy_eV,
                        plan.drag_loss_eV,
                        hard_moller_discrete_enabled);

                    kinetic_energy_eV = amrex::max(
                        amrex::Real(0.0),
                        kinetic_energy_eV - plan.drag_loss_eV);
                    if (plan.reaches_cutoff) {
                        kinetic_energy_eV = coupling.LowEnergyCutoffEv();
                    }
                    return RreaContinuousSegmentLimit{
                        plan.effective_end_fraction, plan.reaches_cutoff, true};
                };

                // The parent key already folds the particle's stored birth-rank
                // cpu() into the id, so per-rank id reuse cannot collide streams.
                std::uint64_t const rng_particle_id = transaction_parent_key;
                // Each rebuilt microchord is one complete subcycle; its local
                // fractions must not be sliced a second time.
                bool exited_domain = false;
                std::uint64_t completed_substeps = 0;
                // Fractions are local to one microchord. Every quantity that
                // is really an elapsed-time fraction of the OUTER step -- above all
                // a secondary's birth fraction -- must come through here, or a
                // secondary born late in the step is stamped early and
                // AdvanceChargedNewborn flies it a large multiple of the residual
                // it is owed.  The validator only range-checks [0,1].
                auto elapsed_fraction_at = [&](amrex::Real path_fraction) {
                    amrex::Real const within = amrex::min(
                        amrex::Real(1.0),
                        amrex::max(amrex::Real(0.0), path_fraction));
                    return amrex::min(
                        amrex::Real(1.0),
                        (static_cast<amrex::Real>(completed_substeps) + within)
                            / static_cast<amrex::Real>(subcycles));
                };
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
                // Substep-engine state and helpers. Under
                // substep_interleaved_v3 each uniform sub-interval advances
                // half field impulse -> drag/discrete hazards (existing
                // machinery, unchanged) -> GS elastic -> half field impulse,
                // with a mid-chord termination when the running energy falls
                // below the tracked cutoff strictly inside the chord.
                bool terminated_mid_chord = false;
                amrex::Real mid_chord_r_m = r_particle;
                amrex::Real mid_chord_z_m = z_particle;
                auto mark_electron_stopped_at = [&](amrex::Real path_fraction) {
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
                    mid_chord_z_m = stop_z;
                    mid_chord_r_m = stop_r;
                    terminated_mid_chord = true;
                };
                auto apply_field_half_kick = [&](
                    PathFieldSubRange const& field_range,
                    amrex::Real half_dt_s) {
                    static_cast<void>(RreaApplyChargedFieldHalfKick(
                        field_range, half_dt_s, amrex::Real(-1.0),
                        kinetic_energy_eV, dir_x, dir_y, dir_z));
                };
                auto apply_electron_gs_scatter = [&](
                    amrex::Real begin_fraction,
                    amrex::Real end_fraction,
                    std::uint64_t substep_idx) {
                    static constexpr GsScatterChannels kElectronGsChannels{
                        RreaRngChannel::ElectronGsFewCount,
                        RreaRngChannel::ElectronGsFewScatter,
                        RreaRngChannel::ElectronGsFewAzimuth,
                        RreaRngChannel::ElectronGsAngle,
                        RreaRngChannel::ElectronGsAzimuth};
                    auto gs_key = [&](RreaRngChannel channel,
                                      std::uint64_t inner) {
                        return RreaRngKey{coupling.RngSeed(), rng_particle_id,
                            static_cast<std::uint64_t>(std::max(step, 0)),
                            (substep_idx << 8) | inner,
                            static_cast<std::uint64_t>(channel)};
                    };
                    apply_gs_elastic_scatter(
                        material_intervals, transport_path,
                        begin_fraction, end_fraction, tables,
                        /*is_positron=*/false, kElectronGsChannels,
                        gs_key, kinetic_energy_eV, dir_x, dir_y, dir_z);
                };
                auto finish_interleaved_substep = [&](
                    std::uint64_t substep_idx,
                    amrex::Real begin_fraction,
                    amrex::Real end_fraction,
                    PathFieldSubRange const& substep_field,
                    amrex::Real substep_dt_s) -> bool {
                    if (kinetic_energy_eV >= coupling.LowEnergyCutoffEv()) {
                        apply_electron_gs_scatter(
                            begin_fraction, end_fraction, substep_idx);
                    }
                    apply_field_half_kick(
                        substep_field, amrex::Real(0.5) * substep_dt_s);
                    if (kinetic_energy_eV < coupling.LowEnergyCutoffEv()) {
                        // Stopped on this microchord: demote at its recorded
                        // endpoint, which is the electron's actual position.
                        // mark_electron_stopped_at itself turns
                        // a stop at the clip root of an exiting chord into an
                        // escape.
                        mark_electron_stopped_at(end_fraction);
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
                        // Rebuild from the current state so field work follows any
                        // within-step backscattering.
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
                                "RREA electron microchord has positive in-domain "
                                "length but no material intervals");
                        }
                        // The density the electron actually flies through, for the
                        // post-loop demotion gate.
                        for (auto const& interval : material_intervals) {
                            transported_density_path_m +=
                                interval.length_m * interval.density_ratio;
                            transported_length_m += interval.length_m;
                        }
                        subcycle_begin = transport_path.in_domain_begin_fraction;
                        subcycle_end = transport_path.in_domain_end_fraction;
                        substep_field = path_field_work_and_average(
                            coupling,
                            transport_path,
                            amrex::Real(-1.0),
                            subcycle_begin,
                            subcycle_end);
                        // Time ACTUALLY spent on this chord.  The work integral is
                        // already restricted to [begin, end], but the half-kick's
                        // ROTATION scales with dt, so handing it the whole subcycle
                        // over-rotates a clipped microchord by 1/(end-begin): energy
                        // exactly right, direction saturated to E, and no conservation
                        // ledger can see it.
                        substep_dt_s = sub_dt_s * (subcycle_end - subcycle_begin);
                        apply_field_half_kick(
                            substep_field, amrex::Real(0.5) * substep_dt_s);
                        if (kinetic_energy_eV < coupling.LowEnergyCutoffEv()) {
                            mark_electron_stopped_at(subcycle_begin);
                            return false;
                        }
                        return true;
                    },
                    [&](std::uint64_t subcycle,
                        amrex::Real subcycle_begin,
                        amrex::Real subcycle_end) {
                    // Event-loop policies: the electron's physics content; the
                    // substep_interleaved_v3 sequencing lives in
                    // RreaRunChargedEventInterleave, where its invariants are testable.
                    amrex::Real event_x = amrex::Real(0.0);
                    amrex::Real event_y = amrex::Real(0.0);
                    amrex::Real event_z = amrex::Real(0.0);
                    amrex::Real event_r = amrex::Real(0.0);
                    std::uint64_t interaction_rng_index = 0;
                    // Frozen STP inverse lengths of the two competing channels,
                    // refreshed by the channel_taus policy at every refreeze
                    // and read by the depth inversion that follows it.
                    std::array<amrex::Real, 3> stp{};
                    RreaRunChargedEventInterleave(
                        subcycle_begin,
                        subcycle_end,
                        kinetic_energy_eV,
                        coupling.LowEnergyCutoffEv(),
                        2,
                        kRreaTransientEventCapacity,
                        "electron",
                        [&](amrex::Real frozen_energy_eV,
                            amrex::Real tau_begin_fraction,
                            amrex::Real tau_end_fraction) {
                            stp = RreaChargedCompetingStpRates(
                                tables, /*is_positron=*/false,
                                hard_moller_threshold_eV,
                                coupling.PhotonCutoffEv(),
                                frozen_energy_eV,
                                hard_moller_discrete_enabled);
                            amrex::Real const segment_density_path_m = density_path_m(
                                material_intervals,
                                transport_path,
                                tau_begin_fraction,
                                tau_end_fraction);
                            amrex::Real const hard_tau =
                                stp[0] * segment_density_path_m;
                            amrex::Real const brems_tau =
                                stp[1] * segment_density_path_m;
                            amrex::Real const total_tau = hard_tau + brems_tau;
                            amrex::Real const total_probability =
                                probability_from_optical_depth(total_tau);
                            amrex::Real const hard_probability = total_tau > amrex::Real(0.0)
                                ? total_probability * hard_tau / total_tau
                                : amrex::Real(0.0);
                            fx.tally.RecordHardMollerExpectation(particle_weight * hard_probability);
                            return std::array<amrex::Real, 3>{
                                hard_tau, brems_tau, amrex::Real(0.0)};
                        },
                        [&](std::uint64_t draw_event_ordinal) {
                            auto const live_event_index = RreaTryLiveHazardRngIndex(
                                RreaSecondarySpecies::Electron,
                                subcycle, draw_event_ordinal);
                            if (!live_event_index.valid) {
                                amrex::Abort(
                                    "RREA electron repeated-hazard RNG index overflow; "
                                    "the uncommitted transaction is rejected");
                            }
                            interaction_rng_index = live_event_index.value;
                            RreaRngChannel const optical_depth_channel = draw_event_ordinal == 0U
                                ? RreaRngChannel::ElectronCompetingOpticalDepth
                                : RreaRngChannel::ElectronContinuationOpticalDepth;
                            RreaRngChannel const competing_channel = draw_event_ordinal == 0U
                                ? RreaRngChannel::ElectronCompetingChannel
                                : RreaRngChannel::ElectronContinuationChannel;
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
                            // tau = (sigma_hard + sigma_brems) * L_chi, so the
                            // depth inverts as a density path.
                            return fraction_at_density_path_m(
                                material_intervals,
                                transport_path,
                                invert_begin_fraction,
                                invert_end_fraction,
                                sampled_optical_depth / (stp[0] + stp[1]));
                        },
                        [&](amrex::Real segment_begin_fraction,
                            amrex::Real segment_end_fraction,
                            amrex::Real frozen_energy_eV) {
                            return apply_electron_continuous_segment(
                                segment_begin_fraction,
                                segment_end_fraction,
                                frozen_energy_eV);
                        },
                        [&](amrex::Real path_fraction) {
                            mark_electron_stopped_at(path_fraction);
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
                                tables, /*is_positron=*/false,
                                hard_moller_threshold_eV,
                                coupling.PhotonCutoffEv(),
                                channel, channel_energy_eV,
                                hard_moller_discrete_enabled);
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
                                RreaApplyHardMollerFinalState(
                                    fx, tables, rng_key, site, particle_weight,
                                    hard_moller_threshold_eV,
                                    kinetic_energy_eV, dir_x, dir_y, dir_z);
                            } else if (channel == 1) {
                                RreaApplyBremsFinalState(
                                    fx, tables, rng_key,
                                    [&](amrex::Real local_energy_eV) {
                                        deposit_local_energy_as_ion_pairs(
                                            fx, event_r, event_z,
                                            particle_weight, local_energy_eV);
                                    },
                                    site, particle_weight,
                                    /*charge_sign=*/amrex::Real(-1.0),
                                    coupling.PhotonCutoffEv(),
                                    kinetic_energy_eV, dir_x, dir_y, dir_z);
                            }
                            return true;
                        });
                    },
                    [&] { return terminated_mid_chord; },
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
                // Evaluate demotion once after transport, using traversed density
                // and the field at the reached endpoint. A per-substep gate would
                // make the disposition timestep-dependent.
                if (transported_length_m > amrex::Real(0.0)) {
                    particle_density_ratio =
                        transported_density_path_m / transported_length_m;
                }
                // run_* is the last COMMITTED point, which is always in-domain: a
                // refused commit leaves it where it was, so this gather can never
                // leave the field cache's halo.
                amrex::Real const endpoint_r_m = std::hypot(run_x, run_y);
                RreaFieldSample const endpoint_field =
                    coupling.GatherFieldRZ(0, endpoint_r_m, run_z);
                // The gate only sees K <= the hard-Moller threshold (50 keV),
                // where the ensemble mean drag is still above 284 kV/m: the drag
                // curve falls below that only above ~205 keV.
                amrex::Real const threshold_v_per_m =
                    electron_ensemble_mean_drag_ev_per_m(
                        tables,
                        amrex::max(kinetic_energy_eV, coupling.LowEnergyCutoffEv()),
                        particle_density_ratio,
                        hard_moller_discrete_enabled);
                auto const state_policy = RreaEvaluateElectronStatePolicy(
                        kinetic_energy_eV,
                        coupling.LowEnergyCutoffEv(),
                        hard_moller_threshold_eV,
                        std::hypot(
                            endpoint_field.er_v_per_m,
                            endpoint_field.ez_v_per_m),
                        threshold_v_per_m);
                // A microchord leaving the domain takes precedence over an
                // in-domain terminal disposition.
                auto const terminal_disposition =
                    RreaResolveChargedTransportTerminal(
                        exited_domain,
                        terminated_mid_chord,
                        RreaChargedTerminalDisposition::ElectronDemotion,
                        /*positron=*/false,
                        state_policy.below_transport_cutoff,
                        state_policy.low_energy_nonrunaway);
                if (terminal_disposition
                    == RreaChargedTerminalDisposition::Escape) {
                    // All in-domain stopping and sampled discrete events have
                    // been booked.  WarpX owns geometric overshoots; RREA owns
                    // exact half-open upper-face invalidation.  Neither may be
                    // demoted at the excluded endpoint.
                    commit_charged_escape();
                    continue;
                }

                // Only below-threshold, locally nonrunaway particles are handed
                // to the mesh; freshly-created above-cutoff knock-on electrons
                // remain in kinetic transport.
                if (terminal_disposition
                    == RreaChargedTerminalDisposition::ElectronDemotion) {
                    // A demoted electron can carry
                    // up to the hard-Moller threshold of kinetic energy;
                    // physically it ionizes ~K/W_air pairs within a sub-metre
                    // range before thermalizing.  Deposit that energy locally
                    // instead of dropping it;
                    // energy 0 in the mesh diagnostic avoids double counting.
                    // A mid-chord termination deposits at the
                    // true in-domain stopping position, not the far endpoint.
                    // Not the pre-push point and not WarpX's straight-line
                    // endpoint: where the electron actually ended up.
                    amrex::Real const demotion_r_m =
                        terminated_mid_chord ? mid_chord_r_m : endpoint_r_m;
                    amrex::Real const demotion_z_m =
                        terminated_mid_chord ? mid_chord_z_m : run_z;
                    fx.BeginGroup(RreaSecondaryProcess::LowEnergyDemotion, true);
                    fx.AccumulateLowEnergyElectron(
                        0,
                        demotion_r_m,
                        demotion_z_m,
                        particle_weight,
                        amrex::Real(0.0));
                    deposit_local_energy_as_ion_pairs(
                        fx,
                        demotion_r_m,
                        demotion_z_m,
                        particle_weight,
                        kinetic_energy_eV);
                    fx.SetInteractionConservationClosure(
                        particle_weight, kinetic_energy_eV, amrex::Real(0.0));
                    // Net chord: captured start to the mid-chord kill point
                    // (or the committed endpoint when the gate fired there).
                    fx.AccumulateChargedSegment(
                        0, r_particle, z_particle, demotion_r_m, demotion_z_m,
                        -particle_weight, kRreaSegmentChannelDemoted);
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
                    endpoint_r_m,
                    /*signed_mesh_weight=*/-particle_weight);
            }
            }  // omp parallel chunk loop

            // Deterministic merge in chunk order.
            for (auto& fx : chunk_fx) {
                pending_side_effects.AppendGroupsFrom(fx);
                invalidated_particles = invalidated_particles || fx.invalidated;
                repositioned_electrons = repositioned_electrons || fx.repositioned;
            }
        }
#else
        amrex::Abort("RREA MC/PIC v2 production transport currently requires WarpX RZ");
#endif
}

}  // namespace rrea::warpx
