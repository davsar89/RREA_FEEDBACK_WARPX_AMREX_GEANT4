#include "RreaTransportStep.H"

#ifdef AMREX_USE_OMP
#include <omp.h>
#endif

namespace rrea::warpx {

void TransportStep::ApplyPhotonTransport()
{
#if defined(WARPX_DIM_RZ)
            auto& photons =
                warpx.GetPartContainer().GetParticleContainerFromName(coupling.PhotonSpeciesName());
            PreviousRzComponentIndices const photon_previous =
                require_previous_rz_components(photons, coupling.PhotonSpeciesName());
            // Set when a surviving Compton photon is repositioned to its mid-flight
            // interaction point + remaining path; drives a collective Redistribute
            // after the loop since the move can cross a tile/box boundary.
            for (WarpXParIter pti(photons, 0); pti.isValid(); ++pti) {
                auto& tile = pti.GetParticleTile();
                auto& attribs = pti.GetAttribs();
                auto* const r_data = attribs[PIdx::r].dataPtr();
                auto* const z_data = attribs[PIdx::z].dataPtr();
                auto* const w_data = attribs[PIdx::w].dataPtr();
                auto* const ux_data = attribs[PIdx::ux].dataPtr();
                auto* const uy_data = attribs[PIdx::uy].dataPtr();
                auto* const uz_data = attribs[PIdx::uz].dataPtr();
                auto* const theta_data = attribs[PIdx::theta].dataPtr();
                auto* const previous_r_data = pti.GetAttribs(photon_previous.r).dataPtr();
                auto* const previous_z_data = pti.GetAttribs(photon_previous.z).dataPtr();
                auto* const previous_theta_data = pti.GetAttribs(photon_previous.theta).dataPtr();
                long const np = pti.numParticles();
                // Same transactional, per-chunk logging as the electron loop.
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
                    std::uint64_t transaction_parent_key =
                        static_cast<std::uint64_t>(std::llabs(static_cast<amrex::Long>(particle_id)))
                        ^ (static_cast<std::uint64_t>(tile.cpu(static_cast<int>(ip))) << 40);
                    fx.SetParentContext(
                        transaction_parent_key, RreaSecondarySpecies::Photon);
                    amrex::Real const particle_weight = w_data[ip];
                    amrex::Real photon_energy_eV =
                        std::sqrt(
                            ux_data[ip] * ux_data[ip]
                            + uy_data[ip] * uy_data[ip]
                            + uz_data[ip] * uz_data[ip]);
                    amrex::Real dir_x = ux_data[ip];
                    amrex::Real dir_y = uy_data[ip];
                    amrex::Real dir_z = uz_data[ip];
                    RreaNormalizeDirection(dir_x, dir_y, dir_z);
                    // Transport precedes WarpX absorbing-boundary deletion. Use
                    // only the captured chord's in-domain part for optical depth
                    // and deposits.
                    if (previous_r_data[ip] > geom.ProbHi(0)
                            + amrex::Real(1.0e-9) * geom.ProbHi(0)
                        || previous_z_data[ip] < geom.ProbLo(1)
                            - amrex::Real(1.0e-9)
                        || previous_z_data[ip] > geom.ProbHi(1)
                            + amrex::Real(1.0e-9) * geom.ProbHi(1)) {
                        std::ostringstream survivor;
                        survivor << std::setprecision(17)
                            << "RREA photon survived outside the domain: prev_r="
                            << previous_r_data[ip]
                            << " prev_z=" << previous_z_data[ip]
                            << " r=" << r_data[ip] << " z=" << z_data[ip]
                            << " w=" << w_data[ip]
                            << " E_eV=" << photon_energy_eV
                            << " id=" << static_cast<amrex::Long>(
                                tile.id(static_cast<int>(ip)))
                            << " cpu=" << tile.cpu(static_cast<int>(ip))
                            << " step=" << step;
                        amrex::Abort(survivor.str());
                    }
                    RreaTransportPathSegment transport_path = ClipRzTransportPath(
                        previous_r_data[ip],
                        previous_theta_data[ip],
                        previous_z_data[ip],
                        r_data[ip],
                        theta_data[ip],
                        z_data[ip],
                        geom.ProbLo(0),
                        geom.ProbHi(0),
                        geom.ProbLo(1),
                        geom.ProbHi(1),
                        periodic_z);
                    std::uint64_t photon_event_ordinal = 0;
                    amrex::Real elapsed_path_start_fraction = amrex::Real(0.0);
                    static constexpr amrex::Real light_speed_m_per_s =
                        kRreaSpeedOfLightMPerS;
                    while (true) {
                    if (photon_event_ordinal >= kRreaTransientEventCapacity) {
                        amrex::Abort(
                            "RREA photon repeated-hazard event ordinal overflow; "
                            "the uncommitted transaction is rejected");
                    }
                    RreaEndpointEscapePolicy const endpoint_escape =
                        RreaClippedTransportEscapePolicyForPath(
                            transport_path,
                            std::hypot(
                                transport_path.end_x_m,
                                transport_path.end_y_m),
                            transport_path.end_z_m);
                    amrex::Real const ds_m = transport_path.in_domain_length_m;
                    // Same material path the charged loops use.  The
                    // piecewise-mean chi places the interaction point with an
                    // error of at most h^2/8H = 9 um on a 0.75 m chord -- far
                    // inside the cell the deposit lands in.
                    auto const photon_intervals = build_material_path_intervals(
                        coupling, transport_path, density_ratio);
                    amrex::Real const chord_density_path_m = density_path_m(
                        photon_intervals,
                        transport_path,
                        transport_path.in_domain_begin_fraction,
                        transport_path.in_domain_end_fraction);
                    if (particle_weight <= amrex::Real(0.0)
                        || photon_energy_eV < coupling.PhotonCutoffEv()) {
                        fx.BeginGroup(RreaSecondaryProcess::LowEnergyDemotion, true);
                        amrex::Real x_deposit = amrex::Real(0.0);
                        amrex::Real y_deposit = amrex::Real(0.0);
                        amrex::Real z_deposit = amrex::Real(0.0);
                        // A photon already below the production cutoff does not
                        // propagate.  Deposit at its captured path start, never
                        // at a clipped absorbing endpoint that lies on the
                        // excluded upper face.
                        transport_path.PointAtFullPathFraction(
                            transport_path.in_domain_begin_fraction,
                            x_deposit,
                            y_deposit,
                            z_deposit);
                        deposit_local_energy_as_ion_pairs(fx,
                            std::hypot(x_deposit, y_deposit),
                            z_deposit,
                            amrex::max(particle_weight, amrex::Real(0.0)),
                            photon_energy_eV);
                        if (particle_weight > amrex::Real(0.0)) {
                            fx.SetInteractionConservationClosure(
                                particle_weight,
                                photon_energy_eV,
                                amrex::Real(0.0));
                        } else {
                            fx.SetExplicitZeroConservationClosure();
                        }
                        fx.EndGroup();
                        particle_id.make_invalid();
                        fx.invalidated = true;
                        break;
                    }

                    // The parent key already folds the particle's stored birth-rank
                    // cpu() into the id, so per-rank id reuse cannot collide streams.
                    std::uint64_t const rng_particle_id = transaction_parent_key;
                    auto commit_no_event_photon_escape = [&]() {
                        amrex::Real endpoint_x = amrex::Real(0.0);
                        amrex::Real endpoint_y = amrex::Real(0.0);
                        amrex::Real endpoint_z = amrex::Real(0.0);
                        transport_path.PointAtFullPathFraction(
                            transport_path.in_domain_end_fraction,
                            endpoint_x, endpoint_y, endpoint_z);
                        r_data[ip] = static_cast<amrex::ParticleReal>(
                            std::hypot(endpoint_x, endpoint_y));
                        theta_data[ip] = static_cast<amrex::ParticleReal>(
                            std::atan2(endpoint_y, endpoint_x));
                        z_data[ip] = static_cast<amrex::ParticleReal>(
                            transport_path.WrapZ(endpoint_z));
                        if (!endpoint_escape.escapes) {
                            return;
                        }
                        fx.BeginGroup(
                            RreaSecondaryProcess::PhotonEscape,
                            /*remove_parent=*/true,
                            endpoint_escape.ParentStillValidAtPreflight());
                        fx.tally.RecordPhotonEscape(particle_weight, photon_energy_eV);
                        fx.AddEscapedConservationEnergy(
                            particle_weight, photon_energy_eV);
                        fx.SetInteractionConservationClosure(
                            particle_weight,
                            photon_energy_eV,
                            amrex::Real(0.0));
                        fx.EndGroup();
                        if (endpoint_escape.invalidate_parent_in_rrea) {
                            // Make the clipped recorded escape
                            // atomic with parent removal before the alive-energy
                            // diagnostic and collective cap preflight.
                            particle_id.make_invalid();
                            fx.invalidated = true;
                        }
                    };
                    if (ds_m <= amrex::Real(0.0)) {
                        commit_no_event_photon_escape();
                        break;
                    }
                    amrex::Real const photon_mfp_stp =
                        tables.PhotonFeedbackMeanFreePath(
                            photon_energy_eV,
                            amrex::Real(1.0));
                    amrex::Real const photon_tau =
                        chord_density_path_m / photon_mfp_stp;
                    // Each pass samples one exact exponential optical depth on
                    // the currently remaining chord.  A surviving Compton
                    // state rebuilds its rates and returns through this loop;
                    // terminal photoelectric/pair states never do.
                    fx.tally.RecordTransportGuardMetrics(
                        amrex::Real(0.0),
                        amrex::Real(0.0),
                        amrex::Real(0.0),
                        photon_tau,
                        amrex::Real(0.0),
                        // One exact exponential optical depth is sampled per pass.
                        1U,
                        amrex::Real(0.0),
                        photon_energy_eV,
                        amrex::Real(0.0),
                        false);
                    // Sample one exponential optical depth over the complete
                    // clipped chord.  This is independent of the diagnostic
                    // subcycle partition and is exact for the step's sampled
                    // density/rate, unlike repeated Bernoulli trials followed by
                    // a uniform-within-substep interaction point.
                    auto const live_event_index = RreaTryLiveHazardRngIndex(
                        RreaSecondarySpecies::Photon,
                        0U, photon_event_ordinal);
                    if (!live_event_index.valid) {
                        amrex::Abort(
                            "RREA photon repeated-hazard RNG index overflow; "
                            "the uncommitted transaction is rejected");
                    }
                    std::uint64_t const interaction_rng_index =
                        live_event_index.value;
                    RreaRngChannel const optical_depth_channel =
                        photon_event_ordinal == 0U
                        ? RreaRngChannel::PhotonInteraction
                        : RreaRngChannel::PhotonContinuationOpticalDepth;
                    amrex::Real const interact_draw = RreaRng::Uniform01(RreaRngKey{
                        coupling.RngSeed(),
                        rng_particle_id,
                        static_cast<std::uint64_t>(std::max(step, 0)),
                        interaction_rng_index,
                        static_cast<std::uint64_t>(optical_depth_channel)});
                    RreaRemainingSegmentHazard const hazard =
                        RreaResolveRemainingSegmentHazard(
                            {photon_tau, amrex::Real(0.0), amrex::Real(0.0)},
                            1,
                            transport_path.in_domain_begin_fraction,
                            transport_path.in_domain_end_fraction,
                            interact_draw,
                            amrex::Real(0.0),
                            [&](amrex::Real sampled_optical_depth) {
                                return fraction_at_density_path_m(
                                    photon_intervals,
                                    transport_path,
                                    transport_path.in_domain_begin_fraction,
                                    transport_path.in_domain_end_fraction,
                                    sampled_optical_depth * photon_mfp_stp);
                            });
                    if (!hazard.selection.event) {
                        commit_no_event_photon_escape();
                        break;
                    }
                    amrex::Real const full_path_fraction =
                        hazard.event_fraction;
                    amrex::Real const interaction_distance_m =
                        transport_path.full_length_m
                        * (full_path_fraction
                            - transport_path.in_domain_begin_fraction);
                    amrex::Real x_particle = amrex::Real(0.0);
                    amrex::Real y_particle = amrex::Real(0.0);
                    amrex::Real z_particle = amrex::Real(0.0);
                    transport_path.PointAtFullPathFraction(
                        full_path_fraction, x_particle, y_particle, z_particle);
                    // A scatter changes whether the photon would have reached
                    // its pre-scatter absorbing-boundary intersection. Preserve the
                    // unused geometric distance from the complete WarpX push,
                    // then clip the new chord independently below.  Using the
                    // pre-scatter in-domain remainder would shorten inward-scattered
                    // photons merely because their unscattered trajectory
                    // would have escaped.
                    amrex::Real const remaining_m = amrex::max(
                        amrex::Real(0.0),
                        transport_path.full_length_m - interaction_distance_m);
                    amrex::Real const birth_elapsed_step_fraction = amrex::min(
                        amrex::Real(1.0),
                        elapsed_path_start_fraction
                            + interaction_distance_m
                                / amrex::max(
                                    light_speed_m_per_s * dt_s,
                                    std::numeric_limits<amrex::Real>::min()));
                    bool repeat_compton_hazard = false;
                    RreaTransportPathSegment next_compton_path;
                    amrex::Real next_compton_elapsed_fraction =
                        birth_elapsed_step_fraction;
                    amrex::Real const r_particle = std::hypot(x_particle, y_particle);
                    RreaRngChannel const photon_channel_rng =
                        photon_event_ordinal == 0U
                        ? RreaRngChannel::PhotonChannel
                        : RreaRngChannel::PhotonContinuationChannel;
                    RreaPhotonChannel const channel =
                        tables.SamplePhotonChannel(
                            photon_energy_eV,
                                RreaRng::Uniform01(RreaRngKey{
                                    coupling.RngSeed(),
                                    rng_particle_id,
                                    static_cast<std::uint64_t>(std::max(step, 0)),
                                    interaction_rng_index,
                                    static_cast<std::uint64_t>(photon_channel_rng)}));
                    RreaInteractionSite const site{
                        x_particle, y_particle, z_particle, r_particle,
                        birth_elapsed_step_fraction};
                    auto rng_key = [&](RreaRngChannel rng_channel) {
                        return RreaRngKey{
                            coupling.RngSeed(),
                            rng_particle_id,
                            static_cast<std::uint64_t>(std::max(step, 0)),
                            interaction_rng_index,
                            static_cast<std::uint64_t>(rng_channel)};
                    };
                    auto deposit_local = [&](amrex::Real local_energy_eV) {
                        deposit_local_energy_as_ion_pairs(
                            fx, r_particle, z_particle,
                            particle_weight, local_energy_eV);
                    };
                    if (channel == RreaPhotonChannel::Compton) {
                        if (RreaApplyComptonFinalState(
                                fx, tables, rng_key, deposit_local, site,
                                particle_weight,
                                coupling.LowEnergyCutoffEv(),
                                coupling.PhotonCutoffEv(),
                                photon_energy_eV, dir_x, dir_y, dir_z)) {
                            ux_data[ip] = static_cast<amrex::ParticleReal>(photon_energy_eV * dir_x);
                            uy_data[ip] = static_cast<amrex::ParticleReal>(photon_energy_eV * dir_y);
                            uz_data[ip] = static_cast<amrex::ParticleReal>(photon_energy_eV * dir_z);
                            // The scattered photon interacted mid-flight, so
                            // it continues from the interaction point along the new
                            // direction for the remaining path.  Write back the RZ
                            // (r, theta, z) position; the move can cross a tile/box
                            // boundary, so flag a collective Redistribute after the loop.
                            amrex::Real const x_target =
                                x_particle + dir_x * remaining_m;
                            amrex::Real const y_target =
                                y_particle + dir_y * remaining_m;
                            amrex::Real const z_target =
                                z_particle + dir_z * remaining_m;
                            RreaTransportPathSegment const continuation_path =
                                ClipRzTransportPath(
                                    r_particle,
                                    std::atan2(y_particle, x_particle),
                                    z_particle,
                                    std::hypot(x_target, y_target),
                                    std::atan2(y_target, x_target),
                                    z_target,
                                    geom.ProbLo(0),
                                    geom.ProbHi(0),
                                    geom.ProbLo(1),
                                    geom.ProbHi(1),
                                    periodic_z);
                            amrex::Real x_new = amrex::Real(0.0);
                            amrex::Real y_new = amrex::Real(0.0);
                            amrex::Real z_new = amrex::Real(0.0);
                            continuation_path.PointAtFullPathFraction(
                                continuation_path.in_domain_end_fraction,
                                x_new, y_new, z_new);
                            r_data[ip] = static_cast<amrex::ParticleReal>(
                                std::hypot(x_new, y_new));
                            theta_data[ip] = static_cast<amrex::ParticleReal>(
                                std::atan2(y_new, x_new));
                            z_data[ip] = static_cast<amrex::ParticleReal>(
                                continuation_path.WrapZ(z_new));
                            // The scattered photon remains transiently live
                            // over the complete unused time.  Do not classify
                            // an absorbing-boundary endpoint until the
                            // continuation hazard has been sampled over its
                            // in-domain prefix.
                            next_compton_path = continuation_path;
                            repeat_compton_hazard = remaining_m > amrex::Real(0.0);
                            fx.repositioned = true;
                        } else {
                            particle_id.make_invalid();
                            fx.invalidated = true;
                        }
                    } else if (channel == RreaPhotonChannel::Photoelectric) {
                        RreaApplyPhotoelectricFinalState(
                            fx, tables, rng_key, deposit_local, site,
                            particle_weight, coupling.LowEnergyCutoffEv(),
                            photon_energy_eV, dir_x, dir_y, dir_z);
                        particle_id.make_invalid();
                        fx.invalidated = true;
                    } else if (channel == RreaPhotonChannel::PairNuclear
                               || channel == RreaPhotonChannel::PairTriplet) {
                        RreaApplyPairFinalState(
                            fx, tables, rng_key, deposit_local,
                            [&](amrex::Real positron_dir_x,
                                amrex::Real positron_dir_y,
                                amrex::Real positron_dir_z,
                                amrex::Real positron_kinetic_eV) {
                                spawn_positron_annihilation(
                                    fx,
                                    x_particle, y_particle, z_particle,
                                    r_particle, z_particle,
                                    positron_dir_x,
                                    positron_dir_y,
                                    positron_dir_z,
                                    particle_weight,
                                    positron_kinetic_eV,
                                    true,
                                    rng_particle_id,
                                    interaction_rng_index,
                                    birth_elapsed_step_fraction);
                            },
                            site, particle_weight,
                            coupling.LowEnergyCutoffEv(),
                            photon_energy_eV, channel,
                            dir_x, dir_y, dir_z);
                        particle_id.make_invalid();
                        fx.invalidated = true;
                    }
                    if (repeat_compton_hazard) {
                        transport_path = std::move(next_compton_path);
                        elapsed_path_start_fraction =
                            next_compton_elapsed_fraction;
                        ++photon_event_ordinal;
                        continue;
                    }
                    break;
                    }
                    // Photon terminal/no-event disposition is complete, or a
                    // Compton survivor exhausted the residual outer-step time.
                    continue;
                }
                }  // omp parallel chunk loop

                // Deterministic merge in chunk order.
                for (auto& fx : chunk_fx) {
                    pending_side_effects.AppendGroupsFrom(fx);
                    invalidated_photons = invalidated_photons || fx.invalidated;
                    repositioned_photons = repositioned_photons || fx.repositioned;
                }
            }
#else
            amrex::Abort(
                "RREA MC/PIC photon feedback currently requires WarpX RZ");
#endif
}

}  // namespace rrea::warpx
