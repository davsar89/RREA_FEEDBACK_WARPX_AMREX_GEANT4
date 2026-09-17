#include "RreaTransportStep.H"

#ifdef AMREX_USE_OMP
#include <omp.h>
#endif

namespace rrea::warpx {

void TransportStep::DrainAndCommitSecondaries()
{
        auto const electron_secondary_range =
            tables.ElectronCertifiedEnergyRangeEv();
        auto const positron_secondary_range =
            tables.PositronCertifiedEnergyRangeEv();
        auto const photon_secondary_range =
            tables.PhotonCertifiedEnergyRangeEv();
        auto transient_energy_is_certified = [&](RreaSecondaryParticle const& p) {
            auto in_range = [](amrex::Real energy_eV,
                               rrea::RreaCertifiedEnergyRange const& range) {
                return std::isfinite(static_cast<double>(energy_eV))
                    && std::isfinite(static_cast<double>(range.min_eV))
                    && std::isfinite(static_cast<double>(range.max_eV))
                    && range.min_eV > amrex::Real(0.0)
                    && range.max_eV >= range.min_eV
                    && range.Contains(energy_eV);
            };
            switch (p.species) {
            case RreaSecondarySpecies::Electron:
                return in_range(
                    p.kinetic_or_photon_energy_eV,
                    electron_secondary_range);
            case RreaSecondarySpecies::Positron:
                return in_range(
                    p.kinetic_or_photon_energy_eV,
                    positron_secondary_range);
            case RreaSecondarySpecies::Photon:
                return in_range(
                    p.kinetic_or_photon_energy_eV,
                    photon_secondary_range);
            }
            return false;
        };
        // Validate the complete deferred payload before any population-cap
        // decision or replay.  This is intentionally defined here, but called
        // only after the transient newborn work graph has been expanded.  A
        // bad group on one rank then aborts the whole original-root
        // transaction before any correlated population or ledger mutation.
        auto validate_complete_transaction = [&]() {
          // [0] position, [1] payload/source-ledger, [2] missing/non-finite
          // conservation contract, [3] finite charge/energy residual
          // violation, [4] tracked charged photon product outside its
          // certified range.
          std::array<amrex::Long, 5> global_transaction_validation_errors{
              0, 0, 0, 0, 0};
          auto photon_flow_matches = [](amrex::Real actual, amrex::Real expected,
                                        amrex::Real scale) {
            amrex::Real constexpr eps64 =
                amrex::Real(64.0) * std::numeric_limits<double>::epsilon();
            return std::isfinite(static_cast<double>(actual))
                && actual >= amrex::Real(0.0)
                && std::abs(actual - expected)
                    <= eps64 * std::max(scale, amrex::Real(1.0));
          };
          for (auto const& group : pending_side_effects.groups) {
            std::uint64_t const invalid_photon_charged_secondaries =
                RreaCountInvalidTrackedPhotonChargedSecondaries(
                    group,
                    electron_secondary_range.min_eV,
                    electron_secondary_range.max_eV,
                    positron_secondary_range.min_eV,
                    positron_secondary_range.max_eV);
            if (invalid_photon_charged_secondaries
                > static_cast<std::uint64_t>(
                    std::numeric_limits<amrex::Long>::max())
                    - static_cast<std::uint64_t>(
                        global_transaction_validation_errors[4])) {
                global_transaction_validation_errors[4] =
                    std::numeric_limits<amrex::Long>::max();
            } else {
                global_transaction_validation_errors[4] +=
                    static_cast<amrex::Long>(
                        invalid_photon_charged_secondaries);
            }
            auto const conservation = RreaValidateSecondaryGroupConservation(
                group, coupling.LowEnergyCutoffEv());
            if (conservation.required
                && (!conservation.present || !conservation.finite)) {
                ++global_transaction_validation_errors[2];
            } else if (conservation.required && !conservation.valid) {
                ++global_transaction_validation_errors[3];
            }
            std::size_t material_source_index = 0;
            for (auto const& op : group.ledger_entries) {
                if (op.kind == RreaLedgerOpKind::RecordComptonInteraction
                    || op.kind == RreaLedgerOpKind::RecordPhotoelectricInteraction
                    || op.kind == RreaLedgerOpKind::RecordPairProduction) {
                    bool valid_flow_payload =
                        std::isfinite(static_cast<double>(op.a[0]))
                        && std::isfinite(static_cast<double>(op.a[1]))
                        && std::isfinite(static_cast<double>(op.a[2]))
                        && op.a[0] > amrex::Real(0.0)
                        && op.a[1] > amrex::Real(0.0)
                        && op.a[2] >= amrex::Real(0.0);
                    RreaPhotonEnergyFlowSplit expected_flow;
                    amrex::Real actual_charged = amrex::Real(0.0);
                    amrex::Real actual_local = amrex::Real(0.0);
                    if (op.kind
                        == RreaLedgerOpKind::RecordComptonInteraction) {
                        expected_flow = RreaComptonPhotonEnergyFlow(
                            op.a[2], op.a[3],
                            coupling.LowEnergyCutoffEv(),
                            coupling.PhotonCutoffEv());
                        actual_charged = op.a[4];
                        actual_local = op.a[5];
                        valid_flow_payload = valid_flow_payload
                            && std::isfinite(static_cast<double>(op.a[3]))
                            && op.a[3] >= amrex::Real(0.0)
                            && photon_flow_matches(
                                op.a[2] + op.a[3], op.a[1], op.a[1]);
                    } else if (op.kind
                               == RreaLedgerOpKind::RecordPhotoelectricInteraction) {
                        expected_flow = RreaPhotoelectricPhotonEnergyFlow(
                            op.a[2], op.a[3],
                            coupling.LowEnergyCutoffEv());
                        actual_charged = op.a[4];
                        actual_local = op.a[5];
                        valid_flow_payload = valid_flow_payload
                            && std::isfinite(static_cast<double>(op.a[3]))
                            && op.a[3] >= amrex::Real(0.0)
                            && photon_flow_matches(
                                op.a[2] + op.a[3], op.a[1], op.a[1]);
                    } else {
                        valid_flow_payload = valid_flow_payload
                            && std::isfinite(static_cast<double>(op.a[3]))
                            && std::isfinite(static_cast<double>(op.a[4]))
                            && op.a[3] >= amrex::Real(0.0)
                            && op.a[4] >= amrex::Real(0.0);
                        amrex::Real const final_state_local_eV = amrex::max(
                            op.a[1]
                                - amrex::Real(2.0)
                                    * RreaSpeciesRestEnergyEv(
                                        RreaSecondarySpecies::Electron)
                                - op.a[2] - op.a[3]
                                - (op.b0 ? op.a[4] : amrex::Real(0.0)),
                            amrex::Real(0.0));
                        expected_flow = RreaPairPhotonEnergyFlow(
                            op.a[2], op.a[3], op.a[4], final_state_local_eV,
                            coupling.LowEnergyCutoffEv(), op.b0);
                        actual_charged = op.a[5];
                        actual_local = op.a[6];
                    }
                    valid_flow_payload = valid_flow_payload
                        && expected_flow.Valid()
                        && photon_flow_matches(
                            actual_charged,
                            expected_flow.charged_kinetic_rest_mass_eV,
                            op.a[1])
                        && photon_flow_matches(
                            actual_local,
                            expected_flow.local_recoil_binding_subcutoff_eV,
                            op.a[1]);
                    if (!valid_flow_payload) {
                        ++global_transaction_validation_errors[1];
                    }
                }
                bool const is_material_source_op =
                    op.kind == RreaLedgerOpKind::AccumulateIonizationEvent
                    || op.kind == RreaLedgerOpKind::AccumulateLowEnergyElectron
                    || op.kind == RreaLedgerOpKind::AccumulatePositiveIonSourceOnly;
                if (!is_material_source_op) {
                    continue;
                }
                if (material_source_index >= group.material_sources.size()
                    || !RreaMaterialSourceMatchesLedgerEntry(
                        group.material_sources[material_source_index], op)) {
                    ++global_transaction_validation_errors[1];
                }
                ++material_source_index;
            }
            if (material_source_index != group.material_sources.size()) {
                ++global_transaction_validation_errors[1];
            }
            for (auto const& source : group.material_sources) {
                bool const finite_payload =
                    std::isfinite(static_cast<double>(source.event_weight))
                    && std::isfinite(static_cast<double>(source.secondary_energy_eV))
                    && std::isfinite(static_cast<double>(source.energy_loss_eV));
                if (!finite_payload || !(source.event_weight > amrex::Real(0.0))
                    || source.secondary_energy_eV < amrex::Real(0.0)
                    || source.energy_loss_eV < amrex::Real(0.0)) {
                    ++global_transaction_validation_errors[1];
                }
                if (!coupling.SecondaryGroupPositionInDomain(
                        source.level, source.r_m, source.z_m)) {
                    ++global_transaction_validation_errors[0];
                }
            }
            for (auto const& particle : group.particles) {
                bool const known_species =
                    particle.species == RreaSecondarySpecies::Electron
                    || particle.species == RreaSecondarySpecies::Photon
                    || particle.species == RreaSecondarySpecies::Positron;
                bool const known_root_species =
                    particle.root_species == RreaSecondarySpecies::Electron
                    || particle.root_species == RreaSecondarySpecies::Photon
                    || particle.root_species == RreaSecondarySpecies::Positron;
                bool const finite_payload =
                    std::isfinite(static_cast<double>(particle.x_m))
                    && std::isfinite(static_cast<double>(particle.y_m))
                    && std::isfinite(static_cast<double>(particle.z_m))
                    && std::isfinite(static_cast<double>(particle.dir_x))
                    && std::isfinite(static_cast<double>(particle.dir_y))
                    && std::isfinite(static_cast<double>(particle.dir_z))
                    && std::isfinite(static_cast<double>(particle.weight))
                    && std::isfinite(static_cast<double>(
                        particle.kinetic_or_photon_energy_eV))
                    && std::isfinite(static_cast<double>(
                        particle.birth_elapsed_step_fraction));
                bool const valid_energy =
                    particle.species == RreaSecondarySpecies::Photon
                    ? particle.kinetic_or_photon_energy_eV > amrex::Real(0.0)
                    : particle.kinetic_or_photon_energy_eV >= amrex::Real(0.0);
                bool const valid_birth_metadata = known_root_species
                    && particle.birth_elapsed_step_fraction >= amrex::Real(0.0)
                    && particle.birth_elapsed_step_fraction <= amrex::Real(1.0)
                    && particle.transient_lineage
                        < kRreaTransientLineageCapacity;
                if (!known_species || !finite_payload || !valid_birth_metadata
                    || !(particle.weight > amrex::Real(0.0)) || !valid_energy) {
                    ++global_transaction_validation_errors[1];
                }
                amrex::Real const particle_r_m =
                    std::hypot(particle.x_m, particle.y_m);
                if (!coupling.SecondaryGroupPositionInDomain(
                        0, particle_r_m, particle.z_m)) {
                    ++global_transaction_validation_errors[0];
                }
            }
          }
          amrex::ParallelDescriptor::ReduceLongSum(
            global_transaction_validation_errors.data(),
            global_transaction_validation_errors.size());
          if (global_transaction_validation_errors[0] != 0
            || global_transaction_validation_errors[1] != 0
            || global_transaction_validation_errors[2] != 0
            || global_transaction_validation_errors[3] != 0
            || global_transaction_validation_errors[4] != 0) {
            std::ostringstream message;
            message
                << "RREA secondary transaction failed collective atomic validation: "
                << global_transaction_validation_errors[0]
                << " source/particle positions are outside the half-open physical domain and "
                << global_transaction_validation_errors[1]
                << " payloads are invalid, "
                << global_transaction_validation_errors[2]
                << " interaction closures are missing/non-finite, and "
                << global_transaction_validation_errors[3]
                << " interaction closures exceed the 64-binary64-epsilon "
                   "charge/energy residual gates, and "
                << global_transaction_validation_errors[4]
                << " tracked charged products of certified-range photons are "
                   "outside their charged certified ranges (sub-floor products "
                   "must be fluid-demoted); no group was committed";
              amrex::Abort(message.str());
          }
        };

        std::vector<amrex::Real> spawn_x;
        std::vector<amrex::Real> spawn_y;
        std::vector<amrex::Real> spawn_z;
        std::vector<amrex::Real> spawn_ux;
        std::vector<amrex::Real> spawn_uy;
        std::vector<amrex::Real> spawn_uz;
        std::vector<amrex::Real> spawn_weight;
        std::vector<amrex::Real> spawn_energy;
        std::vector<amrex::Real> photon_x;
        std::vector<amrex::Real> photon_y;
        std::vector<amrex::Real> photon_z;
        std::vector<amrex::Real> photon_ux;
        std::vector<amrex::Real> photon_uy;
        std::vector<amrex::Real> photon_uz;
        std::vector<amrex::Real> photon_weight;
        std::vector<amrex::Real> photon_energy;
        std::vector<amrex::Real> positron_x;
        std::vector<amrex::Real> positron_y;
        std::vector<amrex::Real> positron_z;
        std::vector<amrex::Real> positron_ux;
        std::vector<amrex::Real> positron_uy;
        std::vector<amrex::Real> positron_uz;
        std::vector<amrex::Real> positron_weight;
        std::vector<amrex::Real> positron_energy;
        std::array<amrex::Long, 3> local_removals{0, 0, 0};
        std::array<std::vector<std::uint64_t>, 3> removed_parent_keys;
        auto species_index = [](RreaSecondarySpecies species) -> std::size_t {
            switch (species) {
            case RreaSecondarySpecies::Electron: return 0;
            case RreaSecondarySpecies::Photon: return 1;
            case RreaSecondarySpecies::Positron: return 2;
            }
            amrex::Abort("RREA secondary transaction contains an unknown species");
            return 0;
        };
        // Payloads are already certified by the collective final_survivor scan
        // below, which uses a strictly stronger predicate.  A rank-local abort
        // here would deadlock the ReduceLongSum that scan performs.
        auto append_surviving_particle = [&](RreaSecondaryParticle const& particle) {
            switch (particle.species) {
            case RreaSecondarySpecies::Electron:
                spawn_x.push_back(particle.x_m); spawn_y.push_back(particle.y_m);
                spawn_z.push_back(particle.z_m); spawn_ux.push_back(particle.dir_x);
                spawn_uy.push_back(particle.dir_y); spawn_uz.push_back(particle.dir_z);
                spawn_weight.push_back(particle.weight);
                spawn_energy.push_back(particle.kinetic_or_photon_energy_eV);
                break;
            case RreaSecondarySpecies::Photon:
                photon_x.push_back(particle.x_m); photon_y.push_back(particle.y_m);
                photon_z.push_back(particle.z_m); photon_ux.push_back(particle.dir_x);
                photon_uy.push_back(particle.dir_y); photon_uz.push_back(particle.dir_z);
                photon_weight.push_back(particle.weight);
                photon_energy.push_back(particle.kinetic_or_photon_energy_eV);
                break;
            case RreaSecondarySpecies::Positron:
                positron_x.push_back(particle.x_m); positron_y.push_back(particle.y_m);
                positron_z.push_back(particle.z_m); positron_ux.push_back(particle.dir_x);
                positron_uy.push_back(particle.dir_y); positron_uz.push_back(particle.dir_z);
                positron_weight.push_back(particle.weight);
                positron_energy.push_back(particle.kinetic_or_photon_energy_eV);
                break;
            }
        };
        // Build and exhaust the deterministic FIFO before population preflight.
        // Bounded snapshots transport in parallel, then merge in cursor order so
        // group replay, descendant order, and RNG lineage remain unchanged.
        // Only final surviving payloads reach AddNParticles.
        std::vector<RreaSecondaryParticle> transient_work;
        std::size_t initial_transient_count = 0;
        for (auto const& group : pending_side_effects.groups) {
            for (auto const& particle : group.particles) {
                bool const known_species =
                    particle.species == RreaSecondarySpecies::Electron
                    || particle.species == RreaSecondarySpecies::Photon
                    || particle.species == RreaSecondarySpecies::Positron;
                bool const known_root_species =
                    particle.root_species == RreaSecondarySpecies::Electron
                    || particle.root_species == RreaSecondarySpecies::Photon
                    || particle.root_species == RreaSecondarySpecies::Positron;
                bool const finite_payload =
                    std::isfinite(static_cast<double>(particle.x_m))
                    && std::isfinite(static_cast<double>(particle.y_m))
                    && std::isfinite(static_cast<double>(particle.z_m))
                    && std::isfinite(static_cast<double>(particle.dir_x))
                    && std::isfinite(static_cast<double>(particle.dir_y))
                    && std::isfinite(static_cast<double>(particle.dir_z))
                    && std::isfinite(static_cast<double>(particle.weight))
                    && std::isfinite(static_cast<double>(
                        particle.kinetic_or_photon_energy_eV))
                    && std::isfinite(static_cast<double>(
                        particle.birth_elapsed_step_fraction));
                bool const valid_energy =
                    particle.species == RreaSecondarySpecies::Photon
                    ? particle.kinetic_or_photon_energy_eV > amrex::Real(0.0)
                    : particle.kinetic_or_photon_energy_eV >= amrex::Real(0.0);
                if (!known_species || !known_root_species || !finite_payload
                    || !(particle.weight > amrex::Real(0.0)) || !valid_energy
                    || particle.birth_elapsed_step_fraction < amrex::Real(0.0)
                    || particle.birth_elapsed_step_fraction > amrex::Real(1.0)
                    || particle.transient_lineage
                        >= kRreaTransientLineageCapacity) {
                    amrex::Abort(
                        "RREA transient FIFO received an invalid birth payload; "
                        "the uncommitted transaction is rejected");
                }
            }
            if (group.particles.size()
                > std::numeric_limits<std::size_t>::max()
                    - initial_transient_count) {
                amrex::Abort(
                    "RREA transient FIFO initial-size overflow; the "
                    "uncommitted transaction is rejected");
            }
            initial_transient_count += group.particles.size();
        }
        transient_work.reserve(initial_transient_count);
        for (auto const& group : pending_side_effects.groups) {
            transient_work.insert(
                transient_work.end(),
                group.particles.begin(), group.particles.end());
        }
        std::unordered_map<std::uint64_t, std::array<std::uint64_t, 3>>
            next_lineage_by_root;
        for (auto const& particle : transient_work) {
            auto& next = next_lineage_by_root[particle.root_rng_particle_id]
                [species_index(particle.root_species)];
            next = std::max(next, particle.transient_lineage + 1U);
        }
        auto allocate_lineage = [&](std::uint64_t key,
                                    RreaSecondarySpecies species) {
            auto& next = next_lineage_by_root[key][species_index(species)];
            if (next >= kRreaTransientLineageCapacity) {
                amrex::Abort(
                    "RREA transient FIFO lineage overflow; the uncommitted "
                    "transaction is rejected");
            }
            return next++;
        };
        std::vector<RreaSecondaryParticle> final_survivors;
        final_survivors.reserve(initial_transient_count);
        struct NewbornResult {
            TransportSideEffects effects;
            bool survives = true;
        };
        constexpr std::size_t newborn_batch_size = 1024;
        std::vector<NewbornResult> results;
#ifdef AMREX_USE_OMP
        int const transport_num_threads =
            (coupling.TransportOmpThreads() > 0)
                ? coupling.TransportOmpThreads()
                : omp_get_max_threads();
#endif
        for (std::size_t cursor = 0; cursor < transient_work.size();) {
            std::size_t const count = std::min(
                newborn_batch_size, transient_work.size() - cursor);
            results.assign(count, NewbornResult{
                TransportSideEffects{coupling.LowEnergyCutoffEv()}, true});
#ifdef AMREX_USE_OMP
            int const batch_num_threads =
                std::min(transport_num_threads, static_cast<int>(count));
#pragma omp parallel for schedule(dynamic, 1) num_threads(batch_num_threads) if(batch_num_threads > 1)
#endif
            for (long offset = 0; offset < static_cast<long>(count); ++offset) {
                auto const i = static_cast<std::size_t>(offset);
                auto& particle = transient_work[cursor + i];
                auto& result = results[i];
                if (!transient_energy_is_certified(particle)) {
                    amrex::Abort(
                        "RREA transient FIFO particle is outside its certified "
                        "transport-energy range; the uncommitted transaction is "
                        "rejected before any table query");
                }
                if (particle.birth_elapsed_step_fraction < amrex::Real(1.0)) {
                    result.survives =
                        particle.species == RreaSecondarySpecies::Photon
                            ? AdvancePhotonNewborn(particle, result.effects)
                            : AdvanceChargedNewborn(particle, result.effects);
                }
            }
            for (std::size_t i = 0; i < count; ++i) {
                RreaSecondaryParticle particle =
                    std::move(transient_work[cursor + i]);
                auto& result = results[i];
                for (auto& created_group : result.effects.groups) {
                    for (auto& child : created_group.particles) {
                        child.root_rng_particle_id = particle.root_rng_particle_id;
                        child.root_species = particle.root_species;
                        child.transient_lineage = allocate_lineage(
                            child.root_rng_particle_id, child.root_species);
                        if (transient_work.size()
                            >= static_cast<std::size_t>(
                                std::numeric_limits<amrex::Long>::max())) {
                            amrex::Abort(
                                "RREA transient FIFO size overflow; the "
                                "uncommitted transaction is rejected");
                        }
                        transient_work.push_back(child);
                    }
                }
                pending_side_effects.AppendGroupsFrom(result.effects);
                if (result.survives) {
                    particle.birth_elapsed_step_fraction = amrex::Real(1.0);
                    final_survivors.push_back(std::move(particle));
                }
            }
            cursor += count;
        }
        validate_complete_transaction();

        // Immutable group payloads retain birth-state products for the
        // conservation ledgers.  Validate the separate final-insertion list
        // at its transported endpoint before preflight mutates any container.
        amrex::Long final_survivor_errors = 0;
        for (auto const& survivor : final_survivors) {
            bool const known_species =
                survivor.species == RreaSecondarySpecies::Electron
                || survivor.species == RreaSecondarySpecies::Photon
                || survivor.species == RreaSecondarySpecies::Positron;
            bool const finite_payload =
                std::isfinite(static_cast<double>(survivor.x_m))
                && std::isfinite(static_cast<double>(survivor.y_m))
                && std::isfinite(static_cast<double>(survivor.z_m))
                && std::isfinite(static_cast<double>(survivor.dir_x))
                && std::isfinite(static_cast<double>(survivor.dir_y))
                && std::isfinite(static_cast<double>(survivor.dir_z))
                && std::isfinite(static_cast<double>(survivor.weight))
                && std::isfinite(static_cast<double>(
                    survivor.kinetic_or_photon_energy_eV));
            bool const valid_energy =
                survivor.species == RreaSecondarySpecies::Photon
                ? survivor.kinetic_or_photon_energy_eV > amrex::Real(0.0)
                : survivor.kinetic_or_photon_energy_eV
                    >= coupling.LowEnergyCutoffEv();
            if (!known_species || !finite_payload
                || !(survivor.weight > amrex::Real(0.0)) || !valid_energy
                || !transient_energy_is_certified(survivor)
                || !coupling.SecondaryGroupPositionInDomain(
                    0, std::hypot(survivor.x_m, survivor.y_m),
                    survivor.z_m)) {
                ++final_survivor_errors;
            }
        }
        amrex::ParallelDescriptor::ReduceLongSum(final_survivor_errors);
        if (final_survivor_errors != 0) {
            amrex::Abort(
                "RREA transient work graph contains invalid final survivors; "
                "no transient or native group was committed");
        }

        for (auto const& group : pending_side_effects.groups) {
            if (RreaSecondaryGroupRemovesNativeParent(group)) {
                if (group.parent_key == 0) {
                    amrex::Abort("RREA secondary transaction removes a parent without a key");
                }
                auto const index = species_index(group.parent_species);
                removed_parent_keys[index].push_back(group.parent_key);
                if (group.parent_still_valid_at_preflight) {
                    ++local_removals[index];
                }
            }
        }
        // One sort per species instead of a linear scan per removing group.
        for (auto& keys : removed_parent_keys) {
            std::sort(keys.begin(), keys.end());
            if (std::adjacent_find(keys.begin(), keys.end()) != keys.end()) {
                amrex::Abort("RREA secondary transaction removes one parent more than once");
            }
        }
        RreaPlaneFluxAccumulator::PreparedUpdate prepared_plane_crossings;
        bool const has_plane_crossings =
            !pending_side_effects.cd_electron_plane_crossings.empty();
        int plane_crossing_preview_error =
            has_plane_crossings
                && !coupling.PrepareCdElectronPlaneCrossings(
                    pending_side_effects.cd_electron_plane_crossings,
                    prepared_plane_crossings)
            ? 1
            : 0;
        amrex::ParallelDescriptor::ReduceIntMax(plane_crossing_preview_error);
        if (plane_crossing_preview_error != 0) {
            amrex::Abort(
                "RREA C&D plane crossing batch is invalid or would overflow; "
                "the complete interaction transaction is rejected before commit");
        }
        coupling.PreflightAndApplySecondaryPopulationControl(
            warpx, final_survivors, removed_parent_keys, local_removals,
            "RreaParticleInteraction::Apply",
            step);
        for (auto const& survivor : final_survivors) {
            append_surviving_particle(survivor);
        }
        // Parent IDs were only marked invalid while the transaction was being
        // assembled.  Physically remove/redistribute them only after the
        // collective overflow decision, before committing any correlated
        // secondary payloads or material/ledger sources.
        // Whether anything moved is a rank-local fact, but Redistribute is
        // collective: a rank that skips it never meets the ranks that entered it.
        // Reduce all three species in one call, before any of them branches, so a
        // species can never be reduced from inside another species' branch.
        auto& electron_container = warpx.GetPartContainer()
            .GetParticleContainerFromName(coupling.ElectronSpeciesName());
        auto& photon_container = warpx.GetPartContainer()
            .GetParticleContainerFromName(coupling.PhotonSpeciesName());
        auto& positron_container = warpx.GetPartContainer()
            .GetParticleContainerFromName(coupling.PositronSpeciesName());
        int repositioned_any[3] = {
            repositioned_electrons ? 1 : 0,
            repositioned_photons ? 1 : 0,
            repositioned_positrons ? 1 : 0};
        amrex::ParallelDescriptor::ReduceIntMax(repositioned_any, 3);
        // Redistribute also drops invalid particles.
        if (repositioned_any[0] != 0) {
            electron_container.Redistribute();
        } else if (invalidated_particles) {
            electron_container.deleteInvalidParticles();
        }
        if (repositioned_any[1] != 0) {
            photon_container.Redistribute();
        } else if (invalidated_photons) {
            photon_container.deleteInvalidParticles();
        }
        if (repositioned_any[2] != 0) {
            positron_container.Redistribute();
        } else if (invalidated_positrons) {
            positron_container.deleteInvalidParticles();
        }
        coupling.CreateLocalKineticElectrons(
            warpx,
            spawn_x,
            spawn_y,
            spawn_z,
            spawn_ux,
            spawn_uy,
            spawn_uz,
            spawn_weight,
            spawn_energy,
            step,
            /*population_ceiling_preflighted=*/true);
        coupling.CreateLocalPhotons(
            warpx,
            photon_x,
            photon_y,
            photon_z,
            photon_ux,
            photon_uy,
            photon_uz,
            photon_weight,
            photon_energy,
            step,
            /*population_ceiling_preflighted=*/true);
        coupling.CreateLocalKineticPositrons(
            warpx,
            positron_x,
            positron_y,
            positron_z,
            positron_ux,
            positron_uy,
            positron_uz,
            positron_weight,
            positron_energy,
            step,
            /*population_ceiling_preflighted=*/true);
        pending_side_effects.ReplayInto(coupling);
        if (has_plane_crossings) {
            coupling.CommitPreparedCdElectronPlaneCrossings(
                std::move(prepared_plane_crossings));
        }
        return;
}

}  // namespace rrea::warpx
