#ifndef RREA_SECONDARY_VALIDATION_H
#define RREA_SECONDARY_VALIDATION_H
#include "RreaKineticStep.H"
namespace rrea::warpx {
// Final insertion state is distinct from each immutable birth transaction.
RREA_HOST_DEVICE inline bool KineticFinalSurvivorValid(
    KineticStep const& context,RreaSecondaryParticle const& survivor) {

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
            >= context.coupling.LowEnergyCutoffEv();
    if (!known_species || !finite_payload
        || !(survivor.weight > amrex::Real(0.0)) || !valid_energy
        || !context.TransientEnergyIsCertified(survivor)
        || !RreaRzPositionInHalfOpenDomain(
            std::hypot(survivor.x_m, survivor.y_m),survivor.z_m,
            context.geom.ProbLo(0),context.geom.ProbHi(0),
            context.geom.ProbLo(1),context.geom.ProbHi(1))) {
        return false;
    }

    return true;
}
// The same complete group validator is used by CPU and CUDA. Only the global
// MPI decision remains on the host, before any material/ledger commit.
template<class Group>
RREA_HOST_DEVICE std::array<amrex::Long,5> ValidateKineticGroup(
    KineticStep const& context,Group const& group) {
    auto const electron_secondary_range=context.tables.ElectronCertifiedEnergyRangeEv();
    auto const positron_secondary_range=context.tables.PositronCertifiedEnergyRangeEv();
    auto position_in_domain=[&](int lev,amrex::Real r,amrex::Real z) {
        return lev==0 && RreaRzPositionInHalfOpenDomain(r,z,
            context.geom.ProbLo(0),context.geom.ProbHi(0),
            context.geom.ProbLo(1),context.geom.ProbHi(1));
    };
    std::array<amrex::Long,5> errors{0,0,0,0,0};
          auto photon_flow_matches = [](amrex::Real actual, amrex::Real expected,
                                        amrex::Real scale) {
            amrex::Real constexpr eps64 =
                amrex::Real(64.0) * std::numeric_limits<double>::epsilon();
            return std::isfinite(static_cast<double>(actual))
                && actual >= amrex::Real(0.0)
                && std::abs(actual - expected)
                    <= eps64 * rrea::kinetic::max(scale, amrex::Real(1.0));
          };
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
                        errors[4])) {
                errors[4] =
                    std::numeric_limits<amrex::Long>::max();
            } else {
                errors[4] +=
                    static_cast<amrex::Long>(
                        invalid_photon_charged_secondaries);
            }
            auto const conservation = RreaValidateSecondaryGroupConservation(
                group, context.coupling.LowEnergyCutoffEv());
            if (conservation.required
                && (!conservation.present || !conservation.finite)) {
                ++errors[2];
            } else if (conservation.required && !conservation.valid) {
                ++errors[3];
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
                            context.coupling.LowEnergyCutoffEv(),
                            context.coupling.PhotonCutoffEv());
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
                            context.coupling.LowEnergyCutoffEv());
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
                        amrex::Real const final_state_local_eV = rrea::kinetic::max(
                            op.a[1]
                                - amrex::Real(2.0)
                                    * RreaSpeciesRestEnergyEv(
                                        RreaSecondarySpecies::Electron)
                                - op.a[2] - op.a[3]
                                - (op.b0 ? op.a[4] : amrex::Real(0.0)),
                            amrex::Real(0.0));
                        expected_flow = RreaPairPhotonEnergyFlow(
                            op.a[2], op.a[3], op.a[4], final_state_local_eV,
                            context.coupling.LowEnergyCutoffEv(), op.b0);
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
                        ++errors[1];
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
                    ++errors[1];
                }
                ++material_source_index;
            }
            if (material_source_index != group.material_sources.size()) {
                ++errors[1];
            }
            for (auto const& source : group.material_sources) {
                bool const finite_payload =
                    std::isfinite(static_cast<double>(source.event_weight))
                    && std::isfinite(static_cast<double>(source.secondary_energy_eV))
                    && std::isfinite(static_cast<double>(source.energy_loss_eV));
                if (!finite_payload || !(source.event_weight > amrex::Real(0.0))
                    || source.secondary_energy_eV < amrex::Real(0.0)
                    || source.energy_loss_eV < amrex::Real(0.0)) {
                    ++errors[1];
                }
                if (!position_in_domain(
                        source.level, source.r_m, source.z_m)) {
                    ++errors[0];
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
                    ++errors[1];
                }
                amrex::Real const particle_r_m =
                    std::hypot(particle.x_m, particle.y_m);
                if (!position_in_domain(
                        0, particle_r_m, particle.z_m)) {
                    ++errors[0];
                }
            }
              return errors;
}
}
#endif
