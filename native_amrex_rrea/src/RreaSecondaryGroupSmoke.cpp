#include "rrea/RreaGpuSmoke.H"
#include "RreaCheckpointRankState.H"
#include "rrea/RreaSmokeRequire.H"
#include "RreaParticleInteraction.H"
#include "RreaRng.H"
#include "RreaManagedSpeciesContract.H"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

using rrea::smoke::require;

using rrea::warpx::RreaSecondaryGroup;
using rrea::warpx::RreaSecondarySpecies;

}  // namespace

int main(int argc, char** argv)
{
    rrea::smoke::GpuSession gpu_runtime;
    using namespace rrea::warpx;
    try {
        {
            auto const stopped = RreaLimitContinuousSegmentAtCutoff(
                amrex::Real(0.25),
                amrex::Real(0.75),
                amrex::Real(10.0),
                amrex::Real(2.0),
                [](amrex::Real end_fraction) {
                    return amrex::Real(32.0)
                        * (end_fraction - amrex::Real(0.25));
                });
            require(
                stopped.valid && stopped.reaches_cutoff
                    && std::abs(
                        stopped.effective_end_fraction - amrex::Real(0.5))
                        <= amrex::Real(1.0e-14),
                "continuous cutoff limiter did not locate the true stop");

            auto const survives = RreaLimitContinuousSegmentAtCutoff(
                amrex::Real(0.25),
                amrex::Real(0.75),
                amrex::Real(20.0),
                amrex::Real(2.0),
                [](amrex::Real end_fraction) {
                    return amrex::Real(4.0)
                        * (end_fraction - amrex::Real(0.25));
                });
            require(
                survives.valid && !survives.reaches_cutoff
                    && survives.effective_end_fraction == amrex::Real(0.75),
                "continuous cutoff limiter shortened a surviving segment");

            auto const already_at_cutoff =
                RreaLimitContinuousSegmentAtCutoff(
                    amrex::Real(0.25),
                    amrex::Real(0.75),
                    amrex::Real(2.0),
                    amrex::Real(2.0),
                    [](amrex::Real) { return amrex::Real(0.0); });
            require(
                already_at_cutoff.valid
                    && already_at_cutoff.reaches_cutoff
                    && already_at_cutoff.effective_end_fraction
                        == amrex::Real(0.25),
                "continuous cutoff limiter advanced a cutoff parent");
        }

        static_cast<void>(argv);
        if (argc != 1) {
            throw std::runtime_error("usage: rrea_secondary_group_smoke");
        }
        RreaParticleInteractionConfig subcycle_config;
        require(
            RreaTransportSubcycleControlsValid(subcycle_config, true),
            "default strict transport subcycle controls are invalid");
        subcycle_config.max_optical_depth_per_substep = amrex::Real(0.0);
        require(
            !RreaTransportSubcycleControlsValid(subcycle_config, true),
            "zero optical-depth target was accepted");
        subcycle_config.max_optical_depth_per_substep =
            std::numeric_limits<amrex::Real>::quiet_NaN();
        require(
            !RreaTransportSubcycleControlsValid(subcycle_config, true),
            "NaN optical-depth target was accepted");
        subcycle_config.max_optical_depth_per_substep = amrex::Real(0.05);
        subcycle_config.max_subcycles = 0U;
        require(
            !RreaTransportSubcycleControlsValid(subcycle_config, true),
            "zero transport subcycle ceiling was accepted");
        subcycle_config.max_subcycles = 256U;
        subcycle_config.strict_transport_guards = false;
        require(
            !RreaTransportSubcycleControlsValid(subcycle_config, true),
            "disabled production strict guards were accepted");
        require(
            RreaTransportSubcycleControlsValid(subcycle_config, false),
            "non-production control validation incorrectly required strict guards");
        RreaSecondaryGroup pair;
        pair.parent_key = 42;
        pair.parent_species = RreaSecondarySpecies::Photon;
        pair.process = RreaSecondaryProcess::PairTriplet;
        pair.remove_parent = true;
        pair.particles = {
            RreaSecondaryParticle{RreaSecondarySpecies::Electron},
            RreaSecondaryParticle{RreaSecondarySpecies::Positron},
            RreaSecondaryParticle{RreaSecondarySpecies::Electron}};
        pair.material_sources.push_back(RreaMaterialSource{
            RreaMaterialSourceKind::IonizationEvent});
        pair.ledger_entries.push_back(RreaLedgerEntry{
            RreaLedgerOpKind::RecordPairProduction});
        require(pair.ParticleCount(RreaSecondarySpecies::Electron) == 2,
            "triplet group lost a correlated electron");
        require(pair.ParticleCount(RreaSecondarySpecies::Positron) == 1,
            "triplet group lost its positron");
        require(pair.ParticleCount(RreaSecondarySpecies::Photon) == 0,
            "triplet group invented a photon");
        require(pair.material_sources.size() == 1 && pair.ledger_entries.size() == 1,
            "group did not retain material and ledger payloads");

        // Locally deposited energy makes N = w E / W_air ion pairs, and
        // booking W_air per created pair returns exactly the deposited energy.
        constexpr amrex::Real w_air_eV = amrex::Real(33.97);
        amrex::Real const pairs = RreaIonPairWeightFromLocalEnergy(
            amrex::Real(2.0), amrex::Real(3397.0), w_air_eV);
        require(std::abs(pairs - amrex::Real(200.0)) <= amrex::Real(1.0e-12),
            "local energy deposit is not w*E/W_air ion pairs");
        require(
            std::abs(pairs * w_air_eV - amrex::Real(6794.0))
                <= amrex::Real(1.0e-9),
            "ion-pair booking does not return the deposited energy");
        require(
            RreaIonPairWeightFromLocalEnergy(
                amrex::Real(2.0), amrex::Real(0.0), w_air_eV)
                == amrex::Real(0.0),
            "a zero local deposit created ion pairs");

        // Every physical interaction carries enough independent data for the
        // production preflight to reconstruct charge and total energy,
        // including lepton rest mass and fluid/ion sources.
        constexpr amrex::Real me_eV = amrex::Real(510998.95000);
        constexpr amrex::Real low_cut_eV = amrex::Real(1000.0);

        // The photon ledger partitions energy at the exact runtime cutoffs.
        // In particular, a 1 keV Compton photon can scatter
        // below the photon cutoff; both the sub-cutoff recoil and scattered
        // photon must then be local so the photon ledger closes at its lower
        // certified endpoint.
        auto const compton_endpoint_flow = RreaComptonPhotonEnergyFlow(
            amrex::Real(3.0),
            amrex::Real(997.0),
            low_cut_eV,
            low_cut_eV);
        require(compton_endpoint_flow.Valid(),
            "1 keV Compton endpoint flow is invalid");
        rrea::smoke::require_close_rel(
            compton_endpoint_flow.charged_kinetic_rest_mass_eV,
            amrex::Real(0.0),
            amrex::Real(0.0),
            "sub-cutoff Compton recoil was reported as charged transfer");
        rrea::smoke::require_close_rel(
            compton_endpoint_flow.local_recoil_binding_subcutoff_eV,
            amrex::Real(1000.0),
            amrex::Real(0.0),
            "sub-cutoff Compton scattered photon was omitted from local flow");
        rrea::smoke::require_close_rel(
            compton_endpoint_flow.TotalEv(),
            amrex::Real(1000.0),
            amrex::Real(0.0),
            "1 keV Compton endpoint photon ledger does not close");

        auto const compton_exact_cut_flow = RreaComptonPhotonEnergyFlow(
            low_cut_eV,
            low_cut_eV,
            low_cut_eV,
            low_cut_eV);
        rrea::smoke::require_close_rel(
            compton_exact_cut_flow.charged_kinetic_rest_mass_eV,
            low_cut_eV,
            amrex::Real(0.0),
            "exact-cutoff Compton recoil was demoted");
        rrea::smoke::require_close_rel(
            compton_exact_cut_flow.local_recoil_binding_subcutoff_eV,
            amrex::Real(0.0),
            amrex::Real(0.0),
            "exact-cutoff scattered photon was deposited locally");

        auto const photo_subcut_flow = RreaPhotoelectricPhotonEnergyFlow(
            amrex::Real(999.0),
            amrex::Real(1.0),
            low_cut_eV);
        rrea::smoke::require_close_rel(
            photo_subcut_flow.charged_kinetic_rest_mass_eV,
            amrex::Real(0.0),
            amrex::Real(0.0),
            "sub-cutoff photoelectron was reported as charged transfer");
        rrea::smoke::require_close_rel(
            photo_subcut_flow.local_recoil_binding_subcutoff_eV,
            amrex::Real(1000.0),
            amrex::Real(0.0),
            "photoelectron kinetic plus binding local flow is incomplete");

        auto const nuclear_pair_subcut_flow = RreaPairPhotonEnergyFlow(
            amrex::Real(999.0),
            low_cut_eV,
            amrex::Real(0.0),
            amrex::Real(50.0),
            low_cut_eV,
            false);
        rrea::smoke::require_close_rel(
            nuclear_pair_subcut_flow.charged_kinetic_rest_mass_eV,
            amrex::Real(2.0) * me_eV + low_cut_eV,
            amrex::Real(0.0),
            "nuclear-pair charged flow lost rest mass or exact-cutoff kinetic energy");
        rrea::smoke::require_close_rel(
            nuclear_pair_subcut_flow.local_recoil_binding_subcutoff_eV,
            amrex::Real(1049.0),
            amrex::Real(0.0),
            "nuclear-pair sub-cutoff kinetic energy was not local");
        rrea::smoke::require_close_rel(
            nuclear_pair_subcut_flow.TotalEv(),
            amrex::Real(2.0) * me_eV + amrex::Real(2049.0),
            amrex::Real(0.0),
            "nuclear-pair photon flow does not close");

        auto const triplet_subcut_flow = RreaPairPhotonEnergyFlow(
            low_cut_eV,
            amrex::Real(999.0),
            amrex::Real(500.0),
            amrex::Real(50.0),
            low_cut_eV,
            true);
        rrea::smoke::require_close_rel(
            triplet_subcut_flow.charged_kinetic_rest_mass_eV,
            amrex::Real(2.0) * me_eV + low_cut_eV,
            amrex::Real(0.0),
            "triplet charged flow lost created rest mass or kinetic energy");
        rrea::smoke::require_close_rel(
            triplet_subcut_flow.local_recoil_binding_subcutoff_eV,
            amrex::Real(1549.0),
            amrex::Real(0.0),
            "triplet recoil/positron sub-cutoff flow is incomplete");
        rrea::smoke::require_close_rel(
            triplet_subcut_flow.TotalEv(),
            amrex::Real(2.0) * me_eV + amrex::Real(2549.0),
            amrex::Real(0.0),
            "triplet photon flow does not close");

        RreaPhotonLedgerFlows compton_local_ledger;
        compton_local_ledger.external_injected_eV = amrex::Real(1000.0);
        compton_local_ledger.local_deposit_eV =
            compton_endpoint_flow.local_recoil_binding_subcutoff_eV;
        rrea::smoke::require_close_rel(
            compton_local_ledger.ResidualEv(),
            amrex::Real(0.0),
            amrex::Real(0.0),
            "synthetic 1 keV Compton-local ledger does not close");

        RreaPhotonLedgerFlows compton_survivor_ledger;
        compton_survivor_ledger.external_injected_eV = amrex::Real(1000.0);
        compton_survivor_ledger.alive_eV = amrex::Real(997.0);
        compton_survivor_ledger.charged_transfer_eV = amrex::Real(3.0);
        rrea::smoke::require_close_rel(
            compton_survivor_ledger.RelativeResidual(),
            amrex::Real(0.0),
            amrex::Real(0.0),
            "synthetic Compton-survivor ledger does not close");

        RreaPhotonLedgerFlows triplet_ledger;
        triplet_ledger.external_injected_eV = triplet_subcut_flow.TotalEv();
        triplet_ledger.charged_transfer_eV =
            triplet_subcut_flow.charged_kinetic_rest_mass_eV;
        triplet_ledger.local_deposit_eV =
            triplet_subcut_flow.local_recoil_binding_subcutoff_eV;
        triplet_ledger.local_deposit_eV += amrex::Real(1.0);
        require(triplet_ledger.ResidualEv() != amrex::Real(0.0),
            "photon ledger helper accepted a one-eV duplicate local sink");

        auto particle = [](
            RreaSecondarySpecies species,
            amrex::Real weight,
            amrex::Real energy_eV) {
            RreaSecondaryParticle value;
            value.species = species;
            value.dir_z = amrex::Real(1.0);
            value.weight = weight;
            value.kinetic_or_photon_energy_eV = energy_eV;
            return value;
        };
        auto ionization = [](
            amrex::Real weight,
            amrex::Real secondary_energy_eV) {
            RreaMaterialSource value;
            value.kind = RreaMaterialSourceKind::IonizationEvent;
            value.event_weight = weight;
            value.secondary_energy_eV = secondary_energy_eV;
            return value;
        };
        auto positive_ion = [](amrex::Real weight) {
            RreaMaterialSource value;
            value.kind = RreaMaterialSourceKind::PositiveIon;
            value.event_weight = weight;
            return value;
        };
        auto low_electron = [](amrex::Real weight) {
            RreaMaterialSource value;
            value.kind = RreaMaterialSourceKind::LowEnergyElectron;
            value.event_weight = weight;
            return value;
        };
        auto detailed = [](
            RreaSecondaryGroup& group,
            amrex::Real weight,
            amrex::Real before_eV,
            amrex::Real after_eV,
            amrex::Real local_eV = amrex::Real(0.0),
            amrex::Real escaped_eV = amrex::Real(0.0)) {
            group.conservation.mode = RreaConservationClosureMode::Detailed;
            group.conservation.parent_weight = weight;
            group.conservation.parent_energy_before_eV = before_eV;
            group.conservation.parent_energy_after_eV = after_eV;
            group.conservation.parent_survives = !group.remove_parent;
            group.conservation.local_recoil_binding_subcutoff_energy_eV =
                weight * local_eV;
            group.conservation.escaped_energy_eV = weight * escaped_eV;
        };
        auto require_conserves = [&](
            RreaSecondaryGroup const& group,
            char const* message) {
            auto const result = RreaValidateSecondaryGroupConservation(
                group, low_cut_eV);
            if (!result.required || !result.present || !result.finite
                || !result.valid) {
                throw std::runtime_error(message);
            }
        };

        amrex::Real const w = amrex::Real(2.0);
        RreaSecondaryGroup moller;
        moller.parent_species = RreaSecondarySpecies::Electron;
        moller.process = RreaSecondaryProcess::HardMoller;
        moller.particles.push_back(particle(
            RreaSecondarySpecies::Electron, w, amrex::Real(0.8e6)));
        moller.material_sources.push_back(ionization(w, amrex::Real(0.8e6)));
        detailed(moller, w, amrex::Real(2.0e6), amrex::Real(1.2e6));
        require_conserves(moller, "hard-Moller group did not conserve");

        RreaSecondaryGroup bhabha;
        bhabha.parent_species = RreaSecondarySpecies::Positron;
        bhabha.process = RreaSecondaryProcess::HardBhabha;
        bhabha.particles.push_back(particle(
            RreaSecondarySpecies::Electron, w, amrex::Real(0.8e6)));
        bhabha.material_sources.push_back(positive_ion(w));
        detailed(bhabha, w, amrex::Real(2.0e6), amrex::Real(1.2e6));
        require_conserves(bhabha, "hard-Bhabha group did not conserve");

        RreaSecondaryGroup electron_brems;
        electron_brems.parent_species = RreaSecondarySpecies::Electron;
        electron_brems.process = RreaSecondaryProcess::Bremsstrahlung;
        electron_brems.particles.push_back(particle(
            RreaSecondarySpecies::Photon, w, amrex::Real(0.6e6)));
        detailed(electron_brems, w, amrex::Real(2.0e6), amrex::Real(1.4e6));
        require_conserves(electron_brems, "electron brems group did not conserve");

        RreaSecondaryGroup positron_brems = electron_brems;
        positron_brems.parent_species = RreaSecondarySpecies::Positron;
        require_conserves(positron_brems, "positron brems group did not conserve");

        RreaSecondaryGroup compton;
        compton.parent_species = RreaSecondarySpecies::Photon;
        compton.process = RreaSecondaryProcess::Compton;
        compton.particles.push_back(particle(
            RreaSecondarySpecies::Electron, w, amrex::Real(0.3e6)));
        compton.material_sources.push_back(ionization(w, amrex::Real(0.3e6)));
        detailed(compton, w, amrex::Real(1.0e6), amrex::Real(0.7e6));
        require_conserves(compton, "Compton group did not conserve");

        RreaSecondaryGroup photoelectric;
        photoelectric.parent_species = RreaSecondarySpecies::Photon;
        photoelectric.process = RreaSecondaryProcess::Photoelectric;
        photoelectric.remove_parent = true;
        photoelectric.particles.push_back(particle(
            RreaSecondarySpecies::Electron, w, amrex::Real(90.0e3)));
        photoelectric.material_sources.push_back(
            ionization(w, amrex::Real(90.0e3)));
        detailed(photoelectric, w, amrex::Real(100.0e3), amrex::Real(0.0),
            amrex::Real(10.0e3));
        require_conserves(photoelectric, "photoelectric group did not conserve");

        RreaSecondaryGroup nuclear_pair;
        nuclear_pair.parent_species = RreaSecondarySpecies::Photon;
        nuclear_pair.process = RreaSecondaryProcess::PairNuclear;
        nuclear_pair.remove_parent = true;
        nuclear_pair.particles = {
            particle(RreaSecondarySpecies::Electron, w, amrex::Real(100.0e3)),
            particle(RreaSecondarySpecies::Positron, w, amrex::Real(150.0e3))};
        nuclear_pair.ledger_entries.push_back(
            RreaLedgerEntry{RreaLedgerOpKind::RecordPairProduction});
        detailed(nuclear_pair, w,
            amrex::Real(2.0) * me_eV + amrex::Real(300.0e3),
            amrex::Real(0.0), amrex::Real(50.0e3));
        require_conserves(nuclear_pair, "nuclear-pair group did not conserve");

        RreaSecondaryGroup triplet;
        triplet.parent_species = RreaSecondarySpecies::Photon;
        triplet.process = RreaSecondaryProcess::PairTriplet;
        triplet.remove_parent = true;
        triplet.particles = {
            particle(RreaSecondarySpecies::Electron, w, amrex::Real(100.0e3)),
            particle(RreaSecondarySpecies::Positron, w, amrex::Real(200.0e3)),
            particle(RreaSecondarySpecies::Electron, w, amrex::Real(250.0e3))};
        triplet.material_sources.push_back(ionization(w, amrex::Real(250.0e3)));
        RreaLedgerEntry triplet_source_op;
        triplet_source_op.kind = RreaLedgerOpKind::AccumulateIonizationEvent;
        triplet_source_op.a[2] = w;
        triplet_source_op.a[3] = amrex::Real(250.0e3);
        triplet.ledger_entries = {
            triplet_source_op,
            RreaLedgerEntry{RreaLedgerOpKind::RecordPairProduction}};
        detailed(triplet, w,
            amrex::Real(2.0) * me_eV + amrex::Real(600.0e3),
            amrex::Real(0.0), amrex::Real(50.0e3));
        require_conserves(triplet, "triplet-pair group did not conserve");

        RreaSecondaryGroup annihilation;
        annihilation.parent_species = RreaSecondarySpecies::Positron;
        annihilation.process = RreaSecondaryProcess::PositronAnnihilation;
        annihilation.remove_parent = true;
        annihilation.particles = {
            particle(RreaSecondarySpecies::Photon, w,
                me_eV + amrex::Real(100.0e3)),
            particle(RreaSecondarySpecies::Photon, w,
                me_eV + amrex::Real(100.0e3))};
        annihilation.material_sources.push_back(positive_ion(w));
        RreaLedgerEntry annihilation_source_op;
        annihilation_source_op.kind =
            RreaLedgerOpKind::AccumulatePositiveIonSourceOnly;
        annihilation_source_op.a[2] = w;
        annihilation.ledger_entries = {
            annihilation_source_op,
            RreaLedgerEntry{RreaLedgerOpKind::RecordPositronAnnihilation}};
        detailed(annihilation, w, amrex::Real(200.0e3), amrex::Real(0.0));
        require_conserves(annihilation, "positron-annihilation group did not conserve");

        RreaSecondaryGroup demotion;
        demotion.parent_species = RreaSecondarySpecies::Electron;
        demotion.process = RreaSecondaryProcess::LowEnergyDemotion;
        demotion.remove_parent = true;
        demotion.material_sources.push_back(low_electron(w));
        detailed(demotion, w, amrex::Real(500.0), amrex::Real(0.0),
            amrex::Real(500.0));
        require_conserves(demotion, "low-energy demotion group did not conserve");

        RreaSecondaryGroup zero_weight_cleanup;
        zero_weight_cleanup.process = RreaSecondaryProcess::LowEnergyDemotion;
        zero_weight_cleanup.remove_parent = true;
        zero_weight_cleanup.conservation.mode =
            RreaConservationClosureMode::ExplicitZero;
        require_conserves(
            zero_weight_cleanup,
            "explicit zero-weight cleanup closure was rejected");

        RreaSecondaryGroup electron_escape;
        electron_escape.parent_species = RreaSecondarySpecies::Electron;
        electron_escape.process = RreaSecondaryProcess::ParticleEscape;
        electron_escape.remove_parent = true;
        electron_escape.parent_still_valid_at_preflight = true;
        detailed(
            electron_escape,
            w,
            amrex::Real(750.0e3),
            amrex::Real(0.0),
            amrex::Real(0.0),
            amrex::Real(750.0e3) + me_eV);
        electron_escape.conservation.escaped_charge_e = -w;
        require_conserves(
            electron_escape,
            "electron boundary-flux closure did not conserve");

        RreaSecondaryGroup positron_escape = electron_escape;
        positron_escape.parent_species = RreaSecondarySpecies::Positron;
        positron_escape.conservation.escaped_charge_e = w;
        require_conserves(
            positron_escape,
            "positron boundary-flux closure did not conserve");

        RreaSecondaryGroup zero_escape_bypass = electron_escape;
        zero_escape_bypass.conservation = RreaInteractionConservationClosure{};
        zero_escape_bypass.conservation.mode =
            RreaConservationClosureMode::ExplicitZero;
        require(!RreaValidateSecondaryGroupConservation(
                    zero_escape_bypass, low_cut_eV).valid,
            "still-valid charged escape incorrectly accepted an explicit-zero bypass");

        RreaSecondaryGroup bad_escaped_charge = electron_escape;
        bad_escaped_charge.conservation.escaped_charge_e += amrex::Real(1.0);
        require(!RreaValidateSecondaryGroupConservation(
                    bad_escaped_charge, low_cut_eV).valid,
            "bad escaped charge was not rejected");
        RreaSecondaryGroup bad_escaped_energy = positron_escape;
        bad_escaped_energy.conservation.escaped_energy_eV += amrex::Real(1.0);
        require(!RreaValidateSecondaryGroupConservation(
                    bad_escaped_energy, low_cut_eV).valid,
            "bad escaped total energy was not rejected");

        RreaSecondaryGroup bad_residual = compton;
        bad_residual.conservation.local_recoil_binding_subcutoff_energy_eV =
            amrex::Real(1.0);
        auto const bad_result = RreaValidateSecondaryGroupConservation(
            bad_residual, low_cut_eV);
        require(bad_result.present && bad_result.finite && !bad_result.valid,
            "deliberately bad interaction residual was not rejected");
        RreaSecondaryGroup missing = compton;
        missing.conservation = RreaInteractionConservationClosure{};
        require(!RreaValidateSecondaryGroupConservation(missing, low_cut_eV).valid,
            "missing interaction closure was not rejected");
        RreaSecondaryGroup nonfinite = compton;
        nonfinite.conservation.parent_energy_before_eV =
            std::numeric_limits<amrex::Real>::quiet_NaN();
        require(!RreaValidateSecondaryGroupConservation(nonfinite, low_cut_eV).valid,
            "non-finite interaction closure was not rejected");

        RreaMaterialSource source{
            RreaMaterialSourceKind::IonizationEvent,
            0,
            amrex::Real(0.25),
            amrex::Real(0.75),
            amrex::Real(3.0),
            amrex::Real(1200.0),
            amrex::Real(37.0)};
        RreaLedgerEntry source_op;
        source_op.kind = RreaLedgerOpKind::AccumulateIonizationEvent;
        source_op.lev = 0;
        source_op.a = {
            source.r_m, source.z_m, source.event_weight,
            source.secondary_energy_eV, source.energy_loss_eV,
            amrex::Real(0.0), amrex::Real(0.0)};
        require(RreaMaterialSourceMatchesLedgerEntry(source, source_op),
            "material source lost its exact replay operation");
        source_op.a[1] = amrex::Real(0.5);
        require(!RreaMaterialSourceMatchesLedgerEntry(source, source_op),
            "material source/replay position mismatch was accepted");

        // Exact half-open endpoint contract for both RZ axes.  These are the
        // same predicates used by the collective transaction preflight.
        amrex::Real const r_lo = amrex::Real(0.0);
        amrex::Real const r_hi = amrex::Real(1.0);
        amrex::Real const z_lo = amrex::Real(-2.0);
        amrex::Real const z_hi = amrex::Real(2.0);
        require(RreaRzPositionInHalfOpenDomain(
                    r_lo, z_lo, r_lo, r_hi, z_lo, z_hi),
            "included lower RZ faces were rejected");
        require(!RreaRzPositionInHalfOpenDomain(
                    r_hi, amrex::Real(0.0), r_lo, r_hi, z_lo, z_hi),
            "excluded radial upper face was accepted");
        require(!RreaRzPositionInHalfOpenDomain(
                    amrex::Real(0.5), z_hi, r_lo, r_hi, z_lo, z_hi),
            "excluded axial upper face was accepted");
        require(RreaRzPositionInHalfOpenDomain(
                    std::nextafter(r_hi, r_lo),
                    std::nextafter(z_hi, z_lo),
                    r_lo, r_hi, z_lo, z_hi),
            "last representable in-domain RZ point was rejected");
        require(!RreaCartesianPositionInHalfOpenRzDomain(
                    r_hi, amrex::Real(0.0), amrex::Real(0.0),
                    r_lo, r_hi, z_lo, z_hi),
            "Cartesian secondary on the radial upper face was accepted");
        require(!RreaCartesianPositionInHalfOpenRzDomain(
                    amrex::Real(0.0), amrex::Real(0.0),
                    std::numeric_limits<amrex::Real>::quiet_NaN(),
                    r_lo, r_hi, z_lo, z_hi),
            "non-finite secondary position was accepted");

        require(RreaManagedSpeciesNamesAreDistinct(
                    "rrea_electrons", "rrea_photons", "rrea_positrons"),
            "distinct managed species names were rejected");
        require(!RreaManagedSpeciesNamesAreDistinct(
                    "rrea_electrons", "rrea_electrons", "rrea_positrons")
                && !RreaManagedSpeciesNamesAreDistinct(
                    "", "rrea_photons", "rrea_positrons"),
            "aliased or empty managed species name was accepted");
        require(RreaManagedSpeciesIdentityMatches(
                    "electron", amrex::Real(2.0), amrex::Real(-3.0),
                    "electron", amrex::Real(2.0), amrex::Real(-3.0)),
            "exact managed species identity was rejected");
        require(!RreaManagedSpeciesIdentityMatches(
                    "positron", amrex::Real(2.0), amrex::Real(-3.0),
                    "electron", amrex::Real(2.0), amrex::Real(-3.0))
                && !RreaManagedSpeciesIdentityMatches(
                    "electron", amrex::Real(2.1), amrex::Real(-3.0),
                    "electron", amrex::Real(2.0), amrex::Real(-3.0))
                && !RreaManagedSpeciesIdentityMatches(
                    "electron", amrex::Real(2.0), amrex::Real(3.0),
                    "electron", amrex::Real(2.0), amrex::Real(-3.0))
                && !RreaManagedSpeciesIdentityMatches(
                    "electron",
                    std::numeric_limits<amrex::Real>::quiet_NaN(),
                    amrex::Real(-3.0),
                    "electron", amrex::Real(2.0), amrex::Real(-3.0)),
            "wrong type, mass, charge, or non-finite identity was accepted");

        require(RreaResolveChargedTerminalDisposition(
                    false, true, true, true)
                    == RreaChargedTerminalDisposition::Escape,
            "electron boundary escape lost precedence over demotion");
        require(RreaResolveChargedTerminalDisposition(
                    true, true, true, false)
                    == RreaChargedTerminalDisposition::Escape,
            "positron boundary escape lost precedence over at-rest annihilation");
        require(RreaResolveChargedTerminalDisposition(
                    false, false, false, true)
                    == RreaChargedTerminalDisposition::ElectronDemotion,
            "interior direction-conditional electron demotion was lost");
        require(RreaResolveChargedTerminalDisposition(
                    true, false, true, false)
                    == RreaChargedTerminalDisposition::PositronAnnihilation,
            "interior sub-cutoff positron annihilation was lost");
        require(RreaResolveChargedTerminalDisposition(
                    false, false, false, false)
                    == RreaChargedTerminalDisposition::Survive,
            "interior above-cutoff electron did not survive");

        std::uint64_t projected = 0;
        require(RreaStrictPopulationProjection(9, 0, 0, 10, projected)
                && projected == 9,
            "cap-1 population should be accepted unchanged");
        require(RreaStrictPopulationProjection(10, 0, 0, 10, projected)
                && projected == 10,
            "cap equality should be accepted unchanged");
        require(!RreaStrictPopulationProjection(
                    10, 0, 0, 10, projected, /*reject_equality=*/true)
                && projected == 10,
            "C&D cap equality must be classified as a ceiling hit");
        require(RreaStrictPopulationProjection(10, 1, 1, 10, projected)
                && projected == 10,
            "same-species remove/create replacement should be accepted at cap");
        require(!RreaStrictPopulationProjection(10, 0, 1, 10, projected)
                && projected == 11,
            "cap+1 must be rejected");
        require(!RreaStrictPopulationProjection(10, 1, 2, 10, projected)
                && projected == 11,
            "net cap+1 correlated group must be rejected");
        require(!RreaStrictPopulationProjection(1, 2, 0, 10, projected),
            "removing more parents than exist must be rejected");
        require(!RreaStrictPopulationProjection(
                    std::numeric_limits<std::uint64_t>::max(),
                    0, 1,
                    std::numeric_limits<std::uint64_t>::max(),
                    projected)
                && projected == std::numeric_limits<std::uint64_t>::max(),
            "uint64 population overflow must saturate and reject, never wrap");

        amrex::Real const hard_cut_eV = amrex::Real(1.0e5);
        amrex::Real const moller_threshold_eV = amrex::Real(2.0) * hard_cut_eV;
        require(!RreaHardMollerChannelOpen(moller_threshold_eV, hard_cut_eV),
            "Moller channel must be closed at its exact zero-rate threshold");
        require(RreaHardMollerChannelOpen(
                    std::nextafter(
                        moller_threshold_eV,
                        std::numeric_limits<amrex::Real>::infinity()),
                    hard_cut_eV),
            "Moller channel must open immediately above its exact threshold");
        require(!RreaHardBhabhaChannelOpen(hard_cut_eV, hard_cut_eV),
            "Bhabha channel must be closed at its exact zero-rate threshold");
        require(RreaHardBhabhaChannelOpen(
                    std::nextafter(
                        hard_cut_eV,
                        std::numeric_limits<amrex::Real>::infinity()),
                    hard_cut_eV),
            "Bhabha channel must open immediately above threshold without a cutoff band");

        // Charged discrete processes compete for one causal event in each
        // existing subcycle.  The occurrence draw is exponential against the
        // summed hazard and the selector uses exact channel shares.
        amrex::Real const event_draw_tau_0p1 =
            amrex::Real(1.0) - std::exp(amrex::Real(-0.1));
        auto const electron_hard_wins = RreaSelectCompetingHazard(
            {amrex::Real(0.2), amrex::Real(0.3), amrex::Real(0.0)},
            2,
            event_draw_tau_0p1,
            amrex::Real(0.1));
        auto const electron_brems_wins = RreaSelectCompetingHazard(
            {amrex::Real(0.2), amrex::Real(0.3), amrex::Real(0.0)},
            2,
            event_draw_tau_0p1,
            amrex::Real(0.8));
        require(electron_hard_wins.valid && electron_hard_wins.event
                && electron_hard_wins.channel == 0
                && electron_brems_wins.valid && electron_brems_wins.event
                && electron_brems_wins.channel == 1,
            "electron competing hazards did not select one channel by total-rate share");
        auto const positron_annihilation_wins = RreaSelectCompetingHazard(
            {amrex::Real(0.1), amrex::Real(0.2), amrex::Real(0.7)},
            3,
            event_draw_tau_0p1,
            amrex::Real(0.5));
        require(positron_annihilation_wins.valid
                && positron_annihilation_wins.event
                && positron_annihilation_wins.channel == 2,
            "positron three-channel competing hazard selected the wrong winner");
        auto const no_competing_event = RreaSelectCompetingHazard(
            {amrex::Real(0.2), amrex::Real(0.3), amrex::Real(0.0)},
            2,
            amrex::Real(1.0) - std::exp(amrex::Real(-0.6)),
            amrex::Real(0.0));
        require(no_competing_event.valid && !no_competing_event.event
                && no_competing_event.channel == -1,
            "competing hazard created an event beyond the segment optical depth");
        require(!RreaSelectCompetingHazard(
                    {amrex::Real(0.1), amrex::Real(-0.1), amrex::Real(0.0)},
                    2, event_draw_tau_0p1, amrex::Real(0.0)).valid,
            "negative competing hazard was accepted");
        // A one-event cap at the configured tau maximum suppresses the Poisson
        // mean by ~tau/2.  Two successive remaining-segment draws must
        // therefore be representable and independently select events.  tau_max
        // reads the production default so a changed per-substep optical-depth
        // target moves the pinned bias band and fails here.
        {
            amrex::Real const tau_max =
                RreaParticleInteractionConfig{}.max_optical_depth_per_substep;
            amrex::Real const capped_mean =
                amrex::Real(1.0) - std::exp(-tau_max);
            amrex::Real const relative_undercount =
                amrex::Real(1.0) - capped_mean / tau_max;
            require(relative_undercount > amrex::Real(0.0245)
                    && relative_undercount < amrex::Real(0.0247),
                "one-event hazard-cap bias changed unexpectedly");
            auto const first = RreaSelectCompetingHazard(
                {amrex::Real(0.03), amrex::Real(0.02), amrex::Real(0.0)},
                2,
                amrex::Real(1.0) - std::exp(amrex::Real(-0.01)),
                amrex::Real(0.1));
            auto const continuation = RreaSelectCompetingHazard(
                {amrex::Real(0.024), amrex::Real(0.016), amrex::Real(0.0)},
                2,
                amrex::Real(1.0) - std::exp(amrex::Real(-0.01)),
                amrex::Real(0.9));
            require(first.valid && first.event && first.channel == 0
                    && continuation.valid && continuation.event
                    && continuation.channel == 1,
                "remaining-segment hazard loop cannot represent two events");
        }
        // Newborn residual-advance streams are append-only
        // additions; pinning them keeps every earlier id stable.
        require(static_cast<std::uint64_t>(RreaRngChannel::AdaptiveResampleRoulette) == 111U
                && static_cast<std::uint64_t>(RreaRngChannel::NewbornElectronGsAngle) == 112U
                && static_cast<std::uint64_t>(RreaRngChannel::NewbornElectronGsFewAzimuth) == 116U
                && static_cast<std::uint64_t>(RreaRngChannel::NewbornPositronGsAngle) == 117U
                && static_cast<std::uint64_t>(RreaRngChannel::NewbornPositronGsFewAzimuth) == 121U
                && static_cast<std::uint64_t>(RreaRngChannel::NewbornCompetingOpticalDepth) == 122U
                && static_cast<std::uint64_t>(RreaRngChannel::NewbornCompetingChannel) == 123U
                && static_cast<std::uint64_t>(RreaRngChannel::ElectronContinuationOpticalDepth) == 124U
                && static_cast<std::uint64_t>(RreaRngChannel::ElectronContinuationChannel) == 125U
                && static_cast<std::uint64_t>(RreaRngChannel::PositronContinuationOpticalDepth) == 126U
                && static_cast<std::uint64_t>(RreaRngChannel::PositronContinuationChannel) == 127U
                && static_cast<std::uint64_t>(RreaRngChannel::PhotonContinuationOpticalDepth) == 128U
                && static_cast<std::uint64_t>(RreaRngChannel::PhotonContinuationChannel) == 129U
                && static_cast<std::uint64_t>(RreaRngChannel::PositronBhabhaAzimuth) == 130U
                && static_cast<std::uint64_t>(RreaRngChannel::AdaptiveResampleCicElectron) == 131U
                && static_cast<std::uint64_t>(RreaRngChannel::AdaptiveResampleCicPositron) == 132U,
            "append-only newborn residual-advance RNG ids changed a reserved stream");
        // Repeated hazards and the transient FIFO share one checked 52-bit
        // layout.  Pin the exact all-maximum value, field separation, root
        // species separation, newborn marker, and every overflow boundary.
        {
            auto const packed_max = RreaTryPackTransientRngIndex(
                true, RreaSecondarySpecies::Positron,
                16383U, 16383U, 8191U, 255U);
            require(packed_max.valid
                    && packed_max.value < (std::uint64_t{1} << 52),
                "transient RNG index exceeds its 52-bit budget");
            auto const live_root = RreaTryPackTransientRngIndex(
                false, RreaSecondarySpecies::Electron,
                kRreaLiveContinuationLineage, 3U, 4U, 5U);
            auto const first_live_hazard = RreaTryLiveHazardRngIndex(
                RreaSecondarySpecies::Electron, 8191U, 0U);
            auto const packed_second = RreaTryLiveHazardRngIndex(
                RreaSecondarySpecies::Electron, 0U, 1U);
            auto const newborn = RreaTryPackTransientRngIndex(
                true, RreaSecondarySpecies::Electron, 2U, 3U, 4U, 5U);
            auto const other_species = RreaTryPackTransientRngIndex(
                true, RreaSecondarySpecies::Photon, 2U, 3U, 4U, 5U);
            auto const other_lineage = RreaTryPackTransientRngIndex(
                true, RreaSecondarySpecies::Electron, 3U, 3U, 4U, 5U);
            auto const other_segment = RreaTryPackTransientRngIndex(
                true, RreaSecondarySpecies::Electron, 2U, 4U, 4U, 5U);
            auto const other_event = RreaTryPackTransientRngIndex(
                true, RreaSecondarySpecies::Electron, 2U, 3U, 5U, 5U);
            auto const other_inner = RreaTryPackTransientRngIndex(
                true, RreaSecondarySpecies::Electron, 2U, 3U, 4U, 6U);
            require(live_root.valid && first_live_hazard.valid
                    && first_live_hazard.value == 8191U
                    && packed_second.valid
                    && packed_second.value >= kRreaTransientSegmentCapacity
                    && newborn.valid && other_species.valid
                    && other_lineage.valid && other_segment.valid
                    && other_event.valid && other_inner.valid
                    && live_root.value != newborn.value
                    && newborn.value != other_species.value
                    && newborn.value != other_lineage.value
                    && newborn.value != other_segment.value
                    && newborn.value != other_event.value
                    && newborn.value != other_inner.value,
                "transient RNG index fields are not collision-free");
            require(live_root.value >= kRreaTransientSegmentCapacity,
                "live electron continuation can alias another packed subcycle index");
            require(!RreaTryPackTransientRngIndex(
                        true, RreaSecondarySpecies::Electron,
                        16384U, 0U, 0U, 0U).valid
                    && !RreaTryPackTransientRngIndex(
                        true, RreaSecondarySpecies::Electron,
                        0U, 16384U, 0U, 0U).valid
                    && !RreaTryPackTransientRngIndex(
                        true, RreaSecondarySpecies::Electron,
                        0U, 0U, 8192U, 0U).valid
                    && !RreaTryPackTransientRngIndex(
                        true, RreaSecondarySpecies::Electron,
                        0U, 0U, 0U, 256U).valid
                    && !RreaTryPackTransientRngIndex(
                        true, static_cast<RreaSecondarySpecies>(3),
                        0U, 0U, 0U, 0U).valid
                    && !RreaTryLiveHazardRngIndex(
                        static_cast<RreaSecondarySpecies>(3),
                        0U, 0U).valid,
                "transient RNG index accepted an overflow or unknown species");
        }
        // Aggregate initialization defaults newborns to no residual transport.
        {
            RreaSecondaryParticle const defaulted_particle{
                RreaSecondarySpecies::Electron,
                amrex::Real(1.0), amrex::Real(0.0), amrex::Real(2.0),
                amrex::Real(0.0), amrex::Real(0.0), amrex::Real(1.0),
                amrex::Real(3.0), amrex::Real(5.0e5)};
            require(defaulted_particle.birth_elapsed_step_fraction == amrex::Real(1.0)
                    && defaulted_particle.root_rng_particle_id == 0U
                    && defaulted_particle.transient_lineage == 0U,
                "aggregate RreaSecondaryParticle no longer defaults "
                "to the no-residual birth fraction");
            RreaSecondaryGroup transient_terminal;
            transient_terminal.remove_parent = true;
            transient_terminal.parent_is_transient = true;
            RreaSecondaryGroup native_terminal;
            native_terminal.remove_parent = true;
            require(!RreaSecondaryGroupRemovesNativeParent(
                        transient_terminal)
                    && RreaSecondaryGroupRemovesNativeParent(native_terminal),
                "transient terminal disposition changed native removal counts");
            require(RreaLowEnergyNonrunawayFieldMagnitudeGate(
                        amrex::Real(5.0e4), amrex::Real(1.0e5),
                        amrex::Real(6.0e4), amrex::Real(7.0e4))
                    && !RreaLowEnergyNonrunawayFieldMagnitudeGate(
                        amrex::Real(5.0e4), amrex::Real(1.0e5),
                        amrex::Real(8.0e4), amrex::Real(7.0e4))
                    && !RreaLowEnergyNonrunawayFieldMagnitudeGate(
                        amrex::Real(2.0e5), amrex::Real(1.0e5),
                        amrex::Real(6.0e4), amrex::Real(7.0e4))
                    && !RreaLowEnergyNonrunawayFieldMagnitudeGate(
                        amrex::Real(5.0e4), amrex::Real(0.0),
                        amrex::Real(6.0e4), amrex::Real(7.0e4)),
                "field-magnitude demotion gate misclassified a state");
            struct PolicyCase {
                amrex::Real energy;
                amrex::Real field;
                bool demote;
            };
            for (PolicyCase const c : {
                     PolicyCase{amrex::Real(9.0e2), amrex::Real(1.0e6), true},
                     PolicyCase{amrex::Real(5.0e4), amrex::Real(6.0e4), true},
                     PolicyCase{amrex::Real(5.0e4), amrex::Real(8.0e4), false},
                     PolicyCase{amrex::Real(2.0e5), amrex::Real(6.0e4), false}}) {
                auto const live = RreaEvaluateElectronStatePolicy(
                    c.energy, amrex::Real(1.0e3), amrex::Real(1.0e5),
                    c.field, amrex::Real(7.0e4));
                auto const newborn = RreaEvaluateElectronStatePolicy(
                    c.energy, amrex::Real(1.0e3), amrex::Real(1.0e5),
                    c.field, amrex::Real(7.0e4));
                require(live.Demote() == c.demote
                        && newborn.Demote() == live.Demote(),
                    "identical live/newborn electron states diverged");
            }
            require(RreaLocalRunawayFieldMagnitudeGate(
                        amrex::Real(3.0), amrex::Real(4.0), amrex::Real(4.9))
                    && RreaLocalRunawayFieldMagnitudeGate(
                        amrex::Real(-3.0), amrex::Real(-4.0), amrex::Real(4.9))
                    && !RreaLocalRunawayFieldMagnitudeGate(
                        amrex::Real(3.0), amrex::Real(4.0), amrex::Real(5.0))
                    && !RreaLocalRunawayFieldMagnitudeGate(
                        amrex::Real(3.0), amrex::Real(4.0), amrex::Real(-1.0)),
                "plane-flux local-runaway selection is not direction-free");
        }
        auto const event_segments = RreaSplitChargedSegmentAtEvent(
            amrex::Real(0.2), amrex::Real(0.35), amrex::Real(0.8));
        require(event_segments.valid
                && event_segments.before_begin_fraction == amrex::Real(0.2)
                && event_segments.before_end_fraction == amrex::Real(0.35)
                && event_segments.after_begin_fraction == amrex::Real(0.35)
                && event_segments.after_end_fraction == amrex::Real(0.8),
            "charged event-depth split introduced a gap, overlap, or moved endpoint");
        require(!RreaSplitChargedSegmentAtEvent(
                    amrex::Real(0.2), amrex::Real(0.1), amrex::Real(0.8)).valid
                && !RreaSplitChargedSegmentAtEvent(
                    amrex::Real(0.2), amrex::Real(0.9), amrex::Real(0.8)).valid
                && !RreaSplitChargedSegmentAtEvent(
                    amrex::Real(0.2),
                    std::numeric_limits<amrex::Real>::quiet_NaN(),
                    amrex::Real(0.8)).valid,
            "charged event-depth split accepted an invalid sampled depth");

        auto const valid_rank_state_layout =
            RreaValidateRankStatePayloadLayout(2U, 43U, 17U, 8U);
        require(valid_rank_state_layout.valid
                && valid_rank_state_layout.integer_count == 86U
                && valid_rank_state_layout.real_count == 34U
                && valid_rank_state_layout.integer_bytes == 86U * 8U
                && valid_rank_state_layout.real_bytes == 34U * 8U,
            "valid checkpoint rank-state layout was rejected");
        require(!RreaValidateRankStatePayloadLayout(0U, 43U, 17U, 8U).valid
                && !RreaValidateRankStatePayloadLayout(1U, 0U, 17U, 8U).valid
                && !RreaValidateRankStatePayloadLayout(1U, 43U, 0U, 8U).valid,
            "zero checkpoint rank-state dimension was accepted");
        require(!RreaValidateRankStatePayloadLayout(
                    std::numeric_limits<std::uint64_t>::max(), 2U, 2U, 8U).valid
                && !RreaValidateRankStatePayloadLayout(
                    2U,
                    static_cast<std::uint64_t>(std::numeric_limits<int>::max()),
                    1U,
                    8U).valid,
            "overflowing checkpoint rank-state dimensions were accepted");
        require(!RreaValidateRankStatePayloadLayout(1U, 1U, 1U, 16U).valid,
            "unsupported checkpoint Real size was accepted");

        // A certified 100 GeV photon can transfer more than the charged 10 GeV
        // table maximum to one child.  Such a tracked child must reject the
        // entire deferred group before any source/particle/ledger mutation.
        amrex::Real const charged_min_eV = amrex::Real(1.0e3);
        amrex::Real const charged_max_eV = amrex::Real(1.0e10);
        require(RreaClassifyTrackedChargedSecondaryEnergy(
                    charged_min_eV, charged_min_eV, charged_max_eV)
                    == RreaTrackedChargedSecondaryDisposition::Track
                && RreaClassifyTrackedChargedSecondaryEnergy(
                    charged_max_eV, charged_min_eV, charged_max_eV)
                    == RreaTrackedChargedSecondaryDisposition::Track,
            "exact charged certified endpoints were rejected");
        require(RreaClassifyTrackedChargedSecondaryEnergy(
                    std::nextafter(charged_min_eV, amrex::Real(0.0)),
                    charged_min_eV, charged_max_eV)
                    == RreaTrackedChargedSecondaryDisposition::DemoteToFluid,
            "sub-1keV photon electron was not assigned to fluid demotion");
        require(RreaClassifyTrackedChargedSecondaryEnergy(
                    std::nextafter(
                        charged_max_eV,
                        std::numeric_limits<amrex::Real>::infinity()),
                    charged_min_eV, charged_max_eV)
                    == RreaTrackedChargedSecondaryDisposition::Reject,
            "above-10GeV charged photon product was accepted");
        RreaSecondaryGroup photon_pair_tip;
        photon_pair_tip.process = RreaSecondaryProcess::PairNuclear;
        photon_pair_tip.parent_species = RreaSecondarySpecies::Photon;
        photon_pair_tip.particles.push_back(RreaSecondaryParticle{
            RreaSecondarySpecies::Electron,
            amrex::Real(0.0), amrex::Real(0.0), amrex::Real(0.0),
            amrex::Real(0.0), amrex::Real(0.0), amrex::Real(1.0),
            amrex::Real(1.0), amrex::Real(9.0e10)});
        require(RreaCountInvalidTrackedPhotonChargedSecondaries(
                    photon_pair_tip,
                    charged_min_eV, charged_max_eV,
                    charged_min_eV, charged_max_eV) == 1U,
            "100GeV-photon charged-child guard did not reject a 90GeV electron");
        photon_pair_tip.particles.front().kinetic_or_photon_energy_eV =
            amrex::Real(999.0);
        require(RreaCountInvalidTrackedPhotonChargedSecondaries(
                    photon_pair_tip,
                    charged_min_eV, charged_max_eV,
                    charged_min_eV, charged_max_eV) == 1U,
            "tracked sub-1keV photon electron bypassed the fluid-demotion guard");
        photon_pair_tip.particles.clear();
        require(RreaCountInvalidTrackedPhotonChargedSecondaries(
                    photon_pair_tip,
                    charged_min_eV, charged_max_eV,
                    charged_min_eV, charged_max_eV) == 0U,
            "fluid-demoted photon electron was still treated as tracked");

        // Absorbing-boundary field work uses only the clipped chord.  The
        // reconstruction starts from captured pre-push energy and uses the
        // partial pushed vector as a direction hint, so its magnitude cannot
        // inherit an out-of-domain full-step kick.
        auto const clipped_path = rrea::ClipRzTransportPath(
            amrex::Real(0.5), amrex::Real(0.0), amrex::Real(0.5),
            amrex::Real(0.5), amrex::Real(0.0), amrex::Real(1.5),
            amrex::Real(0.0), amrex::Real(1.0),
            amrex::Real(0.0), amrex::Real(1.0));
        require(clipped_path.exits_domain, "field-work smoke chord did not exit");
        rrea::smoke::require_close_rel(
            clipped_path.in_domain_length_m,
            amrex::Real(0.5),
            amrex::Real(1.0e-14),
            "field-work smoke clipped the wrong length");
        auto const radial_clipped_path = rrea::ClipRzTransportPath(
            amrex::Real(0.25), amrex::Real(0.0), amrex::Real(0.0),
            amrex::Real(1.25), amrex::Real(0.0), amrex::Real(0.0),
            r_lo, r_hi, amrex::Real(-1.0), amrex::Real(1.0));
        require(radial_clipped_path.exits_domain,
            "radial absorbing-boundary smoke chord did not exit");
        rrea::smoke::require_close_rel(
            radial_clipped_path.in_domain_length_m,
            amrex::Real(0.75),
            amrex::Real(1.0e-14),
            "radial absorbing-boundary smoke clipped the wrong length");
        auto const exact_axial_face_path = rrea::ClipRzTransportPath(
            amrex::Real(0.5), amrex::Real(0.0), amrex::Real(0.5),
            amrex::Real(0.5), amrex::Real(0.0), amrex::Real(1.0),
            r_lo, r_hi, amrex::Real(0.0), amrex::Real(1.0));
        require(!exact_axial_face_path.exits_domain,
            "exact axial endpoint unexpectedly became a geometric crossing");
        auto const exact_axial_escape = RreaEndpointEscapePolicyForPath(
            exact_axial_face_path, amrex::Real(0.5), amrex::Real(1.0));
        require(exact_axial_escape.escapes
                && exact_axial_escape.invalidate_parent_in_rrea
                && !exact_axial_escape.ParentStillValidAtPreflight(),
            "exact excluded axial endpoint was not assigned to boundary escape");
        auto const exact_radial_face_path = rrea::ClipRzTransportPath(
            amrex::Real(0.25), amrex::Real(0.0), amrex::Real(0.0),
            amrex::Real(1.0), amrex::Real(0.0), amrex::Real(0.0),
            r_lo, r_hi, amrex::Real(-1.0), amrex::Real(1.0));
        require(!exact_radial_face_path.exits_domain,
            "exact radial endpoint unexpectedly became a geometric crossing");
        auto const exact_radial_escape = RreaEndpointEscapePolicyForPath(
            exact_radial_face_path, r_hi, amrex::Real(0.0));
        require(exact_radial_escape.escapes
                && exact_radial_escape.invalidate_parent_in_rrea
                && !exact_radial_escape.ParentStillValidAtPreflight(),
            "exact excluded radial endpoint was not assigned to boundary escape");
        require(RreaResolveChargedTerminalDisposition(
                    /*positron=*/false,
                    exact_radial_escape.escapes,
                    /*below_transport_cutoff=*/true,
                    /*electron_directional_demotion=*/true)
                    == RreaChargedTerminalDisposition::Escape
                && RreaResolveChargedTerminalDisposition(
                    /*positron=*/true,
                    exact_radial_escape.escapes,
                    /*below_transport_cutoff=*/true,
                    /*electron_directional_demotion=*/false)
                    == RreaChargedTerminalDisposition::Escape,
            "exact radial face did not remove both charged signs before endpoint physics");
        auto const overshoot_escape = RreaEndpointEscapePolicyForPath(
            clipped_path, amrex::Real(0.5), amrex::Real(1.5));
        require(overshoot_escape.escapes
                && !overshoot_escape.invalidate_parent_in_rrea
                && overshoot_escape.ParentStillValidAtPreflight(),
            "geometric overshoot was not left to WarpX boundary deletion");
        double constexpr pi = 3.141592653589793238462643383279502884;
        auto const annular_reentry_path = rrea::ClipRzTransportPath(
            amrex::Real(0.75), amrex::Real(0.0), amrex::Real(0.5),
            amrex::Real(0.75), amrex::Real(pi), amrex::Real(0.5),
            amrex::Real(0.5), r_hi,
            amrex::Real(0.0), amrex::Real(1.0));
        auto const annular_reentry_escape = RreaEndpointEscapePolicyForPath(
            annular_reentry_path, amrex::Real(0.75), amrex::Real(0.5));
        require(annular_reentry_path.exits_domain
                && annular_reentry_escape.escapes
                && annular_reentry_escape.invalidate_parent_in_rrea
                && !annular_reentry_escape.ParentStillValidAtPreflight(),
            "absorber crossing with an in-domain final endpoint was left to WarpX");
        auto const exact_periodic_face_path = rrea::ClipRzTransportPath(
            amrex::Real(0.5), amrex::Real(0.0), amrex::Real(0.5),
            amrex::Real(0.5), amrex::Real(0.0), amrex::Real(1.0),
            r_lo, r_hi, amrex::Real(0.0), amrex::Real(1.0),
            /*periodic_z=*/true);
        amrex::Real periodic_x = amrex::Real(0.0);
        amrex::Real periodic_y = amrex::Real(0.0);
        amrex::Real periodic_z = amrex::Real(0.0);
        exact_periodic_face_path.PointAtFullPathFraction(
            amrex::Real(1.0), periodic_x, periodic_y, periodic_z);
        auto const periodic_escape = RreaEndpointEscapePolicyForPath(
            exact_periodic_face_path,
            std::hypot(periodic_x, periodic_y),
            amrex::Real(1.0));
        require(periodic_z == amrex::Real(0.0)
                && !periodic_escape.escapes
                && !periodic_escape.invalidate_parent_in_rrea,
            "periodic exact upper face was misclassified as absorbing escape");
        amrex::Real const electron_work_eV = RreaFieldWorkIncrementEv(
            amrex::Real(-1.0),
            amrex::Real(0.0),
            amrex::Real(1000.0),
            amrex::Real(0.0),
            amrex::Real(1.0),
            clipped_path.in_domain_length_m);
        amrex::Real const positron_work_eV = RreaFieldWorkIncrementEv(
            amrex::Real(1.0),
            amrex::Real(0.0),
            amrex::Real(1000.0),
            amrex::Real(0.0),
            amrex::Real(1.0),
            clipped_path.in_domain_length_m);
        rrea::smoke::require_close_rel(electron_work_eV, amrex::Real(-500.0), 0.0,
            "electron clipped field-work sign/magnitude is wrong");
        rrea::smoke::require_close_rel(positron_work_eV, amrex::Real(500.0), 0.0,
            "positron clipped field-work sign/magnitude is wrong");
        // A positron hard-tip brems draw must not reserve the transport cutoff
        // in the photon clamp.  The 300 eV survivor is consumed by the normal
        // at-rest branch; the sampled 10.2 keV photon remains unchanged.
        amrex::Real const hard_tip_parent_eV = amrex::Real(10500.0);
        amrex::Real const hard_tip_photon_eV = amrex::Real(10200.0);
        rrea::smoke::require_close_rel(
            RreaClampTrackedBremsPhotonEnergy(
                hard_tip_photon_eV, hard_tip_parent_eV),
            hard_tip_photon_eV,
            amrex::Real(0.0),
            "positron brems hard tip was distorted by a cutoff reservation");
        rrea::smoke::require_close_rel(
            RreaClampTrackedBremsPhotonEnergy(
                amrex::Real(10600.0), hard_tip_parent_eV),
            hard_tip_parent_eV,
            amrex::Real(0.0),
            "brems roundoff clamp did not stop at physical parent energy");

        std::cout << "rrea_secondary_group_smoke passed\n";
        return 0;
    } catch (std::exception const& exc) {
        std::cerr << "rrea_secondary_group_smoke failed: " << exc.what() << "\n";
        return 1;
    }
}
