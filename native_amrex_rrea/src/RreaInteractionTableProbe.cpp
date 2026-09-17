#include "rrea/RreaGpuSmoke.H"
#include "rrea/RreaInteractionTables.H"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

double parse_number(char const* text, char const* label)
{
    try {
        std::size_t used = 0;
        double const value = std::stod(text, &used);
        if (used != std::string(text).size()) {
            throw std::runtime_error("trailing characters");
        }
        return value;
    } catch (std::exception const& exc) {
        throw std::runtime_error(
            std::string("invalid ") + label + " '" + text + "': " + exc.what());
    }
}

void usage()
{
    std::cerr
        << "usage: rrea_interaction_table_probe "
           "<transport_physics.json> table <name> <energy_eV>\n"
        << "   or: rrea_interaction_table_probe "
           "<transport_physics.json> photon|electron|positron <energy_eV> <density_ratio>\n"
        << "   or: rrea_interaction_table_probe <transport_physics.json> sampler <name> <energy_eV>\n"
        << "   or: rrea_interaction_table_probe <transport_physics.json> final-state <name> <energy_eV>\n";
}

void probe_sampler(
    rrea::RreaInteractionTables const& tables,
    std::string const& name,
    double energy_eV)
{
    double value = 0.0;
    if (name == "electron_gs" || name == "electron_wentzel") {
        value = tables.SampleGsCosTheta(false, energy_eV, 20.0, 0.37);
    } else if (name == "positron_gs" || name == "positron_wentzel") {
        value = tables.SampleGsCosTheta(true, energy_eV, 20.0, 0.37);
    } else if (name == "electron_single_elastic") {
        value = tables.SampleSingleElasticCosTheta(false, energy_eV, 0.37);
    } else if (name == "positron_single_elastic") {
        value = tables.SampleSingleElasticCosTheta(true, energy_eV, 0.37);
    } else if (name == "electron_brems_energy") {
        value = tables.SampleBremsstrahlungPhotonEnergy(energy_eV, 0.37);
    } else if (name == "positron_brems_energy") {
        value = tables.SamplePositronBremsstrahlungPhotonEnergy(energy_eV, 0.37);
    } else if (name == "moller_secondary") {
        value = tables.SampleMollerSecondaryEnergy(energy_eV, 0.37);
    } else if (name == "bhabha_secondary") {
        value = tables.SamplePositronBhabhaSecondaryEnergy(energy_eV, 0.37);
    } else if (name == "brems_cos_theta") {
        value = tables.SampleBremsstrahlungCosTheta(energy_eV, 0.37);
    } else if (name == "photon_channel") {
        value = static_cast<int>(tables.SamplePhotonChannel(energy_eV, 0.37));
    } else if (name == "photoelectron_cos_theta") {
        value = tables.SamplePhotoelectronCosTheta(energy_eV, 0.37);
    } else {
        throw std::runtime_error("unknown sampler probe name: " + name);
    }
    std::cout << "schema_version=" << tables.SchemaVersion()
              << " sampler=" << name
              << " energy_eV=" << energy_eV
              << " value=" << value << "\n";
}

void probe_final_state(
    rrea::RreaInteractionTables const& tables,
    std::string const& name,
    double energy_eV)
{
    std::cout << "schema_version=" << tables.SchemaVersion()
              << " final_state=" << name
              << " energy_eV=" << energy_eV;
    if (name == "compton") {
        auto const state = tables.SampleComptonKleinNishinaFinalState(
            energy_eV, 0.37, 0.61);
        std::cout << " photon_eV=" << state.scattered_photon_energy_eV
                  << " electron_eV=" << state.recoil_electron_energy_eV
                  << " cos_theta=" << state.cos_theta;
    } else if (name == "photoelectric") {
        auto const state = tables.SamplePhotoelectricFinalState(energy_eV, 0.37);
        std::cout << " electron_eV=" << state.photoelectron_energy_eV
                  << " binding_eV=" << state.binding_energy_eV
                  << " relaxation_eV=" << state.relaxation_energy_eV
                  << " element_Z=" << state.element_Z;
    } else if (name == "photoelectric_electron_energy") {
        std::cout << " electron_eV=" << tables.PhotoelectricElectronEnergy(energy_eV);
    } else if (name == "pair_nuclear" || name == "pair_triplet") {
        auto const channel = name == "pair_nuclear"
            ? rrea::RreaPhotonChannel::PairNuclear
            : rrea::RreaPhotonChannel::PairTriplet;
        auto const state = tables.SamplePairFinalState(
            energy_eV, channel, 0.37, 0.41, 0.53, 0.67, 0.79);
        std::cout << " valid=" << (state.valid ? 1 : 0)
                  << " electron_eV=" << state.electron_energy_eV
                  << " positron_eV=" << state.positron_energy_eV
                  << " recoil_electron_eV=" << state.recoil_electron_energy_eV
                  << " local_deposit_eV=" << state.local_deposit_energy_eV;
    } else if (name == "annihilation") {
        auto const state = tables.SamplePositronAnnihilationFinalState(
            energy_eV, 0.37, 0.73);
        std::cout << " valid=" << (state.valid ? 1 : 0)
                  << " at_rest=" << (state.at_rest ? 1 : 0)
                  << " photon1_eV=" << state.photon1_energy_eV
                  << " photon2_eV=" << state.photon2_energy_eV;
    } else {
        throw std::runtime_error("unknown final-state probe name: " + name);
    }
    std::cout << "\n";
}

}  // namespace

int main(int argc, char** argv)
{
    rrea::smoke::GpuSession gpu_runtime;
    try {
        int arg = 1;
        if (argc - arg < 4) {
            usage();
            return EXIT_FAILURE;
        }
        std::string const transport_config = argv[arg++];
        std::string const mode = argv[arg++];
        rrea::RreaInteractionTables tables;
        tables.Load(transport_config);

        std::cout << std::setprecision(17);
        if (mode == "table") {
            std::string const name = argv[arg++];
            double const energy_eV = parse_number(argv[arg++], "energy_eV");
            std::cout << "schema_version=" << tables.SchemaVersion()
                      << " table=" << name
                      << " energy_eV=" << energy_eV
                      << " value=" << tables.ProbeTableValue(name, energy_eV)
                      << "\n";
            return EXIT_SUCCESS;
        }

        if (mode == "sampler" || mode == "final-state") {
            std::string const name = argv[arg++];
            double const energy_eV = parse_number(argv[arg++], "energy_eV");
            if (mode == "sampler") {
                probe_sampler(tables, name, energy_eV);
            } else {
                probe_final_state(tables, name, energy_eV);
            }
            return EXIT_SUCCESS;
        }

        double const energy_eV = parse_number(argv[arg++], "energy_eV");
        double const density_ratio = parse_number(argv[arg++], "density_ratio");
        if (mode == "photon") {
            auto const value = tables.PhotonProcessInverseLengths(
                energy_eV, density_ratio);
            std::cout << "schema_version=" << tables.SchemaVersion()
                      << " energy_eV=" << energy_eV
                      << " density_ratio=" << density_ratio
                      << " compton_per_m=" << value.compton_per_m
                      << " photoelectric_per_m=" << value.photoelectric_per_m
                      << " pair_nuclear_per_m=" << value.pair_nuclear_per_m
                      << " pair_triplet_per_m=" << value.pair_triplet_per_m
                      << " total_per_m=" << value.TotalPerM() << "\n";
        } else if (mode == "electron") {
            auto const value = tables.ElectronTransportCoefficients(
                energy_eV, density_ratio);
            std::cout << "schema_version=" << tables.SchemaVersion()
                      << " energy_eV=" << energy_eV
                      << " density_ratio=" << density_ratio
                      << " collision_stopping_eV_per_m="
                      << value.collision_stopping_eV_per_m
                      << " soft_radiative_stopping_eV_per_m="
                      << value.soft_radiative_stopping_eV_per_m
                      << " hard_moller_inverse_m=" << value.hard_moller_inverse_m
                      << " tracked_brems_inverse_m=" << value.tracked_brems_inverse_m
                      << " elastic_transport_inverse_m="
                      << value.elastic_transport_inverse_m << "\n";
        } else if (mode == "positron") {
            auto const value = tables.PositronTransportCoefficients(
                energy_eV, density_ratio);
            std::cout << "schema_version=" << tables.SchemaVersion()
                      << " energy_eV=" << energy_eV
                      << " density_ratio=" << density_ratio
                      << " collision_stopping_eV_per_m="
                      << value.collision_stopping_eV_per_m
                      << " soft_radiative_stopping_eV_per_m="
                      << value.soft_radiative_stopping_eV_per_m
                      << " hard_bhabha_inverse_m=" << value.hard_bhabha_inverse_m
                      << " tracked_brems_inverse_m=" << value.tracked_brems_inverse_m
                      << " elastic_transport_inverse_m="
                      << value.elastic_transport_inverse_m
                      << " annihilation_inverse_m=" << value.annihilation_inverse_m
                      << "\n";
        } else {
            usage();
            return EXIT_FAILURE;
        }
    } catch (std::exception const& exc) {
        std::cerr << "rrea_interaction_table_probe failed: " << exc.what() << "\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
