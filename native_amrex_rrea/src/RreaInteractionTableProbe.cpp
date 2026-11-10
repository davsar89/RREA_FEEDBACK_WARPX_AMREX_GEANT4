#include "rrea/RreaGpuSmoke.H"
#include "rrea/RreaCsvTextUtil.H"
#include "rrea/RreaInteractionTables.H"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void usage()
{
    std::cerr
        << "usage: rrea_interaction_table_probe "
           "<transport_physics.json> table <name> <energy_eV>\n"
        << "   or: rrea_interaction_table_probe <transport_physics.json> drag < energies_eV\n";
}

}  // namespace

int main(int argc, char** argv)
{
    rrea::smoke::GpuSession gpu_runtime;
    try {
        int arg = 1;
        if (argc - arg < 2) {
            usage();
            return EXIT_FAILURE;
        }
        std::string const transport_config = argv[arg++];
        std::string const mode = argv[arg++];
        rrea::RreaInteractionTables tables;
        tables.Load(transport_config);

        std::cout << std::setprecision(17);
        if (mode == "drag") {
            // Ensemble electron drag (eV/m) at the tables' reference density,
            // one line per energy on stdin: the PARMA birth-rate source's S.
            for (double energy_eV = 0.0; std::cin >> energy_eV;) {
                std::cout << "drag_eV_per_m="
                          << tables.View().ElectronEnsembleDrag(energy_eV, 1.0, true)
                          << "\n";
            }
            return EXIT_SUCCESS;
        }
        if (mode != "table" || argc - arg < 2) {
            usage();
            return EXIT_FAILURE;
        }
        std::string const name = argv[arg++];
        double const energy_eV =
            rrea::csv_text::parse_number<double>(argv[arg++], "energy_eV");
        std::cout << "table=" << name << " energy_eV=" << energy_eV
                  << " value=" << tables.ProbeTableValue(name, energy_eV) << "\n";
    } catch (std::exception const& exc) {
        std::cerr << "rrea_interaction_table_probe failed: " << exc.what() << "\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
