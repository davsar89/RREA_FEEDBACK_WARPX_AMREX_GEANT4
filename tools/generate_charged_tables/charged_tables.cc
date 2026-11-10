// Restricted ionization and its complementary secondary distribution, from
// Geant4 10.7.4 eIoni in G4_AIR. Reuses the reference-dump construction formerly
// in tools/geant4_reference_dump; no Geant4 physics executes in the RREA engine.
//
// Sigma(c) = integral_c^Tmax dSigma/dW dW;
// F(W|c) = 1 - Sigma(W)/Sigma(c). Invert the SAME process rate used in the
// transport table, for both electrons and positrons. The output supports the
// independent check S_restricted(c) + Sigma(c)*<W> = S_unrestricted.
// Units in the files: eV, eV/m and 1/m. The caller supplies energy and CDF grids.
#include "G4Box.hh"
#include "G4EmCalculator.hh"
#include "G4EmStandardPhysics_option4.hh"
#include "G4Electron.hh"
#include "G4LogicalVolume.hh"
#include "G4Material.hh"
#include "G4NistManager.hh"
#include "G4ParticleGun.hh"
#include "G4Positron.hh"
#include "G4PVPlacement.hh"
#include "G4RunManager.hh"
#include "G4SystemOfUnits.hh"
#include "G4VModularPhysicsList.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

class Air : public G4VUserDetectorConstruction {
    G4VPhysicalVolume* Construct() override {
        auto* material = G4NistManager::Instance()->FindOrBuildMaterial("G4_AIR");
        auto* volume = new G4LogicalVolume(new G4Box("air", 5*m, 5*m, 5*m), material, "air");
        return new G4PVPlacement(nullptr, {}, volume, "air", nullptr, false, 0);
    }
};
class Physics : public G4VModularPhysicsList {
public:
    Physics() { RegisterPhysics(new G4EmStandardPhysics_option4()); }
};
class Primary : public G4VUserPrimaryGeneratorAction {
    G4ParticleGun gun{1};
    void GeneratePrimaries(G4Event* event) override { gun.GeneratePrimaryVertex(event); }
public:
    Primary() {
        gun.SetParticleDefinition(G4Electron::Electron());
        gun.SetParticleEnergy(MeV);
        gun.SetParticleMomentumDirection({0, 0, 1});
    }
};

std::vector<double> numbers(char const* path) {
    std::ifstream input(path);
    std::vector<double> values;
    for (double value; input >> value;) values.push_back(value);
    if (values.empty() || !input.eof()) throw std::runtime_error("Invalid numeric grid");
    return values;
}

// F(W|cut)=1-Sigma(W)/Sigma(cut). Reuse one transfer grid across energies:
// Penelope caches a full material table for every distinct cut, so bisection
// at millions of unique cuts is both wasteful and unnecessary. Assess grid
// accuracy using S_restricted + Sigma*integral_0^1 W(u)du = S_unrestricted.
template<class CrossSection>
std::vector<double> secondary_quantiles(double cut, double maximum, double rate,
    std::vector<double> const& energies, std::vector<double> const& uniforms,
    CrossSection sigma)
{
    std::vector<double> w{cut}, f{0}, result;
    if (rate <= 0) return std::vector<double>(uniforms.size(), cut);
    for (double transfer : energies) {
        if (transfer <= cut || transfer >= maximum) continue;
        double const probability = 1 - sigma(transfer)/rate;
        if (probability < f.back()-1e-10 || probability > 1+1e-10)
            throw std::runtime_error("Nonmonotone secondary probability");
        w.push_back(transfer);
        f.push_back(std::min(1.0, std::max(f.back(), probability)));
    }
    w.push_back(maximum); f.push_back(1);
    for (double u : uniforms) {
        if (!(u >= 0 && u <= 1)) throw std::runtime_error("Invalid quantile");
        if (u == 0) { result.push_back(cut); continue; }
        if (u == 1) { result.push_back(maximum); continue; }
        auto const j = std::lower_bound(f.begin(), f.end(), u)-f.begin();
        result.push_back(w[j-1]+(w[j]-w[j-1])*(u-f[j-1])/(f[j]-f[j-1]));
    }
    return result;
}

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--self-test") {
        // Leading free-electron dSigma/dW ~ W^-2: analytic inverse and support.
        double const maximum = 1e5;
        std::vector<double> grid, u{0, .1, .5, .9, .99, .99999, 1};
        for (int j = 0; j <= 1000; ++j) grid.push_back(std::pow(maximum, j/1000.0));
        auto sigma = [=](double w) { return 1/w-1/maximum; };
        auto const w = secondary_quantiles(1, maximum, sigma(1), grid, u, sigma);
        for (std::size_t j = 0; j < u.size(); ++j) {
            double const exact = 1/(1-u[j]+u[j]/maximum);
            if (std::abs(w[j]/exact-1) > 4e-5)
                throw std::runtime_error("W^-2 conditional quantile failed");
        }
        return 0;
    }
    if (argc != 6) {
        std::cerr << "usage: rrea_charged_tables cut_eV energies.txt cdf_energies.txt quantiles.txt output_prefix\n";
        return 2;
    }
    double const cut_eV = std::stod(argv[1]);
    if (!(cut_eV > 0) || !std::isfinite(cut_eV)) throw std::runtime_error("Invalid secondary cut");
    auto const energies = numbers(argv[2]), cdf_energies = numbers(argv[3]), quantiles = numbers(argv[4]);
    G4RunManager manager;
    manager.SetVerboseLevel(0);
    manager.SetUserInitialization(new Air);
    manager.SetUserInitialization(new Physics);
    manager.SetUserAction(new Primary);
    manager.Initialize();
    manager.BeamOn(1);
    auto* air = G4NistManager::Instance()->FindOrBuildMaterial("G4_AIR");
    G4EmCalculator calculator;
    calculator.SetVerbose(0);
    for (bool positron : {false, true}) {
        G4ParticleDefinition* particle = G4Electron::ElectronDefinition();
        if (positron) particle = G4Positron::PositronDefinition();
        std::string const stem = std::string(argv[5]) + (positron ? "_positron" : "_electron");
        std::ofstream rates(stem + ".csv"), cdf(stem + "_cdf.csv");
        rates << std::setprecision(17)
              << "kinetic_energy_eV,collision_stopping_eV_per_m_stp,hard_inverse_m_stp,unrestricted_eV_per_m\n";
        cdf << std::setprecision(17) << "primary_energy_eV,cdf_u,secondary_energy_eV\n";
        for (double energy_eV : energies) {
            if (!(energy_eV > 0) || !std::isfinite(energy_eV)) throw std::runtime_error("Invalid energy");
            double const energy = energy_eV*eV;
            double const maximum = energy_eV/(positron ? 1.0 : 2.0);
            auto sigma = [&](double threshold_eV) {
                return calculator.ComputeCrossSectionPerVolume(
                    energy, particle, "eIoni", air, threshold_eV*eV)/(1/m);
            };
            double const rate = maximum > cut_eV ? sigma(cut_eV) : 0.;
            rates << energy_eV << ','
                  << calculator.ComputeDEDX(energy, particle, "eIoni", air, cut_eV*eV)/(eV/m) << ','
                  << rate << ','
                  << calculator.ComputeDEDX(energy, particle, "eIoni", air, energy)/(eV/m) << '\n';
            if (maximum < cut_eV || !std::binary_search(cdf_energies.begin(), cdf_energies.end(), energy_eV)) continue;
            auto const secondary = secondary_quantiles(cut_eV, maximum, rate, energies, quantiles, sigma);
            for (std::size_t j = 0; j < quantiles.size(); ++j) {
                cdf << energy_eV << ',' << quantiles[j] << ',' << secondary[j] << '\n';
            }
        }
    }
    std::cout << "G4_AIR density " << air->GetDensity()/(kg/m3)
              << " kg/m3; secondary cut " << cut_eV << " eV\n";
}
