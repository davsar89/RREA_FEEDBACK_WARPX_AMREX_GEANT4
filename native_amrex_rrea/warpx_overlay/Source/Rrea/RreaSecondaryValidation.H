#ifndef RREA_SECONDARY_VALIDATION_H
#define RREA_SECONDARY_VALIDATION_H
#include "RreaKineticStep.H"
namespace rrea::warpx {
RREA_HOST_DEVICE inline bool RreaKnownSecondarySpecies(RreaSecondarySpecies species) {
    return species == RreaSecondarySpecies::Electron
        || species == RreaSecondarySpecies::Photon
        || species == RreaSecondarySpecies::Positron;
}
// Known species, finite state and positive weight: required of both a birth
// record and a final survivor.
template<class Particle>
RREA_HOST_DEVICE bool RreaSecondaryStateValid(Particle const& p) {
    return RreaKnownSecondarySpecies(p.species)
        && std::isfinite(static_cast<double>(p.x_m))
        && std::isfinite(static_cast<double>(p.y_m))
        && std::isfinite(static_cast<double>(p.z_m))
        && std::isfinite(static_cast<double>(p.dir_x))
        && std::isfinite(static_cast<double>(p.dir_y))
        && std::isfinite(static_cast<double>(p.dir_z))
        && std::isfinite(static_cast<double>(p.kinetic_or_photon_energy_eV))
        && p.weight > amrex::Real(0.0);
}
// Final insertion state is distinct from each immutable birth transaction.
RREA_HOST_DEVICE inline bool KineticFinalSurvivorValid(
    KineticStep const& context,RreaSecondaryParticle const& survivor) {
    bool const valid_energy =
        survivor.species == RreaSecondarySpecies::Photon
        ? survivor.kinetic_or_photon_energy_eV > amrex::Real(0.0)
        : survivor.kinetic_or_photon_energy_eV
            >= context.coupling.LowEnergyCutoffEv();
    return RreaSecondaryStateValid(survivor) && valid_energy
        && context.TransientEnergyIsCertified(survivor)
        && RreaRzPositionInHalfOpenDomain(
            std::hypot(survivor.x_m, survivor.y_m),survivor.z_m,
            context.geom.ProbLo(0),context.geom.ProbHi(0),
            context.geom.ProbLo(1),context.geom.ProbHi(1));
}
// The same complete group validator is used by CPU and CUDA. Only the global
// MPI decision remains on the host, before any material/ledger commit.
template<class Group>
RREA_HOST_DEVICE std::array<amrex::Long,4> ValidateKineticGroup(
    KineticStep const& context,Group const& group) {
    auto position_in_domain=[&](amrex::Real r,amrex::Real z) {
        return RreaRzPositionInHalfOpenDomain(r,z,
            context.geom.ProbLo(0),context.geom.ProbHi(0),
            context.geom.ProbLo(1),context.geom.ProbHi(1));
    };
    std::array<amrex::Long,4> errors{0,0,0,0};
            auto const conservation = RreaValidateSecondaryGroupConservation(
                group, context.coupling.LowEnergyCutoffEv());
            if (conservation.required
                && (!conservation.present || !conservation.finite)) {
                ++errors[2];
            } else if (conservation.required && !conservation.valid) {
                ++errors[3];
            }
            for (auto const& source : group.material_sources) {
                bool const finite_payload =
                    std::isfinite(static_cast<double>(source.event_weight))
                    && std::isfinite(static_cast<double>(source.secondary_energy_eV));
                if (!finite_payload || !(source.event_weight > amrex::Real(0.0))
                    || source.secondary_energy_eV < amrex::Real(0.0)) {
                    ++errors[1];
                }
                if (!position_in_domain(source.r_m, source.z_m)) {
                    ++errors[0];
                }
            }
            for (auto const& particle : group.particles) {
                bool const valid_energy =
                    particle.species == RreaSecondarySpecies::Photon
                    ? particle.kinetic_or_photon_energy_eV > amrex::Real(0.0)
                    : particle.kinetic_or_photon_energy_eV >= amrex::Real(0.0);
                // The [0, 1] range test also rejects a NaN birth fraction.
                bool const valid_birth_metadata =
                    RreaKnownSecondarySpecies(particle.root_species)
                    && particle.birth_elapsed_step_fraction >= amrex::Real(0.0)
                    && particle.birth_elapsed_step_fraction <= amrex::Real(1.0)
                    && particle.transient_lineage
                        < kRreaTransientLineageCapacity;
                if (!RreaSecondaryStateValid(particle) || !valid_birth_metadata
                    || !valid_energy) {
                    ++errors[1];
                }
                amrex::Real const particle_r_m =
                    std::hypot(particle.x_m, particle.y_m);
                if (!position_in_domain(particle_r_m, particle.z_m)) {
                    ++errors[0];
                }
            }
              return errors;
}
}
#endif
