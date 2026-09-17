#ifndef RREA_PARTICLE_SWEEP_H
#define RREA_PARTICLE_SWEEP_H

// The ONE rank-local sweep over live RZ macroparticles.
//
// Soft-deleted macros keep their SoA slots until the next compaction, so a
// sum that does not skip invalid ids counts particles the engine has already
// removed.  Every coupling and diagnostic reduction goes through this loop so
// that guard has a single owner.
//
// Energies: a charged macro's kinetic energy is T = (gamma - 1) m_e c^2 with
// gamma = sqrt(1 + u^2/c^2) (u is the stored proper velocity), evaluated by
// RreaKineticEnergyEvFromProperVelocity in its cancellation-free form; a
// photon stores its energy in eV as |u| itself.

#include "RreaParticleInteraction.H"

#include "Particles/WarpXParticleContainer.H"

#include <AMReX.H>
#include <AMReX_REAL.H>

#include <cmath>
#include <cstdint>
#include <cstdlib>

namespace rrea::warpx {

// One live macroparticle.  The attributes are copies (read-only to the
// visitor); `weight` and `idcpu` are references into the tile, so the
// population controller can rewrite a weight or invalidate a slot in place
// even though the visitor receives the particle by const reference.
struct RreaLiveParticle {
    amrex::ParticleReal r;
    amrex::ParticleReal z;
    amrex::ParticleReal theta;
    amrex::ParticleReal ux;
    amrex::ParticleReal uy;
    amrex::ParticleReal uz;
    amrex::ParticleReal& weight;
    std::uint64_t& idcpu;
    amrex::Long id;
    int cpu;

    [[nodiscard]] amrex::Real KineticEnergyEv() const noexcept
    {
        return RreaKineticEnergyEvFromProperVelocity(ux, uy, uz);
    }
    // gamma = sqrt(1 + u^2/c^2); v_z = u_z / gamma.
    [[nodiscard]] amrex::Real Gamma() const noexcept
    {
        constexpr amrex::Real c = kRreaSpeedOfLightMPerS;
        return std::sqrt(
            amrex::Real(1.0) + (ux * ux + uy * uy + uz * uz) / (c * c));
    }
    [[nodiscard]] amrex::Real PhotonEnergyEv() const noexcept
    {
        return std::sqrt(ux * ux + uy * uy + uz * uz);
    }
    // The parent key the secondary transaction stages removals under.
    [[nodiscard]] std::uint64_t StableKey() const noexcept
    {
        return static_cast<std::uint64_t>(std::llabs(id))
            ^ (static_cast<std::uint64_t>(cpu) << 40);
    }
};

template <typename Fn>
void RreaForEachValidParticle(WarpXParticleContainer& species, Fn&& fn)
{
#if defined(WARPX_DIM_RZ)
    for (WarpXParIter pti(species, 0); pti.isValid(); ++pti) {
        auto& tile = pti.GetParticleTile();
        auto tile_data = tile.getParticleTileData();
        auto& attribs = pti.GetAttribs();
        auto* const r = attribs[PIdx::r].dataPtr();
        auto* const z = attribs[PIdx::z].dataPtr();
        auto* const theta = attribs[PIdx::theta].dataPtr();
        auto* const ux = attribs[PIdx::ux].dataPtr();
        auto* const uy = attribs[PIdx::uy].dataPtr();
        auto* const uz = attribs[PIdx::uz].dataPtr();
        auto* const w = attribs[PIdx::w].dataPtr();
        long const np = pti.numParticles();
        for (long ip = 0; ip < np; ++ip) {
            int const slot = static_cast<int>(ip);
            auto const particle_id = tile.id(slot);
            if (!particle_id.is_valid()) {
                continue;
            }
            fn(RreaLiveParticle{
                r[ip], z[ip], theta[ip], ux[ip], uy[ip], uz[ip], w[ip],
                tile_data.idcpu(slot),
                static_cast<amrex::Long>(particle_id),
                static_cast<int>(tile.cpu(slot))});
        }
    }
#else
    amrex::ignore_unused(species, fn);
    amrex::Abort("RREA macroparticle sweeps require the WarpX RZ build");
#endif
}

}  // namespace rrea::warpx

#endif  // RREA_PARTICLE_SWEEP_H
