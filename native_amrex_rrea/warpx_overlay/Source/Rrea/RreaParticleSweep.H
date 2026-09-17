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

    [[nodiscard]] RREA_HOST_DEVICE amrex::Real KineticEnergyEv() const noexcept
    {
        return RreaKineticEnergyEvFromProperVelocity(ux, uy, uz);
    }
    // gamma = sqrt(1 + u^2/c^2); v_z = u_z / gamma.
    [[nodiscard]] RREA_HOST_DEVICE amrex::Real Gamma() const noexcept
    {
        constexpr amrex::Real c = kRreaSpeedOfLightMPerS;
        return std::sqrt(
            amrex::Real(1.0) + (ux * ux + uy * uy + uz * uz) / (c * c));
    }
    [[nodiscard]] RREA_HOST_DEVICE amrex::Real PhotonEnergyEv() const noexcept
    {
        return std::sqrt(ux * ux + uy * uy + uz * uz);
    }
    // The parent key the secondary transaction stages removals under.
    [[nodiscard]] RREA_HOST_DEVICE std::uint64_t StableKey() const noexcept
    {
        return static_cast<std::uint64_t>(rrea::kinetic::abs_id(id))
            ^ (static_cast<std::uint64_t>(cpu) << 40);
    }
};

template <typename Fn>
void RreaForEachValidParticle(WarpXParticleContainer& species, Fn&& fn)
{
#if defined(WARPX_DIM_RZ)
    GpuSynchronize();  // this visitor and its dynamic bookkeeping are host-only
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

// Named functor rather than an extended __device__ lambda: the enclosing
// template takes the caller's Kernel and Consume as template arguments, and
// nvcc forbids a function-local type there when the body contains an extended
// lambda. The device body is unchanged.
#if defined(RREA_USE_CUDA) && defined(WARPX_DIM_RZ)
template<class Tile,class Output,class Kernel>
struct RreaLiveParticleLaunch {
    Tile tile;
    amrex::ParticleReal* r;
    amrex::ParticleReal* z;
    amrex::ParticleReal* theta;
    amrex::ParticleReal* ux;
    amrex::ParticleReal* uy;
    amrex::ParticleReal* uz;
    amrex::ParticleReal* w;
    Output* out;
    unsigned char* keep;
    Kernel kernel;
    AMREX_GPU_DEVICE void operator()(amrex::Long i) const noexcept {
        auto& idcpu=tile.idcpu(static_cast<int>(i));
        amrex::ParticleIDWrapper<> id(idcpu);
        amrex::ParticleCPUWrapper cpu(idcpu);
        if(!id.is_valid())return;
        keep[i]=kernel(RreaLiveParticle{r[i],z[i],theta[i],ux[i],uy[i],uz[i],w[i],
            idcpu,static_cast<amrex::Long>(id),static_cast<int>(cpu)},out[i]);
    }
};
#endif
// Map a live particle into a plain record on its execution device. The host
// consumes only those records, never the underlying particle arrays. Output
// order follows tile/slot order, so RNG identities and insertion order remain
// stable. The CUDA launch resides in a free function: this is also usable from
// private coupling methods without NVCC extended-lambda access restrictions.
template<class Output,class Kernel,class Consume>
void RreaMapLiveParticles(WarpXParticleContainer& species,bool gpu,Kernel kernel,Consume consume)
{
#if defined(RREA_USE_CUDA) && defined(WARPX_DIM_RZ)
    if(gpu) {
        for(WarpXParIter pti(species,0);pti.isValid();++pti) {
            auto const tile=pti.GetParticleTile().getParticleTileData();
            auto& a=pti.GetAttribs();
            auto* r=a[PIdx::r].dataPtr();auto* z=a[PIdx::z].dataPtr();
            auto* theta=a[PIdx::theta].dataPtr();auto* ux=a[PIdx::ux].dataPtr();
            auto* uy=a[PIdx::uy].dataPtr();auto* uz=a[PIdx::uz].dataPtr();
            auto* w=a[PIdx::w].dataPtr();
            auto const n=pti.numParticles();
            rrea::GpuVector<Output> records(static_cast<std::size_t>(n));
            rrea::GpuVector<unsigned char> accepted(static_cast<std::size_t>(n),0);
            auto* out=records.data();auto* keep=accepted.data();
            amrex::ParallelFor(n,RreaLiveParticleLaunch<decltype(tile),Output,Kernel>{
                tile,r,z,theta,ux,uy,uz,w,out,keep,kernel});
            rrea::GpuSynchronize();
            for(amrex::Long i=0;i<n;++i)if(accepted[i])consume(records[i]);
        }
        return;
    }
#else
    amrex::ignore_unused(gpu);
#endif
    RreaForEachValidParticle(species,[&](RreaLiveParticle const& p) {
        Output result{};
        if(kernel(p,result))consume(result);
    });
}

}  // namespace rrea::warpx

#endif  // RREA_PARTICLE_SWEEP_H
