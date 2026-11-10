#include "RreaTransportStep.H"
#include "RreaKineticGpu.H"

#include <utility>

namespace rrea::warpx {

RreaParticleInteraction::RreaParticleInteraction(
    RreaParticleInteractionConfig config)
    : m_config(std::move(config))
{}

void RreaParticleInteraction::Apply(
    WarpX& warpx,
    RreaWarpXCoupling& coupling,
    rrea::RreaInteractionTables const& tables,
    int step,
    amrex::Real time_s,
    amrex::Real dt_s)
{
    RequireHybridManagedMemory();
    if (KineticGpuEnabled()) {  // also rejects gpu on a CPU build
#ifdef RREA_USE_CUDA
        ConfigureKineticGpu();
#endif
    }
    TransportStep ctx{
        warpx, coupling, tables, m_config, step, time_s, dt_s, warpx.Geom(0)};
#if defined(WARPX_DIM_RZ)
    // Keep nested MFIter permission active across every transport phase.
    ScopedMultipleMFIters allow_nested_mesh_access;
#endif
    ctx.ApplyElectronTransport();
    ctx.ApplyPhotonTransport();
    ctx.ApplyPositronTransport();
    ctx.DrainAndCommitSecondaries();
}

}  // namespace rrea::warpx
