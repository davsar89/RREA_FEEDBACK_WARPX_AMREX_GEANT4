#include "RreaTransportStep.H"

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
    amrex::ignore_unused(time_s);

    if (!tables.Loaded()) {
        amrex::Abort("Schema-6 transport requires loaded interaction tables");
    }

    auto const& geom = warpx.Geom(0);
    bool const periodic_z = geom.isPeriodic(1);
#if (AMREX_SPACEDIM >= 2)
    amrex::Real const reference_z =
        amrex::Real(0.5) * (geom.ProbLo(1) + geom.ProbHi(1));
#else
    amrex::Real const reference_z = amrex::Real(0.0);
#endif
    amrex::Real const density_ratio =
        positive_or(coupling.TransportDensityRatioAtZ(reference_z),
            positive_or(m_config.density_ratio, amrex::Real(1.0)));
    {
        if (dt_s <= amrex::Real(0.0)) {
            return;
        }

        TransportStep ctx{
            warpx,
            coupling,
            tables,
            m_config,
            step,
            time_s,
            dt_s,
            geom,
            periodic_z,
            density_ratio};
#if defined(WARPX_DIM_RZ)
        // Keep nested MFIter permission active across every transport phase.
        ScopedMultipleMFIters allow_nested_mesh_access;
#endif
        ctx.ApplyElectronTransport();
        ctx.ApplyPhotonTransport();
        ctx.ApplyPositronTransport();
        ctx.DrainAndCommitSecondaries();
    }
}

}  // namespace rrea::warpx
