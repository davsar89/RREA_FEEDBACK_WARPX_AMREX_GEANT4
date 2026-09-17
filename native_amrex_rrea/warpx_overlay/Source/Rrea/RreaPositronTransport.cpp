#include "RreaTransportStep.H"
#include "RreaKineticGpu.H"
#include <AMReX_Particle.H>

#if defined(AMREX_USE_OMP) || defined(RREA_USE_HOST_OMP)
#include <omp.h>
#endif

namespace rrea::warpx {

void TransportStep::ApplyPositronTransport()
{
#if defined(WARPX_DIM_RZ)
            auto& positrons =
                warpx.GetPartContainer().GetParticleContainerFromName(coupling.PositronSpeciesName());
            PreviousRzComponentIndices const positron_previous =
                require_previous_rz_components(positrons, coupling.PositronSpeciesName());
            // Use the same deterministic per-chunk replay as the electron and
            // photon paths when scoped transport threading is enabled.
        auto const kernel=Kinetic();
            for (WarpXParIter pti(positrons, 0); pti.isValid(); ++pti) {
                auto& tile = pti.GetParticleTile();
                auto& attribs = pti.GetAttribs();
                auto* const r_data = attribs[PIdx::r].dataPtr();
                auto* const z_data = attribs[PIdx::z].dataPtr();
                auto* const w_data = attribs[PIdx::w].dataPtr();
                auto* const ux_data = attribs[PIdx::ux].dataPtr();
                auto* const uy_data = attribs[PIdx::uy].dataPtr();
                auto* const uz_data = attribs[PIdx::uz].dataPtr();
                auto* const theta_data = attribs[PIdx::theta].dataPtr();
                auto* const previous_r_data = pti.GetAttribs(positron_previous.r).dataPtr();
                auto* const previous_z_data = pti.GetAttribs(positron_previous.z).dataPtr();
                auto* const previous_theta_data = pti.GetAttribs(positron_previous.theta).dataPtr();
                auto* const previous_ux_data = pti.GetAttribs(positron_previous.ux).dataPtr();
                auto* const previous_uy_data = pti.GetAttribs(positron_previous.uy).dataPtr();
                auto* const previous_uz_data = pti.GetAttribs(positron_previous.uz).dataPtr();
            ChargedSoAView const soa{
                r_data, z_data, w_data, ux_data, uy_data, uz_data,
                theta_data,
                previous_r_data, previous_z_data, previous_theta_data,
                previous_ux_data, previous_uy_data, previous_uz_data};
                long const np = pti.numParticles();
#ifdef RREA_USE_CUDA
                if(KineticGpuEnabled()) {
                    auto const tile_data=tile.getParticleTileData();
                    RunKineticGpuHistories(kernel,np,
                        [=] AMREX_GPU_DEVICE(KineticStep const& k,long ip,DeviceTransportSideEffects& fx) noexcept {
                            auto& idcpu=tile_data.idcpu(static_cast<int>(ip));
                            amrex::ParticleIDWrapper<> id(idcpu);
                            amrex::ParticleCPUWrapper cpu(idcpu);
                            k.Positron(fx,soa,ip,id,static_cast<int>(cpu));
                        },
                        [&](long,TransportSideEffects& fx) {
                            pending_side_effects.AppendGroupsFrom(fx);
                            invalidated_positrons=invalidated_positrons || fx.invalidated;
                            repositioned_positrons=repositioned_positrons || fx.repositioned;
                        });
                    continue;
                }
#endif
                constexpr long transport_chunk_size = 64;
                long const n_chunks =
                    (np + transport_chunk_size - 1) / transport_chunk_size;
                std::vector<TransportSideEffects> chunk_fx(
                    static_cast<std::size_t>(std::max<long>(n_chunks, 0)),
                    TransportSideEffects{coupling.LowEnergyCutoffEv()});
#if defined(AMREX_USE_OMP) || defined(RREA_USE_HOST_OMP)
                int const transport_num_threads =
                    (coupling.TransportOmpThreads() > 0)
                        ? coupling.TransportOmpThreads()
                        : omp_get_max_threads();
#pragma omp parallel for schedule(dynamic, 1) num_threads(transport_num_threads) if(transport_num_threads > 1 && n_chunks > 1)
#endif
                for (long chunk = 0; chunk < n_chunks; ++chunk) {
                TransportSideEffects& fx = chunk_fx[static_cast<std::size_t>(chunk)];
                long const ip_begin = chunk * transport_chunk_size;
                long const ip_end = std::min(np, ip_begin + transport_chunk_size);
                for (long ip = ip_begin; ip < ip_end; ++ip) {
                kernel.Positron(fx,soa,ip,tile.id(static_cast<int>(ip)),tile.cpu(static_cast<int>(ip)));
            }
                }  // chunk loop

                for (auto& chunk_effects : chunk_fx) {
                    pending_side_effects.AppendGroupsFrom(chunk_effects);
                    invalidated_positrons =
                        invalidated_positrons || chunk_effects.invalidated;
                    repositioned_positrons =
                        repositioned_positrons || chunk_effects.repositioned;
                }
            }
#else
        amrex::Abort("RREA MC/PIC positron transport currently requires WarpX RZ");
#endif
}

}  // namespace rrea::warpx
