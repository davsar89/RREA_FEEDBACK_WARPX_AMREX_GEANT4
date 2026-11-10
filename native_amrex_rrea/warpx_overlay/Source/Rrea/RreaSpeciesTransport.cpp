// The electron, positron and photon transport phases: one tile loop for all
// three.  Each tile runs its histories as a GPU batch when GPU kinetics is on,
// otherwise in 64-particle chunks on the scoped host transport threads.  Every
// chunk logs its side effects, and they are merged in chunk order after the
// loop, so the replay is deterministic whatever the thread count; the
// collective cap check runs before any of it is committed.

#include "RreaTransportStep.H"
#include "RreaKineticGpu.H"
#include <AMReX_Particle.H>

#if defined(AMREX_USE_OMP) || defined(RREA_USE_HOST_OMP)
#include <omp.h>
#endif

namespace rrea::warpx {

struct ChargedHistoryCall {
    bool is_positron = false;
    template <class Effects, class Id>
    RREA_HOST_DEVICE void operator()(KineticStep const& k, Effects& fx,
        ChargedSoAView const& soa, long ip, Id id, int cpu) const
    {
        k.Charged(fx, soa, ip, id, cpu, is_positron);
    }
};

struct PhotonHistoryCall {
    template <class Effects, class Id>
    RREA_HOST_DEVICE void operator()(KineticStep const& k, Effects& fx,
        ChargedSoAView const& soa, long ip, Id id, int cpu) const
    {
        k.Photon(fx, soa, ip, id, cpu);
    }
};

template <class History>
void TransportStep::TransportSpecies(
    std::string const& species, bool& invalidated, bool& repositioned,
    History history)
{
#if defined(WARPX_DIM_RZ)
    auto& particles = warpx.GetPartContainer().GetParticleContainerFromName(species);
    PreviousRzComponentIndices const previous =
        require_previous_rz_components(particles, species);
    auto const kernel = Kinetic();
    for (WarpXParIter pti(particles, 0); pti.isValid(); ++pti) {
        auto& tile = pti.GetParticleTile();
        auto& attribs = pti.GetAttribs();
        auto prev = [&pti](int component) { return pti.GetAttribs(component).dataPtr(); };
        ChargedSoAView const soa{
            attribs[PIdx::r].dataPtr(), attribs[PIdx::z].dataPtr(),
            attribs[PIdx::w].dataPtr(), attribs[PIdx::ux].dataPtr(),
            attribs[PIdx::uy].dataPtr(), attribs[PIdx::uz].dataPtr(),
            attribs[PIdx::theta].dataPtr(),
            prev(previous.r), prev(previous.z), prev(previous.theta),
            prev(previous.ux), prev(previous.uy), prev(previous.uz),
            particles.HasRealComp(rrea::electron_creation_attribute)
                ? pti.GetAttribs(rrea::electron_creation_attribute).dataPtr() : nullptr};
        long const np = pti.numParticles();
        auto merge = [&](TransportSideEffects& fx) {
            pending_side_effects.AppendGroupsFrom(fx);
            invalidated = invalidated || fx.invalidated;
            repositioned = repositioned || fx.repositioned;
        };
#ifdef RREA_USE_CUDA
        if (KineticGpuEnabled()) {
            auto const tile_data = tile.getParticleTileData();
            RunKineticGpuHistories(kernel, np,
                [=] AMREX_GPU_DEVICE(KineticStep const& k, long ip,
                                     TransportSideEffects& fx) noexcept {
                    auto& idcpu = tile_data.idcpu(static_cast<int>(ip));
                    amrex::ParticleIDWrapper<> id(idcpu);
                    amrex::ParticleCPUWrapper cpu(idcpu);
                    history(k, fx, soa, ip, id, static_cast<int>(cpu));
                },
                [&](long, TransportSideEffects& fx) { merge(fx); });
            continue;
        }
#endif
        constexpr long chunk_size = 64;
        long const n_chunks = (np + chunk_size - 1) / chunk_size;
        std::vector<TransportSideEffects> chunk_fx(
            static_cast<std::size_t>(std::max<long>(n_chunks, 0)),
            TransportSideEffects{coupling.LowEnergyCutoffEv()});
#if defined(AMREX_USE_OMP) || defined(RREA_USE_HOST_OMP)
        int const threads = coupling.TransportOmpThreads() > 0
            ? coupling.TransportOmpThreads() : omp_get_max_threads();
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads) if(threads > 1 && n_chunks > 1)
#endif
        for (long chunk = 0; chunk < n_chunks; ++chunk) {
            TransportSideEffects& fx = chunk_fx[static_cast<std::size_t>(chunk)];
            long const end = std::min(np, (chunk + 1) * chunk_size);
            for (long ip = chunk * chunk_size; ip < end; ++ip) {
                history(kernel, fx, soa, ip, tile.id(static_cast<int>(ip)),
                        tile.cpu(static_cast<int>(ip)));
            }
        }
        for (auto& fx : chunk_fx) {
            merge(fx);
        }
    }
#else
    amrex::ignore_unused(species, invalidated, repositioned, history);
    amrex::Abort("RREA production transport requires WarpX RZ");
#endif
}

void TransportStep::ApplyElectronTransport()
{
    TransportSpecies(coupling.ElectronSpeciesName(), invalidated_particles,
                     repositioned_electrons, ChargedHistoryCall{false});
}

void TransportStep::ApplyPositronTransport()
{
    TransportSpecies(coupling.PositronSpeciesName(), invalidated_positrons,
                     repositioned_positrons, ChargedHistoryCall{true});
}

void TransportStep::ApplyPhotonTransport()
{
    TransportSpecies(coupling.PhotonSpeciesName(), invalidated_photons,
                     repositioned_photons, PhotonHistoryCall{});
}

}  // namespace rrea::warpx
