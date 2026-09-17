#ifndef RREA_POPULATION_PARTICLES_H
#define RREA_POPULATION_PARTICLES_H
#include "RreaParticleSweep.H"
#include "RreaPopulationRecord.H"
#include "RreaRng.H"
namespace rrea::warpx {
// Namespace-scope POD and functor keep local host plan types out of CUDA
// kernel signatures and avoid nested extended-lambda capture restrictions.
struct PopulationWeightCommit {
    std::uint64_t* idcpu = nullptr;
    amrex::ParticleReal* weight = nullptr;
    amrex::ParticleReal planned_weight = 0;
};
struct ApplyPopulationWeightCommit {
    PopulationWeightCommit const* records = nullptr;
    RREA_HOST_DEVICE void operator()(std::size_t i) const noexcept {
        auto const& record = records[i];
        *record.weight = record.planned_weight;
        if (record.planned_weight == 0) {
            amrex::ParticleIDWrapper<> id(*record.idcpu);
            id.make_invalid();
        }
    }
};
static_assert(std::is_trivially_copyable_v<PopulationWeightCommit>);

struct PopulationLiveRecord {
    rrea::RreaGlobalCicParticle particle;
    std::uint64_t* idcpu=nullptr;
    amrex::ParticleReal* weight=nullptr;
    amrex::Real energy=0;
};
struct BuildPopulationLiveRecord {
    PopulationRecordBuilder record;
    rrea::KineticSpan<std::uint64_t const> pending_removals;
    RREA_HOST_DEVICE bool operator()(RreaLiveParticle const& p,PopulationLiveRecord& out) const noexcept {
        auto const key=p.StableKey();
        auto const it=rrea::kinetic::lower_bound(pending_removals.begin(),pending_removals.end(),key);
        if(it!=pending_removals.end() && *it==key)return false;
        out.particle=record(p.r,p.z,p.ux,p.uy,p.uz,p.theta,p.weight,false,0,p.idcpu);
        out.idcpu=&p.idcpu;out.weight=&p.weight;out.energy=p.KineticEnergyEv();
        return true;
    }
};
struct PopulationSplitRecord {
    amrex::ParticleReal x=0,y=0,z=0,ux=0,uy=0,uz=0,weight=0;
};
struct SplitPopulationParticle {
    double minimum_weight=0;
    RREA_HOST_DEVICE bool operator()(RreaLiveParticle const& p,PopulationSplitRecord& out) const noexcept {
        auto const split=RreaAdaptiveResampleSplitOutcome(p.weight,minimum_weight);
        if(!split.split)return false;
        auto const w=static_cast<amrex::ParticleReal>(split.child_weight);
        p.weight=w;
        out={p.r*std::cos(p.theta),p.r*std::sin(p.theta),p.z,p.ux,p.uy,p.uz,w};
        return true;
    }
};
} // namespace rrea::warpx
#endif
