#ifndef RREA_POPULATION_GPU_H
#define RREA_POPULATION_GPU_H

#include "RreaKineticGpu.H"
#include "RreaRng.H"
#include "rrea/RreaCicConservingResample.H"
#include <AMReX_Print.H>
#include <vector>

namespace rrea::warpx {
struct PopulationSupportRequest {
    std::size_t begin=0,end=0,request=0,capacity=0,floor=0;
};
struct PopulationPlanResult {
    rrea::RreaGlobalCicPlanStatus status=rrea::RreaGlobalCicPlanStatus::Ready;
    std::size_t capacity=0,floor=0,removed=0,updates=0;
};
struct PopulationRandomStream {
    std::uint64_t seed=0,salt=0,step=0;
    RreaAdaptiveResampleSpecies species=RreaAdaptiveResampleSpecies::Electron;
    RREA_HOST_DEVICE double operator()(std::uint64_t group,std::size_t reduction) const noexcept {
        return RreaAdaptiveResampleCicUniform(seed,group^salt,step,reduction,species);
    }
};
RREA_HOST_DEVICE inline bool PopulationPlanAcceptable(
    PopulationSupportRequest const& request,PopulationPlanResult const& plan) noexcept {
    return plan.status==rrea::RreaGlobalCicPlanStatus::Ready
        && plan.capacity==request.capacity && plan.floor==request.floor
        && plan.removed>=request.request && plan.removed<=request.capacity
        && plan.updates==request.end-request.begin;
}
// The host owns support routing and the global commit barrier. Independent
// support-local constrained reductions execute here. A support is never split
// across kernels: doing so would change the four CIC constraints and RNG order.
// Device binary64 arithmetic lacks the host's optional extended long double.
// A failed rank/moment/progress check is re-evaluated by the original host
// arithmetic, or rejected when gpu_population_cpu_fallback=0. No partial
// device plan is consumed. The fallback is explicit and reported.
// Named functor rather than an extended __device__ lambda: the enclosing
// template's Consume argument is a caller lambda, and nvcc forbids a
// function-local type among the template arguments of a function that
// contains an extended lambda. The device body is unchanged.
struct PopulationPlanLaunch {
    std::size_t const* indices;
    PopulationSupportRequest const* desc;
    rrea::RreaGlobalCicParticle const* particles;
    PopulationPlanResult* result;
    rrea::RreaGlobalCicUpdate* out;
    std::size_t begin;
    int ranks;
    std::uint64_t stream_key;
    PopulationRandomStream rng;
    AMREX_GPU_DEVICE void operator()(int i) const noexcept {
        auto const index=indices[begin+static_cast<std::size_t>(i)];
        auto const r=desc[index];
        if(r.request==0)return;
        // All dynamic arrays are born and destroyed on this device.
        // Only plain updates are copied to the runtime-allocated output.
        rrea::KineticVector<rrea::RreaGlobalCicParticle> support(
            particles+r.begin,particles+r.end);
        auto const plan=rrea::RreaPlanGlobalCicThinning<amrex::ParticleReal>(
            support,support.size()-r.request,ranks,stream_key,rng);
        PopulationPlanResult const summary{plan.status,plan.capacity,
            plan.geometric_floor,plan.removed,plan.updates.size()};
        result[index]=summary;
        if(PopulationPlanAcceptable(r,summary)) {
            for(std::size_t j=0;j<plan.updates.size();++j)
                out[r.begin+j]=plan.updates[j];
        }
    }
};
template<class Consume>
void ExecutePopulationSupportPlans(
    std::vector<rrea::RreaGlobalCicParticle> const& owned,
    std::vector<PopulationSupportRequest> const& requests,
    int ranks,std::uint64_t stream_key,PopulationRandomStream rng,
    Consume consume) {
    auto cpu_plan=[&](PopulationSupportRequest const& r) {
        std::vector<rrea::RreaGlobalCicParticle> input(
            owned.begin()+static_cast<std::ptrdiff_t>(r.begin),
            owned.begin()+static_cast<std::ptrdiff_t>(r.end));
        auto const plan=rrea::RreaPlanGlobalCicThinning<amrex::ParticleReal>(
            input,input.size()-r.request,ranks,stream_key,rng);
        PopulationPlanResult const result{plan.status,plan.capacity,
            plan.geometric_floor,plan.removed,plan.updates.size()};
        if(!PopulationPlanAcceptable(r,result))
            amrex::Abort("adaptive_resample_v1 owner rejected a support plan");
        consume(result,rrea::KineticSpan<rrea::RreaGlobalCicUpdate const>{
            plan.updates.data(),plan.updates.size()});
    };
    for(auto const& r:requests) {
        if(r.begin>r.end || r.end>owned.size() || r.request>r.capacity
           || r.capacity>r.end-r.begin || r.floor!=r.end-r.begin-r.capacity)
            amrex::Abort("adaptive_resample_v1 malformed GPU support descriptor");
    }
#ifdef RREA_USE_CUDA
    if(KineticGpuEnabled() && !requests.empty()) {
        ConfigureKineticGpu();
        int batch=4096,fallback=1;
        amrex::ParmParse pp("rrea");
        pp.query("gpu_population_batch_size",batch);
        pp.query("gpu_population_cpu_fallback",fallback);
        if(batch<1 || batch>65536 || (fallback!=0 && fallback!=1))
            amrex::Abort("invalid GPU population configuration");
        rrea::GpuVector<rrea::RreaGlobalCicParticle> input(owned.begin(),owned.end());
        rrea::GpuVector<PopulationSupportRequest> descriptors(requests.begin(),requests.end());
        rrea::GpuVector<PopulationPlanResult> results(requests.size());
        rrea::GpuVector<rrea::RreaGlobalCicUpdate> updates(owned.size());
        auto const* particles=input.data();auto const* desc=descriptors.data();
        auto* result=results.data();auto* out=updates.data();
        rrea::GpuVector<std::size_t> active;
        for(std::size_t i=0;i<requests.size();++i)if(requests[i].request)active.push_back(i);
        auto const* indices=active.data();
        std::size_t fallback_count=0;
        for(std::size_t begin=0;begin<active.size();) {
            // Bound typical scratch demand as well as thread count. An
            // unusually large single support is never split or truncated.
            int n=0;
            std::size_t particles_in_batch=0;
            while(begin+static_cast<std::size_t>(n)<active.size() && n<batch) {
                auto const& r=requests[active[begin+static_cast<std::size_t>(n)]];
                auto const size=r.end-r.begin;
                if(n && (size>262144 || particles_in_batch>262144-size))break;
                particles_in_batch+=size;++n;
            }
            amrex::ParallelFor<64>(n,PopulationPlanLaunch{
                indices,desc,particles,result,out,begin,ranks,stream_key,rng});
            rrea::GpuSynchronize();
            for(int i=0;i<n;++i) {
                auto const index=active[begin+static_cast<std::size_t>(i)];
                auto const& r=requests[index];
                if(r.request==0)continue;
                if(PopulationPlanAcceptable(r,results[index])) {
                    consume(results[index],rrea::KineticSpan<rrea::RreaGlobalCicUpdate const>{
                        out+r.begin,r.end-r.begin});
                } else {
                    if(!fallback)amrex::Abort(
                        "GPU CIC plan failed rank/moment/progress checks; no weights committed");
                    ++fallback_count;
                    cpu_plan(r);
                }
            }
            begin+=static_cast<std::size_t>(n);
        }
        if(fallback_count)amrex::AllPrint()
            <<"RREA_GPU_POPULATION_FALLBACK supports="<<fallback_count<<"\n";
        return;
    }
#endif
    for(auto const& r:requests)if(r.request)cpu_plan(r);
}
} // namespace rrea::warpx
#endif
