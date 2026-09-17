#ifndef RREA_KINETIC_GPU_H
#define RREA_KINETIC_GPU_H
#include "RreaKineticStep.H"
#include "RreaKineticRecords.H"
#include "RreaSecondaryValidation.H"
#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <string>
#include <limits>
#ifdef RREA_USE_CUDA
#include <AMReX_Gpu.H>
#include <AMReX_Arena.H>
#include <cuda_runtime.h>
#endif
namespace rrea::warpx {
// CPU mode in a CUDA build is a diagnostic reference, not a fast hybrid path.
inline bool KineticGpuEnabled() {
#ifdef RREA_USE_CUDA
    std::string backend="gpu";
#else
    std::string backend="cpu";
#endif
    amrex::ParmParse pp("rrea");pp.query("kinetic_backend",backend);
    if(backend=="cpu")return false;
    if(backend!="gpu")amrex::Abort("rrea.kinetic_backend must be cpu or gpu");
#ifdef RREA_USE_CUDA
    return true;
#else
    amrex::Abort("rrea.kinetic_backend=gpu requires a CUDA build");
    return false;
#endif
}
#ifdef RREA_USE_CUDA
struct KineticGpuOptions { int batch=8192;std::size_t heap_bytes=1024ULL*1024ULL*1024ULL; };
inline KineticGpuOptions ConfigureKineticGpu() {
    KineticGpuOptions result;
    int heap_mib=1024;
    amrex::ParmParse pp("rrea");
    pp.query("gpu_kinetic_batch_size",result.batch);
    pp.query("gpu_kinetic_heap_mib",heap_mib);
    if(result.batch<1 || result.batch>1048576 || heap_mib<8)
        amrex::Abort("invalid kinetic batch/heap configuration");
    result.heap_bytes=static_cast<std::size_t>(heap_mib)*1024ULL*1024ULL;
    std::size_t existing=0;
    auto err=cudaDeviceGetLimit(&existing,cudaLimitMallocHeapSize);
    if(err==cudaSuccess && existing<result.heap_bytes)
        err=cudaDeviceSetLimit(cudaLimitMallocHeapSize,result.heap_bytes);
    if(err!=cudaSuccess)amrex::Abort(std::string("kinetic CUDA heap configuration: ")+cudaGetErrorString(err));
    return result;
}
// Named launchers: nvcc refuses an extended __device__ lambda inside a
// function whose address cannot be taken, which is every constructor and
// destructor. The lifetime contract below is unchanged.
inline void ConstructDeviceLogs(DeviceTransportSideEffects* p,int n,amrex::Real cutoff) {
    amrex::ParallelFor(n,[=] AMREX_GPU_DEVICE(int i) noexcept {
        new (p+i) DeviceTransportSideEffects(cutoff);
    });
    rrea::GpuSynchronize();
}
inline void DestroyDeviceLogs(DeviceTransportSideEffects* p,int n) {
    amrex::ParallelFor(n,[=] AMREX_GPU_DEVICE(int i) noexcept {
        p[i].~DeviceTransportSideEffects();
    });
    rrea::GpuSynchronize();
}
// A batch owns device-constructed nested logs. Their destructors MUST execute
// on the device, before the runtime allocation containing them is released.
class KineticDeviceLogBatch {
    DeviceTransportSideEffects* p_=nullptr;
    int n_=0;
public:
    KineticDeviceLogBatch(int n,amrex::Real cutoff):n_(n) {
        p_=static_cast<DeviceTransportSideEffects*>(amrex::The_Device_Arena()->alloc(
            static_cast<std::size_t>(n)*sizeof(DeviceTransportSideEffects)));
        ConstructDeviceLogs(p_,n,cutoff);
    }
    KineticDeviceLogBatch(KineticDeviceLogBatch const&)=delete;
    KineticDeviceLogBatch& operator=(KineticDeviceLogBatch const&)=delete;
    ~KineticDeviceLogBatch() {
        DestroyDeviceLogs(p_,n_);
        amrex::The_Device_Arena()->free(p_);
    }
    DeviceTransportSideEffects* data()const{return p_;}
};
// nvcc rejects an extended __device__ lambda inside a function whose template
// arguments include a function-local type, and both the advance kernel and the
// host consumer below are caller lambdas. A named functor carries the same
// captures and the same device body without being an extended lambda.
template<class Advance>
struct KineticAdvanceLaunch {
    KineticStep const* ctx;
    DeviceTransportSideEffects* effects;
    KineticRecordCounts* counts;
    KineticHistorySummary* info;
    long begin;
    Advance advance;
    AMREX_GPU_DEVICE void operator()(int i) const noexcept {
        advance(*ctx,begin+i,effects[i]);
        counts[i]=CountKineticRecords(effects[i]);
        info[i]={effects[i].tally,effects[i].invalidated,effects[i].repositioned};
    }
};
struct KineticPackLaunch {
    DeviceTransportSideEffects* effects;
    KineticRecordCounts const* offsets;
    PackedKineticBuffers packed;
    AMREX_GPU_DEVICE void operator()(int i) const noexcept {
        PackKineticRecords(effects[i],offsets[i],packed);
    }
};
// Physics executes once, then count/prefix/pack transfers complete variable-
// length transactions without truncating a history. Prefixing these small
// per-history metadata arrays on the host retains stable history order. The
// large record copy and all nested-heap traversal execute on the device.
// Consume is host orchestration: MPI preflight remains collective/fail-closed.
template<class Advance,class Consume>
void RunKineticGpuHistories(KineticStep const& context,long count,
                           Advance advance,Consume consume) {
    if(count<0)amrex::Abort("negative kinetic history count");
    if(count==0)return;
    auto const options=ConfigureKineticGpu();
    rrea::GpuVector<KineticStep> state(1,context);
    auto const* ctx=state.data();
    for(long begin=0;begin<count;) {
        int const n=static_cast<int>(std::min<long>(options.batch,count-begin));
        KineticDeviceLogBatch logs(n,context.coupling.LowEnergyCutoffEv());
        auto* effects=logs.data();
        rrea::GpuVector<KineticRecordCounts> counts(n),offsets(n);
        rrea::GpuVector<KineticHistorySummary> summary(n);
        auto* c=counts.data();auto* info=summary.data();
        // Smaller blocks leave register space for the full collision history.
        // The default batch exposes 128 such blocks; these are starting
        // settings, not a measured GH200 optimum.
        amrex::ParallelFor<64>(n,
            KineticAdvanceLaunch<Advance>{ctx,effects,c,info,begin,advance});
        rrea::GpuSynchronize();
        KineticRecordCounts total;
        for(int i=0;i<n;++i){offsets[i]=total;total=AddKineticCounts(total,counts[i]);}
        rrea::GpuVector<PackedSecondaryGroup> groups(total.groups);
        rrea::GpuVector<RreaSecondaryParticle> particles(total.particles);
        rrea::GpuVector<RreaMaterialSource> sources(total.sources);
        rrea::GpuVector<RreaLedgerEntry> operations(total.operations);
        rrea::GpuVector<DeferredCdElectronPlaneCrossing> planes(total.planes);
        PackedKineticBuffers packed{groups.data(),particles.data(),sources.data(),operations.data(),planes.data()};
        auto const* off=offsets.data();
        amrex::ParallelFor(n,KineticPackLaunch{effects,off,packed});
        rrea::GpuSynchronize();
        for(int i=0;i<n;++i){
            auto fx=UnpackKineticRecords(summary[i],offsets[i],counts[i],packed);
            consume(begin+i,fx);
        }
        begin+=n;
    }
}
#endif // RREA_USE_CUDA
inline amrex::Long ValidateKineticSurvivors(
    KineticStep const& context,std::vector<RreaSecondaryParticle> const& survivors) {
    if(survivors.size()>static_cast<std::size_t>(std::numeric_limits<amrex::Long>::max()))
        amrex::Abort("kinetic final survivor count overflow");
#ifdef RREA_USE_CUDA
    if(KineticGpuEnabled() && !survivors.empty()) {
        rrea::GpuVector<RreaSecondaryParticle> input(survivors.begin(),survivors.end());
        rrea::GpuVector<KineticStep> state(1,context);
        auto const* p=input.data();auto const* k=state.data();
        amrex::ReduceOps<amrex::ReduceOpSum> op;
        amrex::ReduceData<amrex::Long> data(op);
        using Tuple=typename decltype(data)::Type;
        op.eval(static_cast<amrex::Long>(input.size()),data,
            [=] AMREX_GPU_DEVICE(amrex::Long i) noexcept -> Tuple {
                return {KineticFinalSurvivorValid(*k,p[i])?amrex::Long(0):amrex::Long(1)};
            });
        auto const result=amrex::get<0>(data.value());rrea::GpuSynchronize();return result;
    }
#endif
    amrex::Long errors=0;
    for(auto const& p:survivors)if(!KineticFinalSurvivorValid(context,p))++errors;
    return errors;
}
inline std::array<amrex::Long,5> ValidateKineticGroups(
    KineticStep const& context,TransportSideEffects const& fx) {
    std::array<amrex::Long,5> result{0,0,0,0,0};
#ifdef RREA_USE_CUDA
    if(KineticGpuEnabled() && !fx.groups.empty()) {
        auto const total=CountKineticRecords(fx);
        rrea::GpuVector<PackedSecondaryGroup> groups(total.groups);
        rrea::GpuVector<RreaSecondaryParticle> particles(total.particles);
        rrea::GpuVector<RreaMaterialSource> sources(total.sources);
        rrea::GpuVector<RreaLedgerEntry> operations(total.operations);
        rrea::GpuVector<DeferredCdElectronPlaneCrossing> planes(total.planes);
        PackedKineticBuffers packed{groups.data(),particles.data(),sources.data(),operations.data(),planes.data()};
        PackKineticRecords(fx,{},packed);
        rrea::GpuVector<KineticStep> state(1,context);
        auto const* k=state.data();auto const* g=groups.data();
        using Long=amrex::Long;
        amrex::ReduceOps<amrex::ReduceOpSum,amrex::ReduceOpSum,amrex::ReduceOpSum,
                         amrex::ReduceOpSum,amrex::ReduceOpSum> ops;
        amrex::ReduceData<Long,Long,Long,Long,Long> data(ops);
        using Tuple=typename decltype(data)::Type;
        ops.eval(static_cast<Long>(total.groups),data,
            [=] AMREX_GPU_DEVICE(Long i) noexcept -> Tuple {
                auto const e=ValidateKineticGroup(*k,g[i]);
                return {e[0],e[1],e[2],e[3],e[4]};
            });
        auto const sums=data.value();
        result={amrex::get<0>(sums),amrex::get<1>(sums),amrex::get<2>(sums),
                amrex::get<3>(sums),amrex::get<4>(sums)};
        rrea::GpuSynchronize();
        return result;
    }
#endif
    for(auto const& g:fx.groups) {
        auto const e=ValidateKineticGroup(context,g);
        for(int j=0;j<5;++j) {
            if(e[j]>std::numeric_limits<amrex::Long>::max()-result[j])
                amrex::Abort("kinetic validation counter overflow");
            result[j]+=e[j];
        }
    }
    return result;
}
} // namespace rrea::warpx
#endif
