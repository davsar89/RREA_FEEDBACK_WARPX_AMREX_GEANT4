#ifndef RREA_KINETIC_GPU_H
#define RREA_KINETIC_GPU_H
#include "RreaKineticStep.H"
#include "RreaKineticRecords.H"
#include "RreaSecondaryValidation.H"
#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <string>
#include <limits>
#include <AMReX_Print.H>
#ifdef RREA_USE_CUDA
#include <AMReX_Gpu.H>
#include <AMReX_Arena.H>
#include <AMReX_GpuContainers.H>
#include <cuda_runtime.h>
#endif
namespace rrea::warpx {
// Both backends are supported. A CUDA build defaults to GPU kinetics; the
// production GH200 layout sets rrea.kinetic_backend=cpu (host kinetics, GPU
// fluid/Maxwell).
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
struct KineticGpuOptions {
    int batch=8192, arena_kib=13, arena_audit=0, arena_spill=1;
    std::size_t heap_bytes=1024ULL*1024ULL*1024ULL;
};
inline KineticGpuOptions ConfigureKineticGpu() {
    KineticGpuOptions result;
    int heap_mib=1024;
    amrex::ParmParse pp("rrea");
    pp.query("gpu_kinetic_batch_size",result.batch);
    pp.query("gpu_kinetic_heap_mib",heap_mib);
    pp.query("gpu_kinetic_arena_kib",result.arena_kib);
    pp.query("gpu_kinetic_arena_audit",result.arena_audit);
    pp.query("gpu_kinetic_arena_spill",result.arena_spill);
    if (result.arena_spill!=0 && result.arena_spill!=1)
        amrex::Abort("gpu_kinetic_arena_spill must be 0 or 1");
    if (result.arena_kib<0 || result.arena_kib>1048576
        || (result.arena_audit!=0 && result.arena_audit!=1))
        amrex::Abort("invalid kinetic arena size/audit flag");
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
// Explicit staging: DeviceVector uses The_Arena(), which is managed in
// this hybrid application. NonManagedDeviceVector is required here. Keep
// capacity across batches; no CPU initialization dirties device output pages.
template<class T> struct KineticTransferBuffer {
    amrex::Gpu::NonManagedDeviceVector<T> device;
    amrex::Gpu::PinnedVector<T> host;
    void resize(std::size_t n) {
        // Previous batch data are dead: growing need not copy them on device.
        device.clear(); host.clear(); device.resize(n); host.resize(n);
    }
    void upload() {
        if (!host.empty()) amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
            host.begin(),host.end(),device.begin());
    }
    void download() {
        if (!host.empty()) amrex::Gpu::copyAsync(amrex::Gpu::deviceToHost,
            device.begin(),device.end(),host.begin());
    }
};
inline void DestroyDeviceLogs(TransportSideEffects* p,int n) {
    amrex::ParallelFor(n,[=] AMREX_GPU_DEVICE(int i) noexcept {
        p[i].~TransportSideEffects();
    });
    rrea::GpuSynchronize();
}
// Normally construction is fused with advance, destruction with packing.
// Retain device-side cleanup if host allocation/prefixing throws in between.
class KineticDeviceLogBatch {
    TransportSideEffects* p_=nullptr;
    int live_=0;
public:
    explicit KineticDeviceLogBatch(int capacity) {
        p_=static_cast<TransportSideEffects*>(amrex::The_Device_Arena()->alloc(
            static_cast<std::size_t>(capacity)*sizeof(TransportSideEffects)));
    }
    KineticDeviceLogBatch(KineticDeviceLogBatch const&)=delete;
    KineticDeviceLogBatch& operator=(KineticDeviceLogBatch const&)=delete;
    ~KineticDeviceLogBatch() {
        if(live_) DestroyDeviceLogs(p_,live_);
        amrex::The_Device_Arena()->free(p_);
    }
    void live(int n) noexcept { live_=n; }
    TransportSideEffects* data()const{return p_;}
};
// nvcc rejects an extended __device__ lambda inside a function whose template
// arguments include a function-local type, and both the advance kernel and the
// host consumer below are caller lambdas. A named functor carries the same
// captures and the same device body without being an extended lambda.
template<class Advance>
struct KineticAdvanceLaunch {
    KineticStep const* ctx;
    TransportSideEffects* effects;
    KineticRecordCounts* counts;
    KineticHistorySummary* info;
    long begin;
    Advance advance;
    rrea::KineticArena* arenas;
    unsigned char* pool;
    std::size_t stride;
    bool audit, allow_spill;
    AMREX_GPU_DEVICE void operator()(int i) const noexcept {
        auto& arena=arenas[i];
        arena={pool ? pool+static_cast<std::size_t>(i)*stride : nullptr,
               stride,0,0,audit};
        arena.allow_spill=allow_spill;
        rrea::KineticResource resource{&arena};
        auto local=*ctx;
        local.coupling.allocation=resource;
        new (effects+i) TransportSideEffects(
            ctx->coupling.LowEnergyCutoffEv(),resource);
        advance(local,begin+i,effects[i]);
        counts[i]=CountKineticRecords(effects[i]);
        info[i]={effects[i].tally,effects[i].invalidated,effects[i].repositioned,
                 arena.used,arena.allocations,
                 arena.pool_used,arena.spills,arena.spill_bytes};
    }
};
struct KineticPackLaunch {
    TransportSideEffects* effects;
    KineticRecordCounts const* offsets;
    PackedKineticBuffers packed;
    AMREX_GPU_DEVICE void operator()(int i) const noexcept {
        PackKineticRecords(effects[i],offsets[i],packed);
        effects[i].~TransportSideEffects();
    }
};
// Physics executes once; two launches per nonempty batch. Only flat
// runtime-owned records are copied, never pointers into CUDA's device heap.
// Host prefixing preserves history order. All host buffer mutation/allocation
// precedes async work; both joins remain because host replay may touch the
// managed WarpX arrays (including on systems without concurrent UM access).
template<class Advance,class Consume>
void RunKineticGpuHistories(KineticStep const& context,long count,
                           Advance advance,Consume consume) {
    if(count<0)amrex::Abort("negative kinetic history count");
    if(count==0)return;
    auto const options=ConfigureKineticGpu();
    KineticTransferBuffer<KineticStep> state;
    state.resize(1); state.host[0]=context;
    KineticTransferBuffer<KineticRecordCounts> counts,offsets;
    KineticTransferBuffer<KineticHistorySummary> summary;
    KineticTransferBuffer<PackedSecondaryGroup> groups;
    KineticTransferBuffer<RreaSecondaryParticle> particles;
    KineticTransferBuffer<RreaMaterialSource> sources;
    KineticTransferBuffer<RreaLedgerEntry> operations;
    KineticTransferBuffer<DeferredCdElectronPlaneCrossing> planes;
    auto const capacity=static_cast<std::size_t>(
        std::min<long>(options.batch,count));
    auto const stride=static_cast<std::size_t>(options.arena_kib)*1024;
    if (stride && capacity>std::numeric_limits<std::size_t>::max()/stride)
        amrex::Abort("kinetic pool allocation size overflow");
    // Declare pool owners before logs: exception cleanup destroys logs first.
    amrex::Gpu::NonManagedDeviceVector<rrea::KineticArena> arenas(capacity);
    amrex::Gpu::NonManagedDeviceVector<unsigned char> pool(capacity*stride);
    KineticDeviceLogBatch logs(static_cast<int>(capacity));
    auto* effects=logs.data();
    for(long begin=0;begin<count;) {
        int const n=static_cast<int>(std::min<long>(options.batch,count-begin));
        counts.resize(n); offsets.resize(n); summary.resize(n);
        if(begin==0) state.upload();
        amrex::ParallelFor<64>(n,KineticAdvanceLaunch<Advance>{
            state.device.data(),effects,counts.device.data(),
            summary.device.data(),begin,advance,
            arenas.data(),stride ? pool.data() : nullptr,
            stride,options.arena_audit!=0,options.arena_spill!=0});
        logs.live(n);
        counts.download(); summary.download();
        rrea::GpuSynchronize();
        {
            std::size_t high=0, calls=0, bytes=0;
            std::size_t pool_high=0, spills=0, spill_bytes=0;
            for (int i=0;i<n;++i) {
                auto const& h=summary.host[i];
                spills=KineticCheckedAdd(spills,h.arena_spills);
                spill_bytes=KineticCheckedAdd(spill_bytes,h.arena_spill_bytes);
                if (options.arena_audit) {
                    high=std::max(high,h.arena_used);
                    pool_high=std::max(pool_high,h.arena_pool_used);
                    calls=KineticCheckedAdd(calls,h.arena_allocations);
                    bytes=KineticCheckedAdd(bytes,h.arena_used);
                }
            }
            if (options.arena_audit) {
                amrex::Print() << "RREA_KINETIC_ARENA {\"histories\":" << n
                    << ",\"stride\":" << stride << ",\"max_bytes\":" << high
                    << ",\"requests\":" << calls << ",\"bytes\":" << bytes
                    << ",\"pool_high\":" << pool_high << ",\"spills\":" << spills
                    << ",\"spill_bytes\":" << spill_bytes << "}\n";
            } else if (spills) {
                amrex::Print() << "RREA_KINETIC_ARENA_SPILL histories=" << n
                    << " allocations=" << spills
                    << " bytes=" << spill_bytes << "\n";
            }
        }
        KineticRecordCounts total;
        for(int i=0;i<n;++i){offsets.host[i]=total;total=AddKineticCounts(total,counts.host[i]);}
        groups.resize(total.groups); particles.resize(total.particles);
        sources.resize(total.sources); operations.resize(total.operations); planes.resize(total.planes);
        PackedKineticBuffers packed{groups.device.data(),particles.device.data(),sources.device.data(),
            operations.device.data(),planes.device.data()};
        offsets.upload();
        amrex::ParallelFor(n,KineticPackLaunch{effects,offsets.device.data(),packed});
        groups.download(); particles.download(); sources.download(); operations.download(); planes.download();
        rrea::GpuSynchronize();
        logs.live(0);
        PackedKineticBuffers host{groups.host.data(),particles.host.data(),sources.host.data(),
            operations.host.data(),planes.host.data()};
        for(int i=0;i<n;++i){
            auto fx=UnpackKineticRecords(summary.host[i],offsets.host[i],counts.host[i],host);
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
inline std::array<amrex::Long,4> ValidateKineticGroups(
    KineticStep const& context,TransportSideEffects const& fx) {
    std::array<amrex::Long,4> result{0,0,0,0};
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
                         amrex::ReduceOpSum> ops;
        amrex::ReduceData<Long,Long,Long,Long> data(ops);
        using Tuple=typename decltype(data)::Type;
        ops.eval(static_cast<Long>(total.groups),data,
            [=] AMREX_GPU_DEVICE(Long i) noexcept -> Tuple {
                auto const e=ValidateKineticGroup(*k,g[i]);
                return {e[0],e[1],e[2],e[3]};
            });
        auto const sums=data.value();
        result={amrex::get<0>(sums),amrex::get<1>(sums),amrex::get<2>(sums),
                amrex::get<3>(sums)};
        rrea::GpuSynchronize();
        return result;
    }
#endif
    for(auto const& g:fx.groups) {
        auto const e=ValidateKineticGroup(context,g);
        for(int j=0;j<4;++j) {
            if(e[j]>std::numeric_limits<amrex::Long>::max()-result[j])
                amrex::Abort("kinetic validation counter overflow");
            result[j]+=e[j];
        }
    }
    return result;
}
} // namespace rrea::warpx
#endif
