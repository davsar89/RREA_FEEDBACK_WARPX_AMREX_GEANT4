#ifndef RREA_GPU_H
#define RREA_GPU_H

// Execution/storage shared by the standalone physics primitives and AMReX.
// CUDA builds use managed allocations, not copies hidden behind host pointers.
// Calls are synchronous at this interface because checkpoint, transaction
// replay and MPI callers remain host consumers. Synchronization does not
// itself copy the arrays; the field kernels can reuse their GPU-resident pages.
#include <cstddef>
#include <string>
#include <cmath>
#include <limits>
#include <vector>

// Annotate only when the CUDA execution paths below are actually active.
// A device compiler alone is not enough: CMake source-file properties are
// directory scoped, so a target that never defines RREA_USE_CUDA (the
// AMReX-free RNG/CIC smokes) can still have a shared .cpp handed to nvcc.
// Such a translation unit takes the host fallback in every launcher here,
// and a __device__ lambda called from that host loop is not callable.
#if defined(RREA_USE_CUDA) && (defined(__CUDACC__) || defined(__HIPCC__))
#  define RREA_HOST_DEVICE __host__ __device__
#  define RREA_DEVICE __device__
#else
#  define RREA_HOST_DEVICE
#  define RREA_DEVICE
#endif

#ifdef RREA_USE_CUDA
#  include <AMReX.H>
#  include <AMReX_Arena.H>
#  include <AMReX_Gpu.H>
#  include <AMReX_GpuAllocators.H>
#  include <AMReX_GpuLaunch.H>
#  include <AMReX_Reduce.H>
#  include <AMReX_ParmParse.H>
#endif

namespace rrea {

// Which mesh modules actually execute on the device. Managed-memory migration
// can make a device path a net loss, and the only way to learn which module
// earns its place is to switch them independently and measure. Kernels tagged
// Always (the GPU primitive smokes) are what the device path is tested with,
// so they do not follow the switches.
//   rrea.gpu_maxwell=0|1   field solver          (default 1)
//   rrea.gpu_fluid=0|1     field/fluid advance   (default 1)
// The particle side keeps its own rrea.kinetic_backend.
enum class GpuModule { Maxwell, Fluid, Always };

inline bool GpuModuleOnDevice([[maybe_unused]] GpuModule module)
{
#ifdef RREA_USE_CUDA
    auto const flag = [](char const* name) {
        int value = 1;
        amrex::ParmParse pp("rrea");
        pp.query(name, value);
        if (value != 0 && value != 1) {
            amrex::Abort(std::string(name) + " must be 0 or 1");
        }
        return value == 1;
    };
    // Queried once: ParmParse is fixed after initialization, and these sit in
    // the inner field loops.
    static bool const maxwell = flag("gpu_maxwell");
    static bool const fluid = flag("gpu_fluid");
    switch (module) {
        case GpuModule::Maxwell: return maxwell;
        case GpuModule::Fluid: return fluid;
        case GpuModule::Always: break;
    }
    return true;
#else
    return false;
#endif
}

#ifdef RREA_USE_CUDA
template<class T>
using GpuVector = std::vector<T, amrex::ManagedArenaAllocator<T>>;
#else
template<class T>
using GpuVector = std::vector<T>;
#endif

inline void GpuSynchronize()
{
#ifdef RREA_USE_CUDA
    amrex::Gpu::synchronize();
#endif
}

// Some transaction orchestration, diagnostic and checkpoint callbacks still dereference
// WarpX particle and mesh arrays on the CPU. Reject device-only allocation
// rather than relying on GH200 page accessibility or changing the arena late.
inline void RequireHybridManagedMemory()
{
#ifdef RREA_USE_CUDA
    if (!amrex::The_Arena()->isManaged()) {
        amrex::Abort(
            "RREA hybrid CUDA requires amrex.the_arena_is_managed=1 "
            "on the command line before AMReX initialization; "
            "host transaction orchestration and diagnostics still access shared arrays");
    }
    GpuSynchronize();
#endif
}

template<class F>
void GpuFor2D(GpuModule module, int ilo, int ihi, int jlo, int jhi, F const& f)
{
    if (ihi <= ilo || jhi <= jlo) { return; }
#ifdef RREA_USE_CUDA
    if (GpuModuleOnDevice(module)) {
        auto const ni = static_cast<amrex::Long>(ihi - ilo);
        auto const n = ni * static_cast<amrex::Long>(jhi - jlo);
        amrex::ParallelFor(n, [=] AMREX_GPU_DEVICE(amrex::Long k) noexcept {
            f(ilo + static_cast<int>(k % ni), jlo + static_cast<int>(k / ni));
        });
        GpuSynchronize();
        return;
    }
#endif
    F host_f = f;
    for (int j = jlo; j < jhi; ++j) {
        for (int i = ilo; i < ihi; ++i) { host_f(i, j); }
    }
}

template<class F>
double GpuSum2D(GpuModule module, int ilo, int ihi, int jlo, int jhi, F const& f)
{
    if (ihi <= ilo || jhi <= jlo) { return 0.0; }
#ifdef RREA_USE_CUDA
    if (GpuModuleOnDevice(module)) {
    auto const ni = static_cast<amrex::Long>(ihi - ilo);
    auto const n = ni * static_cast<amrex::Long>(jhi - jlo);
    amrex::ReduceOps<amrex::ReduceOpSum> op;
    amrex::ReduceData<double> data(op);
    using Tuple = typename decltype(data)::Type;
    op.eval(n, data, [=] AMREX_GPU_DEVICE(amrex::Long k) noexcept -> Tuple {
        return {f(ilo + static_cast<int>(k % ni), jlo + static_cast<int>(k / ni))};
    });
    return amrex::get<0>(data.value());
    }
#endif
    F host_f = f;
    double result = 0.0;
    for (int j = jlo; j < jhi; ++j) {
        for (int i = ilo; i < ihi; ++i) { result += host_f(i, j); }
    }
    return result;
}

// One-dimensional helpers share the CPU fallback and stream/host handoff.
template<class F>
void GpuFor(GpuModule module, std::size_t count, F const& f)
{
#ifdef RREA_USE_CUDA
    if (GpuModuleOnDevice(module)) {
        amrex::ParallelFor(static_cast<amrex::Long>(count),
            [=] AMREX_GPU_DEVICE(amrex::Long i) noexcept { f(i); });
        GpuSynchronize();
        return;
    }
#endif
    F host_f = f;
    for (std::size_t i = 0; i < count; ++i) { host_f(i); }
}

template<class F>
double GpuMax(GpuModule module, std::size_t count, F const& f)
{
    if (count == 0) { return 0.0; }
#ifdef RREA_USE_CUDA
    if (GpuModuleOnDevice(module)) {
    amrex::ReduceOps<amrex::ReduceOpMax> op;
    amrex::ReduceData<double> data(op);
    using Tuple = typename decltype(data)::Type;
    op.eval(static_cast<amrex::Long>(count), data,
        [=] AMREX_GPU_DEVICE(amrex::Long i) noexcept -> Tuple {
            double const value = f(i);
            return {std::isnan(value) ? std::numeric_limits<double>::infinity() : value};
        });
    return amrex::get<0>(data.value());
    }
#endif
    F host_f = f;
    double result = -std::numeric_limits<double>::max();
    for (std::size_t i = 0; i < count; ++i) {
        double const value = host_f(i);
        if (std::isnan(value)) { return std::numeric_limits<double>::infinity(); }
        result = result > value ? result : value;
    }
    return result;
}

// Named launchers also let private implementation methods avoid NVCC's
// extended-lambda restriction on taking a private enclosing method address.
template<class T>
void GpuCopy(GpuModule module, T const* source, T* destination, std::size_t count)
{
    GpuFor(module, count, [=] RREA_HOST_DEVICE(std::size_t i) noexcept {
        destination[i] = source[i];
    });
}

template<class T>
void GpuAdd(GpuModule module, T const* a, T const* b, T* destination, std::size_t count)
{
    GpuFor(module, count, [=] RREA_HOST_DEVICE(std::size_t i) noexcept {
        destination[i] = a[i] + b[i];
    });
}

template<class T>
void GpuFill(GpuModule module, T* data, std::size_t count, T value)
{
#ifdef RREA_USE_CUDA
    if (GpuModuleOnDevice(module)) {
        amrex::ParallelFor(static_cast<amrex::Long>(count),
            [=] AMREX_GPU_DEVICE(amrex::Long i) noexcept { data[i] = value; });
        GpuSynchronize();
        return;
    }
#endif
    for (std::size_t i = 0; i < count; ++i) { data[i] = value; }
}

} // namespace rrea
#endif
