#ifndef RREA_GPU_SMOKE_H
#define RREA_GPU_SMOKE_H

// Test-only runtime ownership: the same primitive smokes compile without
// AMReX on a CPU, or launch real kernels in the CUDA CMake build. Managed
// allocations must die before the AMReX arenas; no fake CUDA executor is used.
#include "rrea/RreaGpu.H"
#if defined(RREA_USE_CUDA) || defined(RREA_AMREX_FIELD_SOLVER)
#  include <AMReX.H>
#  include <AMReX_ParmParse.H>
#  include <functional>
#endif

namespace rrea::smoke {

#if defined(RREA_USE_CUDA) || defined(RREA_AMREX_FIELD_SOLVER)
inline void InitializeAmrexSmoke(int& argc, char**& argv)
{
    amrex::Initialize(argc, argv, std::function<void()>([] {
#ifdef RREA_USE_CUDA
        amrex::ParmParse pp("amrex");
        if (!pp.contains("the_arena_is_managed")) {
            pp.add("the_arena_is_managed", 1);
        }
#endif
    }));
}
#endif

class GpuSession {
#if defined(RREA_USE_CUDA) || defined(RREA_AMREX_FIELD_SOLVER)
    bool owns_runtime_ = false;
#endif
public:
    GpuSession()
    {
#if defined(RREA_USE_CUDA) || defined(RREA_AMREX_FIELD_SOLVER)
        if (amrex::Initialized()) { return; }
        // Table smokes have their own positional JSON argument. Do not feed
        // that document to AMReX's input-deck parser or modify caller argv.
        int argc = 1;
        char name[] = "rrea_gpu_smoke";
        char* arguments[] = {name, nullptr};
        char** argv = arguments;
        InitializeAmrexSmoke(argc, argv);
        owns_runtime_ = true;
#endif
    }
    ~GpuSession()
    {
#if defined(RREA_USE_CUDA) || defined(RREA_AMREX_FIELD_SOLVER)
        if (owns_runtime_) {
            GpuSynchronize();
            amrex::Finalize();
        }
#endif
    }
    GpuSession(GpuSession const&) = delete;
    GpuSession& operator=(GpuSession const&) = delete;
};

} // namespace rrea::smoke
#endif
