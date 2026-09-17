#include "rrea/RreaKineticSupport.H"
#include "rrea/RreaGpuSmoke.H"
#include "rrea/RreaSmokeRequire.H"
#include "RreaKineticRecords.H"
#ifdef RREA_USE_CUDA
# include "RreaKineticGpu.H"
#endif
#include <iostream>
namespace {
RREA_HOST_DEVICE std::uint64_t Exercise() {
    std::uint64_t errors=0;
    rrea::KineticVector<int> a;
    for(int i=0;i<1025;++i)a.push_back(1024-i);
    rrea::kinetic::sort(a.begin(),a.end());
    for(int i=0;i<1025;++i)if(a[i]!=i)++errors;
    for(int i=-2;i<1028;++i){auto* p=rrea::kinetic::lower_bound(a.begin(),a.end(),i);
        auto const expected=i<0?0:(i>1024?1025:i);
        if(p-a.begin()!=expected)++errors;}
    // Spell the type: nvcc resolves decltype(a) to a reference here, so
    // decltype(a)&& collapses to an lvalue reference and this selects the
    // copy constructor instead of the move it is meant to pin.
    auto copy=a;auto moved=static_cast<rrea::KineticVector<int>&&>(copy);
    if(!copy.empty() || moved.size()!=1025)++errors;
    moved.erase(moved.begin()+1,moved.begin()+1024);
    if(moved.size()!=2 || moved[0]!=0 || moved[1]!=1024)++errors;
    moved.resize(5000,moved.back());
    if(moved.size()!=5000 || moved.back()!=1024)++errors;
    moved.clear();moved.erase(moved.begin(),moved.end());
    if(!moved.empty())++errors;
    rrea::KineticVector<rrea::KineticVector<int>> nested;
    nested.push_back(a);nested.push_back(nested[0]);
    nested.resize(12,nested[0]);
    auto nested_copy=nested; nested.clear();
    if(nested_copy.size()!=12 || nested_copy[11][1024]!=1024)++errors;
    for(int i=0;i<12;++i)nested_copy[i][0]=i;
    for(int i=0;i<12;++i)if(nested_copy[i][0]!=i)++errors;
    auto counts=rrea::warpx::AddKineticCounts({1,2,3,4,5},{2,3,4,5,6});
    if(counts.groups!=3 || counts.particles!=5 || counts.sources!=7 || counts.operations!=9 || counts.planes!=11)++errors;
    return errors;
}
}
int main(){rrea::smoke::GpuSession session;try {
    rrea::smoke::require(Exercise()==0,"CPU nested storage/algorithm checks failed");
#ifdef RREA_USE_CUDA
    rrea::warpx::ConfigureKineticGpu();
    rrea::GpuVector<std::uint64_t> errors(128);
    auto* result=errors.data();
    rrea::GpuFor(rrea::GpuModule::Always, errors.size(),[=] RREA_DEVICE(std::size_t i) noexcept {result[i]=Exercise();});
    for(auto n:errors)rrea::smoke::require(n==0,"CUDA nested storage/algorithm checks failed");
    std::cout<<"real CUDA nested allocations/destructors and algorithms passed\n";
#else
    std::cout<<"CPU nested storage/algorithm checks passed (NO CUDA execution)\n";
#endif
    return 0;
}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}}
