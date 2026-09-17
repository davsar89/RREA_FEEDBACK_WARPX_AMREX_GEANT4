// Exercises the exact production planner; CUDA builds use its real support dispatcher.
#include "rrea/RreaGpuSmoke.H"
#include "rrea/RreaSmokeRequire.H"
#include "rrea/RreaCicConservingResample.H"
#include "RreaRng.H"
#ifdef RREA_USE_CUDA
#  include "RreaPopulationGpu.H"
#endif
#include <array>
#include <iostream>
#include <map>
#include <vector>
namespace {
using namespace rrea;
using namespace rrea::warpx;
using namespace rrea::smoke;
struct Stream {
    RREA_HOST_DEVICE double operator()(std::uint64_t group,std::size_t reduction)const {
        return RreaAdaptiveResampleCicUniform(424242,group,42,reduction,RreaAdaptiveResampleSpecies::Electron);
    }
};
std::array<long double,4> Moments(std::vector<RreaGlobalCicParticle> const& input,
                                std::size_t a,std::size_t b,std::map<std::uint64_t,double> const* after=nullptr) {
    std::array<long double,4> m{};
    for(std::size_t i=a;i<b;++i){auto const& p=input[i];
        double const w=after?after->at(p.stable_id):p.weight;
        m[0]+=w*(1-p.fraction_r)*(1-p.fraction_z);m[1]+=w*p.fraction_r*(1-p.fraction_z);
        m[2]+=w*(1-p.fraction_r)*p.fraction_z;m[3]+=w*p.fraction_r*p.fraction_z;
    }return m;
}
}
int main() {
    rrea::smoke::GpuSession session;
    try {
        constexpr int supports=256;
        std::vector<RreaGlobalCicParticle> input;
        std::array<std::size_t,supports+1> offsets{};
        std::array<std::size_t,supports> capacities{};
        for(int s=0;s<supports;++s){offsets[s]=input.size();
            int const n=17+s%47;
            for(int j=0;j<n;++j){RreaGlobalCicParticle p;
                p.support_r=s;p.support_z=s%19;p.phase=j%3;p.source_rank=j%4;
                p.stable_id=1+input.size();p.candidate=j%2;p.stable_id_high=j%5;
                p.fraction_r=(s%7)?(0.03+0.91*((7*j+3)%67)/66.0):0.2;
                p.fraction_z=(s%11)?(0.04+0.89*((11*j+5)%71)/70.0):0.6;
                p.weight=0.25+((13*j+2)%23)/7.0;input.push_back(p);
            }
            capacities[s]=RreaCicResampleRemovalCapacity(KineticSpan<RreaGlobalCicParticle const>{
                input.data()+offsets[s],input.size()-offsets[s]});
        }
        offsets[supports]=input.size();
        std::map<std::uint64_t,double> weights;
        std::size_t planned=0;
        auto consume=[&](auto const&,KineticSpan<RreaGlobalCicUpdate const> updates){
            ++planned;for(auto const& u:updates){
                require(u.stable_id>0 && u.stable_id<=input.size(),"update identity invalid");
                auto const& old=input[u.stable_id-1];
                require(u.source_rank==old.source_rank && u.candidate==old.candidate
                    && u.stable_id_high==old.stable_id_high,"routing/identity metadata changed");
                require(std::isfinite(u.stored_weight) && u.stored_weight>=0,"invalid planned weight");
                require(weights.emplace(u.stable_id,u.stored_weight).second,"duplicate planned weight");
            }
        };
#ifdef RREA_USE_CUDA
        amrex::ParmParse pp("rrea");pp.add("kinetic_backend",std::string("gpu"));
        pp.add("gpu_population_batch_size",17);pp.add("gpu_population_cpu_fallback",0);
        ConfigureKineticGpu();
        std::vector<PopulationSupportRequest> requests;
        for(int s=0;s<supports;++s)requests.push_back({offsets[s],offsets[s+1],
            capacities[s],capacities[s],offsets[s+1]-offsets[s]-capacities[s]});
        ExecutePopulationSupportPlans(input,requests,4,0,PopulationRandomStream{
            424242,0,42,RreaAdaptiveResampleSpecies::Electron},consume);
#else
        for(int s=0;s<supports;++s){
            KineticVector<RreaGlobalCicParticle> block(input.data()+offsets[s],input.data()+offsets[s+1]);
            auto const plan=RreaPlanGlobalCicThinning<double>(block,block.size()-capacities[s],4,0,Stream{});
            require(plan.status==RreaGlobalCicPlanStatus::Ready,"portable support plan failed");
            require(plan.capacity==capacities[s] && plan.removed>=capacities[s],"support rank/progress mismatch");
            consume(plan,{plan.updates.data(),plan.updates.size()});
        }
#endif
        require(planned==supports && weights.size()==input.size(),"lost support plan/update");
        for(int s=0;s<supports;++s){
            auto before=Moments(input,offsets[s],offsets[s+1]);
            auto after=Moments(input,offsets[s],offsets[s+1],&weights);
            std::size_t count=0;for(auto i=offsets[s];i<offsets[s+1];++i)count+=weights.at(input[i].stable_id)>0;
            require(count<=offsets[s+1]-offsets[s]-capacities[s],"support exceeded its geometric floor");
            for(int c=0;c<4;++c)require(std::abs(before[c]-after[c])<=
                512*std::numeric_limits<double>::epsilon()*(offsets[s+1]-offsets[s])*std::max(1.0L,std::abs(before[c])),
                "one of four deposited cell charges changed");
        }
        std::cout<<"population: "<<supports<<" supports, "<<input.size()<<" records; all four CIC moments preserved\n";
#ifdef RREA_USE_CUDA
        std::cout<<"real CUDA support planner passed (CPU fallback disabled)\n";
#else
        std::cout<<"CPU portable support planner passed (NO CUDA execution)\n";
#endif
        return 0;
    }catch(std::exception const& e){std::cerr<<"population smoke: "<<e.what()<<'\n';return 1;}
}
