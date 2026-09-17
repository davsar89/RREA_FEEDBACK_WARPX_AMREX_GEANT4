#include "RreaKineticStep.H"
#include "RreaKineticRecords.H"
#include "RreaSecondaryValidation.H"
#include "RreaKineticTestDigest.H"
#include "rrea/RreaInteractionTables.H"
#include "rrea/RreaGpuSmoke.H"
#ifdef RREA_USE_CUDA
#  include "RreaKineticGpu.H"
#endif
#include <array>
#include <iostream>
#include <utility>
#include <vector>

namespace {
using namespace rrea;
using namespace rrea::warpx;
using namespace rrea::smoke;
using namespace rrea::smoke::kinetic_test;
struct TestId {
    long long* value;
    RREA_HOST_DEVICE bool is_valid()const{return *value>0;}
    RREA_HOST_DEVICE void make_invalid(){*value=-std::abs(*value);}
    RREA_HOST_DEVICE operator long long()const{return *value;}
};
struct Particles {
    std::array<GpuVector<double>,13> a;
    GpuVector<long long> ids;
    explicit Particles(std::size_t n):ids(n){for(auto& v:a)v.resize(n);}
    ChargedSoAView view(){return {a[0].data(),a[1].data(),a[2].data(),a[3].data(),
        a[4].data(),a[5].data(),a[6].data(),a[7].data(),a[8].data(),a[9].data(),
        a[10].data(),a[11].data(),a[12].data()};}
};
struct AdvanceLive {
    ChargedSoAView soa;long long* ids;int species;
    RREA_HOST_DEVICE void operator()(KineticStep const& k,long i,
                                     DeviceTransportSideEffects& fx)const {
        if(species==0)k.Electron(fx,soa,i,TestId{ids+i},0);
        else if(species==1)k.Positron(fx,soa,i,TestId{ids+i},0);
        else k.Photon(fx,soa,i,TestId{ids+i},0);
    }
};
struct AdvanceNewborn {
    RreaSecondaryParticle* payload;int* survive;
    RREA_HOST_DEVICE void operator()(KineticStep const& k,long i,
                                     DeviceTransportSideEffects& fx)const {
        auto& p=payload[i];
        survive[i]=p.species==RreaSecondarySpecies::Photon
            ?k.AdvancePhotonNewborn(p,fx):k.AdvanceChargedNewborn(p,fx);
    }
};
std::array<std::size_t,13> processes{};
std::size_t total_groups=0,total_payloads=0,total_ops=0;
void CheckLog(KineticStep const& k,TransportSideEffects const& fx) {
    auto counts=CountKineticRecords(fx);
    // Pack two records with nonzero offsets to exercise segmented output.
    auto total=AddKineticCounts(counts,counts);
    GpuVector<PackedSecondaryGroup> g(total.groups);
    GpuVector<RreaSecondaryParticle> p(total.particles);
    GpuVector<RreaMaterialSource> s(total.sources);
    GpuVector<RreaLedgerEntry> o(total.operations);
    GpuVector<DeferredCdElectronPlaneCrossing> planes(total.planes);
    PackedKineticBuffers buffers{g.data(),p.data(),s.data(),o.data(),planes.data()};
    PackKineticRecords(fx,{},buffers);PackKineticRecords(fx,counts,buffers);
    auto restored=UnpackKineticRecords({fx.tally,fx.invalidated,fx.repositioned},counts,counts,buffers);
    Compare(MakeDigest(restored),MakeDigest(fx));
    for(auto const& group:restored.groups) {
        auto errors=ValidateKineticGroup(k,group);
        for(int j=0;j<5;++j)require(errors[j]==0,"group validation category "+std::to_string(j)+
            " process "+std::to_string(static_cast<int>(group.process)));
        ++processes[static_cast<int>(group.process)];
    }
#ifdef RREA_USE_CUDA
    for (auto const errors : ValidateKineticGroups(k, fx)) {
        require(errors == 0, "device group validator rejected a valid transaction");
    }
#endif
    total_groups+=counts.groups;total_payloads+=counts.particles;total_ops+=counts.operations;
}
void Fill(Particles& p,KineticStep const& k,int species) {
    double const energies[]={1.1e3,6e4,5e5,5e6,5e7,5e8};
    for(std::size_t i=0;i<p.ids.size();++i) {
        auto const e=energies[i%6];double const sign=(i%2)?1:-1;
        double const dx=0.2,dy=0.3,dz=sign*std::sqrt(0.87);
        double const r=300.0,theta=0.35,z=(i%11==0)?sign*799.9:0.0;
        p.ids[i]=static_cast<long long>(10000+species*10000+i);
        p.a[0][i]=p.a[7][i]=r;p.a[1][i]=p.a[8][i]=z;
        p.a[6][i]=p.a[9][i]=theta;p.a[2][i]=1.0+0.125*(i%5);
        double const gamma=1+e/kRreaElectronRestEnergyEv;
        double const magnitude=species==2?e:kRreaSpeedOfLightMPerS*std::sqrt(gamma*gamma-1);
        p.a[3][i]=p.a[10][i]=magnitude*dx;
        p.a[4][i]=p.a[11][i]=magnitude*dy;
        p.a[5][i]=p.a[12][i]=magnitude*dz;
        if(species==2) {
            double const x=r*std::cos(theta)+kRreaSpeedOfLightMPerS*k.dt_s*dx;
            double const y=r*std::sin(theta)+kRreaSpeedOfLightMPerS*k.dt_s*dy;
            p.a[0][i]=std::hypot(x,y);p.a[6][i]=std::atan2(y,x);
            p.a[1][i]=z+kRreaSpeedOfLightMPerS*k.dt_s*dz;
        }
    }
}
void CheckLives(KineticStep const& k) {
    constexpr int n=180;
    for(int species=0;species<3;++species) {
        Particles reference(n);Fill(reference,k,species);
        auto const original=reference;
        std::vector<Digest> expected;
        AdvanceLive cpu{reference.view(),reference.ids.data(),species};
        for(int i=0;i<n;++i) {
            TransportSideEffects fx(k.coupling.low_cutoff);cpu(k,i,fx);
            CheckLog(k,fx);expected.push_back(MakeDigest(fx));
        }
#ifdef RREA_USE_CUDA
        auto device=original;
        AdvanceLive gpu{device.view(),device.ids.data(),species};
        RunKineticGpuHistories(k,n,gpu,[&](long i,auto const& fx) {
            CheckLog(k,fx);Compare(MakeDigest(fx),expected[i],2e-8);
        });
        require(device.ids==reference.ids,"live particle identity mismatch");
        for(std::size_t a=0;a<device.a.size();++a)for(int i=0;i<n;++i)
            require_close_rel(device.a[a][i],reference.a[a][i],2e-8,"live particle state");
#else
        // A second execution verifies that storage and batch scheduling do not
        // feed back into the counter-keyed random stream.
        auto repeated=original;AdvanceLive repeat{repeated.view(),repeated.ids.data(),species};
        for(int i=n-1;i>=0;--i){TransportSideEffects fx(k.coupling.low_cutoff);repeat(k,i,fx);
            Compare(MakeDigest(fx),expected[i]);}
        require(repeated.ids==reference.ids,"reordered history identity mismatch");
        require(repeated.a==reference.a,"reordered history state mismatch");
#endif
    }
}
void CheckNewborns(KineticStep const& k) {
    constexpr int n=120;
    GpuVector<RreaSecondaryParticle> reference(n);
    GpuVector<int> alive(n);
    for(int i=0;i<n;++i){auto& p=reference[i];
        p.species=static_cast<RreaSecondarySpecies>(i%3);p.root_species=p.species;
        p.x_m=250;p.y_m=50;p.z_m=(i%13==0)?799.99:-1;
        p.weight=1.125;p.kinetic_or_photon_energy_eV=(i%5==0)?6e4:5e6;
        p.dir_x=0;p.dir_y=0;p.dir_z=1;
        p.birth_elapsed_step_fraction=0.25*(i%5);
        p.root_rng_particle_id=100000+i;p.transient_lineage=2+i;
    }
    auto const original=reference;
    std::vector<Digest> expected;
    AdvanceNewborn cpu{reference.data(),alive.data()};
    for(int i=0;i<n;++i){TransportSideEffects fx(k.coupling.low_cutoff);cpu(k,i,fx);
        CheckLog(k,fx);expected.push_back(MakeDigest(fx));}
#ifdef RREA_USE_CUDA
    auto device=original;GpuVector<int> gpu_alive(n);
    RunKineticGpuHistories(k,n,AdvanceNewborn{device.data(),gpu_alive.data()},
        [&](long i,auto const& fx){CheckLog(k,fx);Compare(MakeDigest(fx),expected[i],2e-8);});
    require(gpu_alive==alive,"newborn survivor mask mismatch");
    for(int i=0;i<n;++i){Digest a,b;Add(a,device[i]);Add(b,reference[i]);Compare(a,b,2e-8);}
#else
    auto repeated=original;GpuVector<int> repeat_alive(n);
    AdvanceNewborn repeat{repeated.data(),repeat_alive.data()};
    for(int i=n-1;i>=0;--i){TransportSideEffects fx(k.coupling.low_cutoff);repeat(k,i,fx);
        Compare(MakeDigest(fx),expected[i]);Digest a,b;Add(a,repeated[i]);Add(b,reference[i]);Compare(a,b);}
    require(repeat_alive==alive,"reordered newborn survivor mismatch");
#endif
    std::vector<RreaSecondaryParticle> survivors;
    for (int i = 0; i < n; ++i) {
        if (alive[i]) {
            require(KineticFinalSurvivorValid(k, reference[i]),
                    "valid newborn rejected at final insertion");
            survivors.push_back(reference[i]);
        }
    }
    require(!survivors.empty(), "newborn fixture produced no final survivors");
#ifdef RREA_USE_CUDA
    require(ValidateKineticSurvivors(k, survivors) == 0,
            "device survivor validator rejected valid newborns");
#endif
    auto invalid = survivors.front();
    invalid.weight = std::numeric_limits<amrex::Real>::quiet_NaN();
    require(!KineticFinalSurvivorValid(k, invalid), "NaN survivor weight accepted");
    survivors.push_back(invalid);
    invalid = survivors.front();
    invalid.z_m = k.geom.ProbHi(1);
    require(!KineticFinalSurvivorValid(k, invalid), "half-open upper boundary accepted");
    survivors.push_back(invalid);
#ifdef RREA_USE_CUDA
    require(ValidateKineticSurvivors(k, survivors) == 2,
            "device survivor validator did not count invalid payloads");
#endif
}
}
int main(int argc,char** argv) {
    rrea::smoke::GpuSession session;
    if(argc!=2){std::cerr<<"usage: rrea_kinetic_engine_smoke <transport_physics.json>\n";return 2;}
    try {
#ifdef RREA_USE_CUDA
        amrex::ParmParse pp("rrea");pp.add("kinetic_backend",std::string("gpu"));
        pp.add("gpu_kinetic_batch_size",67); // deliberately crosses batch boundaries
        ConfigureKineticGpu();
#endif
        rrea::RreaInteractionTables tables;tables.Load(argv[1]);
        rrea::MaxwellTMRZ field;field.nr=8;field.nz=16;field.dr=100;field.dz=100;field.z_lo=-800;field.resize();
        auto ambient=field;std::fill(ambient.ez.begin(),ambient.ez.end(),-1e4);
        KineticStep k;k.tables=tables.View();k.step=42;k.dt_s=1e-7;k.time_s=4.2e-6;
        k.geom={{0,-800},{800,800}};k.m_config.max_subcycles=4096;
        k.coupling.seed=918273;k.coupling.low_cutoff=5e4;
        k.coupling.moller_cutoff=5e4;k.coupling.photon_cutoff=5e4;
        k.coupling.field.field=std::as_const(field).view();
        k.coupling.field.ambient=std::as_const(ambient).view();
        k.coupling.field.prob_lo_z=-800;k.coupling.field.hi[0]=7;k.coupling.field.hi[1]=15;
        rrea::GpuVector<double> planes{-1,0,1};k.coupling.planes={planes.data(),planes.size()};k.coupling.planes_enabled=true;
        CheckLives(k);CheckNewborns(k);
        require(total_groups>0 && total_payloads>0 && total_ops>0,"transport test produced no transactions");
        std::cout<<"kinetic histories: 540 live + 120 newborn; groups="<<total_groups
                 <<" payloads="<<total_payloads<<" operations="<<total_ops<<"\nprocess counts:";
        for(std::size_t i=0;i<processes.size();++i)std::cout<<' '<<i<<':'<<processes[i];
#ifdef RREA_USE_CUDA
        std::cout<<"\nreal CUDA history/pack/validation parity passed\n";
#else
        std::cout<<"\nCPU shared-history/reordering/pack/validation tests passed (NO CUDA execution)\n";
#endif
        return 0;
    }catch(std::exception const& e){std::cerr<<"kinetic smoke: "<<e.what()<<'\n';return 1;}
}
