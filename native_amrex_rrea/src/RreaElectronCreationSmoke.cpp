#include "rrea/RreaElectronCreation.H"
#include "rrea/RreaSmokeRequire.H"
#include <cstdio>
#include <iostream>
#include <numeric>
int main() {
    using namespace rrea;
    using namespace rrea::smoke;
    try {
        bool seen=false;
        require_close(ElectronFirstPassage(seen,0.8e6,1.2e6),0.5,"crossing location");
        require(ElectronFirstPassage(seen,0.5e6,2e6)<0,"recrossing counted twice");
        bool clone=seen;
        require(ElectronFirstPassage(clone,2e6,3e6)<0,"counted split counted again");
        seen=false;
        require_close(ElectronFirstPassage(seen,2e6,2e6),0,"energetic birth");
        seen=false;
        require(ElectronFirstPassageInDragSegment(seen,1.01e6,0.04e6,0.04e6,
            0,1,0,1)<0 && !seen,"Strang half-kick overshoot falsely counted");
        require_close(ElectronFirstPassageInDragSegment(seen,1.0e6,0.1e6,0.4e6,
            0,1,0,1),2.0/3,"field minus drag crossing");
        ElectronCreationMap map;
        map.Initialize(25,{0,100,0,100});
        map.Add(5,5,12,false);
        map.Add(5,5,3,false);
        map.Add(5,5,7,true);
        // Splitting an uncounted weight into two disjoint representatives
        // preserves the physical first-passage weight, not the macro count.
        bool a=false,b=false;
        if(ElectronFirstPassage(a,0,2e6)>=0)map.Add(30,30,5,false);
        if(ElectronFirstPassage(b,0,2e6)>=0)map.Add(30,30,5,false);
        require_close(map.weights[0],15,"weighted count");
        require_close(std::accumulate(map.weights.begin(),map.weights.end(),0.),32,
                      "physical weight conservation");
        char const* history = "electron_creation_smoke.bin.gz";
        std::remove(history);
        map.Write(history,1e-3,6000,map.weights,true);
        map.history_bytes = std::filesystem::file_size(history);
        map.Write("electron_creation_smoke.bin",1e-3,6000,map.weights);
        map.Write(history,2e-3,6000,map.weights,true); // uncheckpointed tail
        ElectronCreationMap restored;
        restored.Initialize(25,{0,100,0,100});
        restored.Read("electron_creation_smoke.bin");
        restored.RestoreHistory(history);
        require(std::filesystem::file_size(history)==map.history_bytes,"restart retained a future history tail");
        gzFile file = gzopen(history,"rb");
        std::vector<char> record(80+map.weights.size()*sizeof(double));
        require(file && gzread(file,record.data(),record.size())==static_cast<int>(record.size()),
                "compressed history record did not decode");
        char extra;
        require(gzread(file,&extra,1)==0,"restart duplicated a history record");
        require(gzclose(file)==Z_OK,"invalid compressed history footer");
        std::remove(history);
        for(std::size_t i=0;i<map.weights.size();++i)
            require_close(restored.weights[i],map.weights[i],"checkpoint counts");
        std::remove("electron_creation_smoke.bin");
        require_throws([&]{map.Add(100,0,1,false);},"outside-domain event accepted");
        std::cout<<"electron first passage: birth, acceleration, recrossing, weights, split and checkpoint passed\n";
        return 0;
    } catch(std::exception const& e) {std::cerr<<e.what()<<'\n';return 1;}
}
