#ifndef RREA_KINETIC_RECORDS_H
#define RREA_KINETIC_RECORDS_H
#include "RreaTransportLog.H"
#include <limits>
#include <type_traits>
namespace rrea::warpx {
// Only the packed arrays are runtime-allocated and host accessible. CUDA's
// device-heap pointers MUST NOT be passed to cudaMemcpy or freed on the host.
using PackedSecondaryGroup=BasicSecondaryGroup<rrea::KineticSpan>;
struct KineticRecordCounts {
    std::size_t groups=0,particles=0,sources=0,operations=0,planes=0;
};
struct KineticHistorySummary {
    rrea::RreaEventLedger tally;
    bool invalidated=false,repositioned=false;
};
template<class To,class From>
RREA_HOST_DEVICE void CopySecondaryGroupHeader(To& to,From const& from) {
    to.parent_key=from.parent_key;
    to.process=from.process;
    to.remove_parent=from.remove_parent;
    to.parent_is_transient=from.parent_is_transient;
    to.parent_still_valid_at_preflight=from.parent_still_valid_at_preflight;
    to.parent_species=from.parent_species;
    to.conservation=from.conservation;
}
RREA_HOST_DEVICE inline std::size_t KineticCheckedAdd(std::size_t a,std::size_t b) {
    if(b>std::numeric_limits<std::size_t>::max()-a)
        rrea::KineticFail("kinetic output count overflow");
    return a+b;
}
RREA_HOST_DEVICE inline KineticRecordCounts AddKineticCounts(
    KineticRecordCounts a,KineticRecordCounts const& b) {
    a.groups=KineticCheckedAdd(a.groups,b.groups);
    a.particles=KineticCheckedAdd(a.particles,b.particles);
    a.sources=KineticCheckedAdd(a.sources,b.sources);
    a.operations=KineticCheckedAdd(a.operations,b.operations);
    a.planes=KineticCheckedAdd(a.planes,b.planes);
    return a;
}
template<class Effects>
RREA_HOST_DEVICE KineticRecordCounts CountKineticRecords(Effects const& fx) {
    if(fx.group_open)rrea::KineticFail("unfinished kinetic transaction");
    KineticRecordCounts result;
    result.groups=fx.groups.size();result.planes=fx.cd_electron_plane_crossings.size();
    for(auto const& g:fx.groups) {
        result.particles=KineticCheckedAdd(result.particles,g.particles.size());
        result.sources=KineticCheckedAdd(result.sources,g.material_sources.size());
        result.operations=KineticCheckedAdd(result.operations,g.ledger_entries.size());
    }
    return result;
}
struct PackedKineticBuffers {
    PackedSecondaryGroup* groups=nullptr;
    RreaSecondaryParticle* particles=nullptr;
    RreaMaterialSource* sources=nullptr;
    RreaLedgerEntry* operations=nullptr;
    DeferredCdElectronPlaneCrossing* planes=nullptr;
};
template<class T> RREA_HOST_DEVICE T* KineticOffset(T* p,std::size_t n) {
    return p ? p+n : nullptr;
}
template<class Effects>
RREA_HOST_DEVICE void PackKineticRecords(Effects const& fx,KineticRecordCounts off,
                                       PackedKineticBuffers out) {
    for(auto const& g:fx.groups) {
        auto& packed=out.groups[off.groups++];
        CopySecondaryGroupHeader(packed,g);
        packed.particles={KineticOffset(out.particles,off.particles),g.particles.size()};
        packed.material_sources={KineticOffset(out.sources,off.sources),g.material_sources.size()};
        packed.ledger_entries={KineticOffset(out.operations,off.operations),g.ledger_entries.size()};
        for(auto const& p:g.particles)out.particles[off.particles++]=p;
        for(auto const& p:g.material_sources)out.sources[off.sources++]=p;
        for(auto const& p:g.ledger_entries)out.operations[off.operations++]=p;
    }
    for(auto const& p:fx.cd_electron_plane_crossings)out.planes[off.planes++]=p;
}
inline TransportSideEffects UnpackKineticRecords(KineticHistorySummary const& summary,
    KineticRecordCounts off,KineticRecordCounts count,PackedKineticBuffers packed) {
    TransportSideEffects fx(summary.tally.low_energy_cutoff_eV);
    fx.tally=summary.tally;fx.invalidated=summary.invalidated;fx.repositioned=summary.repositioned;
    fx.groups.reserve(count.groups);
    for(std::size_t i=0;i<count.groups;++i) {
        auto const& g=packed.groups[off.groups+i];
        RreaSecondaryGroup host;
        CopySecondaryGroupHeader(host,g);
        if(!g.particles.empty())host.particles.assign(g.particles.begin(),g.particles.end());
        if(!g.material_sources.empty())host.material_sources.assign(g.material_sources.begin(),g.material_sources.end());
        if(!g.ledger_entries.empty())host.ledger_entries.assign(g.ledger_entries.begin(),g.ledger_entries.end());
        fx.groups.push_back(std::move(host));
    }
    if(count.planes)fx.cd_electron_plane_crossings.assign(
        packed.planes+off.planes,packed.planes+off.planes+count.planes);
    return fx;
}
static_assert(std::is_trivially_copyable_v<PackedSecondaryGroup>);
static_assert(std::is_trivially_copyable_v<RreaSecondaryParticle>);
static_assert(std::is_trivially_copyable_v<RreaMaterialSource>);
static_assert(std::is_trivially_copyable_v<RreaLedgerEntry>);
static_assert(std::is_trivially_copyable_v<KineticHistorySummary>);
} // namespace rrea::warpx
#endif
