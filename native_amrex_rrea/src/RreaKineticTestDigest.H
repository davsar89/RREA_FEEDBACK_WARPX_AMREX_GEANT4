// Test-only field-wise serialization. Never memcpy an owning log or compare padding.
#ifndef RREA_KINETIC_TEST_DIGEST_H
#define RREA_KINETIC_TEST_DIGEST_H
#include "RreaKineticRecords.H"
#include "rrea/RreaSmokeRequire.H"
#include <type_traits>
#include <vector>
namespace rrea::smoke::kinetic_test {
using namespace rrea::warpx;
struct Digest {
    std::vector<double> real;
    std::vector<std::uint64_t> integer;
    template<class T> void Add(T v) {
        if constexpr(std::is_floating_point_v<T>) real.push_back(v);
        else integer.push_back(static_cast<std::uint64_t>(v));
    }
};
inline void Add(Digest& d,RreaSecondaryParticle const& p) {
    d.Add(p.species);
    d.Add(p.x_m);
    d.Add(p.y_m);
    d.Add(p.z_m);
    d.Add(p.dir_x);
    d.Add(p.dir_y);
    d.Add(p.dir_z);
    d.Add(p.weight);
    d.Add(p.kinetic_or_photon_energy_eV);
    d.Add(p.birth_elapsed_step_fraction);
    d.Add(p.root_rng_particle_id);
    d.Add(p.transient_lineage);
    d.Add(p.root_species);
    d.Add(p.electron_counted);
}
inline void Add(Digest& d,RreaMaterialSource const& p) {
    d.Add(p.kind);
    d.Add(p.r_m);
    d.Add(p.z_m);
    d.Add(p.event_weight);
    d.Add(p.secondary_energy_eV);
}
inline void Add(Digest& d,RreaLedgerEntry const& p) {
    d.Add(p.kind);
    d.Add(p.u0);
    d.Add(p.u1);
    d.Add(p.b0);
    d.Add(p.b1);
    for(auto v:p.a)d.Add(v);
}
inline void Add(Digest& d,RreaInteractionConservationClosure const& p) {
    d.Add(p.mode);
    d.Add(p.parent_weight);
    d.Add(p.parent_energy_before_eV);
    d.Add(p.parent_energy_after_eV);
    d.Add(p.parent_survives);
    d.Add(p.local_recoil_binding_subcutoff_energy_eV);
    d.Add(p.escaped_energy_eV);
    d.Add(p.escaped_charge_e);
}
inline void Add(Digest& d,RreaEventLedger const& p) {
    d.Add(p.low_energy_cutoff_eV);
#define RREA_DIGEST_LEDGER(f) d.Add(p.f);
    RREA_LEDGER_COUNTS(RREA_DIGEST_LEDGER)
    RREA_LEDGER_REALS(RREA_DIGEST_LEDGER)
#undef RREA_DIGEST_LEDGER
    d.Add(p.max_transport_subcycles);
}
inline void Add(Digest& d,RreaPlaneFluxCrossing const& p) {
    d.Add(p.plane_index);
    d.Add(p.direction);
    d.Add(p.weight);
    d.Add(p.radial_position_m);
    d.Add(p.local_runaway);
    d.Add(p.at_least_one_mev);
}
inline Digest MakeDigest(TransportSideEffects const& fx) {
    Digest d;d.Add(fx.invalidated);d.Add(fx.repositioned);Add(d,fx.tally);
    d.Add(fx.groups.size());d.Add(fx.cd_electron_plane_crossings.size());
    for(auto const& g:fx.groups) {
        d.Add(g.parent_key);d.Add(g.process);d.Add(g.remove_parent);
        d.Add(g.parent_is_transient);d.Add(g.parent_still_valid_at_preflight);d.Add(g.parent_species);
        Add(d,g.conservation);
        d.Add(g.particles.size());d.Add(g.material_sources.size());d.Add(g.ledger_entries.size());
        for(auto const& p:g.particles)Add(d,p);
        for(auto const& p:g.material_sources)Add(d,p);
        for(auto const& p:g.ledger_entries)Add(d,p);
    }
    for(auto const& p:fx.cd_electron_plane_crossings)Add(d,p);
    return d;
}
inline void Compare(Digest const& actual,Digest const& expected,double tolerance=0) {
    require(actual.integer==expected.integer,"kinetic discrete record mismatch");
    require(actual.real.size()==expected.real.size(),"kinetic real record length mismatch");
    for(std::size_t i=0;i<actual.real.size();++i)
        require_close_rel(actual.real[i],expected.real[i],tolerance,
                          "kinetic record real field "+std::to_string(i));
}
} // namespace
#endif
