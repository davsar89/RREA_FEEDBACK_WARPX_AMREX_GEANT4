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
}
inline void Add(Digest& d,RreaMaterialSource const& p) {
    d.Add(p.kind);
    d.Add(p.level);
    d.Add(p.r_m);
    d.Add(p.z_m);
    d.Add(p.event_weight);
    d.Add(p.secondary_energy_eV);
    d.Add(p.energy_loss_eV);
}
inline void Add(Digest& d,RreaLedgerEntry const& p) {
    d.Add(p.kind);
    d.Add(p.lev);
    d.Add(p.electron_total_competing_tau);
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
    d.Add(p.photon_interaction_count);
    d.Add(p.electron_brems_photon_count);
    d.Add(p.total_brems_photon_count);
    d.Add(p.photon_interaction_weight);
    d.Add(p.electron_brems_photon_weight);
    d.Add(p.electron_brems_photon_energy_eV);
    d.Add(p.total_brems_photon_weight);
    d.Add(p.total_brems_photon_energy_eV);
    d.Add(p.photon_continuous_soft_brems_energy_eV);
    d.Add(p.compton_interaction_count);
    d.Add(p.photoelectric_interaction_count);
    d.Add(p.pair_production_count);
    d.Add(p.pair_nuclear_count);
    d.Add(p.pair_triplet_count);
    d.Add(p.compton_interaction_weight);
    d.Add(p.photoelectric_interaction_weight);
    d.Add(p.pair_production_weight);
    d.Add(p.pair_nuclear_weight);
    d.Add(p.pair_triplet_weight);
    d.Add(p.pair_recoil_electron_energy_eV);
    d.Add(p.pair_local_deposit_energy_eV);
    d.Add(p.photon_seeded_electron_weight);
    d.Add(p.photon_seeded_positron_weight);
    d.Add(p.photon_seeded_kinetic_electron_weight);
    d.Add(p.photon_seeded_low_electron_weight);
    d.Add(p.photon_seeded_kinetic_positron_weight);
    d.Add(p.photon_seeded_below_cutoff_positron_weight);
    d.Add(p.compton_kinetic_electron_weight);
    d.Add(p.compton_low_electron_weight);
    d.Add(p.photoelectric_kinetic_electron_weight);
    d.Add(p.photoelectric_low_electron_weight);
    d.Add(p.pair_kinetic_electron_weight);
    d.Add(p.pair_low_electron_weight);
    d.Add(p.pair_kinetic_positron_weight);
    d.Add(p.pair_below_cutoff_positron_weight);
    d.Add(p.pair_triplet_recoil_kinetic_electron_weight);
    d.Add(p.pair_triplet_recoil_low_electron_weight);
    d.Add(p.photon_absorbed_energy_eV);
    d.Add(p.photon_escaped_energy_eV);
    d.Add(p.photon_to_charged_energy_eV);
    d.Add(p.photon_local_deposit_energy_eV);
    d.Add(p.positron_transport_energy_loss_eV);
    d.Add(p.positron_thermalized_count);
    d.Add(p.positron_bhabha_event_count);
    d.Add(p.positron_bhabha_event_weight);
    d.Add(p.positron_bhabha_secondary_energy_eV);
    d.Add(p.positron_bhabha_kinetic_secondary_weight);
    d.Add(p.positron_brems_event_count);
    d.Add(p.positron_brems_event_weight);
    d.Add(p.positron_brems_photon_energy_eV);
    d.Add(p.transport_ionization_weight);
    d.Add(p.transport_low_secondary_weight);
    d.Add(p.transport_kinetic_secondary_weight);
    d.Add(p.transport_energy_loss_eV);
    d.Add(p.hard_moller_event_count);
    d.Add(p.hard_moller_event_weight);
    d.Add(p.direct_hard_moller_electron_weight);
    d.Add(p.hard_moller_expected_event_weight);
    d.Add(p.hard_moller_secondary_energy_weighted_sum_eV);
    d.Add(p.hard_moller_secondary_energy_weight_sum);
    d.Add(p.hard_moller_secondary_energy_max_eV);
    d.Add(p.parent_drag_loss_eV);
    d.Add(p.ion_pair_source_loss_eV);
    d.Add(p.restricted_collision_loss_eV);
    d.Add(p.explicit_hard_moller_energy_loss_eV);
    d.Add(p.max_transport_hard_moller_tau);
    d.Add(p.max_transport_brems_tau);
    d.Add(p.max_transport_electron_total_competing_tau);
    d.Add(p.max_transport_photon_tau);
    d.Add(p.max_transport_positron_tau);
    d.Add(p.max_transport_subcycles);
    d.Add(p.transport_optical_depth_violation_count);
    d.Add(p.max_transport_electron_energy_eV);
    d.Add(p.max_transport_photon_energy_seen_eV);
    d.Add(p.max_transport_positron_energy_eV);
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
