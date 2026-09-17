#ifndef RREA_TRANSPORT_LOG_H
#define RREA_TRANSPORT_LOG_H

// Ordered per-chunk transaction log of the transport step: every side effect a
// species loop wants to apply is recorded here first and committed only after
// the collective population/conservation preflight has accepted the whole
// step, so a rank that rejects one interaction group has mutated nothing.
//
// Two kinds of side effect live here, and they are kept apart on purpose.
//   * ORDERED operations (mesh sources, charged segment/escape deposits, and
//     the three photon-interaction records the transaction validator reads)
//     stay as replayed ledger entries: their targets are grid cells or the
//     validator's expected-flow contract, so the deterministic chunk-order
//     replay is part of their definition.
//   * SCALAR sums (rates, weights, energies and running maxima) go straight
//     into `tally`, a per-chunk RreaEventLedger.  Addition and max are
//     associative, so folding chunk tallies in chunk order and the step tally
//     into the coupling ledger once reproduces the serial per-event
//     accumulation up to floating-point summation order.
//
// Pinned by rrea_secondary_group_smoke (group payload/conservation contracts)
// and rrea_warpx_transport_apply (the replay itself).

#include "RreaParticleInteraction.H"
#include "RreaWarpXCoupling.H"

#include "rrea/RreaEventLedger.H"
#include "rrea/RreaPlaneFlux.H"

#include <AMReX.H>
#include <AMReX_REAL.H>

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <limits>
#include <utility>
#include <vector>

namespace rrea::warpx {

// Fold one scalar tally into another.  RreaEventLedger holds nothing but
// order-independent sums apart from the ten running maxima below;
// low_energy_cutoff_eV is configuration, not state, and is never folded.
inline void RreaFoldEventLedger(
    rrea::RreaEventLedger& into, rrea::RreaEventLedger const& from)
{
#define RREA_FOLD_SUM(f) into.f += from.f;
#define RREA_FOLD_MAX(f) into.f = std::max(into.f, from.f);
    RREA_FOLD_SUM(photon_interaction_count)
    RREA_FOLD_SUM(electron_brems_photon_count)
    RREA_FOLD_SUM(total_brems_photon_count)
    RREA_FOLD_SUM(photon_interaction_weight)
    RREA_FOLD_SUM(electron_brems_photon_weight)
    RREA_FOLD_SUM(electron_brems_photon_energy_eV)
    RREA_FOLD_SUM(total_brems_photon_weight)
    RREA_FOLD_SUM(total_brems_photon_energy_eV)
    RREA_FOLD_SUM(photon_continuous_soft_brems_energy_eV)
    RREA_FOLD_SUM(compton_interaction_count)
    RREA_FOLD_SUM(photoelectric_interaction_count)
    RREA_FOLD_SUM(pair_production_count)
    RREA_FOLD_SUM(pair_nuclear_count)
    RREA_FOLD_SUM(pair_triplet_count)
    RREA_FOLD_SUM(compton_interaction_weight)
    RREA_FOLD_SUM(photoelectric_interaction_weight)
    RREA_FOLD_SUM(pair_production_weight)
    RREA_FOLD_SUM(pair_nuclear_weight)
    RREA_FOLD_SUM(pair_triplet_weight)
    RREA_FOLD_SUM(pair_recoil_electron_energy_eV)
    RREA_FOLD_SUM(pair_local_deposit_energy_eV)
    RREA_FOLD_SUM(photon_seeded_electron_weight)
    RREA_FOLD_SUM(photon_seeded_positron_weight)
    RREA_FOLD_SUM(photon_seeded_kinetic_electron_weight)
    RREA_FOLD_SUM(photon_seeded_low_electron_weight)
    RREA_FOLD_SUM(photon_seeded_kinetic_positron_weight)
    RREA_FOLD_SUM(photon_seeded_below_cutoff_positron_weight)
    RREA_FOLD_SUM(compton_kinetic_electron_weight)
    RREA_FOLD_SUM(compton_low_electron_weight)
    RREA_FOLD_SUM(photoelectric_kinetic_electron_weight)
    RREA_FOLD_SUM(photoelectric_low_electron_weight)
    RREA_FOLD_SUM(pair_kinetic_electron_weight)
    RREA_FOLD_SUM(pair_low_electron_weight)
    RREA_FOLD_SUM(pair_kinetic_positron_weight)
    RREA_FOLD_SUM(pair_below_cutoff_positron_weight)
    RREA_FOLD_SUM(pair_triplet_recoil_kinetic_electron_weight)
    RREA_FOLD_SUM(pair_triplet_recoil_low_electron_weight)
    RREA_FOLD_SUM(photon_absorbed_energy_eV)
    RREA_FOLD_SUM(photon_escaped_energy_eV)
    RREA_FOLD_SUM(photon_to_charged_energy_eV)
    RREA_FOLD_SUM(photon_local_deposit_energy_eV)
    RREA_FOLD_SUM(positron_transport_energy_loss_eV)
    RREA_FOLD_SUM(positron_thermalized_count)
    RREA_FOLD_SUM(positron_bhabha_event_count)
    RREA_FOLD_SUM(positron_bhabha_event_weight)
    RREA_FOLD_SUM(positron_bhabha_secondary_energy_eV)
    RREA_FOLD_SUM(positron_bhabha_kinetic_secondary_weight)
    RREA_FOLD_SUM(positron_brems_event_count)
    RREA_FOLD_SUM(positron_brems_event_weight)
    RREA_FOLD_SUM(positron_brems_photon_energy_eV)
    RREA_FOLD_SUM(transport_ionization_weight)
    RREA_FOLD_SUM(transport_low_secondary_weight)
    RREA_FOLD_SUM(transport_kinetic_secondary_weight)
    RREA_FOLD_SUM(transport_energy_loss_eV)
    RREA_FOLD_SUM(hard_moller_event_count)
    RREA_FOLD_SUM(hard_moller_event_weight)
    RREA_FOLD_SUM(direct_hard_moller_electron_weight)
    RREA_FOLD_SUM(hard_moller_expected_event_weight)
    RREA_FOLD_SUM(hard_moller_secondary_energy_weighted_sum_eV)
    RREA_FOLD_SUM(hard_moller_secondary_energy_weight_sum)
    RREA_FOLD_SUM(parent_drag_loss_eV)
    RREA_FOLD_SUM(ion_pair_source_loss_eV)
    RREA_FOLD_SUM(restricted_collision_loss_eV)
    RREA_FOLD_SUM(explicit_hard_moller_energy_loss_eV)
    RREA_FOLD_SUM(transport_optical_depth_violation_count)
    RREA_FOLD_MAX(hard_moller_secondary_energy_max_eV)
    RREA_FOLD_MAX(max_transport_hard_moller_tau)
    RREA_FOLD_MAX(max_transport_brems_tau)
    RREA_FOLD_MAX(max_transport_electron_total_competing_tau)
    RREA_FOLD_MAX(max_transport_photon_tau)
    RREA_FOLD_MAX(max_transport_positron_tau)
    RREA_FOLD_MAX(max_transport_subcycles)
    RREA_FOLD_MAX(max_transport_electron_energy_eV)
    RREA_FOLD_MAX(max_transport_photon_energy_seen_eV)
    RREA_FOLD_MAX(max_transport_positron_energy_eV)
#undef RREA_FOLD_SUM
#undef RREA_FOLD_MAX
}

// Every spawn carries the elapsed outer-step fraction of its creating event
// (default 1.0 = end of step / no residual).  Fixed charged chords convert
// their geometric fraction linearly; photon paths use traveled distance/c.
// The commit pass advances f < 1 newborns through the remaining (1-f)*dt.
// One birth state for every secondary species; the species is the tag passed
// alongside it.  energy_eV is kinetic energy for e-/e+ and total energy for a
// photon, exactly as RreaSecondaryParticle carries it.
struct Spawn {
    amrex::Real x_m = 0.0;
    amrex::Real y_m = 0.0;
    amrex::Real z_m = 0.0;
    amrex::Real dir_x = 0.0;
    amrex::Real dir_y = 0.0;
    amrex::Real dir_z = 1.0;
    amrex::Real weight = 0.0;
    amrex::Real energy_eV = 0.0;
    amrex::Real birth_elapsed_step_fraction = 1.0;
};

using DeferredCdElectronPlaneCrossing = rrea::RreaPlaneFluxCrossing;

// Ordered per-chunk side-effect log for the threaded electron/photon loops.
// One-thread transport applies side effects directly; multi-thread transport
// records per-chunk logs and replays them in chunk order.
struct TransportSideEffects {
    using OpKind = RreaLedgerOpKind;
    using Op = RreaLedgerEntry;

    std::vector<RreaSecondaryGroup> groups;
    std::uint64_t parent_key = 0;
    RreaSecondarySpecies parent_species = RreaSecondarySpecies::Electron;
    bool parent_is_transient = false;
    std::uint64_t parent_context_epoch = 0;
    std::uint64_t continuous_group_epoch = 0;
    std::size_t continuous_group_index =
        std::numeric_limits<std::size_t>::max();
    RreaSecondaryGroup current_group;
    bool group_open = false;
    bool invalidated = false;
    bool repositioned = false;
    // Observation-only crossings are deferred with the physical side effects.
    // The complete batch is previewed (including cumulative overflow) before
    // any physical commit, then installed by a no-fail swap after replay.
    std::vector<DeferredCdElectronPlaneCrossing> cd_electron_plane_crossings;

    // RNG lineage is stamped onto every charged/photon spawn so
    // the newborn residual-advance pass can draw from the parent's streams.
    // parent_key doubles as the parent rng_particle_id (they are the same
    // (id,cpu)-folded value at every transport call site).  The lineage is a
    // per-parent per-step counter over every charged and photon product.
    std::uint64_t spawn_transient_lineage = 0;

    // Scalar tally of this chunk's order-independent ledger sums.  Every rate,
    // weight, energy and running maximum a transport event contributes lands
    // here in place instead of in a replayed ledger entry, so one particle-step
    // adds ONE tally contribution rather than five or more heap-allocated
    // operations per substep.  Chunk tallies fold into the step log in chunk
    // order; the step log folds into the coupling ledger once, at replay.
    rrea::RreaEventLedger tally;

    explicit TransportSideEffects(
        amrex::Real low_energy_cutoff_eV = amrex::Real(0.0))
    {
        tally.low_energy_cutoff_eV = low_energy_cutoff_eV;
    }

    void SetParentContext(
        std::uint64_t key,
        RreaSecondarySpecies species,
        bool transient = false)
    {
        if (group_open) {
            amrex::Abort(
                "RREA secondary transaction changed parent inside an open group");
        }
        if (parent_context_epoch == std::numeric_limits<std::uint64_t>::max()) {
            amrex::Abort("RREA secondary parent-context epoch overflow");
        }
        ++parent_context_epoch;
        parent_key = key;
        parent_species = species;
        parent_is_transient = transient;
        continuous_group_index = std::numeric_limits<std::size_t>::max();
        spawn_transient_lineage = 0;
    }

    void BeginGroup(
        RreaSecondaryProcess process,
        bool remove_parent = false,
        bool parent_still_valid_at_preflight = false)
    {
        if (group_open) {
            amrex::Abort("RREA secondary transaction attempted to nest groups");
        }
        continuous_group_index = std::numeric_limits<std::size_t>::max();
        current_group = RreaSecondaryGroup{};
        current_group.parent_key = parent_key;
        current_group.parent_species = parent_species;
        current_group.process = process;
        current_group.remove_parent = remove_parent;
        current_group.parent_is_transient = parent_is_transient;
        current_group.parent_still_valid_at_preflight =
            parent_still_valid_at_preflight;
        group_open = true;
    }

    void EndGroup()
    {
        if (!group_open) {
            amrex::Abort("RREA secondary transaction ended without an open group");
        }
        groups.push_back(std::move(current_group));
        current_group = RreaSecondaryGroup{};
        group_open = false;
    }

    [[nodiscard]] bool CurrentGroupRemovesParent() const
    {
        if (!group_open) {
            amrex::Abort(
                "RREA secondary transaction queried removal without an open group");
        }
        return current_group.remove_parent;
    }

    void SetInteractionConservationClosure(
        amrex::Real parent_weight,
        amrex::Real parent_energy_before_eV,
        amrex::Real parent_energy_after_eV)
    {
        if (!group_open) {
            amrex::Abort(
                "RREA interaction conservation closure requires an open group");
        }
        auto& closure = current_group.conservation;
        if (closure.mode != RreaConservationClosureMode::Missing) {
            amrex::Abort(
                "RREA interaction conservation closure was supplied more than once");
        }
        closure.mode = RreaConservationClosureMode::Detailed;
        closure.parent_weight = parent_weight;
        closure.parent_energy_before_eV = parent_energy_before_eV;
        closure.parent_survives = !current_group.remove_parent;
        closure.parent_energy_after_eV = closure.parent_survives
            ? parent_energy_after_eV
            : amrex::Real(0.0);
    }

    void SetExplicitZeroConservationClosure()
    {
        if (!group_open) {
            amrex::Abort(
                "RREA zero conservation closure requires an open group");
        }
        auto& closure = current_group.conservation;
        if (closure.mode != RreaConservationClosureMode::Missing) {
            amrex::Abort(
                "RREA interaction conservation closure was supplied more than once");
        }
        closure.mode = RreaConservationClosureMode::ExplicitZero;
    }

    void AddLocalConservationEnergy(
        amrex::Real macro_weight,
        amrex::Real energy_per_particle_eV)
    {
        if (!group_open) {
            amrex::Abort(
                "RREA local conservation energy requires an open interaction group");
        }
        current_group.conservation.local_recoil_binding_subcutoff_energy_eV
            += macro_weight * energy_per_particle_eV;
    }

    void AddEscapedConservationEnergy(
        amrex::Real macro_weight,
        amrex::Real energy_per_particle_eV)
    {
        if (!group_open) {
            amrex::Abort(
                "RREA escaped conservation energy requires an open interaction group");
        }
        current_group.conservation.escaped_energy_eV
            += macro_weight * energy_per_particle_eV;
    }

    void AddEscapedConservationCharge(
        amrex::Real macro_weight,
        amrex::Real signed_charge_e_per_particle)
    {
        if (!group_open) {
            amrex::Abort(
                "RREA escaped conservation charge requires an open interaction group");
        }
        current_group.conservation.escaped_charge_e
            += macro_weight * signed_charge_e_per_particle;
    }

    void SetChargedEscapeConservationClosure(
        amrex::Real parent_weight,
        amrex::Real parent_kinetic_energy_eV)
    {
        if (!group_open
            || current_group.process != RreaSecondaryProcess::ParticleEscape
            || current_group.parent_species == RreaSecondarySpecies::Photon) {
            amrex::Abort(
                "RREA charged escape closure requires an open charged ParticleEscape group");
        }
        // The boundary flux carries the complete charged-particle energy.
        // Parent kinetic energy is recorded separately from rest mass in the
        // closure input, so add rest mass exactly once to the escaped output.
        AddEscapedConservationEnergy(
            parent_weight,
            parent_kinetic_energy_eV
                + RreaSpeciesRestEnergyEv(current_group.parent_species));
        AddEscapedConservationCharge(
            parent_weight,
            RreaSpeciesChargeE(current_group.parent_species));
        SetInteractionConservationClosure(
            parent_weight,
            parent_kinetic_energy_eV,
            amrex::Real(0.0));
    }

    void AppendGroupsFrom(TransportSideEffects& other)
    {
        if (other.group_open) {
            amrex::Abort("RREA secondary transaction merge saw an unfinished group");
        }
        continuous_group_index = std::numeric_limits<std::size_t>::max();
        other.continuous_group_index = std::numeric_limits<std::size_t>::max();
        groups.insert(
            groups.end(),
            std::make_move_iterator(other.groups.begin()),
            std::make_move_iterator(other.groups.end()));
        other.groups.clear();
        cd_electron_plane_crossings.insert(
            cd_electron_plane_crossings.end(),
            std::make_move_iterator(other.cd_electron_plane_crossings.begin()),
            std::make_move_iterator(other.cd_electron_plane_crossings.end()));
        other.cd_electron_plane_crossings.clear();
        RreaFoldEventLedger(tally, other.tally);
    }

    void DeferCdElectronPlaneCrossing(
        std::size_t plane_index,
        int direction,
        amrex::Real weight,
        amrex::Real radial_position_m,
        bool local_runaway,
        bool at_least_one_mev)
    {
        cd_electron_plane_crossings.push_back(
            DeferredCdElectronPlaneCrossing{
                plane_index,
                direction,
                weight,
                radial_position_m,
                local_runaway,
                at_least_one_mev});
    }

    RreaSecondaryGroup& TargetGroup()
    {
        if (group_open) {
            return current_group;
        }
        // Continuous drag/source bookkeeping emits several ordered operations
        // per transport substep.  Keep adjacent operations for one parent in
        // one transaction group instead of allocating an empty heap-backed
        // group for every operation.  An explicit interaction group, parent
        // change, or transient/native boundary starts a new group, so replay
        // and newborn ordering remain exactly the same.
        if (!groups.empty()
            && continuous_group_index == groups.size() - 1U
            && continuous_group_epoch == parent_context_epoch) {
            auto& group = groups[continuous_group_index];
            if (group.process == RreaSecondaryProcess::ContinuousTransport
                && group.parent_key == parent_key
                && group.parent_species == parent_species
                && group.parent_is_transient == parent_is_transient
                && !group.remove_parent) {
                return group;
            }
        }
        groups.emplace_back();
        auto& group = groups.back();
        group.parent_key = parent_key;
        group.parent_species = parent_species;
        group.process = RreaSecondaryProcess::ContinuousTransport;
        group.parent_is_transient = parent_is_transient;
        continuous_group_index = groups.size() - 1U;
        continuous_group_epoch = parent_context_epoch;
        return group;
    }

    void AppendOp(Op op)
    {
        TargetGroup().ledger_entries.push_back(std::move(op));
    }

    void AppendMaterialSource(RreaMaterialSource source, Op op)
    {
        auto& group = TargetGroup();
        group.material_sources.push_back(std::move(source));
        group.ledger_entries.push_back(std::move(op));
    }

    void AddSecondary(RreaSecondarySpecies species, Spawn const& spawn)
    {
        TargetGroup().particles.push_back(RreaSecondaryParticle{
            species,
            spawn.x_m, spawn.y_m, spawn.z_m,
            spawn.dir_x, spawn.dir_y, spawn.dir_z,
            spawn.weight, spawn.energy_eV,
            spawn.birth_elapsed_step_fraction,
            parent_key, spawn_transient_lineage++,
            parent_species});
    }

    void AccumulateIonizationEvent(
        int lev, amrex::Real r_m, amrex::Real z_m, amrex::Real event_weight,
        amrex::Real secondary_energy_eV, amrex::Real energy_loss_eV)
    {
        Op op;
        op.kind = OpKind::AccumulateIonizationEvent;
        op.lev = lev;
        op.a = {r_m, z_m, event_weight, secondary_energy_eV, energy_loss_eV,
                amrex::Real(0.0), amrex::Real(0.0)};
        AppendMaterialSource(
            RreaMaterialSource{RreaMaterialSourceKind::IonizationEvent,
                lev, r_m, z_m, event_weight, secondary_energy_eV, energy_loss_eV},
            std::move(op));
    }
    void AccumulateLowEnergyElectron(
        int lev, amrex::Real r_m, amrex::Real z_m, amrex::Real electron_weight,
        amrex::Real energy_loss_eV)
    {
        Op op;
        op.kind = OpKind::AccumulateLowEnergyElectron;
        op.lev = lev;
        op.a = {r_m, z_m, electron_weight, energy_loss_eV,
                amrex::Real(0.0), amrex::Real(0.0), amrex::Real(0.0)};
        AppendMaterialSource(
            RreaMaterialSource{RreaMaterialSourceKind::LowEnergyElectron,
                lev, r_m, z_m, electron_weight, amrex::Real(0.0), energy_loss_eV},
            std::move(op));
    }
    void AccumulatePositiveIonSourceOnly(
        int lev, amrex::Real r_m, amrex::Real z_m, amrex::Real ion_weight,
        amrex::Real energy_loss_eV)
    {
        Op op;
        op.kind = OpKind::AccumulatePositiveIonSourceOnly;
        op.lev = lev;
        op.a = {r_m, z_m, ion_weight, energy_loss_eV,
                amrex::Real(0.0), amrex::Real(0.0), amrex::Real(0.0)};
        AppendMaterialSource(
            RreaMaterialSource{RreaMaterialSourceKind::PositiveIon,
                lev, r_m, z_m, ion_weight, amrex::Real(0.0), energy_loss_eV},
            std::move(op));
    }
    void RecordComptonInteraction(
        amrex::Real event_weight,
        amrex::Real photon_energy_eV,
        amrex::Real recoil_energy_eV,
        amrex::Real scattered_photon_energy_eV,
        amrex::Real charged_transfer_energy_eV,
        amrex::Real local_deposit_energy_eV)
    {
        Op op;
        op.kind = OpKind::RecordComptonInteraction;
        op.a = {event_weight, photon_energy_eV, recoil_energy_eV,
                scattered_photon_energy_eV, charged_transfer_energy_eV,
                local_deposit_energy_eV, amrex::Real(0.0)};
        AppendOp(std::move(op));
    }
    void RecordPhotoelectricInteraction(
        amrex::Real event_weight,
        amrex::Real photon_energy_eV,
        amrex::Real electron_energy_eV,
        amrex::Real binding_energy_eV,
        amrex::Real charged_transfer_energy_eV,
        amrex::Real local_deposit_energy_eV)
    {
        Op op;
        op.kind = OpKind::RecordPhotoelectricInteraction;
        op.a = {event_weight, photon_energy_eV, electron_energy_eV,
                binding_energy_eV, charged_transfer_energy_eV,
                local_deposit_energy_eV, amrex::Real(0.0)};
        AppendOp(std::move(op));
    }
    void RecordPairProduction(
        amrex::Real event_weight, amrex::Real photon_energy_eV,
        amrex::Real electron_energy_eV, amrex::Real positron_energy_eV,
        amrex::Real recoil_electron_energy_eV,
        amrex::Real charged_transfer_energy_eV,
        amrex::Real local_deposit_energy_eV,
        bool triplet_channel)
    {
        Op op;
        op.kind = OpKind::RecordPairProduction;
        op.a = {event_weight, photon_energy_eV, electron_energy_eV,
                positron_energy_eV, recoil_electron_energy_eV,
                charged_transfer_energy_eV, local_deposit_energy_eV};
        op.b0 = triplet_channel;
        AppendOp(std::move(op));
    }
    void RecordPositronAnnihilation(
        amrex::Real event_weight, amrex::Real local_deposit_energy_eV,
        amrex::Real photon1_energy_eV, amrex::Real photon2_energy_eV,
        bool at_rest_or_below_cutoff)
    {
        Op op;
        op.kind = OpKind::RecordPositronAnnihilation;
        op.a = {event_weight, local_deposit_energy_eV, photon1_energy_eV,
                photon2_energy_eV, amrex::Real(0.0), amrex::Real(0.0),
                amrex::Real(0.0)};
        op.b0 = at_rest_or_below_cutoff;
        AppendOp(std::move(op));
    }
    // One charge-conserving (r,z)
    // segment per charged particle per step -- captured start to committed
    // endpoint / kill point -- plus one per newborn residual leg.  Riding the
    // ordered per-chunk log gives the deposit the SAME deterministic
    // chunk-order merge as every other side effect.  charge_e_weight is the
    // signed charge in elementary charges times the macro weight.
    void AccumulateChargedSegment(
        int lev, amrex::Real r0_m, amrex::Real z0_m,
        amrex::Real r1_m, amrex::Real z1_m,
        amrex::Real charge_e_weight, std::uint64_t channels)
    {
        Op op;
        op.kind = OpKind::AccumulateChargedSegment;
        op.lev = lev;
        op.a = {r0_m, z0_m, r1_m, z1_m, charge_e_weight,
                amrex::Real(0.0), amrex::Real(0.0)};
        op.u0 = channels;
        AppendOp(std::move(op));
    }

    // One atomic escape deposit: the net prefix ends at the actual final
    // microchord's start, that microchord reaches the exact clipped face, and
    // the boundary-cloud closure removes its remaining valid-cell CIC support.
    void AccumulateChargedEscape(
        int lev,
        amrex::Real trajectory_start_r_m,
        amrex::Real trajectory_start_z_m,
        RreaChargedBoundaryCrossing const& crossing,
        amrex::Real charge_e_weight,
        std::uint64_t channels)
    {
        if (!crossing.valid) {
            amrex::Abort(
                "RREA charged escape has no certified absorbing-face crossing");
        }
        Op op;
        op.kind = OpKind::AccumulateChargedEscape;
        op.lev = lev;
        op.a = {
            trajectory_start_r_m,
            trajectory_start_z_m,
            crossing.final_microchord_start_r_m,
            crossing.final_microchord_start_z_m,
            crossing.crossing_r_m,
            crossing.crossing_z_m,
            charge_e_weight};
        op.u0 = channels;
        op.u1 = crossing.faces;
        AppendOp(std::move(op));
    }

    // Replays coupling mutations group-by-group, in the same deterministic
    // parent/process order in which transport assembled them.
    void ReplayInto(RreaWarpXCoupling& coupling) const
    {
        RreaFoldEventLedger(coupling.Ledger(), tally);
        for (auto const& group : groups) {
        for (auto const& op : group.ledger_entries) {
            switch (op.kind) {
            case OpKind::AccumulateIonizationEvent:
                coupling.AccumulateIonizationEvent(
                    op.lev, op.a[0], op.a[1], op.a[2], op.a[3], op.a[4]);
                break;
            case OpKind::AccumulateLowEnergyElectron:
                coupling.AccumulateLowEnergyElectron(
                    op.lev, op.a[0], op.a[1], op.a[2], op.a[3]);
                break;
            case OpKind::AccumulatePositiveIonSourceOnly:
                coupling.AccumulatePositiveIonSourceOnly(
                    op.lev, op.a[0], op.a[1], op.a[2], op.a[3]);
                break;
            case OpKind::RecordComptonInteraction:
                coupling.Ledger().RecordComptonInteraction(
                    op.a[0], op.a[1], op.a[2], op.a[3], op.a[4], op.a[5]);
                break;
            case OpKind::RecordPhotoelectricInteraction:
                coupling.Ledger().RecordPhotoelectricInteraction(
                    op.a[0], op.a[1], op.a[2], op.a[3], op.a[4], op.a[5]);
                break;
            case OpKind::RecordPairProduction:
                coupling.Ledger().RecordPairProduction(
                    op.a[0], op.a[1], op.a[2], op.a[3], op.a[4], op.a[5],
                    op.a[6], op.b0);
                break;
            case OpKind::RecordPositronAnnihilation:
                coupling.RecordPositronAnnihilation(
                    op.a[0], op.a[1], op.a[2], op.a[3], op.b0);
                break;
            case OpKind::AccumulateChargedSegment:
                coupling.AccumulateChargedSegment(
                    op.lev, op.a[0], op.a[1], op.a[2], op.a[3], op.a[4],
                    op.u0);
                break;
            case OpKind::AccumulateChargedEscape:
                coupling.AccumulateChargedEscape(
                    op.lev,
                    op.a[0], op.a[1], op.a[2], op.a[3],
                    op.a[4], op.a[5], op.a[6],
                    op.u0, static_cast<std::uint32_t>(op.u1));
                break;
            }
        }
        }
    }
};

}  // namespace rrea::warpx

#endif
