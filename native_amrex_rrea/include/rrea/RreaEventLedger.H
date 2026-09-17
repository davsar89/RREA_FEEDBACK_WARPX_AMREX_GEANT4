#ifndef RREA_EVENT_LEDGER_H
#define RREA_EVENT_LEDGER_H

// Per-step event ledger: the scalar charge and energy bookkeeping every
// discrete transport event writes into, and the reduced-diagnostics columns
// read back out of.  Pure arithmetic -- no WarpX, no grids, no MPI -- so it
// can be read, reviewed and pinned on its own.

#include "rrea/RreaConstants.H"

#include <AMReX_REAL.H>
#include "rrea/RreaKineticSupport.H"

#include <algorithm>
#include <cstdint>

namespace rrea {

struct RreaEventLedger {
    // Configuration, not state: the charged cutoff used to classify
    // secondaries, derived from the selected table unless a study overrides it.
    amrex::Real low_energy_cutoff_eV = amrex::Real(0.0);

    std::uint64_t photon_interaction_count = 0;
    std::uint64_t electron_brems_photon_count = 0;
    std::uint64_t total_brems_photon_count = 0;
    amrex::Real photon_interaction_weight = amrex::Real(0.0);
    amrex::Real electron_brems_photon_weight = amrex::Real(0.0);
    amrex::Real electron_brems_photon_energy_eV = amrex::Real(0.0);
    amrex::Real total_brems_photon_weight = amrex::Real(0.0);
    amrex::Real total_brems_photon_energy_eV = amrex::Real(0.0);
    amrex::Real photon_continuous_soft_brems_energy_eV = amrex::Real(0.0);
    std::uint64_t compton_interaction_count = 0;
    std::uint64_t photoelectric_interaction_count = 0;
    std::uint64_t pair_production_count = 0;
    std::uint64_t pair_nuclear_count = 0;
    std::uint64_t pair_triplet_count = 0;
    amrex::Real compton_interaction_weight = amrex::Real(0.0);
    amrex::Real photoelectric_interaction_weight = amrex::Real(0.0);
    amrex::Real pair_production_weight = amrex::Real(0.0);
    amrex::Real pair_nuclear_weight = amrex::Real(0.0);
    amrex::Real pair_triplet_weight = amrex::Real(0.0);
    amrex::Real pair_recoil_electron_energy_eV = amrex::Real(0.0);
    amrex::Real pair_local_deposit_energy_eV = amrex::Real(0.0);
    amrex::Real photon_seeded_electron_weight = amrex::Real(0.0);
    amrex::Real photon_seeded_positron_weight = amrex::Real(0.0);
    amrex::Real photon_seeded_kinetic_electron_weight = amrex::Real(0.0);
    amrex::Real photon_seeded_low_electron_weight = amrex::Real(0.0);
    amrex::Real photon_seeded_kinetic_positron_weight = amrex::Real(0.0);
    amrex::Real photon_seeded_below_cutoff_positron_weight = amrex::Real(0.0);
    amrex::Real compton_kinetic_electron_weight = amrex::Real(0.0);
    amrex::Real compton_low_electron_weight = amrex::Real(0.0);
    amrex::Real photoelectric_kinetic_electron_weight = amrex::Real(0.0);
    amrex::Real photoelectric_low_electron_weight = amrex::Real(0.0);
    amrex::Real pair_kinetic_electron_weight = amrex::Real(0.0);
    amrex::Real pair_low_electron_weight = amrex::Real(0.0);
    amrex::Real pair_kinetic_positron_weight = amrex::Real(0.0);
    amrex::Real pair_below_cutoff_positron_weight = amrex::Real(0.0);
    amrex::Real pair_triplet_recoil_kinetic_electron_weight = amrex::Real(0.0);
    amrex::Real pair_triplet_recoil_low_electron_weight = amrex::Real(0.0);
    amrex::Real photon_absorbed_energy_eV = amrex::Real(0.0);
    amrex::Real photon_escaped_energy_eV = amrex::Real(0.0);
    amrex::Real photon_to_charged_energy_eV = amrex::Real(0.0);
    amrex::Real photon_local_deposit_energy_eV = amrex::Real(0.0);
    amrex::Real positron_transport_energy_loss_eV = amrex::Real(0.0);
    std::uint64_t positron_thermalized_count = 0;
    std::uint64_t positron_bhabha_event_count = 0;
    amrex::Real positron_bhabha_event_weight = amrex::Real(0.0);
    amrex::Real positron_bhabha_secondary_energy_eV = amrex::Real(0.0);
    amrex::Real positron_bhabha_kinetic_secondary_weight = amrex::Real(0.0);
    std::uint64_t positron_brems_event_count = 0;
    amrex::Real positron_brems_event_weight = amrex::Real(0.0);
    amrex::Real positron_brems_photon_energy_eV = amrex::Real(0.0);
    amrex::Real transport_ionization_weight = amrex::Real(0.0);
    amrex::Real transport_low_secondary_weight = amrex::Real(0.0);
    amrex::Real transport_kinetic_secondary_weight = amrex::Real(0.0);
    amrex::Real transport_energy_loss_eV = amrex::Real(0.0);
    std::uint64_t hard_moller_event_count = 0;
    amrex::Real hard_moller_event_weight = amrex::Real(0.0);
    amrex::Real direct_hard_moller_electron_weight = amrex::Real(0.0);
    amrex::Real hard_moller_expected_event_weight = amrex::Real(0.0);
    amrex::Real hard_moller_secondary_energy_weighted_sum_eV = amrex::Real(0.0);
    amrex::Real hard_moller_secondary_energy_weight_sum = amrex::Real(0.0);
    amrex::Real hard_moller_secondary_energy_max_eV = amrex::Real(0.0);
    amrex::Real parent_drag_loss_eV = amrex::Real(0.0);
    amrex::Real ion_pair_source_loss_eV = amrex::Real(0.0);
    amrex::Real restricted_collision_loss_eV = amrex::Real(0.0);
    amrex::Real explicit_hard_moller_energy_loss_eV = amrex::Real(0.0);
    amrex::Real max_transport_hard_moller_tau = amrex::Real(0.0);
    amrex::Real max_transport_brems_tau = amrex::Real(0.0);
    amrex::Real max_transport_electron_total_competing_tau = amrex::Real(0.0);
    amrex::Real max_transport_photon_tau = amrex::Real(0.0);
    amrex::Real max_transport_positron_tau = amrex::Real(0.0);
    std::uint64_t max_transport_subcycles = 0;
    std::uint64_t transport_optical_depth_violation_count = 0;
    amrex::Real max_transport_electron_energy_eV = amrex::Real(0.0);
    amrex::Real max_transport_photon_energy_seen_eV = amrex::Real(0.0);
    amrex::Real max_transport_positron_energy_eV = amrex::Real(0.0);

    RREA_HOST_DEVICE void RecordTransportIonization(
        amrex::Real event_weight,
        amrex::Real energy_loss_eV,
        bool low_energy_secondary,
        bool kinetic_secondary)
    {
        if (event_weight <= amrex::Real(0.0)) {
            return;
        }
        transport_ionization_weight += event_weight;
        transport_energy_loss_eV += event_weight * rrea::kinetic::max(energy_loss_eV, amrex::Real(0.0));
        if (low_energy_secondary) {
            transport_low_secondary_weight += event_weight;
        }
        if (kinetic_secondary) {
            transport_kinetic_secondary_weight += event_weight;
        }
    }

    RREA_HOST_DEVICE void RecordContinuousLossPartition(
        amrex::Real event_weight,
        amrex::Real drag_eV,
        amrex::Real collision_eV,
        amrex::Real hard_moller_eV)
    {
        if (event_weight <= amrex::Real(0.0)) {
            return;
        }
        parent_drag_loss_eV += event_weight * rrea::kinetic::max(drag_eV, amrex::Real(0.0));
        ion_pair_source_loss_eV += event_weight
            * rrea::kinetic::max(collision_eV, amrex::Real(0.0));
        restricted_collision_loss_eV += event_weight
            * rrea::kinetic::max(collision_eV, amrex::Real(0.0));
        explicit_hard_moller_energy_loss_eV += event_weight
            * rrea::kinetic::max(hard_moller_eV, amrex::Real(0.0));
    }

    RREA_HOST_DEVICE void RecordContinuousSoftBrems(
        amrex::Real event_weight,
        amrex::Real energy_eV)
    {
        if (event_weight > amrex::Real(0.0) && energy_eV > amrex::Real(0.0)) {
            photon_continuous_soft_brems_energy_eV += event_weight * energy_eV;
        }
    }

    RREA_HOST_DEVICE void RecordHardMollerExpectation(amrex::Real expected_event_weight)
    {
        if (expected_event_weight > amrex::Real(0.0)) {
            hard_moller_expected_event_weight += expected_event_weight;
        }
    }

    RREA_HOST_DEVICE void RecordHardMollerEvent(
        amrex::Real event_weight,
        amrex::Real secondary_energy_eV)
    {
        if (event_weight <= amrex::Real(0.0)) {
            return;
        }
        ++hard_moller_event_count;
        hard_moller_event_weight += event_weight;
        if (secondary_energy_eV >= low_energy_cutoff_eV) {
            direct_hard_moller_electron_weight += event_weight;
        }
        hard_moller_secondary_energy_weighted_sum_eV +=
            event_weight * rrea::kinetic::max(secondary_energy_eV, amrex::Real(0.0));
        hard_moller_secondary_energy_weight_sum += event_weight;
        hard_moller_secondary_energy_max_eV =
            rrea::kinetic::max(hard_moller_secondary_energy_max_eV, secondary_energy_eV);
    }

    RREA_HOST_DEVICE void RecordPhotonInteraction(
        amrex::Real event_weight,
        amrex::Real photon_energy_eV)
    {
        rrea::kinetic::ignore_unused(photon_energy_eV);
        if (event_weight <= amrex::Real(0.0)) {
            return;
        }
        ++photon_interaction_count;
        photon_interaction_weight += event_weight;
    }

    RREA_HOST_DEVICE void RecordPhotonEscape(
        amrex::Real event_weight,
        amrex::Real photon_energy_eV)
    {
        if (event_weight > amrex::Real(0.0) && photon_energy_eV > amrex::Real(0.0)) {
            photon_escaped_energy_eV += event_weight * photon_energy_eV;
        }
    }

    RREA_HOST_DEVICE void RecordBremsPhoton(
        amrex::Real event_weight,
        amrex::Real photon_energy_eV)
    {
        if (event_weight <= amrex::Real(0.0) || photon_energy_eV <= amrex::Real(0.0)) {
            return;
        }
        ++electron_brems_photon_count;
        ++total_brems_photon_count;
        electron_brems_photon_weight += event_weight;
        electron_brems_photon_energy_eV += event_weight * photon_energy_eV;
        total_brems_photon_weight += event_weight;
        total_brems_photon_energy_eV += event_weight * photon_energy_eV;
    }

    RREA_HOST_DEVICE void RecordComptonInteraction(
        amrex::Real event_weight,
        amrex::Real photon_energy_eV,
        amrex::Real recoil_energy_eV,
        amrex::Real scattered_photon_energy_eV,
        amrex::Real charged_transfer_energy_eV,
        amrex::Real local_deposit_energy_eV)
    {
        rrea::kinetic::ignore_unused(scattered_photon_energy_eV);
        RecordPhotonInteraction(event_weight, photon_energy_eV);
        ++compton_interaction_count;
        compton_interaction_weight += event_weight;
        if (recoil_energy_eV >= amrex::Real(0.0)) {
            photon_seeded_electron_weight += event_weight;
            if (recoil_energy_eV >= low_energy_cutoff_eV) {
                photon_seeded_kinetic_electron_weight += event_weight;
                compton_kinetic_electron_weight += event_weight;
            } else {
                photon_seeded_low_electron_weight += event_weight;
                compton_low_electron_weight += event_weight;
            }
        }
        photon_to_charged_energy_eV += event_weight
            * rrea::kinetic::max(charged_transfer_energy_eV, amrex::Real(0.0));
        photon_local_deposit_energy_eV += event_weight
            * rrea::kinetic::max(local_deposit_energy_eV, amrex::Real(0.0));
        transport_energy_loss_eV += event_weight
            * (rrea::kinetic::max(charged_transfer_energy_eV, amrex::Real(0.0))
               + rrea::kinetic::max(local_deposit_energy_eV, amrex::Real(0.0)));
    }

    RREA_HOST_DEVICE void RecordPhotoelectricInteraction(
        amrex::Real event_weight,
        amrex::Real photon_energy_eV,
        amrex::Real electron_energy_eV,
        amrex::Real binding_energy_eV,
        amrex::Real charged_transfer_energy_eV,
        amrex::Real local_deposit_energy_eV)
    {
        rrea::kinetic::ignore_unused(binding_energy_eV);
        RecordPhotonInteraction(event_weight, photon_energy_eV);
        ++photoelectric_interaction_count;
        photoelectric_interaction_weight += event_weight;
        if (electron_energy_eV >= amrex::Real(0.0)) {
            photon_seeded_electron_weight += event_weight;
            if (electron_energy_eV >= low_energy_cutoff_eV) {
                photon_seeded_kinetic_electron_weight += event_weight;
                photoelectric_kinetic_electron_weight += event_weight;
            } else {
                photon_seeded_low_electron_weight += event_weight;
                photoelectric_low_electron_weight += event_weight;
            }
        }
        photon_absorbed_energy_eV += event_weight * rrea::kinetic::max(photon_energy_eV, amrex::Real(0.0));
        photon_to_charged_energy_eV += event_weight
            * rrea::kinetic::max(charged_transfer_energy_eV, amrex::Real(0.0));
        photon_local_deposit_energy_eV += event_weight
            * rrea::kinetic::max(local_deposit_energy_eV, amrex::Real(0.0));
        // Photoelectric absorption transfers the full photon energy to local
        // charged/relaxation energy. The photoelectron kinetic energy excludes the
        // shell binding energy, so using electron_energy_eV here undercounts the
        // transport energy ledger for shell-resolved events.
        transport_energy_loss_eV += event_weight * rrea::kinetic::max(photon_energy_eV, amrex::Real(0.0));
    }

    RREA_HOST_DEVICE void RecordPairProduction(
        amrex::Real event_weight,
        amrex::Real photon_energy_eV,
        amrex::Real electron_energy_eV,
        amrex::Real positron_energy_eV,
        amrex::Real recoil_electron_energy_eV,
        amrex::Real charged_transfer_energy_eV,
        amrex::Real local_deposit_energy_eV,
        bool triplet_channel)
    {
        RecordPhotonInteraction(event_weight, photon_energy_eV);
        ++pair_production_count;
        pair_production_weight += event_weight;
        if (triplet_channel) {
            ++pair_triplet_count;
            pair_triplet_weight += event_weight;
        } else {
            ++pair_nuclear_count;
            pair_nuclear_weight += event_weight;
        }
        if (electron_energy_eV >= amrex::Real(0.0)) {
            photon_seeded_electron_weight += event_weight;
            if (electron_energy_eV >= low_energy_cutoff_eV) {
                photon_seeded_kinetic_electron_weight += event_weight;
                pair_kinetic_electron_weight += event_weight;
            } else {
                photon_seeded_low_electron_weight += event_weight;
                pair_low_electron_weight += event_weight;
            }
        }
        photon_seeded_positron_weight += event_weight;
        if (positron_energy_eV >= low_energy_cutoff_eV) {
            photon_seeded_kinetic_positron_weight += event_weight;
            pair_kinetic_positron_weight += event_weight;
        } else {
            photon_seeded_below_cutoff_positron_weight += event_weight;
            pair_below_cutoff_positron_weight += event_weight;
        }
        if (triplet_channel) {
            photon_seeded_electron_weight += event_weight;
            if (recoil_electron_energy_eV >= low_energy_cutoff_eV) {
                photon_seeded_kinetic_electron_weight += event_weight;
                pair_triplet_recoil_kinetic_electron_weight += event_weight;
            } else {
                photon_seeded_low_electron_weight += event_weight;
                pair_triplet_recoil_low_electron_weight += event_weight;
            }
        }
        pair_recoil_electron_energy_eV += event_weight
            * rrea::kinetic::max(recoil_electron_energy_eV, amrex::Real(0.0));
        pair_local_deposit_energy_eV += event_weight
            * rrea::kinetic::max(local_deposit_energy_eV, amrex::Real(0.0));
        photon_absorbed_energy_eV += event_weight * rrea::kinetic::max(photon_energy_eV, amrex::Real(0.0));
        photon_local_deposit_energy_eV += event_weight
            * rrea::kinetic::max(local_deposit_energy_eV, amrex::Real(0.0));
        photon_to_charged_energy_eV += event_weight
            * rrea::kinetic::max(charged_transfer_energy_eV, amrex::Real(0.0));
        static constexpr amrex::Real electron_rest_energy_eV = amrex::Real(rrea::me_c2_eV);
        transport_energy_loss_eV += event_weight
            * rrea::kinetic::max(
                charged_transfer_energy_eV + local_deposit_energy_eV
                    - amrex::Real(2.0) * electron_rest_energy_eV,
                amrex::Real(0.0));
    }

    RREA_HOST_DEVICE void RecordPositronTransport(
        amrex::Real event_weight,
        amrex::Real energy_loss_eV,
        bool thermalized)
    {
        if (event_weight <= amrex::Real(0.0)) {
            return;
        }
        positron_transport_energy_loss_eV += event_weight
            * rrea::kinetic::max(energy_loss_eV, amrex::Real(0.0));
        if (thermalized) {
            ++positron_thermalized_count;
        }
    }

    RREA_HOST_DEVICE void RecordPositronBhabhaEvent(
        amrex::Real event_weight,
        amrex::Real secondary_electron_energy_eV,
        bool kinetic_secondary)
    {
        if (event_weight <= amrex::Real(0.0)) {
            return;
        }
        ++positron_bhabha_event_count;
        positron_bhabha_event_weight += event_weight;
        positron_bhabha_secondary_energy_eV += event_weight
            * rrea::kinetic::max(secondary_electron_energy_eV, amrex::Real(0.0));
        if (kinetic_secondary) {
            positron_bhabha_kinetic_secondary_weight += event_weight;
        }
    }

    RREA_HOST_DEVICE void RecordPositronBremsPhoton(
        amrex::Real event_weight,
        amrex::Real photon_energy_eV)
    {
        if (event_weight <= amrex::Real(0.0) || photon_energy_eV <= amrex::Real(0.0)) {
            return;
        }
        ++positron_brems_event_count;
        ++total_brems_photon_count;
        positron_brems_event_weight += event_weight;
        positron_brems_photon_energy_eV += event_weight * photon_energy_eV;
        total_brems_photon_weight += event_weight;
        total_brems_photon_energy_eV += event_weight * photon_energy_eV;
    }

    RREA_HOST_DEVICE void RecordTransportGuardMetrics(
        amrex::Real hard_moller_tau,
        amrex::Real brems_tau,
        amrex::Real electron_total_competing_tau,
        amrex::Real photon_tau,
        amrex::Real positron_tau,
        std::uint64_t subcycles,
        amrex::Real electron_energy_eV,
        amrex::Real photon_energy_eV,
        amrex::Real positron_energy_eV,
        bool optical_depth_violation)
    {
        max_transport_hard_moller_tau =
            rrea::kinetic::max(max_transport_hard_moller_tau, hard_moller_tau);
        max_transport_brems_tau = rrea::kinetic::max(max_transport_brems_tau, brems_tau);
        max_transport_electron_total_competing_tau = rrea::kinetic::max(
            max_transport_electron_total_competing_tau,
            electron_total_competing_tau);
        max_transport_photon_tau = rrea::kinetic::max(max_transport_photon_tau, photon_tau);
        max_transport_positron_tau = rrea::kinetic::max(max_transport_positron_tau, positron_tau);
        max_transport_subcycles = rrea::kinetic::max(max_transport_subcycles, subcycles);
        max_transport_electron_energy_eV =
            rrea::kinetic::max(max_transport_electron_energy_eV, electron_energy_eV);
        max_transport_photon_energy_seen_eV =
            rrea::kinetic::max(max_transport_photon_energy_seen_eV, photon_energy_eV);
        max_transport_positron_energy_eV =
            rrea::kinetic::max(max_transport_positron_energy_eV, positron_energy_eV);
        if (optical_depth_violation) {
            ++transport_optical_depth_violation_count;
        }
    }
};

}  // namespace rrea

#endif  // RREA_EVENT_LEDGER_H
