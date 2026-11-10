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

// The ledger's state, one list: counts and reals are order-independent sums
// (per-chunk merge and checkpoint rank-count fold); max_transport_subcycles is
// the one running maximum.  The declaration, the fold, the checkpoint registry
// and the test digest all expand these lists.
#define RREA_LEDGER_COUNTS(X) \
    X(photon_interaction_count) X(electron_brems_photon_count) \
    X(compton_interaction_count) X(photoelectric_interaction_count) \
    X(pair_production_count) X(pair_nuclear_count) X(pair_triplet_count) \
    X(positron_thermalized_count) X(positron_bhabha_event_count) \
    X(positron_brems_event_count) X(hard_moller_event_count)
#define RREA_LEDGER_REALS(X) \
    X(photon_interaction_weight) X(total_brems_photon_energy_eV) \
    X(photon_continuous_soft_brems_energy_eV) X(compton_interaction_weight) \
    X(photoelectric_interaction_weight) X(pair_production_weight) \
    X(pair_nuclear_weight) X(pair_triplet_weight) \
    X(pair_recoil_electron_energy_eV) X(pair_local_deposit_energy_eV) \
    X(photon_seeded_electron_weight) X(photon_seeded_positron_weight) \
    X(photon_seeded_kinetic_electron_weight) X(photon_seeded_low_electron_weight) \
    X(photon_seeded_kinetic_positron_weight) \
    X(photon_seeded_below_cutoff_positron_weight) X(photon_absorbed_energy_eV) \
    X(photon_escaped_energy_eV) X(photon_to_charged_energy_eV) \
    X(photon_local_deposit_energy_eV) X(positron_transport_energy_loss_eV) \
    X(transport_ionization_weight) X(transport_low_secondary_weight) \
    X(transport_kinetic_secondary_weight) X(transport_energy_loss_eV)

struct RreaEventLedger {
    // Configuration, not state: the charged cutoff used to classify
    // secondaries, derived from the selected table unless a study overrides it.
    amrex::Real low_energy_cutoff_eV = amrex::Real(0.0);

#define RREA_LEDGER_DECLARE_COUNT(f) std::uint64_t f = 0;
#define RREA_LEDGER_DECLARE_REAL(f) amrex::Real f = amrex::Real(0.0);
    RREA_LEDGER_COUNTS(RREA_LEDGER_DECLARE_COUNT)
    RREA_LEDGER_REALS(RREA_LEDGER_DECLARE_REAL)
#undef RREA_LEDGER_DECLARE_COUNT
#undef RREA_LEDGER_DECLARE_REAL
    std::uint64_t max_transport_subcycles = 0;

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

    RREA_HOST_DEVICE void RecordContinuousSoftBrems(
        amrex::Real event_weight,
        amrex::Real energy_eV)
    {
        if (event_weight > amrex::Real(0.0) && energy_eV > amrex::Real(0.0)) {
            photon_continuous_soft_brems_energy_eV += event_weight * energy_eV;
        }
    }

    RREA_HOST_DEVICE void RecordHardMollerEvent(amrex::Real event_weight)
    {
        if (event_weight > amrex::Real(0.0)) {
            ++hard_moller_event_count;
        }
    }

    RREA_HOST_DEVICE void RecordPhotonInteraction(amrex::Real event_weight)
    {
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
        total_brems_photon_energy_eV += event_weight * photon_energy_eV;
    }

    // A photon-seeded electron is kinetic at or above the charged cutoff.
    RREA_HOST_DEVICE void RecordPhotonSeededElectron(
        amrex::Real event_weight, amrex::Real energy_eV)
    {
        photon_seeded_electron_weight += event_weight;
        if (energy_eV >= low_energy_cutoff_eV) {
            photon_seeded_kinetic_electron_weight += event_weight;
        } else {
            photon_seeded_low_electron_weight += event_weight;
        }
    }

    RREA_HOST_DEVICE void RecordComptonInteraction(
        amrex::Real event_weight,
        amrex::Real photon_energy_eV,
        amrex::Real recoil_energy_eV,
        amrex::Real scattered_photon_energy_eV,
        amrex::Real charged_transfer_energy_eV,
        amrex::Real local_deposit_energy_eV)
    {
        rrea::kinetic::ignore_unused(photon_energy_eV, scattered_photon_energy_eV);
        RecordPhotonInteraction(event_weight);
        ++compton_interaction_count;
        compton_interaction_weight += event_weight;
        if (recoil_energy_eV >= amrex::Real(0.0)) {
            RecordPhotonSeededElectron(event_weight, recoil_energy_eV);
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
        RecordPhotonInteraction(event_weight);
        ++photoelectric_interaction_count;
        photoelectric_interaction_weight += event_weight;
        if (electron_energy_eV >= amrex::Real(0.0)) {
            RecordPhotonSeededElectron(event_weight, electron_energy_eV);
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
        RecordPhotonInteraction(event_weight);
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
            RecordPhotonSeededElectron(event_weight, electron_energy_eV);
        }
        photon_seeded_positron_weight += event_weight;
        if (positron_energy_eV >= low_energy_cutoff_eV) {
            photon_seeded_kinetic_positron_weight += event_weight;
        } else {
            photon_seeded_below_cutoff_positron_weight += event_weight;
        }
        if (triplet_channel) {
            RecordPhotonSeededElectron(event_weight, recoil_electron_energy_eV);
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

    RREA_HOST_DEVICE void RecordPositronBhabhaEvent(amrex::Real event_weight)
    {
        if (event_weight > amrex::Real(0.0)) {
            ++positron_bhabha_event_count;
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
        total_brems_photon_energy_eV += event_weight * photon_energy_eV;
    }

    RREA_HOST_DEVICE void RecordTransportSubcycles(std::uint64_t subcycles)
    {
        max_transport_subcycles = rrea::kinetic::max(max_transport_subcycles, subcycles);
    }
};

}  // namespace rrea

#endif  // RREA_EVENT_LEDGER_H
