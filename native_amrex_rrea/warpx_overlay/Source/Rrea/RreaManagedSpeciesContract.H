#ifndef RREA_MANAGED_SPECIES_CONTRACT_H
#define RREA_MANAGED_SPECIES_CONTRACT_H
#include <AMReX_REAL.H>
#include <cmath>
#include <string>
namespace rrea::warpx {
// Managed species are not aliases for arbitrary WarpX containers.  The RREA
// transport equations assume these exact physical identities, including the
// effective mass and charge after WarpX has parsed all species overrides.
// Keeping the predicate pure gives startup fail-closed behavior focused native
// coverage without constructing a WarpX singleton in the smoke executable.
[[nodiscard]] inline bool RreaManagedSpeciesIdentityMatches(
    std::string const& effective_type,
    amrex::Real effective_mass_kg,
    amrex::Real effective_charge_c,
    std::string const& expected_type,
    amrex::Real expected_mass_kg,
    amrex::Real expected_charge_c) noexcept
{
    return effective_type == expected_type
        && std::isfinite(static_cast<double>(effective_mass_kg))
        && std::isfinite(static_cast<double>(effective_charge_c))
        && effective_mass_kg == expected_mass_kg
        && effective_charge_c == expected_charge_c;
}

[[nodiscard]] inline bool RreaManagedSpeciesNamesAreDistinct(
    std::string const& electron_name,
    std::string const& photon_name,
    std::string const& positron_name) noexcept
{
    return !electron_name.empty()
        && !photon_name.empty()
        && !positron_name.empty()
        && electron_name != photon_name
        && electron_name != positron_name
        && photon_name != positron_name;
}

}
#endif
