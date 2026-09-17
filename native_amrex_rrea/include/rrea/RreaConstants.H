#ifndef RREA_CONSTANTS_H
#define RREA_CONSTANTS_H

namespace rrea {

inline constexpr double qe = 1.602176634e-19;
inline constexpr double eps0 = 8.8541878128e-12;
inline constexpr double pi = 3.141592653589793238462643383279502884;
inline constexpr double me_c2_eV = 510998.95000;
inline constexpr double c_m_per_s = 299792458.0;
// Loschmidt number density (dry air, 0 C, 1 atm): the density every REDUCED
// fluid mobility/attachment coefficient is quoted at, and the density the
// fluid transport density ratio is relative to.  1 Td = 1e-21 V m^2.
inline constexpr double n_loschmidt_m3 = 2.6868e25;
inline constexpr double townsend_v_m2 = 1.0e-21;
// Dry-air molar mass (CIPM-2007 composition) and the Avogadro constant, so
// the Loschmidt MASS density below is derived rather than quoted a second
// time: rho_L = n_L M / N_A = 1.29230 kg/m^3.  That is the density a REDUCED
// fluid mobility is defined at, and never the density the transport tables
// were generated at (20 C, 1.20479 kg/m^3); the two differ by 7 percent.
inline constexpr double molar_mass_air_kg_per_mol = 28.96546e-3;
inline constexpr double avogadro_per_mol = 6.02214076e23;
inline constexpr double rho_loschmidt_kg_m3 =
    n_loschmidt_m3 * molar_mass_air_kg_per_mol / avogadro_per_mol;
// Mean energy expended per ion pair in dry air.  ICRU Report 31 (1979),
// electron W value for air.  This is one project constant, not an optional
// transport-table property.
inline constexpr double mean_energy_per_ion_pair_air_eV = 33.97;

// Volume of one axisymmetric shell cell bounded by the radial faces r_in and
// r_out:  V = pi (r_out^2 - r_in^2) dz.  Both radii are clamped to the axis,
// so a guard cell straddling r = 0 keeps only its physical part and one
// entirely below the axis has zero volume.  Every RZ caller (event
// accumulator, fluid transport, reduced observables, Maxwell metric, video
// rebinning) passes r_in >= 0 for its valid cells -- geometry.prob_lo[0] = 0
// is a startup contract -- so the clamp only ever acts on guard cells.
[[nodiscard]] constexpr double rz_shell_volume(
    double r_inner_m, double r_outer_m, double dz_m) noexcept
{
    double const inner = r_inner_m > 0.0 ? r_inner_m : 0.0;
    double const outer = r_outer_m > 0.0 ? r_outer_m : 0.0;
    return pi * (outer * outer - inner * inner) * dz_m;
}

}  // namespace rrea

#endif
