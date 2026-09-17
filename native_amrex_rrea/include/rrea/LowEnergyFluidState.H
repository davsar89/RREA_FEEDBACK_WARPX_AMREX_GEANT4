#ifndef RREA_LOW_ENERGY_FLUID_STATE_H
#define RREA_LOW_ENERGY_FLUID_STATE_H

#include <AMReX_Array.H>
#include <AMReX_Array4.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <string>

namespace rrea {

class LowEnergyElectronClosure;

// The ONE carrier mobility law, shared by the cell-centred conductivity build
// and the face-flux drift:
//   "reduced_stp_density_scaled":  mu = mu_reduced / max(n/n_L, floor)
//   "absolute" (anything else):    mu = mu_absolute
// A REDUCED mobility is defined at Loschmidt (dry air, 0 C, 1 atm), so the
// density_ratio argument must be the Loschmidt-relative one
// (RreaWarpXCoupling::MobilityDensityRatioAtZ), never the transport ratio the
// Monte Carlo uses -- they differ by 7 percent.  The floor bounds the drift
// speed where the profile density goes to zero.  Resolve once per call,
// outside the innermost loops.
struct MobilityLaw {
    bool reduced = false;
    double absolute = 0.0;
    double reduced_mobility = 0.0;
    double density_floor = 1.0;

    [[nodiscard]] double At(double density_ratio) const noexcept
    {
        return reduced
            ? reduced_mobility / std::max(density_ratio, density_floor)
            : absolute;
    }
};

[[nodiscard]] inline MobilityLaw resolve_mobility_law(
    std::string const& model,
    double absolute_mobility,
    double reduced_mobility,
    double density_floor)
{
    return MobilityLaw{
        model == "reduced_stp_density_scaled",
        absolute_mobility,
        reduced_mobility,
        density_floor};
}

// The density ratio at one cell: the bound profile field where there is one,
// else the deck's scalar.  The profile is validated strictly positive at load
// (RreaAltitudeProfile) and the scalar at parse, so a nonpositive or
// non-finite entry is an engine bug, not data to mask; the positive fallback
// is a release backstop so no such value reaches transport.
[[nodiscard]] inline double checked_density_ratio(
    amrex::Array4<amrex::Real const> const& density,
    bool has_density,
    int i,
    int j,
    int k,
    double scalar_density) noexcept
{
    double const value =
        has_density ? static_cast<double>(density(i, j, k)) : scalar_density;
    if (!std::isfinite(value) || value <= 0.0) {
        return scalar_density > 0.0 ? scalar_density : 1.0;
    }
    return value;
}

struct LowEnergyFluidConfig {
    double electron_mobility_m2_per_vs = 0.1;
    double positive_ion_mobility_m2_per_vs = 2.0e-4;
    double negative_ion_mobility_m2_per_vs = 2.0e-4;
    bool ion_drift_enable = false;
    // Low-energy electrons advect so the fluid charge update and the face
    // current consumed by Ampere's law describe the same carrier motion.
    bool electron_drift_enable = true;
    // Electron models: "absolute" (constant mu, the smokes' analytic
    // control) or "en_table_flux_v1", the production closure -- a
    // field-dependent flux mobility mu(E/N) from the hybrid dry-air bundle,
    // supplied by the caller as an evaluated cell field.  There is no
    // density-only production arm: the table owns the E/N dependence.
    std::string electron_mobility_model = "absolute";
    std::string ion_mobility_model = "absolute";
    double positive_ion_reduced_mobility_stp_m2_per_vs = 0.0;
    double negative_ion_reduced_mobility_stp_m2_per_vs = 0.0;
    double ion_mobility_density_ratio_floor = 0.0;
    double ion_drift_cfl = 0.5;
    double transport_density_ratio = 1.0;
    double attachment_frequency_s = 2.0e6;
    // Attachment models: "constant" applies attachment_frequency_s
    // everywhere (the smokes' analytic control); "en_table_attachment_v1" is
    // the production closure and consumes a caller-evaluated nu(E/N, rho)
    // cell field, including its two- and three-body density dependence.
    std::string attachment_model = "constant";
    // Effective room-temperature dry-air chemistry.  Keep electron-ion and
    // ion-ion pair losses separate: they consume different species even
    // though both conserve charge exactly.
    double electron_ion_recombination_coefficient_m3_per_s = 0.0;
    double ion_ion_recombination_coefficient_m3_per_s = 0.0;
    // Collisional detachment n_neg -> n_e + M (charge conserving). Default 0
    // because the active closure has no source-validated detachment rate; the
    // channel remains available for an explicitly justified study value.
    double detachment_frequency_s = 0.0;
};

// Rank-local realized-E/N accounting from one closure cell-field build.
// Callers reduce across ranks and must abort on above_range/invalid before
// committing any state.  Integrated and peak activity measures distinguish
// quiet cells below a table endpoint from active plasma.  The production v3
// endpoint is the measured zero-field asymptote, so holding it is physical;
// the fractions remain useful for tables whose endpoint is not an asymptote.
// The conductivity proxy omits qe because it cancels in held/total ratios.
struct LowEnergyClosureFieldDiagnostics {
    double min_en_td = std::numeric_limits<double>::max();
    double max_en_td = 0.0;
    long held_below_cells = 0;
    long above_range_cells = 0;
    long invalid_cells = 0;
    double max_above_range_en_td = -1.0;
    int max_above_range_i = -1;
    int max_above_range_j = -1;
    double max_above_range_electron_density_m3 = 0.0;
    double max_above_range_source_rate_m3_s = 0.0;
    double max_above_range_conductivity_proxy_per_m_v_s = 0.0;
    double electron_count = 0.0;
    double held_below_electron_count = 0.0;
    double electron_source_rate_per_s = 0.0;
    double held_below_electron_source_rate_per_s = 0.0;
    double electron_conductivity_proxy_m2_per_v_s = 0.0;
    double held_below_electron_conductivity_proxy_m2_per_v_s = 0.0;
    double max_electron_density_m3 = 0.0;
    double max_held_below_electron_density_m3 = 0.0;
    double max_electron_source_rate_m3_s = 0.0;
    double max_held_below_electron_source_rate_m3_s = 0.0;
    double max_electron_conductivity_proxy_per_m_v_s = 0.0;
    double max_held_below_electron_conductivity_proxy_per_m_v_s = 0.0;
};

// Closure evaluation from the complete cell-centered field magnitude
// |E| = hypot(Er, Ez) and the
// Loschmidt-relative density ratio chi, evaluate E/N = |E|*1e21/(chi*N0) and
// write the LOCAL coefficients mu_e = K0(E/N)/chi [m^2/Vs],
// nu_att = nu2_ref(E/N)*chi + nu3_ref(E/N)*chi^2 [1/s], and the RZ tensor
// D = D_T I + (D_L-D_T) b b with b = E/|E| and D_L,D_T ~ 1/chi.
// Any output may be null when its model is inactive.  The tensor output has
// components (Drr, Drz, Dzz); at exactly zero field it becomes D_T I because
// no preferred direction exists.  No mobility floor enters E/N: chi is
// validated positive and a nonpositive/nonfinite chi or field is counted in
// diagnostics.invalid_cells.  Optional electron-density and source fields
// supply activity weights for the counted lower-asymptote hold; they do not
// alter either coefficient.  Outputs need >= 1 ghost cell; valid cells are
// written and FillBoundary supplies the rest, so the face-transport stencil
// never reads fabricated out-of-domain physics.
void BuildElectronClosureCellFields(
    amrex::Geometry const& geom,
    amrex::MultiFab const& efield_x_cell,
    amrex::MultiFab const& efield_z_cell,
    amrex::MultiFab const* density_ratio,
    double scalar_density_ratio,
    LowEnergyElectronClosure const& closure,
    amrex::MultiFab const* electron_density,
    amrex::MultiFab const* electron_source_rate,
    amrex::MultiFab* mu_e_cell,
    amrex::MultiFab* nu_att_cell,
    amrex::MultiFab* electron_diffusion_tensor_cell,
    LowEnergyClosureFieldDiagnostics& diagnostics);

struct LowEnergyCarrierDriftDiagnostics {
    double max_drift_speed_m_per_s = 0.0;
    double cfl_dt_s = 1.0e300;
    int substeps = 0;
    double max_outgoing_cfl = 0.0;
    double unlimited_max_outgoing_cfl = 0.0;
    double min_outflow_scale = 1.0;
    double limited_outflow_fraction = 0.0;
    double limited_electron_number_fraction = 0.0;
    double positive_ion_center_r_m = 0.0;
    double positive_ion_center_z_m = 0.0;
    double positive_ion_initial_center_z_m = 0.0;
    double positive_ion_drift_z_m = 0.0;
    double negative_ion_center_r_m = 0.0;
    double negative_ion_center_z_m = 0.0;
    double density_ratio_at_positive_ion_center = 1.0;
    double mu_pos_at_positive_ion_center_m2_per_vs = 0.0;
};

// A density below this value is transported as vacuum, but remains stored in
// the conservative carrier and charge state.  Even if the entire canonical
// 10-km-radius by 7-km domain were filled at this density, it would contain
// only 2.2e-18 carriers.  This is a numerical representation rule, not a
// configurable physical model parameter.
inline constexpr double carrier_transport_vacuum_density_m3 = 1.0e-30;

struct LowEnergyFaceTransportDiagnostics {
    double max_face_conductivity_siemens_per_m = 0.0;
    double max_explicit_diffusion_number = 0.0;
};

// Optional audit of independently written nodal-face copies immediately
// before OverrideSync. Production leaves this disabled; decomposition tests
// use it to prove that synchronization is canonicalization, not a hidden
// physics selection between disagreeing copies.
struct LowEnergyFaceSyncDiagnostics {
    double max_pre_sync_duplicate_difference = 0.0;
};

struct LowEnergyFluidHistoricalState {
    LowEnergyCarrierDriftDiagnostics last_carrier_drift;
    bool positive_ion_initial_center_set = false;
    double positive_ion_initial_center_z_m = 0.0;
};

class LowEnergyFluidState {
public:
    LowEnergyFluidState(
        amrex::BoxArray const& ba,
        amrex::DistributionMapping const& dm,
        int n_grow);

    amrex::MultiFab& LowElectronDensity() noexcept { return *m_ne_low; }
    amrex::MultiFab& PositiveIonDensity() noexcept { return *m_n_pos; }
    amrex::MultiFab& NegativeIonDensity() noexcept { return *m_n_neg; }
    amrex::MultiFab& LowElectronSourceRate() noexcept { return *m_s_low_electron; }
    amrex::MultiFab& PositiveIonSourceRate() noexcept { return *m_s_positive_ion; }
    amrex::MultiFab& DirectNegativeIonSourceRate() noexcept { return *m_s_direct_negative_ion; }
    amrex::MultiFab& Conductivity() noexcept { return *m_sigma; }
    amrex::MultiFab& MaxwellTime() noexcept { return *m_tau_m; }
    amrex::MultiFab& FluidChargeDensity() noexcept { return *m_rho_fluid; }
    amrex::MultiFab const& LowElectronDensity() const noexcept { return *m_ne_low; }
    amrex::MultiFab const& PositiveIonDensity() const noexcept { return *m_n_pos; }
    amrex::MultiFab const& NegativeIonDensity() const noexcept { return *m_n_neg; }
    amrex::MultiFab const& LowElectronSourceRate() const noexcept { return *m_s_low_electron; }
    amrex::MultiFab const& PositiveIonSourceRate() const noexcept { return *m_s_positive_ion; }
    amrex::MultiFab const& DirectNegativeIonSourceRate() const noexcept { return *m_s_direct_negative_ion; }
    amrex::MultiFab const& Conductivity() const noexcept { return *m_sigma; }
    amrex::MultiFab const& MaxwellTime() const noexcept { return *m_tau_m; }
    amrex::MultiFab const& FluidChargeDensity() const noexcept { return *m_rho_fluid; }

    void SetVal(double value);
    // attachment_frequency_cell is the en_table_attachment_v1 binding: the
    // caller owns evaluation, range policy, and finite/nonnegative
    // validation of the field (BuildElectronClosureCellFields), it must be
    // non-null exactly when that model is selected, and the table path never
    // reads the density-only attachment cache.  clear_sources=false retains
    // the outer-step rates for another fractional reaction interval.
    void UpdateFromIonizationSource(
        double dt,
        LowEnergyFluidConfig const& config,
        amrex::MultiFab const* density_ratio = nullptr,
        amrex::MultiFab const* attachment_frequency_cell = nullptr,
        bool clear_sources = true);
    // Spectral-rate bound for the local reaction Jacobian,
    //   nu_att + nu_det + alpha_ei(ne+n_pos) + alpha_ii(n_pos+n_neg),
    // used only to choose/check the reaction/transport splitting interval.
    // Exact channel solves remain in UpdateFromIonizationSource.
    double MaxReactionFrequency(
        LowEnergyFluidConfig const& config,
        amrex::MultiFab const* attachment_frequency_cell = nullptr);
    // FluidChargeDensity is an independently persisted conservative variable.
    // After a caller directly changes any carrier density, call
    // ComputeFluidChargeDensity once before the next chemistry/production
    // advance.  Chemistry and drift then update charge from their exact
    // source/current balance instead of reconstructing a near-cancellation.
    // collect_center_diagnostics gates the
    // post-drift center-of-mass sweep (a pure-diagnostic full-mesh pass with
    // global reductions, consumed only by the reduced CSV on diag steps).
    // The initial-center latch runs until it captures the first-step value.
    // Freeze the post-chemistry state used by the explicit conservative
    // carrier flux and its identical Maxwell current.
    void PrepareExplicitCarrierTransport(
        amrex::Geometry const& geom,
        LowEnergyFluidConfig const& config,
        amrex::MultiFab const* density_ratio = nullptr);
    // Governing face law for species s:
    //   Gamma_s = sign(q_s) mu_s E n_upwind - delta_se D.grad(n_e).
    // Values 0 < n <= carrier_transport_vacuum_density_m3 are read as vacuum
    // by this flux law only; the stored conservative state is never erased.
    // The same Gamma is consumed by ApplyExplicitCarrierTransport and Maxwell.
    // electron_mobility_cell and electron_diffusion_tensor_cell are the
    // en_table_flux_v1 bindings (non-null exactly when that model is selected,
    // >= 1 filled ghost cell).  The same drift-plus-diffusion number flux is
    // consumed by the density update and by Maxwell current; diffusion is
    // never folded into sigma*E.
    LowEnergyFaceTransportDiagnostics BuildExplicitFaceTransport(
        double dt,
        amrex::Geometry const& geom,
        amrex::Array<amrex::MultiFab const*, AMREX_SPACEDIM> const& face_efield,
        LowEnergyFluidConfig const& config,
        amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM> const& face_number_flux,
        amrex::MultiFab const* electron_mobility_cell = nullptr,
        amrex::MultiFab const* electron_diffusion_tensor_cell = nullptr);
    // Finite-volume update:
    //   n_s^(k+1) = n_s^k - dt/V sum_faces(Gamma_s A).
    // A conservative donor-cell scale caps the total outgoing Courant number;
    // the limited shared face flux is also the current consumed by Maxwell.
    void ApplyExplicitCarrierTransport(
        double dt,
        amrex::Geometry const& geom,
        amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM> const& face_number_flux,
        LowEnergyFluidConfig const& config,
        amrex::MultiFab const* density_ratio = nullptr,
        bool collect_center_diagnostics = true,
        LowEnergyFaceSyncDiagnostics* face_sync_diagnostics = nullptr);
    // Cell sigma/tau_m stay diagnostic.  The en_table_flux_v1 model passes the
    // same evaluated cell mobility here as the face flux uses.
    void ComputeConductivity(
        LowEnergyFluidConfig const& config,
        amrex::MultiFab const* density_ratio = nullptr,
        amrex::MultiFab const* electron_mobility_cell = nullptr);
    void ComputeFluidChargeDensity();
    LowEnergyCarrierDriftDiagnostics const& LastCarrierDriftDiagnostics() const noexcept
    {
        return m_last_carrier_drift;
    }
    double LastReactionChargeConservationResidual() const noexcept
    {
        return m_last_reaction_charge_residual;
    }
    [[nodiscard]] LowEnergyFluidHistoricalState SnapshotHistoricalState() const noexcept
    {
        return LowEnergyFluidHistoricalState{
            m_last_carrier_drift,
            m_positive_ion_initial_center_set,
            m_positive_ion_initial_center_z_m};
    }
    void RestoreHistoricalState(LowEnergyFluidHistoricalState const& state) noexcept
    {
        m_last_carrier_drift = state.last_carrier_drift;
        m_positive_ion_initial_center_set = state.positive_ion_initial_center_set;
        m_positive_ion_initial_center_z_m = state.positive_ion_initial_center_z_m;
    }

private:
    static void ValidateCarrierTransportConfig(
        LowEnergyFluidConfig const& config,
        bool require_drift_cfl);
    void CollectCarrierDriftCenterDiagnostics(
        amrex::Geometry const& geom,
        LowEnergyFluidConfig const& config,
        amrex::MultiFab const* density_ratio,
        bool collect_center_diagnostics);
    std::unique_ptr<amrex::MultiFab> m_ne_low;
    std::unique_ptr<amrex::MultiFab> m_n_pos;
    std::unique_ptr<amrex::MultiFab> m_n_neg;
    std::unique_ptr<amrex::MultiFab> m_s_low_electron;
    std::unique_ptr<amrex::MultiFab> m_s_positive_ion;
    std::unique_ptr<amrex::MultiFab> m_s_direct_negative_ion;
    std::unique_ptr<amrex::MultiFab> m_sigma;
    std::unique_ptr<amrex::MultiFab> m_tau_m;
    std::unique_ptr<amrex::MultiFab> m_rho_fluid;
    // Scratch used only to audit that attachment/recombination preserve the
    // species-derived charge.  The authoritative conservative rho field is
    // never overwritten with a normalized residual.
    std::unique_ptr<amrex::MultiFab> m_charge_residual;
    LowEnergyCarrierDriftDiagnostics m_last_carrier_drift;
    double m_last_reaction_charge_residual = 0.0;
    bool m_positive_ion_initial_center_set = false;
    double m_positive_ion_initial_center_z_m = 0.0;
    // Frozen post-chemistry state for one explicit face-transport update.
    std::unique_ptr<amrex::MultiFab> m_transport_old_pos;
    std::unique_ptr<amrex::MultiFab> m_transport_old_neg;
    std::unique_ptr<amrex::MultiFab> m_transport_old_ele;
    std::unique_ptr<amrex::MultiFab> m_transport_den_g;
    // Per-donor-cell/species positivity limiter of one transport update,
    // reallocated only when the layout changes.
    std::unique_ptr<amrex::MultiFab> m_outflow_scale;
    bool m_explicit_transport_prepared = false;
};

}  // namespace rrea

#endif
