#ifndef RREA_INTERACTION_TABLES_H
#define RREA_INTERACTION_TABLES_H
#include "rrea/RreaInteractionPhysics.H"
#include "rrea/RreaElectronClosure.H"
#include <map>
#include <string>
#include <vector>
namespace rrea {
struct RreaInteractionTableStorage {
    RreaInteractionTablesView view;
    std::vector<GpuVector<amrex::Real>> reals;
    std::array<GpuVector<RreaConditionalCdfGroupView>,4> conditional;
    GpuVector<RreaPhotoelectricSubshellCdfGroupView> photo;
    std::vector<GpuVector<RreaPhotoelectricSubshellCdfRowView>> photo_rows;
    GpuVector<RreaPairEmpiricalBlockView> pair;
    std::vector<GpuVector<RreaPairEmpiricalRecord>> pair_rows;
    template<class Sequence> KineticSpan<amrex::Real const> Keep(Sequence const& a) {
        reals.emplace_back(a.begin(),a.end());
        return {reals.back().data(),reals.back().size()};
    }
};
class RreaInteractionTables {
public:
    RreaInteractionTables()=default;
    RreaInteractionTables(RreaInteractionTables const&)=delete;
    RreaInteractionTables& operator=(RreaInteractionTables const&)=delete;
    RreaInteractionTables(RreaInteractionTables&&)=default;
    RreaInteractionTables& operator=(RreaInteractionTables&&)=default;
    operator RreaInteractionTablesView const&() const noexcept { return View(); }

    [[nodiscard]] RreaInteractionTablesView const& View() const noexcept { return m_kinetic_storage.view; }
    // Only the production schema-6 contract is accepted.
    void Load(std::string const& config_path);

    [[nodiscard]] bool Loaded() const noexcept { return m_loaded; }
    // Electrons and positrons share one charged range.
    [[nodiscard]] RreaCertifiedEnergyRange ChargedCertifiedEnergyRangeEv() const noexcept
    {
        return m_charged_certified_range;
    }
    [[nodiscard]] RreaCertifiedEnergyRange PhotonCertifiedEnergyRangeEv() const noexcept
    {
        return m_photon_certified_range;
    }
    [[nodiscard]] std::string const& TableClass() const noexcept { return m_table_class; }
    [[nodiscard]] std::string const& TransportModel() const noexcept { return m_transport_model; }
    [[nodiscard]] amrex::Real GsFewCollisionThreshold() const noexcept { return m_gs_few_collision_threshold; }

    // Native probe API: uses the exact production interpolation implementation
    // and therefore shares its strict, endpoint-inclusive range contract.
    [[nodiscard]] amrex::Real ProbeTableValue(
        std::string const& name, amrex::Real energy_eV) const;

    [[nodiscard]] amrex::Real WAirEvPerIonPair() const noexcept
    {
        return static_cast<amrex::Real>(mean_energy_per_ion_pair_air_eV);
    }
    [[nodiscard]] amrex::Real HardMollerSecondaryThresholdEv() const noexcept
    {
        return m_hard_moller_secondary_threshold_eV;
    }
    // Single owner of the tracked-brems rule: below this a photon is continuous
    // drag, above it an explicit particle.  A flat energy, not a fraction of the
    // parent, so there is one number and nothing restates it.
    [[nodiscard]] amrex::Real BremsAbsoluteCutoffEv() const noexcept
    {
        return m_brems_absolute_cutoff_eV;
    }
    // Air density these tables were generated at.  Every density_ratio argument
    // below means rho(z) divided by THIS, and by nothing else.  Load() requires
    // the configuration to declare it.
    [[nodiscard]] amrex::Real ReferenceDensityKgM3() const noexcept
    {
        return m_reference_density_kg_m3;
    }
    // Numerical rates and samplers live on View(); this host sampler adds the
    // element symbol and shell id of the drawn photoelectric subshell.
    [[nodiscard]] RreaPhotoelectricFinalState SamplePhotoelectricFinalState(
        amrex::Real photon_energy_eV,
        amrex::Real cdf_u) const;

private:
    void BuildKineticStorage();
    RreaInteractionTableStorage m_kinetic_storage;
    // Loader/validation/probe path only: it takes a name and may fail.
    [[nodiscard]] RreaInteractionTable const& Require(std::string const& name) const;

    bool m_loaded = false;
    std::string m_table_class;
    std::string m_transport_model;
    amrex::Real m_hard_moller_secondary_threshold_eV = 0.0;
    amrex::Real m_brems_absolute_cutoff_eV = 0.0;
    amrex::Real m_reference_density_kg_m3 = 0.0;
    // One certified window for both charged species: the configuration declares a
    // single charged_energy_range_eV.
    RreaCertifiedEnergyRange m_charged_certified_range;
    RreaCertifiedEnergyRange m_photon_certified_range;
    std::map<std::string, RreaInteractionTable> m_tables;
    std::map<std::string, amrex::Vector<RreaConditionalCdfGroup>> m_conditional_cdfs;
    amrex::Vector<RreaPhotoelectricSubshellCdfGroup> m_photoelectric_subshell_cdf;
    // Elastic parameter and conditional-angle tables for the GS and WentzelVI branches.
    RreaElasticMfpTable m_electron_elastic_mfp;
    RreaElasticMfpTable m_positron_elastic_mfp;
    RreaGsAngleTable m_electron_gs_angle;
    RreaGsAngleTable m_positron_gs_angle;
    RreaGsAngleTable m_electron_wentzel_angle;
    RreaGsAngleTable m_positron_wentzel_angle;
    amrex::Vector<RreaPairEmpiricalBlock> m_pair_empirical_blocks;
    amrex::Real m_gs_few_collision_threshold = 10.0;
    amrex::Real m_gs_isotropic_above_n = 1.0e5;
    amrex::Real m_gs_wentzel_transition_energy_eV = 0.0;
};

// ---- Low-energy electron closure (measured-data E/N table) -----------------
//
// Field-dependent reduced FLUX mobility K0(E/N) and reference two-/three-body
// attachment frequencies for the en_table_flux_v1 /
// en_table_attachment_v1 fluid models, loaded directly from one semantic CSV.
// Local coefficients are
// K0/chi and nu2_ref*chi + nu3_ref*chi^2, with chi the density ratio;
// the loader refuses a bundle whose reference density is not Loschmidt.
//
// Range policy is part of the loaded model, never silent: inside the
// valid-range interpolation is piecewise log-linear on the positive
// quantities (exact at nodes); BELOW range the lowest node is held and the
// caller counts it (the production geometry guarantees ~zero-field volume,
// so a strict below-range abort could never run); ABOVE range the caller
// must abort before committing any state. The two-body onset is nonnegative
// linear-in-log(E/N) across a zero bracket and log-linear once positive.
//
// The owner validates tables on the host; View() exposes only immutable
// numerical arrays to the GPU. The interpolation equation is shared.

using RreaElectronClosureCoefficients = ElectronClosureCoefficientsT<amrex::Real>;
using RreaElectronClosureView = ElectronClosureViewT<amrex::Real>;

class LowEnergyElectronClosure {
public:
    LowEnergyElectronClosure() = default;

    // Loads and fully validates one table: exact unit-bearing CSV header,
    // strictly increasing E/N nodes, finite positive coefficients, and the
    // internal derivation identities
    // k2 = W_F*eta/N, nu2_ref = k2*N0, k3 = W_F*eta/N^2,
    // nu3_ref = k3*N0^2, K0 = W_F/(E/N*N0). Throws
    // std::runtime_error on any violation.
    static LowEnergyElectronClosure Load(std::string const& table_path);

    [[nodiscard]] bool Loaded() const noexcept { return !m_en_td.empty(); }
    [[nodiscard]] RreaElectronClosureView View() const noexcept
    {
        return {m_en_td.size(),
                m_en_td.data(),
                m_log_en_td.data(),
                m_k0_flux_ref.data(),
                m_log_k0_flux_ref.data(),
                m_nu2_ref.data(),
                m_nu3_ref.data(),
                m_log_nu3_ref.data(),
                m_longitudinal_diffusion_ref.data(),
                m_log_longitudinal_diffusion_ref.data(),
                m_transverse_diffusion_ref.data(),
                m_log_transverse_diffusion_ref.data()};
    }
    [[nodiscard]] RreaElectronClosureCoefficients Evaluate(
        double en_td, RreaClosureRangeStatus& status) const noexcept;
    // Strict variant for startup validation and tests: throws on
    // AboveRange/Invalid (a below-range hold is legal and returned).
    [[nodiscard]] RreaElectronClosureCoefficients EvaluateHost(
        double en_td) const;
    [[nodiscard]] double MinEnTd() const noexcept { return m_en_td.front(); }
    [[nodiscard]] double MaxEnTd() const noexcept { return m_en_td.back(); }

private:
    GpuVector<double> m_en_td;
    GpuVector<double> m_log_en_td;
    GpuVector<double> m_k0_flux_ref;
    GpuVector<double> m_log_k0_flux_ref;
    GpuVector<double> m_nu2_ref;
    GpuVector<double> m_nu3_ref;
    GpuVector<double> m_log_nu3_ref;
    GpuVector<double> m_longitudinal_diffusion_ref;
    GpuVector<double> m_log_longitudinal_diffusion_ref;
    GpuVector<double> m_transverse_diffusion_ref;
    GpuVector<double> m_log_transverse_diffusion_ref;
};

}  // namespace rrea

#endif
