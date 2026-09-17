#ifndef RREA_AMREX_ADVANCE_H
#define RREA_AMREX_ADVANCE_H

#include "rrea/RreaFieldInitializer.H"
#include "rrea/LowEnergyFluidState.H"
#include "rrea/RreaMaxwellTMRZ.H"
#include "rrea/RreaMaxwellTMRZRecord.H"

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Vector.H>

#include <cmath>
#include <limits>
#include <memory>

namespace rrea {

inline constexpr double kMaterialContinuityRelativeTolerance = 1.0e-10;
// Binary64 cancellation floor for the six-term material identity.  Deposit-
// preserving bookkeeping -- null-space resampling above all, whose residue
// lands in a support cell absolutely, not in proportion to its deposit share
// -- leaves per-cell residue of order eps times the LARGEST charge quantum it
// manipulates, so the runtime supplies that live scale.  The sqrt(eps) floor
// leaves cancellation headroom without masking loss of a resolved macro fraction.
inline const double kMaterialContinuityAbsoluteMacroFraction =
    8.0 * std::sqrt(std::numeric_limits<double>::epsilon());

// The AMReX-side field bindings of the ONE level the coupling runs on: the
// particle charge deposit (read) and the cell-centred total field the
// particle gather and closure read (written by CommitMaxwellTotalField).
struct RreaAmrexLevelBinding {
    amrex::Geometry geom;
    amrex::BoxArray box_array;
    amrex::DistributionMapping distribution_map;
    amrex::MultiFab const* rho_c_per_m3 = nullptr;
    amrex::MultiFab* efield_x = nullptr;
    amrex::MultiFab* efield_z = nullptr;
    amrex::MultiFab const* density_ratio = nullptr;
};

struct RreaAmrexAdvanceConfig {
    LowEnergyFluidConfig fluid;
    FieldInitializerConfig field_initializer;
    // Accuracy targets select equal material substeps; hard guards certify
    // every realized substep.  All are dimensionless: dt*sigma/eps0,
    // explicit tensor-diffusion number, and dt*reaction spectral rate.
    double material_conduction_target = 0.08;
    double material_conduction_guard = 0.1;
    double material_diffusion_target = 0.8;
    double material_diffusion_guard = 1.0;
    double material_reaction_target = 0.08;
    double material_reaction_guard = 0.1;
    int material_min_substeps = 2;
    int material_max_substeps = 64;
    // Non-owning binding for the en_table_* fluid models; must outlive the
    // advance and be Loaded() whenever either table model is selected.
    LowEnergyElectronClosure const* electron_closure = nullptr;
    // Largest live represented macro weight in physical electrons, over the
    // charged AND photon populations: a mid-step pair conversion moves charge
    // at photon weight.  The continuity gate permits at most
    // kMaterialContinuityAbsoluteMacroFraction of this charge per cell as an
    // absolute binary64 accumulation residual.
    double maximum_represented_charge_e =
        std::numeric_limits<double>::quiet_NaN();
    // The ion-drift center-of-mass sweep is pure
    // diagnostics; the coupling sets this true only on reduced-CSV diag steps.
    bool collect_ion_drift_center_diagnostics = true;
};

struct RreaAmrexAdvanceDiagnostics {
    double max_normalized_reaction_charge_residual = 0.0;
    // Realized-E/N accounting when an en_table_* closure evaluated this call:
    // min/max span the chemistry field and all fluid/Maxwell substeps;
    // there is no Picard loop;
    // held_below_cells is the LAST evaluation's count of the declared
    // hold-lowest-node policy.  Above-range or invalid states abort before
    // any commit, so they never appear here.
    bool closure_evaluated = false;
    double closure_min_en_td = 0.0;
    double closure_max_en_td = 0.0;
    long closure_held_below_cells = 0;
    // Maximum over every closure evaluation in the step.  Each category is
    // itself max(volume-integrated held/total, held peak/global peak), so a
    // compact but locally dominant screened core cannot hide in a whole-
    // domain integral.
    double closure_max_held_below_electron_fraction = 0.0;
    double closure_max_held_below_source_fraction = 0.0;
    double closure_max_held_below_conductivity_fraction = 0.0;
    // eps0-Gauss residual is MONITORED, never gated: the fail-closed
    // material-continuity check below keeps its initial value invariant
    // because discrete div(curl) vanishes, and the current-free seed
    // injection adds to that baseline.
    double maxwell_gauss_residual_normalized = 0.0;
    // Before the outer step is published, persisted particle + fluid charge
    // at both time levels is checked against the time-averaged face currents
    // already handed to the Maxwell substeps. Unlike the Gauss monitor, this
    // is a fail-closed material-continuity invariant.
    double maxwell_charge_continuity_max_relative = 0.0;
    double maxwell_charge_continuity_max_abs_e_per_cell = 0.0;
    // Signed sum of the same residual over the domain, accumulated over the
    // steps since start/restart (electrons): a max norm cannot see a
    // one-signed drift.
    double maxwell_charge_continuity_signed_accum_e = 0.0;
    double maxwell_btheta_max_t = 0.0;
    double maxwell_scattered_energy_j = 0.0;
    // Maximum actual-substep material-centering parameter, formed from the
    // face conductivity used to build the shared carrier/Ampere current.
    // Values << 1 bound the forward material response independently of how
    // many orders of magnitude sigma grew from its ambient value.
    double maxwell_max_dt_sigma_face_over_eps0 = 0.0;
    // Maximum actual-substep tensor-diffusion stability number
    // 2*dt_sub*lambda_max(D)*(dr^-2+dz^-2); required < 1.
    double fluid_max_explicit_diffusion_number = 0.0;
    int material_substeps = 0;
    double fluid_max_reaction_number = 0.0;
    // Total-field energy change relative to the immutable ambient, including
    // eps0 integral(Eambient.Escattered) dV -- the quantity that closes a
    // Joule/Poynting balance.  Joule and the exact discrete Poynting term
    // are bilinear means over each Ampere substep, so the residual below is
    // only the O(dt^2) difference between B^2 at the outer half-time and
    // the leapfrog's mixed-time magnetic energy (the smoke pins the exact
    // identity in the mixed form).
    double maxwell_energy_change_from_ambient_j = 0.0;
    double maxwell_joule_work_j = 0.0;
    double maxwell_poynting_out_j = 0.0;
    double maxwell_energy_balance_residual_j = 0.0;
    double maxwell_jz_moment_a_m = 0.0;
};

// The AMReX glue of the coupled low-energy fluid + Maxwell TM RZ advance:
// binds the mesh, reduces face fields to the replicated solver vectors,
// runs the adaptive material substeps and owns the material-continuity
// gate.  The field equations themselves live in RreaMaxwellTMRZ.H.
//
// Physics: d(scattered E)/dt = c^2 curl B - J_total/eps0 with the static
// ambient profile carried separately; conduction enters through the
// fluid's own committed face currents (which respond to E_total), so there
// is no sigma term to double count and fluid continuity and the field share
// one current exactly.  State is replicated on every rank; each substep
// Allreduces the shared J vectors.  Persistent B keeps the outer
// E(t),B(t-dt/2) checkpoint/radio staggering; the advance re-staggers it
// temporarily into the substep spacing.
class RreaAmrexAdvance {
public:
    RreaAmrexAdvance(
        amrex::Vector<RreaAmrexLevelBinding> level_bindings,
        int fluid_n_grow = 1);

    int FinestLevel() const noexcept;

    LowEnergyFluidState& Fluid(int lev);
    LowEnergyFluidState const& Fluid(int lev) const;

    // Realise the analytic thundercloud ambient on the Yee faces (verified
    // curl-free) and commit ambient + current scattered field as the total
    // field every consumer reads.  Fresh runs enter with a zero scattered
    // field; restarts after ReadMaxwellState.
    void InitializeBackgroundState(FieldInitializerConfig const& config);

    void AdvanceOneStep(
        double dt_s,
        RreaAmrexAdvanceConfig const& config);

    // Rank-local kinetic net-chord face currents for THIS step, in the padded
    // index-space layout of the coupling's segment deposit (converted to
    // physical A/m^2 and Allreduced inside).  Must be bound before
    // AdvanceOneStep on every step; cleared after use.
    void BindKineticCurrent(
        std::vector<double> const* jr_index,
        std::vector<double> const* jz_index,
        int pad_cells,
        double charge_per_index_flux_c);

    // Capture the actual bound particle + fluid charge as rho^n for the next
    // continuity check.  The coupling calls this after initialization and
    // again after any explicit scheduled injection, before particles move.
    void PrimeMaxwellContinuityReference();

    // Scattered-state persistence and the Huygens record (RreaMaxwellTMRZRecord.H).
    void WriteMaxwellState(std::string const& checkpoint_dir) const
    {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            rrea::WriteMaxwellState(m_maxwell, checkpoint_dir);
        }
    }
    void ReadMaxwellState(std::string const& checkpoint_dir)
    {
        rrea::ReadMaxwellState(m_maxwell, checkpoint_dir);
        m_maxwell_boundary_record_ready = false;
    }
    // Empty path disables recording.  An off-axis vector surface integral is
    // required for a radio observable; exact-axis m=0 output is longitudinal.
    void SetMaxwellBoundaryRecordDir(std::string dir)
    {
        if (dir != m_maxwell_boundary_dir) { m_maxwell_boundary_record_ready = false; }
        m_maxwell_boundary_dir = std::move(dir);
    }
    void AppendMaxwellBoundaryRecord(double time_s, double dt_s)
    {
        if (!m_maxwell_boundary_dir.empty()
            && amrex::ParallelDescriptor::IOProcessor()) {
            rrea::AppendMaxwellBoundaryRecord(
                m_maxwell, m_maxwell_boundary_dir, time_s, dt_s, !m_maxwell_boundary_record_ready);
            m_maxwell_boundary_record_ready = true;
        }
    }
    [[nodiscard]] double MaxwellBthetaAt(double r_m, double z_m) const noexcept;
    // Total field for the particle gather: the same cell-centred face average
    // CommitMaxwellTotalField publishes, interpolated bilinearly between cell
    // centres on the shared stencil rule, with Er continued oddly to the axis.
    // Replicated state, hence valid on every rank for any in-domain point.
    void GatherTotalFieldRZ(
        double r_m, double z_m,
        double& er, double& ez, double& btheta) const noexcept;

    RreaAmrexAdvanceDiagnostics const& LastDiagnostics() const noexcept
    {
        return m_last_diagnostics;
    }

private:
    void ValidateLevelBindings() const;
    void ValidateEvolutionConfig(RreaAmrexAdvanceConfig const& config) const;
    void AssembleTotalCharge();
    // en_table closure support: evaluate the closure cell fields from the
    // CURRENT bound cell fields, then collectively reduce the range
    // diagnostics and abort on any above-range/invalid state before it can
    // be committed.  Called from the current field at each explicit
    // fluid/Maxwell substep; there is no Picard loop.
    void EvaluateClosureFields(
        RreaAmrexAdvanceConfig const& config,
        bool build_mobility,
        bool build_attachment,
        char const* stage);
    // J_total on the physical faces (into m_jr_total/m_jz_total): the fluid's
    // committed species currents (qe * [Gamma+ - Gamma- - Gamma_e], components
    // 0/1/2) plus the kinetic net-chord deposit converted from index space --
    // the ONE current that enters Ampere's law, the continuity gate, and the
    // Joule ledger, assembled in ONE Allreduce of a concatenated buffer.
    void AssembleTotalFaceCurrent();
    // Fail-closed MATERIAL continuity before the outer step is committed:
    // rho^{n+1} - rho^n + dt div J = 0 per cell in the physical RZ metric,
    // relative to the net state, face-current and gross source operands, with
    // an absolute binary64 floor tied to the largest live macro charge.
    // rho^n is the primed previous total; rho^{n+1} is reassembled from the
    // same bound objects, so the identity covers survival, birth,
    // kinetic/fluid transfer, annihilation, resampling and absorbing escape.
    // Throws collectively on violation.
    void VerifyMaterialContinuity(
        RreaAmrexAdvanceConfig const& config,
        double dt_s,
        std::vector<double> const& jr_total,
        std::vector<double> const& jz_total);
    // eps0-Gauss MONITOR (never a gate): eps0 div E_scattered tracks the free
    // charge up to the current-free injection baseline plus round-off.
    void RecordGaussMonitor();
    // Publish ambient + scattered as the total Yee faces every consumer reads
    // and rebuild the cell averages the gather and closure use.
    void CommitMaxwellTotalField();

    amrex::Vector<RreaAmrexLevelBinding> m_levels;
    amrex::Vector<std::unique_ptr<LowEnergyFluidState>> m_fluids;
    amrex::Vector<std::unique_ptr<amrex::MultiFab>> m_rho_total;
    using FaceFabArray =
        amrex::Array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM>;
    amrex::Vector<FaceFabArray> m_face_efield;
    amrex::Vector<FaceFabArray> m_face_number_flux;
    // en_table closure cell fields (1 ghost for the face stencil), allocated
    // lazily on the first step that selects a table model.
    amrex::Vector<std::unique_ptr<amrex::MultiFab>> m_mu_e_cell;
    amrex::Vector<std::unique_ptr<amrex::MultiFab>> m_nu_att_cell;
    // Components (Drr, Drz, Dzz), built from the closure's D_L/D_T and the
    // current cell-centered field.  The identical face flux derived from this
    // tensor advances electron density and supplies Maxwell current.
    amrex::Vector<std::unique_ptr<amrex::MultiFab>> m_diffusion_tensor_e_cell;
    bool m_background_field_initialized = false;
    RreaAmrexAdvanceDiagnostics m_last_diagnostics;

    std::string m_maxwell_boundary_dir;
    bool m_maxwell_boundary_record_ready = false;
    MaxwellTMRZ m_maxwell;   // scattered Er/Ez/Btheta, replicated
    MaxwellTMRZ m_ambient;   // static curl-free ambient Er/Ez (no Btheta)
    // Faces shared by adjacent boxes are counted once per owner in the
    // replicated reduction; the multiplicities are mesh geometry, fixed once.
    std::vector<double> m_er_face_count;
    std::vector<double> m_ez_face_count;
    // Full-domain face scratch, sized once and reused every substep: the
    // committed total field, the concatenated current Allreduce buffer, its
    // kinetic arm, and the per-substep / time-averaged total current.
    std::vector<double> m_total_er;
    std::vector<double> m_total_ez;
    std::vector<double> m_current_buffer;
    std::vector<double> m_jr_kinetic;
    std::vector<double> m_jz_kinetic;
    std::vector<double> m_jr_total;
    std::vector<double> m_jz_total;
    std::vector<double> m_jr_average;
    std::vector<double> m_jz_average;
    std::unique_ptr<amrex::MultiFab> m_maxwell_rho_previous;
    bool m_maxwell_continuity_primed = false;
    // Previous step's realized substep count: logs changes, and rescales the
    // lagged drift Courant number the next selector reads.
    int m_maxwell_coupling_substeps = 0;
    std::vector<double> const* m_kinetic_jr_index = nullptr;
    std::vector<double> const* m_kinetic_jz_index = nullptr;
    int m_kinetic_pad_cells = 0;
    double m_kinetic_charge_per_index_flux_c = 0.0;
};

}  // namespace rrea

#endif
