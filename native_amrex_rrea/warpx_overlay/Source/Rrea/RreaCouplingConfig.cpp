// Configuration, startup contracts and table/schedule loading for
// RreaWarpXCoupling: everything that runs before any coupled state is
// allocated, mutated, or consumes RNG.

#include "RreaWarpXCoupling.H"

#include "RreaCouplingDetail.H"
#include "RreaParticleInteraction.H"
#include "RreaRng.H"

#include "rrea/RreaDebugOptions.H"

#include "EmbeddedBoundary/Enabled.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Utility.H>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <initializer_list>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace rrea::warpx {

using detail::checkpoint_complete_marker;
using detail::dynamic_regrid_requested;
using detail::join_path;
using detail::lower_copy;
using detail::photon_species_energy;
using detail::read_binary_file;
using detail::validate_rrea_external_field_contract;

namespace {

constexpr char kProductionTransportModel[] =
    "rrea_mc_pic_v6_geant4_10_7_4p04";
constexpr char kValidationTransportModel[] =
    "rrea_schema6_validation_debug";


bool query_bool(amrex::ParmParse& pp, char const* name, bool value)
{
    int parsed = value ? 1 : 0;
    pp.query(name, parsed);
    return parsed != 0;
}

template <PhysicalSpecies ExpectedSpecies>
void validate_managed_species_identity(
    WarpXParticleContainer& particles,
    std::string const& configured_name,
    char const* role,
    char const* expected_type,
    amrex::Real expected_mass_kg,
    amrex::Real expected_charge_c)
{
    amrex::Real const effective_mass_kg =
        static_cast<amrex::Real>(particles.getMass());
    amrex::Real const effective_charge_c =
        static_cast<amrex::Real>(particles.getCharge());
    std::string const effective_type = particles.getSpeciesTypeName();
    if (!particles.AmIA<ExpectedSpecies>()
        || !RreaManagedSpeciesIdentityMatches(
            effective_type,
            effective_mass_kg,
            effective_charge_c,
            expected_type,
            expected_mass_kg,
            expected_charge_c)) {
        std::ostringstream message;
        message.imbue(std::locale::classic());
        message << std::setprecision(17)
                << "RREA managed " << role << " species '" << configured_name
                << "' has an uncertified physical identity: effective type="
                << effective_type << ", mass_kg=" << effective_mass_kg
                << ", charge_C=" << effective_charge_c
                << "; required type=" << expected_type
                << ", mass_kg=" << expected_mass_kg
                << ", charge_C=" << expected_charge_c;
        amrex::Abort(message.str());
    }
}

void validate_rrea_managed_species(
    WarpX& warpx,
    std::string const& electron_name,
    std::string const& photon_name,
    std::string const& positron_name)
{
    if (!RreaManagedSpeciesNamesAreDistinct(
            electron_name, photon_name, positron_name)) {
        amrex::Abort(
            "RREA electron, photon, and positron species names must be nonempty "
            "and pairwise distinct");
    }

    auto& containers = warpx.GetPartContainer();
    auto& electrons = containers.GetParticleContainerFromName(electron_name);
    auto& photons = containers.GetParticleContainerFromName(photon_name);
    auto& positrons = containers.GetParticleContainerFromName(positron_name);

    // Match the precision of WarpXParticleContainer's stored effective values
    // before promoting them to amrex::Real for the exact comparison.
    amrex::Real const electron_mass_kg = static_cast<amrex::Real>(
        static_cast<amrex::ParticleReal>(PhysConst::m_e));
    amrex::Real const elementary_charge_c = static_cast<amrex::Real>(
        static_cast<amrex::ParticleReal>(PhysConst::q_e));
    validate_managed_species_identity<PhysicalSpecies::electron>(
        electrons,
        electron_name,
        "electron",
        "electron",
        electron_mass_kg,
        -elementary_charge_c);
    validate_managed_species_identity<PhysicalSpecies::photon>(
        photons,
        photon_name,
        "photon",
        "photon",
        amrex::Real(0.0),
        amrex::Real(0.0));
    validate_managed_species_identity<PhysicalSpecies::positron>(
        positrons,
        positron_name,
        "positron",
        "positron",
        electron_mass_kg,
        elementary_charge_c);
}

void validate_managed_species_lifecycle(
    WarpXParticleContainer& particles,
    std::string const& species_name,
    bool charged)
{
    amrex::ParmParse pp(species_name);
    std::string injection_style = "none";
    pp.query("injection_style", injection_style);
    injection_style = lower_copy(injection_style);
    auto queried_bool = [&pp](char const* key, bool default_value = false) {
        int value = default_value ? 1 : 0;
        pp.query(key, value);
        return value != 0;
    };
    bool const do_not_push = queried_bool("do_not_push");
    bool const do_not_gather = queried_bool("do_not_gather");
    bool const do_not_deposit = queried_bool("do_not_deposit");
    bool const native_process =
        particles.DoFieldIonization() != 0
        || particles.DoQED() != 0
        || queried_bool("do_classical_radiation_reaction")
        || queried_bool("do_qed_quantum_sync")
        || queried_bool("do_qed_breit_wheeler")
        || queried_bool("do_qed_virtual_photons")
        || queried_bool("initialize_self_fields");
    if (injection_style != "none"
        || particles.doContinuousInjection()
        || particles.do_splitting
        || queried_bool("do_resampling")
        || do_not_push
        || (charged && (do_not_deposit || do_not_gather
                        || particles.do_not_deposit != 0))
        || native_process) {
        std::ostringstream message;
        message
            << "RREA-managed species '" << species_name
            << "' has an uncertified native WarpX lifecycle/process setting. "
               "Required: injection_style=none, "
               "do_continuous_injection=0, "
               "do_splitting=0, do_resampling=0, do_not_push=0, no native "
               "self-field/field-ionization/QED/radiation-reaction process";
        if (charged) {
            message << ", and do_not_deposit=do_not_gather=0";
        }
        amrex::Abort(message.str());
    }
}

void validate_rrea_warpx_mode_contract(
    WarpX& warpx,
    std::string const& electron_name,
    std::string const& photon_name,
    std::string const& positron_name)
{
    if (warpx.getdo_moving_window() != 0
        || WarpX::gamma_boost != amrex::Real(1.0)
        || WarpX::do_single_precision_comms
        || WarpX::n_rz_azimuthal_modes != 1
        || warpx.finestLevel() != 0
        || warpx.maxLevel() != 0
        || warpx.get_load_balance_intervals().isActivated()
        || dynamic_regrid_requested()
        || EB::enabled()
        || WarpX::electromagnetic_solver_id
            != ElectromagneticSolverAlgo::None
        || WarpX::electrostatic_solver_id
            != ElectrostaticSolverAlgo::LabFrame
        || WarpX::poisson_solver_id != PoissonSolverAlgo::Multigrid
        || warpx.evolve_scheme != EvolveScheme::Explicit) {
        amrex::Abort(
            "RREA schema-6 requires a fixed single-level RZ mesh with "
            "n_rz_azimuthal_modes=1, no moving window, no boost, no embedded "
            "boundary, no dynamic load balancing/regridding, full-precision "
            "communications, explicit evolution, electromagnetic solver None, "
            "and electrostatic LabFrame multigrid mode");
    }

    auto& containers = warpx.GetPartContainer();
    validate_managed_species_lifecycle(
        containers.GetParticleContainerFromName(electron_name),
        electron_name,
        true);
    validate_managed_species_lifecycle(
        containers.GetParticleContainerFromName(photon_name),
        photon_name,
        false);
    validate_managed_species_lifecycle(
        containers.GetParticleContainerFromName(positron_name),
        positron_name,
        true);

    amrex::ParmParse collisions("collisions");
    amrex::Vector<std::string> collision_names;
    collisions.queryarr("collision_names", collision_names);
    if (!collision_names.empty()) {
        amrex::Abort(
            "RREA schema-6 rejects native WarpX collisions: managed-species "
            "participation cannot be excluded from the certified transport "
            "and conservation ledger");
    }
}

// Native Esirkepov current deposition reconstructs the previous position as
// x - v*dt, and its shifted shape factor assumes at most one cell of
// displacement. A relativistic RREA-managed macro can exceed that stencil.
// Whenever c*dt exceeds the smallest cell size, every native diagnostic
// that could trigger a current deposit must be certified inert: checkpoint
// format always deposits current and must be fully neutralized; other Full
// formats must declare an explicit fields_to_plot without current
// components (the WarpX default field set includes j).
bool intervals_string_is_never(std::string const& intervals)
{
    return intervals == "0" || intervals == "0:0";
}

void validate_rrea_native_diag_current_deposit_contract(WarpX& warpx)
{
    amrex::Real const dt_s = warpx.getdt(0);
    if (dt_s <= amrex::Real(0.0)) {
        amrex::Abort(
            "RREA schema-6 requires a positive fixed timestep before the "
            "native-diagnostic current-deposit contract can be certified");
    }
    auto const& geom = warpx.Geom(0);
    amrex::Real min_cell_m = geom.CellSize(0);
    for (int dim = 1; dim < AMREX_SPACEDIM; ++dim) {
        min_cell_m = std::min(min_cell_m, geom.CellSize(dim));
    }
    amrex::Real const light_step_m =
        static_cast<amrex::Real>(PhysConst::c) * dt_s;
    if (light_step_m <= min_cell_m) {
        return;
    }
    amrex::ParmParse pp_diag("diagnostics");
    amrex::Vector<std::string> diag_names;
    pp_diag.queryarr("diags_names", diag_names);
    for (auto const& name : diag_names) {
        amrex::ParmParse pp(name);
        std::string intervals;
        pp.query("intervals", intervals);
        int dump_last_timestep = 1;
        pp.query("dump_last_timestep", dump_last_timestep);
        bool const can_fire =
            !intervals_string_is_never(intervals) || dump_last_timestep != 0;
        if (!can_fire) {
            continue;
        }
        std::string format = "plotfile";
        pp.query("format", format);
        bool violation = false;
        if (lower_copy(format) == "checkpoint") {
            violation = true;
        } else {
            amrex::Vector<std::string> fields;
            bool const fields_set = pp.queryarr("fields_to_plot", fields);
            if (!fields_set) {
                violation = true;
            } else {
                for (auto const& field : fields) {
                    std::string const lowered = lower_copy(field);
                    if (!lowered.empty() && lowered.front() == 'j') {
                        violation = true;
                        break;
                    }
                }
            }
        }
        if (violation) {
            amrex::Abort(
                "RREA schema-6 forbids native diagnostic '" + name
                + "': c*dt exceeds the smallest cell size, so a flush that "
                  "deposits Esirkepov current (checkpoint format, a default "
                  "field set, or any j component in fields_to_plot) would "
                  "index past the one-cell shape-factor support of an "
                  "RREA-transported macro. Neutralize the diagnostic "
                  "(intervals=0 and dump_last_timestep=0), remove current "
                  "components, or reduce warpx.const_dt below "
                  "min_cell/c.");
        }
    }
}

template <typename T>
void query_real(amrex::ParmParse& pp, char const* name, T& value)
{
    amrex::Real parsed = static_cast<amrex::Real>(value);
    pp.query(name, parsed);
    value = static_cast<T>(parsed);
}

void validate_rrea_particle_boundary_contract(WarpX const& warpx)
{
#if !defined(WARPX_DIM_RZ)
    amrex::ignore_unused(warpx);
    amrex::Abort("RREA schema-6 transport boundary contract requires WarpX RZ");
#else
    if (warpx.finestLevel() != 0) {
        amrex::Abort("RREA schema-6 pre-boundary transport is not certified for AMR levels > 0");
    }
    auto const& lo = WarpX::particle_boundary_lo;
    auto const& hi = WarpX::particle_boundary_hi;
    bool const axis_none = lo[0] == ParticleBoundaryType::None;
    amrex::Real const axis_tolerance = amrex::Real(64.0)
        * std::numeric_limits<amrex::Real>::epsilon()
        * amrex::max(amrex::Real(1.0), std::abs(warpx.Geom(0).ProbHi(0)));
    if (std::abs(warpx.Geom(0).ProbLo(0)) > axis_tolerance) {
        amrex::Abort(
            "RREA schema-6 RZ coupling requires geometry.prob_lo[0]=0 so "
            "the low-r field boundary is the physical symmetry axis; annular "
            "or other nonzero-ProbLo(r) domains are not certified");
    }
    bool const radial_certified =
        (axis_none || lo[0] == ParticleBoundaryType::Absorbing)
        && hi[0] == ParticleBoundaryType::Absorbing;
    bool const z_absorbing = lo[1] == ParticleBoundaryType::Absorbing
        && hi[1] == ParticleBoundaryType::Absorbing;
    bool const z_periodic = lo[1] == ParticleBoundaryType::Periodic
        && hi[1] == ParticleBoundaryType::Periodic;
    bool const geometry_z_periodic = warpx.Geom(0).isPeriodic(1);
    if (!radial_certified || (!z_absorbing && !z_periodic)
        || z_periodic != geometry_z_periodic) {
        std::ostringstream message;
        message << "RREA schema-6 transport certifies absorbing radial boundaries "
                   "(plus None at the r=0 axis) and either paired absorbing or paired "
                   "periodic axial boundaries; open, reflecting, thermal, mixed, and "
                   "other boundary modes are unsupported. "
                   "Configured boundary enum values: rlo=" << static_cast<int>(lo[0])
                << " rhi=" << static_cast<int>(hi[0])
                << " zlo=" << static_cast<int>(lo[1])
                << " zhi=" << static_cast<int>(hi[1]);
        amrex::Abort(message.str());
    }
#endif
}

void validate_rrea_field_boundary_contract(WarpX const& warpx)
{
#if !defined(WARPX_DIM_RZ)
    amrex::ignore_unused(warpx);
    amrex::Abort("RREA schema-6 field-boundary contract requires WarpX RZ");
#else
    auto const& lo = WarpX::field_boundary_lo;
    auto const& hi = WarpX::field_boundary_hi;
    bool const radial_axis_pec =
        lo[0] == FieldBoundaryType::None
        && hi[0] == FieldBoundaryType::PEC;
    bool const z_pec = lo[1] == FieldBoundaryType::PEC
        && hi[1] == FieldBoundaryType::PEC;
    bool const z_periodic = lo[1] == FieldBoundaryType::Periodic
        && hi[1] == FieldBoundaryType::Periodic;
    bool const geometry_z_periodic = warpx.Geom(0).isPeriodic(1);
    bool const particle_z_periodic =
        WarpX::particle_boundary_lo[1] == ParticleBoundaryType::Periodic
        && WarpX::particle_boundary_hi[1] == ParticleBoundaryType::Periodic;
    if (!radial_axis_pec || (!z_pec && !z_periodic)
        || z_periodic != geometry_z_periodic
        || z_periodic != particle_z_periodic) {
        std::ostringstream message;
        message
            << "RREA Maxwell TM supports exactly field boundaries "
               "[rlo=None(axis), rhi=PEC] with either [zlo=PEC, zhi=PEC] "
               "or paired axial Periodic boundaries. Field, particle, and "
               "Geometry axial periodicity must agree; PMC, open, PML, "
               "PEC-insulator, damped, mixed, and other field boundaries are "
               "unsupported. Configured field enum values: rlo="
            << static_cast<int>(lo[0])
            << " rhi=" << static_cast<int>(hi[0])
            << " zlo=" << static_cast<int>(lo[1])
            << " zhi=" << static_cast<int>(hi[1]);
        amrex::Abort(message.str());
    }
#endif
}

[[noreturn]] void abort_rrea_external_field(std::string const& key)
{
    amrex::Abort(
        "RREA-managed species forbid native WarpX external E/B field '" + key
        + "'. Remove the external-field setting and express electrostatic "
          "backgrounds through rrea.initial_field_mode / the RREA field profile; "
          "external magnetic fields are not certified. Native external fields "
          "are not included in clipped charged field-work integration or the "
          "enabled-carrier conductivity contract.");
    std::abort();
}

}  // namespace

namespace detail {

void validate_rrea_external_field_contract(
    std::string const& electron_species,
    std::string const& photon_species,
    std::string const& positron_species)
{
    auto reject_key = [](amrex::ParmParse const& pp, std::string const& prefix,
                         char const* key) {
        if (pp.contains(key)) {
            abort_rrea_external_field(prefix + "." + key);
        }
    };
    auto reject_nondefault_style = [](
                                       amrex::ParmParse const& pp,
                                       std::string const& prefix,
                                       char const* key,
                                       std::initializer_list<char const*> allowed) {
        std::string style;
        pp.query(key, style);
        style = lower_copy(style);
        bool accepted = false;
        for (char const* value : allowed) {
            accepted = accepted || style == value;
        }
        if (!accepted) {
            abort_rrea_external_field(prefix + "." + key + "=" + style);
        }
    };

    amrex::ParmParse const warpx_pp("warpx");
    reject_nondefault_style(
        warpx_pp, "warpx", "E_ext_grid_init_style", {"", "default"});
    reject_nondefault_style(
        warpx_pp, "warpx", "B_ext_grid_init_style", {"", "default"});
    reject_key(warpx_pp, "warpx", "E_external_grid");
    reject_key(warpx_pp, "warpx", "B_external_grid");

    amrex::ParmParse const particles_pp("particles");
    reject_nondefault_style(
        particles_pp, "particles", "E_ext_particle_init_style", {"", "none"});
    reject_nondefault_style(
        particles_pp, "particles", "B_ext_particle_init_style", {"", "none"});
    // WarpX currently reads these constant arrays from the global particles
    // namespace, so reject them in addition to the per-species spellings.
    reject_key(particles_pp, "particles", "E_external_particle");
    reject_key(particles_pp, "particles", "B_external_particle");

    for (std::string const& species : {
             electron_species, photon_species, positron_species}) {
        if (species.empty()) {
            amrex::Abort(
                "RREA external-field validation requires all managed species names");
        }
        amrex::ParmParse const species_pp(species);
        reject_key(species_pp, species, "E_external_particle");
        reject_key(species_pp, species, "B_external_particle");
    }
}

}  // namespace detail

void RreaWarpXCoupling::ReadParameters()
{
    if (m_parameters_read) {
        return;
    }

    amrex::ParmParse pp("rrea");
    m_enabled = query_bool(pp, "enable", false);
    m_debug_disable_transport = query_bool(pp, "debug_disable_transport", false);
    m_debug_disable_diagnostics = query_bool(pp, "debug_disable_diagnostics", false);
    m_debug_disable_field_copyback = query_bool(pp, "debug_disable_field_copyback", false);
    m_debug_disable_checkpoint = query_bool(pp, "debug_disable_checkpoint", false);
    m_debug_actual_continuity_fixture =
        query_bool(pp, "debug_actual_continuity_fixture", false);
    // Continuation-epoch marker for an intentional engine, MPI-layout or
    // physics-model transition. Each use
    // must state its reason, which is recorded into the next
    // checkpoint metadata alongside every tolerated deviation.
    m_allow_restart_epoch_change =
        query_bool(pp, "allow_restart_epoch_change", false);
    pp.query("restart_epoch_change_reason", m_restart_epoch_change_reason);
    if (m_allow_restart_epoch_change == m_restart_epoch_change_reason.empty()) {
        amrex::Abort(
            "rrea.allow_restart_epoch_change and rrea.restart_epoch_change_reason "
            "are required together: the escalation must state why this "
            "continuation epoch is allowed to break.");
    }
    pp.query("population_ceiling_policy", m_population_ceiling_policy);
    m_population_ceiling_policy = lower_copy(m_population_ceiling_policy);
    pp.query("low_energy_cutoff_eV", m_low_energy_cutoff_eV);
    m_hard_moller_secondary_threshold_input_set =
        pp.query(
            "hard_moller_secondary_threshold_eV",
            m_hard_moller_secondary_threshold_eV) > 0;
    pp.query("photon_cutoff_eV", m_photon_cutoff_eV);
    pp.query("case_id", m_case_id);
    pp.query("seed_schedule_path", m_seed_schedule_path);
    pp.query("seed_species", m_seed_species_name);
    pp.query("electron_species", m_seed_species_name);
    pp.query("photon_species", m_photon_species_name);
    pp.query("positron_species", m_positron_species_name);
    if (m_enabled) {
        validate_rrea_external_field_contract(
            m_seed_species_name,
            m_photon_species_name,
            m_positron_species_name);
    }
    pp.query("interaction_table_config", m_interaction_table_config_path);
    pp.query("transport_model", m_transport_model);
    pp.query("output_dir", m_output_dir);
    m_video_diag_enable = query_bool(pp, "video_diag_enable", false);
    pp.query("video_diag_output_dir", m_video_diag_output_dir);
    pp.query("video_diag_frame_schedule_file", m_video_diag_frame_schedule_path);
    pp.query("video_diag_nr", m_video_diag_nr);
    pp.query("video_diag_nz", m_video_diag_nz);
    pp.query("video_diag_r_min_m", m_video_diag_r_min_m);
    pp.query("video_diag_r_max_m", m_video_diag_r_max_m);
    pp.query("video_diag_z_min_m", m_video_diag_z_min_m);
    pp.query("video_diag_z_max_m", m_video_diag_z_max_m);
    pp.query("video_diag_energy_threshold_eV", m_video_diag_energy_threshold_eV);
    amrex::Vector<amrex::Real> cd_plane_z_m;
    bool const cd_planes_set = pp.queryarr("cd_plane_z_m", cd_plane_z_m);
    amrex::Real cd_plane_output_interval_s = amrex::Real(0.0);
    bool const cd_interval_set =
        pp.query("cd_plane_output_interval_s", cd_plane_output_interval_s) > 0;
    bool const cd_core_radius_set =
        pp.query("cd_plane_core_radius_m", m_cd_plane_core_radius_m) > 0;
    if (cd_planes_set && !cd_plane_z_m.empty()) {
        std::vector<amrex::Real> planes(
            cd_plane_z_m.begin(), cd_plane_z_m.end());
        if (!cd_interval_set || !cd_core_radius_set
            || !RreaPlaneLocationsAreStrictlyIncreasingFinite(planes)
            || !std::isfinite(static_cast<double>(cd_plane_output_interval_s))
            || !(cd_plane_output_interval_s > amrex::Real(0.0))
            || !std::isfinite(static_cast<double>(m_cd_plane_core_radius_m))
            || !(m_cd_plane_core_radius_m > amrex::Real(0.0))) {
            amrex::Abort(
                "rrea.cd_plane_* requires strictly increasing finite plane z, "
                "a positive finite output interval, and a positive finite core radius");
        }
        m_cd_plane_flux.Configure(
            std::move(planes), cd_plane_output_interval_s);
    } else if (cd_planes_set || cd_interval_set || cd_core_radius_set) {
        amrex::Abort(
            "rrea.cd_plane_output_interval_s/core_radius_m requires a non-empty "
            "rrea.cd_plane_z_m list");
    }
    pp.query("ion_mobility_model", m_ion_mobility_model);
    m_ion_mobility_model = lower_copy(m_ion_mobility_model);
    if (m_ion_mobility_model.empty()) {
        m_ion_mobility_model = "absolute";
    }
    if (m_ion_mobility_model != "absolute"
        && m_ion_mobility_model != "reduced_stp_density_scaled") {
        amrex::Abort(
            "rrea.ion_mobility_model must be 'absolute' or "
            "'reduced_stp_density_scaled'");
    }
    pp.query(
        "positive_ion_reduced_mobility_stp_m2_per_vs",
        m_advance_config.fluid.positive_ion_reduced_mobility_stp_m2_per_vs);
    pp.query(
        "negative_ion_reduced_mobility_stp_m2_per_vs",
        m_advance_config.fluid.negative_ion_reduced_mobility_stp_m2_per_vs);
    pp.query(
        "ion_mobility_density_ratio_floor",
        m_advance_config.fluid.ion_mobility_density_ratio_floor);
    m_advance_config.fluid.ion_mobility_model = m_ion_mobility_model;
    pp.query("diag_interval", m_diag_interval);
    pp.query("mesh_diag_interval", m_mesh_diag_interval);
    pp.query("max_electron_macros", m_max_electron_macros);
    pp.query("max_photon_macros", m_max_photon_macros);
    pp.query("max_positron_macros", m_max_positron_macros);
    pp.query("population_target_electron_macros", m_population_target_electron_macros);
    pp.query("population_target_positron_macros", m_population_target_positron_macros);
    pp.query("population_control_interval", m_population_control_interval);
    query_real(pp, "population_split_min_weight", m_population_split_min_weight);
    if (m_population_ceiling_policy != "strict_abort_v1"
        && m_population_ceiling_policy != "adaptive_resample_v1") {
        amrex::Abort(
            "rrea.population_ceiling_policy must be 'strict_abort_v1' or "
            "'adaptive_resample_v1'");
    }
    if (m_population_ceiling_policy == "adaptive_resample_v1") {
        // adaptive_resample_v1: global survivor-plus-staged-candidate CIC
        // thinning above 1.25x target, then exact macro doubling below 0.5x
        // target (never below population_split_min_weight). Strict ceilings
        // remain fail-closed resource backstops above the control band.
        if (m_population_target_electron_macros <= 0
            && m_population_target_positron_macros <= 0) {
            amrex::Abort(
                "rrea.population_ceiling_policy=adaptive_resample_v1 requires "
                "a positive electron or positron population target");
        }
        if (m_population_control_interval <= 0) {
            amrex::Abort("rrea.population_control_interval must be positive");
        }
        if (m_population_split_min_weight < amrex::Real(0.0)) {
            amrex::Abort(
                "rrea.population_split_min_weight must be zero (splitting "
                "disabled) or positive");
        }
        auto require_band_below_ceiling = [](long target, long ceiling,
                                             char const* what) {
            if (target > 0 && ceiling > 0 && 2 * target > ceiling) {
                amrex::Abort(std::string(
                    "adaptive_resample_v1: the strict ceiling for ")
                    + what
                    + " must leave headroom above the control band "
                      "(require max >= 2*target)");
            }
        };
        require_band_below_ceiling(
            m_population_target_electron_macros, m_max_electron_macros,
            "electrons");
        require_band_below_ceiling(
            m_population_target_positron_macros, m_max_positron_macros,
            "positrons");
    } else if (m_population_target_electron_macros > 0
               || m_population_target_positron_macros > 0) {
        amrex::Abort(
            "rrea.population_target_*_macros requires "
            "rrea.population_ceiling_policy=adaptive_resample_v1");
    }
    if (m_max_electron_macros < 0 || m_max_photon_macros < 0
        || m_max_positron_macros < 0) {
        amrex::Abort("RREA macro ceilings must be zero (disabled) or positive");
    }
    if (m_video_diag_enable) {
        if (m_video_diag_output_dir.empty()) {
            m_video_diag_output_dir = join_path(m_output_dir, "rrea_video");
        }
        if (m_video_diag_frame_schedule_path.empty()) {
            amrex::Abort(
                "rrea.video_diag_enable=1 requires rrea.video_diag_frame_schedule_file");
        }
        if (m_video_diag_nr <= 0 || m_video_diag_nz <= 0) {
            amrex::Abort("rrea.video_diag_nr and rrea.video_diag_nz must be positive");
        }
        if (std::isfinite(static_cast<double>(m_video_diag_r_min_m))
            && std::isfinite(static_cast<double>(m_video_diag_r_max_m))
            && m_video_diag_r_max_m <= m_video_diag_r_min_m) {
            amrex::Abort("rrea.video_diag_r_max_m must exceed rrea.video_diag_r_min_m");
        }
        if (std::isfinite(static_cast<double>(m_video_diag_z_min_m))
            && std::isfinite(static_cast<double>(m_video_diag_z_max_m))
            && m_video_diag_z_max_m <= m_video_diag_z_min_m) {
            amrex::Abort("rrea.video_diag_z_max_m must exceed rrea.video_diag_z_min_m");
        }
        if (m_video_diag_energy_threshold_eV <= amrex::Real(0.0)) {
            amrex::Abort("rrea.video_diag_energy_threshold_eV must be positive");
        }
        LoadVideoFrameSchedule();
    }

    if (m_seed_schedule_path.empty()) {
        amrex::ParmParse pp_electrons("rrea_electrons");
        pp_electrons.query("source_schedule_file", m_seed_schedule_path);
        pp_electrons.query("source_schedule_case_id", m_case_id);
    }

    pp.query("fluid_attachment_frequency_s", m_advance_config.fluid.attachment_frequency_s);
    pp.query("fluid_attachment_model", m_advance_config.fluid.attachment_model);
    m_advance_config.fluid.attachment_model =
        lower_copy(m_advance_config.fluid.attachment_model);
    if (m_advance_config.fluid.attachment_model.empty()) {
        m_advance_config.fluid.attachment_model = "constant";
    }
    if (m_advance_config.fluid.attachment_model != "constant"
        && m_advance_config.fluid.attachment_model != "en_table_attachment_v1") {
        amrex::Abort(
            "rrea.fluid_attachment_model must be 'constant'"
            " or 'en_table_attachment_v1'");
    }
    pp.query(
        "electron_ion_recombination_coefficient_m3_s",
        m_advance_config.fluid.electron_ion_recombination_coefficient_m3_per_s);
    pp.query(
        "ion_ion_recombination_coefficient_m3_s",
        m_advance_config.fluid.ion_ion_recombination_coefficient_m3_per_s);
    pp.query("fluid_detachment_frequency_s", m_advance_config.fluid.detachment_frequency_s);
    pp.query("material_conduction_target", m_advance_config.material_conduction_target);
    pp.query("material_conduction_guard", m_advance_config.material_conduction_guard);
    pp.query("material_diffusion_target", m_advance_config.material_diffusion_target);
    pp.query("material_diffusion_guard", m_advance_config.material_diffusion_guard);
    pp.query("material_reaction_target", m_advance_config.material_reaction_target);
    pp.query("material_reaction_guard", m_advance_config.material_reaction_guard);
    pp.query("material_min_substeps", m_advance_config.material_min_substeps);
    pp.query("material_max_substeps", m_advance_config.material_max_substeps);
    pp.query("electron_mobility_m2_V_s", m_advance_config.fluid.electron_mobility_m2_per_vs);
    pp.query("electron_mobility_m2_per_vs", m_advance_config.fluid.electron_mobility_m2_per_vs);
    pp.query("positive_ion_mobility_m2_V_s", m_advance_config.fluid.positive_ion_mobility_m2_per_vs);
    pp.query("positive_ion_mobility_m2_per_vs", m_advance_config.fluid.positive_ion_mobility_m2_per_vs);
    pp.query("negative_ion_mobility_m2_V_s", m_advance_config.fluid.negative_ion_mobility_m2_per_vs);
    pp.query("negative_ion_mobility_m2_per_vs", m_advance_config.fluid.negative_ion_mobility_m2_per_vs);
    m_advance_config.fluid.ion_drift_enable = query_bool(pp, "ion_drift_enable", false);
    // Electron drift persists conduction charge. Disabling it is diagnostic
    // only because a memoryless screening response is not the production model.
    m_advance_config.fluid.electron_drift_enable = query_bool(
        pp,
        "electron_drift_enable",
        m_advance_config.fluid.electron_drift_enable);
    pp.query("electron_mobility_model", m_advance_config.fluid.electron_mobility_model);
    m_advance_config.fluid.electron_mobility_model =
        lower_copy(m_advance_config.fluid.electron_mobility_model);
    if (m_advance_config.fluid.electron_mobility_model != "absolute"
        && m_advance_config.fluid.electron_mobility_model
            != "en_table_flux_v1") {
        amrex::Abort(
            "rrea.electron_mobility_model must be 'absolute'"
            " or 'en_table_flux_v1'");
    }
    // The en_table models read one unit-bearing CSV and validate its physics
    // directly. A supplied table without a table model is a configuration
    // error, never a silent no-op.
    pp.query("electron_closure_table", m_electron_closure_table_path);
    {
        bool const table_fluid_model =
            m_advance_config.fluid.electron_mobility_model
                == "en_table_flux_v1"
            || m_advance_config.fluid.attachment_model
                == "en_table_attachment_v1";
        if (table_fluid_model) {
            if (m_electron_closure_table_path.empty()) {
                amrex::Abort(
                    "en_table_* fluid models require "
                    "rrea.electron_closure_table");
            }
            try {
                m_electron_closure = LowEnergyElectronClosure::Load(
                    m_electron_closure_table_path);
            } catch (std::exception const& exc) {
                amrex::Abort(
                    std::string("electron closure bundle rejected: ")
                    + exc.what());
            }
            m_advance_config.electron_closure = &m_electron_closure;
        } else if (!m_electron_closure_table_path.empty()) {
            amrex::Abort(
                "rrea.electron_closure_table supplied but no en_table "
                "fluid model is selected");
        }
    }
    auto const finite_nonnegative = [](amrex::Real value) {
        return std::isfinite(static_cast<double>(value))
            && value >= amrex::Real(0.0);
    };
    if (!finite_nonnegative(
            m_advance_config.fluid.positive_ion_reduced_mobility_stp_m2_per_vs)
        || !finite_nonnegative(
            m_advance_config.fluid.negative_ion_reduced_mobility_stp_m2_per_vs)
        || !finite_nonnegative(
            m_advance_config.fluid.positive_ion_mobility_m2_per_vs)
        || !finite_nonnegative(
            m_advance_config.fluid.negative_ion_mobility_m2_per_vs)
        || !std::isfinite(
            m_advance_config.fluid.ion_mobility_density_ratio_floor)) {
        amrex::Abort(
            "rrea inactive ion mobilities must be finite/non-negative");
    }
    if (m_advance_config.fluid.ion_drift_enable) {
        if (m_ion_mobility_model == "reduced_stp_density_scaled") {
            if (!(m_advance_config.fluid.positive_ion_reduced_mobility_stp_m2_per_vs
                    > amrex::Real(0.0))
                || !(m_advance_config.fluid.negative_ion_reduced_mobility_stp_m2_per_vs
                    > amrex::Real(0.0))
                || !(m_advance_config.fluid.ion_mobility_density_ratio_floor
                    > amrex::Real(0.0))) {
                amrex::Abort(
                    "rrea active density-scaled ion drift requires positive "
                    "reduced mobilities and density-ratio floor");
            }
        } else if (!(m_advance_config.fluid.positive_ion_mobility_m2_per_vs
                         > amrex::Real(0.0))
            || !(m_advance_config.fluid.negative_ion_mobility_m2_per_vs
                         > amrex::Real(0.0))) {
            amrex::Abort(
                "rrea active absolute ion drift requires positive ion mobilities");
        }
    }
    pp.query("ion_drift_cfl", m_advance_config.fluid.ion_drift_cfl);
    if (m_advance_config.fluid.ion_drift_cfl <= amrex::Real(0.0)
        || m_advance_config.fluid.ion_drift_cfl > amrex::Real(1.0)) {
        amrex::Abort("rrea.ion_drift_cfl must be in (0, 1]");
    }
    // OpenMP thread count for the transport particle loops only (0 = follow the
    // global OpenMP setting); per-chunk replay fixes the side-effect order.
    pp.query("transport_omp_threads", m_transport_omp_threads);
    if (m_transport_omp_threads < 0) {
        amrex::Abort("rrea.transport_omp_threads must be >= 0");
    }
    // Ion-drift finite-volume substeps use tile-disjoint stencil writes and no
    // reductions.
    pp.query("initial_field_mode", m_initial_field_mode);
    m_initial_field_mode = lower_copy(m_initial_field_mode);
    if (m_initial_field_mode.empty()) {
        m_initial_field_mode = "constant";
    }
    if (m_initial_field_mode != "constant" && m_initial_field_mode != "altitude_profile") {
        amrex::Abort("rrea.initial_field_mode must be 'constant' or 'altitude_profile'");
    }
    // Maxwell TM is the one field equation exposed by the RREA coupling.
    pp.query("field_model", m_field_model);
    m_field_model = lower_copy(m_field_model);
    if (m_field_model.empty()) {
        m_field_model = "maxwell_rz_yee_material_coupled";
    }
    if (m_field_model != "maxwell_rz_yee_material_coupled") {
        amrex::Abort(
            "rrea.field_model '" + m_field_model
            + "' is not supported; RREA uses only "
              "'maxwell_rz_yee_material_coupled'");
    }
    pp.query("acceleration_field_V_m", m_advance_config.field_initializer.ez_acceleration_v_per_m);
    pp.query("acceleration_field_v_per_m", m_advance_config.field_initializer.ez_acceleration_v_per_m);
    amrex::Real profile_e0_peak_v_per_m =
        std::abs(m_advance_config.field_initializer.ez_acceleration_v_per_m);
    int profile_e0_key_count = 0;
    profile_e0_key_count += pp.query("profile_e0_peak_V_m", profile_e0_peak_v_per_m);
    profile_e0_key_count += pp.query("profile_e0_peak_v_per_m", profile_e0_peak_v_per_m);
    if (profile_e0_key_count > 0) {
        m_advance_config.field_initializer.ez_acceleration_v_per_m =
            -std::abs(profile_e0_peak_v_per_m);
    }
    pp.query("acceleration_z_max_m", m_advance_config.field_initializer.acceleration_z_max_m);
    bool const profile_altitude_set =
        pp.query("profile_altitude_msl_at_z0_m", m_profile_altitude_msl_at_z0_m) > 0;
    m_advance_config.field_initializer.profile_altitude_msl_at_z0_m =
        m_profile_altitude_msl_at_z0_m;
    bool const profile_phi_step_set = pp.query(
        "profile_phi_integration_step_m",
        m_advance_config.field_initializer.profile_phi_integration_step_m) > 0;
    pp.query(
        "field_taper_r_start_m",
        m_advance_config.field_initializer.taper_r_start_m);
    pp.query(
        "field_taper_r_end_m",
        m_advance_config.field_initializer.taper_r_end_m);
    {
        auto const& field = m_advance_config.field_initializer;
        bool const taper_configured =
            field.taper_r_start_m >= 0.0 || field.taper_r_end_m >= 0.0;
        if (taper_configured && !field.TaperActive()) {
            amrex::Abort(
                "rrea.field_taper requires 0 <= field_taper_r_start_m < "
                "field_taper_r_end_m (set both, or neither)");
        }
    }
    pp.query("field_profile_path", m_field_profile_path);
    std::string field_profile_altitude_column = "altitude_km_MSL";
    std::string field_profile_value_column = "minus_Ez_kV_per_m";
    pp.query("field_profile_altitude_column", field_profile_altitude_column);
    pp.query("field_profile_value_column", field_profile_value_column);
    bool const field_profile_path_provided = !m_field_profile_path.empty();
    if (field_profile_path_provided && m_initial_field_mode != "altitude_profile") {
        amrex::Abort(
            "rrea.field_profile_path requires rrea.initial_field_mode=altitude_profile");
    }
    if (m_initial_field_mode == "altitude_profile") {
        if (!profile_altitude_set
            || !std::isfinite(static_cast<double>(m_profile_altitude_msl_at_z0_m))
            || !profile_phi_step_set
            || !(m_advance_config.field_initializer.profile_phi_integration_step_m
                > 0.0)) {
            amrex::Abort(
                "rrea.initial_field_mode=altitude_profile requires explicit "
                "finite profile_altitude_msl_at_z0_m and positive "
                "profile_phi_integration_step_m");
        }
        if (m_field_profile_path.empty()) {
            amrex::Abort("rrea.initial_field_mode=altitude_profile requires rrea.field_profile_path");
        }
        if (!std::isfinite(m_advance_config.field_initializer.ez_acceleration_v_per_m)) {
            amrex::Abort("rrea.profile_e0_peak_v_per_m must be finite for altitude_profile");
        }
        m_advance_config.field_initializer.use_altitude_profile = true;
        m_advance_config.field_initializer.ez_altitude_profile.LoadCsv(
            m_field_profile_path,
            field_profile_altitude_column,
            field_profile_value_column,
            amrex::Real(1000.0),
            amrex::Real(1000.0));
    }
    pp.query("profile_active_z_min_m", m_profile_active_z_min_m);
    pp.query("profile_active_z_max_m", m_profile_active_z_max_m);
    pp.query("profile_injection_z_min_m", m_profile_injection_z_min_m);
    pp.query("profile_injection_z_max_m", m_profile_injection_z_max_m);
    auto const validate_optional_z_band =
        [](char const* name, amrex::Real z_min, amrex::Real z_max) {
            bool const min_set = std::isfinite(static_cast<double>(z_min));
            bool const max_set = std::isfinite(static_cast<double>(z_max));
            if (min_set != max_set) {
                amrex::Abort(std::string("rrea.") + name + " requires both z_min_m and z_max_m");
            }
            if (min_set && z_max <= z_min) {
                amrex::Abort(std::string("rrea.") + name + " requires z_max_m > z_min_m");
            }
        };
    validate_optional_z_band(
        "profile_active",
        m_profile_active_z_min_m,
        m_profile_active_z_max_m);
    validate_optional_z_band(
        "profile_injection",
        m_profile_injection_z_min_m,
        m_profile_injection_z_max_m);

    m_ledger.low_energy_cutoff_eV = m_low_energy_cutoff_eV;

    long long parsed_seed = static_cast<long long>(m_rng_seed);
    pp.query("rng_seed", parsed_seed);
    m_rng_seed = static_cast<std::uint64_t>(std::max(parsed_seed, 0LL));
    long long controller_salt =
        static_cast<long long>(m_population_controller_rng_salt);
    pp.query("population_controller_rng_salt", controller_salt);
    if (controller_salt < 0) {
        amrex::Abort("rrea.population_controller_rng_salt must be non-negative");
    }
    m_population_controller_rng_salt =
        static_cast<std::uint64_t>(controller_salt);
    m_rng_scheme = RreaRng::Scheme();
    std::string air_density_model = "constant_altitude";
    pp.query("air_density_model", air_density_model);
    air_density_model = lower_copy(air_density_model);
    if (air_density_model.empty()) {
        air_density_model = "constant_altitude";
    }
    if (air_density_model != "constant_altitude" && air_density_model != "altitude_profile") {
        amrex::Abort("rrea.air_density_model must be 'constant_altitude' or 'altitude_profile'");
    }
    pp.query("transport_density_ratio", m_transport_density_ratio);
    if (m_transport_density_ratio <= amrex::Real(0.0)) {
        amrex::Abort("rrea.transport_density_ratio must be positive");
    }
    m_advance_config.fluid.transport_density_ratio = m_transport_density_ratio;
    pp.query("density_profile_path", m_density_profile_path);
    pp.query("transport_density_profile_path", m_density_profile_path);
    if (!m_density_profile_path.empty() && air_density_model != "altitude_profile") {
        amrex::Abort(
            "rrea.density_profile_path requires rrea.air_density_model=altitude_profile");
    }
    std::string density_profile_altitude_column = "altitude_km_MSL";
    // Absolute density, not either of the CSV's pre-divided ratio columns:
    // those are 1.0 at the sounding's own warm surface row, which is not a
    // physical reference for anything.
    std::string density_profile_value_column = "mass_density_g_cm3";
    pp.query("density_profile_altitude_column", density_profile_altitude_column);
    pp.query("density_profile_value_column", density_profile_value_column);
    m_transport_density_profile_enabled = air_density_model == "altitude_profile";
    if (m_transport_density_profile_enabled) {
        if (m_density_profile_path.empty()) {
            amrex::Abort("rrea.air_density_model=altitude_profile requires rrea.density_profile_path");
        }
        m_transport_density_profile.LoadCsv(
            m_density_profile_path,
            density_profile_altitude_column,
            density_profile_value_column,
            amrex::Real(1000.0),
            // g/cm^3 -> kg/m^3. Tables load after profile initialization, so
            // the profile stores absolute density and each consumer divides
            // by its own reference density.
            amrex::Real(1000.0),
            /*require_positive=*/true);
    }
    m_transport_config.rng_seed = m_rng_seed;
    m_transport_config.density_ratio = m_transport_density_ratio;
    pp.query(
        "transport_max_optical_depth_per_substep",
        m_transport_config.max_optical_depth_per_substep);
    long long max_subcycles =
        static_cast<long long>(m_transport_config.max_subcycles);
    pp.query("transport_max_subcycles", max_subcycles);
    if (max_subcycles <= 0) {
        amrex::Abort(
            "rrea.transport_max_subcycles must be a positive integer; zero and "
            "negative values are invalid and are not coerced");
    }
    m_transport_config.max_subcycles =
        static_cast<std::uint64_t>(max_subcycles);
    pp.query(
        "transport_max_gs_collisions_per_substep",
        m_transport_config.transport_max_gs_collisions_per_substep);
    pp.query(
        "transport_gs_total_collision_cap",
        m_transport_config.transport_gs_total_collision_cap);
    pp.query(
        "transport_max_field_impulse_fraction",
        m_transport_config.transport_max_field_impulse_fraction);
    pp.query(
        "transport_field_impulse_floor_eV",
        m_transport_config.transport_field_impulse_floor_eV);
    auto const require_positive_finite = [](amrex::Real value, char const* name) {
        if (!std::isfinite(static_cast<double>(value))
            || value <= amrex::Real(0.0)) {
            amrex::Abort(
                std::string("rrea.") + name + " must be finite and positive");
        }
    };
    require_positive_finite(
        m_transport_config.transport_max_gs_collisions_per_substep,
        "transport_max_gs_collisions_per_substep");
    require_positive_finite(
        m_transport_config.transport_gs_total_collision_cap,
        "transport_gs_total_collision_cap");
    require_positive_finite(
        m_transport_config.transport_max_field_impulse_fraction,
        "transport_max_field_impulse_fraction");
    require_positive_finite(
        m_transport_config.transport_field_impulse_floor_eV,
        "transport_field_impulse_floor_eV");
    if (m_transport_config.transport_gs_total_collision_cap
        > amrex::Real(1.0e6)) {
        amrex::Abort(
            "rrea.transport_gs_total_collision_cap above 1e6 is not "
            "meaningful (the GS sampler is isotropic long before that) and "
            "risks integer overflow in the substep drivers");
    }
    if (m_transport_config.max_subcycles > 16383U) {
        // The repeated-hazard RNG layout gives the transport segment 14 bits.
        // Event ordinals and transient lineages have their own checked fields
        // and reject the still-uncommitted transaction if they overflow.
        // Production decks run 8192; fail closed at input parsing instead of
        // reaching an unrepresentable late segment.
        amrex::Abort(
            "rrea.transport_max_subcycles > 16383 is incompatible with the "
            "newborn residual-advance RNG packing (14-bit fields)");
    }
    m_transport_config.debug_disable_hard_moller_secondaries =
        query_bool(pp, "debug_disable_hard_moller_secondaries", false);
    m_transport_config.strict_transport_guards =
        query_bool(pp, "transport_strict_guards", true);
    if (m_enabled && !RreaTransportSubcycleControlsValid(
            m_transport_config,
            /*require_strict_guards=*/true)) {
        amrex::Abort(
            "Schema-6 production transport requires "
            "rrea.transport_strict_guards=1, a finite positive "
            "rrea.transport_max_optical_depth_per_substep, and a positive "
            "rrea.transport_max_subcycles");
    }

    // A production physics epoch may not use validation shortcuts to bypass
    // the schema-6 table/model semantics or the coupled EM contract. The
    // separate validation model is retained for static/fixed-source fixtures
    // that never claim production transport.
    if (m_enabled) {
        bool const production_model =
            m_transport_model == kProductionTransportModel;
        bool const validation_model =
            m_transport_model == kValidationTransportModel;
        if (!production_model && !validation_model) {
            amrex::Abort(
                "Enabled schema-6 RREA requires rrea.transport_model="
                "rrea_mc_pic_v6_geant4_10_7_4p04, or the explicitly "
                "non-production rrea_schema6_validation_debug fixture model.");
        }
        if (production_model) {
            if (m_interaction_table_config_path.empty()) {
                amrex::Abort(
                    "Schema-6 production requires "
                    "rrea.interaction_table_config before any state mutation.");
            }
            if (m_debug_disable_transport
                || m_debug_disable_field_copyback
                || m_debug_actual_continuity_fixture) {
                amrex::Abort(
                    "Schema-6 production transport forbids "
                    "rrea.debug_disable_transport, "
                    "rrea.debug_disable_field_copyback, and "
                    "rrea.debug_actual_continuity_fixture. Use "
                    "rrea_schema6_validation_debug for non-production fixtures.");
            }
            bool const sensitivity_study =
                rrea::DebugTransportSensitivityOptions().enable;
            if (m_transport_config.debug_disable_hard_moller_secondaries
                && !sensitivity_study) {
                amrex::Abort(
                    "Schema-6 production permits "
                    "rrea.debug_disable_hard_moller_secondaries=1 only when "
                    "rrea.debug_transport_sensitivity_enable=1 records an "
                    "explicitly labeled sensitivity study.");
            }
        }
        if (m_debug_actual_continuity_fixture) {
            if (!validation_model) {
                amrex::Abort(
                    "rrea.debug_actual_continuity_fixture is available only "
                    "with rrea.transport_model=rrea_schema6_validation_debug");
            }
            if (m_debug_disable_transport
                || !m_debug_disable_checkpoint
                || m_population_ceiling_policy != "adaptive_resample_v1"
                || m_population_target_electron_macros != 6
                || m_population_target_positron_macros != 0
                || m_population_control_interval != 1
                || m_population_split_min_weight != amrex::Real(1.0)) {
                amrex::Abort(
                    "rrea.debug_actual_continuity_fixture requires the pinned "
                    "Maxwell validation deck (electron target 6, "
                    "positron target 0, interval 1, split floor 1, no fixed "
                    "source or disabled transport, checkpointing disabled)");
            }
        }
    }

    m_parameters_read = true;
}

void RreaWarpXCoupling::InitFromWarpX(WarpX& warpx)
{
    ReadParameters();
    if (!m_enabled || m_initialized) {
        return;
    }

    RequireHybridManagedMemory();
#ifdef RREA_USE_CUDA
    amrex::Print() << "RREA backend: experimental hybrid CUDA; "
                   << "GPU field/fluid kernels, CPU particle collisions/secondaries; "
                   << "managed memory required\n";
#endif

    if (warpx.DoFluidSpecies()) {
        amrex::Abort(
            "RREA Maxwell evolution does not support native WarpX fluid species");
    }

    // Boundary semantics affect clipping, Maxwell, and field copyback.
    // Reject unsupported combinations before loading
    // tables, allocating coupled state, mutating fields, or consuming RNG.
    validate_rrea_particle_boundary_contract(warpx);
    validate_rrea_field_boundary_contract(warpx);

    // Event-source deposition is intentionally the exact cell-centered
    // WarpX order-1 RZ stencil.  A different particle shape would make the
    // kinetic charge and low-energy reaction sources discretely inconsistent,
    // so fail closed even if a non-canonical launcher bypasses Python's
    // protected-override checks.
    if (WarpX::nox != 1 || WarpX::noz != 1) {
        amrex::Abort(
            "RREA coupled transport requires algo.particle_shape=1 so event "
            "sources match WarpX cell-centered charge deposition exactly");
    }

    // Validate the effective WarpX identities before table loading, coupled
    // state allocation, or RNG use.  A species name alone is insufficient:
    // user mass/charge overrides would otherwise silently change the pusher,
    // field work, charge deposition, and conservation bookkeeping.
    validate_rrea_managed_species(
        warpx,
        m_seed_species_name,
        m_photon_species_name,
        m_positron_species_name);
    validate_rrea_warpx_mode_contract(
        warpx,
        m_seed_species_name,
        m_photon_species_name,
        m_positron_species_name);
    validate_rrea_native_diag_current_deposit_contract(warpx);

    // Validate the production model, every deployed transport table, and the
    // certified cutoff contract semantically before allocating or mutating
    // any coupled field/fluid/event state.  This path consumes no RNG.
    LoadInteractionTables();
    LoadSeedSchedule();
    ValidateSeedSchedule(warpx);
    BuildBindings(warpx);

    amrex::Vector<RreaEventAccumulatorLevelBinding> event_bindings;
    event_bindings.reserve(m_level_bindings.size());
    for (auto const& binding : m_level_bindings) {
        event_bindings.push_back(RreaEventAccumulatorLevelBinding{
            binding.geom,
            binding.box_array,
            binding.distribution_map});
    }

    m_advance = std::make_unique<RreaAmrexAdvance>(m_level_bindings, 1);
    // Four grow cells cover backtracked photon interaction points up to four
    // cells off-box (c*dt is below one production cell); deposits land in
    // the grow region and SumBoundary routes them to the owning rank.
    m_events = std::make_unique<RreaEventAccumulator>(event_bindings, 4);
    m_air_interaction = std::make_unique<RreaParticleInteraction>(m_transport_config);
    m_initialized = true;
    bool const restoring = !m_pending_checkpoint_dir.empty();
    LoadPendingCheckpointIfAny();
    m_restored_from_checkpoint = restoring;
    // Fresh runs commit ambient + zero scattered field; restarts commit
    // ambient + the scattered state LoadPendingCheckpointIfAny restored.
    m_advance->InitializeBackgroundState(m_advance_config.field_initializer);
    EnsurePreviousPositionComponents(warpx);
    if (!restoring) {
        m_initial_photon_alive_energy_eV = photon_species_energy(m_photon_species_name);
    }
    EnforcePopulationCeiling(
        warpx, m_seed_species_name, m_max_electron_macros, 0,
        "initial_population", -1);
    EnforcePopulationCeiling(
        warpx, m_photon_species_name, m_max_photon_macros, 0,
        "initial_population", -1);
    EnforcePopulationCeiling(
        warpx, m_positron_species_name, m_max_positron_macros, 0,
        "initial_population", -1);

    amrex::Print() << "RREA coupling initialized: Maxwell RZ-Yee material coupling"
                   << ", field_model=" << m_field_model
                   << ", low_energy_cutoff_eV=" << m_low_energy_cutoff_eV
                   << ", seed_events=" << m_seed_schedule.Size()
                   << ", initial_field_mode=" << m_initial_field_mode
                   << ", density_profile=" << (m_transport_density_profile_enabled ? 1 : 0)
                   << ", altitude_z0_msl_m=" << m_profile_altitude_msl_at_z0_m
                   << ", rng_scheme=" << m_rng_scheme
                   << "\n";
}

void RreaWarpXCoupling::LoadSeedSchedule()
{
    if (m_schedule_loaded) {
        return;
    }
    // RreaSeedSchedule owns parsing, sorting and semantic validation.
    if (!m_seed_schedule_path.empty()) {
        m_seed_schedule.Load(m_seed_schedule_path, m_case_id);
    }
    m_schedule_loaded = true;
}

void RreaWarpXCoupling::ValidateSeedSchedule(WarpX const& warpx) const
{
    if (m_seed_schedule.Empty()) {
        return;
    }
    if (!InteractionTablesRequired()) {
        // Table-driven transport is off (rrea.debug_disable_transport or the
        // fixed-ionization debug source without a transport configuration): the seeds only
        // exercise injection and direct PIC deposition, and the schema-6
        // production gate separately forbids every one of these debug
        // switches, so there are legitimately no certified table ranges to
        // validate against.  The direct_pic_charge and checkpoint_v2 gates
        // and the debug smoke deck all seed particles in exactly this
        // configuration.
        return;
    }
    if (m_interaction_tables.SchemaVersion() != 6) {
        amrex::Abort(
            "A nonempty RREA seed schedule requires loaded production schema-6 "
            "interaction tables before seed validation");
    }
#if !defined(WARPX_DIM_RZ)
    amrex::ignore_unused(warpx);
    amrex::Abort("RREA production seed validation requires WarpX RZ geometry");
#else
    auto const electron_range =
        m_interaction_tables.ElectronCertifiedEnergyRangeEv();
    auto const& geometry = warpx.Geom(0);
    try {
        m_seed_schedule.ValidateProduction(
            electron_range.min_eV,
            electron_range.max_eV,
            geometry.ProbLo(0),
            geometry.ProbHi(0),
            geometry.ProbLo(1),
            geometry.ProbHi(1));
    } catch (std::exception const& error) {
        amrex::Abort(
            std::string("RREA seed schedule production validation failed: ")
            + error.what());
    }
#endif
}

bool RreaWarpXCoupling::InteractionTablesRequired() const
{
    // Single authority shared with ValidateSeedSchedule: when this is false
    // the run performs no table-driven transport and loads no tables.
    return !m_debug_disable_transport
        && !m_debug_actual_continuity_fixture;
}

void RreaWarpXCoupling::LoadInteractionTables()
{
    if (m_interaction_tables_loaded) {
        return;
    }
    bool const needs_tables = InteractionTablesRequired();
    if (!needs_tables) {
        m_interaction_tables_loaded = true;
        return;
    }
    if (m_interaction_table_config_path.empty()) {
        amrex::Abort(
            "Schema-6 transport requires "
            "rrea.interaction_table_config unless debug fixed source is enabled");
    }
    m_interaction_tables.Load(m_interaction_table_config_path);
    if (m_transport_model.empty()
        || m_transport_model != m_interaction_tables.TransportModel()) {
        amrex::Abort(
            "rrea.transport_model must exactly match the loaded table configuration");
    }
    auto const electron_range =
        m_interaction_tables.ElectronCertifiedEnergyRangeEv();
    if (!std::isfinite(static_cast<double>(m_low_energy_cutoff_eV))) {
        m_low_energy_cutoff_eV = electron_range.min_eV;
    }
    amrex::Real const table_photon_cutoff_eV =
        m_interaction_tables.BremsAbsoluteCutoffEv();
    if (std::isfinite(static_cast<double>(m_photon_cutoff_eV))
        && m_photon_cutoff_eV != table_photon_cutoff_eV) {
        amrex::Abort(
            "rrea.photon_cutoff_eV does not match the selected interaction "
            "table configuration's brems_absolute_cutoff_eV");
    }
    m_photon_cutoff_eV = table_photon_cutoff_eV;
    m_ledger.low_energy_cutoff_eV = m_low_energy_cutoff_eV;
    if (!RreaTransportSubcycleControlsValid(
            m_transport_config,
            /*require_strict_guards=*/true)) {
        amrex::Abort(
            "Loaded schema-6 production tables require strict, finite positive "
            "transport subcycle controls");
    }
    amrex::Real const configured_hard_moller_threshold =
        m_interaction_tables.HardMollerSecondaryThresholdEv();
    if (configured_hard_moller_threshold > amrex::Real(0.0)) {
        if (m_hard_moller_secondary_threshold_input_set) {
            amrex::Real const denom =
                amrex::max(std::abs(configured_hard_moller_threshold), amrex::Real(1.0));
            if (std::abs(m_hard_moller_secondary_threshold_eV - configured_hard_moller_threshold)
                / denom > amrex::Real(1.0e-9)) {
                amrex::Abort(
                    "rrea.hard_moller_secondary_threshold_eV does not match "
                    "the selected interaction-table transport configuration");
            }
        } else {
            m_hard_moller_secondary_threshold_eV = configured_hard_moller_threshold;
        }
    }
    // The deck's scalar rrea.transport_density_ratio is relative to the density
    // the tables were generated at, so the fluid -- whose reduced mobilities are
    // quoted at Loschmidt -- needs it re-expressed against that density instead.
    // The transport reference density is available only after table parsing.
    m_advance_config.fluid.transport_density_ratio =
        m_transport_density_ratio * TransportReferenceDensityKgM3()
        / kReducedMobilityReferenceDensityKgM3;
    // Hard contract, not a hint: low_energy_cutoff_eV < hard_moller_secondary
    // _threshold_eV.  Hard secondaries are sampled only at or above the
    // threshold while restricted dE/dx already carries everything below it as
    // continuous drag, so an inverted configuration counts that energy twice --
    // once as drag, once as a born-then-dropped particle -- and silently loses
    // ion-pair yield.  The threshold is TABLE-defined and the production deck
    // does not set it, so a bundle built at a different Tcut can invert this
    // with the deck untouched.  A warning in a 64-rank log is not a guard.
    if (m_hard_moller_secondary_threshold_eV > amrex::Real(0.0)
        && m_low_energy_cutoff_eV >= m_hard_moller_secondary_threshold_eV) {
        std::ostringstream fatal;
        fatal << "RREA config: rrea.low_energy_cutoff_eV (" << m_low_energy_cutoff_eV
              << ") >= hard_moller_secondary_threshold_eV ("
              << m_hard_moller_secondary_threshold_eV
              << "); hard secondaries would spawn at or below the transport "
                 "cutoff and their energy would be double-counted against "
                 "restricted dE/dx -- lower the cutoff or use a bundle whose "
                 "threshold is above it";
        amrex::Abort(fatal.str());
    }

    // Schema 6 has one inclusive certified range per species and every
    // production interpolation is strict: there is no endpoint clamping or
    // extrapolation to warn about.  Validate the configured entry energies
    // against those authorities before transport/state mutation or RNG use.
    auto require_in_certified_range = [](
        char const* setting,
        amrex::Real value_eV,
        char const* species,
        rrea::RreaCertifiedEnergyRange const& range) {
        if (!std::isfinite(static_cast<double>(value_eV))
            || !range.Contains(value_eV)) {
            std::ostringstream message;
            message << setting << "=" << std::setprecision(17) << value_eV
                    << " eV is outside the schema-6 " << species
                    << " certified inclusive range [" << range.min_eV << ", "
                    << range.max_eV << "] eV; clamping and extrapolation are forbidden";
            amrex::Abort(message.str());
        }
    };
    auto const positron_range = m_interaction_tables.PositronCertifiedEnergyRangeEv();
    auto const photon_range = m_interaction_tables.PhotonCertifiedEnergyRangeEv();
    require_in_certified_range(
        "rrea.low_energy_cutoff_eV", m_low_energy_cutoff_eV,
        "electron", electron_range);
    require_in_certified_range(
        "rrea.low_energy_cutoff_eV", m_low_energy_cutoff_eV,
        "positron", positron_range);
    require_in_certified_range(
        "rrea.photon_cutoff_eV", m_photon_cutoff_eV,
        "photon", photon_range);
    require_in_certified_range(
        "rrea.hard_moller_secondary_threshold_eV",
        m_hard_moller_secondary_threshold_eV,
        "electron", electron_range);
    require_in_certified_range(
        "rrea.hard_moller_secondary_threshold_eV",
        m_hard_moller_secondary_threshold_eV,
        "positron", positron_range);
    m_interaction_tables_loaded = true;
}

void RreaWarpXCoupling::LoadPendingCheckpointIfAny()
{
    if (!m_pending_checkpoint_dir.empty()) {
        std::string const nested = join_path(m_pending_checkpoint_dir, "rrea_checkpoint");
        std::string const root = amrex::FileExists(join_path(nested, "metadata.json"))
            ? nested
            : m_pending_checkpoint_dir;
        std::filesystem::path const complete_path =
            std::filesystem::path(root) / "COMPLETE";
        if (!std::filesystem::is_regular_file(complete_path)
            || std::filesystem::is_symlink(complete_path)) {
            amrex::Abort("RREA checkpoint v2 is incomplete (missing COMPLETE marker)");
        }
        if (read_binary_file(complete_path.string()) != checkpoint_complete_marker) {
            amrex::Abort(
                "RREA checkpoint v2 has invalid COMPLETE marker bytes; expected "
                "exactly rrea_checkpoint_v2 followed by LF");
        }
        ReadMetadata(root);
        ReadMeshFields(root);
        ReadRankState(root);
        ++m_checkpoint_read_count;
        amrex::Print() << "RREA checkpoint restored from "
                       << m_pending_checkpoint_dir
                       << ", next_seed_event=" << m_next_seed_event
                       << ", injected_macro_count=" << m_injected_macro_count
                       << "\n";
        m_pending_checkpoint_dir.clear();
    }
}

}  // namespace rrea::warpx
