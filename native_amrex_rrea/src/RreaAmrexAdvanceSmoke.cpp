#include "rrea/RreaAmrexAdvance.H"
#include "rrea/RreaSmokeRequire.H"
#include "rrea/RreaConstants.H"

#include <AMReX.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Print.H>

#include <limits>
#include <stdexcept>
#include <vector>

namespace {

struct TestDomain {
    amrex::Geometry geom;
    amrex::BoxArray ba;
    amrex::DistributionMapping dm;
};

TestDomain make_domain()
{
    amrex::IntVect lo(0);
    amrex::IntVect hi(0);
    hi[0] = 15;
#if (AMREX_SPACEDIM >= 2)
    hi[1] = 31;
#endif
    amrex::Box domain(lo, hi);
    amrex::RealBox real_box;
    real_box.setLo(0, 0.0);
    real_box.setHi(0, 16.0);
#if (AMREX_SPACEDIM >= 2)
    real_box.setLo(1, 0.0);
    real_box.setHi(1, 32.0);
#endif
    amrex::Vector<int> periodic(AMREX_SPACEDIM, 0);
    amrex::Geometry geom(domain, &real_box, 0, periodic.data());
    amrex::BoxArray ba(domain);
    ba.maxSize(8);
    return {geom, ba, amrex::DistributionMapping(ba)};
}

using rrea::smoke::require;
using rrea::smoke::require_throws;

void run_maxwell_material_continuity_smoke()
{
    auto domain = make_domain();
    amrex::MultiFab rho(domain.ba, domain.dm, 1, 0);
    amrex::MultiFab er(domain.ba, domain.dm, 1, 0);
    amrex::MultiFab ez(domain.ba, domain.dm, 1, 0);
    rho.setVal(0.0);
    rrea::RreaAmrexLevelBinding binding{
        domain.geom, domain.ba, domain.dm, &rho, &er, &ez, nullptr};
    rrea::RreaAmrexAdvance advance({binding}, 1);
    rrea::RreaAmrexAdvanceConfig config;
    config.maximum_represented_charge_e = 1.0;
    config.fluid.electron_drift_enable = true;
    config.fluid.ion_drift_enable = true;
    config.fluid.attachment_frequency_s = 0.0;
    config.fluid.ion_ion_recombination_coefficient_m3_per_s = 0.0;
    advance.InitializeBackgroundState(config.field_initializer);
    advance.PrimeMaxwellContinuityReference();

    int constexpr pad = 2;
    int const nr = domain.geom.Domain().length(0);
    int const nz = domain.geom.Domain().length(1);
    std::vector<double> jr(
        static_cast<std::size_t>(nr + pad + 1) * (nz + 2 * pad), 0.0);
    std::vector<double> jz(
        static_cast<std::size_t>(nr + pad) * (nz + 2 * pad + 1), 0.0);
    double constexpr stable_dt_s = 1.0e-9;
    advance.BindKineticCurrent(&jr, &jz, pad, rrea::qe);
    advance.AdvanceOneStep(stable_dt_s, config);
    require(
        advance.LastDiagnostics().maxwell_charge_continuity_max_relative
            <= 1.0e-14,
        "zero-source Maxwell step failed material continuity");

    // The material-continuity norm is mixed relative/absolute: its absolute
    // scale is the smallest normal double, so physically empty subnormal
    // charge tails cannot become false O(1) relative defects.
    rho.setVal(std::numeric_limits<double>::denorm_min());
    advance.BindKineticCurrent(&jr, &jz, pad, rrea::qe);
    advance.AdvanceOneStep(stable_dt_s, config);
    require(
        advance.LastDiagnostics().maxwell_charge_continuity_max_relative
            <= 1.0e-14,
        "subnormal Maxwell charge tail produced a false continuity defect");

    // Exercise two distinct subnormal magnitudes, not only a one-ULP creation
    // from zero.
    rho.setVal(-3.55e-320);
    advance.PrimeMaxwellContinuityReference();
    rho.setVal(-8.32e-320);
    advance.BindKineticCurrent(&jr, &jz, pad, rrea::qe);
    advance.AdvanceOneStep(stable_dt_s, config);
    require(
        advance.LastDiagnostics().maxwell_charge_continuity_max_relative
            <= rrea::kMaterialContinuityRelativeTolerance,
        "distinct subnormal Maxwell charge tails exceeded the absolute floor");

    // A full 1 ns outer step would have dt*sigma/eps0=0.25.  Adaptive shared
    // coupling must choose four steps (0.0625 each) without weakening the
    // per-substep 0.1 bound or losing material continuity.
    double constexpr full_step_relaxation = 0.25;
    double const sigma = full_step_relaxation * rrea::kMaxwellEps0
        / stable_dt_s;
    double const ne = sigma / (rrea::qe * config.fluid.electron_mobility_m2_per_vs);
    rho.setVal(0.0);
    advance.Fluid(0).SetVal(0.0);
    advance.Fluid(0).LowElectronDensity().setVal(ne);
    advance.Fluid(0).FluidChargeDensity().setVal(-rrea::qe * ne);
    advance.PrimeMaxwellContinuityReference();
    advance.BindKineticCurrent(&jr, &jz, pad, rrea::qe);
    advance.AdvanceOneStep(stable_dt_s, config);
    require(
        std::abs(
            advance.LastDiagnostics().maxwell_max_dt_sigma_face_over_eps0
            - 0.25 * full_step_relaxation) < 1.0e-12,
        "fluid/Maxwell coupling chose the wrong conductive substep size");
    require(
        advance.LastDiagnostics().maxwell_charge_continuity_max_relative
            <= rrea::kMaterialContinuityRelativeTolerance,
        "time-averaged substep current failed full-step continuity");

    // A 0.25 outer-step reaction number selects four material steps at the
    // default 0.08 target; symmetric half reactions preserve exact constant
    // attachment in the zero-field/no-transport limit.
    advance.Fluid(0).SetVal(0.0);
    advance.Fluid(0).LowElectronDensity().setVal(1.0);
    advance.Fluid(0).FluidChargeDensity().setVal(-rrea::qe);
    config.fluid.attachment_frequency_s = 0.25 / stable_dt_s;
    advance.PrimeMaxwellContinuityReference();
    advance.BindKineticCurrent(&jr, &jz, pad, rrea::qe);
    advance.AdvanceOneStep(stable_dt_s, config);
    require(advance.LastDiagnostics().material_substeps == 4,
            "reaction stiffness selected the wrong material substep count");
    require(std::abs(advance.LastDiagnostics().fluid_max_reaction_number - 0.0625)
                < 1.0e-14,
            "realized reaction stiffness diagnostic is wrong");
    require(
        std::abs(advance.Fluid(0).LowElectronDensity().min(0, 0, false)
                 - std::exp(-0.25)) < 1.0e-14,
        "symmetric material substeps changed exact constant attachment");
    config.fluid.attachment_frequency_s = 0.0;

    advance.Fluid(0).SetVal(0.0);
    rho.setVal(0.0);
    advance.PrimeMaxwellContinuityReference();
    rho.setVal(1.0e-12);
    advance.BindKineticCurrent(&jr, &jz, pad, rrea::qe);
    require_throws(
        [&]() { advance.AdvanceOneStep(stable_dt_s, config); },
        "Maxwell accepted charge creation without current");

    rho.setVal(0.0);
    advance.PrimeMaxwellContinuityReference();
    advance.BindKineticCurrent(&jr, &jz, pad, rrea::qe);
    require_throws(
        [&]() { advance.AdvanceOneStep(3.0e-9, config); },
        "Maxwell accepted a timestep above the two-dimensional Yee CFL");

    config.material_conduction_target = 0.2;
    advance.BindKineticCurrent(&jr, &jz, pad, rrea::qe);
    require_throws(
        [&]() { advance.AdvanceOneStep(stable_dt_s, config); },
        "Maxwell accepted a material target above its guard");
}

// The particle gather must reproduce the committed cell field exactly: the
// same bilinear of the cell-centred averages, with Er continued oddly to zero
// on the axis.
void run_gather_total_field_smoke()
{
    auto domain = make_domain();
    amrex::MultiFab rho(domain.ba, domain.dm, 1, 0);
    amrex::MultiFab er(domain.ba, domain.dm, 1, 0);
    amrex::MultiFab ez(domain.ba, domain.dm, 1, 0);
    rho.setVal(0.0);
    rrea::RreaAmrexLevelBinding binding{
        domain.geom, domain.ba, domain.dm, &rho, &er, &ez, nullptr};
    rrea::RreaAmrexAdvance advance({binding}, 1);
    rrea::RreaAmrexAdvanceConfig config;
    advance.InitializeBackgroundState(config.field_initializer);
    auto const dx = domain.geom.CellSizeArray();
    auto const cell = [&](amrex::MultiFab const& field, int i, int j) {
        double value = 0.0;
        for (amrex::MFIter mfi(field); mfi.isValid(); ++mfi) {
            if (mfi.validbox().contains(amrex::IntVect(AMREX_D_DECL(i, j, 0)))) {
                value = field[mfi](amrex::IntVect(AMREX_D_DECL(i, j, 0)), 0);
            }
        }
        amrex::ParallelDescriptor::ReduceRealSum(value);
        return value;
    };
    // Binary-exact offsets, so the gathered value must reproduce the cell
    // bilinear to the last bit rather than to an interpolation tolerance.
    int constexpr i = 4;
    int constexpr j = 7;
    for (double const frac : {0.25, 0.5, 0.75}) {
        double gathered[3] = {0.0, 0.0, 0.0};
        advance.GatherTotalFieldRZ(
            (i + 0.5 + frac) * dx[0], (j + 0.5 + frac) * dx[1],
            gathered[0], gathered[1], gathered[2]);
        char const* const names[2] = {
            "gathered Er does not match the committed cell field",
            "gathered Ez does not match the committed cell field"};
        amrex::MultiFab const* const fields[2] = {&er, &ez};
        for (int component = 0; component < 2; ++component) {
            auto const& f = *fields[component];
            double const f00 = cell(f, i, j);
            double const f01 = cell(f, i, j + 1);
            double const f10 = cell(f, i + 1, j);
            double const f11 = cell(f, i + 1, j + 1);
            double const expect =
                (1.0 - frac) * ((1.0 - frac) * f00 + frac * f01)
                + frac * ((1.0 - frac) * f10 + frac * f11);
            require(std::abs(gathered[component] - expect)
                        <= 1.0e-15 * std::max(std::abs(expect), 1.0),
                    names[component]);
        }
    }
    // Inside the first half cell Er is the odd continuation of the first cell
    // centre, so it must vanish exactly on the axis.
    double axis[3] = {1.0, 0.0, 0.0};
    advance.GatherTotalFieldRZ(0.0, 8.5 * dx[1], axis[0], axis[1], axis[2]);
    require(axis[0] == 0.0, "gathered Er is not odd-continued to zero on the axis");
    advance.GatherTotalFieldRZ(
        0.25 * dx[0], 8.5 * dx[1], axis[0], axis[1], axis[2]);
    double const first_cell_er = cell(er, 0, 8);
    require(std::abs(axis[0] - 0.5 * first_cell_er)
                <= 1.0e-15 * std::max(std::abs(first_cell_er), 1.0),
            "the odd radial axis scale is not applied inside r=dr/2");
}

void run_conditioned_continuity_case(
    double maximum_charge_e,
    double particle_charge_e,
    double fluid_charge_e,
    double residual_charge_e,
    bool should_pass,
    double source_pair_e = 0.0,
    double source_imbalance_e = 0.0)
{
    auto domain = make_domain();
    amrex::MultiFab rho(domain.ba, domain.dm, 1, 0);
    amrex::MultiFab er(domain.ba, domain.dm, 1, 0);
    amrex::MultiFab ez(domain.ba, domain.dm, 1, 0);
    rho.setVal(0.0);
    rrea::RreaAmrexLevelBinding binding{
        domain.geom, domain.ba, domain.dm, &rho, &er, &ez, nullptr};
    rrea::RreaAmrexAdvance advance({binding}, 1);
    rrea::RreaAmrexAdvanceConfig config;
    config.maximum_represented_charge_e = maximum_charge_e;
    config.fluid.electron_drift_enable = true;
    config.fluid.ion_drift_enable = true;
    config.fluid.attachment_frequency_s = 0.0;
    config.fluid.ion_ion_recombination_coefficient_m3_per_s = 0.0;
    advance.InitializeBackgroundState(config.field_initializer);
    advance.PrimeMaxwellContinuityReference();

    int constexpr target_i = 4;
    int constexpr target_j = 5;
    auto const dx = domain.geom.CellSizeArray();
    double const r_lo = target_i * dx[0];
    double const r_hi = r_lo + dx[0];
    double const volume = rrea::pi * (r_hi * r_hi - r_lo * r_lo) * dx[1];
    auto set_cell = [&](amrex::MultiFab& field, int i, int j, double value) {
        for (amrex::MFIter mfi(field); mfi.isValid(); ++mfi) {
            auto const bx = mfi.validbox();
            if (bx.contains(amrex::IntVect(AMREX_D_DECL(i, j, 0)))) {
                field[mfi](
                    amrex::IntVect(AMREX_D_DECL(i, j, 0)), 0) = value;
            }
        }
    };
    set_cell(
        rho, target_i, target_j, rrea::qe * particle_charge_e / volume);
    set_cell(
        advance.Fluid(0).FluidChargeDensity(),
        target_i,
        target_j,
        rrea::qe * fluid_charge_e / volume);

    int constexpr pad = 2;
    int const nr = domain.geom.Domain().length(0);
    int const nz = domain.geom.Domain().length(1);
    int const nr_pad = nr + pad;
    std::vector<double> jr(
        static_cast<std::size_t>(nr_pad + 1) * (nz + 2 * pad), 0.0);
    std::vector<double> jz(
        static_cast<std::size_t>(nr_pad) * (nz + 2 * pad + 1), 0.0);
    double constexpr dt_s = 1.0e-9;
    set_cell(
        advance.Fluid(0).LowElectronSourceRate(), target_i, target_j,
        source_pair_e / (volume * dt_s));
    set_cell(
        advance.Fluid(0).PositiveIonSourceRate(), target_i, target_j,
        (source_pair_e + source_imbalance_e) / (volume * dt_s));
    double const net_charge_e = particle_charge_e + fluid_charge_e;
    std::size_t const upper_face = static_cast<std::size_t>(target_i)
        + static_cast<std::size_t>(nr_pad) * (target_j + pad + 1);
    if (amrex::ParallelDescriptor::IOProcessor()) {
        jz[upper_face] = (-net_charge_e + residual_charge_e) / dt_s;
    }
    set_cell(
        rho,
        target_i,
        target_j + 1,
        rrea::qe * (-net_charge_e + residual_charge_e) / volume);
    advance.BindKineticCurrent(&jr, &jz, pad, rrea::qe);

    if (should_pass) {
        advance.AdvanceOneStep(dt_s, config);
        auto const& diagnostics = advance.LastDiagnostics();
        require(
            diagnostics.maxwell_charge_continuity_max_relative
                <= rrea::kMaterialContinuityRelativeTolerance,
            "conditioned continuity fixture rejected a representational residual");
        if (source_pair_e == 0.0) {
            require(
                std::abs(
                    diagnostics.maxwell_charge_continuity_max_abs_e_per_cell
                    - std::abs(residual_charge_e))
                    <= 1.0e-8 * std::max(std::abs(residual_charge_e), 1.0),
                "conditioned continuity fixture reported the wrong absolute residual");
        } else {
            require(
                diagnostics.maxwell_charge_continuity_max_abs_e_per_cell
                    > rrea::kMaterialContinuityAbsoluteMacroFraction
                        * maximum_charge_e,
                "source-conditioned fixture did not exceed the absolute floor");
            require(
                advance.Fluid(0).LowElectronSourceRate().max(0, 0, false) == 0.0
                    && advance.Fluid(0).PositiveIonSourceRate().max(0, 0, false) == 0.0,
                "source rates were not cleared after the continuity gate");
        }
    } else {
        require_throws(
            [&]() { advance.AdvanceOneStep(dt_s, config); },
            "conditioned continuity fixture accepted an unrepresented charge");
    }
}

void run_conditioned_continuity_smoke()
{
    // Express a cancellation-limited case in units of represented macro charge
    // so the fixture is independent of any run's seed count.
    double constexpr maximum_charge_e = 1.0;
    double const allowance =
        rrea::kMaterialContinuityAbsoluteMacroFraction * maximum_charge_e;
    double constexpr particle_e = -0.020957538863424576;
    double constexpr fluid_e = 0.02191829374705534;
    double constexpr observed_residual_e = 2.8093064505064065e-13;
    run_conditioned_continuity_case(
        maximum_charge_e, particle_e, fluid_e, observed_residual_e, true);
    run_conditioned_continuity_case(
        maximum_charge_e, particle_e, fluid_e, 0.99 * allowance, true);
    run_conditioned_continuity_case(
        maximum_charge_e, particle_e, fluid_e, 1.01 * allowance, false);
    run_conditioned_continuity_case(
        maximum_charge_e, particle_e, fluid_e, maximum_charge_e, false);

    // An eps-scale residue from a heavy macro can land in a quiet cell. The
    // absolute floor must scale with that macro, not with the initial seed.
    double constexpr heavy_macro_charge_e = 3.470060620638e12;
    double constexpr seed_macro_charge_e = 364.898070162699887;
    double constexpr quiet_cell_residual_e = 7.77174759244298044e-05;
    double constexpr quiet_cell_particle_e = -2.468e5;
    run_conditioned_continuity_case(
        heavy_macro_charge_e, quiet_cell_particle_e, 0.0, quiet_cell_residual_e, true);
    run_conditioned_continuity_case(
        seed_macro_charge_e, quiet_cell_particle_e, 0.0, quiet_cell_residual_e, false);
    run_conditioned_continuity_case(
        heavy_macro_charge_e, quiet_cell_particle_e, 0.0,
        1.01 * rrea::kMaterialContinuityAbsoluteMacroFraction
            * heavy_macro_charge_e,
        false);

    // Large, separately accumulated charge-neutral source channels carry a
    // binary64 cancellation scale even when their net charge is nearly zero.
    run_conditioned_continuity_case(
        seed_macro_charge_e, 0.0, 0.0, 0.0, true, 3.0e11, 1.0e-3);
    run_conditioned_continuity_case(
        seed_macro_charge_e, 0.0, 0.0, 0.0, false, 3.0e11, seed_macro_charge_e);
}

}  // namespace

int main(int argc, char** argv)
{
    amrex::Initialize(argc, argv);
    int result = 0;
    try {
        run_maxwell_material_continuity_smoke();
        run_gather_total_field_smoke();
        run_conditioned_continuity_smoke();
        amrex::Print() << "rrea_amrex_advance_smoke passed\n";
    } catch (std::exception const& exc) {
        result = 1;
        amrex::AllPrint() << "rrea_amrex_advance_smoke failed: "
                          << exc.what() << "\n";
    }
    amrex::Finalize();
    return result;
}
