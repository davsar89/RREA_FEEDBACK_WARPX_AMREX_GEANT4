#include "rrea/LowEnergyFluidState.H"
#include "rrea/RreaConstants.H"
#include "rrea/RreaSmokeRequire.H"

#include <AMReX.H>
#include <AMReX_BoxArray.H>
#include <AMReX_BoxList.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

struct TestDomain {
    amrex::Geometry geom;
    amrex::BoxArray ba;
    amrex::DistributionMapping dm;
};

enum class BoxLayout {
    Single,
    RadialStripes,
    AxialStripes,
    Tiles,
    ReversedTiles
};

using FaceFields =
    amrex::Array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM>;

using rrea::smoke::require;
using rrea::smoke::require_throws;

amrex::BoxArray make_box_array(amrex::Box const& domain, BoxLayout layout)
{
    if (layout == BoxLayout::Single) {
        return amrex::BoxArray(domain);
    }
    amrex::Vector<amrex::Box> boxes;
    int constexpr block = 4;
    int const r_lo = domain.smallEnd(0);
    int const r_hi = domain.bigEnd(0);
#if (AMREX_SPACEDIM >= 2)
    int const z_lo = domain.smallEnd(1);
    int const z_hi = domain.bigEnd(1);
#else
    int const z_lo = 0;
    int const z_hi = 0;
#endif
    int const r_step = layout == BoxLayout::AxialStripes ? r_hi - r_lo + 1 : block;
    int const z_step = layout == BoxLayout::RadialStripes ? z_hi - z_lo + 1 : block;
    for (int radial = r_lo; radial <= r_hi; radial += r_step) {
        for (int axial = z_lo; axial <= z_hi; axial += z_step) {
            amrex::IntVect lo(AMREX_D_DECL(radial, axial, 0));
            amrex::IntVect hi(AMREX_D_DECL(
                std::min(radial + r_step - 1, r_hi),
                std::min(axial + z_step - 1, z_hi), 0));
            boxes.emplace_back(lo, hi, domain.ixType());
        }
    }
    if (layout == BoxLayout::ReversedTiles) {
        std::reverse(boxes.begin(), boxes.end());
    }
    return amrex::BoxArray(amrex::BoxList(std::move(boxes)));
}

TestDomain make_domain(
    bool periodic_z,
    BoxLayout layout = BoxLayout::Tiles)
{
    amrex::IntVect lo(0);
    amrex::IntVect hi(0);
    hi[0] = 15;
#if (AMREX_SPACEDIM >= 2)
    hi[1] = 31;
#endif
    amrex::Box const domain(lo, hi);
    amrex::RealBox real_box;
    real_box.setLo(0, 0.0);
    real_box.setHi(0, 16.0);
#if (AMREX_SPACEDIM >= 2)
    real_box.setLo(1, 0.0);
    real_box.setHi(1, 32.0);
#endif
    amrex::Vector<int> periodic(AMREX_SPACEDIM, 0);
#if (AMREX_SPACEDIM >= 2)
    periodic[1] = periodic_z ? 1 : 0;
#else
    amrex::ignore_unused(periodic_z);
#endif
    amrex::Geometry geom(domain, &real_box, 0, periodic.data());
    amrex::BoxArray ba = make_box_array(domain, layout);
    return {geom, ba, amrex::DistributionMapping(ba)};
}

FaceFields make_face_fields(
    amrex::BoxArray const& ba,
    amrex::DistributionMapping const& dm,
    int components)
{
    FaceFields result;
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        amrex::BoxArray face_ba = ba;
        face_ba.surroundingNodes(dir);
        result[dir] = std::make_unique<amrex::MultiFab>(
            face_ba, dm, components, 0);
        result[dir]->setVal(0.0);
    }
    return result;
}

double cell_volume(amrex::Geometry const& geom, int i)
{
    auto const dx = geom.CellSizeArray();
    int const radial_lo = geom.Domain().smallEnd(0);
    double const r_lo = geom.ProbLo(0)
        + static_cast<double>(i - radial_lo) * dx[0];
    double const r_hi = r_lo + dx[0];
#if (AMREX_SPACEDIM >= 2)
    double const dz = dx[1];
#else
    double const dz = 1.0;
#endif
    return rrea::pi * (r_hi * r_hi - r_lo * r_lo) * dz;
}

double total_number(
    amrex::MultiFab const& density,
    amrex::Geometry const& geom)
{
    double result = 0.0;
    for (amrex::MFIter mfi(density); mfi.isValid(); ++mfi) {
        auto const bx = mfi.validbox();
        auto const n = density.const_array(mfi);
        amrex::LoopOnCpu(bx, [&](int i, int j, int k) noexcept {
            amrex::ignore_unused(j, k);
            result += static_cast<double>(n(i, j, k)) * cell_volume(geom, i);
        });
    }
    amrex::ParallelDescriptor::ReduceRealSum(result);
    return result;
}

void set_cell(amrex::MultiFab& field, int i, int j, double value)
{
    for (amrex::MFIter mfi(field); mfi.isValid(); ++mfi) {
        auto const bx = mfi.validbox();
        if (bx.contains(amrex::IntVect(AMREX_D_DECL(i, j, 0)))) {
            field[mfi](amrex::IntVect(AMREX_D_DECL(i, j, 0)), 0) = value;
        }
    }
}

double cell_value(amrex::MultiFab const& field, int i, int j)
{
    double value = 0.0;
    double count = 0.0;
    amrex::IntVect const iv(AMREX_D_DECL(i, j, 0));
    for (amrex::MFIter mfi(field); mfi.isValid(); ++mfi) {
        if (mfi.validbox().contains(iv)) {
            value += static_cast<double>(field[mfi](iv, 0));
            count += 1.0;
        }
    }
    double reduced[2] = {value, count};
    amrex::ParallelDescriptor::ReduceRealSum(reduced, 2);
    require(reduced[1] == 1.0, "cell sample did not have exactly one owner");
    return reduced[0];
}

struct FaceCopies {
    double minimum = std::numeric_limits<double>::infinity();
    double maximum = -std::numeric_limits<double>::infinity();
    double count = 0.0;
};

FaceCopies face_copies(
    amrex::MultiFab const& field,
    int i,
    int j,
    int component)
{
    FaceCopies result;
    amrex::IntVect const iv(AMREX_D_DECL(i, j, 0));
    for (amrex::MFIter mfi(field); mfi.isValid(); ++mfi) {
        if (mfi.validbox().contains(iv)) {
            double const value = static_cast<double>(field[mfi](iv, component));
            result.minimum = std::min(result.minimum, value);
            result.maximum = std::max(result.maximum, value);
            result.count += 1.0;
        }
    }
    amrex::ParallelDescriptor::ReduceRealMin(result.minimum);
    amrex::ParallelDescriptor::ReduceRealMax(result.maximum);
    amrex::ParallelDescriptor::ReduceRealSum(result.count);
    require(result.count > 0.0, "face sample was absent from the global mesh");
    return result;
}

double max_species_charge_error(rrea::LowEnergyFluidState const& fluid)
{
    double error = 0.0;
    auto const& rho = fluid.FluidChargeDensity();
    for (amrex::MFIter mfi(rho); mfi.isValid(); ++mfi) {
        auto const bx = mfi.validbox();
        auto const charge = rho.const_array(mfi);
        auto const positive = fluid.PositiveIonDensity().const_array(mfi);
        auto const negative = fluid.NegativeIonDensity().const_array(mfi);
        auto const electron = fluid.LowElectronDensity().const_array(mfi);
        amrex::LoopOnCpu(bx, [&](int i, int j, int k) noexcept {
            double const expected = rrea::qe * (
                static_cast<double>(positive(i, j, k))
                - static_cast<double>(negative(i, j, k))
                - static_cast<double>(electron(i, j, k)));
            error = std::max(
                error,
                std::abs(static_cast<double>(charge(i, j, k)) - expected));
        });
    }
    amrex::ParallelDescriptor::ReduceRealMax(error);
    return error;
}

double electron_boundary_outflow(
    FaceFields const& flux,
    amrex::Geometry const& geom)
{
    auto const domain = geom.Domain();
    auto const dx = geom.CellSizeArray();
    double result = 0.0;
    int const outer_i = domain.bigEnd(0) + 1;
    double const outer_r = geom.ProbHi(0);
#if (AMREX_SPACEDIM >= 2)
    double const dz = dx[1];
#else
    double const dz = 1.0;
#endif
    for (amrex::MFIter mfi(*flux[0]); mfi.isValid(); ++mfi) {
        auto const bx = mfi.validbox();
        auto const values = flux[0]->const_array(mfi);
        if (outer_i >= bx.smallEnd(0) && outer_i <= bx.bigEnd(0)) {
            int const j_lo = bx.smallEnd(1);
            int const j_hi = bx.bigEnd(1);
            for (int j = j_lo; j <= j_hi; ++j) {
                result += static_cast<double>(values(outer_i, j, 0, 2))
                    * (2.0 * rrea::pi * outer_r * dz);
            }
        }
    }
#if (AMREX_SPACEDIM >= 2)
    int const lower_j = domain.smallEnd(1);
    int const upper_j = domain.bigEnd(1) + 1;
    for (amrex::MFIter mfi(*flux[1]); mfi.isValid(); ++mfi) {
        auto const bx = mfi.validbox();
        auto const values = flux[1]->const_array(mfi);
        for (int boundary_j : {lower_j, upper_j}) {
            if (boundary_j < bx.smallEnd(1) || boundary_j > bx.bigEnd(1)) {
                continue;
            }
            for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                int const radial_lo = domain.smallEnd(0);
                double const r_lo = geom.ProbLo(0)
                    + static_cast<double>(i - radial_lo) * dx[0];
                double const r_hi = r_lo + dx[0];
                double const area = rrea::pi * (r_hi * r_hi - r_lo * r_lo);
                double const sign = boundary_j == lower_j ? -1.0 : 1.0;
                result += sign
                    * static_cast<double>(values(i, boundary_j, 0, 2)) * area;
            }
        }
    }
#endif
    amrex::ParallelDescriptor::ReduceRealSum(result);
    return result;
}

rrea::LowEnergyFluidConfig electron_config(bool table)
{
    rrea::LowEnergyFluidConfig config;
    config.electron_drift_enable = true;
    config.ion_drift_enable = false;
    config.electron_mobility_model = table ? "en_table_flux_v1" : "absolute";
    config.electron_mobility_m2_per_vs = table ? 0.0 : 1.0;
    config.ion_drift_cfl = 0.5;
    return config;
}

void run_transport_vacuum_and_seam_test()
{
    auto domain = make_domain(true);
    rrea::LowEnergyFluidState fluid(domain.ba, domain.dm, 1);
    auto efield = make_face_fields(domain.ba, domain.dm, 1);
    auto flux = make_face_fields(domain.ba, domain.dm, 3);
#if (AMREX_SPACEDIM >= 2)
    efield[1]->setVal(-1000.0); // electron velocity +1000 m/s
#endif
    auto& electron = fluid.LowElectronDensity();
    electron.setVal(0.0);

    double const floor = rrea::carrier_transport_vacuum_density_m3;
    struct Sample {
        int j;
        double density;
    };
    std::array<Sample, 4> const vacuum_samples = {{
        {8, std::numeric_limits<double>::denorm_min()},
        {12, 1.58e-319},
        {16, std::nextafter(floor, 0.0)},
        {20, floor}}};
    int constexpr sample_i = 5;
    for (auto const& sample : vacuum_samples) {
        set_cell(electron, sample_i, sample.j, sample.density);
    }
    // j=3 ends a maxSize(4) cell box; its outgoing j=4 nodal face is shared.
    double const active_density = std::nextafter(floor, 2.0 * floor);
    set_cell(electron, sample_i, 3, active_density);
    fluid.ComputeFluidChargeDensity();

    std::array<double, 4> density_before{};
    std::array<double, 4> charge_before{};
    for (std::size_t n = 0; n < vacuum_samples.size(); ++n) {
        density_before[n] = cell_value(electron, sample_i, vacuum_samples[n].j);
        charge_before[n] = cell_value(
            fluid.FluidChargeDensity(), sample_i, vacuum_samples[n].j);
    }
    double const number_before = total_number(electron, domain.geom);

    auto const config = electron_config(false);
    double constexpr dt = 1.0;
    rrea::LowEnergyFaceSyncDiagnostics sync_diagnostics;
    fluid.PrepareExplicitCarrierTransport(domain.geom, config);
    auto const build = fluid.BuildExplicitFaceTransport(
        dt, domain.geom, amrex::GetArrOfConstPtrs(efield), config,
        amrex::GetArrOfPtrs(flux));
    require(build.max_explicit_diffusion_number == 0.0,
        "absolute-mobility transport reported a diffusion number");
    fluid.ApplyExplicitCarrierTransport(
        dt, domain.geom, amrex::GetArrOfPtrs(flux), config,
        nullptr, false, &sync_diagnostics);
    require(
        sync_diagnostics.max_pre_sync_duplicate_difference == 0.0,
        "independent limited face copies disagreed before OverrideSync");

    for (std::size_t n = 0; n < vacuum_samples.size(); ++n) {
        require(
            cell_value(electron, sample_i, vacuum_samples[n].j)
                    == density_before[n]
                && cell_value(
                    fluid.FluidChargeDensity(), sample_i, vacuum_samples[n].j)
                    == charge_before[n],
            "transport-vacuum state or conservative charge was erased");
        auto const copies = face_copies(
            *flux[1], sample_i, vacuum_samples[n].j + 1, 2);
        require(copies.minimum == 0.0 && copies.maximum == 0.0,
            "transport-vacuum donor emitted a nonzero face flux");
    }

    auto const seam = face_copies(*flux[1], sample_i, 4, 2);
    require(seam.count >= 2.0 && seam.minimum > 0.0,
        "limiter-active seam face was not represented by duplicate copies");
    require(
        std::abs(seam.maximum - seam.minimum)
            <= 16.0 * std::numeric_limits<double>::epsilon() * seam.maximum,
        "OverrideSync did not make shared limited face copies identical");

    double const number_after = total_number(electron, domain.geom);
    require(
        std::abs(number_after - number_before)
            <= 1.0e-12 * std::max(number_before, floor),
        "periodic limiter path did not conserve electron number");
    auto const& drift = fluid.LastCarrierDriftDiagnostics();
    require(
        drift.min_outflow_scale > 0.0
            && drift.min_outflow_scale < 1.0
            && cell_value(electron, sample_i, 3) >= 0.5 * active_density
            && fluid.LowElectronDensity().min(0, 0, false) >= 0.0,
        "normal-density limiter did not preserve the donor reserve");
    require(
        max_species_charge_error(fluid)
            <= 2048.0 * std::numeric_limits<double>::epsilon()
                * rrea::qe * active_density,
        "electron face current and conservative fluid charge diverged");
}

struct LimiterCaseResult {
    double active_face_flux = 0.0;
    double conserved_number_ratio = 0.0;
    double donor_remaining_fraction = 0.0;
    double charge_error = 0.0;
    double pre_sync_difference = 0.0;
    double active_face_copy_count = 0.0;
};

LimiterCaseResult run_limiter_layout_case(
    BoxLayout layout,
    int direction,
    int velocity_sign)
{
    auto domain = make_domain(true, layout);
    rrea::LowEnergyFluidState fluid(domain.ba, domain.dm, 1);
    auto efield = make_face_fields(domain.ba, domain.dm, 1);
    auto flux = make_face_fields(domain.ba, domain.dm, 3);
    efield[direction]->setVal(velocity_sign > 0 ? -1.0 : 1.0);

    int const face_i = direction == 0 ? 4 : 6;
    int const face_j = direction == 0 ? 10 : 4;
    int const donor_i = direction == 0 && velocity_sign > 0
        ? face_i - 1
        : face_i;
    int const donor_j = direction == 1 && velocity_sign > 0
        ? face_j - 1
        : face_j;
    fluid.LowElectronDensity().setVal(0.0);
    set_cell(fluid.LowElectronDensity(), donor_i, donor_j, 1.0);
    fluid.ComputeFluidChargeDensity();
    double const number_before = total_number(
        fluid.LowElectronDensity(), domain.geom);

    auto const config = electron_config(false);
    rrea::LowEnergyFaceSyncDiagnostics sync_diagnostics;
    fluid.PrepareExplicitCarrierTransport(domain.geom, config);
    fluid.BuildExplicitFaceTransport(
        1.0, domain.geom, amrex::GetArrOfConstPtrs(efield), config,
        amrex::GetArrOfPtrs(flux));
    fluid.ApplyExplicitCarrierTransport(
        1.0, domain.geom, amrex::GetArrOfPtrs(flux), config,
        nullptr, false, &sync_diagnostics);

    auto const active = face_copies(
        *flux[direction], face_i, face_j, 2);
    double const number_after = total_number(
        fluid.LowElectronDensity(), domain.geom);
    return LimiterCaseResult{
        active.maximum,
        number_after / number_before,
        cell_value(fluid.LowElectronDensity(), donor_i, donor_j),
        max_species_charge_error(fluid),
        sync_diagnostics.max_pre_sync_duplicate_difference,
        active.count};
}

void run_layout_and_sign_matrix()
{
    std::array<BoxLayout, 5> const layouts = {
        BoxLayout::Single,
        BoxLayout::RadialStripes,
        BoxLayout::AxialStripes,
        BoxLayout::Tiles,
        BoxLayout::ReversedTiles};
    for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
        for (int velocity_sign : {-1, 1}) {
            LimiterCaseResult const reference = run_limiter_layout_case(
                BoxLayout::Single, direction, velocity_sign);
            for (BoxLayout layout : layouts) {
                LimiterCaseResult const result = layout == BoxLayout::Single
                    ? reference
                    : run_limiter_layout_case(layout, direction, velocity_sign);
                require(
                    result.pre_sync_difference == 0.0,
                    "BoxArray layout produced unequal pre-sync face copies");
                require(
                    result.active_face_flux * velocity_sign > 0.0
                        && result.donor_remaining_fraction
                            >= 0.5 - 64.0 * std::numeric_limits<double>::epsilon(),
                    "signed limiter case violated flux direction or donor reserve");
                require(
                    std::abs(result.conserved_number_ratio - 1.0) <= 1.0e-13,
                    "BoxArray layout changed globally conserved carrier number");
                require(
                    result.charge_error <= 4096.0
                        * std::numeric_limits<double>::epsilon() * rrea::qe,
                    "BoxArray layout broke conservative fluid charge");
                double const flux_scale = std::max(
                    std::abs(reference.active_face_flux), 1.0);
                require(
                    std::abs(result.active_face_flux
                             - reference.active_face_flux)
                        <= 64.0 * std::numeric_limits<double>::epsilon()
                            * flux_scale,
                    "BoxArray layout changed the physical limited face flux");
                bool const split_in_direction =
                    layout == BoxLayout::Tiles
                    || layout == BoxLayout::ReversedTiles
                    || (direction == 0 && layout == BoxLayout::RadialStripes)
                    || (direction == 1 && layout == BoxLayout::AxialStripes);
                if (split_in_direction) {
                    require(
                        result.active_face_copy_count >= 2.0,
                        "active limiter face did not lie on the intended box seam");
                }
            }
        }
    }
}

void fill_linear_density(
    amrex::MultiFab& density,
    amrex::Geometry const& geom,
    double base,
    double grad_r,
    double grad_z)
{
    auto const dx = geom.CellSizeArray();
    for (amrex::MFIter mfi(density); mfi.isValid(); ++mfi) {
        auto const bx = mfi.validbox();
        auto const values = density.array(mfi);
        amrex::LoopOnCpu(bx, [&](int i, int j, int k) noexcept {
            double const r = geom.ProbLo(0)
                + (static_cast<double>(i) + 0.5) * dx[0];
#if (AMREX_SPACEDIM >= 2)
            double const z = geom.ProbLo(1)
                + (static_cast<double>(j) + 0.5) * dx[1];
#else
            double const z = 0.0;
#endif
            values(i, j, k) = base + grad_r * r + grad_z * z;
        });
    }
}

void set_tensor(amrex::MultiFab& tensor, double drr, double drz, double dzz)
{
    tensor.setVal(drr, 0, 1, 1);
    tensor.setVal(drz, 1, 1, 1);
    tensor.setVal(dzz, 2, 1, 1);
}

void run_anisotropic_diffusion_test()
{
    auto domain = make_domain(false);
    rrea::LowEnergyFluidState fluid(domain.ba, domain.dm, 1);
    auto efield = make_face_fields(domain.ba, domain.dm, 1);
    auto flux = make_face_fields(domain.ba, domain.dm, 3);
    amrex::MultiFab mobility(domain.ba, domain.dm, 1, 1);
    amrex::MultiFab tensor(domain.ba, domain.dm, 3, 1);
    mobility.setVal(0.2);
    double constexpr drr = 2.0;
    double constexpr drz = 0.5;
    double constexpr dzz = 1.5;
    set_tensor(tensor, drr, drz, dzz);
    double constexpr base = 100.0;
    double constexpr grad_r = 0.25;
    double constexpr grad_z = -0.10;
    fill_linear_density(
        fluid.LowElectronDensity(), domain.geom, base, grad_r, grad_z);
    fluid.ComputeFluidChargeDensity();

    auto const config = electron_config(true);
    double constexpr dt = 0.01;
    double const number_before = total_number(
        fluid.LowElectronDensity(), domain.geom);
    fluid.PrepareExplicitCarrierTransport(domain.geom, config);
    auto const build = fluid.BuildExplicitFaceTransport(
        dt, domain.geom, amrex::GetArrOfConstPtrs(efield), config,
        amrex::GetArrOfPtrs(flux), &mobility, &tensor);

    double const lambda_max = 0.5 * (
        drr + dzz + std::hypot(drr - dzz, 2.0 * drz));
    auto const dx = domain.geom.CellSizeArray();
    double inverse_spacing_squared = 1.0 / (dx[0] * dx[0]);
#if (AMREX_SPACEDIM >= 2)
    inverse_spacing_squared += 1.0 / (dx[1] * dx[1]);
#endif
    double const expected_diffusion_number =
        2.0 * dt * lambda_max * inverse_spacing_squared;
    require(
        std::abs(build.max_explicit_diffusion_number
                 - expected_diffusion_number)
            <= 64.0 * std::numeric_limits<double>::epsilon()
                * expected_diffusion_number,
        "explicit anisotropic diffusion number disagreed with its eigenvalue bound");

    int constexpr interior_i = 6;
    int constexpr interior_j = 10;
    auto const radial = face_copies(*flux[0], interior_i, interior_j, 2);
    auto const axial = face_copies(*flux[1], interior_i, interior_j, 2);
    double const expected_radial = -(drr * grad_r + drz * grad_z);
    double const expected_axial = -(drz * grad_r + dzz * grad_z);
    require(
        std::abs(radial.maximum - expected_radial) <= 1.0e-12
            && std::abs(axial.maximum - expected_axial) <= 1.0e-12,
        "anisotropic face flux did not reproduce the analytic linear-gradient law");
    auto const axis = face_copies(*flux[0], 0, interior_j, 2);
    require(axis.minimum == 0.0 && axis.maximum == 0.0,
        "RZ axis emitted a radial diffusive flux");

    double const boundary_rate = electron_boundary_outflow(flux, domain.geom);
    fluid.ApplyExplicitCarrierTransport(
        dt, domain.geom, amrex::GetArrOfPtrs(flux), config);
    double const number_after = total_number(
        fluid.LowElectronDensity(), domain.geom);
    double const expected_after = number_before - dt * boundary_rate;
    require(
        std::abs(number_after - expected_after)
            <= 2.0e-12 * std::max(number_before, 1.0),
        "absorbing-boundary carrier loss did not equal the integrated face flux");
    require(fluid.LowElectronDensity().min(0, 0, false) >= 0.0,
        "anisotropic diffusion produced a negative carrier density");
    require(
        max_species_charge_error(fluid)
            <= 4096.0 * std::numeric_limits<double>::epsilon()
                * rrea::qe * (base + 16.0 * grad_r),
        "anisotropic electron current and conservative charge diverged");

    fluid.PrepareExplicitCarrierTransport(domain.geom, config);
    set_tensor(tensor, 1.0, 2.0, 1.0);
    require_throws(
        [&]() {
            fluid.BuildExplicitFaceTransport(
                dt, domain.geom, amrex::GetArrOfConstPtrs(efield), config,
                amrex::GetArrOfPtrs(flux), &mobility, &tensor);
        },
        "non-positive-semidefinite diffusion tensor was accepted");

    set_tensor(tensor, 100.0, 0.0, 100.0);
    require_throws(
        [&]() {
            fluid.BuildExplicitFaceTransport(
                1.0, domain.geom, amrex::GetArrOfConstPtrs(efield), config,
                amrex::GetArrOfPtrs(flux), &mobility, &tensor);
        },
        "unstable explicit diffusion number was accepted");
}

} // namespace

int main(int argc, char** argv)
{
    amrex::Initialize(argc, argv);
    int result = 0;
    try {
        run_transport_vacuum_and_seam_test();
        run_layout_and_sign_matrix();
        run_anisotropic_diffusion_test();
        amrex::Print() << "rrea_low_energy_face_transport_smoke passed\n";
    } catch (std::exception const& exception) {
        amrex::Print() << "rrea_low_energy_face_transport_smoke failed: "
                       << exception.what() << "\n";
        result = 1;
    }
    amrex::Finalize();
    return result;
}
