#include "rrea/RreaAmrexAdvance.H"

#include "rrea/RreaConstants.H"
#include "rrea/RreaFieldGatherStencil.H"
#include "rrea/RreaSegmentDepositionRZ.H"
#include "rrea/RreaInteractionTables.H"

#include <AMReX_Array4.H>
#include <AMReX_MFIter.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_Reduce.H>
#include <AMReX_ParallelDescriptor.H>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace rrea {

namespace {

struct MaterialContinuityArrayView {
    MaxwellTMRZLayout layout;
    double dt, allowance;
    double const* jr;
    double const* jz;
    amrex::Array4<amrex::Real const> old_charge, new_charge, particles, fluid;
    amrex::Array4<amrex::Real const> source_e, source_pos, source_neg;

    RREA_HOST_DEVICE MaterialContinuitySample operator()(int i, int j, int k) const noexcept
    {
        return EvaluateMaterialContinuity(layout, i, j, dt, allowance,
            {static_cast<double>(old_charge(i,j,k)),
             static_cast<double>(new_charge(i,j,k)),
             static_cast<double>(particles(i,j,k)),
             static_cast<double>(fluid(i,j,k)),
             static_cast<double>(source_e(i,j,k)),
             static_cast<double>(source_pos(i,j,k)),
             static_cast<double>(source_neg(i,j,k))}, jr, jz);
    }
};

// Public POD call operators, rather than extended lambdas in private
// RreaAmrexAdvance methods (an NVCC language restriction).
struct ContinuityReduction {
    MaterialContinuityArrayView view;
    RREA_HOST_DEVICE amrex::GpuTuple<double, double, double>
    operator()(int i, int j, int k) const noexcept
    {
        auto const value = view(i, j, k);
        return {value.relative, value.absolute_e, value.signed_e};
    }
};

struct ContinuityLocation {
    MaterialContinuityArrayView view;
    double maximum;
    int nr;
    RREA_HOST_DEVICE amrex::GpuTuple<amrex::Long>
    operator()(int i, int j, int k) const noexcept
    {
        return {view(i,j,k).relative == maximum
            ? static_cast<amrex::Long>(i) + static_cast<amrex::Long>(nr) * j
            : std::numeric_limits<amrex::Long>::max()};
    }
};

struct ContinuityWitness {
    MaterialContinuityArrayView view;
    int i, j;
    double* out;
    RREA_HOST_DEVICE void operator()(std::size_t) const noexcept
    {
        auto const value = view(i,j,0);
        out[0] = value.old_charge; out[1] = value.new_charge;
        out[2] = value.divergence; out[3] = value.residual;
        out[4] = value.absolute_e;
    }
};

struct GaussReduction {
    MaxwellTMRZConstView field;
    amrex::Array4<amrex::Real const> rho;
    RREA_HOST_DEVICE amrex::GpuTuple<double, double>
    operator()(int i, int j, int k) const noexcept
    {
        double const density = static_cast<double>(rho(i,j,k));
        double const residual = std::abs(
            eps0 * field.DivergenceRZ(field.er, field.ez, i, j) - density);
        double const invalid = std::numeric_limits<double>::infinity();
        return {std::isfinite(residual) ? residual : invalid,
                std::isfinite(density) ? std::abs(density) : invalid};
    }
};

struct CellFieldFromFaces {
    MaxwellTMRZLayout field;
    double const* er;
    double const* ez;
    amrex::Array4<amrex::Real> ex, ez_cell;
    RREA_HOST_DEVICE void operator()(int i, int j, int k) const noexcept
    {
        ex(i,j,k) = static_cast<amrex::Real>(0.5 * (
            er[field.ErIndex(i,j)] + er[field.ErIndex(i+1,j)]));
        ez_cell(i,j,k) = static_cast<amrex::Real>(0.5 * (
            ez[field.EzIndex(i,j)] + ez[field.EzIndex(i,j+1)]));
    }
};

using FaceFabArray =
    amrex::Array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM>;

FaceFabArray make_face_fields(
    amrex::BoxArray const& cell_ba,
    amrex::DistributionMapping const& dm,
    int ncomp)
{
    FaceFabArray fields;
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        amrex::BoxArray face_ba = cell_ba;
        face_ba.surroundingNodes(dir);
        fields[dir] = std::make_unique<amrex::MultiFab>(face_ba, dm, ncomp, 0);
        fields[dir]->setVal(0.0);
    }
    return fields;
}

// The two en_table selections, gated exactly as the fluid bindings gate
// them: mobility only participates while electron drift is enabled.
bool table_mobility_selected(LowEnergyFluidConfig const& fluid)
{
    return fluid.electron_drift_enable
        && fluid.electron_mobility_model == "en_table_flux_v1";
}

bool table_attachment_selected(LowEnergyFluidConfig const& fluid)
{
    return fluid.attachment_model == "en_table_attachment_v1";
}

// Build the analytic ambient directly as the compatible Yee-face gradient of
// one scalar potential.  This is stronger than averaging cell-centered
// derivatives: the mixed face differences commute exactly, so the interior
// discrete curl seen by MaxwellAdvanceB is round-off zero.  The physical
// boundary ghosts written by InitializePhi live on the boundary face, hence
// the half-cell denominator at non-periodic high/low faces.
void compatible_gradient_to_faces(
    amrex::MultiFab const& phi,
    amrex::Geometry const& geom,
    FaceFabArray& faces)
{
    auto const dx = geom.CellSizeArray();
    auto const domain = geom.Domain();
    bool const periodic_z = geom.isPeriodic(1);
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        auto& face = *faces[dir];
        for (amrex::MFIter mfi(face); mfi.isValid(); ++mfi) {
            auto const box = mfi.validbox();
            auto const p = phi.const_array(mfi);
            auto const e = face.array(mfi);
            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                if (dir == 0) {
                    int const ilo = domain.smallEnd(0);
                    int const ihi = domain.bigEnd(0);
                    if (i <= ilo) {
                        e(i, j, k) = 0.0;  // axis symmetry
                    } else {
                        double const spacing = i > ihi
                            ? 0.5 * dx[0]
                            : dx[0];
                        e(i, j, k) = -(
                            static_cast<double>(p(i, j, k))
                            - static_cast<double>(p(i - 1, j, k)))
                            / spacing;
                    }
                }
#if (AMREX_SPACEDIM >= 2)
                else {
                    int const jlo = domain.smallEnd(1);
                    int const jhi = domain.bigEnd(1);
                    double spacing = dx[1];
                    if (!periodic_z && (j <= jlo || j > jhi)) {
                        spacing = 0.5 * dx[1];
                    }
                    e(i, j, k) = -(
                        static_cast<double>(p(i, j, k))
                        - static_cast<double>(p(i, j - 1, k)))
                        / spacing;
                }
#endif
            });
        }
        face.OverrideSync(geom.periodicity());
    }
}

// Owner multiplicity of every face of the replicated layout: adjacent boxes
// share interface faces with identical values, so a reduced sum is divided
// by this once-computed count.
GpuVector<double> face_multiplicity(
    amrex::MultiFab const& faces, int stride, std::size_t size)
{
    GpuVector<double> count(size);
    double* const out = count.data();
    GpuFill(GpuModule::Fluid, out, size, 0.0);
    for (amrex::MFIter mfi(faces); mfi.isValid(); ++mfi) {
        amrex::ParallelFor(mfi.validbox(),
            [=] AMREX_GPU_DEVICE(int i, int j, int) noexcept {
                amrex::Gpu::Atomic::Add(out + static_cast<std::size_t>(i)
                    + static_cast<std::size_t>(stride) * j, 1.0);
            });
    }
    GpuSynchronize(); // MPI below is a host consumer; all MFIter streams join.
    amrex::ParallelDescriptor::ReduceRealSum(
        count.data(), static_cast<int>(count.size()));
    return count;
}

// Weighted component sum of a face MultiFab into a RANK-LOCAL span, already
// divided by the replicated owner multiplicity so the caller's single
// Allreduce of the concatenated buffer recovers the replicated face value.
void face_to_full_vector(
    amrex::MultiFab const& faces,
    std::initializer_list<double> component_weights,
    int stride,
    GpuVector<double> const& count,
    double* value_out,
    std::size_t value_size)
{
    if (component_weights.size() > 3 || component_weights.size() >
        static_cast<std::size_t>(faces.nComp())) {
        throw std::runtime_error("invalid component count in face assembly");
    }
    amrex::GpuArray<double, 3> weights{};
    int ncomp = 0;
    for (double weight : component_weights) { weights[ncomp++] = weight; }
    double const* const multiplicity = count.data();
    GpuFill(GpuModule::Fluid, value_out, value_size, 0.0);
    for (amrex::MFIter mfi(faces); mfi.isValid(); ++mfi) {
        auto const arr = faces.const_array(mfi);
        amrex::ParallelFor(mfi.validbox(),
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                std::size_t const f = static_cast<std::size_t>(i)
                    + static_cast<std::size_t>(stride) * j;
                double value = 0.0;
                for (int c = 0; c < ncomp; ++c) {
                    value += weights[c] * static_cast<double>(arr(i, j, k, c));
                }
                // Face copies in different boxes can run on different streams.
                amrex::Gpu::Atomic::Add(value_out + f, value);
            });
    }
    GpuSynchronize();
    GpuFor(GpuModule::Fluid, value_size, [=] RREA_HOST_DEVICE(std::size_t f) {
        if (multiplicity[f] > 0.0) { value_out[f] /= multiplicity[f]; }
    });
}

// Write a replicated full-domain face vector back into a face MultiFab.
void full_vector_to_face(
    GpuVector<double> const& values,
    int stride,
    amrex::MultiFab& faces)
{
    double const* const input = values.data();
    for (amrex::MFIter mfi(faces); mfi.isValid(); ++mfi) {
        auto const arr = faces.array(mfi);
        amrex::ParallelFor(mfi.validbox(),
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            arr(i, j, k) = static_cast<amrex::Real>(
                input[static_cast<std::size_t>(i)
                       + static_cast<std::size_t>(stride) * j]);
        });
    }
}

}  // namespace

RreaAmrexAdvance::RreaAmrexAdvance(
    amrex::Vector<RreaAmrexLevelBinding> level_bindings,
    int fluid_n_grow)
    : m_levels(std::move(level_bindings))
{
    RequireHybridManagedMemory();
    ValidateLevelBindings();
    m_fluids.reserve(m_levels.size());
    m_rho_total.reserve(m_levels.size());
    for (auto const& level : m_levels) {
        m_fluids.push_back(std::make_unique<LowEnergyFluidState>(
            level.box_array, level.distribution_map, fluid_n_grow));
        m_rho_total.push_back(std::make_unique<amrex::MultiFab>(
            level.box_array, level.distribution_map, 1, 0));
        m_rho_total.back()->setVal(0.0);
        m_face_efield.push_back(make_face_fields(
            level.box_array, level.distribution_map, 1));
        m_face_number_flux.push_back(make_face_fields(
            level.box_array, level.distribution_map, 3));
    }
    auto const& geom = m_levels[0].geom;
    m_maxwell.nr = geom.Domain().length(0);
    m_maxwell.nz = geom.Domain().length(1);
    m_maxwell.dr = geom.CellSize(0);
    m_maxwell.dz = geom.CellSize(1);
    m_maxwell.z_lo = geom.ProbLo(1);
    m_maxwell.resize();
    m_ambient = m_maxwell;
    m_ambient.btheta = {};
    m_er_face_count = face_multiplicity(
        *m_face_efield[0][0], m_maxwell.nr + 1, m_maxwell.er.size());
    m_ez_face_count = face_multiplicity(
        *m_face_efield[0][1], m_maxwell.nr, m_maxwell.ez.size());
}

int RreaAmrexAdvance::FinestLevel() const noexcept
{
    return static_cast<int>(m_levels.size()) - 1;
}

LowEnergyFluidState& RreaAmrexAdvance::Fluid(int lev)
{
    return *m_fluids.at(static_cast<std::size_t>(lev));
}

LowEnergyFluidState const& RreaAmrexAdvance::Fluid(int lev) const
{
    return *m_fluids.at(static_cast<std::size_t>(lev));
}

// The static ambient is reconstructed from its analytic potential rather than
// checkpointed.  Taking its gradient directly on the Maxwell faces makes the
// discrete curl identically zero; in particular the production radial taper
// retains the Er fringe that cancels d(Ez)/dr.
void RreaAmrexAdvance::InitializeBackgroundState(
    FieldInitializerConfig const& config)
{
    auto const& level = m_levels[0];
    amrex::MultiFab phi(level.box_array, level.distribution_map, 1, 1);
    RreaFieldInitializer::InitializePhi(phi, level.geom, config);
    FaceFabArray faces = make_face_fields(
        level.box_array, level.distribution_map, 1);
    compatible_gradient_to_faces(phi, level.geom, faces);
    face_to_full_vector(
        *faces[0], {1.0}, m_maxwell.nr + 1, m_er_face_count,
        m_ambient.er.data(), m_ambient.er.size());
    face_to_full_vector(
        *faces[1], {1.0}, m_maxwell.nr, m_ez_face_count,
        m_ambient.ez.data(), m_ambient.ez.size());
    amrex::ParallelDescriptor::ReduceRealSum(
        m_ambient.er.data(), static_cast<int>(m_ambient.er.size()));
    amrex::ParallelDescriptor::ReduceRealSum(
        m_ambient.ez.data(), static_cast<int>(m_ambient.ez.size()));

    // The ambient must be curl-free BY CONSTRUCTION (both face families are
    // differences of the one phi, so the discrete (curl E)_theta telescopes
    // to exact cancellation): verify it at every interior Btheta corner
    // rather than document it.  A violation means the construction changed
    // and the retained radial fringe is no longer the legitimate curl-free
    // complement of the tapered column -- fail closed at initialization.
    double curl_max = 0.0;
    double scale_max = 0.0;
    for (int j = 1; j < m_maxwell.nz; ++j) {
        for (int i = 1; i < m_maxwell.nr; ++i) {
            double const er_hi = m_ambient.Er(i, j);
            double const er_lo = m_ambient.Er(i, j - 1);
            double const ez_hi = m_ambient.Ez(i, j);
            double const ez_lo = m_ambient.Ez(i - 1, j);
            curl_max = std::max(curl_max, std::abs(
                (er_hi - er_lo) / m_maxwell.dz
                - (ez_hi - ez_lo) / m_maxwell.dr));
            scale_max = std::max(scale_max,
                (std::abs(er_hi) + std::abs(er_lo)) / m_maxwell.dz
                + (std::abs(ez_hi) + std::abs(ez_lo)) / m_maxwell.dr);
        }
    }
    if (!(curl_max <= 1.0e-12 * std::max(scale_max, 1.0))) {
        throw std::runtime_error(
            "ambient field construction is no longer curl-free: "
            "max |curl_theta| = " + std::to_string(curl_max)
            + " against gradient scale " + std::to_string(scale_max));
    }
    m_background_field_initialized = true;
    CommitMaxwellTotalField();
}

void RreaAmrexAdvance::EvaluateClosureFields(
    RreaAmrexAdvanceConfig const& config,
    bool build_mobility,
    bool build_attachment,
    char const* stage)
{
    if (config.electron_closure == nullptr
        || !config.electron_closure->Loaded()) {
        throw std::runtime_error(
            "en_table fluid models require a loaded electron closure binding");
    }
    auto ensure_fields =
        [this](amrex::Vector<std::unique_ptr<amrex::MultiFab>>& fields,
               int ncomp) {
            if (!fields.empty()) {
                return;
            }
            fields.reserve(m_levels.size());
            for (auto const& level : m_levels) {
                fields.push_back(std::make_unique<amrex::MultiFab>(
                    level.box_array, level.distribution_map, ncomp, 1));
                fields.back()->setVal(0.0);
            }
        };
    if (build_mobility) {
        ensure_fields(m_mu_e_cell, 1);
        ensure_fields(m_diffusion_tensor_e_cell, 3);
    }
    if (build_attachment) {
        ensure_fields(m_nu_att_cell, 1);
    }
    LowEnergyClosureFieldDiagnostics local;
    for (int lev = 0; lev < static_cast<int>(m_levels.size()); ++lev) {
        auto const& level = m_levels[lev];
        auto const& fluid = *m_fluids[lev];
        BuildElectronClosureCellFields(
            level.geom,
            *level.efield_x,
            *level.efield_z,
            level.density_ratio,
            config.fluid.transport_density_ratio,
            *config.electron_closure,
            &fluid.LowElectronDensity(),
            &fluid.LowElectronSourceRate(),
            build_mobility ? m_mu_e_cell[lev].get() : nullptr,
            build_attachment ? m_nu_att_cell[lev].get() : nullptr,
            build_mobility ? m_diffusion_tensor_e_cell[lev].get() : nullptr,
            local);
    }
    // Collective gate: every rank reaches this reduction in lockstep (the
    // evaluation points are structural, not data-dependent), so an
    // above-range or invalid state aborts globally before any commit.
    double extrema[2] = {-local.min_en_td, local.max_en_td};
    amrex::ParallelDescriptor::ReduceRealMax(extrema, 2);
    amrex::Long counts[3] = {
        static_cast<amrex::Long>(local.held_below_cells),
        static_cast<amrex::Long>(local.above_range_cells),
        static_cast<amrex::Long>(local.invalid_cells)};
    amrex::ParallelDescriptor::ReduceLongSum(counts, 3);
    double active_weights[6] = {
        local.electron_count,
        local.held_below_electron_count,
        local.electron_source_rate_per_s,
        local.held_below_electron_source_rate_per_s,
        local.electron_conductivity_proxy_m2_per_v_s,
        local.held_below_electron_conductivity_proxy_m2_per_v_s};
    amrex::ParallelDescriptor::ReduceRealSum(active_weights, 6);
    double active_maxima[6] = {
        local.max_electron_density_m3,
        local.max_held_below_electron_density_m3,
        local.max_electron_source_rate_m3_s,
        local.max_held_below_electron_source_rate_m3_s,
        local.max_electron_conductivity_proxy_per_m_v_s,
        local.max_held_below_electron_conductivity_proxy_per_m_v_s};
    amrex::ParallelDescriptor::ReduceRealMax(active_maxima, 6);
    double const min_en_td = -extrema[0];
    double const max_en_td = extrema[1];
    int above_location[2] = {-1, -1};
    double above_details[4] = {max_en_td, 0.0, 0.0, 0.0};
    if (counts[1] > 0) {
        int owner = local.max_above_range_en_td == max_en_td
            ? amrex::ParallelDescriptor::MyProc()
            : std::numeric_limits<int>::max();
        amrex::ParallelDescriptor::ReduceIntMin(owner);
        if (amrex::ParallelDescriptor::MyProc() == owner) {
            above_location[0] = local.max_above_range_i;
            above_location[1] = local.max_above_range_j;
            above_details[1] = local.max_above_range_electron_density_m3;
            above_details[2] = local.max_above_range_source_rate_m3_s;
            above_details[3] =
                local.max_above_range_conductivity_proxy_per_m_v_s;
        }
        amrex::ParallelDescriptor::Bcast(above_location, 2, owner);
        amrex::ParallelDescriptor::Bcast(above_details, 4, owner);
    }
    if (counts[1] > 0 || counts[2] > 0) {
        std::ostringstream message;
        message << std::scientific << std::setprecision(17)
                << "electron closure range violation at " << stage
                << ": above_range_cells=" << counts[1]
                << ", invalid_cells=" << counts[2]
                << ", realized_min_en_td=" << min_en_td
                << ", realized_max_en_td=" << max_en_td
                << ", valid=[" << config.electron_closure->MinEnTd()
                << ", " << config.electron_closure->MaxEnTd() << "] Td";
        if (counts[1] > 0) {
            message << ", max_above_cell=(" << above_location[0] << ","
                    << above_location[1] << ")"
                    << ", max_above_electron_density_m3=" << above_details[1]
                    << ", max_above_source_rate_m3_s=" << above_details[2]
                    << ", max_above_electron_conductivity_S_per_m="
                    << qe * above_details[3];
        }
        throw std::runtime_error(message.str());
    }
    auto const held_fraction = [](double held, double total) {
        return total > 0.0 ? held / total : 0.0;
    };
    double const electron_integral_fraction =
        held_fraction(active_weights[1], active_weights[0]);
    double const source_integral_fraction =
        held_fraction(active_weights[3], active_weights[2]);
    double const conductivity_integral_fraction =
        held_fraction(active_weights[5], active_weights[4]);
    double const electron_peak_fraction =
        held_fraction(active_maxima[1], active_maxima[0]);
    double const source_peak_fraction =
        held_fraction(active_maxima[3], active_maxima[2]);
    double const conductivity_peak_fraction =
        held_fraction(active_maxima[5], active_maxima[4]);
    double const electron_fraction = std::max(
        electron_integral_fraction, electron_peak_fraction);
    double const source_fraction = std::max(
        source_integral_fraction, source_peak_fraction);
    double const conductivity_fraction = build_mobility
        ? std::max(conductivity_integral_fraction, conductivity_peak_fraction)
        : 0.0;
    if (!std::isfinite(electron_fraction)
        || !std::isfinite(source_fraction)
        || !std::isfinite(conductivity_fraction)) {
        throw std::runtime_error(
            "electron closure active below-range fractions are non-finite");
    }
    // Below the positive proxy node the v3 table holds its physical E/N=0
    // asymptote; retain the fractions as diagnostics, not validity warnings.
    if (m_last_diagnostics.closure_evaluated) {
        m_last_diagnostics.closure_min_en_td =
            std::min(m_last_diagnostics.closure_min_en_td, min_en_td);
        m_last_diagnostics.closure_max_en_td =
            std::max(m_last_diagnostics.closure_max_en_td, max_en_td);
    } else {
        m_last_diagnostics.closure_min_en_td = min_en_td;
        m_last_diagnostics.closure_max_en_td = max_en_td;
        m_last_diagnostics.closure_evaluated = true;
    }
    m_last_diagnostics.closure_held_below_cells =
        static_cast<long>(counts[0]);
    m_last_diagnostics.closure_max_held_below_electron_fraction = std::max(
        m_last_diagnostics.closure_max_held_below_electron_fraction,
        electron_fraction);
    m_last_diagnostics.closure_max_held_below_source_fraction = std::max(
        m_last_diagnostics.closure_max_held_below_source_fraction,
        source_fraction);
    m_last_diagnostics.closure_max_held_below_conductivity_fraction = std::max(
        m_last_diagnostics.closure_max_held_below_conductivity_fraction,
        conductivity_fraction);
}

void RreaAmrexAdvance::BindKineticCurrent(
    std::vector<double> const* jr_index,
    std::vector<double> const* jz_index,
    int pad_cells,
    double charge_per_index_flux_c)
{
    m_kinetic_jr_index = jr_index;
    m_kinetic_jz_index = jz_index;
    m_kinetic_pad_cells = pad_cells;
    m_kinetic_charge_per_index_flux_c = charge_per_index_flux_c;
}

void RreaAmrexAdvance::PrimeMaxwellContinuityReference()
{
    if (!m_maxwell_rho_previous) {
        auto const& level = m_levels[0];
        m_maxwell_rho_previous = std::make_unique<amrex::MultiFab>(
            level.box_array, level.distribution_map, 1, 0);
    }
    AssembleTotalCharge();
    amrex::MultiFab::Copy(
        *m_maxwell_rho_previous, *m_rho_total[0], 0, 0, 1, 0);
    m_maxwell_continuity_primed = true;
}

// Publish the authoritative total Yee faces to every consumer and rebuild the
// cell averages used by closure evaluation and diagnostics.  This is also the
// restart handoff: the scattered checkpoint plus reconstructed ambient, not a
// face field re-interpolated from checkpointed cell averages, owns the state.
void RreaAmrexAdvance::CommitMaxwellTotalField()
{
    auto const& level = m_levels[0];
    auto& total_r = m_total_er;
    auto& total_z = m_total_ez;
    total_r.resize(m_maxwell.er.size());
    total_z.resize(m_maxwell.ez.size());
    auto const field = m_maxwell.view();
    auto const ambient = m_ambient.view();
    auto* const er = total_r.data();
    auto* const ez = total_z.data();
    GpuAdd(GpuModule::Fluid, field.er, ambient.er, er, total_r.size());
    GpuAdd(GpuModule::Fluid, field.ez, ambient.ez, ez, total_z.size());
    full_vector_to_face(total_r, field.nr + 1, *m_face_efield[0][0]);
    full_vector_to_face(total_z, field.nr, *m_face_efield[0][1]);

    for (amrex::MFIter mfi(*level.efield_x); mfi.isValid(); ++mfi) {
        auto const ex = level.efield_x->array(mfi);
        auto const ez_cell = level.efield_z->array(mfi);
        amrex::ParallelFor(mfi.validbox(), CellFieldFromFaces{field, er, ez, ex, ez_cell});
    }
    level.efield_x->FillBoundary(level.geom.periodicity());
    level.efield_z->FillBoundary(level.geom.periodicity());
}

RreaKineticFieldView RreaAmrexAdvance::KineticFieldView() const noexcept
{
    auto const& geom=m_levels[0].geom;
    auto const& domain=geom.Domain();
    return {m_maxwell.view(), m_ambient.view(), geom.ProbLo(0), geom.ProbLo(1),
        {domain.smallEnd(0),domain.smallEnd(1)}, {domain.bigEnd(0),domain.bigEnd(1)},
        geom.isPeriodic(1)};
}

double RreaAmrexAdvance::MaxwellBthetaAt(double r_m,double z_m) const noexcept
{
    return KineticFieldView().MaxwellBthetaAt(r_m,z_m);
}
void RreaAmrexAdvance::GatherTotalFieldRZ(double r_m,double z_m,
    double& er,double& ez,double& btheta) const noexcept
{
    KineticFieldView().Gather(r_m,z_m,er,ez,btheta);
}


// ---------------------------------------------------------------------------
// Governing update per step (SI, axisymmetric m=0, TM triplet Er/Ez/Btheta):
//     d(E_scattered)/dt = c^2 curl B - J_total / eps0
//     d(Btheta)/dt      = -(curl E)_theta
// with the static ambient profile E_amb carried separately (its sources are
// the thundercloud, quasi-static on our record).  Conduction enters through
// J_total = J_fluid + J_kinetic where J_fluid is the fluid's OWN committed
// donor-upwind face current (which responded to E_total), so there is no
// sigma term to double count, and fluid continuity and the field equation
// share one identical current. Each adaptive material substep is certified
// against the configured conductive, diffusive and reaction guards below.
// ---------------------------------------------------------------------------
void RreaAmrexAdvance::AdvanceOneStep(
    double dt_s,
    RreaAmrexAdvanceConfig const& config)
{
    if (!std::isfinite(dt_s) || dt_s <= 0.0) {
        throw std::runtime_error(
            "RreaAmrexAdvance requires finite, positive dt");
    }
    ValidateEvolutionConfig(config);
    double const maxwell_cfl = m_maxwell.CflNumber(dt_s);
    if (!std::isfinite(maxwell_cfl) || !(maxwell_cfl < 1.0)) {
        throw std::runtime_error(
            "Maxwell TM violates the two-dimensional Yee CFL condition: "
            "c*dt*sqrt(dr^-2+dz^-2)=" + std::to_string(maxwell_cfl)
            + " must be finite and < 1");
    }
    if (m_kinetic_jr_index == nullptr || m_kinetic_jz_index == nullptr) {
        throw std::runtime_error(
            "the Maxwell TM step requires the kinetic net-chord current to "
            "be bound before AdvanceOneStep");
    }
    // Kinetic transport is a retained host stage. Its outer-step current is
    // constant during all adaptive fluid/Maxwell substeps: stage it once.
    GpuSynchronize();
    ConvertPaddedIndexCurrentToPhysicalRZ(
        m_maxwell.nr, m_maxwell.nz, m_maxwell.dr, m_maxwell.dz,
        m_kinetic_pad_cells, m_kinetic_charge_per_index_flux_c,
        *m_kinetic_jr_index, *m_kinetic_jz_index, m_jr_kinetic, m_jz_kinetic);

    m_last_diagnostics.max_normalized_reaction_charge_residual = 0.0;
    m_last_diagnostics.closure_evaluated = false;
    m_last_diagnostics.closure_min_en_td = 0.0;
    m_last_diagnostics.closure_max_en_td = 0.0;
    m_last_diagnostics.closure_held_below_cells = 0;
    m_last_diagnostics.closure_max_held_below_electron_fraction = 0.0;
    m_last_diagnostics.closure_max_held_below_source_fraction = 0.0;
    m_last_diagnostics.closure_max_held_below_conductivity_fraction = 0.0;
    m_last_diagnostics.maxwell_charge_continuity_max_relative = 0.0;
    m_last_diagnostics.maxwell_charge_continuity_max_abs_e_per_cell = 0.0;
    m_last_diagnostics.maxwell_max_dt_sigma_face_over_eps0 = 0.0;
    m_last_diagnostics.fluid_max_explicit_diffusion_number = 0.0;
    m_last_diagnostics.material_substeps = 0;
    m_last_diagnostics.fluid_max_reaction_number = 0.0;
    m_last_diagnostics.maxwell_joule_work_j = 0.0;
    m_last_diagnostics.maxwell_poynting_out_j = 0.0;
    m_last_diagnostics.maxwell_energy_balance_residual_j = 0.0;
    m_last_diagnostics.maxwell_jz_moment_a_m = 0.0;

    bool const mobility_table = table_mobility_selected(config.fluid);
    bool const attachment_table = table_attachment_selected(config.fluid);
    if (mobility_table || attachment_table) {
        EvaluateClosureFields(
            config, mobility_table, attachment_table,
            "beginning of step (maxwell)");
    }

    // Reactions, carrier transport and Maxwell share equal adaptive material
    // substeps. The outer-step source rates remain intact through the
    // continuity gate, so their known time integral is distributed uniformly
    // and their gross accumulation scale remains available to that gate.
    auto& fluid = *m_fluids[0];
    auto const& level = m_levels[0];
    // A tiny non-mutating face build measures the current stiffness.  Choose
    // enough shared fluid/Maxwell steps to meet all configured accuracy
    // targets; the realized transport and reaction builds enforce guards.
    fluid.ComputeConductivity(
        config.fluid, level.density_ratio,
        mobility_table ? m_mu_e_cell[0].get() : nullptr);
    fluid.PrepareExplicitCarrierTransport(
        level.geom, config.fluid, level.density_ratio);
    double constexpr probe_fraction = 1.0e-6;
    auto const probe = fluid.BuildExplicitFaceTransport(
        dt_s * probe_fraction, level.geom,
        amrex::GetArrOfConstPtrs(m_face_efield[0]), config.fluid,
        amrex::GetArrOfPtrs(m_face_number_flux[0]),
        mobility_table ? m_mu_e_cell[0].get() : nullptr,
        mobility_table ? m_diffusion_tensor_e_cell[0].get() : nullptr);
    double const full_conduction = dt_s
        * probe.max_face_conductivity_siemens_per_m / eps0;
    double const full_diffusion =
        probe.max_explicit_diffusion_number / probe_fraction;
    double const full_reaction = dt_s * fluid.MaxReactionFrequency(
        config.fluid, attachment_table ? m_nu_att_cell[0].get() : nullptr);
    if (!std::isfinite(full_conduction) || !std::isfinite(full_diffusion)
        || !std::isfinite(full_reaction)) {
        throw std::runtime_error("fluid/Maxwell coupling stiffness is non-finite");
    }
    double const required_substeps = std::max({
        full_conduction / config.material_conduction_target,
        full_diffusion / config.material_diffusion_target,
        full_reaction / config.material_reaction_target});
    if (required_substeps > config.material_max_substeps) {
        throw std::runtime_error(
            "fluid/Maxwell coupling exceeds configured maximum substeps");
    }
    int const coupling_substeps = std::max(
        config.material_min_substeps,
        static_cast<int>(std::ceil(required_substeps)));
    m_last_diagnostics.material_substeps = coupling_substeps;
    if (coupling_substeps != m_maxwell_coupling_substeps) {
        amrex::Print() << "RREA fluid/Maxwell coupling substeps="
            << coupling_substeps << " (unsplit conduction="
            << full_conduction << ", diffusion=" << full_diffusion
            << ", reaction=" << full_reaction << ")\n";
        m_maxwell_coupling_substeps = coupling_substeps;
    }
    double const coupling_dt_s = dt_s / static_cast<double>(coupling_substeps);
    m_jr_average.resize(m_maxwell.er.size());
    m_jz_average.resize(m_maxwell.ez.size());
    GpuFill(GpuModule::Fluid, m_jr_average.data(), m_jr_average.size(), 0.0);
    GpuFill(GpuModule::Fluid, m_jz_average.data(), m_jz_average.size(), 0.0);
    double const energy_before = MaxwellFieldEnergy(m_maxwell, &m_ambient);
    double const b_restagger_dt = 0.5 * (dt_s - coupling_dt_s);
    MaxwellAdvanceB(m_maxwell, b_restagger_dt);
    auto const advance_half_reaction = [&](bool clear_sources) {
        double const reaction_number = coupling_dt_s
            * fluid.MaxReactionFrequency(
                config.fluid,
                attachment_table ? m_nu_att_cell[0].get() : nullptr);
        m_last_diagnostics.fluid_max_reaction_number = std::max(
            m_last_diagnostics.fluid_max_reaction_number, reaction_number);
        if (reaction_number > config.material_reaction_guard) {
            throw std::runtime_error(
                "reaction substep out of configured range: dt_sub*rate = "
                + std::to_string(reaction_number) + " > "
                + std::to_string(config.material_reaction_guard));
        }
        fluid.UpdateFromIonizationSource(
            0.5 * coupling_dt_s, config.fluid, level.density_ratio,
            attachment_table ? m_nu_att_cell[0].get() : nullptr,
            clear_sources);
        m_last_diagnostics.max_normalized_reaction_charge_residual = std::max(
            m_last_diagnostics.max_normalized_reaction_charge_residual,
            fluid.LastReactionChargeConservationResidual());
    };
    for (int substep = 0; substep < coupling_substeps; ++substep) {
        advance_half_reaction(false);
        fluid.ComputeConductivity(
            config.fluid, level.density_ratio,
            mobility_table ? m_mu_e_cell[0].get() : nullptr);
        fluid.PrepareExplicitCarrierTransport(
            level.geom, config.fluid, level.density_ratio);
        auto const face_transport = fluid.BuildExplicitFaceTransport(
            coupling_dt_s, level.geom,
            amrex::GetArrOfConstPtrs(m_face_efield[0]), config.fluid,
            amrex::GetArrOfPtrs(m_face_number_flux[0]),
            mobility_table ? m_mu_e_cell[0].get() : nullptr,
            mobility_table ? m_diffusion_tensor_e_cell[0].get() : nullptr);
        amrex::Real max_dt_sigma_over_eps0 = static_cast<amrex::Real>(
            coupling_dt_s
            * face_transport.max_face_conductivity_siemens_per_m / eps0);
        amrex::ParallelDescriptor::ReduceRealMax(max_dt_sigma_over_eps0);
        m_last_diagnostics.maxwell_max_dt_sigma_face_over_eps0 = std::max(
            m_last_diagnostics.maxwell_max_dt_sigma_face_over_eps0,
            static_cast<double>(max_dt_sigma_over_eps0));
        if (max_dt_sigma_over_eps0
            > amrex::Real(config.material_conduction_guard)) {
            throw std::runtime_error(
                "explicit conduction substep out of configured range: "
                "max(dt_sub*sigma_face/eps0) = "
                + std::to_string(static_cast<double>(max_dt_sigma_over_eps0))
                + " > " + std::to_string(config.material_conduction_guard));
        }
        m_last_diagnostics.fluid_max_explicit_diffusion_number = std::max(
            m_last_diagnostics.fluid_max_explicit_diffusion_number,
            face_transport.max_explicit_diffusion_number);
        if (face_transport.max_explicit_diffusion_number
            > config.material_diffusion_guard) {
            throw std::runtime_error(
                "explicit diffusion substep out of configured range: number = "
                + std::to_string(face_transport.max_explicit_diffusion_number)
                + " > " + std::to_string(config.material_diffusion_guard));
        }
        fluid.ApplyExplicitCarrierTransport(
            coupling_dt_s, level.geom,
            amrex::GetArrOfPtrs(m_face_number_flux[0]), config.fluid,
            level.density_ratio, config.collect_ion_drift_center_diagnostics);

        AssembleTotalFaceCurrent();
        auto* const jr_average = m_jr_average.data();
        auto* const jz_average = m_jz_average.data();
        auto const* const jr_total = m_jr_total.data();
        auto const* const jz_total = m_jz_total.data();
        GpuFor(GpuModule::Fluid, m_jr_total.size(), [=] RREA_HOST_DEVICE(std::size_t f) {
            jr_average[f] += jr_total[f] / coupling_substeps;
        });
        GpuFor(GpuModule::Fluid, m_jz_total.size(), [=] RREA_HOST_DEVICE(std::size_t f) {
            jz_average[f] += jz_total[f] / coupling_substeps;
        });
        // Leapfrog B -> absorber -> E.  Joule work and the discrete boundary
        // term are bilinear in E, so their mean over E^m and E^{m+1} at the
        // fixed half-step B is the exact work of this Ampere update.
        MaxwellAdvanceB(m_maxwell, coupling_dt_s);
        MaxwellSilverMueller(m_maxwell);
        double const poynting_before =
            MaxwellBoundaryPoyntingPower(m_maxwell, &m_ambient);
        double const joule_before = MaxwellJoulePower(
            m_maxwell, &m_ambient, m_jr_total, m_jz_total,
            &m_last_diagnostics.maxwell_jz_moment_a_m);
        MaxwellAdvanceE(
            m_maxwell, coupling_dt_s, nullptr, nullptr,
            &m_jr_total, &m_jz_total, nullptr, nullptr);
        m_last_diagnostics.maxwell_joule_work_j += 0.5 * coupling_dt_s
            * (joule_before
               + MaxwellJoulePower(
                   m_maxwell, &m_ambient, m_jr_total, m_jz_total));
        m_last_diagnostics.maxwell_poynting_out_j += 0.5 * coupling_dt_s
            * (poynting_before
               + MaxwellBoundaryPoyntingPower(m_maxwell, &m_ambient));
        CommitMaxwellTotalField();
        // ONE closure evaluation per substep, on the field just committed:
        // the mobility it builds is exactly what the next substep's transport
        // would have re-derived from that same field, and the attachment is
        // what this substep's second half reaction needs.
        if (mobility_table || attachment_table) {
            EvaluateClosureFields(
                config, mobility_table, attachment_table,
                "end of fluid/Maxwell substep");
        }
        advance_half_reaction(false);
    }
    // Checkpoints and Huygens frames retain the outer leapfrog convention:
    // E(t), B(t-dt/2).  This also makes changing the next step's adaptive
    // count independent of the previous count.
    MaxwellAdvanceB(m_maxwell, -b_restagger_dt);
    MaxwellSilverMueller(m_maxwell);
    m_last_diagnostics.maxwell_jz_moment_a_m /= coupling_substeps;
    fluid.ComputeConductivity(
        config.fluid, level.density_ratio,
        mobility_table ? m_mu_e_cell[0].get() : nullptr);

    if (!m_maxwell_continuity_primed || !m_maxwell_rho_previous) {
        throw std::runtime_error(
            "Maxwell material continuity reference was not primed before the step");
    }
    AssembleTotalCharge();
    VerifyMaterialContinuity(config, dt_s, m_jr_average, m_jz_average);
    fluid.LowElectronSourceRate().setVal(0.0);
    fluid.PositiveIonSourceRate().setVal(0.0);
    fluid.DirectNegativeIonSourceRate().setVal(0.0);
    auto const* const btheta = m_maxwell.btheta.data();
    m_last_diagnostics.maxwell_btheta_max_t = GpuMax(GpuModule::Fluid, m_maxwell.btheta.size(),
        [=] RREA_HOST_DEVICE(std::size_t f) { return std::abs(btheta[f]); });
    m_last_diagnostics.maxwell_scattered_energy_j =
        MaxwellFieldEnergy(m_maxwell);
    m_last_diagnostics.maxwell_energy_change_from_ambient_j =
        MaxwellFieldEnergy(m_maxwell, &m_ambient);
    m_last_diagnostics.maxwell_energy_balance_residual_j =
        m_last_diagnostics.maxwell_energy_change_from_ambient_j - energy_before
        + m_last_diagnostics.maxwell_joule_work_j
        + m_last_diagnostics.maxwell_poynting_out_j;

    RecordGaussMonitor();
    amrex::MultiFab::Copy(
        *m_maxwell_rho_previous, *m_rho_total[0], 0, 0, 1, 0);
    m_kinetic_jr_index = nullptr;
    m_kinetic_jz_index = nullptr;
}

void RreaAmrexAdvance::AssembleTotalFaceCurrent()
{
    // Fluid: qe * [Gamma+ - Gamma- - Gamma_e] from the committed species
    // number fluxes (components 0/1/2).  Kinetic: rank-local padded
    // index-space sums -> physical faces; that conversion is shared with the
    // standalone continuity smoke so the RZ areas and the absorbing-boundary
    // loss of transverse pad support are tested, not only documented.  Normal
    // current on the physical boundary face is retained; no guard current is
    // clamped inward.  Both arms are rank-local, so both directions of both
    // arms concatenate into ONE Allreduce per substep.
    std::size_t const er_faces = m_maxwell.er.size();
    std::size_t const ez_faces = m_maxwell.ez.size();
    m_current_buffer.resize(er_faces + ez_faces);
    face_to_full_vector(
        *m_face_number_flux[0][0], {qe, -qe, -qe}, m_maxwell.nr + 1,
        m_er_face_count, m_current_buffer.data(), er_faces);
    face_to_full_vector(
        *m_face_number_flux[0][1], {qe, -qe, -qe}, m_maxwell.nr,
        m_ez_face_count, m_current_buffer.data() + er_faces, ez_faces);
    auto* const buffer = m_current_buffer.data();
    auto const* const kinetic_r = m_jr_kinetic.data();
    auto const* const kinetic_z = m_jz_kinetic.data();
    GpuAdd(GpuModule::Fluid, buffer, kinetic_r, buffer, er_faces);
    GpuAdd(GpuModule::Fluid, buffer + er_faces, kinetic_z, buffer + er_faces, ez_faces);
    // Retains the replicated MPI layout. On one rank this is a no-op; a
    // distributed Maxwell layout is a separate multi-GPU scalability task.
    amrex::ParallelDescriptor::ReduceRealSum(
        buffer, static_cast<int>(m_current_buffer.size()));
    m_jr_total.resize(er_faces);
    m_jz_total.resize(ez_faces);
    auto* const jr = m_jr_total.data();
    auto* const jz = m_jz_total.data();
    GpuCopy(GpuModule::Fluid, buffer, jr, er_faces);
    GpuCopy(GpuModule::Fluid, buffer + er_faces, jz, ez_faces);
}

void RreaAmrexAdvance::VerifyMaterialContinuity(
    RreaAmrexAdvanceConfig const& config,
    double dt_s,
    GpuVector<double> const& jr_total,
    GpuVector<double> const& jz_total)
{
    // End-to-end MATERIAL continuity before publishing the outer step. rho^n was
    // captured from the actual bound WarpX particle deposit plus persisted
    // fluid charge.  rho^{n+1} is assembled from those same objects by the
    // caller (AssembleTotalCharge). jr_total/jz_total are the time averages
    // of the physical face vectors already used by Ampere's law. This therefore
    // covers survival, birth, kinetic/fluid transfer, annihilation,
    // resampling, and absorbing escape without using the segment
    // depositor's synthetic endpoint-rho scratch.
    auto const& level = m_levels[0];
    auto& fluid = *m_fluids[0];
    double const absolute_allowance_e = kMaterialContinuityAbsoluteMacroFraction
        * config.maximum_represented_charge_e;
    auto const& rho_new = *m_rho_total[0];
    auto const& rho_old = *m_maxwell_rho_previous;
    auto const view_for = [&](amrex::MFIter const& mfi) {
        return MaterialContinuityArrayView{
            static_cast<MaxwellTMRZLayout const&>(m_maxwell), dt_s,
            absolute_allowance_e, jr_total.data(), jz_total.data(),
            rho_old.const_array(mfi), rho_new.const_array(mfi),
            level.rho_c_per_m3->const_array(mfi),
            fluid.FluidChargeDensity().const_array(mfi),
            fluid.LowElectronSourceRate().const_array(mfi),
            fluid.PositiveIonSourceRate().const_array(mfi),
            fluid.DirectNegativeIonSourceRate().const_array(mfi)};
    };
    amrex::ReduceOps<amrex::ReduceOpMax, amrex::ReduceOpMax, amrex::ReduceOpSum> op;
    amrex::ReduceData<double, double, double> data(op);
    for (amrex::MFIter mfi(rho_new); mfi.isValid(); ++mfi) {
        auto const view = view_for(mfi);
        op.eval(mfi.validbox(), data, ContinuityReduction{view});
    }
    auto const result = data.value();
    double local_max_relative = amrex::get<0>(result);
    double local_max_abs_e = amrex::get<1>(result);
    double local_signed_e = amrex::get<2>(result);
    double const rank_local_max_relative = local_max_relative;
    double maxima[2] = {local_max_relative, local_max_abs_e};
    amrex::ParallelDescriptor::ReduceRealMax(maxima, 2);
    local_max_relative = maxima[0];
    local_max_abs_e = maxima[1];
    m_last_diagnostics.maxwell_charge_continuity_max_relative =
        local_max_relative;
    m_last_diagnostics.maxwell_charge_continuity_max_abs_e_per_cell =
        local_max_abs_e;
    amrex::ParallelDescriptor::ReduceRealSum(local_signed_e);
    m_last_diagnostics.maxwell_charge_continuity_signed_accum_e += local_signed_e;
    if (!std::isfinite(local_max_relative)
        || local_max_relative > kMaterialContinuityRelativeTolerance) {
        int owner_rank = rank_local_max_relative == local_max_relative
            ? amrex::ParallelDescriptor::MyProc()
            : std::numeric_limits<int>::max();
        amrex::ParallelDescriptor::ReduceIntMin(owner_rank);
        int location[2] = {-1, -1};
        double details[5] = {};
        if (amrex::ParallelDescriptor::MyProc() == owner_rank) {
            // Only a failing step pays for locating a witness. No full-grid
            // copy or host scan is needed on the normal path.
            amrex::ReduceOps<amrex::ReduceOpMin> where_op;
            amrex::ReduceData<amrex::Long> where_data(where_op);
            int const nr = m_maxwell.nr;
            for (amrex::MFIter mfi(rho_new); mfi.isValid(); ++mfi) {
                auto const view = view_for(mfi);
                where_op.eval(mfi.validbox(), where_data,
                    ContinuityLocation{view, local_max_relative, nr});
            }
            auto const q = amrex::get<0>(where_data.value());
            if (q != std::numeric_limits<amrex::Long>::max()) {
                int const i = static_cast<int>(q % nr);
                int const j = static_cast<int>(q / nr);
                location[0] = i; location[1] = j;
                GpuVector<double> witness(5);
                auto* const out = witness.data();
                for (amrex::MFIter mfi(rho_new); mfi.isValid(); ++mfi) {
                    if (!mfi.validbox().contains(amrex::IntVect(AMREX_D_DECL(i,j,0)))) {
                        continue;
                    }
                    auto const view = view_for(mfi);
                    GpuFor(GpuModule::Fluid, 1, ContinuityWitness{view, i, j, out});
                }
                std::copy(witness.begin(), witness.end(), details);
            }
        }
        amrex::ParallelDescriptor::Bcast(location, 2, owner_rank);
        amrex::ParallelDescriptor::Bcast(details, 5, owner_rank);
        std::ostringstream message;
        message << std::scientific << std::setprecision(17)
                << "actual particle+fluid Maxwell continuity failed before "
                   "outer-step commit: max relative residual="
                << local_max_relative
                << ", tolerance=" << kMaterialContinuityRelativeTolerance
                << ", local_max_cell=(" << location[0] << ","
                << location[1] << ")"
                << ", rho_old=" << details[0]
                << ", rho_new=" << details[1]
                << ", div_j=" << details[2]
                << ", residual=" << details[3]
                << ", residual_e_per_cell=" << details[4]
                << ", absolute_allowance_e=" << absolute_allowance_e
                << ", maximum_represented_charge_e="
                << config.maximum_represented_charge_e;
        throw std::runtime_error(message.str());
    }
}

void RreaAmrexAdvance::RecordGaussMonitor()
{
    // Scattered Gauss law: E_s starts at zero and evolves by exactly
    // -div(J)/eps0 (discrete div curl = 0), so eps0 div(E_s) tracks the FREE
    // charge; the ambient never enters the scattered divergence.  Every
    // material channel is fail-closed by the continuity gate, so the
    // residual retains only charge injected WITHOUT a current -- the seed
    // stream, teleported in by design and growing under a continuous source --
    // plus round-off drift.  Normalization by max |rho| is diagnostic, not
    // a direct bound on longitudinal E error.  m_rho_total already holds
    // rho^{n+1} from the continuity gate.
    double residual_max = 0.0;
    double scale_max = 0.0;
    auto const& rho = *m_rho_total[0];
    auto const field = std::as_const(m_maxwell).view();
    amrex::ReduceOps<amrex::ReduceOpMax, amrex::ReduceOpMax> op;
    amrex::ReduceData<double, double> data(op);
    for (amrex::MFIter mfi(rho); mfi.isValid(); ++mfi) {
        auto const rho_arr = rho.const_array(mfi);
        op.eval(mfi.validbox(), data, GaussReduction{field, rho_arr});
    }
    auto const result = data.value();
    residual_max = amrex::get<0>(result);
    scale_max = amrex::get<1>(result);
    amrex::ParallelDescriptor::ReduceRealMax(residual_max);
    amrex::ParallelDescriptor::ReduceRealMax(scale_max);
    m_last_diagnostics.maxwell_gauss_residual_normalized =
        !std::isfinite(residual_max) || !std::isfinite(scale_max)
            ? std::numeric_limits<double>::infinity()
            : (scale_max > 0.0 ? residual_max / scale_max : 0.0);
}

void RreaAmrexAdvance::ValidateEvolutionConfig(
    RreaAmrexAdvanceConfig const& config) const
{
    if (!m_background_field_initialized) {
        throw std::runtime_error(
            "RreaAmrexAdvance background state is not initialized; call InitializeBackgroundState explicitly");
    }
    if (!std::isfinite(config.maximum_represented_charge_e)
        || !(config.maximum_represented_charge_e > 0.0)) {
        throw std::runtime_error(
            "RreaAmrexAdvance requires a positive finite maximum represented charge");
    }
    double const controls[] = {
        config.material_conduction_target,
        config.material_conduction_guard,
        config.material_diffusion_target,
        config.material_diffusion_guard,
        config.material_reaction_target,
        config.material_reaction_guard};
    for (double const value : controls) {
        if (!std::isfinite(value) || !(value > 0.0)) {
            throw std::runtime_error(
                "material substep targets and guards must be positive and finite");
        }
    }
    if (config.material_conduction_target > config.material_conduction_guard
        || config.material_diffusion_target > config.material_diffusion_guard
        || config.material_reaction_target > config.material_reaction_guard
        || config.material_conduction_guard > 1.0
        || config.material_diffusion_guard > 1.0
        || config.material_reaction_guard > 1.0
        || config.material_min_substeps < 1
        || config.material_max_substeps < config.material_min_substeps) {
        throw std::runtime_error(
            "material targets must not exceed guards, guards must be <= 1, "
            "and substep bounds must satisfy 1 <= min <= max");
    }
}

void RreaAmrexAdvance::AssembleTotalCharge()
{
    for (int lev = 0; lev < static_cast<int>(m_levels.size()); ++lev) {
        auto const& level = m_levels[lev];
        auto const& fluid_rho = m_fluids[lev]->FluidChargeDensity();
        amrex::MultiFab::Copy(
            *m_rho_total[lev], *level.rho_c_per_m3, 0, 0, 1, 0);
        amrex::MultiFab::Add(
            *m_rho_total[lev], fluid_rho, 0, 0, 1, 0);
    }
}

void RreaAmrexAdvance::ValidateLevelBindings() const
{
    if (m_levels.size() != 1) {
        throw std::runtime_error(
            "RreaAmrexAdvance is single-level: the Maxwell TM state is one "
            "replicated mesh");
    }
    auto const& level = m_levels[0];
    if (level.rho_c_per_m3 == nullptr
        || level.efield_x == nullptr || level.efield_z == nullptr) {
        throw std::runtime_error("RreaAmrexAdvance level has null field binding");
    }
    if (level.box_array.empty()) {
        throw std::runtime_error("RreaAmrexAdvance level has empty BoxArray");
    }
    double const axis_tolerance = 64.0
        * static_cast<double>(std::numeric_limits<amrex::Real>::epsilon())
        * std::max(
            1.0,
            std::abs(static_cast<double>(level.geom.ProbHi(0))));
    if (std::abs(static_cast<double>(level.geom.ProbLo(0)))
        > axis_tolerance) {
        throw std::runtime_error(
            "RreaAmrexAdvance requires the physical r=0 axis; "
            "annular/nonzero ProbLo(r) RZ domains are uncertified");
    }
}

}  // namespace rrea
