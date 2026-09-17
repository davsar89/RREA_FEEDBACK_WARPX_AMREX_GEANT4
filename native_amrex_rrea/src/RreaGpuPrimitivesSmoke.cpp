// Shared CPU/CUDA numerical kernels: indexing, closure interpolation and the
// finite-volume material identity. CUDA CMake builds launch the same tests
// on the GPU; direct g++ builds exercise the CPU path without AMReX.
#include "rrea/RreaElectronClosure.H"
#include "rrea/RreaGpuSmoke.H"
#include "rrea/RreaMaterialContinuity.H"
#include "rrea/RreaSmokeRequire.H"

#include <cmath>
#include <iostream>
#include <limits>

namespace {
using rrea::smoke::require;
using rrea::smoke::require_close;

void execution_and_reductions()
{
    constexpr int ni = 17, nj = 13;
    rrea::GpuVector<double> values(ni * nj);
    auto* const data = values.data();
    rrea::GpuFill(rrea::GpuModule::Always, data, values.size(), -1.0);
    rrea::GpuFor2D(rrea::GpuModule::Always, -3, ni - 3, 7, nj + 7,
        [=] RREA_DEVICE(int i, int j) noexcept {
            auto const q = (j - 7) * ni + i + 3;
            data[q] = static_cast<double>(q);
        });
    for (std::size_t q = 0; q < values.size(); ++q) {
        require(values[q] == static_cast<double>(q), "2D kernel indexing");
    }
    double const total = rrea::GpuSum2D(rrea::GpuModule::Always, 0, ni, 0, nj,
        [=] RREA_DEVICE(int i, int j) noexcept { return data[j * ni + i]; });
    require_close(total, values.size() * (values.size() - 1.0) / 2.0,
                  "2D reduction sum");
    require_close(rrea::GpuMax(rrea::GpuModule::Always, values.size(),
        [=] RREA_DEVICE(std::size_t q) noexcept { return -1.0 - data[q]; }),
        -1.0, "maximum of negative values");
    rrea::GpuFor(rrea::GpuModule::Always, 0, [=] RREA_DEVICE(std::size_t) noexcept { data[0] = -1.0; });
    rrea::GpuFor2D(rrea::GpuModule::Always, 1, 1, 0, 1,
        [=] RREA_DEVICE(int, int) noexcept { data[0] = -1.0; });
    require(data[0] == 0.0, "empty launches must not write");
    require(rrea::GpuMax(rrea::GpuModule::Always, 0,
        [=] RREA_DEVICE(std::size_t) noexcept { return data[0]; }) == 0.0,
        "empty maximum");
    values[5] = std::numeric_limits<double>::quiet_NaN();
    require(std::isinf(rrea::GpuMax(rrea::GpuModule::Always, values.size(),
        [=] RREA_DEVICE(std::size_t q) noexcept { return data[q]; })),
        "maximum must not silently discard a NaN");
}

void closure_interpolation()
{
    // Analytic fixtures, not production-table byte comparisons: K=2 EN,
    // nu3=3 EN, DL=5 EN, DT=7 EN; nu2 has a zero-onset bracket.
    rrea::GpuVector<double> table(33);
    double const en[] = {1.0, 10.0, 100.0};
    double const nu2[] = {0.0, 4.0, 16.0};
    for (int j = 0; j < 3; ++j) {
        table[j] = en[j];
        table[3+j] = std::log(en[j]);
        table[6+j] = 2.0 * en[j];
        table[9+j] = std::log(table[6+j]);
        table[12+j] = nu2[j];
        table[15+j] = 3.0 * en[j];
        table[18+j] = std::log(table[15+j]);
        table[21+j] = 5.0 * en[j];
        table[24+j] = std::log(table[21+j]);
        table[27+j] = 7.0 * en[j];
        table[30+j] = std::log(table[27+j]);
    }
    auto const* p = table.data();
    rrea::ElectronClosureViewT<double> const view{
        3, p, p+3, p+6, p+9, p+12, p+15, p+18, p+21, p+24, p+27, p+30};
    rrea::GpuVector<double> queries{
        0.0, 0.1, 1.0, std::sqrt(10.0), 10.0, std::sqrt(1000.0), 100.0,
        101.0, -1.0, std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::infinity()};
    using Coeff = rrea::ElectronClosureCoefficientsT<double>;
    rrea::GpuVector<Coeff> answers(queries.size());
    rrea::GpuVector<int> statuses(queries.size());
    auto const* q = queries.data();
    auto* result = answers.data();
    auto* range = statuses.data();
    rrea::GpuFor(rrea::GpuModule::Always, queries.size(), [=] RREA_DEVICE(std::size_t i) noexcept {
        rrea::RreaClosureRangeStatus status;
        result[i] = view.Evaluate(q[i], status);
        range[i] = static_cast<int>(status);
    });
    using Status = rrea::RreaClosureRangeStatus;
    for (std::size_t i = 0; i < queries.size(); ++i) {
        double const x = queries[i];
        Status const expected = !std::isfinite(x) || x < 0.0 ? Status::Invalid
            : x < 1.0 ? Status::HeldBelow
            : x > 100.0 ? Status::AboveRange : Status::InRange;
        require(statuses[i] == static_cast<int>(expected), "closure range policy");
        if (expected == Status::Invalid) { continue; }
        double const clipped = std::max(1.0, std::min(100.0, x));
        require_close(answers[i].k0_flux_ref_m2_per_vs, 2 * clipped, "closure K");
        require_close(answers[i].nu3_ref_per_s, 3 * clipped, "closure nu3");
        require_close(answers[i].longitudinal_diffusion_ref_m2_per_s,
                      5 * clipped, "closure DL");
        require_close(answers[i].transverse_diffusion_ref_m2_per_s,
                      7 * clipped, "closure DT");
    }
    require_close(answers[3].nu2_ref_per_s, 2.0, "zero-onset interpolation");
    require_close(answers[5].nu2_ref_per_s, 8.0, "positive log interpolation");
    require(answers[4].nu2_ref_per_s == 4.0, "exact interior node");
    rrea::GpuFor(rrea::GpuModule::Always, 1, [=] RREA_DEVICE(std::size_t) noexcept {
        rrea::ElectronClosureViewT<double> const empty{};
        Status status;
        result[0] = empty.Evaluate(1.0, status);
        range[0] = static_cast<int>(status);
    });
    require(statuses[0] == static_cast<int>(Status::Invalid), "empty closure");
}

void material_continuity()
{
    constexpr int nr = 7, nz = 11;
    rrea::MaxwellTMRZLayout const layout{nr, nz, 0.7, 2.3, -12.0};
    constexpr double a = 2.0e-9, b = 3.0e-9, dt = 1.0e-9;
    rrea::GpuVector<double> jr((nr+1)*nz), jz(nr*(nz+1));
    // Jr=a*r and Jz=b*z have analytic cylindrical divergence 2a+b,
    // including the axis cell and non-square grid spacing.
    for (int j = 0; j < nz; ++j) {
        for (int i = 0; i <= nr; ++i) { jr[layout.ErIndex(i,j)] = a*i*layout.dr; }
    }
    for (int j = 0; j <= nz; ++j) {
        for (int i = 0; i < nr; ++i) { jz[layout.EzIndex(i,j)] = b*j*layout.dz; }
    }
    rrea::GpuVector<rrea::MaterialChargeCell> charges(nr*nz);
    for (int j = 0; j < nz; ++j) {
        for (int i = 0; i < nr; ++i) {
            double const old = 100*rrea::qe / layout.CellVolume(i);
            double const now = old - dt*(2*a+b);
            charges[j*nr+i] = {old, now, 0.3*now, 0.7*now, 0, 0, 0};
        }
    }
    rrea::GpuVector<rrea::MaterialContinuitySample> samples(nr*nz);
    auto* out = samples.data();
    auto const* charge = charges.data();
    auto const* radial = jr.data();
    auto const* axial = jz.data();
    auto evaluate = [&] {
        rrea::GpuFor2D(rrea::GpuModule::Always, 0, nr, 0, nz, [=] RREA_DEVICE(int i, int j) noexcept {
            out[j*nr+i] = rrea::EvaluateMaterialContinuity(layout, i, j, dt,
                1.0e-7, charge[j*nr+i], radial, axial);
        });
    };
    evaluate();
    for (auto const& value : samples) {
        require(std::abs(value.divergence-(2*a+b)) <= 1.0e-22,
                "cylindrical divergence of linear currents");
        require(value.relative < rrea::kMaterialContinuityRelativeTolerance,
                "conservative material state must pass");
    }
    charges[0].new_charge += rrea::qe / layout.CellVolume(0);
    evaluate();
    require(samples[0].relative > rrea::kMaterialContinuityRelativeTolerance,
            "one unmatched electron must fail the continuity gate");
    require_close(samples[0].absolute_e, 1.0, "absolute charge residual", 512);
    charges[1].old_charge = std::numeric_limits<double>::quiet_NaN();
    evaluate();
    require(std::isinf(samples[1].relative) && std::isinf(samples[1].absolute_e),
            "nonfinite material state must not disappear in a max reduction");
}
} // namespace

int main()
{
    rrea::smoke::GpuSession runtime;
    try {
        execution_and_reductions();
        closure_interpolation();
        material_continuity();
    } catch (std::exception const& error) {
        std::cerr << "rrea_gpu_primitives_smoke: FAIL: " << error.what() << '\n';
        return 1;
    }
#ifdef RREA_USE_CUDA
    std::cout << "rrea_gpu_primitives_smoke: PASS (CUDA kernels)\n";
#else
    std::cout << "rrea_gpu_primitives_smoke: PASS (CPU backend)\n";
#endif
}
