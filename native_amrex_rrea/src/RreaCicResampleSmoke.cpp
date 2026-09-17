// Native contract smoke for charge-exact CIC population thinning.

#include "RreaRng.H"
#include "rrea/RreaSmokeRequire.H"
#include "rrea/RreaCicConservingResample.H"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using rrea::RreaCicResampleParticle;
using rrea::warpx::RreaAdaptiveResampleCicUniform;
using rrea::warpx::RreaAdaptiveResampleSpecies;

using rrea::smoke::require;

std::array<long double, 4> deposited_cells(
    std::vector<RreaCicResampleParticle> const& particles)
{
    std::array<long double, 4> cells{};
    for (auto const& particle : particles) {
        double const fr = particle.fraction_r;
        double const fz = particle.fraction_z;
        double const w = particle.weight;
        cells[0] += w * (1.0 - fr) * (1.0 - fz);
        cells[1] += w * fr * (1.0 - fz);
        cells[2] += w * (1.0 - fr) * fz;
        cells[3] += w * fr * fz;
    }
    return cells;
}

std::size_t live_count(std::vector<RreaCicResampleParticle> const& particles)
{
    std::size_t live = 0;
    for (auto const& particle : particles) {
        live += particle.weight > 0.0 ? 1U : 0U;
    }
    return live;
}

std::vector<RreaCicResampleParticle> fixture()
{
    std::vector<RreaCicResampleParticle> particles;
    for (std::size_t i = 0; i < 19; ++i) {
        particles.push_back(RreaCicResampleParticle{
            0.03 + 0.91 * static_cast<double>((7 * i + 3) % 19) / 18.0,
            0.04 + 0.89 * static_cast<double>((11 * i + 5) % 19) / 18.0,
            0.25 + static_cast<double>((13 * i + 2) % 23) / 7.0});
    }
    return particles;
}

void exercise_exact_cic_conservation_and_reduction()
{
    auto particles = fixture();
    auto const before = deposited_cells(particles);
    std::size_t const before_count = live_count(particles);
    std::size_t const capacity =
        rrea::RreaCicResampleRemovalCapacity(particles);
    require(capacity == before_count - 4,
        "generic four-moment fixture must have CIC rank four");

    auto const result = rrea::RreaThinCicWeights(
        particles, capacity,
        [](std::size_t draw) {
            // Irrational rotation: deterministic open-interval uniforms that
            // exercise both null-space endpoints without involving the RNG
            // implementation under test in the conservation assertion.
            double const x = std::fmod(
                0.3141592653589793
                    + 0.6180339887498948 * static_cast<double>(draw),
                1.0);
            return x > 0.0 ? x : 0.5;
        });
    require(result.valid, "CIC thinning rejected a valid finite group");
    require(result.removed >= capacity,
        "CIC thinning did not realize the requested count reduction");
    require(live_count(particles) <= 4,
        "full CIC thinning left more particles than the constraint rank");

    auto const after = deposited_cells(particles);
    for (std::size_t cell = 0; cell < 4; ++cell) {
        long double const scale = std::max(1.0L, std::abs(before[cell]));
        require(
            std::abs(after[cell] - before[cell])
                <= 2048.0L * std::numeric_limits<double>::epsilon() * scale,
            "one realization changed a deposited CIC cell charge");
    }
    for (auto const& particle : particles) {
        require(std::isfinite(particle.weight) && particle.weight >= 0.0,
            "CIC thinning produced a negative or non-finite weight");
    }
}

void exercise_per_particle_unbiasedness()
{
    std::vector<RreaCicResampleParticle> const original{
        {0.08, 0.16, 0.7},
        {0.77, 0.13, 1.3},
        {0.21, 0.82, 2.1},
        {0.91, 0.68, 3.4},
        {0.47, 0.43, 5.2},
    };
    constexpr std::size_t trials = 120000;
    std::array<long double, 5> sum{};
    std::array<long double, 5> sum_square{};
    long double observable_sum = 0.0L;
    long double const coefficient[5] = {-2.0L, 0.5L, 7.0L, -0.25L, 3.0L};
    long double observable_expected = 0.0L;
    for (std::size_t i = 0; i < original.size(); ++i) {
        observable_expected += coefficient[i] * original[i].weight;
    }

    for (std::size_t trial = 0; trial < trials; ++trial) {
        auto particles = original;
        auto const result = rrea::RreaThinCicWeights(
            particles, 1,
            [&](std::size_t draw) {
                return RreaAdaptiveResampleCicUniform(
                    0x91f5a17ULL + trial,
                    0x2bdc545d6b4b87ULL,
                    2560,
                    draw,
                    RreaAdaptiveResampleSpecies::Electron);
            });
        require(result.valid && result.removed >= 1,
            "one-step unbiasedness fixture failed to remove a particle");
        long double observable = 0.0L;
        for (std::size_t i = 0; i < particles.size(); ++i) {
            long double const w = particles[i].weight;
            sum[i] += w;
            sum_square[i] += w * w;
            observable += coefficient[i] * w;
        }
        observable_sum += observable;
    }

    for (std::size_t i = 0; i < original.size(); ++i) {
        long double const mean = sum[i] / trials;
        long double const variance = std::max(
            0.0L, sum_square[i] / trials - mean * mean);
        long double const standard_error = std::sqrt(variance / trials);
        require(
            std::abs(mean - original[i].weight)
                <= 7.0L * standard_error + 2.0e-11L,
            "a particle weight is biased beyond seven standard errors");
    }
    long double const observable_mean = observable_sum / trials;
    require(std::abs(observable_mean - observable_expected) < 0.08L,
        "a heterogeneous weight-linear observable is biased");
}

void exercise_degenerate_and_boundary_supports()
{
    // Coincident particles give a rank-one CIC matrix.  The controller must
    // use that extra degeneracy rather than stopping mechanically at four.
    std::vector<RreaCicResampleParticle> coincident{
        {0.5, 0.0, 0.5},
        {0.5, 0.0, 0.75},
        {0.5, 0.0, 1.25},
        {0.5, 0.0, 2.0},
        {0.5, 0.0, 3.5},
        {0.5, 0.0, 5.0},
    };
    auto const before = deposited_cells(coincident);
    std::size_t const capacity =
        rrea::RreaCicResampleRemovalCapacity(coincident);
    require(capacity == coincident.size() - 1,
        "coincident CIC columns must expose rank-one removal capacity");
    auto const result = rrea::RreaThinCicWeights(
        coincident, capacity,
        [](std::size_t draw) {
            return draw % 2 == 0 ? 0.25 : 0.75;
        });
    require(result.valid && live_count(coincident) == 1,
        "rank-one CIC group did not thin to one representative");
    auto const after = deposited_cells(coincident);
    for (std::size_t cell = 0; cell < 4; ++cell) {
        long double const scale = std::max(1.0L, std::abs(before[cell]));
        require(
            std::abs(after[cell] - before[cell])
                <= 2048.0L * std::numeric_limits<double>::epsilon() * scale,
            "degenerate CIC group changed a deposited cell charge");
    }

    std::vector<RreaCicResampleParticle> rank_two;
    std::vector<RreaCicResampleParticle> rank_three;
    for (std::size_t i = 0; i < 7; ++i) {
        double const x = 0.08 + 0.13 * static_cast<double>(i);
        rank_two.push_back({x, 0.25, 1.0 + 0.2 * i});
        rank_three.push_back({x, x, 0.8 + 0.3 * i});
    }
    require(
        rrea::RreaCicResampleRemovalCapacity(rank_two)
            == rank_two.size() - 2,
        "constant-z CIC columns must expose rank-two capacity");
    require(
        rrea::RreaCicResampleRemovalCapacity(rank_three)
            == rank_three.size() - 3,
        "diagonal CIC columns must expose rank-three capacity");
    auto const rank_two_before = deposited_cells(rank_two);
    auto const rank_two_result = rrea::RreaThinCicWeights(
        rank_two, 2,
        [](std::size_t draw) { return draw == 0 ? 0.23 : 0.71; });
    require(rank_two_result.valid && rank_two_result.removed >= 2
            && live_count(rank_two) <= 5,
        "partial removal request did not reduce a rank-two group");
    auto const rank_two_after = deposited_cells(rank_two);
    for (std::size_t cell = 0; cell < 4; ++cell) {
        long double const scale =
            std::max(1.0L, std::abs(rank_two_before[cell]));
        require(
            std::abs(rank_two_after[cell] - rank_two_before[cell])
                <= 2048.0L * std::numeric_limits<double>::epsilon() * scale,
            "partial rank-two reduction changed a CIC cell charge");
    }

    // fr=1/2 is the raw support immediately below/above the RZ axis for a
    // particle at r=0; fz=0 is an exact cell-center/boundary shape endpoint.
    // Preserve the raw four entries before WarpX folds the radial ghost into
    // cell zero or drops a physical-z ghost contribution.
    std::vector<RreaCicResampleParticle> axis_boundary{
        {0.5, 0.0, 0.9},
        {0.5, 0.2, 1.1},
        {0.5, 0.4, 1.4},
        {0.5, 0.6, 1.8},
        {0.5, 0.8, 2.2},
    };
    auto const axis_before = deposited_cells(axis_boundary);
    auto const axis_result = rrea::RreaThinCicWeights(
        axis_boundary,
        rrea::RreaCicResampleRemovalCapacity(axis_boundary),
        [](std::size_t draw) { return draw % 2 == 0 ? 0.37 : 0.63; });
    require(axis_result.valid && axis_result.removed > 0,
        "axis/boundary raw-support group did not thin");
    auto const axis_after = deposited_cells(axis_boundary);
    for (std::size_t cell = 0; cell < 4; ++cell) {
        long double const scale = std::max(1.0L, std::abs(axis_before[cell]));
        require(
            std::abs(axis_after[cell] - axis_before[cell])
                <= 2048.0L * std::numeric_limits<double>::epsilon() * scale,
            "axis/boundary raw CIC contribution changed");
    }
}

void exercise_species_rng_separation()
{
    double const electron = RreaAdaptiveResampleCicUniform(
        9001, 42, 7, 3, RreaAdaptiveResampleSpecies::Electron);
    double const electron_repeat = RreaAdaptiveResampleCicUniform(
        9001, 42, 7, 3, RreaAdaptiveResampleSpecies::Electron);
    double const positron = RreaAdaptiveResampleCicUniform(
        9001, 42, 7, 3, RreaAdaptiveResampleSpecies::Positron);
    require(electron == electron_repeat,
        "CIC resampling RNG is not deterministic for a fixed group key");
    require(electron != positron,
        "electron and positron CIC resampling reused the same RNG stream");
    require(electron > 0.0 && electron < 1.0
            && positron > 0.0 && positron < 1.0,
        "CIC resampling RNG escaped the open unit interval");
}

std::vector<rrea::RreaGlobalCicParticle> global_fixture()
{
    std::vector<rrea::RreaGlobalCicParticle> particles;
    auto const local = fixture();
    for (std::size_t i = 0; i < local.size(); ++i) {
        particles.push_back(rrea::RreaGlobalCicParticle{
            17, -4, static_cast<int>(i % 5), static_cast<int>(i % 3),
            1000U + i, local[i].fraction_r, local[i].fraction_z,
            local[i].weight, i >= 15});
    }
    return particles;
}

auto deterministic_global_plan(
    std::vector<rrea::RreaGlobalCicParticle> const& particles,
    std::size_t target)
{
    return rrea::RreaPlanGlobalCicThinning<double>(
        particles, target, 4, 0x656c656374726f6eULL,
        [](std::uint64_t group, std::size_t reduction) {
            double const value = std::fmod(
                0.2718281828459045
                    + 0.6180339887498948 * static_cast<double>(reduction)
                    + static_cast<double>(group & 255U) / 521.0,
                1.0);
            return value > 0.0 ? value : 0.5;
        });
}

void exercise_global_candidate_capacity_and_floor()
{
    auto particles = global_fixture();
    std::vector<RreaCicResampleParticle> rank0;
    std::vector<RreaCicResampleParticle> rank1;
    for (std::size_t i = 0; i < 5; ++i) {
        auto const& p = particles[i];
        auto& local = i < 3 ? rank0 : rank1;
        local.push_back({p.fraction_r, p.fraction_z, p.weight});
    }
    require(rrea::RreaCicResampleRemovalCapacity(rank0) == 0
            && rrea::RreaCicResampleRemovalCapacity(rank1) == 0,
        "rank-local fragments unexpectedly had CIC capacity");
    auto five = std::vector<rrea::RreaGlobalCicParticle>(
        particles.begin(), particles.begin() + 5);
    five.back().candidate = true;
    // A lower-key singleton has zero capacity. It must be skipped, not
    // mistaken for a failed reduction before the thinnable mixed group.
    five.insert(five.begin(), rrea::RreaGlobalCicParticle{
        0, 0, 0, 0, 999U, 0.5, 0.5, 1.0, false});
    auto const candidate_plan = deterministic_global_plan(five, 5);
    require(candidate_plan.valid() && candidate_plan.capacity == 1
            && candidate_plan.geometric_floor == 5
            && candidate_plan.removed >= 1,
        "global survivor-plus-candidate support did not expose its true capacity");
    auto const mixed_support = std::find_if(
        candidate_plan.supports.begin(), candidate_plan.supports.end(),
        [](auto const& support) {
            return support.support_r == 17 && support.support_z == -4;
        });
    require(mixed_support != candidate_plan.supports.end()
            && mixed_support->owner_rank == rrea::RreaGlobalCicSupportOwner(
                17, -4, 4, 0x656c656374726f6eULL),
        "global CIC support owner was not deterministic");

    std::vector<rrea::RreaGlobalCicParticle> floors{
        {1, 2, 0, 0, 1, 0.5, 0.5, 1.0, false},
        {1, 2, 0, 1, 2, 0.5, 0.5, 2.0, true},
        {1, 2, 0, 2, 3, 0.5, 0.5, 3.0, false},
        {8, 9, 0, 0, 4, 0.1, 0.2, 1.0, false},
        {8, 9, 0, 1, 5, 0.7, 0.1, 1.0, false},
        {8, 9, 0, 2, 6, 0.2, 0.8, 1.0, false},
        {8, 9, 0, 3, 7, 0.9, 0.7, 1.0, false},
        {8, 9, 0, 0, 8, 0.4, 0.3, 1.0, true}};
    auto const floor_plan = deterministic_global_plan(floors, floors.size());
    require(floor_plan.valid() && floor_plan.geometric_floor == 5
            && floor_plan.capacity == 3,
        "global CIC geometric floor did not equal the sum of support ranks");
}

void exercise_decomposition_invariance_and_atomic_rejection()
{
    auto first = global_fixture();
    auto second = first;
    std::reverse(second.begin(), second.end());
    for (std::size_t i = 0; i < second.size(); ++i) {
        second[i].source_rank = static_cast<int>((2 * i + 1) % 4);
    }
    auto const plan_a = deterministic_global_plan(first, 7);
    auto const plan_b = deterministic_global_plan(second, 7);
    require(plan_a.valid() && plan_b.valid()
            && plan_a.capacity == plan_b.capacity
            && plan_a.geometric_floor == plan_b.geometric_floor
            && plan_a.removed == plan_b.removed
            && plan_a.updates.size() == plan_b.updates.size(),
        "global CIC plan changed under an MPI decomposition permutation");
    for (std::size_t i = 0; i < plan_a.updates.size(); ++i) {
        require(plan_a.updates[i].stable_id == plan_b.updates[i].stable_id
                && plan_a.updates[i].stored_weight
                    == plan_b.updates[i].stored_weight,
            "global CIC weights depended on source rank or input order");
    }

    auto const unchanged = first;
    auto const rejected = rrea::RreaPlanGlobalCicThinning<double>(
        first, 7, 4, 99,
        [](std::uint64_t, std::size_t) { return 0.0; });
    require(!rejected.valid() && rejected.updates.empty(),
        "invalid global CIC draw did not reject the complete plan atomically");
    for (std::size_t i = 0; i < first.size(); ++i) {
        require(first[i].stable_id == unchanged[i].stable_id
                && first[i].weight == unchanged[i].weight
                && first[i].candidate == unchanged[i].candidate,
            "rejected global CIC plan mutated a survivor, candidate, or ledger weight");
    }
}

void exercise_stored_particle_moment_preflight()
{
    std::vector<RreaCicResampleParticle> before{
        {0.08, 0.16, 0.7}, {0.77, 0.13, 1.3}, {0.21, 0.82, 2.1},
        {0.91, 0.68, 3.4}, {0.47, 0.43, 5.2}};
    auto planned = before;
    auto const result = rrea::RreaThinCicWeights(
        planned, 1, [](std::size_t) { return 0.37; });
    require(result.valid && result.removed >= 1,
        "stored-moment fixture could not produce a valid plan");
    std::vector<double> stored_double;
    require(rrea::RreaRoundAndVerifyCicWeights(
                before, planned, stored_double)
            && stored_double.size() == before.size(),
        "double ParticleReal rounding failed a charge-exact CIC plan");
    std::vector<float> stored_float;
    require(!rrea::RreaRoundAndVerifyCicWeights(
                before, planned, stored_float)
            && stored_float.empty(),
        "lower-precision stored weights bypassed the deposited-moment preflight");
}

}  // namespace

int main()
{
    try {
        exercise_exact_cic_conservation_and_reduction();
        exercise_per_particle_unbiasedness();
        exercise_degenerate_and_boundary_supports();
        exercise_species_rng_separation();
        exercise_global_candidate_capacity_and_floor();
        exercise_decomposition_invariance_and_atomic_rejection();
        exercise_stored_particle_moment_preflight();
    } catch (std::exception const& error) {
        std::cerr << "RREA CIC resample smoke failed: " << error.what() << '\n';
        return 1;
    }
    std::cout << "RREA CIC resample smoke passed\n";
    return 0;
}
