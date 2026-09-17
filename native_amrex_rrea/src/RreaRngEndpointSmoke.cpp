// Endpoint and stream-pin smoke for the counter-based overlay RNG.
//
// Uniform01 must stay strictly inside (0,1) at every CDF/optical-depth
// endpoint. The exact form uses the top 52 counter bits: every c + 0.5 is
// representable there, so (c52 + 0.5) * 2^-52 lies in [2^-53, 1 - 2^-53]
// with every operation exact.
//
// The golden values below pin the schema-6 RNG stream
// (counter_splitmix64_v1 + 52-bit midpoint).  Any change to splitmix64, the
// key mixing, or the endpoint mapping is a physics-stream change and MUST
// bump the scheme name; this smoke is what catches a silent one.

#include "RreaRng.H"
#include "rrea/RreaSmokeRequire.H"
#include "rrea/RreaUniformStream.H"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using rrea::warpx::RreaRng;
using rrea::warpx::RreaRngKey;

struct GoldenPin {
    RreaRngKey key;
    std::uint64_t counter;
    double uniform;
};

// Values independently derived from the documented counter_splitmix64_v1
// definition (seed/particle/step/interaction/channel splitmix64 XOR mix).
constexpr GoldenPin kGoldenPins[] = {
    {{0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL},
     0x3db5adc86dfdd12dULL, 0x1.edad6e436feecp-3},
    {{0x1ULL, 0x2ULL, 0x3ULL, 0x4ULL, 0x5ULL},
     0x5d3617c86eb12224ULL, 0x1.74d85f21bac4ap-2},
    {{0xdeadbeefULL, 0x2bdc545d6b4b87ULL, 0x2aULL, 0x7ULL, 0x33ULL},
     0xeb74a9e550130b92ULL, 0x1.d6e953caa0261p-1},
    {{0x7eaULL, 0x1ULL, 0x186a0ULL, 0x3ULL, 0x63ULL},
     0xaceff5afeec79e3cULL, 0x1.59dfeb5fdd8f3p-1},
    {{0xffffffffffffffffULL, 0xffffffffffffffffULL, 0xffffffffffffffffULL,
      0xffffffffffffffffULL, 0xffffffffffffffffULL},
     0xb22a341e3093f92fULL, 0x1.6454683c6127fp-1},
    {{0x75bcd15ULL, 0x3ade68b1ULL, 0x37ULL, 0x0ULL, 0x6bULL},
     0x9633b556ab3a3a1fULL, 0x1.2c676aad56747p-1},
};

using rrea::smoke::require;

// The split/refill branch's decision rules are deterministic and use no RNG.
// The checks pin the hysteresis
// band, the inclusive 2x-floor guard, and the exact-FP weight conservation
// of halving.  A wrong floor silently dilutes macros below the injected
// seed weight; a wrong trigger makes the controller thrash or never refill.
void exercise_adaptive_resample_split()
{
    using rrea::warpx::RreaAdaptiveResampleAction;
    using rrea::warpx::RreaAdaptiveResampleSplitOutcome;
    using rrea::warpx::RreaResampleAction;

    // Hysteresis band at the production target (adaptive_resample_v1
    // target 1e6): thin strictly above 1.25x, split strictly below 0.5x,
    // dead band between, and an empty species never splits.
    long const target = 1000000;
    double const floor_w = 1.0e9;  // production split floor = seed weight
    require(
        RreaAdaptiveResampleAction(target + target / 4 + 1, target, floor_w)
            == RreaResampleAction::Thin,
        "count just above 1.25x target must thin");
    require(
        RreaAdaptiveResampleAction(target + target / 4, target, floor_w)
            == RreaResampleAction::None,
        "count at exactly 1.25x target is inside the dead band");
    require(
        RreaAdaptiveResampleAction(target / 2, target, floor_w)
            == RreaResampleAction::None,
        "count at exactly 0.5x target is inside the dead band");
    require(
        RreaAdaptiveResampleAction(target / 2 - 1, target, floor_w)
            == RreaResampleAction::Split,
        "count just below 0.5x target must split");
    require(
        RreaAdaptiveResampleAction(0, target, floor_w)
            == RreaResampleAction::None,
        "an empty species must never split");
    require(
        RreaAdaptiveResampleAction(target / 2 - 1, target, 0.0)
            == RreaResampleAction::None,
        "splitting must stay disabled when no split floor is configured");
    require(
        RreaAdaptiveResampleAction(3 * target, target, 0.0)
            == RreaResampleAction::Thin,
        "thinning must not depend on the split floor");

    // Truncating integer division of the triggers (odd target=7:
    // thin above 7+7/4=8, split below 7/2=3).
    require(
        RreaAdaptiveResampleAction(8, 7, 1.0) == RreaResampleAction::None
            && RreaAdaptiveResampleAction(9, 7, 1.0)
                == RreaResampleAction::Thin
            && RreaAdaptiveResampleAction(3, 7, 1.0)
                == RreaResampleAction::None
            && RreaAdaptiveResampleAction(2, 7, 1.0)
                == RreaResampleAction::Split,
        "trigger integer division changed (target/4, target/2 truncation)");

    // Floor guard: strictly below 2x floor never splits and the weight
    // passes through untouched; exactly 2x floor splits inclusively with
    // children landing exactly ON the floor, never below it.
    auto const below = RreaAdaptiveResampleSplitOutcome(
        std::nextafter(2.0 * floor_w, 0.0), floor_w);
    require(!below.split, "parent below 2x split floor must not split");
    require(
        below.child_weight == std::nextafter(2.0 * floor_w, 0.0),
        "ineligible parent weight must pass through unchanged");
    auto const at_floor =
        RreaAdaptiveResampleSplitOutcome(2.0 * floor_w, floor_w);
    require(
        at_floor.split && at_floor.child_weight == floor_w,
        "parent at exactly 2x floor must split to children at the floor");
    require(
        !RreaAdaptiveResampleSplitOutcome(4.0 * floor_w, 0.0).split,
        "a non-positive split floor must disable splitting entirely");

    // One synthetic doubling round over heterogeneous weights spanning the
    // floor: halving a normal double only decrements its exponent, so
    // parent + clone reconstruct the original weight BIT-FOR-BIT and the
    // round conserves every weight sum exactly -- the split analogue of the
    // roulette's total_after == 2*survivor_before identity above.
    double total_before = 0.0;
    double total_after = 0.0;
    std::size_t clones = 0;
    for (std::size_t i = 0; i < 1000; ++i) {
        double const w =
            floor_w * (0.5 + 0.01 * static_cast<double>(i % 400));
        total_before += w;
        auto const outcome = RreaAdaptiveResampleSplitOutcome(w, floor_w);
        if (outcome.split) {
            require(
                outcome.child_weight + outcome.child_weight == w,
                "split halving must be exact in binary floating point");
            require(
                outcome.child_weight >= floor_w,
                "split child fell below the configured floor");
            total_after += outcome.child_weight + outcome.child_weight;
            ++clones;
        } else {
            require(
                outcome.child_weight == w,
                "unsplit parent weight must be unchanged");
            total_after += outcome.child_weight;
        }
    }
    require(
        total_after == total_before,
        "a split round must conserve total weight exactly");
    require(
        clones > 0 && clones < 1000,
        "synthetic population must exercise both sides of the floor guard");
}

}  // namespace

int main()
{
    try {
        require(
            std::string(RreaRng::Scheme()) == "counter_splitmix64_v1",
            "unexpected RNG scheme name");
        exercise_adaptive_resample_split();

        for (auto const& pin : kGoldenPins) {
            std::uint64_t const counter = RreaRng::Counter(pin.key);
            double const uniform = RreaRng::Uniform01(pin.key);
            if (counter != pin.counter || uniform != pin.uniform) {
                std::fprintf(
                    stderr,
                    "golden pin mismatch for key {%llu,%llu,%llu,%llu,%llu}: "
                    "counter=0x%016llx (expected 0x%016llx) uniform=%a "
                    "(expected %a)\n",
                    static_cast<unsigned long long>(pin.key.seed),
                    static_cast<unsigned long long>(pin.key.particle_id),
                    static_cast<unsigned long long>(pin.key.step),
                    static_cast<unsigned long long>(pin.key.interaction_index),
                    static_cast<unsigned long long>(pin.key.channel_id),
                    static_cast<unsigned long long>(counter),
                    static_cast<unsigned long long>(pin.counter),
                    uniform, pin.uniform);
                throw std::runtime_error("RNG golden stream pin mismatch");
            }
        }

        // Exact open-interval bounds of the 52-bit midpoint mapping, used by
        // the sweep below both as the admissible range and to reconstruct
        // Uniform01 from Counter exactly.
        constexpr double scale = 0x1.0p-52;
        constexpr double lowest = (0.0 + 0.5) * scale;
        constexpr double highest =
            (static_cast<double>((1ULL << 52U) - 1ULL) + 0.5) * scale;

        // Sweep a large deterministic key family: strict open interval,
        // exact consistency with Counter, and basic distribution sanity.
        constexpr std::uint64_t kSweep = 1u << 20U;
        double sum = 0.0;
        double minimum = 1.0;
        double maximum = 0.0;
        for (std::uint64_t index = 0; index < kSweep; ++index) {
            RreaRngKey key;
            key.seed = 0x51ab5eedULL + (index % 7U);
            key.particle_id = index * 0x9e3779b9ULL;
            key.step = index % 4096U;
            key.interaction_index = index / 4096U;
            key.channel_id = 51U + (index % 60U);
            double const uniform = RreaRng::Uniform01(key);
            std::uint64_t const c52 = RreaRng::Counter(key) >> 12U;
            require(
                uniform == (static_cast<double>(c52) + 0.5) * scale,
                "Uniform01 is not the exact midpoint of its counter");
            require(
                uniform >= lowest && uniform <= highest,
                "Uniform01 escaped the open interval");
            sum += uniform;
            minimum = std::fmin(minimum, uniform);
            maximum = std::fmax(maximum, uniform);
        }
        double const mean = sum / static_cast<double>(kSweep);
        require(
            std::fabs(mean - 0.5) < 2.0e-3,
            "Uniform01 sweep mean is not near 0.5: " + std::to_string(mean));
        require(minimum < 1.0e-4, "Uniform01 sweep never approached zero");
        require(maximum > 1.0 - 1.0e-4, "Uniform01 sweep never approached one");

        // RreaUniformStream expands two pre-drawn uniforms into the unbounded
        // sequence a rejection loop needs.  Nothing constructed one before, so
        // the annihilation accept path ran on an unverified generator.  Its
        // whole purpose is to stay uniform and bit-reproducible for a fixed
        // key, so those are what is checked.
        {
            int const draws = 200000;
            int const buckets = 20;
            std::vector<long> occupancy(buckets, 0);
            rrea::RreaUniformStream stream(0.25, 0.75);
            double stream_sum = 0.0;
            for (int i = 0; i < draws; ++i) {
                double const value = stream.Next();
                require(value >= 0.0 && value < 1.0,
                        "RreaUniformStream escaped [0, 1): " + std::to_string(value));
                stream_sum += value;
                occupancy[static_cast<std::size_t>(value * buckets)] += 1;
            }
            double const stream_mean = stream_sum / draws;
            require(std::fabs(stream_mean - 0.5) < 5.0e-3,
                    "RreaUniformStream mean is not near 0.5: "
                        + std::to_string(stream_mean));
            double const expected = static_cast<double>(draws) / buckets;
            for (long count : occupancy) {
                require(std::fabs(count - expected) < 0.05 * expected,
                        "RreaUniformStream bucket occupancy is not uniform: "
                            + std::to_string(count));
            }

            // Bit-reproducible for a fixed key: two streams seeded alike must
            // agree exactly, or a rerun of the same event would not replay.
            rrea::RreaUniformStream first(0.25, 0.75);
            rrea::RreaUniformStream second(0.25, 0.75);
            for (int i = 0; i < 256; ++i) {
                require(first.Next() == second.Next(),
                        "RreaUniformStream diverged for an identical key");
            }

            // One ulp of seed separation must decorrelate the sequences,
            // otherwise neighbouring events share rejection outcomes.
            rrea::RreaUniformStream nearby(0.25, std::nextafter(0.75, 1.0));
            rrea::RreaUniformStream base(0.25, 0.75);
            int matches = 0;
            for (int i = 0; i < 256; ++i) {
                if (base.Next() == nearby.Next()) { ++matches; }
            }
            require(matches == 0,
                    "RreaUniformStream repeated a one-ulp-separated key: "
                        + std::to_string(matches) + " of 256 draws");
        }
    } catch (std::exception const& error) {
        std::cerr << "RREA RNG endpoint smoke failed: " << error.what() << '\n';
        return 1;
    }
    std::cout << "RREA RNG endpoint smoke passed\n";
    return 0;
}
