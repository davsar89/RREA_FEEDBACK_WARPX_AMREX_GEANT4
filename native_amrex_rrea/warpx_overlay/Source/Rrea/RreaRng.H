#ifndef WARPX_RREA_RNG_H
#define WARPX_RREA_RNG_H

#include <cstdint>
#include "rrea/RreaKineticSupport.H"

namespace rrea::warpx {

struct RreaRngKey {
    std::uint64_t seed = 0;
    std::uint64_t particle_id = 0;
    std::uint64_t step = 0;
    std::uint64_t interaction_index = 0;
    std::uint64_t channel_id = 0;
};

enum class RreaRngChannel : std::uint64_t {
    ElectronHardMollerEvent = 21,
    ElectronHardMollerEnergy = 22,
    ElectronHardMollerAzimuth = 23,
    BremsEvent = 41,
    BremsPhotonEnergy = 42,
    BremsPhotonAngle = 43,
    BremsPhotonAzimuth = 44,
    PhotonInteraction = 51,
    PhotonChannel = 52,
    ComptonRecoil = 53,
    ComptonScatter = 54,
    Photoelectric = 55,
    PairAngle = 57,
    ComptonScatterAzimuth = 58,
    PairAzimuth = 59,
    PhotoelectricAngle = 60,
    PhotoelectricAzimuth = 65,
    PairInvariant = 66,
    PairLeptonPolar = 67,
    PairLeptonAzimuth = 68,
    PositronBhabhaEvent = 73,
    PositronBhabhaEnergy = 74,
    PositronBremsPhotonEnergy = 76,
    PositronBremsPhotonAngle = 77,
    PositronBremsPhotonAzimuth = 78,
    PositronAnnihilationPolar = 80,
    PositronAnnihilationAzimuth = 81,
    // Goudsmit-Saunderson elastic streams.
    // The few-collision Poisson branch varies interaction_index by the sub-scatter
    // index k so each single scatter draws an independent (cos,phi) stream.
    ElectronGsAngle = 89,
    ElectronGsAzimuth = 90,
    ElectronGsFewCount = 91,
    ElectronGsFewScatter = 92,
    ElectronGsFewAzimuth = 93,
    PositronGsAngle = 94,
    PositronGsAzimuth = 95,
    PositronGsFewCount = 96,
    PositronGsFewScatter = 97,
    PositronGsFewAzimuth = 98,
    // Charged competing-hazard transport. Earlier channel ids remain reserved
    // so no numerical stream is reassigned.
    ElectronCompetingOpticalDepth = 107,
    ElectronCompetingChannel = 108,
    PositronCompetingOpticalDepth = 109,
    PositronCompetingChannel = 110,
    // Reserved retired roulette id; CIC-conserving resampling uses 131--132.
    AdaptiveResampleRoulette = 111,
    // Newborn residual-advance GS elastic streams. The child
    // has no AMReX id until AddNParticles, so these streams are keyed on the
    // original tracked root and the checked 52-bit transient index packed by
    // RreaTryPackTransientRngIndex.  Append-only.
    NewbornElectronGsAngle = 112,
    NewbornElectronGsAzimuth = 113,
    NewbornElectronGsFewCount = 114,
    NewbornElectronGsFewScatter = 115,
    NewbornElectronGsFewAzimuth = 116,
    NewbornPositronGsAngle = 117,
    NewbornPositronGsAzimuth = 118,
    NewbornPositronGsFewCount = 119,
    NewbornPositronGsFewScatter = 120,
    NewbornPositronGsFewAzimuth = 121,
    // Optical-depth and channel draws for transient newborns.
    NewbornCompetingOpticalDepth = 122,
    NewbornCompetingChannel = 123,
    // Continuation hazards are disjoint from the first-event streams.
    ElectronContinuationOpticalDepth = 124,
    ElectronContinuationChannel = 125,
    PositronContinuationOpticalDepth = 126,
    PositronContinuationChannel = 127,
    PhotonContinuationOpticalDepth = 128,
    PhotonContinuationChannel = 129,

    // Dedicated Bhabha azimuth channel, disjoint from the energy draw.
    PositronBhabhaAzimuth = 130,
    // CIC-conserving adaptive resampling.  Electron and positron reductions
    // own distinct append-only channels so equal AMReX ids cannot correlate
    // the two species.  Keyed {seed, stable_group_id, step, reduction_index,
    // species channel}.
    AdaptiveResampleCicElectron = 131,
    AdaptiveResampleCicPositron = 132,
};

class RreaRng {
public:
    RREA_HOST_DEVICE static char const* Scheme() noexcept;
    RREA_HOST_DEVICE static std::uint64_t Counter(RreaRngKey key) noexcept;
    RREA_HOST_DEVICE static double Uniform01(RreaRngKey key) noexcept;
};

enum class RreaAdaptiveResampleSpecies : std::uint64_t {
    Electron = 0,
    Positron = 1,
};

[[nodiscard]] RREA_HOST_DEVICE inline double RreaAdaptiveResampleCicUniform(
    std::uint64_t rng_seed,
    std::uint64_t stable_group_id,
    std::uint64_t step,
    std::uint64_t reduction_index,
    RreaAdaptiveResampleSpecies species) noexcept
{
    RreaRngChannel const channel =
        species == RreaAdaptiveResampleSpecies::Electron
            ? RreaRngChannel::AdaptiveResampleCicElectron
            : RreaRngChannel::AdaptiveResampleCicPositron;
    return RreaRng::Uniform01(RreaRngKey{
        rng_seed,
        stable_group_id,
        step,
        reduction_index,
        static_cast<std::uint64_t>(channel)});
}

// The population controller's action selector, extracted header-inline so
// the native smoke can pin the hysteresis band (roulette-outcome precedent
// above): thinning engages strictly above 1.25x target, the split refill
// strictly below 0.5x target and only when a positive split floor is
// configured; the band between is a dead zone that prevents thrash.
// Deterministic, no RNG.
enum class RreaResampleAction { None, Thin, Split };

[[nodiscard]] RREA_HOST_DEVICE inline RreaResampleAction RreaAdaptiveResampleAction(
    long count, long target, double split_min_weight) noexcept
{
    if (count > target + target / 4) {
        return RreaResampleAction::Thin;
    }
    if (count > 0 && count < target / 2 && split_min_weight > 0.0) {
        return RreaResampleAction::Split;
    }
    return RreaResampleAction::None;
}

// The split/refill rule for one macro: a parent at or above twice the
// split floor halves its weight and emits an identical clone.  Halving a
// normal double is exact, so parent + clone reconstruct the original
// weight bit-for-bit -- conservation is exact, not just in expectation --
// and the inclusive 2x-floor guard means children never fall below the
// configured floor (the injected seed weight in production, so splitting
// never dilutes macros below physical seed granularity).
struct RreaSplitOutcome {
    bool split = false;
    double child_weight = 0.0;  // parent's new weight AND the clone weight
};

[[nodiscard]] RREA_HOST_DEVICE inline RreaSplitOutcome RreaAdaptiveResampleSplitOutcome(
    double weight, double split_min_weight) noexcept
{
    if (!(split_min_weight > 0.0) || weight < 2.0 * split_min_weight) {
        return RreaSplitOutcome{false, weight};
    }
    return RreaSplitOutcome{true, 0.5 * weight};
}



namespace rng_detail {

RREA_HOST_DEVICE inline std::uint64_t splitmix64(std::uint64_t value) noexcept
{
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

}  // namespace

RREA_HOST_DEVICE inline char const* RreaRng::Scheme() noexcept
{
    return "counter_splitmix64_v1";
}

RREA_HOST_DEVICE inline std::uint64_t RreaRng::Counter(RreaRngKey key) noexcept
{
    std::uint64_t state = rng_detail::splitmix64(key.seed);
    state ^= rng_detail::splitmix64(key.particle_id + 0x100000001b3ULL);
    state ^= rng_detail::splitmix64(key.step + 0x9e3779b97f4a7c15ULL);
    state ^= rng_detail::splitmix64(key.interaction_index + 0xbf58476d1ce4e5b9ULL);
    state ^= rng_detail::splitmix64(key.channel_id + 0x94d049bb133111ebULL);
    return rng_detail::splitmix64(state);
}

RREA_HOST_DEVICE inline double RreaRng::Uniform01(RreaRngKey key) noexcept
{
    // Midpoint sampling on the top 52 bits with exact double arithmetic.
    // A uint64-to-double conversion near 2^64 can round to exactly 1.0, and
    // (2^53 - 1) + 0.5 is also not representable. With 52 bits every c + 0.5
    // is representable (spacing at most 0.5 below
    // 2^52), so each operation is exact and the result lies in
    // [2^-53, 1 - 2^-53].
    constexpr double scale = 0x1.0p-52;
    return (static_cast<double>(Counter(key) >> 12U) + 0.5) * scale;
}


}  // namespace rrea::warpx

#endif
