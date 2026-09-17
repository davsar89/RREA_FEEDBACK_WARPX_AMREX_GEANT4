#include "RreaRng.H"

#include <limits>

namespace rrea::warpx {

namespace {

std::uint64_t splitmix64(std::uint64_t value) noexcept
{
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

}  // namespace

char const* RreaRng::Scheme() noexcept
{
    return "counter_splitmix64_v1";
}

std::uint64_t RreaRng::Counter(RreaRngKey key) noexcept
{
    std::uint64_t state = splitmix64(key.seed);
    state ^= splitmix64(key.particle_id + 0x100000001b3ULL);
    state ^= splitmix64(key.step + 0x9e3779b97f4a7c15ULL);
    state ^= splitmix64(key.interaction_index + 0xbf58476d1ce4e5b9ULL);
    state ^= splitmix64(key.channel_id + 0x94d049bb133111ebULL);
    return splitmix64(state);
}

double RreaRng::Uniform01(RreaRngKey key) noexcept
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
