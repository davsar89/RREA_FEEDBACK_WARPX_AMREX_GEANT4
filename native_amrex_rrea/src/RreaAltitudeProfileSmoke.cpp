#include "rrea/RreaAltitudeProfile.H"
#include "rrea/RreaSmokeRequire.H"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

std::filesystem::path write_csv(char const* stem, std::string const& text)
{
    auto const path = std::filesystem::temp_directory_path()
        / (std::string("rrea_altitude_profile_smoke_") + stem + ".csv");
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("could not write smoke fixture: " + path.string());
    }
    output << text;
    return path;
}

using rrea::smoke::require_throws;

// Two strictly-increasing altitude rows in km with a density ratio column,
// matching the production profile's shape.
constexpr char const* kGoodCsv =
    "altitude_km_MSL,ratio\n"
    "0.0,1.0\n"
    "10.0,0.25\n";

}  // namespace

int main()
{
    try {
        auto const good = write_csv("good", kGoodCsv);
        auto const zero = write_csv(
            "zero",
            "altitude_km_MSL,ratio\n"
            "0.0,1.0\n"
            "10.0,0.0\n");
        auto const negative = write_csv(
            "negative",
            "altitude_km_MSL,ratio\n"
            "0.0,1.0\n"
            "10.0,-0.25\n");

        auto load = [](std::filesystem::path const& path, bool require_positive) {
            rrea::RreaAltitudeProfile profile;
            profile.LoadCsv(
                path.string(), "altitude_km_MSL", "ratio",
                amrex::Real(1000.0), amrex::Real(1.0), require_positive);
            return profile;
        };

        // The density profile must fail closed at ingestion.  LowEnergyFluidState
        // documents this exact invariant ("validated strictly-positive at load")
        // and its AMREX_ASSERT compiles out in release, so an unenforced guard
        // silently substitutes a positive fallback instead of aborting.
        require_throws([&] { load(zero, true); }, "zero density row with require_positive");
        require_throws([&] { load(negative, true); }, "negative density row with require_positive");

        // Pin that the flag is what rejects them: with the permissive default the
        // same data loads, which is why every density call site must pass true.
        load(zero, false);
        load(negative, false);

        auto const profile = load(good, true);
        if (!profile.Loaded()) {
            throw std::runtime_error("good profile did not load");
        }
        rrea::smoke::require_close(profile.Interpolate(amrex::Real(0.0)), 1.0, "value at lower edge", 256.0);
        rrea::smoke::require_close(profile.Interpolate(amrex::Real(10000.0)), 0.25, "value at upper edge", 256.0);
        rrea::smoke::require_close(profile.Interpolate(amrex::Real(5000.0)), 0.625, "linear midpoint", 256.0);

        // RreaWarpXCoupling::TransportDensityRatioAtZ relies on these throws: it
        // queries unclamped so that a domain reaching past profile coverage
        // aborts instead of silently gaining a constant edge density.
        require_throws(
            [&] { static_cast<void>(profile.Interpolate(amrex::Real(-1.0))); },
            "altitude below profile range");
        require_throws(
            [&] { static_cast<void>(profile.Interpolate(amrex::Real(10001.0))); },
            "altitude above profile range");

        for (auto const& path : {good, zero, negative}) {
            std::filesystem::remove(path);
        }
    } catch (std::exception const& error) {
        std::cerr << "RREA altitude-profile smoke failed: " << error.what() << '\n';
        return 1;
    }
    std::cout << "RREA altitude-profile smoke passed\n";
    return 0;
}
