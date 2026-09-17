#include "rrea/RreaSeedSchedule.H"
#include "rrea/RreaSmokeRequire.H"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

std::string const header =
    "case_id,macro_index,time_s,x_m,y_m,z_m,r_m,kinetic_energy_eV,"
    "ux,uy,uz,weight_real_electrons\n";

std::string row(
    int macro_index,
    std::string const& time_s,
    std::string const& x_m,
    std::string const& y_m,
    std::string const& z_m,
    std::string const& r_m,
    std::string const& energy_eV,
    std::string const& ux,
    std::string const& uy,
    std::string const& uz,
    std::string const& weight)
{
    std::ostringstream output;
    output << "case," << macro_index << ',' << time_s << ',' << x_m << ','
           << y_m << ',' << z_m << ',' << r_m << ',' << energy_eV << ','
           << ux << ',' << uy << ',' << uz << ',' << weight << '\n';
    return output.str();
}

void write_file(std::filesystem::path const& path, std::string const& body)
{
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output << header << body;
    if (!output) {
        throw std::runtime_error("could not write seed-schedule smoke fixture");
    }
}

void validate(std::filesystem::path const& path)
{
    rrea::RreaSeedSchedule schedule;
    schedule.Load(path.string(), "case");
    schedule.ValidateProduction(1.0e3, 1.0e10, 0.0, 10.0, -5.0, 5.0);
}

void require_invalid(
    std::filesystem::path const& path,
    std::string const& body,
    char const* label)
{
    write_file(path, body);
    try {
        validate(path);
    } catch (std::exception const&) {
        return;
    }
    throw std::runtime_error(std::string("invalid schedule was accepted: ") + label);
}

std::string number(double value)
{
    std::ostringstream output;
    output << std::setprecision(17) << value;
    return output.str();
}

using rrea::smoke::require;

// Proper-velocity conversion.  Independent references:
//  (1) EXACT hand-derivable pin: E = rest/4 gives gamma = 1.25 exactly
//      (rest/4 and its ratio are exact in binary FP), gamma^2 - 1 =
//      0.5625 = 9/16, sqrt = 0.75 exact, so u == 0.75 * c BIT-FOR-BIT.
//  (2) Closure through the INVERSE identity the production validator
//      uses (E = rest * q * (q / (gamma + 1)) with gamma = hypot(1, q))
//      -- different algebra, so a transcription error in either
//      direction breaks the round trip.
void exercise_proper_velocity_conversion()
{
    using rrea::RreaElectronProperVelocityFromKineticEv;
    constexpr double rest_eV = 510998.95;
    constexpr double c = 299792458.0;
    require(
        RreaElectronProperVelocityFromKineticEv(rest_eV / 4.0) == 0.75 * c,
        "u(rest/4) must be exactly 3c/4");
    for (double e_eV : {1.0e3, 5.11e5, 1.0e6, 1.0e7, 1.0e9}) {
        double const u = RreaElectronProperVelocityFromKineticEv(e_eV);
        double const q = u / c;
        double const gamma = std::hypot(1.0, q);
        double const e_back = rest_eV * q * (q / (gamma + 1.0));
        // The production forward uses the ill-conditioned gamma^2 - 1
        // form, whose cancellation error is ~eps * gamma^2 relative to
        // gamma^2 - 1, i.e. an ABSOLUTE energy error ~eps * (E + rest)
        // -- at 1 keV that alone is ~250 eps of E.  Bound with the
        // correct error model rather than a relative fudge; a wrong sign
        // or a dropped -1 is off by ~rest and still fails by 10^9 x.
        require(
            std::abs(e_back - e_eV)
                <= 64.0 * std::numeric_limits<double>::epsilon()
                    * (e_eV + rest_eV),
            "proper velocity does not close through the validator inverse");
    }
    require(RreaElectronProperVelocityFromKineticEv(0.0) == 0.0
            && RreaElectronProperVelocityFromKineticEv(-5.0) == 0.0,
        "nonpositive energy must map to zero proper velocity");
}

// The magnitude-dispatch rule.  Hand-stated outcomes on both sides of the
// 1e6 m/s cliff and at the boundary itself (the guard is strict <, so
// exactly 1e6 passes through).
void exercise_seed_momentum_dispatch()
{
    using rrea::RreaNormalizeSeedMomentum;
    using rrea::RreaSeedDirectionHintMaxMPerS;
    constexpr double c = 299792458.0;
    double const u_1mev =
        rrea::RreaElectronProperVelocityFromKineticEv(1.0e6);
    // Direction hint (production path): magnitude replaced, direction
    // kept.  Norm of (3,0,4) is exactly 5, so expected components use the
    // same (u_mag * u) / norm operation order as the primitive -- the
    // CHECK is the dispatch branch and direction preservation, not the
    // arithmetic.
    auto const hint = RreaNormalizeSeedMomentum(3.0, 0.0, 4.0, 1.0e6);
    require(
        hint.ux == (u_1mev * 3.0) / 5.0 && hint.uy == 0.0
            && hint.uz == (u_1mev * 4.0) / 5.0,
        "direction hint must rescale to proper velocity along itself");
    // Exactly at the cliff: passthrough (strict <).
    auto const at = RreaNormalizeSeedMomentum(
        0.0, 0.0,
        static_cast<double>(RreaSeedDirectionHintMaxMPerS), 1.0e6);
    require(
        at.uz == static_cast<double>(RreaSeedDirectionHintMaxMPerS),
        "|u| exactly at the hint ceiling must pass through unchanged");
    // Proper-velocity input (validated branch): passthrough.
    auto const pv = RreaNormalizeSeedMomentum(0.0, 0.0, 0.9 * c, 1.0e6);
    require(pv.uz == 0.9 * c, "m/s input must pass through unchanged");
    // Hazard pins (documented, deliberate): zero/negative energy or zero
    // norm disables the rescale and hands the raw triple through.
    require(RreaNormalizeSeedMomentum(0.0, 0.0, 2.78, 0.0).uz == 2.78,
        "zero energy must disable the rescale (documented hazard)");
    require(RreaNormalizeSeedMomentum(0.0, 0.0, 0.0, 1.0e6).uz == 0.0,
        "zero direction must pass through unchanged");
}

// Window classifier boundaries: begin-inclusive, end-exclusive.
void exercise_seed_event_window()
{
    using rrea::RreaSeedEventWindowClass;
    using rrea::RreaSeedWindowClass;
    double const t = 5.0e-6;
    double const dt = 2.5e-9;
    require(RreaSeedEventWindowClass(t - 1.0e-12, t, t + dt)
            == RreaSeedWindowClass::SkipStale,
        "event before the window must be stale");
    require(RreaSeedEventWindowClass(t, t, t + dt)
            == RreaSeedWindowClass::Inject,
        "event exactly at window begin must inject");
    require(RreaSeedEventWindowClass(t + dt, t, t + dt)
            == RreaSeedWindowClass::BeyondWindow,
        "event exactly at window end belongs to the NEXT step");
    require(RreaSeedEventWindowClass(
                std::nextafter(t + dt, t), t, t + dt)
            == RreaSeedWindowClass::Inject,
        "event just inside the window end must inject");
}

}  // namespace

int main()
{
    try {
        exercise_proper_velocity_conversion();
        exercise_seed_momentum_dispatch();
        exercise_seed_event_window();
        auto const path = std::filesystem::temp_directory_path()
            / "rrea_seed_schedule_smoke.csv";
        constexpr double rest_eV = 510998.95;
        constexpr double c = 299792458.0;
        double const proper_1MeV = c * std::sqrt(
            std::pow(1.0 + 1.0e6 / rest_eV, 2) - 1.0);
        write_file(
            path,
            row(0, "0", "1", "0", "0", "1", "1000", "0", "0", "1", "1")
            + row(1, "1", "2", "0", "1", "2", "10000000000", "0", "0", "1", "2")
            + row(2, "2", "3", "0", "2", "3", "1000000", "0", "0",
                  number(proper_1MeV), "3")
            // Off-axis position (y != 0, r = hypot nontrivial) with a full
            // 3-D downward direction hint -- the general continuous-source
            // row shape (PARMA cosmic seeds).
            + row(3, "3", "3", "4", "1", "5", "2500000",
                  "0.3", "-0.4", "-0.866", "4"));
        validate(path);

        auto const base = row(
            0, "0", "1", "0", "0", "1", "1000", "0", "0", "1", "1");
        require_invalid(
            path,
            row(0, "0", "1", "0", "0", "1",
                number(std::nextafter(1.0e3, 0.0)), "0", "0", "1", "1"),
            "below minimum energy");
        require_invalid(
            path,
            row(0, "0", "1", "0", "0", "1",
                number(std::nextafter(1.0e10, INFINITY)), "0", "0", "1", "1"),
            "above maximum energy");
        require_invalid(
            path,
            row(0, "0", "1", "0", "0", "1", "nan", "0", "0", "1", "1"),
            "NaN energy");
        require_invalid(
            path,
            row(0, "-1", "1", "0", "0", "1", "1000", "0", "0", "1", "1"),
            "negative time");
        require_invalid(
            path,
            row(0, "0", "1", "0", "0", "1", "1000", "0", "0", "1", "0"),
            "zero weight");
        require_invalid(
            path,
            row(-1, "0", "1", "0", "0", "1", "1000", "0", "0", "1", "1"),
            "negative macro index");
        require_invalid(path, base + base, "duplicate macro index");
        require_invalid(
            path,
            row(0, "0", "10", "0", "0", "10", "1000", "0", "0", "1", "1"),
            "position on high boundary");
        require_invalid(
            path,
            row(0, "0", "1", "0", "0", "2", "1000", "0", "0", "1", "1"),
            "declared radius mismatch");
        require_invalid(
            path,
            row(0, "0", "1", "0", "0", "1", "1000", "0", "0", "0", "1"),
            "zero direction");
        require_invalid(
            path,
            row(0, "0", "1", "0", "0", "1", "1000", "0", "0",
                number(c), "1"),
            "proper velocity energy mismatch");

        // species_id column: absent (all fixtures above) pins the electron
        // default; present it must be 1 or 2 -- a positron row (PARMA
        // cosmic seeds) loads and validates through the same primitives.
        auto const write_species_file = [&](std::string const& body) {
            std::ofstream output(path, std::ios::binary | std::ios::trunc);
            output << "case_id,macro_index,time_s,x_m,y_m,z_m,r_m,"
                      "kinetic_energy_eV,ux,uy,uz,weight_real_electrons,"
                      "species_id\n"
                   << body;
            if (!output) {
                throw std::runtime_error(
                    "could not write seed-schedule species fixture");
            }
        };
        write_species_file(
            "case,0,0,1,0,0,1,1000,0,0,1,1,1\n"
            "case,1,1,3,4,1,5,2500000,0.3,-0.4,-0.866,4,2\n");
        validate(path);
        write_species_file("case,0,0,1,0,0,1,1000,0,0,1,1,3\n");
        bool species_rejected = false;
        try {
            validate(path);
        } catch (std::exception const&) {
            species_rejected = true;
        }
        require(species_rejected, "species_id=3 must be rejected");

        std::filesystem::remove(path);
        std::cout << "PASS: schema-6 retained seed schedule validation\n";
        return 0;
    } catch (std::exception const& error) {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }
}
