#include "rrea/RreaSeedSchedule.H"
#include "rrea/RreaCsvTextUtil.H"

#include <AMReX_BLassert.H>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <limits>
#include <iterator>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace rrea {

namespace {

using rrea::csv_text::split_csv_line;
using rrea::csv_text::trim;

template <typename T>
T parse_number(std::string const& text, char const* column)
{
    std::istringstream stream(text);
    T value{};
    stream >> value;
    if (!stream || !stream.eof()) {
        throw std::runtime_error(std::string("Invalid numeric value in seed schedule column ")
            + column + ": " + text);
    }
    return value;
}

std::string require_cell(
    amrex::Vector<std::string> const& row,
    std::map<std::string, int> const& columns,
    char const* name)
{
    auto const it = columns.find(name);
    if (it == columns.end()) {
        throw std::runtime_error(std::string("Missing seed schedule column: ") + name);
    }
    int const index = it->second;
    if (index < 0 || static_cast<std::size_t>(index) >= row.size()) {
        return {};
    }
    return row[static_cast<std::size_t>(index)];
}

std::string optional_cell(
    amrex::Vector<std::string> const& row,
    std::map<std::string, int> const& columns,
    char const* name)
{
    auto const it = columns.find(name);
    if (it == columns.end()) {
        return {};
    }
    int const index = it->second;
    if (index < 0 || static_cast<std::size_t>(index) >= row.size()) {
        return {};
    }
    return row[static_cast<std::size_t>(index)];
}

}  // namespace

void RreaSeedSchedule::Load(std::string const& path, std::string const& case_id)
{
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Could not open RREA seed schedule: " + path);
    }

    std::string header_line;
    if (!std::getline(input, header_line)) {
        throw std::runtime_error("Empty RREA seed schedule: " + path);
    }
    auto const headers = split_csv_line(header_line);
    std::map<std::string, int> columns;
    for (std::size_t i = 0; i < headers.size(); ++i) {
        auto const [_, inserted] = columns.emplace(headers[i], static_cast<int>(i));
        if (!inserted) {
            throw std::runtime_error("Duplicate seed schedule column: " + headers[i]);
        }
    }

    amrex::Vector<RreaSeedEvent> events;
    std::string line;
    int line_number = 1;
    while (std::getline(input, line)) {
        ++line_number;
        if (trim(line).empty()) {
            continue;
        }
        auto const row = split_csv_line(line);
        RreaSeedEvent event;
        event.case_id = require_cell(row, columns, "case_id");
        if (!case_id.empty() && event.case_id != case_id) {
            continue;
        }
        std::string const macro = optional_cell(row, columns, "macro_index");
        event.macro_index = macro.empty() ? line_number : parse_number<int>(macro, "macro_index");
        std::string const species = optional_cell(row, columns, "species_id");
        event.species_id = species.empty()
            ? RreaSeedSpeciesElectron
            : parse_number<int>(species, "species_id");
        event.time_s = parse_number<amrex::Real>(
            require_cell(row, columns, "time_s"),
            "time_s");
        event.x_m = parse_number<amrex::Real>(require_cell(row, columns, "x_m"), "x_m");
        event.y_m = parse_number<amrex::Real>(require_cell(row, columns, "y_m"), "y_m");
        event.z_m = parse_number<amrex::Real>(require_cell(row, columns, "z_m"), "z_m");
        event.r_m = parse_number<amrex::Real>(require_cell(row, columns, "r_m"), "r_m");
        event.kinetic_energy_eV = parse_number<amrex::Real>(
            require_cell(row, columns, "kinetic_energy_eV"),
            "kinetic_energy_eV");
        event.ux = parse_number<amrex::Real>(require_cell(row, columns, "ux"), "ux");
        event.uy = parse_number<amrex::Real>(require_cell(row, columns, "uy"), "uy");
        event.uz = parse_number<amrex::Real>(require_cell(row, columns, "uz"), "uz");
        event.weight_real_electrons = parse_number<amrex::Real>(
            require_cell(row, columns, "weight_real_electrons"),
            "weight_real_electrons");
        events.push_back(std::move(event));
    }

    std::sort(events.begin(), events.end(), [](auto const& lhs, auto const& rhs) {
        if (lhs.time_s != rhs.time_s) {
            return lhs.time_s < rhs.time_s;
        }
        return lhs.macro_index < rhs.macro_index;
    });

    m_path = path;
    m_case_id = case_id;
    m_events = std::move(events);
}

void RreaSeedSchedule::ValidateProduction(
    amrex::Real electron_min_energy_eV,
    amrex::Real electron_max_energy_eV,
    amrex::Real radial_min_m,
    amrex::Real radial_max_m,
    amrex::Real axial_min_m,
    amrex::Real axial_max_m) const
{
    auto const finite = [](amrex::Real value) {
        return std::isfinite(static_cast<double>(value));
    };
    if (!finite(electron_min_energy_eV) || !finite(electron_max_energy_eV)
        || !(electron_min_energy_eV > amrex::Real(0.0))
        || electron_max_energy_eV < electron_min_energy_eV
        || !finite(radial_min_m) || !finite(radial_max_m)
        || !finite(axial_min_m) || !finite(axial_max_m)
        || !(radial_max_m > radial_min_m)
        || !(axial_max_m > axial_min_m)) {
        throw std::runtime_error(
            "RREA seed validation received invalid schema-6 energy/domain bounds");
    }

    constexpr amrex::Real rest_energy_eV = amrex::Real(rrea::me_c2_eV);
    constexpr amrex::Real c_m_per_s = amrex::Real(rrea::c_m_per_s);
    // Shared with the injection hook's momentum dispatch -- see the
    // documentation on the constant in RreaSeedSchedule.H.
    constexpr amrex::Real direction_hint_max_m_per_s =
        RreaSeedDirectionHintMaxMPerS;
    std::set<int> macro_indices;
    for (auto const& event : m_events) {
        auto fail = [&](std::string const& detail) {
            std::ostringstream message;
            message << "Invalid retained RREA seed macro_index="
                    << event.macro_index << ": " << detail;
            throw std::runtime_error(message.str());
        };
        if (event.macro_index < 0) {
            fail("macro_index must be nonnegative");
        }
        if (!macro_indices.insert(event.macro_index).second) {
            fail("macro_index must be unique within the retained case schedule");
        }
        if (event.species_id != RreaSeedSpeciesElectron
            && event.species_id != RreaSeedSpeciesPositron) {
            fail("species_id must be 1 (electron) or 2 (positron)");
        }
        if (!finite(event.time_s) || event.time_s < amrex::Real(0.0)) {
            fail("time_s must be finite and nonnegative");
        }
        if (!finite(event.weight_real_electrons)
            || !(event.weight_real_electrons > amrex::Real(0.0))) {
            fail("weight_real_electrons must be finite and positive");
        }
        if (!finite(event.x_m) || !finite(event.y_m) || !finite(event.r_m)
            || !finite(event.z_m)) {
            fail("initial position components must be finite");
        }
        amrex::Real const actual_radius = std::hypot(event.x_m, event.y_m);
        amrex::Real const radius_tolerance = amrex::Real(128.0)
            * std::numeric_limits<amrex::Real>::epsilon()
            * std::max({amrex::Real(1.0), actual_radius, std::abs(event.r_m)});
        if (std::abs(event.r_m - actual_radius) > radius_tolerance) {
            fail("r_m is inconsistent with hypot(x_m,y_m)");
        }
        if (actual_radius < radial_min_m || !(actual_radius < radial_max_m)
            || event.z_m < axial_min_m || !(event.z_m < axial_max_m)) {
            fail("initial position is outside the half-open physical RZ domain");
        }
        if (!finite(event.kinetic_energy_eV)
            || event.kinetic_energy_eV < electron_min_energy_eV
            || event.kinetic_energy_eV > electron_max_energy_eV) {
            fail("declared kinetic energy is outside the certified inclusive charged range");
        }
        if (!finite(event.ux) || !finite(event.uy) || !finite(event.uz)) {
            fail("direction/proper-velocity components must be finite");
        }
        amrex::Real const u_norm = std::hypot(std::hypot(event.ux, event.uy), event.uz);
        if (!finite(u_norm) || !(u_norm > amrex::Real(0.0))) {
            fail("direction/proper-velocity vector must be nonzero and finite");
        }
        amrex::Real actual_energy_eV = event.kinetic_energy_eV;
        if (!(u_norm < direction_hint_max_m_per_s)) {
            amrex::Real const q = u_norm / c_m_per_s;
            amrex::Real const gamma = std::hypot(amrex::Real(1.0), q);
            actual_energy_eV = rest_energy_eV * q * (q / (gamma + amrex::Real(1.0)));
        }
        if (!finite(actual_energy_eV)
            || actual_energy_eV < electron_min_energy_eV
            || actual_energy_eV > electron_max_energy_eV) {
            fail("effective electron kinetic energy is outside the certified inclusive range");
        }
        amrex::Real const energy_tolerance = amrex::Real(256.0)
            * std::numeric_limits<amrex::Real>::epsilon()
            * std::max({amrex::Real(1.0), actual_energy_eV, event.kinetic_energy_eV});
        if (std::abs(actual_energy_eV - event.kinetic_energy_eV) > energy_tolerance) {
            fail("proper-velocity kinetic energy disagrees with kinetic_energy_eV");
        }
    }
}

bool RreaSeedSchedule::Empty() const noexcept
{
    return m_events.empty();
}

std::size_t RreaSeedSchedule::Size() const noexcept
{
    return m_events.size();
}

std::string const& RreaSeedSchedule::Path() const noexcept
{
    return m_path;
}

RreaSeedEvent const& RreaSeedSchedule::Event(std::size_t index) const
{
    AMREX_ALWAYS_ASSERT(index < m_events.size());
    return m_events[index];
}

}  // namespace rrea
