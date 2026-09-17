#include "rrea/RreaAltitudeProfile.H"
#include "rrea/RreaCsvTextUtil.H"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>

namespace rrea {

namespace {

using rrea::csv_text::split_csv_line;
using rrea::csv_text::trim;

amrex::Real parse_real(std::string const& text, std::string const& column)
{
    std::istringstream stream(text);
    amrex::Real value = amrex::Real(0.0);
    stream >> value;
    if (!stream || !stream.eof() || !std::isfinite(static_cast<double>(value))) {
        throw std::runtime_error("Invalid numeric altitude profile value in column "
            + column + ": " + text);
    }
    return value;
}

}  // namespace

void RreaAltitudeProfile::LoadCsv(
    std::string const& path,
    std::string const& altitude_column,
    std::string const& value_column,
    amrex::Real altitude_scale,
    amrex::Real value_scale,
    bool require_positive)
{
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Could not open RREA altitude profile: " + path);
    }
    if (altitude_scale <= amrex::Real(0.0) || value_scale == amrex::Real(0.0)) {
        throw std::runtime_error("Invalid scale for RREA altitude profile: " + path);
    }

    std::string header_line;
    if (!std::getline(input, header_line)) {
        throw std::runtime_error("Empty RREA altitude profile: " + path);
    }
    auto const headers = split_csv_line(header_line);
    std::map<std::string, int> columns;
    for (std::size_t i = 0; i < headers.size(); ++i) {
        auto const [_, inserted] = columns.emplace(headers[i], static_cast<int>(i));
        if (!inserted) {
            throw std::runtime_error("Duplicate RREA altitude profile column: " + headers[i]);
        }
    }
    auto const altitude_it = columns.find(altitude_column);
    if (altitude_it == columns.end()) {
        throw std::runtime_error("Missing RREA altitude profile column: " + altitude_column);
    }
    auto const value_it = columns.find(value_column);
    if (value_it == columns.end()) {
        throw std::runtime_error("Missing RREA altitude profile column: " + value_column);
    }

    amrex::Vector<amrex::Real> altitudes;
    amrex::Vector<amrex::Real> values;
    std::string line;
    while (std::getline(input, line)) {
        if (trim(line).empty()) {
            continue;
        }
        auto const row = split_csv_line(line);
        int const altitude_index = altitude_it->second;
        int const value_index = value_it->second;
        if (altitude_index >= static_cast<int>(row.size())
            || value_index >= static_cast<int>(row.size())) {
            throw std::runtime_error("Short row in RREA altitude profile: " + path);
        }
        altitudes.push_back(parse_real(row[altitude_index], altitude_column) * altitude_scale);
        values.push_back(parse_real(row[value_index], value_column) * value_scale);
    }

    if (altitudes.size() < 2) {
        throw std::runtime_error("RREA altitude profile needs at least two rows: " + path);
    }
    amrex::Real max_abs_value = amrex::Real(0.0);
    for (std::size_t i = 0; i < altitudes.size(); ++i) {
        if (i > 0 && altitudes[i] <= altitudes[i - 1]) {
            throw std::runtime_error(
                "RREA altitude profile altitudes must be strictly increasing: " + path);
        }
        // Fail on nonpositive density: the not-identically-zero check below
        // uses std::abs, so an all-negative
        // or mixed-sign column would otherwise pass and be silently masked to
        // a positive scalar in the fluid/optical-depth consumers.
        if (require_positive && !(values[i] > amrex::Real(0.0))) {
            throw std::runtime_error(
                "RREA altitude profile requires strictly positive values (column "
                + value_column + "): " + path);
        }
        max_abs_value = std::max(max_abs_value, std::abs(values[i]));
    }
    if (max_abs_value <= amrex::Real(0.0)) {
        throw std::runtime_error("RREA altitude profile has zero value everywhere: " + path);
    }

    m_path = path;
    m_altitude_column = altitude_column;
    m_value_column = value_column;
    m_altitude_m.assign(altitudes.begin(), altitudes.end());
    m_value.assign(values.begin(), values.end());
    m_max_abs_value = max_abs_value;
}

bool RreaAltitudeProfile::Loaded() const noexcept
{
    return !m_altitude_m.empty();
}

amrex::Real RreaAltitudeProfile::Interpolate(amrex::Real altitude_m) const
{
    if (!Loaded()) {
        throw std::runtime_error("RREA altitude profile was not loaded");
    }
    if (altitude_m < m_altitude_m.front() || altitude_m > m_altitude_m.back()) {
        std::ostringstream message;
        message << "RREA altitude " << altitude_m
                << " m is outside profile range ["
                << m_altitude_m.front() << ", " << m_altitude_m.back() << "] m";
        throw std::runtime_error(message.str());
    }
    return View().Interpolate(altitude_m);
}

amrex::Real RreaAltitudeProfile::MaxAbsValue() const noexcept
{
    return m_max_abs_value;
}

amrex::Real RreaAltitudeProfile::MinAltitudeM() const noexcept
{
    return Loaded() ? m_altitude_m.front() : amrex::Real(0.0);
}

amrex::Real RreaAltitudeProfile::MaxAltitudeM() const noexcept
{
    return Loaded() ? m_altitude_m.back() : amrex::Real(0.0);
}

std::string const& RreaAltitudeProfile::Path() const noexcept
{
    return m_path;
}

std::string const& RreaAltitudeProfile::AltitudeColumn() const noexcept
{
    return m_altitude_column;
}

std::string const& RreaAltitudeProfile::ValueColumn() const noexcept
{
    return m_value_column;
}

}  // namespace rrea
