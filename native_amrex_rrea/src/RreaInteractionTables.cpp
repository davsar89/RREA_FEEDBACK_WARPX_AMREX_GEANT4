#include "rrea/RreaInteractionTables.H"
#include "rrea/RreaCsvTextUtil.H"
#include "rrea/RreaTableContainer.H"
#include "rrea/RreaConstants.H"
#include "rrea/RreaUniformStream.H"
#include "rrea/RreaDebugOptions.H"

#include <AMReX.H>
#include <AMReX_BLassert.H>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>
#include <filesystem>

namespace rrea {


namespace {

constexpr amrex::Real kElectronRestEnergyEv = amrex::Real(rrea::me_c2_eV);
constexpr amrex::Real kPairNuclearThresholdEv = amrex::Real(2.0) * kElectronRestEnergyEv;
constexpr amrex::Real kPairTripletThresholdEv = amrex::Real(4.0) * kElectronRestEnergyEv;
constexpr amrex::Real kTwoPi = amrex::Real(6.283185307179586476925286766559);
constexpr char kOption4GsBranch[] = "goudsmit_saunderson_option4";
constexpr char kOption4GsModel[] = "G4GoudsmitSaundersonMscModel";
constexpr char kOption4WentzelBranch[] = "wentzel_vi_option4";
constexpr char kOption4WentzelModel[] = "G4WentzelVIModel";

amrex::Real clamp_unit(amrex::Real value)
{
    if (!std::isfinite(static_cast<double>(value))) {
        return amrex::Real(0.5);
    }
    return std::max(amrex::Real(0.0), std::min(amrex::Real(1.0), value));
}

amrex::Real wrap_unit_interval(amrex::Real value)
{
    if (!std::isfinite(static_cast<double>(value))) {
        return amrex::Real(0.0);
    }
    value = value - std::floor(value);
    if (value < amrex::Real(0.0)) {
        value += amrex::Real(1.0);
    }
    return value;
}

amrex::Real sample_annihilation_photon_energy_fraction(
    amrex::Real gamma,
    amrex::Real draw_energy)
{
    // Exact Heitler spectrum. The support uses the CM beta; using the lab beta
    // admits kinematically impossible draws. The nonconstant rejection kernel
    // is required in addition to the 1/eps proposal:
    //   f(eps) ~ (1/eps) * g(eps),  g(eps) = 1 + 2g/(g+1)^2 - eps
    //                                        - 1/((g+1)^2 eps)
    // over the physical support eps in [.5(1-b_cm), .5(1+b_cm)],
    // b_cm = sqrt((g-1)/(g+1)), by deterministic bisection on the analytic
    // antiderivative F(eps) = a ln eps - eps + 1/((g+1)^2 eps),
    // a = 1 + 2g/(g+1)^2.  The inversion consumes one uniform, no rejection.
    if (gamma <= amrex::Real(1.0 + 1.0e-12)) {
        return amrex::Real(0.5);
    }
    amrex::Real const beta_cm = std::sqrt(std::max(
        amrex::Real(0.0),
        (gamma - amrex::Real(1.0)) / (gamma + amrex::Real(1.0))));
    amrex::Real const eps_min = amrex::Real(0.5) * (amrex::Real(1.0) - beta_cm);
    amrex::Real const eps_max = amrex::Real(0.5) * (amrex::Real(1.0) + beta_cm);
    amrex::Real const u = clamp_unit(draw_energy);
    amrex::Real const gp1 = gamma + amrex::Real(1.0);
    amrex::Real const a = amrex::Real(1.0) + amrex::Real(2.0) * gamma / (gp1 * gp1);
    auto antiderivative = [a, gp1](amrex::Real e) {
        return a * std::log(e) - e + amrex::Real(1.0) / (gp1 * gp1 * e);
    };
    amrex::Real const f_lo = antiderivative(eps_min);
    amrex::Real const f_hi = antiderivative(eps_max);
    amrex::Real const span = f_hi - f_lo;
    if (!(span > amrex::Real(0.0))) {
        return amrex::Real(0.5);
    }
    amrex::Real const target = f_lo + u * span;
    amrex::Real lo = eps_min;
    amrex::Real hi = eps_max;
    for (int iter = 0; iter < 64; ++iter) {
        amrex::Real const mid = amrex::Real(0.5) * (lo + hi);
        if (antiderivative(mid) < target) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return std::max(eps_min, std::min(eps_max, amrex::Real(0.5) * (lo + hi)));
}

amrex::Real mean_free_path_from_inverse_length(amrex::Real inverse_length_per_m)
{
    if (!std::isfinite(static_cast<double>(inverse_length_per_m))
        || inverse_length_per_m < amrex::Real(0.0)) {
        amrex::Abort("RREA process inverse length must be finite and nonnegative");
    }
    return inverse_length_per_m > amrex::Real(0.0)
        ? amrex::Real(1.0) / inverse_length_per_m
        : amrex::Real(1.0e99);
}

void normalize_local_direction(amrex::Real& x, amrex::Real& y, amrex::Real& z)
{
    amrex::Real const norm = std::sqrt(x * x + y * y + z * z);
    if (norm <= amrex::Real(0.0) || !std::isfinite(static_cast<double>(norm))) {
        x = amrex::Real(0.0);
        y = amrex::Real(0.0);
        z = amrex::Real(1.0);
        return;
    }
    x /= norm;
    y /= norm;
    z /= norm;
}

using rrea::csv_text::split_csv_line;
using rrea::csv_text::trim;

template <typename T>
T parse_number(std::string const& text, std::string const& column)
{
    std::istringstream stream(text);
    T value{};
    stream >> value;
    if (!stream || !stream.eof()) {
        throw std::runtime_error("Invalid numeric value in " + column + ": " + text);
    }
    return value;
}

std::string file_text(std::string const& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Could not open file: " + path);
    }
    return std::string(
        (std::istreambuf_iterator<char>(input)),
        std::istreambuf_iterator<char>());
}

std::string json_string(std::string const& object, std::string const& key)
{
    std::string const quoted_key = "\"" + key + "\"";
    std::size_t pos = object.find(quoted_key);
    if (pos == std::string::npos) {
        throw std::runtime_error("Missing transport configuration string key: " + key);
    }
    pos = object.find(':', pos + quoted_key.size());
    if (pos == std::string::npos) {
        throw std::runtime_error("Malformed transport configuration string key: " + key);
    }
    pos = object.find('"', pos);
    if (pos == std::string::npos) {
        throw std::runtime_error("Malformed transport configuration string value: " + key);
    }
    ++pos;
    std::string value;
    bool escaped = false;
    for (; pos < object.size(); ++pos) {
        char const c = object[pos];
        if (escaped) {
            value.push_back(c);
            escaped = false;
        } else if (c == '\\') {
            escaped = true;
        } else if (c == '"') {
            return value;
        } else {
            value.push_back(c);
        }
    }
    throw std::runtime_error("Unterminated transport configuration string value: " + key);
}

std::string json_string_or_empty(std::string const& object, std::string const& key)
{
    try {
        return json_string(object, key);
    } catch (std::runtime_error const&) {
        return {};
    }
}

std::string json_string_or_number(std::string const& object, std::string const& key)
{
    std::string const quoted_key = "\"" + key + "\"";
    std::size_t pos = object.find(quoted_key);
    if (pos == std::string::npos) {
        throw std::runtime_error("Missing transport configuration key: " + key);
    }
    pos = object.find(':', pos + quoted_key.size());
    if (pos == std::string::npos) {
        throw std::runtime_error("Malformed transport configuration key: " + key);
    }
    ++pos;
    while (pos < object.size() && std::isspace(static_cast<unsigned char>(object[pos])) != 0) {
        ++pos;
    }
    if (pos < object.size() && object[pos] == '"') {
        --pos;
        return json_string(object, key);
    }
    std::size_t end = pos;
    while (end < object.size()) {
        char const c = object[end];
        if ((c >= '0' && c <= '9') || c == '-' || c == '+'
            || c == '.' || c == 'e' || c == 'E') {
            ++end;
        } else {
            break;
        }
    }
    if (end == pos) {
        throw std::runtime_error("Malformed transport configuration numeric value: " + key);
    }
    return trim(object.substr(pos, end - pos));
}


bool json_bool_or_default(
    std::string const& object,
    std::string const& key,
    bool default_value)
{
    std::string const quoted_key = "\"" + key + "\"";
    std::size_t pos = object.find(quoted_key);
    if (pos == std::string::npos) {
        return default_value;
    }
    pos = object.find(':', pos + quoted_key.size());
    if (pos == std::string::npos) {
        throw std::runtime_error("Malformed transport configuration boolean key: " + key);
    }
    ++pos;
    while (pos < object.size() && std::isspace(static_cast<unsigned char>(object[pos])) != 0) {
        ++pos;
    }
    if (object.compare(pos, 4, "true") == 0) {
        return true;
    }
    if (object.compare(pos, 5, "false") == 0) {
        return false;
    }
    throw std::runtime_error("Malformed transport configuration boolean value: " + key);
}

amrex::Real json_number_or_default(
    std::string const& object,
    std::string const& key,
    amrex::Real default_value)
{
    std::string const quoted_key = "\"" + key + "\"";
    std::size_t pos = object.find(quoted_key);
    if (pos == std::string::npos) {
        return default_value;
    }
    pos = object.find(':', pos + quoted_key.size());
    if (pos == std::string::npos) {
        throw std::runtime_error("Malformed transport configuration numeric key: " + key);
    }
    ++pos;
    while (pos < object.size() && std::isspace(static_cast<unsigned char>(object[pos])) != 0) {
        ++pos;
    }
    std::size_t end = pos;
    while (end < object.size()) {
        char const c = object[end];
        if ((c >= '0' && c <= '9') || c == '-' || c == '+' || c == '.'
            || c == 'e' || c == 'E') {
            ++end;
        } else {
            break;
        }
    }
    if (end == pos) {
        throw std::runtime_error("Malformed transport configuration numeric value: " + key);
    }
    return parse_number<amrex::Real>(object.substr(pos, end - pos), key);
}

std::array<amrex::Real, 2> json_number_pair(
    std::string const& object,
    std::string const& key)
{
    std::string const quoted_key = "\"" + key + "\"";
    std::size_t pos = object.find(quoted_key);
    if (pos == std::string::npos) {
        throw std::runtime_error("Missing transport configuration numeric-pair key: " + key);
    }
    pos = object.find(':', pos + quoted_key.size());
    if (pos == std::string::npos) {
        throw std::runtime_error("Malformed transport configuration numeric-pair key: " + key);
    }
    std::size_t const begin = object.find('[', pos);
    std::size_t const end = begin == std::string::npos
        ? std::string::npos
        : object.find(']', begin + 1);
    if (begin == std::string::npos || end == std::string::npos) {
        throw std::runtime_error("Malformed transport configuration numeric-pair value: " + key);
    }
    auto const cells = split_csv_line(object.substr(begin + 1, end - begin - 1));
    if (cells.size() != 2U) {
        throw std::runtime_error("Transport configuration numeric-pair must have exactly two values: " + key);
    }
    std::array<amrex::Real, 2> result{
        parse_number<amrex::Real>(cells[0], key),
        parse_number<amrex::Real>(cells[1], key)};
    if (!std::isfinite(static_cast<double>(result[0]))
        || !std::isfinite(static_cast<double>(result[1]))
        || !(result[0] > amrex::Real(0.0))
        || !(result[1] > result[0])) {
        throw std::runtime_error(
            "Transport configuration numeric-pair must be finite, positive, and increasing: " + key);
    }
    return result;
}

std::string json_object_or_empty(std::string const& object, std::string const& key)
{
    std::string const quoted_key = "\"" + key + "\"";
    std::size_t pos = object.find(quoted_key);
    if (pos == std::string::npos) {
        return {};
    }
    pos = object.find(':', pos + quoted_key.size());
    if (pos == std::string::npos) {
        throw std::runtime_error("Malformed transport configuration object key: " + key);
    }
    pos = object.find('{', pos);
    if (pos == std::string::npos) {
        throw std::runtime_error("Malformed transport configuration object value: " + key);
    }

    int depth = 0;
    bool in_quotes = false;
    bool escaped = false;
    for (std::size_t i = pos; i < object.size(); ++i) {
        char const c = object[i];
        if (escaped) {
            escaped = false;
            continue;
        }
        if (c == '\\' && in_quotes) {
            escaped = true;
            continue;
        }
        if (c == '"') {
            in_quotes = !in_quotes;
            continue;
        }
        if (in_quotes) {
            continue;
        }
        if (c == '{') {
            ++depth;
        } else if (c == '}') {
            --depth;
            if (depth == 0) {
                return object.substr(pos, i - pos + 1);
            }
        }
        if (depth < 0) {
            throw std::runtime_error("Malformed transport configuration object braces: " + key);
        }
    }
    throw std::runtime_error("Unterminated transport configuration object value: " + key);
}

amrex::Vector<std::string> json_array_objects(
    std::string const& text, std::string const& key, bool require_nonempty = true)
{
    auto const tables_pos = text.find("\"" + key + "\"");
    if (tables_pos == std::string::npos) {
        throw std::runtime_error("JSON object missing " + key + " array");
    }
    auto const array_begin = text.find('[', tables_pos);
    if (array_begin == std::string::npos) {
        throw std::runtime_error("JSON object has malformed " + key + " array");
    }
    std::size_t array_end = std::string::npos;
    int bracket_depth = 0;
    bool in_quotes = false;
    bool escaped = false;
    for (std::size_t i = array_begin; i < text.size(); ++i) {
        char const c = text[i];
        if (escaped) {
            escaped = false;
            continue;
        }
        if (c == '\\' && in_quotes) {
            escaped = true;
            continue;
        }
        if (c == '"') {
            in_quotes = !in_quotes;
            continue;
        }
        if (in_quotes) {
            continue;
        }
        if (c == '[') {
            ++bracket_depth;
        } else if (c == ']') {
            --bracket_depth;
            if (bracket_depth == 0) {
                array_end = i;
                break;
            }
        }
    }
    if (array_end == std::string::npos) {
        throw std::runtime_error("JSON object has malformed " + key + " array");
    }
    std::string const array_text = text.substr(array_begin + 1, array_end - array_begin - 1);
    amrex::Vector<std::string> objects;
    int depth = 0;
    std::size_t start = std::string::npos;
    in_quotes = false;
    escaped = false;
    for (std::size_t i = 0; i < array_text.size(); ++i) {
        char const c = array_text[i];
        if (escaped) {
            escaped = false;
            continue;
        }
        if (c == '\\' && in_quotes) {
            escaped = true;
            continue;
        }
        if (c == '"') {
            in_quotes = !in_quotes;
            continue;
        }
        if (in_quotes) {
            continue;
        }
        if (c == '{') {
            if (depth == 0) {
                start = i;
            }
            ++depth;
        } else if (c == '}') {
            --depth;
            if (depth == 0 && start != std::string::npos) {
                objects.push_back(array_text.substr(start, i - start + 1));
                start = std::string::npos;
            }
        }
        if (depth < 0) {
            throw std::runtime_error("JSON object has malformed " + key + " object braces");
        }
    }
    if (depth != 0 || (require_nonempty && objects.empty())) {
        throw std::runtime_error("JSON object has no complete " + key + " objects");
    }
    return objects;
}

amrex::Vector<std::string> table_objects(std::string const& text)
{
    return json_array_objects(text, "tables");
}

RreaInteractionTable load_table(
    std::filesystem::path const& config_dir,
    std::string const& object,
    std::string const& x_column = "energy_eV")
{
    std::string actual_x_column = json_string_or_empty(object, "x_column");
    if (actual_x_column.empty()) {
        actual_x_column = x_column;
    }
    RreaInteractionTable table;
    table.name = json_string(object, "name");
    table.path = (config_dir / json_string(object, "path")).string();
    table.value_column = json_string_or_empty(object, "value_column");
    if (table.value_column.empty()) {
        table.value_column = json_string(object, "primary_value_column");
    }
    table.units = json_string(object, "units");
    table.interpolation = json_string_or_empty(object, "interpolation");
    if (table.interpolation.empty()) {
        table.interpolation = "loglog_linear";
    }
    table.extrapolation = json_string_or_empty(object, "extrapolation");
    if (table.extrapolation.empty()) {
        table.extrapolation = "forbidden";
    }
    if (table.extrapolation != "forbidden") {
        throw std::runtime_error(
            "Interaction table extrapolation must be 'forbidden': " + table.name);
    }
    bool const allow_duplicate_energy =
        table.name == "photon_interaction_mfp_air"
        || json_bool_or_default(object, "allow_duplicate_x", false);
    if (json_bool_or_default(object, "allow_duplicate_x", false)
        && json_string_or_empty(object, "duplicate_x_exact_policy")
            != "low_side_at_exact_edge_high_side_immediately_above") {
        throw std::runtime_error(
            "Duplicate-edge table must declare the exact-query one-sided policy: "
            + table.name);
    }

    auto const container = rrea::table_container::Read(table.path);
    table.energy_eV = rrea::table_container::Column(container, actual_x_column);
    table.values = rrea::table_container::Column(container, table.value_column);
    if (table.energy_eV.size() != table.values.size()) {
        throw std::runtime_error("Interaction table columns differ in length: " + table.path);
    }
    if (table.energy_eV.size() < 2) {
        throw std::runtime_error("Interaction table needs at least two rows: " + table.path);
    }
    for (std::size_t i = 0; i < table.energy_eV.size(); ++i) {
        if (!std::isfinite(static_cast<double>(table.energy_eV[i]))
            || !std::isfinite(static_cast<double>(table.values[i]))
            || table.energy_eV[i] <= amrex::Real(0.0)
            || table.values[i] < amrex::Real(0.0)
            || (table.interpolation == "loglog_linear"
                && table.values[i] <= amrex::Real(0.0))) {
            throw std::runtime_error(
                "Interaction table energies must be positive and values finite/nonnegative "
                "(strictly positive for loglog_linear): " + table.path);
        }
        if (i > 0) {
            bool const nondecreasing = table.energy_eV[i] >= table.energy_eV[i - 1];
            bool const strictly_increasing = table.energy_eV[i] > table.energy_eV[i - 1];
            if (!nondecreasing || (!strictly_increasing && !allow_duplicate_energy)) {
                throw std::runtime_error(
                    allow_duplicate_energy
                    ? "Interaction table energy grid must be nondecreasing: " + table.path
                    : "Interaction table energy grid must be strictly increasing: " + table.path);
            }
        }
    }
    table.Finalize();
    return table;
}

amrex::Vector<RreaConditionalCdfGroup> load_conditional_cdf(
    std::filesystem::path const& config_dir,
    std::string const& object,
    std::string const& primary_column,
    std::string const& value_column)
{
    std::string const table_name = json_string(object, "name");
    std::filesystem::path const path = config_dir / json_string(object, "path");
    auto const container = rrea::table_container::Read(path.string());
    auto const primaries = rrea::table_container::Column(container, primary_column);
    auto const cdfs = rrea::table_container::Column(container, "cdf_u");
    auto const values = rrea::table_container::Column(container, value_column);
    if (primaries.size() != cdfs.size() || primaries.size() != values.size()) {
        throw std::runtime_error(
            "Conditional CDF columns differ in length: " + path.string());
    }

    amrex::Vector<RreaConditionalCdfGroup> groups;
    for (std::size_t row_index = 0; row_index < primaries.size(); ++row_index) {
        amrex::Real const primary = primaries[row_index];
        amrex::Real const cdf = cdfs[row_index];
        amrex::Real const value = values[row_index];
        if (!std::isfinite(static_cast<double>(primary))
            || !(primary > amrex::Real(0.0))) {
            throw std::runtime_error(
                "Conditional CDF primary energies must be finite and positive: "
                + path.string());
        }
        if (!std::isfinite(static_cast<double>(cdf))
            || cdf < amrex::Real(0.0) || cdf > amrex::Real(1.0)) {
            throw std::runtime_error(
                "Conditional CDF probability nodes must be finite and in [0,1]: "
                + path.string());
        }
        if (!std::isfinite(static_cast<double>(value))
            || value < amrex::Real(0.0)) {
            throw std::runtime_error(
                "Conditional inverse-CDF sampled values must be finite and nonnegative: "
                + path.string());
        }
        if (groups.empty() || groups.back().primary_energy_eV != primary) {
            if (!groups.empty() && primary <= groups.back().primary_energy_eV) {
                throw std::runtime_error(
                    "Conditional CDF primary-energy groups must be strictly increasing: "
                    + path.string());
            }
            groups.push_back(RreaConditionalCdfGroup{});
            groups.back().primary_energy_eV = primary;
        }
        auto& group = groups.back();
        if (!group.cdf_u.empty() && cdf <= group.cdf_u.back()) {
            throw std::runtime_error(
                "Conditional CDF grid must be strictly increasing within a primary group: "
                + path.string());
        }
        group.cdf_u.push_back(cdf);
        group.values.push_back(value);
    }
    ValidateRreaConditionalCdfGroups(groups, path.string());
    return groups;
}

// --- GS / Wentzel elastic table loaders and samplers -------------------------

// Strict semi-log-x interpolation for the elastic MFP tables.
amrex::Real interp_semilogx_strict(
    amrex::Vector<amrex::Real> const& x,
    amrex::Vector<amrex::Real> const& y,
    amrex::Real q)
{
    if (x.size() < 2) {
        return y.empty() ? amrex::Real(0.0) : y.front();
    }
    if (!std::isfinite(static_cast<double>(q)) || q < x.front() || q > x.back()) {
        amrex::Abort("RREA elastic-table query is outside its inclusive energy range");
    }
    if (q == x.front()) return y.front();
    if (q == x.back()) return y.back();
    auto const upper = std::lower_bound(x.begin(), x.end(), q);
    std::size_t const hi = static_cast<std::size_t>(std::distance(x.begin(), upper));
    std::size_t const lo = hi - 1U;
    // Node-exact fast path (every other interpolator here has one).  Without
    // it a query at the nextafter side of a sub-ulp anchor pair (the exact
    // 1e8 GS/Wentzel transition rows in the elastic tables) divides by a
    // zero log-segment width because log(1e8) and log(nextafter(1e8)) are the
    // same double.
    if (x[hi] == q) return y[hi];
    amrex::Real const t = (std::log(q) - std::log(x[lo])) / (std::log(x[hi]) - std::log(x[lo]));
    return y[lo] + t * (y[hi] - y[lo]);
}

RreaElasticMfpTable load_elastic_mfp_table(
    std::filesystem::path const& config_dir, std::string const& object)
{
    std::string const table_name = json_string(object, "name");
    std::filesystem::path const path = config_dir / json_string(object, "path");
    auto const container = rrea::table_container::Read(path.string());
    RreaElasticMfpTable tab;
    tab.energy_eV = rrea::table_container::Column(container, "primary_energy_eV");
    tab.elastic_mfp_m_stp = rrea::table_container::Column(container, "elastic_mfp_m_stp");
    tab.screening_eta = rrea::table_container::Column(container, "screening_eta");
    tab.transport_mfp_m_stp = rrea::table_container::Column(container, "transport_mfp_m_stp");
    if (tab.elastic_mfp_m_stp.size() != tab.energy_eV.size()
        || tab.screening_eta.size() != tab.energy_eV.size()
        || tab.transport_mfp_m_stp.size() != tab.energy_eV.size()) {
        throw std::runtime_error("Elastic MFP columns differ in length: " + path.string());
    }
    if (tab.energy_eV.size() < 2) {
        throw std::runtime_error("Elastic MFP table needs at least two rows: " + path.string());
    }
    for (std::size_t i = 0; i < tab.energy_eV.size(); ++i) {
        if (!std::isfinite(static_cast<double>(tab.energy_eV[i]))
            || !std::isfinite(static_cast<double>(tab.elastic_mfp_m_stp[i]))
            || !std::isfinite(static_cast<double>(tab.screening_eta[i]))
            || !std::isfinite(static_cast<double>(tab.transport_mfp_m_stp[i]))
            || !(tab.energy_eV[i] > amrex::Real(0.0))
            || !(tab.elastic_mfp_m_stp[i] > amrex::Real(0.0))
            || !(tab.screening_eta[i] > amrex::Real(0.0))
            || !(tab.transport_mfp_m_stp[i] > amrex::Real(0.0))
            || (i > 0 && !(tab.energy_eV[i] > tab.energy_eV[i - 1]))) {
            throw std::runtime_error(
                "Elastic parameter table values must be finite/positive with "
                "strictly increasing energy: " + path.string());
        }
    }
    return tab;
}

RreaGsAngleTable load_gs_angle_table(
    std::filesystem::path const& config_dir, std::string const& object)
{
    std::string const table_name = json_string(object, "name");
    std::filesystem::path const path = config_dir / json_string(object, "path");
    auto const container = rrea::table_container::Read(path.string());
    auto const energies = rrea::table_container::Column(container, "primary_energy_eV");
    auto const paths = rrea::table_container::Column(container, "path_over_elastic_mfp");
    auto const us = rrea::table_container::Column(container, "cdf");
    auto const xs = rrea::table_container::Column(container, "one_minus_cos_theta");
    if (paths.size() != energies.size() || us.size() != energies.size()
        || xs.size() != energies.size()) {
        throw std::runtime_error(
            "Elastic angle CDF columns differ in length: " + path.string());
    }
    // energy -> path -> ordered (u, one_minus_cos)
    std::map<amrex::Real, std::map<amrex::Real, std::vector<std::pair<amrex::Real, amrex::Real>>>> by_ep;
    for (std::size_t row_index = 0; row_index < energies.size(); ++row_index) {
        by_ep[energies[row_index]][paths[row_index]].emplace_back(
            us[row_index], xs[row_index]);
    }
    if (by_ep.size() < 2) {
        throw std::runtime_error("Elastic angle CDF needs >= 2 energy groups: " + path.string());
    }
    RreaGsAngleTable tab;
    for (auto const& ep : by_ep) tab.energy_eV.push_back(ep.first);
    for (auto const& pp : by_ep.begin()->second) tab.path_n.push_back(pp.first);
    for (auto const& up : by_ep.begin()->second.begin()->second) tab.cdf_u.push_back(up.first);
    int const ne = static_cast<int>(tab.energy_eV.size());
    int const np = static_cast<int>(tab.path_n.size());
    int const nu = static_cast<int>(tab.cdf_u.size());
    if (np < 2 || nu < 2) {
        throw std::runtime_error("Elastic angle CDF needs >= 2 path and u nodes: " + path.string());
    }
    // The sampler indexes one_minus_cos positionally across u-nodes, so every
    // (E,n) group MUST share the same, strictly-increasing u grid; a group
    // with the right COUNT but different u-values would silently mis-map.
    for (int iu = 1; iu < nu; ++iu) {
        if (!(tab.cdf_u[iu] > tab.cdf_u[iu - 1])) {
            throw std::runtime_error("Elastic angle CDF u-nodes not strictly increasing: " + path.string());
        }
    }
    tab.one_minus_cos.resize(static_cast<std::size_t>(ne) * np * nu, amrex::Real(1.0));
    for (int ie = 0; ie < ne; ++ie) {
        auto const& pm = by_ep.at(tab.energy_eV[ie]);
        if (static_cast<int>(pm.size()) != np) {
            throw std::runtime_error("Elastic angle CDF path count varies across energy: " + path.string());
        }
        for (int ip = 0; ip < np; ++ip) {
            auto const& v = pm.at(tab.path_n[ip]);
            if (static_cast<int>(v.size()) != nu) {
                throw std::runtime_error("Elastic angle CDF node count varies across group: " + path.string());
            }
            std::size_t const base = (static_cast<std::size_t>(ie) * np + ip) * nu;
            for (int iu = 0; iu < nu; ++iu) {
                if (v[iu].first != tab.cdf_u[iu]) {
                    throw std::runtime_error("Elastic angle CDF u-nodes differ across (E,n) groups: "
                        + path.string());
                }
                amrex::Real const x = v[iu].second;
                if (!std::isfinite(static_cast<double>(x))
                    || !(x > amrex::Real(0.0))
                    || x > amrex::Real(2.0)) {
                    throw std::runtime_error(
                        "Elastic angle CDF one_minus_cos_theta must be finite in (0,2]: "
                        + path.string());
                }
                if (iu > 0 && x > v[iu - 1].second) {
                    throw std::runtime_error(
                        "Elastic angle inverse CDF is not monotone in cdf: "
                        + path.string());
                }
                tab.one_minus_cos[base + iu] = x;
            }
        }
    }
    if (!(tab.energy_eV.front() > amrex::Real(0.0))
        || !(tab.path_n.front() > amrex::Real(0.0))
        || tab.cdf_u.front() < amrex::Real(0.0)
        || tab.cdf_u.back() > amrex::Real(1.0)) {
        throw std::runtime_error(
            "Elastic angle CDF grids must use positive energy/path and cdf in [0,1]: "
            + path.string());
    }
    return tab;
}

void require_scattering_table_contract(
    std::string const& object,
    char const* expected_branch,
    char const* expected_model,
    amrex::Real expected_min_eV,
    amrex::Real expected_max_eV)
{
    std::string const name = json_string(object, "name");
    if (json_string_or_empty(object, "scattering_branch") != expected_branch) {
        throw std::runtime_error(
            "schema-6 elastic angle table has wrong or missing scattering_branch: " + name);
    }
    if (json_string_or_empty(object, "model") != expected_model) {
        throw std::runtime_error(
            "schema-6 elastic angle table has wrong or missing model: " + name);
    }
    if (json_string_or_empty(object, "interpolation")
        != "log_energy_log_path_log_one_minus_cos_bilinear_then_cdf_linear_v1") {
        throw std::runtime_error(
            "schema-6 elastic angle table has wrong or missing interpolation: " + name);
    }
    if (json_string_or_empty(object, "extrapolation") != "forbidden") {
        throw std::runtime_error(
            "schema-6 elastic angle table must forbid extrapolation: " + name);
    }
    auto const range = json_number_pair(object, "active_energy_range_eV");
    if (range[0] != expected_min_eV || range[1] != expected_max_eV) {
        throw std::runtime_error(
            "schema-6 elastic angle table has an invalid active_energy_range_eV: " + name);
    }
}

// Bracket q in a sorted grid; returns lower index and log-fraction to next node.
void log_bracket(amrex::Vector<amrex::Real> const& g, amrex::Real q, int& i, amrex::Real& frac)
{
    int const last = static_cast<int>(g.size()) - 1;
    if (!std::isfinite(static_cast<double>(q)) || q < g[0] || q > g[last]) {
        amrex::Abort("RREA elastic-angle query is outside its inclusive tabulated range");
    }
    if (q == g[0]) { i = 0; frac = amrex::Real(0.0); return; }
    if (q == g[last]) { i = last - 1; frac = amrex::Real(1.0); return; }
    int lo = 0, hi = last;
    while (hi - lo > 1) { int const m = (lo + hi) / 2; if (g[m] <= q) lo = m; else hi = m; }
    i = lo;
    amrex::Real const log_lo = std::log(g[lo]);
    amrex::Real const log_hi = std::log(g[hi]);
    if (log_hi == log_lo) {
        // Sub-ulp bracket (duplicated-edge/nextafter anchor rows): the only
        // representable queries inside it are its endpoints.
        frac = (q == g[lo]) ? amrex::Real(0.0) : amrex::Real(1.0);
        return;
    }
    frac = (std::log(q) - log_lo) / (log_hi - log_lo);
}

// Sample cos(theta) from the 2D GS inverse CDF: bilinear in (log E, log n) of
// log(one_minus_cos) at fixed u-nodes, then linear in cos(theta) across u.
amrex::Real sample_gs_angle_cos(
    RreaGsAngleTable const& t, amrex::Real energy_eV, amrex::Real path_n, amrex::Real u)
{
    int const np = static_cast<int>(t.path_n.size());
    int const nu = static_cast<int>(t.cdf_u.size());
    int ie = 0, ip = 0;
    amrex::Real te = 0.0, tn = 0.0;
    log_bracket(t.energy_eV, energy_eV, ie, te);
    log_bracket(t.path_n, path_n, ip, tn);
    auto cos_at_node = [&](int iu) -> amrex::Real {
        auto val = [&](int e_i, int p_i) -> amrex::Real {
            return t.one_minus_cos[(static_cast<std::size_t>(e_i) * np + p_i) * nu + iu];
        };
        amrex::Real const la = std::log(val(ie, ip));
        amrex::Real const lb = std::log(val(ie + 1, ip));
        amrex::Real const lc = std::log(val(ie, ip + 1));
        amrex::Real const ld = std::log(val(ie + 1, ip + 1));
        amrex::Real const lx = (1 - te) * (1 - tn) * la + te * (1 - tn) * lb
                             + (1 - te) * tn * lc + te * tn * ld;
        return amrex::Real(1.0) - std::exp(lx);
    };
    // u nodes 0 and nu-1 are hard bounds, not samples: every (E, n) group stores
    // one_minus_cos = 2.0 at u = 0 and 1e-15 at u = 1.  The interior nodes are
    // diffusive (mean deflection scales as n^1.05 to n^1.14) and the resolved
    // tail follows one_minus_cos ~ 1/u, so interpolate log(one_minus_cos)
    // against log(u) over interior nodes only -- matching the log treatment this
    // function already uses on the E and n axes -- and continue that power law
    // below the first interior node.  The 2.0 bound is enforced as a clamp only.
    auto one_minus_cos_at = [&](int iu) -> amrex::Real {
        return std::max(amrex::Real(1.0) - cos_at_node(iu),
                        std::numeric_limits<amrex::Real>::min());
    };
    int const first = 1;          // first interior node
    int const last = nu - 2;      // last interior node
    amrex::Real one_minus_cos;
    if (last <= first) {
        one_minus_cos = one_minus_cos_at(first);
    } else if (u <= t.cdf_u[first]) {
        // Extreme tail: continue the resolved power law rather than the bound.
        amrex::Real const d1 = one_minus_cos_at(first);
        amrex::Real const d2 = one_minus_cos_at(first + 1);
        amrex::Real const u1 = t.cdf_u[first];
        amrex::Real const u2 = t.cdf_u[first + 1];
        amrex::Real const slope = std::log(d2 / d1) / std::log(u2 / u1);
        amrex::Real const u_eff = std::max(u, u1 * amrex::Real(1.0e-6));
        one_minus_cos = d1 * std::exp(slope * std::log(u_eff / u1));
    } else if (u >= t.cdf_u[last]) {
        one_minus_cos = one_minus_cos_at(last);
    } else {
        int lo = first, hi = last;
        while (hi - lo > 1) { int const m = (lo + hi) / 2; if (t.cdf_u[m] <= u) lo = m; else hi = m; }
        amrex::Real const d0 = one_minus_cos_at(lo);
        amrex::Real const d1 = one_minus_cos_at(hi);
        amrex::Real const fu =
            std::log(u / t.cdf_u[lo]) / std::log(t.cdf_u[hi] / t.cdf_u[lo]);
        one_minus_cos = d0 * std::exp(fu * std::log(d1 / d0));
    }
    amrex::Real const cos_val = amrex::Real(1.0)
        - std::min(amrex::Real(2.0), one_minus_cos);
    return std::min(amrex::Real(1.0), std::max(amrex::Real(-1.0), cos_val));
}

amrex::Vector<RreaPhotoelectricSubshellCdfGroup> load_photoelectric_subshell_cdf(
    std::filesystem::path const& config_dir,
    std::string const& object)
{
    std::string const table_name = json_string(object, "name");
    std::filesystem::path const path = config_dir / json_string(object, "path");
    auto const container = rrea::table_container::Read(path.string());
    auto const primaries = rrea::table_container::Column(container, "primary_energy_eV");
    auto const cdfs = rrea::table_container::Column(container, "cdf_u");
    auto const probabilities =
        rrea::table_container::Column(container, "subshell_probability");
    auto const atomic_numbers = rrea::table_container::Column(container, "element_Z");
    auto const bindings = rrea::table_container::Column(container, "binding_energy_eV");
    auto const relaxations =
        rrea::table_container::Column(container, "relaxation_energy_eV");
    auto const symbols = rrea::table_container::TextColumn(container, "element_symbol");
    auto const shells = rrea::table_container::TextColumn(container, "shell_id");
    std::size_t const row_count = primaries.size();
    if (cdfs.size() != row_count || probabilities.size() != row_count
        || atomic_numbers.size() != row_count || bindings.size() != row_count
        || relaxations.size() != row_count || symbols.size() != row_count
        || shells.size() != row_count) {
        throw std::runtime_error(
            "Photoelectric subshell CDF columns differ in length: " + path.string());
    }

    amrex::Vector<RreaPhotoelectricSubshellCdfGroup> groups;
    for (std::size_t line_number = 0; line_number < row_count; ++line_number) {
        amrex::Real const primary = primaries[line_number];
        RreaPhotoelectricSubshellCdfRow entry;
        entry.cdf_u = cdfs[line_number];
        entry.subshell_probability = probabilities[line_number];
        // element_Z is 6/7/8/18, exact in a double, so the round trip is exact.
        entry.element_Z = static_cast<int>(std::lround(
            static_cast<double>(atomic_numbers[line_number])));
        entry.element_symbol = symbols[line_number];
        entry.shell_id = shells[line_number];
        entry.binding_energy_eV = bindings[line_number];
        entry.relaxation_energy_eV = relaxations[line_number];
        if (primary <= amrex::Real(0.0) || entry.element_Z <= 0
            || entry.binding_energy_eV < amrex::Real(0.0)
            || entry.relaxation_energy_eV < amrex::Real(0.0)
            || entry.relaxation_energy_eV
                   > entry.binding_energy_eV * (amrex::Real(1.0) + amrex::Real(1.0e-9))
            || entry.binding_energy_eV > primary
            || entry.cdf_u < amrex::Real(0.0) || entry.cdf_u > amrex::Real(1.0)
            || entry.subshell_probability < amrex::Real(0.0)) {
            throw std::runtime_error(
                "Invalid photoelectric subshell CDF row in " + path.string()
                + " at line " + std::to_string(line_number));
        }
        bool const cdf_reset = !groups.empty()
            && groups.back().primary_energy_eV == primary
            && !groups.back().rows.empty()
            && entry.cdf_u <= groups.back().rows.back().cdf_u;
        if (groups.empty() || groups.back().primary_energy_eV != primary || cdf_reset) {
            if (!groups.empty() && primary < groups.back().primary_energy_eV) {
                throw std::runtime_error(
                    "Photoelectric subshell primary-energy groups must be nondecreasing: "
                    + path.string());
            }
            groups.push_back(RreaPhotoelectricSubshellCdfGroup{});
            groups.back().primary_energy_eV = primary;
        }
        auto& group = groups.back();
        if (!group.rows.empty() && entry.cdf_u <= group.rows.back().cdf_u) {
            throw std::runtime_error(
                "Photoelectric subshell CDF grid must be strictly increasing within a group: "
                + path.string());
        }
        group.rows.push_back(std::move(entry));
    }
    if (groups.size() < 2) {
        throw std::runtime_error("Photoelectric subshell CDF table needs at least two primary-energy groups");
    }
    for (auto const& group : groups) {
        if (group.rows.empty() || std::abs(group.rows.back().cdf_u - amrex::Real(1.0)) > amrex::Real(1.0e-12)) {
            throw std::runtime_error(
                "Photoelectric subshell CDF group must end at cdf_u=1: " + path.string());
        }
        amrex::Real prob_sum = amrex::Real(0.0);
        amrex::Real previous_cdf = amrex::Real(0.0);
        for (auto const& row : group.rows) {
            prob_sum += row.subshell_probability;
            if (row.cdf_u < previous_cdf) {
                throw std::runtime_error("Photoelectric subshell CDF is not monotone: " + path.string());
            }
            previous_cdf = row.cdf_u;
        }
        if (std::abs(prob_sum - amrex::Real(1.0)) > amrex::Real(1.0e-10)) {
            throw std::runtime_error("Photoelectric subshell probabilities must sum to one: " + path.string());
        }
    }
    return groups;
}

// Shared inverse-CDF primitive under the brems-quantile, Moller and Bhabha
// samplers.  CONVENTION THAT MUST NOT BE CONFUSED WITH GS: here the u = 0 and
// u = 1 nodes ARE data: empirical brems extrema, or the analytic Moller/Bhabha
// kinematic limits. They are interpolated like any interior node. The GS angle table is
// the opposite (its endpoints are BOUNDS, clamped, never interpolated;
// see sample_gs_angle_cos).
amrex::Real sample_conditional_cdf(
    amrex::Vector<RreaConditionalCdfGroup> const& groups,
    amrex::Real primary_energy_eV,
    amrex::Real cdf_u,
    bool log_interp)
{
    if (groups.empty()) {
        amrex::Abort("RREA conditional CDF table is not loaded");
    }
    cdf_u = std::max(amrex::Real(0.0), std::min(amrex::Real(1.0), cdf_u));

    auto const sample_group = [cdf_u](RreaConditionalCdfGroup const& group) {
        auto const upper = std::lower_bound(group.cdf_u.begin(), group.cdf_u.end(), cdf_u);
        if (upper == group.cdf_u.begin()) {
            return group.values.front();
        }
        if (upper == group.cdf_u.end()) {
            return group.values.back();
        }
        std::size_t const hi =
            static_cast<std::size_t>(std::distance(group.cdf_u.begin(), upper));
        std::size_t const lo = hi - 1U;
        amrex::Real const denom = group.cdf_u[hi] - group.cdf_u[lo];
        amrex::Real const t = denom > amrex::Real(0.0)
            ? (cdf_u - group.cdf_u[lo]) / denom
            : amrex::Real(0.0);
        return group.values[lo] + t * (group.values[hi] - group.values[lo]);
    };

    if (!std::isfinite(static_cast<double>(primary_energy_eV))
        || primary_energy_eV < groups.front().primary_energy_eV
        || primary_energy_eV > groups.back().primary_energy_eV) {
        std::ostringstream message;
        message << "RREA conditional-CDF primary energy " << std::setprecision(17)
                << primary_energy_eV << " eV is outside the inclusive range ["
                << groups.front().primary_energy_eV << ", "
                << groups.back().primary_energy_eV
                << "] eV; endpoint clamping is forbidden";
        amrex::Abort(message.str());
    }
    if (primary_energy_eV == groups.front().primary_energy_eV) {
        return sample_group(groups.front());
    }
    if (primary_energy_eV == groups.back().primary_energy_eV) {
        return sample_group(groups.back());
    }

    auto const upper = std::lower_bound(
        groups.begin(),
        groups.end(),
        primary_energy_eV,
        [](RreaConditionalCdfGroup const& group, amrex::Real energy) {
            return group.primary_energy_eV < energy;
        });
    if (upper == groups.begin()) {
        return sample_group(*upper);
    }
    if (upper == groups.end()) {
        return sample_group(groups.back());
    }
    std::size_t const hi = static_cast<std::size_t>(std::distance(groups.begin(), upper));
    std::size_t const lo = hi - 1U;
    if (groups[hi].primary_energy_eV == primary_energy_eV) {
        return sample_group(groups[hi]);
    }
    amrex::Real const y0 = sample_group(groups[lo]);
    amrex::Real const y1 = sample_group(groups[hi]);
    amrex::Real const t =
        (std::log(primary_energy_eV) - std::log(groups[lo].primary_energy_eV))
        / (std::log(groups[hi].primary_energy_eV) - std::log(groups[lo].primary_energy_eV));
    if (log_interp && y0 > amrex::Real(0.0) && y1 > amrex::Real(0.0)) {
        return std::exp(std::log(y0) + t * (std::log(y1) - std::log(y0)));
    }
    return y0 + t * (y1 - y0);
}

amrex::Real mean_conditional_cdf(
    amrex::Vector<RreaConditionalCdfGroup> const& groups,
    amrex::Real primary_energy_eV)
{
    auto group_mean = [](RreaConditionalCdfGroup const& group) {
        amrex::Real mean = group.cdf_u.front() * group.values.front();
        for (std::size_t i = 1; i < group.cdf_u.size(); ++i) {
            mean += amrex::Real(0.5)
                * (group.values[i - 1] + group.values[i])
                * (group.cdf_u[i] - group.cdf_u[i - 1]);
        }
        mean += (amrex::Real(1.0) - group.cdf_u.back())
            * group.values.back();
        return mean;
    };
    if (groups.empty()
        || primary_energy_eV < groups.front().primary_energy_eV
        || primary_energy_eV > groups.back().primary_energy_eV) {
        amrex::Abort("RREA conditional-CDF mean query is outside its range");
    }
    auto const upper = std::lower_bound(
        groups.begin(), groups.end(), primary_energy_eV,
        [](RreaConditionalCdfGroup const& group, amrex::Real energy) {
            return group.primary_energy_eV < energy;
        });
    if (upper == groups.begin()) return group_mean(*upper);
    if (upper == groups.end()) return group_mean(groups.back());
    if (upper->primary_energy_eV == primary_energy_eV) return group_mean(*upper);
    std::size_t const hi = static_cast<std::size_t>(std::distance(groups.begin(), upper));
    std::size_t const lo = hi - 1U;
    amrex::Real const y0 = group_mean(groups[lo]);
    amrex::Real const y1 = group_mean(groups[hi]);
    amrex::Real const t =
        (std::log(primary_energy_eV) - std::log(groups[lo].primary_energy_eV))
        / (std::log(groups[hi].primary_energy_eV)
           - std::log(groups[lo].primary_energy_eV));
    return y0 > amrex::Real(0.0) && y1 > amrex::Real(0.0)
        ? std::exp(std::log(y0) + t * (std::log(y1) - std::log(y0)))
        : y0 + t * (y1 - y0);
}

RreaPhotonChannel parse_pair_channel(std::string const& value)
{
    if (value == "nuclear") {
        return RreaPhotonChannel::PairNuclear;
    }
    if (value == "triplet") {
        return RreaPhotonChannel::PairTriplet;
    }
    throw std::runtime_error("unknown pair final-state index channel: " + value);
}

void validate_pair_empirical_record_contract(
    RreaPairEmpiricalRecord const& record,
    RreaPhotonChannel expected_channel,
    amrex::Real expected_primary_energy_eV,
    std::uint32_t flags,
    std::array<std::uint32_t, 3> const& reserved_words)
{
    int const expected_count =
        expected_channel == RreaPhotonChannel::PairTriplet ? 3 : 2;
    if ((record.target_Z != 6 && record.target_Z != 7
         && record.target_Z != 8 && record.target_Z != 18)
        || (flags & 1U) == 0U || (flags & ~3U) != 0U
        || record.channel != expected_channel || record.lepton_count != expected_count
        || !std::isfinite(static_cast<double>(record.primary_energy_eV))
        || !std::isfinite(static_cast<double>(record.local_recoil_fraction))
        || std::abs(record.primary_energy_eV - expected_primary_energy_eV)
            > amrex::Real(1.0e-12) * expected_primary_energy_eV
        || record.local_recoil_fraction < amrex::Real(0.0)
        || record.local_recoil_fraction > amrex::Real(1.0)) {
        throw std::runtime_error("pair final-state record metadata is invalid");
    }

    amrex::Real fraction_sum = record.local_recoil_fraction;
    for (int slot = 0; slot < 3; ++slot) {
        auto const& particle = record.particles[slot];
        if (reserved_words[static_cast<std::size_t>(slot)] != 0U
            || !std::isfinite(static_cast<double>(particle.kinetic_fraction))
            || !std::isfinite(static_cast<double>(particle.dir_x))
            || !std::isfinite(static_cast<double>(particle.dir_y))
            || !std::isfinite(static_cast<double>(particle.dir_z))) {
            throw std::runtime_error("pair final-state particle reserved/numeric fields are invalid");
        }
        if (slot < expected_count) {
            amrex::Real const norm = std::sqrt(
                particle.dir_x * particle.dir_x
                + particle.dir_y * particle.dir_y
                + particle.dir_z * particle.dir_z);
            if (particle.kinetic_fraction < amrex::Real(0.0)
                || std::abs(norm - amrex::Real(1.0)) > amrex::Real(1.0e-8)) {
                throw std::runtime_error("pair final-state particle record is invalid");
            }
            fraction_sum += particle.kinetic_fraction;
        } else if (particle.pdg != 0
                   || particle.kinetic_fraction != amrex::Real(0.0)
                   || particle.dir_x != amrex::Real(0.0)
                   || particle.dir_y != amrex::Real(0.0)
                   || particle.dir_z != amrex::Real(0.0)) {
            throw std::runtime_error("pair final-state unused slot is not all-zero");
        }
    }
    int electron_count = 0;
    int positron_count = 0;
    for (int slot = 0; slot < expected_count; ++slot) {
        electron_count += record.particles[slot].pdg == 11 ? 1 : 0;
        positron_count += record.particles[slot].pdg == -11 ? 1 : 0;
    }
    amrex::Real const closure_tolerance = amrex::Real(64.0)
        * std::numeric_limits<amrex::Real>::epsilon();
    if (electron_count != (expected_count - 1) || positron_count != 1
        || std::abs(fraction_sum - amrex::Real(1.0)) > closure_tolerance) {
        throw std::runtime_error("pair final-state record topology/energy closure is invalid");
    }
}

amrex::Vector<RreaPairEmpiricalBlock> load_pair_empirical_blocks(
    std::filesystem::path const& binary_path,
    std::filesystem::path const& index_path)
{
    // The 144-byte record is stored as twenty named columns; the field names
    // are the container header, not offset arithmetic repeated on both sides.
    auto const bank = rrea::table_container::Read(binary_path.string());
    if (rrea::json_text::value(bank.header, "kind", "pair final-state") != "records"
        || rrea::json_text::value(bank.header, "record_bytes", "pair final-state")
            != "144") {
        throw std::runtime_error(
            "pair final-state container is not a 144-byte record set");
    }
    auto column = [&bank](char const* name) {
        return rrea::table_container::Column(bank, name);
    };
    auto const col_target_Z = column("target_Z");
    auto const col_channel_code = column("channel_code");
    auto const col_lepton_count = column("lepton_count");
    auto const col_flags = column("flags");
    auto const col_primary_energy = column("primary_energy_eV");
    auto const col_recoil = column("local_recoil_fraction");
    std::array<amrex::Vector<amrex::Real>, 3> col_pdg;
    std::array<amrex::Vector<amrex::Real>, 3> col_reserved;
    std::array<amrex::Vector<amrex::Real>, 3> col_fraction;
    std::array<amrex::Vector<amrex::Real>, 3> col_dir_x;
    std::array<amrex::Vector<amrex::Real>, 3> col_dir_y;
    std::array<amrex::Vector<amrex::Real>, 3> col_dir_z;
    for (int slot = 0; slot < 3; ++slot) {
        std::string const prefix = "p" + std::to_string(slot) + "_";
        col_pdg[slot] = column((prefix + "pdg").c_str());
        col_reserved[slot] = column((prefix + "reserved").c_str());
        col_fraction[slot] = column((prefix + "kinetic_fraction").c_str());
        col_dir_x[slot] = column((prefix + "dir_x").c_str());
        col_dir_y[slot] = column((prefix + "dir_y").c_str());
        col_dir_z[slot] = column((prefix + "dir_z").c_str());
    }
    std::uint64_t const record_count =
        static_cast<std::uint64_t>(rrea::table_container::Rows(bank));
    if (col_target_Z.size() != record_count || col_channel_code.size() != record_count
        || col_lepton_count.size() != record_count || col_flags.size() != record_count
        || col_primary_energy.size() != record_count || col_recoil.size() != record_count) {
        throw std::runtime_error("pair final-state columns differ from the row count");
    }
    for (int slot = 0; slot < 3; ++slot) {
        if (col_pdg[slot].size() != record_count
            || col_reserved[slot].size() != record_count
            || col_fraction[slot].size() != record_count
            || col_dir_x[slot].size() != record_count
            || col_dir_y[slot].size() != record_count
            || col_dir_z[slot].size() != record_count) {
            throw std::runtime_error("pair final-state columns differ from the row count");
        }
    }
    // Size and record-count validation above catch malformed payloads; the
    // physical distribution is checked by independent train/holdout tests.
    std::string const index_text = file_text(index_path.string());
    // The producer's own failure ledger: the one index field that can report a
    // real problem instead of restating the schema.
    if (!json_array_objects(index_text, "failures", false).empty()) {
        throw std::runtime_error("pair final-state index reports producer failures");
    }

    amrex::Vector<RreaPairEmpiricalBlock> blocks;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> occupied_record_ranges;
    for (std::string const& object : json_array_objects(index_text, "blocks")) {
        std::string const split = json_string(object, "split");
        RreaPhotonChannel const channel = parse_pair_channel(json_string(object, "channel"));
        amrex::Real const primary_energy_eV =
            json_number_or_default(object, "primary_energy_eV", amrex::Real(-1.0));
        std::uint64_t const byte_offset = parse_number<std::uint64_t>(
            json_string_or_number(object, "byte_offset"), "byte_offset");
        std::uint64_t const record_offset = parse_number<std::uint64_t>(
            json_string_or_number(object, "record_offset"), "record_offset");
        std::uint64_t const count = parse_number<std::uint64_t>(
            json_string_or_number(object, "count"), "count");
        if (!(primary_energy_eV > amrex::Real(0.0))
            || record_offset > record_count || count > record_count - record_offset
            || record_offset > (std::numeric_limits<std::uint64_t>::max() - 64U) / 144U
            || byte_offset != 64U + record_offset * 144U) {
            throw std::runtime_error("pair final-state index block has invalid offset/count");
        }
        if (count > 0U) {
            occupied_record_ranges.emplace_back(record_offset, record_offset + count);
        }
        if (split != "training") {
            if (split != "holdout") {
                throw std::runtime_error("pair final-state index split must be training or holdout");
            }
            continue;
        }
        if (count == 0U) {
            continue;  // analytic threshold node
        }
        if (count < 4096U) {
            throw std::runtime_error("pair final-state training block has fewer than 4096 records");
        }

        RreaPairEmpiricalBlock block;
        block.channel = channel;
        block.primary_energy_eV = primary_energy_eV;
        block.training_records.reserve(static_cast<std::size_t>(count));
        for (std::uint64_t i = 0; i < count; ++i) {
            std::size_t const row = static_cast<std::size_t>(record_offset + i);
            RreaPairEmpiricalRecord record;
            record.target_Z = static_cast<int>(std::llround(
                static_cast<double>(col_target_Z[row])));
            auto const channel_code = static_cast<std::uint32_t>(std::llround(
                static_cast<double>(col_channel_code[row])));
            record.lepton_count = static_cast<int>(std::llround(
                static_cast<double>(col_lepton_count[row])));
            auto const flags = static_cast<std::uint32_t>(std::llround(
                static_cast<double>(col_flags[row])));
            record.primary_energy_eV = col_primary_energy[row];
            record.local_recoil_fraction = col_recoil[row];
            if (channel_code != 1U && channel_code != 2U) {
                throw std::runtime_error("pair final-state record channel code is invalid");
            }
            record.channel = channel_code == 1U
                ? RreaPhotonChannel::PairNuclear
                : RreaPhotonChannel::PairTriplet;
            std::array<std::uint32_t, 3> reserved_words{};
            for (int slot = 0; slot < 3; ++slot) {
                record.particles[slot].pdg = static_cast<int>(std::llround(
                    static_cast<double>(col_pdg[slot][row])));
                reserved_words[static_cast<std::size_t>(slot)] =
                    static_cast<std::uint32_t>(std::llround(
                        static_cast<double>(col_reserved[slot][row])));
                record.particles[slot].kinetic_fraction = col_fraction[slot][row];
                record.particles[slot].dir_x = col_dir_x[slot][row];
                record.particles[slot].dir_y = col_dir_y[slot][row];
                record.particles[slot].dir_z = col_dir_z[slot][row];
            }
            validate_pair_empirical_record_contract(
                record, channel, primary_energy_eV, flags, reserved_words);
            block.training_records.push_back(record);
        }
        blocks.push_back(std::move(block));
    }
    // Index vs binary, two separate files: the index's blocks must tile the
    // payload the binary header declares -- contiguous, no overlap, no gap, and
    // ending exactly on record_count.  A short or duplicate payload fails
    // here, which the per-block offset arithmetic above cannot see.
    std::sort(occupied_record_ranges.begin(), occupied_record_ranges.end());
    std::uint64_t next_record = 0U;
    for (auto const& range : occupied_record_ranges) {
        if (range.first != next_record) {
            throw std::runtime_error(
                "pair final-state index blocks overlap or leave a record gap");
        }
        next_record = range.second;
    }
    if (next_record != record_count) {
        throw std::runtime_error(
            "pair final-state index blocks do not cover the binary payload");
    }
    std::sort(
        blocks.begin(), blocks.end(),
        [](RreaPairEmpiricalBlock const& lhs, RreaPairEmpiricalBlock const& rhs) {
            if (lhs.channel != rhs.channel) {
                return static_cast<int>(lhs.channel) < static_cast<int>(rhs.channel);
            }
            return lhs.primary_energy_eV < rhs.primary_energy_eV;
        });
    for (RreaPhotonChannel channel : {
             RreaPhotonChannel::PairNuclear, RreaPhotonChannel::PairTriplet}) {
        std::size_t count = 0;
        amrex::Real last_energy = amrex::Real(0.0);
        for (auto const& block : blocks) {
            if (block.channel != channel) {
                continue;
            }
            if (!(block.primary_energy_eV > last_energy)) {
                throw std::runtime_error("pair final-state training energies are not unique/increasing");
            }
            last_energy = block.primary_energy_eV;
            ++count;
        }
        if (count < 2U || last_energy < amrex::Real(1.0e11)) {
            throw std::runtime_error("pair final-state training blocks do not cover 100 GeV");
        }
    }
    return blocks;
}

}  // namespace

bool RreaPairChannelHasPhaseSpace(
    amrex::Real photon_energy_eV,
    RreaPhotonChannel channel) noexcept
{
    amrex::Real const threshold = channel == RreaPhotonChannel::PairTriplet
        ? kPairTripletThresholdEv : kPairNuclearThresholdEv;
    return std::isfinite(static_cast<double>(photon_energy_eV))
        && photon_energy_eV > threshold;
}

RreaPairFinalState BuildRreaPairEmpiricalFinalState(
    RreaPairEmpiricalRecord const& record,
    amrex::Real photon_energy_eV,
    RreaPhotonChannel channel,
    amrex::Real draw_pair_azimuth)
{
    RreaPairFinalState state;
    state.channel = channel;
    bool const triplet = channel == RreaPhotonChannel::PairTriplet;
    bool const nuclear = channel == RreaPhotonChannel::PairNuclear;
    if ((!triplet && !nuclear) || record.channel != channel
        || !RreaPairChannelHasPhaseSpace(photon_energy_eV, channel)) {
        return state;
    }

    int electron_slot = -1;
    int positron_slot = -1;
    int recoil_slot = -1;
    for (int slot = 0; slot < record.lepton_count; ++slot) {
        if (record.particles[slot].pdg == -11) {
            positron_slot = slot;
        } else if (record.particles[slot].pdg == 11) {
            if (electron_slot < 0) {
                electron_slot = slot;
            } else {
                recoil_slot = slot;
            }
        }
    }
    if (electron_slot < 0 || positron_slot < 0 || (triplet && recoil_slot < 0)) {
        amrex::Abort("schema-6 selected pair record has invalid lepton topology");
    }

    amrex::Real const available_kinetic_eV =
        photon_energy_eV - amrex::Real(2.0) * kElectronRestEnergyEv;
    amrex::Real const rotation = kTwoPi * clamp_unit(draw_pair_azimuth);
    amrex::Real const cos_rotation = std::cos(rotation);
    amrex::Real const sin_rotation = std::sin(rotation);
    auto rotated_direction = [&] (
        RreaPairEmpiricalParticle const& particle,
        amrex::Real& x,
        amrex::Real& y,
        amrex::Real& z) {
        x = cos_rotation * particle.dir_x - sin_rotation * particle.dir_y;
        y = sin_rotation * particle.dir_x + cos_rotation * particle.dir_y;
        z = particle.dir_z;
        normalize_local_direction(x, y, z);
    };

    state.valid = true;
    state.electron_energy_eV = available_kinetic_eV
        * record.particles[electron_slot].kinetic_fraction;
    state.positron_energy_eV = available_kinetic_eV
        * record.particles[positron_slot].kinetic_fraction;
    rotated_direction(
        record.particles[electron_slot], state.electron_dir_x,
        state.electron_dir_y, state.electron_dir_z);
    rotated_direction(
        record.particles[positron_slot], state.positron_dir_x,
        state.positron_dir_y, state.positron_dir_z);
    if (triplet) {
        state.creates_target_ion = true;
        state.has_recoil_electron = true;
        state.recoil_electron_energy_eV = available_kinetic_eV
            * record.particles[recoil_slot].kinetic_fraction;
        rotated_direction(
            record.particles[recoil_slot], state.recoil_dir_x,
            state.recoil_dir_y, state.recoil_dir_z);
    }

    amrex::Real const tolerance = amrex::Real(64.0)
        * std::numeric_limits<amrex::Real>::epsilon()
        * std::max(amrex::Real(1.0), available_kinetic_eV);
    auto explicit_sum = [&]() {
        return state.electron_energy_eV + state.positron_energy_eV
            + state.recoil_electron_energy_eV;
    };
    amrex::Real explicit_kinetic_eV = explicit_sum();
    if (!std::isfinite(static_cast<double>(explicit_kinetic_eV))
        || state.electron_energy_eV < amrex::Real(0.0)
        || state.positron_energy_eV < amrex::Real(0.0)
        || state.recoil_electron_energy_eV < amrex::Real(0.0)) {
        amrex::Abort("schema-6 pair final-state record produced a negative/nonfinite energy");
    }
    if (explicit_kinetic_eV > available_kinetic_eV) {
        if (explicit_kinetic_eV - available_kinetic_eV > tolerance) {
            amrex::Abort("schema-6 pair final-state record failed energy closure");
        }
        amrex::Real const scale = available_kinetic_eV / explicit_kinetic_eV;
        state.electron_energy_eV *= scale;
        state.positron_energy_eV *= scale;
        state.recoil_electron_energy_eV *= scale;
        explicit_kinetic_eV = explicit_sum();
    }
    state.local_deposit_energy_eV = available_kinetic_eV - explicit_kinetic_eV;
    auto reduce_largest = [&] (amrex::Real correction) {
        std::array<amrex::Real*, 3> energies{{
            &state.electron_energy_eV,
            &state.positron_energy_eV,
            &state.recoil_electron_energy_eV,
        }};
        auto largest = std::max_element(
            energies.begin(), energies.end(),
            [](amrex::Real const* lhs, amrex::Real const* rhs) {
                return *lhs < *rhs;
            });
        if (largest == energies.end() || **largest < correction) {
            amrex::Abort("schema-6 pair final-state cannot close nonnegative remainder");
        }
        **largest -= correction;
    };
    if (state.local_deposit_energy_eV < amrex::Real(0.0)) {
        amrex::Real const correction = -state.local_deposit_energy_eV;
        if (correction > tolerance) {
            amrex::Abort("schema-6 pair final-state exact remainder is negative");
        }
        reduce_largest(correction);
        state.local_deposit_energy_eV = amrex::Real(0.0);
    }
    amrex::Real accounted = explicit_sum() + state.local_deposit_energy_eV;
    state.energy_residual_eV = available_kinetic_eV - accounted;
    if (std::abs(state.energy_residual_eV) > tolerance) {
        amrex::Abort("schema-6 pair final-state record failed energy closure");
    }
    if (state.energy_residual_eV >= amrex::Real(0.0)) {
        state.local_deposit_energy_eV += state.energy_residual_eV;
    } else if (state.local_deposit_energy_eV >= -state.energy_residual_eV) {
        state.local_deposit_energy_eV += state.energy_residual_eV;
    } else {
        reduce_largest(-state.energy_residual_eV);
    }
    accounted = explicit_sum() + state.local_deposit_energy_eV;
    if (state.electron_energy_eV < amrex::Real(0.0)
        || state.positron_energy_eV < amrex::Real(0.0)
        || state.recoil_electron_energy_eV < amrex::Real(0.0)
        || state.local_deposit_energy_eV < amrex::Real(0.0)
        || std::abs(available_kinetic_eV - accounted) > tolerance) {
        amrex::Abort("schema-6 pair final-state nonnegative closure correction failed");
    }
    state.energy_residual_eV = amrex::Real(0.0);
    return state;
}

void ValidateRreaConditionalCdfGroups(
    amrex::Vector<RreaConditionalCdfGroup> const& groups,
    std::string const& source_label)
{
    if (groups.size() < 2U) {
        throw std::runtime_error(
            "Conditional CDF table needs at least two primary-energy groups: "
            + source_label);
    }
    amrex::Real previous_primary = amrex::Real(0.0);
    for (std::size_t group_index = 0; group_index < groups.size(); ++group_index) {
        auto const& group = groups[group_index];
        if (!std::isfinite(static_cast<double>(group.primary_energy_eV))
            || !(group.primary_energy_eV > amrex::Real(0.0))
            || (group_index > 0U
                && !(group.primary_energy_eV > previous_primary))) {
            throw std::runtime_error(
                "Conditional CDF primary-energy groups must be finite, positive, "
                "and strictly increasing: " + source_label);
        }
        previous_primary = group.primary_energy_eV;
        if (group.cdf_u.size() < 2U
            || group.cdf_u.size() != group.values.size()) {
            throw std::runtime_error(
                "Conditional CDF group has an invalid node count: " + source_label);
        }
        if (group.cdf_u.front() != amrex::Real(0.0)
            || group.cdf_u.back() != amrex::Real(1.0)) {
            throw std::runtime_error(
                "Conditional CDF group must have exact cdf_u endpoints 0 and 1: "
                + source_label);
        }
        for (std::size_t node = 0; node < group.cdf_u.size(); ++node) {
            amrex::Real const cdf = group.cdf_u[node];
            amrex::Real const value = group.values[node];
            if (!std::isfinite(static_cast<double>(cdf))
                || cdf < amrex::Real(0.0) || cdf > amrex::Real(1.0)
                || (node > 0U && !(cdf > group.cdf_u[node - 1U]))) {
                throw std::runtime_error(
                    "Conditional CDF probability nodes must be finite and strictly "
                    "increasing from 0 to 1: " + source_label);
            }
            if (!std::isfinite(static_cast<double>(value))
                || value < amrex::Real(0.0)
                || (node > 0U && value < group.values[node - 1U])) {
                throw std::runtime_error(
                    "Conditional inverse-CDF sampled values must be finite, "
                    "nonnegative, and nondecreasing: " + source_label);
            }
        }
    }
}

amrex::Real RreaInteractionTable::Interpolate(amrex::Real energy_eV_query) const
{
    if (energy_eV.empty() || values.empty() || energy_eV.size() != values.size()
        || (mode != RreaInterpolationMode::Linear
            && log_energy.size() != energy_eV.size())) {
        amrex::Abort(
            "RREA interaction table is empty, malformed, or not Finalize()d: " + name);
    }
    if (!std::isfinite(static_cast<double>(energy_eV_query))
        || energy_eV_query < energy_eV.front() || energy_eV_query > energy_eV.back()) {
        std::ostringstream message;
        message << "RREA interaction table '" << name << "' query "
                << std::setprecision(17) << energy_eV_query
                << " eV is outside the certified inclusive range ["
                << energy_eV.front() << ", " << energy_eV.back()
                << "] eV; extrapolation/clamping is forbidden";
        amrex::Abort(message.str());
    }
    auto const upper = std::lower_bound(energy_eV.begin(), energy_eV.end(), energy_eV_query);
    if (upper == energy_eV.begin()) {
        return values.front();
    }
    if (upper == energy_eV.end()) {
        return values.back();
    }
    std::size_t const hi = static_cast<std::size_t>(std::distance(energy_eV.begin(), upper));
    std::size_t const lo = hi - 1U;
    if (energy_eV[hi] == energy_eV_query) {
        return values[hi];
    }
    if (mode == RreaInterpolationMode::Linear) {
        amrex::Real const t = (energy_eV_query - energy_eV[lo])
            / (energy_eV[hi] - energy_eV[lo]);
        return values[lo] + t * (values[hi] - values[lo]);
    }
    amrex::Real const t = (std::log(energy_eV_query) - log_energy[lo])
        / (log_energy[hi] - log_energy[lo]);
    bool const positive_bracket =
        values[lo] > amrex::Real(0.0) && values[hi] > amrex::Real(0.0);
    if (mode == RreaInterpolationMode::LogXLinear
        || (mode == RreaInterpolationMode::LogLogPositiveWithZeroThreshold
            && !positive_bracket)) {
        return values[lo] + t * (values[hi] - values[lo]);
    }
    if (!positive_bracket) {
        amrex::Abort(
            "loglog_linear requires positive bracketing values in table '"
            + name + "'");
    }
    return std::exp(log_values[lo] + t * (log_values[hi] - log_values[lo]));
}

void RreaInteractionTable::Finalize()
{
    if (interpolation == "loglog_linear") {
        mode = RreaInterpolationMode::LogLogLinear;
    } else if (interpolation == "loglog_positive_with_zero_threshold") {
        mode = RreaInterpolationMode::LogLogPositiveWithZeroThreshold;
    } else if (interpolation == "logx_linear") {
        mode = RreaInterpolationMode::LogXLinear;
    } else if (interpolation == "linear") {
        mode = RreaInterpolationMode::Linear;
    } else {
        throw std::runtime_error(
            "Unsupported RREA interaction interpolation '" + interpolation
            + "' for table '" + name + "'");
    }
    if (mode == RreaInterpolationMode::Linear) {
        return;
    }
    log_energy.resize(energy_eV.size());
    log_values.resize(values.size());
    for (std::size_t i = 0; i < energy_eV.size(); ++i) {
        log_energy[i] = std::log(energy_eV[i]);
        log_values[i] = values[i] > amrex::Real(0.0)
            ? std::log(values[i]) : amrex::Real(0.0);
    }
}

void RreaInteractionTables::Load(std::string const& config_path)
{
    if (config_path.empty()) {
        throw std::runtime_error("rrea.interaction_table_config is required");
    }
    std::filesystem::path const config_fs(config_path);
    std::filesystem::path const config_dir = config_fs.parent_path();
    std::string const text = file_text(config_path);
    std::string const schema = json_string_or_number(text, "schema_version");
    if (schema != "6") {
        throw std::runtime_error(
            "Unsupported interaction transport configuration schema_version: " + schema
            + " (only production schema 6 is supported)");
    }
    amrex::Real gs_wentzel_transition_energy_eV = amrex::Real(0.0);
    amrex::Real schema6_gs_few_collision_threshold = amrex::Real(0.0);
    amrex::Real schema6_gs_isotropic_above_n = amrex::Real(0.0);
    std::string const scattering =
        json_object_or_empty(text, "elastic_scattering_model");
    if (scattering.empty()) {
        throw std::runtime_error(
            "schema-6 transport configuration is missing elastic_scattering_model");
    }
    gs_wentzel_transition_energy_eV = json_number_or_default(
        scattering, "transition_energy_eV", amrex::Real(-1.0));
    schema6_gs_few_collision_threshold = json_number_or_default(
        scattering, "few_collision_threshold_n", amrex::Real(-1.0));
    schema6_gs_isotropic_above_n = json_number_or_default(
        scattering, "isotropic_above_path_n", amrex::Real(-1.0));
    // The transport configuration owns the GS/WentzelVI split energy; the engine binds to it
    // (branch selection, and the per-table coverage bounds below) and requires
    // only that it is usable.  The per-table branch/model contract enforced by
    // require_scattering_table_contract is what ties each loaded angle table to
    // the model that produced it.
    if (!std::isfinite(static_cast<double>(gs_wentzel_transition_energy_eV))
        || !(gs_wentzel_transition_energy_eV > amrex::Real(0.0))) {
        throw std::runtime_error(
            "schema-6 elastic_scattering_model must declare a positive "
            "transition_energy_eV");
    }

    std::map<std::string, RreaInteractionTable> tables;
    amrex::Vector<RreaMollerCdfGroup> moller_cdf;
    std::map<std::string, amrex::Vector<RreaConditionalCdfGroup>> conditional_cdfs;
    std::map<std::string, std::string> conditional_active_rate_tables;
    amrex::Vector<RreaPhotoelectricSubshellCdfGroup> photoelectric_subshell_cdf;
    RreaElasticMfpTable electron_elastic_mfp;
    RreaElasticMfpTable positron_elastic_mfp;
    RreaGsAngleTable electron_gs_angle;
    RreaGsAngleTable positron_gs_angle;
    RreaGsAngleTable electron_wentzel_angle;
    RreaGsAngleTable positron_wentzel_angle;
    amrex::Vector<RreaPairEmpiricalBlock> pair_empirical_blocks;
    std::set<std::string> declared_table_names;
    for (auto const& object : table_objects(text)) {
        std::string const table_name = json_string(object, "name");
        if (!declared_table_names.emplace(table_name).second) {
            throw std::runtime_error(
                "Duplicate interaction table name in transport configuration: " + table_name);
        }
        std::string const table_kind = json_string_or_empty(object, "table_kind");
        if (table_kind == "gs_angle_cdf_2d"
            || table_kind == "wentzel_vi_angle_cdf_2d") {
            auto angle = load_gs_angle_table(config_dir, object);
            if (table_name == "electron_gs_angle_cdf_air") {
                if (table_kind != "gs_angle_cdf_2d") {
                    throw std::runtime_error(
                        "electron GS angle table has the wrong table_kind");
                }
                require_scattering_table_contract(
                    object, kOption4GsBranch, kOption4GsModel,
                    amrex::Real(1.0e3), gs_wentzel_transition_energy_eV);
                electron_gs_angle = std::move(angle);
            } else if (table_name == "positron_gs_angle_cdf_air") {
                if (table_kind != "gs_angle_cdf_2d") {
                    throw std::runtime_error(
                        "positron GS angle table has the wrong table_kind");
                }
                require_scattering_table_contract(
                    object, kOption4GsBranch, kOption4GsModel,
                    amrex::Real(1.0e3), gs_wentzel_transition_energy_eV);
                positron_gs_angle = std::move(angle);
            } else if (table_name == "electron_wentzel_vi_angle_cdf_air") {
                if (table_kind != "wentzel_vi_angle_cdf_2d") {
                    throw std::runtime_error(
                        "WentzelVI angle tables are valid only for production schema 6");
                }
                require_scattering_table_contract(
                    object, kOption4WentzelBranch, kOption4WentzelModel,
                    gs_wentzel_transition_energy_eV, amrex::Real(1.0e10));
                electron_wentzel_angle = std::move(angle);
            } else if (table_name == "positron_wentzel_vi_angle_cdf_air") {
                if (table_kind != "wentzel_vi_angle_cdf_2d") {
                    throw std::runtime_error(
                        "WentzelVI angle tables are valid only for production schema 6");
                }
                require_scattering_table_contract(
                    object, kOption4WentzelBranch, kOption4WentzelModel,
                    gs_wentzel_transition_energy_eV, amrex::Real(1.0e10));
                positron_wentzel_angle = std::move(angle);
            } else {
                throw std::runtime_error(
                    "Unexpected elastic angle CDF table: " + table_name);
            }
            continue;
        }
        if (table_name == "electron_elastic_mfp_air" || table_name == "positron_elastic_mfp_air") {
            auto em = load_elastic_mfp_table(config_dir, object);
            if (table_name == "electron_elastic_mfp_air") {
                electron_elastic_mfp = std::move(em);
            } else {
                positron_elastic_mfp = std::move(em);
            }
            continue;
        }
        if (table_kind == "conditional_cdf") {
            std::string value_column = json_string_or_empty(object, "value_column");
            if (value_column.empty()) {
                value_column = json_string(object, "primary_value_column");
            }
            auto groups = load_conditional_cdf(
                config_dir,
                object,
                "primary_energy_eV",
                value_column);
            conditional_cdfs.emplace(table_name, std::move(groups));
            std::string const active_rate_table =
                json_string_or_empty(object, "active_rate_table");
            if (!active_rate_table.empty()) {
                conditional_active_rate_tables.emplace(table_name, active_rate_table);
            }
            if (table_name == "electron_moller_secondary_cdf_air") {
                // Populate the electron-specific view consumed by the Moller sampler.
                auto const& generic = conditional_cdfs.at(table_name);
                for (auto const& group : generic) {
                    RreaMollerCdfGroup converted;
                    converted.primary_energy_eV = group.primary_energy_eV;
                    converted.cdf_u = group.cdf_u;
                    converted.secondary_energy_eV = group.values;
                    moller_cdf.push_back(std::move(converted));
                }
            }
            continue;
        }
        if (table_kind == "photoelectric_subshell_cdf") {
            if (!json_bool_or_default(object, "allow_duplicate_x", false)
                || json_string_or_empty(object, "duplicate_x_exact_policy")
                    != "low_side_at_exact_edge_high_side_immediately_above") {
                throw std::runtime_error(
                    "schema-6 photoelectric subshell CDF must declare the exact "
                    "low-side/high-side duplicate shell-edge policy");
            }
            photoelectric_subshell_cdf = load_photoelectric_subshell_cdf(config_dir, object);
            continue;
        }
        std::string x_column = "energy_eV";
        auto table = load_table(config_dir, object, x_column);
        auto const [_, inserted] = tables.emplace(table.name, table);
        if (!inserted) {
            throw std::runtime_error("Duplicate interaction table name: " + table.name);
        }
    }

    for (auto const& required : {
             "photon_compton_inverse_length_air",
             "photon_photoelectric_inverse_length_air",
             "photon_pair_nuclear_inverse_length_air",
             "photon_pair_triplet_inverse_length_air",
             "electron_collision_stopping_air",
             "electron_soft_radiative_stopping_air",
             "electron_hard_moller_inverse_length_air",
             "electron_tracked_brems_inverse_length_air",
             "electron_elastic_transport_inverse_length_air",
             "positron_collision_stopping_air",
             "positron_soft_radiative_stopping_air",
             "positron_hard_bhabha_inverse_length_air",
             "positron_tracked_brems_inverse_length_air",
             "positron_elastic_transport_inverse_length_air",
             "positron_annihilation_inverse_length_air",
         }) {
        if (tables.find(required) == tables.end()) {
            throw std::runtime_error(
                std::string("Missing required production schema-6 process column: ")
                + required);
        }
    }
    std::map<std::string, std::string> const expected_active_rates{
        {"electron_moller_secondary_cdf_air",
         "electron_hard_moller_inverse_length_air"},
        {"brems_photon_energy_cdf_air",
         "electron_tracked_brems_inverse_length_air"},
        {"positron_hard_bhabha_secondary_cdf_air",
         "positron_hard_bhabha_inverse_length_air"},
        {"positron_brems_photon_energy_cdf_air",
         "positron_tracked_brems_inverse_length_air"},
    };
    if (conditional_active_rate_tables != expected_active_rates) {
        throw std::runtime_error(
            "schema-6 conditional CDF active_rate_table mapping is missing or invalid");
    }
    // Both pinned option4 branches are mandatory.
    if (electron_elastic_mfp.energy_eV.empty() || positron_elastic_mfp.energy_eV.empty()) {
        throw std::runtime_error(
            "Missing required RREA elastic parameter table: "
            "{electron,positron}_elastic_mfp_air");
    }
    if (electron_gs_angle.energy_eV.empty() || positron_gs_angle.energy_eV.empty()) {
        throw std::runtime_error(
            "Missing required RREA GS angle table: "
            "{electron,positron}_gs_angle_cdf_air");
    }
    if (electron_wentzel_angle.energy_eV.empty()
        || positron_wentzel_angle.energy_eV.empty()) {
        throw std::runtime_error(
            "Missing required schema-6 WentzelVI angle table: "
            "{electron,positron}_wentzel_vi_angle_cdf_air");
    }
    for (auto const& required : {
             "electron_moller_secondary_cdf_air",
             "brems_photon_energy_cdf_air",
             "positron_hard_bhabha_secondary_cdf_air",
         }) {
        if (conditional_cdfs.find(required) == conditional_cdfs.end()) {
            throw std::runtime_error(
                std::string("Missing required schema-6 conditional CDF table: ")
                + required);
        }
    }
    for (char const* relative : {
             "final_state/pair_final_state_samples_air.bin.rtb",
             "final_state/pair_final_state_samples_air.index.json"}) {
        if (!std::filesystem::is_regular_file(config_dir / relative)) {
            throw std::runtime_error(
                std::string("schema-6 final-state artifact is missing: ")
                + relative);
        }
    }
    pair_empirical_blocks = load_pair_empirical_blocks(
        config_dir / "final_state/pair_final_state_samples_air.bin.rtb",
        config_dir / "final_state/pair_final_state_samples_air.index.json");
    // Geant4-measured positron photon-energy CDF.
    if (conditional_cdfs.find("positron_brems_photon_energy_cdf_air") == conditional_cdfs.end()) {
        throw std::runtime_error(
            "Missing required schema-6 CDF table: "
            "positron_brems_photon_energy_cdf_air");
    }
    if (photoelectric_subshell_cdf.empty()) {
        throw std::runtime_error(
            "Missing required schema-6 table: photoelectric_subshell_cdf_air");
    }

    m_table_class = json_string_or_empty(text, "table_class");
    m_transport_model = json_string_or_empty(text, "transport_model");
    if (m_transport_model != "rrea_mc_pic_v6_geant4_10_7_4p04") {
        throw std::runtime_error(
            "schema-6 transport_model must be "
            "rrea_mc_pic_v6_geant4_10_7_4p04, got '" + m_transport_model + "'");
    }
    if (!json_bool_or_default(text, "photon_feedback_enabled", false)
        || !json_bool_or_default(text, "positron_transport_enabled", false)) {
        throw std::runtime_error(
            "schema-6 transport configuration must explicitly enable photon feedback and positron transport");
    }
    m_hard_moller_secondary_threshold_eV =
        json_number_or_default(text, "hard_moller_secondary_threshold_eV", amrex::Real(0.0));
    m_brems_absolute_cutoff_eV =
        json_number_or_default(text, "brems_absolute_cutoff_eV", amrex::Real(0.0));
    // Required, not defaulted: every rate below is linear in density_ratio, so
    // a missing reference density would silently rescale all of transport.  The
    // NaN default makes "absent" and "malformed" fail the same way.
    m_reference_density_kg_m3 = json_number_or_default(
        text,
        "reference_density_kg_m3",
        std::numeric_limits<amrex::Real>::quiet_NaN());
    if (!std::isfinite(static_cast<double>(m_reference_density_kg_m3))
        || !(m_reference_density_kg_m3 > amrex::Real(0.0))) {
        throw std::runtime_error(
            "schema-6 transport configuration must declare a finite positive "
            "reference_density_kg_m3");
    }
    // The transport configuration owns both cuts; the engine requires only that they are
    // usable, never given values.  Both are flat energies -- there is no
    // parent-dependent fraction -- so there is nothing else to validate.
    if (!(m_hard_moller_secondary_threshold_eV > amrex::Real(0.0))
        || !std::isfinite(static_cast<double>(m_hard_moller_secondary_threshold_eV))
        || !(m_brems_absolute_cutoff_eV > amrex::Real(0.0))
        || !std::isfinite(static_cast<double>(m_brems_absolute_cutoff_eV))) {
        throw std::runtime_error(
            "schema-6 hard-secondary threshold and tracked-brems cutoff must be "
            "finite and positive");
    }
    m_tables = std::move(tables);
    m_moller_cdf = std::move(moller_cdf);
    m_conditional_cdfs = std::move(conditional_cdfs);
    m_photoelectric_subshell_cdf = std::move(photoelectric_subshell_cdf);
    m_electron_elastic_mfp = std::move(electron_elastic_mfp);
    m_positron_elastic_mfp = std::move(positron_elastic_mfp);
    m_electron_gs_angle = std::move(electron_gs_angle);
    m_positron_gs_angle = std::move(positron_gs_angle);
    m_electron_wentzel_angle = std::move(electron_wentzel_angle);
    m_positron_wentzel_angle = std::move(positron_wentzel_angle);
    m_pair_empirical_blocks = std::move(pair_empirical_blocks);
    m_gs_few_collision_threshold = schema6_gs_few_collision_threshold;
    m_gs_isotropic_above_n = schema6_gs_isotropic_above_n;
    m_gs_wentzel_transition_energy_eV = gs_wentzel_transition_energy_eV;
    if (!std::isfinite(static_cast<double>(m_gs_few_collision_threshold))
        || !std::isfinite(static_cast<double>(m_gs_isotropic_above_n))
        || !(m_gs_few_collision_threshold > amrex::Real(0.0))
        || !(m_gs_isotropic_above_n > m_gs_few_collision_threshold)) {
        throw std::runtime_error(
            "elastic scattering path thresholds must be finite, positive, and increasing");
    }
    // The transport configuration owns the certified windows.  json_number_pair already
    // rejects a non-finite, non-positive, or non-increasing pair, and
    // require_coverage below proves the shipped CSV grids actually span what is
    // declared -- the cross-file half of this check.
    auto const charged_range = json_number_pair(text, "charged_energy_range_eV");
    auto const photon_range = json_number_pair(text, "photon_energy_range_eV");
    m_charged_certified_range = {charged_range[0], charged_range[1]};
    m_photon_certified_range = {photon_range[0], photon_range[1]};
        auto require_coverage = [this](
            char const* table_name,
            RreaCertifiedEnergyRange const& range,
            char const* species) {
            auto const& table = Require(table_name);
            if (table.energy_eV.empty()
                || table.energy_eV.front() != range.min_eV
                || table.energy_eV.back() != range.max_eV) {
                std::ostringstream message;
                message << "schema-6 " << species << " table '" << table_name
                        << "' does not cover the certified inclusive range ["
                        << range.min_eV << ", " << range.max_eV << "] eV";
                throw std::runtime_error(message.str());
            }
        };
        for (char const* name : {
                 "electron_collision_stopping_air",
                 "electron_soft_radiative_stopping_air",
                 "electron_hard_moller_inverse_length_air",
                 "electron_tracked_brems_inverse_length_air",
                 "electron_elastic_transport_inverse_length_air"}) {
            require_coverage(name, m_charged_certified_range, "electron");
        }
        for (char const* name : {
                 "positron_collision_stopping_air",
                 "positron_soft_radiative_stopping_air",
                 "positron_hard_bhabha_inverse_length_air",
                 "positron_tracked_brems_inverse_length_air",
                 "positron_elastic_transport_inverse_length_air",
                 "positron_annihilation_inverse_length_air"}) {
            require_coverage(name, m_charged_certified_range, "positron");
        }
        for (char const* name : {
                 "photon_compton_inverse_length_air",
                 "photon_photoelectric_inverse_length_air",
                 "photon_pair_nuclear_inverse_length_air",
                 "photon_pair_triplet_inverse_length_air"}) {
            require_coverage(name, m_photon_certified_range, "photon");
        }
        auto require_group_coverage = [this, &conditional_active_rate_tables](
            char const* name, RreaCertifiedEnergyRange const& range) {
            auto const it = m_conditional_cdfs.find(name);
            auto const active_it = conditional_active_rate_tables.find(name);
            if (it == m_conditional_cdfs.end() || it->second.empty()
                || active_it == conditional_active_rate_tables.end()) {
                throw std::runtime_error(
                    std::string("schema-6 conditional table lacks its active-rate contract: ")
                    + name);
            }
            auto const& rate = Require(active_it->second);
            auto const first_positive = std::find_if(
                rate.values.begin(), rate.values.end(),
                [](amrex::Real value) { return value > amrex::Real(0.0); });
            if (first_positive == rate.values.end()) {
                throw std::runtime_error(
                    std::string("schema-6 conditional active-rate table has no positive node: ")
                    + active_it->second);
            }
            std::size_t const active_index = static_cast<std::size_t>(
                std::distance(rate.values.begin(), first_positive));
            // The rate is linearly opened from the final explicit zero node to
            // the first positive node.  A deterministic limiting CDF is stored
            // at that zero-rate node but is never queried at the node itself;
            // it supplies the mathematically required one-sided interpolation
            // immediately above threshold without inventing zero-statistics
            // Geant4 emissions.
            std::size_t const cdf_start_index = active_index > 0U
                ? active_index - 1U : active_index;
            amrex::Real const cdf_start = rate.energy_eV[cdf_start_index];
            if (rate.interpolation != "loglog_positive_with_zero_threshold"
                || it->second.front().primary_energy_eV != cdf_start
                || it->second.back().primary_energy_eV != range.max_eV) {
                throw std::runtime_error(
                    std::string("schema-6 conditional table does not exactly cover the ")
                    + "one-sided active-rate interpolation domain: " + name);
            }
        };
        for (char const* name : {
                 "electron_moller_secondary_cdf_air",
                 "brems_photon_energy_cdf_air",
                 "positron_hard_bhabha_secondary_cdf_air",
                 "positron_brems_photon_energy_cdf_air"}) {
            require_group_coverage(name, m_charged_certified_range);
        }
        auto require_elastic_coverage = [](RreaElasticMfpTable const& table) {
            return !table.energy_eV.empty()
                && table.energy_eV.front() == amrex::Real(1.0e3)
                && table.energy_eV.back() == amrex::Real(1.0e10);
        };
        auto require_angle_coverage = [this](
            RreaGsAngleTable const& table,
            amrex::Real min_eV,
            amrex::Real max_eV) {
            return !table.energy_eV.empty()
                && table.energy_eV.front() == min_eV
                && table.energy_eV.back() == max_eV
                && !table.path_n.empty()
                && table.path_n.front() <= m_gs_few_collision_threshold
                && table.path_n.back() >= m_gs_isotropic_above_n;
        };
        if (!require_elastic_coverage(m_electron_elastic_mfp)
            || !require_elastic_coverage(m_positron_elastic_mfp)) {
            throw std::runtime_error(
                "schema-6 elastic parameter tables must cover 1 keV through 10 GeV");
        }
        auto require_transport_anchor_consistency = [this](
            RreaElasticMfpTable const& parameters,
            char const* inverse_length_table) {
            auto const& inverse_length = Require(inverse_length_table);
            for (std::size_t i = 0; i < parameters.energy_eV.size(); ++i) {
                amrex::Real const inverse_m =
                    inverse_length.Interpolate(parameters.energy_eV[i]);
                amrex::Real const product =
                    inverse_m * parameters.transport_mfp_m_stp[i];
                if (!std::isfinite(static_cast<double>(product))
                    || std::abs(product - amrex::Real(1.0)) > amrex::Real(0.01)) {
                    std::ostringstream message;
                    message << "schema-6 elastic transport-rate authority disagrees "
                            << "with the GS/Wentzel G1 audit anchor by more than 1% at "
                            << std::setprecision(17) << parameters.energy_eV[i]
                            << " eV: " << inverse_length_table;
                    throw std::runtime_error(message.str());
                }
            }
        };
        require_transport_anchor_consistency(
            m_electron_elastic_mfp,
            "electron_elastic_transport_inverse_length_air");
        require_transport_anchor_consistency(
            m_positron_elastic_mfp,
            "positron_elastic_transport_inverse_length_air");
        if (!require_angle_coverage(
                m_electron_gs_angle,
                m_charged_certified_range.min_eV,
                m_gs_wentzel_transition_energy_eV)
            || !require_angle_coverage(
                m_positron_gs_angle,
                m_charged_certified_range.min_eV,
                m_gs_wentzel_transition_energy_eV)) {
            throw std::runtime_error(
                "schema-6 option4 GS angle tables must exactly cover 1 keV "
                "through the inclusive 100 MeV transition and the certified path range");
        }
        if (!require_angle_coverage(
                m_electron_wentzel_angle,
                m_gs_wentzel_transition_energy_eV,
                m_charged_certified_range.max_eV)
            || !require_angle_coverage(
                m_positron_wentzel_angle,
                m_gs_wentzel_transition_energy_eV,
                m_charged_certified_range.max_eV)) {
            throw std::runtime_error(
                "schema-6 option4 WentzelVI angle tables must exactly cover the "
                "100 MeV transition through 10 GeV and the certified path range");
        }
    if (m_photoelectric_subshell_cdf.empty()
        || m_photoelectric_subshell_cdf.front().primary_energy_eV
            != m_photon_certified_range.min_eV
        || m_photoelectric_subshell_cdf.back().primary_energy_eV
            != m_photon_certified_range.max_eV) {
        throw std::runtime_error(
            "schema-6 photoelectric subshell CDF must cover 1 keV through 100 GeV");
    }
    // Mean energy carried off per discrete hard-Moller event, tabulated once on
    // the hard-Moller rate grid.  ElectronHardMollerEnergyLoss turns it into a
    // drag; doing the quadrature here keeps that accessor allocation-free in
    // the per-particle path.
    {
        auto const& rate = Require("electron_hard_moller_inverse_length_air");
        m_hard_moller_mean_secondary_eV.assign(rate.energy_eV.size(), amrex::Real(0.0));
        int const samples = 4096;
        for (std::size_t i = 0; i < rate.energy_eV.size(); ++i) {
            amrex::Real const energy_eV = rate.energy_eV[i];
            // A node with no hard-Moller rate carries no events, so a zero mean
            // is the right answer there.  Key on the rate itself rather than on
            // twice the secondary threshold: that 2x kinematic factor is already
            // owned by the rate table's own onset, and restating it here differs
            // from the CDF's bisected onset node by ULPs.
            if (!(rate.values[i] > amrex::Real(0.0))) {
                continue;
            }
            // A node that HAS rate must lie inside the CDF's tabulated range --
            // require_group_coverage guarantees it.  Falling through to a zero
            // mean instead would silently delete the hard-Moller drag.
            if (m_moller_cdf.empty()
                || energy_eV < m_moller_cdf.front().primary_energy_eV
                || energy_eV > m_moller_cdf.back().primary_energy_eV) {
                throw std::runtime_error(
                    "electron_hard_moller_inverse_length_air has a positive rate "
                    "outside the tabulated Moller secondary CDF primary range");
            }
            double mean = 0.0;
            for (int s = 0; s < samples; ++s) {
                mean += static_cast<double>(SampleMollerSecondaryEnergy(
                    energy_eV,
                    amrex::Real((static_cast<double>(s) + 0.5) / samples)));
            }
            m_hard_moller_mean_secondary_eV[i] =
                amrex::Real(mean / static_cast<double>(samples));
        }
    }
    // Explicit validation sensitivities warp the loaded tables ONCE, here, after
    // every certification check has run on the pristine data, so the
    // per-particle accessors carry no debug branch.  Production leaves the study
    // disabled and this block is a no-op.  A mean-free-path scale enters as its
    // reciprocal on the stored inverse length: the accessor forms 1/(sigma*rho),
    // so dividing sigma by the factor is the same product.  Scaling the column
    // rather than one accessor means every consumer of that rate -- the
    // transport coefficients and the discrete-channel drag included -- sees the
    // same scaled process.
    auto const& sensitivity = DebugTransportSensitivityOptions();
    if (sensitivity.enable) {
        for (auto const& [name, factor] : {
                 std::pair<char const*, amrex::Real>{
                     "electron_collision_stopping_air",
                     sensitivity.electron_collision_loss},
                 {"electron_hard_moller_inverse_length_air",
                  amrex::Real(1.0) / sensitivity.hard_moller_mfp},
                 {"electron_soft_radiative_stopping_air",
                  sensitivity.soft_radiative_drag},
                 {"positron_soft_radiative_stopping_air",
                  sensitivity.soft_radiative_drag}}) {
            auto& table = m_tables.at(name);
            for (auto& value : table.values) {
                value *= factor;
            }
            table.Finalize();
        }
        for (auto& mfp : m_electron_elastic_mfp.elastic_mfp_m_stp) {
            mfp *= sensitivity.electron_wentzel_mfp;
        }
    }
    // Bind the per-particle accessors to their tables once.  This is the only
    // "table not loaded" check the hot path needs: after it, Require(RreaTable)
    // is an array index.
    static constexpr std::array<
        char const*, static_cast<std::size_t>(RreaTable::Count)> hot_table_names{
#define RREA_TABLE_NAME(id, table_name) table_name,
        RREA_HOT_TABLES(RREA_TABLE_NAME)
#undef RREA_TABLE_NAME
    };
    for (std::size_t i = 0; i < hot_table_names.size(); ++i) {
        auto const it = m_tables.find(hot_table_names[i]);
        if (it == m_tables.end()) {
            throw std::runtime_error(
                std::string("schema-6 bundle is missing the required transport table: ")
                + hot_table_names[i]);
        }
        m_hot[i] = &it->second;
    }
    m_loaded = true;
}

RreaInteractionTable const& RreaInteractionTables::Require(std::string const& name) const
{
    auto const it = m_tables.find(name);
    if (it == m_tables.end()) {
        amrex::Abort("RREA interaction table not loaded: " + name);
    }
    return it->second;
}

void RreaInteractionTables::RequireSchema6SamplerEnergy(
    char const* sampler,
    char const* species,
    RreaCertifiedEnergyRange const& certified_range,
    amrex::Real energy_eV,
    bool allow_exact_zero) const
{
    if (allow_exact_zero && energy_eV == amrex::Real(0.0)) {
        return;
    }
    if (!std::isfinite(static_cast<double>(energy_eV))
        || !certified_range.Contains(energy_eV)) {
        std::ostringstream message;
        message << sampler << ": schema-6 " << species << " energy "
                << std::setprecision(17) << energy_eV
                << " eV is outside the certified inclusive range ["
                << certified_range.min_eV << ", " << certified_range.max_eV
                << "] eV; endpoint clamping and extrapolation are forbidden";
        if (allow_exact_zero) {
            message << " (exact zero is the only permitted physical exception)";
        }
        amrex::Abort(message.str());
    }
}

amrex::Real RreaInteractionTables::ProbeTableValue(
    std::string const& name, amrex::Real energy_eV) const
{
    return Require(name).Interpolate(energy_eV);
}

RreaPhotonProcessInverseLengths RreaInteractionTables::PhotonProcessInverseLengths(
    amrex::Real energy_eV, amrex::Real density_ratio) const
{
    if (!std::isfinite(static_cast<double>(density_ratio))
        || density_ratio <= amrex::Real(0.0)) {
        amrex::Abort("PhotonProcessInverseLengths density_ratio must be finite and positive");
    }
    if (!m_photon_certified_range.Contains(energy_eV)) {
        std::ostringstream message;
        message << "schema-6 photon energy " << std::setprecision(17)
                << energy_eV << " eV is outside the certified inclusive range ["
                << m_photon_certified_range.min_eV << ", "
                << m_photon_certified_range.max_eV << "] eV";
        amrex::Abort(message.str());
    }
    return RreaPhotonProcessInverseLengths{
        Require(RreaTable::PhotonCompton).Interpolate(energy_eV)
            * density_ratio,
        Require(RreaTable::PhotonPhotoelectric).Interpolate(energy_eV)
            * density_ratio,
        Require(RreaTable::PhotonPairNuclear).Interpolate(energy_eV)
            * density_ratio,
        Require(RreaTable::PhotonPairTriplet).Interpolate(energy_eV)
            * density_ratio};
}

RreaElectronTransportCoefficients RreaInteractionTables::ElectronTransportCoefficients(
    amrex::Real energy_eV, amrex::Real density_ratio) const
{
    if (!std::isfinite(static_cast<double>(density_ratio))
        || density_ratio <= amrex::Real(0.0)) {
        amrex::Abort("ElectronTransportCoefficients density_ratio must be finite and positive");
    }
    if (!m_charged_certified_range.Contains(energy_eV)) {
        std::ostringstream message;
        message << "schema-6 electron energy " << std::setprecision(17)
                << energy_eV << " eV is outside the certified inclusive range ["
                << m_charged_certified_range.min_eV << ", "
                << m_charged_certified_range.max_eV << "] eV";
        amrex::Abort(message.str());
    }
    return RreaElectronTransportCoefficients{
        Require(RreaTable::ElectronCollisionStopping).Interpolate(energy_eV)
            * density_ratio,
        Require(RreaTable::ElectronSoftRadiativeStopping).Interpolate(energy_eV)
            * density_ratio,
        Require(RreaTable::ElectronHardMollerInverseLength).Interpolate(energy_eV)
            * density_ratio,
        Require(RreaTable::ElectronTrackedBremsInverseLength).Interpolate(energy_eV)
            * density_ratio,
        Require(RreaTable::ElectronElasticTransportInverseLength).Interpolate(energy_eV)
            * density_ratio};
}

RreaPositronTransportCoefficients RreaInteractionTables::PositronTransportCoefficients(
    amrex::Real energy_eV, amrex::Real density_ratio) const
{
    if (!std::isfinite(static_cast<double>(density_ratio))
        || density_ratio <= amrex::Real(0.0)) {
        amrex::Abort("PositronTransportCoefficients density_ratio must be finite and positive");
    }
    if (!m_charged_certified_range.Contains(energy_eV)) {
        std::ostringstream message;
        message << "schema-6 positron energy " << std::setprecision(17)
                << energy_eV << " eV is outside the certified inclusive range ["
                << m_charged_certified_range.min_eV << ", "
                << m_charged_certified_range.max_eV << "] eV";
        amrex::Abort(message.str());
    }
    return RreaPositronTransportCoefficients{
        Require(RreaTable::PositronCollisionStopping).Interpolate(energy_eV)
            * density_ratio,
        Require(RreaTable::PositronSoftRadiativeStopping).Interpolate(energy_eV)
            * density_ratio,
        Require(RreaTable::PositronHardBhabhaInverseLength).Interpolate(energy_eV)
            * density_ratio,
        Require(RreaTable::PositronTrackedBremsInverseLength).Interpolate(energy_eV)
            * density_ratio,
        Require(RreaTable::PositronElasticTransportInverseLength).Interpolate(energy_eV)
            * density_ratio,
        Require(RreaTable::PositronAnnihilationInverseLength).Interpolate(energy_eV)
            * density_ratio};
}

bool RreaInteractionTables::MinTableValueInRange(
    std::string const& name,
    amrex::Real lo_eV,
    amrex::Real hi_eV,
    amrex::Real& min_value) const noexcept
{
    auto const it = m_tables.find(name);
    if (it == m_tables.end() || it->second.energy_eV.empty() || !(lo_eV <= hi_eV)) {
        return false;
    }
    auto const& table = it->second;
    if (lo_eV < table.energy_eV.front() || hi_eV > table.energy_eV.back()) {
        return false;
    }
    amrex::Real const lo = lo_eV;
    amrex::Real const hi = hi_eV;
    amrex::Real best = std::min(table.Interpolate(lo), table.Interpolate(hi));
    for (std::size_t i = 0; i < table.energy_eV.size(); ++i) {
        if (table.energy_eV[i] >= lo && table.energy_eV[i] <= hi) {
            best = std::min(best, table.values[i]);
        }
    }
    min_value = best;
    return true;
}

amrex::Real RreaInteractionTables::ElectronCollisionLoss(
    amrex::Real energy_eV,
    amrex::Real density_ratio) const
{
    if (density_ratio <= amrex::Real(0.0)) {
        amrex::Abort("density_ratio must be positive");
    }
    amrex::Real const loss_stp_eV_per_m =
        Require(RreaTable::ElectronCollisionStopping).Interpolate(energy_eV);
    return loss_stp_eV_per_m * density_ratio;
}

amrex::Real RreaInteractionTables::ElectronHardMollerMeanFreePath(
    amrex::Real energy_eV,
    amrex::Real density_ratio) const
{
    if (density_ratio <= amrex::Real(0.0)) {
        amrex::Abort("density_ratio must be positive");
    }
    // Validate the certified energy before the zero-phase-space guard; an
    // out-of-range query must never be disguised as an infinite MFP.
    amrex::Real const inverse_m_stp =
        Require(RreaTable::ElectronHardMollerInverseLength).Interpolate(energy_eV);
    // The rate table is exactly zero at 2*Tcut and authoritative immediately
    // above it.  Do not widen that zero-rate point into an unmodelled band.
    if (m_hard_moller_secondary_threshold_eV > amrex::Real(0.0)
        && energy_eV <= amrex::Real(2.0) * m_hard_moller_secondary_threshold_eV) {
        return amrex::Real(1.0e99);
    }
    return mean_free_path_from_inverse_length(inverse_m_stp * density_ratio);
}

amrex::Real RreaInteractionTables::ElectronHardMollerEnergyLoss(
    amrex::Real energy_eV,
    amrex::Real density_ratio) const
{
    if (density_ratio <= amrex::Real(0.0)) {
        amrex::Abort("density_ratio must be positive");
    }
    // Mean drag from the DISCRETE delta-ray channel: <T> / lambda_hardMoller.
    // The parent's actual loss is realized stochastically by sampled Moller
    // events, so this must never be added to the continuous drag that transport
    // applies -- that would double-count.  It exists because restricted
    // stopping power alone is not the break-even criterion: at the 1.2 MeV
    // minimum it is 180.5 kV/m against a true total of 199.4, and a runaway
    // gate built on the restricted value fires about 10 percent too easily.
    auto const& rate = Require(RreaTable::ElectronHardMollerInverseLength);
    amrex::Real const inverse_m_stp = rate.Interpolate(energy_eV);
    if (!(inverse_m_stp > amrex::Real(0.0))
        || m_hard_moller_mean_secondary_eV.size() != rate.energy_eV.size()) {
        return amrex::Real(0.0);
    }
    amrex::Real const mean_secondary_eV = interp_semilogx_strict(
        rate.energy_eV, m_hard_moller_mean_secondary_eV, energy_eV);
    if (!(mean_secondary_eV > amrex::Real(0.0))) {
        return amrex::Real(0.0);
    }
    return mean_secondary_eV * inverse_m_stp * density_ratio;
}

amrex::Real RreaInteractionTables::ElectronElasticTransportMeanFreePath(
    amrex::Real energy_eV,
    amrex::Real density_ratio) const
{
    if (density_ratio <= amrex::Real(0.0)) {
        amrex::Abort("density_ratio must be positive");
    }
    return mean_free_path_from_inverse_length(
        ElectronTransportCoefficients(energy_eV, density_ratio)
            .elastic_transport_inverse_m);
}

// --- GS / Wentzel elastic runtime accessors ----------------------------------

namespace {

// Apply the explicit scattering sensitivity scale to every elastic branch.
amrex::Real apply_electron_theta_debug_scale(bool is_positron, amrex::Real cos_theta)
{
    if (is_positron) {
        return cos_theta;
    }
    auto const& sensitivity = DebugTransportSensitivityOptions();
    amrex::Real const scale = sensitivity.electron_scattering_theta;
    if (!sensitivity.enable || scale == amrex::Real(1.0)) {
        return cos_theta;
    }
    double const clamped =
        std::min(std::max(static_cast<double>(cos_theta), -1.0), 1.0);
    return static_cast<amrex::Real>(
        std::cos(static_cast<double>(scale) * std::acos(clamped)));
}

}  // namespace

amrex::Real RreaInteractionTables::ElasticMeanFreePath(
    bool is_positron, amrex::Real energy_eV, amrex::Real density_ratio) const
{
    if (density_ratio <= amrex::Real(0.0)) {
        amrex::Abort("density_ratio must be positive");
    }
    auto const& tab = is_positron ? m_positron_elastic_mfp : m_electron_elastic_mfp;
    if (tab.energy_eV.empty()) {
        amrex::Abort("ElasticMeanFreePath: elastic collision-MFP table not loaded");
    }
    // This is the collision MFP used in n=s/lambda_el by the few-collision and
    // conditional-angle samplers. It is deliberately distinct from the
    // unified coefficient table's elastic *transport* inverse length (1/lambda1).
    // STP collision MFP shortens with density (lambda = lambda_stp/rho_ratio).
    return interp_semilogx_strict(tab.energy_eV, tab.elastic_mfp_m_stp, energy_eV)
        / density_ratio;
}

amrex::Real RreaInteractionTables::ScreeningEta(bool is_positron, amrex::Real energy_eV) const
{
    auto const& tab = is_positron ? m_positron_elastic_mfp : m_electron_elastic_mfp;
    if (tab.energy_eV.empty()) {
        amrex::Abort("ScreeningEta: elastic parameter table not loaded");
    }
    return interp_semilogx_strict(tab.energy_eV, tab.screening_eta, energy_eV);
}

amrex::Real RreaInteractionTables::SampleGsCosTheta(
    bool is_positron, amrex::Real energy_eV, amrex::Real path_n, amrex::Real cdf_u) const
{
    RequireSchema6SamplerEnergy(
        "SampleGsCosTheta",
        is_positron ? "positron" : "electron",
        m_charged_certified_range,
        energy_eV);
    if (!std::isfinite(static_cast<double>(cdf_u))
        || cdf_u < amrex::Real(0.0)
        || cdf_u > amrex::Real(1.0)) {
        amrex::Abort("SampleGsCosTheta: schema-6 CDF query must be in [0,1]");
    }
    bool const use_gs = energy_eV <= m_gs_wentzel_transition_energy_eV;
    RreaGsAngleTable const* tab = nullptr;
    if (is_positron) {
        tab = use_gs ? &m_positron_gs_angle : &m_positron_wentzel_angle;
    } else {
        tab = use_gs ? &m_electron_gs_angle : &m_electron_wentzel_angle;
    }
    if (tab == nullptr || tab->energy_eV.empty()) {
        amrex::Abort("SampleGsCosTheta: selected elastic angle table is not loaded");
    }
    if (!std::isfinite(static_cast<double>(energy_eV))
        || energy_eV < tab->energy_eV.front()
        || energy_eV > tab->energy_eV.back()) {
        amrex::Abort("SampleGsCosTheta: energy is outside the selected model branch");
    }
    amrex::Real const u = cdf_u;
    if (!std::isfinite(static_cast<double>(path_n))
        || path_n < m_gs_few_collision_threshold) {
        amrex::Abort(
            "SampleGsCosTheta: path n is below the multiple-scattering branch "
            "or is non-finite");
    }
    if (path_n >= m_gs_isotropic_above_n) {
        return amrex::Real(2.0) * u - amrex::Real(1.0);  // isotropic
    }
    amrex::Real const cos_theta = sample_gs_angle_cos(*tab, energy_eV, path_n, u);
    return apply_electron_theta_debug_scale(is_positron, cos_theta);
}

amrex::Real RreaInteractionTables::SampleSingleElasticCosTheta(
    bool is_positron, amrex::Real energy_eV, amrex::Real cdf_u) const
{
    RequireSchema6SamplerEnergy(
        "SampleSingleElasticCosTheta",
        is_positron ? "positron" : "electron",
        m_charged_certified_range,
        energy_eV);
    // Analytic screened-Rutherford inverse-CDF draw.
    amrex::Real const eta = ScreeningEta(is_positron, energy_eV);
    amrex::Real const u = std::min(std::max(cdf_u, amrex::Real(0.0)), amrex::Real(1.0) - amrex::Real(1.0e-15));
    amrex::Real const cos_theta =
        amrex::Real(1.0) - amrex::Real(2.0) * eta * u / (amrex::Real(1.0) + eta - u);
    return apply_electron_theta_debug_scale(is_positron, cos_theta);
}

amrex::Real RreaInteractionTables::SamplePositronBremsstrahlungPhotonEnergy(
    amrex::Real primary_energy_eV, amrex::Real cdf_u) const
{
    RequireSchema6SamplerEnergy(
        "SamplePositronBremsstrahlungPhotonEnergy",
        "positron",
        m_charged_certified_range,
        primary_energy_eV);
    auto const it = m_conditional_cdfs.find("positron_brems_photon_energy_cdf_air");
    if (it == m_conditional_cdfs.end()) {
        amrex::Abort("SamplePositronBremsstrahlungPhotonEnergy: positron brems CDF not loaded");
    }
    return sample_conditional_cdf(it->second, primary_energy_eV, cdf_u, /*log_interp=*/true);
}

// ===== Hard Moller / Bhabha secondary-energy samplers =====
// Conditional CDFs from the Geant4-generated bundle.  Endpoint kinematics
// (data, not bounds): Moller spans [T_cut, E/2] -- identical particles, the
// faster lepton is primary by convention, so the channel opens only at
// E > 2 T_cut; Bhabha spans [T_cut, E] (distinguishable) and opens at
// E > T_cut.  The zero-phase-space band below each opening carries an
// infinite MFP in the rate tables, not a clamp.
amrex::Real RreaInteractionTables::SampleMollerSecondaryEnergy(
    amrex::Real primary_energy_eV,
    amrex::Real cdf_u) const
{
    RequireSchema6SamplerEnergy(
        "SampleMollerSecondaryEnergy",
        "electron",
        m_charged_certified_range,
        primary_energy_eV);
    if (m_moller_cdf.empty()) {
        amrex::Abort("Moller CDF table is not loaded");
    }
    if (!std::isfinite(static_cast<double>(primary_energy_eV))
        || primary_energy_eV < m_moller_cdf.front().primary_energy_eV
        || primary_energy_eV > m_moller_cdf.back().primary_energy_eV) {
        std::ostringstream message;
        message << "Moller CDF primary energy " << std::setprecision(17)
                << primary_energy_eV << " eV is outside the inclusive range ["
                << m_moller_cdf.front().primary_energy_eV << ", "
                << m_moller_cdf.back().primary_energy_eV
                << "] eV; endpoint clamping is forbidden";
        amrex::Abort(message.str());
    }
    cdf_u = std::max(amrex::Real(0.0), std::min(amrex::Real(1.0), cdf_u));

    auto const sample_group = [cdf_u](RreaMollerCdfGroup const& group) {
        auto const upper = std::lower_bound(group.cdf_u.begin(), group.cdf_u.end(), cdf_u);
        if (upper == group.cdf_u.begin()) {
            return group.secondary_energy_eV.front();
        }
        if (upper == group.cdf_u.end()) {
            return group.secondary_energy_eV.back();
        }
        std::size_t const hi =
            static_cast<std::size_t>(std::distance(group.cdf_u.begin(), upper));
        std::size_t const lo = hi - 1U;
        amrex::Real const denom = group.cdf_u[hi] - group.cdf_u[lo];
        amrex::Real const t = denom > amrex::Real(0.0)
            ? (cdf_u - group.cdf_u[lo]) / denom
            : amrex::Real(0.0);
        return group.secondary_energy_eV[lo]
            + t * (group.secondary_energy_eV[hi] - group.secondary_energy_eV[lo]);
    };

    auto const upper = std::lower_bound(
        m_moller_cdf.begin(),
        m_moller_cdf.end(),
        primary_energy_eV,
        [](RreaMollerCdfGroup const& group, amrex::Real energy) {
            return group.primary_energy_eV < energy;
        });
    if (upper == m_moller_cdf.begin()) {
        return sample_group(*upper);
    }
    if (upper == m_moller_cdf.end()) {
        return sample_group(m_moller_cdf.back());
    }
    std::size_t const hi =
        static_cast<std::size_t>(std::distance(m_moller_cdf.begin(), upper));
    std::size_t const lo = hi - 1U;
    if (m_moller_cdf[hi].primary_energy_eV == primary_energy_eV) {
        return sample_group(m_moller_cdf[hi]);
    }
    amrex::Real const y0 = sample_group(m_moller_cdf[lo]);
    amrex::Real const y1 = sample_group(m_moller_cdf[hi]);
    amrex::Real const x0 = std::log(m_moller_cdf[lo].primary_energy_eV);
    amrex::Real const x1 = std::log(m_moller_cdf[hi].primary_energy_eV);
    amrex::Real const t = (std::log(primary_energy_eV) - x0) / (x1 - x0);
    return std::exp(std::log(y0) + t * (std::log(y1) - std::log(y0)));
}

amrex::Real RreaInteractionTables::ElectronSoftRadiativeDrag(
    amrex::Real energy_eV,
    amrex::Real density_ratio) const
{
    if (density_ratio <= amrex::Real(0.0)) {
        amrex::Abort("density_ratio must be positive");
    }
    return Require(RreaTable::ElectronSoftRadiativeStopping).Interpolate(energy_eV)
        * density_ratio;
}

amrex::Real RreaInteractionTables::ElectronBremsstrahlungMeanFreePath(
    amrex::Real energy_eV,
    amrex::Real density_ratio) const
{
    if (density_ratio <= amrex::Real(0.0)) {
        amrex::Abort("density_ratio must be positive");
    }
    return mean_free_path_from_inverse_length(
        Require(RreaTable::ElectronTrackedBremsInverseLength).Interpolate(energy_eV)
            * density_ratio);
}

amrex::Real RreaInteractionTables::ElectronTrackedBremsEnergyLoss(
    amrex::Real energy_eV,
    amrex::Real density_ratio) const
{
    if (density_ratio <= amrex::Real(0.0)) {
        amrex::Abort("density_ratio must be positive");
    }
    auto const it = m_conditional_cdfs.find("brems_photon_energy_cdf_air");
    if (it == m_conditional_cdfs.end()) {
        amrex::Abort("brems_photon_energy_cdf_air is not loaded");
    }
    amrex::Real const inverse_length =
        Require(RreaTable::ElectronTrackedBremsInverseLength).Interpolate(energy_eV)
        * density_ratio;
    if (!(inverse_length > amrex::Real(0.0))) {
        return amrex::Real(0.0);
    }
    return inverse_length * mean_conditional_cdf(it->second, energy_eV);
}

amrex::Real RreaInteractionTables::SampleBremsstrahlungPhotonEnergy(
    amrex::Real primary_energy_eV,
    amrex::Real cdf_u) const
{
    RequireSchema6SamplerEnergy(
        "SampleBremsstrahlungPhotonEnergy",
        "electron",
        m_charged_certified_range,
        primary_energy_eV);
    auto const it = m_conditional_cdfs.find("brems_photon_energy_cdf_air");
    if (it == m_conditional_cdfs.end()) {
        amrex::Abort("brems_photon_energy_cdf_air is not loaded");
    }
    return sample_conditional_cdf(it->second, primary_energy_eV, cdf_u, true);
}

amrex::Real RreaInteractionTables::SampleBremsstrahlungCosTheta(
    amrex::Real primary_energy_eV,
    amrex::Real cdf_u) const
{
    RequireSchema6SamplerEnergy(
        "SampleBremsstrahlungCosTheta",
        "charged-lepton",
        m_charged_certified_range,
        primary_energy_eV);
    // Bounded ModifiedTsai (Geant4 10.7.4 G4ModifiedTsai::SampleCosTheta;
    // Urban, Geant3 Phys211, derived from Tsai, Rev. Mod. Phys. 49, 421).
    // F(u) = .25 G(a*u) + .75 G(3*a*u), G(x)=1-(1+x)exp(-x), a=.625.
    // Condition the whole mixture on u <= 2*gamma, then use
    // cos(theta)=1-2*(u/(2*gamma))^2. Inverting p*F(2*gamma) is equivalent
    // to Geant4's rejection of out-of-support draws, with one uniform input.
    cdf_u = std::max(amrex::Real(0.0), std::min(amrex::Real(1.0), cdf_u));
    if (cdf_u == amrex::Real(0.0)) { return amrex::Real(1.0); }
    if (cdf_u == amrex::Real(1.0)) { return amrex::Real(-1.0); }
    amrex::Real constexpr tsai_a = amrex::Real(0.625);
    auto gamma2_cdf = [](amrex::Real x) {
        if (x < amrex::Real(1.0e-3)) {
            return x*x * (amrex::Real(0.5) + x * (amrex::Real(-1.0/3.0)
                + x * (amrex::Real(1.0/8.0) + x * (amrex::Real(-1.0/30.0)
                + x * amrex::Real(1.0/144.0)))));
        }
        return -std::expm1(-x) - x * std::exp(-x);
    };
    auto mixture_cdf = [&](amrex::Real u) {
        return amrex::Real(0.25) * gamma2_cdf(tsai_a * u)
            + amrex::Real(0.75) * gamma2_cdf(amrex::Real(3.0) * tsai_a * u);
    };
    amrex::Real const u_max = amrex::Real(2.0)
        * (amrex::Real(1.0) + primary_energy_eV / kElectronRestEnergyEv);
    amrex::Real const target = cdf_u * mixture_cdf(u_max);
    amrex::Real lo = amrex::Real(0.0);
    amrex::Real hi = u_max;
    for (int iter = 0; iter < 60; ++iter) {
        amrex::Real const mid = amrex::Real(0.5) * (lo + hi);
        if (mixture_cdf(mid) < target) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    amrex::Real const fraction = amrex::Real(0.5) * (lo + hi) / u_max;
    return amrex::Real(1.0) - amrex::Real(2.0) * fraction * fraction;
}

amrex::Real RreaInteractionTables::PhotonFeedbackMeanFreePath(
    amrex::Real energy_eV,
    amrex::Real density_ratio) const
{
    if (density_ratio <= amrex::Real(0.0)) {
        amrex::Abort("density_ratio must be positive");
    }
    return mean_free_path_from_inverse_length(
        PhotonProcessInverseLengths(energy_eV, density_ratio).TotalPerM());
}

// ===== Photon channel selection =====
// Partitions ONE uniform draw by the same per-channel inverse lengths that
// form the total photon MFP (photoelectric / Compton / pair nuclear / pair
// triplet), so selection and hazard rate cannot disagree; the density ratio
// cancels in the branching fractions, which is why none is passed here.
RreaPhotonChannel RreaInteractionTables::SamplePhotonChannel(
    amrex::Real energy_eV,
    amrex::Real cdf_u) const
{
    RequireSchema6SamplerEnergy(
        "SamplePhotonChannel", "photon", m_photon_certified_range, energy_eV);
    cdf_u = std::max(amrex::Real(0.0), std::min(amrex::Real(1.0), cdf_u));
    auto const rates = PhotonProcessInverseLengths(
        energy_eV, amrex::Real(1.0));
    amrex::Real const total = rates.TotalPerM();
    if (!(total > amrex::Real(0.0))
        || !std::isfinite(static_cast<double>(total))) {
        amrex::Abort("schema-6 photon process inverse lengths sum to zero/non-finite");
    }
    amrex::Real const target = cdf_u * total;
    if (target < rates.compton_per_m) {
        return RreaPhotonChannel::Compton;
    }
    if (target < rates.compton_per_m + rates.photoelectric_per_m) {
        return RreaPhotonChannel::Photoelectric;
    }
    if (target < rates.compton_per_m + rates.photoelectric_per_m
            + rates.pair_nuclear_per_m) {
        return RreaPhotonChannel::PairNuclear;
    }
    return RreaPhotonChannel::PairTriplet;
}

RreaComptonFinalState RreaInteractionTables::SampleComptonKleinNishinaFinalState(
    amrex::Real primary_energy_eV,
    amrex::Real draw_mu,
    amrex::Real draw_accept) const
{
    RequireSchema6SamplerEnergy(
        "SampleComptonKleinNishinaFinalState",
        "photon",
        m_photon_certified_range,
        primary_energy_eV);
    if (primary_energy_eV <= amrex::Real(0.0)) {
        amrex::Abort("Compton sampler requires positive photon energy");
    }
    // Exact free-electron Klein-Nishina final state via the Butcher-Messel
    // rejection scheme used by G4KleinNishinaCompton.
    //
    // The runtime hands this sampler two coupled uniforms (channels
    // ComptonRecoil=53, ComptonScatter=54).  The accept/reject loop needs
    // more: round 0 consumes the two pre-drawn uniforms directly, while the
    // accept draw and later rounds use a deterministic stream seeded from them.
    //
    // Deliberate rate/final-state mismatch: photon_compton_inverse_length_air
    // carries Geant4's incoherent scattering function (-4.4% vs free KN at
    // 50 keV). MFP and channel probabilities retain that correction, but the
    // sampled energy-angle distribution assumes a free stationary electron.
    // Per-event energy is conserved; bound-electron kinematics are omitted.
    draw_mu = std::max(amrex::Real(0.0), std::min(amrex::Real(1.0), draw_mu));
    draw_accept = std::max(amrex::Real(0.0), std::min(amrex::Real(1.0), draw_accept));
    RreaUniformStream stream(static_cast<double>(draw_mu), static_cast<double>(draw_accept));

    amrex::Real const alpha = primary_energy_eV / kElectronRestEnergyEv;  // E0 / mc^2
    amrex::Real const eps0 = amrex::Real(1.0) / (amrex::Real(1.0) + amrex::Real(2.0) * alpha);
    amrex::Real const eps0sq = eps0 * eps0;
    amrex::Real const a1 = -std::log(eps0);                       // ln(1 + 2 alpha)
    amrex::Real const a2 = amrex::Real(0.5) * (amrex::Real(1.0) - eps0sq);
    amrex::Real epsilon = eps0;   // scattered / incident photon energy ratio
    amrex::Real onecost = amrex::Real(0.0);
    int constexpr max_iter = 128;
    for (int iter = 0; iter < max_iter; ++iter) {
        amrex::Real const u1 = (iter == 0) ? draw_mu : static_cast<amrex::Real>(stream.Next());
        amrex::Real const u2 = (iter == 0) ? draw_accept : static_cast<amrex::Real>(stream.Next());
        amrex::Real epsilonsq;
        if (a1 > (a1 + a2) * u1) {
            epsilon = std::exp(-a1 * u2);          // 1/eps sampled ~ 1/eps  (eps in [eps0,1])
            epsilonsq = epsilon * epsilon;
        } else {
            epsilonsq = eps0sq + (amrex::Real(1.0) - eps0sq) * u2;  // eps sampled ~ eps
            epsilon = std::sqrt(epsilonsq);
        }
        onecost = (amrex::Real(1.0) - epsilon) / (epsilon * alpha);  // 1 - cos(theta)
        amrex::Real const sint2 = onecost * (amrex::Real(2.0) - onecost);
        amrex::Real const greject =
            amrex::Real(1.0) - epsilon * sint2 / (amrex::Real(1.0) + epsilonsq);
        if (greject >= static_cast<amrex::Real>(stream.Next())) {
            break;
        }
    }
    amrex::Real mu = amrex::Real(1.0) - onecost;
    mu = std::max(amrex::Real(-1.0), std::min(amrex::Real(1.0), mu));
    amrex::Real const scattered = epsilon * primary_energy_eV;
    RreaComptonFinalState state;
    state.scattered_photon_energy_eV = std::max(amrex::Real(0.0), scattered);
    state.recoil_electron_energy_eV = std::max(amrex::Real(0.0), primary_energy_eV - state.scattered_photon_energy_eV);
    state.cos_theta = mu;
    return state;
}

// ===== Photoelectric final state =====
// Subshell selection by tabulated per-shell probabilities (summing to one at
// load), photoelectron energy = E_gamma - E_binding of the selected shell,
// with the residual binding booked through atomic relaxation; the emission
// angle is Sauter-Gavrila (its own documented sampler below).
RreaPhotoelectricFinalState RreaInteractionTables::SamplePhotoelectricFinalState(
    amrex::Real photon_energy_eV,
    amrex::Real cdf_u) const
{
    RequireSchema6SamplerEnergy(
        "SamplePhotoelectricFinalState",
        "photon",
        m_photon_certified_range,
        photon_energy_eV);
    if (photon_energy_eV <= amrex::Real(0.0)) {
        amrex::Abort("Photoelectric sampler requires positive photon energy");
    }
    if (m_photoelectric_subshell_cdf.empty()) {
        amrex::Abort("Schema-6 photoelectric subshell CDF table is not loaded");
    }
    if (photon_energy_eV < m_photoelectric_subshell_cdf.front().primary_energy_eV
        || photon_energy_eV > m_photoelectric_subshell_cdf.back().primary_energy_eV) {
        amrex::Abort("Photoelectric subshell CDF primary energy out of range");
    }
    cdf_u = std::max(amrex::Real(0.0), std::min(amrex::Real(1.0), cdf_u));
    auto const upper = std::lower_bound(
        m_photoelectric_subshell_cdf.begin(),
        m_photoelectric_subshell_cdf.end(),
        photon_energy_eV,
        [](RreaPhotoelectricSubshellCdfGroup const& group, amrex::Real energy) {
            return group.primary_energy_eV < energy;
        });
    RreaPhotoelectricSubshellCdfGroup const* group = nullptr;
    if (upper == m_photoelectric_subshell_cdf.begin()) {
        group = &(*upper);
    } else if (upper == m_photoelectric_subshell_cdf.end()) {
        group = &m_photoelectric_subshell_cdf.back();
    } else if (upper->primary_energy_eV == photon_energy_eV) {
        // At duplicated XCOM edge energies this intentionally picks the low-side
        // branch. Queries just above the edge use the high-side duplicate through
        // the previous-group rule below.
        group = &(*upper);
    } else {
        group = &(*(upper - 1));
    }
    if (group == nullptr || group->rows.empty()) {
        amrex::Abort("Photoelectric subshell CDF group is malformed");
    }
    auto const row_it = std::lower_bound(
        group->rows.begin(),
        group->rows.end(),
        cdf_u,
        [](RreaPhotoelectricSubshellCdfRow const& row, amrex::Real u) {
            return row.cdf_u < u;
        });
    RreaPhotoelectricSubshellCdfRow const& row =
        row_it == group->rows.end() ? group->rows.back() : *row_it;
    if (row.binding_energy_eV > photon_energy_eV) {
        amrex::Abort("Photoelectric shell binding energy exceeds photon energy");
    }
    RreaPhotoelectricFinalState state;
    state.binding_energy_eV = row.binding_energy_eV;
    state.relaxation_energy_eV = row.relaxation_energy_eV;
    state.photoelectron_energy_eV = std::max(amrex::Real(0.0), photon_energy_eV - row.binding_energy_eV);
    state.element_Z = row.element_Z;
    state.element_symbol = row.element_symbol;
    state.shell_id = row.shell_id;
    return state;
}

amrex::Real RreaInteractionTables::PhotoelectricElectronEnergy(
    amrex::Real photon_energy_eV) const
{
    return SamplePhotoelectricFinalState(photon_energy_eV, amrex::Real(0.5)).photoelectron_energy_eV;
}

amrex::Real RreaInteractionTables::SamplePhotoelectronCosTheta(
    amrex::Real photoelectron_energy_eV,
    amrex::Real draw_polar) const
{
    RequireSchema6SamplerEnergy(
        "SamplePhotoelectronCosTheta",
        "electron",
        m_charged_certified_range,
        photoelectron_energy_eV,
        /*allow_exact_zero=*/true);
    // Sample the polar cos(theta) of the photoelectron relative to the incident
    // photon momentum from the relativistic Sauter-Gavrila K-shell distribution
    // (the model G4LivermorePhotoElectricModel / SauterGavrila uses in option4):
    //   dsigma/dmu ~ (1 - mu^2) / (1 - beta mu)^4 * [1 + b (1 - beta mu)]
    // with mu = cos(theta) and b = 0.5 gamma (gamma-1)(gamma-2).  In
    // t = 1-beta*mu, the target density (apart from a positive constant) is
    //
    //   C t^-4 + (2+bC) t^-3 + (2b-1) t^-2 - b t^-1,
    //   C = beta^2-1.
    //
    // Its analytic primitive is monotone on [1-beta,1+beta], so a bracketed
    // inversion consumes exactly one PhotoelectricAngle uniform and avoids
    // bounded-rejection bias at high energy.
    long double const u = static_cast<long double>(clamp_unit(draw_polar));
    if (u <= 0.0L) {
        return amrex::Real(-1.0);
    }
    if (u >= 1.0L) {
        return amrex::Real(1.0);
    }

    long double const gamma = 1.0L
        + static_cast<long double>(photoelectron_energy_eV)
            / static_cast<long double>(kElectronRestEnergyEv);
    long double const inv_gamma_sq = 1.0L / (gamma * gamma);
    long double const beta = std::sqrt(std::max(0.0L, 1.0L - inv_gamma_sq));

    if (beta < 1.0e-4L) {
        // The exact beta->0 limit is f(mu) = 3/4 (1-mu^2), with
        // F(mu) = (2+3mu-mu^3)/4. Invert it monotonically to preserve that
        // limiting distribution.
        long double lo = -1.0L;
        long double hi = 1.0L;
        for (int iter = 0; iter < 96; ++iter) {
            long double const mid = 0.5L * (lo + hi);
            if (mid == lo || mid == hi) {
                break;
            }
            long double const cdf = (2.0L + mid * (3.0L - mid * mid)) / 4.0L;
            if (cdf < u) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        return static_cast<amrex::Real>(0.5L * (lo + hi));
    }

    long double const b = 0.5L * gamma * (gamma - 1.0L) * (gamma - 2.0L);
    long double const c = -inv_gamma_sq;
    // Compute 1-beta without subtractive cancellation; this remains accurate
    // at the certified 10 GeV endpoint even when long double aliases double.
    long double const t_lo = inv_gamma_sq / (1.0L + beta);  // mu = +1
    long double const t_hi = 1.0L + beta;                   // mu = -1
    auto const primitive = [c, b](long double t) {
        return -c / (3.0L * t * t * t)
            - (2.0L + b * c) / (2.0L * t * t)
            - (2.0L * b - 1.0L) / t
            - b * std::log(t);
    };

    long double const f_lo = primitive(t_lo);
    long double const f_hi = primitive(t_hi);
    long double const total = f_hi - f_lo;
    if (!(total > 0.0L) || !std::isfinite(total)) {
        amrex::Abort(
            "Sauter-Gavrila photoelectron inverse CDF has a non-positive/non-finite normalization");
    }
    long double const target = (1.0L - u) * f_hi + u * f_lo;
    long double lo = t_lo;
    long double hi = t_hi;
    for (int iter = 0; iter < 128; ++iter) {
        long double const mid = 0.5L * (lo + hi);
        if (mid == lo || mid == hi) {
            break;
        }
        if (primitive(mid) < target) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    long double const t = 0.5L * (lo + hi);
    long double const mu = std::max(-1.0L, std::min(1.0L, (1.0L - t) / beta));
    return static_cast<amrex::Real>(mu);
}

// ===== Pair-production final state =====
// This is a BANK REPLAY, not a distribution sampler: draw_invariant indexes
// the stored Geant4 final-state records and draw_pair_polar picks between
// adjacent energy blocks, so comparing its output to the bank validates the
// LOADER, not a sampled distribution.  The lepton polar/azimuth draws are
// deliberately unused (counter-based RNG: discarding them perturbs nothing);
// energy closure against the record is a hard abort at build time.
RreaPairFinalState RreaInteractionTables::SamplePairFinalState(
    amrex::Real photon_energy_eV,
    RreaPhotonChannel channel,
    amrex::Real draw_invariant,
    amrex::Real draw_pair_polar,
    amrex::Real draw_pair_azimuth,
    amrex::Real draw_lepton_polar,
    amrex::Real draw_lepton_azimuth) const
{
    RequireSchema6SamplerEnergy(
        "SamplePairFinalState", "photon", m_photon_certified_range, photon_energy_eV);
    RreaPairFinalState state;
    state.channel = channel;

    bool const triplet = channel == RreaPhotonChannel::PairTriplet;
    bool const nuclear = channel == RreaPhotonChannel::PairNuclear;
    if (!triplet && !nuclear) {
        amrex::Abort("SamplePairFinalState called for a non-pair photon channel");
    }

    if (!RreaPairChannelHasPhaseSpace(photon_energy_eV, channel)) {
        return state;
    }

    if (m_pair_empirical_blocks.empty()) {
        amrex::Abort("schema-6 correlated pair final-state samples are not loaded");
    }
    // Blocks are stored under the split PairNuclear/PairTriplet channels.
    RreaPhotonChannel const block_channel = triplet
        ? RreaPhotonChannel::PairTriplet
        : RreaPhotonChannel::PairNuclear;
    amrex::Vector<RreaPairEmpiricalBlock const*> channel_blocks;
    for (auto const& block : m_pair_empirical_blocks) {
        if (block.channel == block_channel) {
            channel_blocks.push_back(&block);
        }
    }
    if (channel_blocks.empty()) {
        amrex::Abort("schema-6 correlated pair channel has no training blocks");
    }
    RreaPairEmpiricalBlock const* selected = channel_blocks.front();
    if (photon_energy_eV >= channel_blocks.back()->primary_energy_eV) {
        selected = channel_blocks.back();
    } else if (photon_energy_eV > channel_blocks.front()->primary_energy_eV) {
        auto const upper = std::lower_bound(
            channel_blocks.begin(), channel_blocks.end(), photon_energy_eV,
            [](RreaPairEmpiricalBlock const* block, amrex::Real energy) {
                return block->primary_energy_eV < energy;
            });
        if ((*upper)->primary_energy_eV == photon_energy_eV) {
            selected = *upper;
        } else {
            auto const lower = *(upper - 1);
            amrex::Real const t =
                (std::log(photon_energy_eV) - std::log(lower->primary_energy_eV))
                / (std::log((*upper)->primary_energy_eV)
                    - std::log(lower->primary_energy_eV));
            selected = clamp_unit(draw_pair_polar) < t ? *upper : lower;
        }
    }
    if (selected->training_records.empty()) {
        amrex::Abort("schema-6 selected pair final-state block is empty");
    }
    std::size_t const training_size =
        static_cast<std::size_t>(selected->training_records.size());
    std::size_t const record_index = std::min(
        training_size - 1U,
        static_cast<std::size_t>(
            clamp_unit(draw_invariant)
            * static_cast<amrex::Real>(training_size)));
    auto const& record = selected->training_records[record_index];
    amrex::ignore_unused(draw_lepton_polar, draw_lepton_azimuth);
    return BuildRreaPairEmpiricalFinalState(
        record, photon_energy_eV, channel, draw_pair_azimuth);
}

amrex::Real RreaInteractionTables::PositronCollisionLoss(
    amrex::Real energy_eV,
    amrex::Real density_ratio) const
{
    if (density_ratio <= amrex::Real(0.0)) {
        amrex::Abort("density_ratio must be positive");
    }
    return Require(RreaTable::PositronCollisionStopping).Interpolate(energy_eV)
        * density_ratio;
}

amrex::Real RreaInteractionTables::PositronSoftRadiativeDrag(
    amrex::Real energy_eV,
    amrex::Real density_ratio) const
{
    if (density_ratio <= amrex::Real(0.0)) {
        amrex::Abort("density_ratio must be positive");
    }
    return Require(RreaTable::PositronSoftRadiativeStopping).Interpolate(energy_eV)
        * density_ratio;
}

amrex::Real RreaInteractionTables::PositronBremsstrahlungMeanFreePath(
    amrex::Real energy_eV,
    amrex::Real density_ratio) const
{
    if (density_ratio <= amrex::Real(0.0)) {
        amrex::Abort("density_ratio must be positive");
    }
    return mean_free_path_from_inverse_length(
        Require(RreaTable::PositronTrackedBremsInverseLength).Interpolate(energy_eV)
            * density_ratio);
}

amrex::Real RreaInteractionTables::PositronHardBhabhaMeanFreePath(
    amrex::Real energy_eV,
    amrex::Real density_ratio) const
{
    if (density_ratio <= amrex::Real(0.0)) {
        amrex::Abort("density_ratio must be positive");
    }
    amrex::Real const inverse_m_stp =
        Require(RreaTable::PositronHardBhabhaInverseLength).Interpolate(energy_eV);
    if (m_hard_moller_secondary_threshold_eV > amrex::Real(0.0)
        && energy_eV <= m_hard_moller_secondary_threshold_eV) {
        return amrex::Real(1.0e99);
    }
    return mean_free_path_from_inverse_length(
        inverse_m_stp * density_ratio);
}

amrex::Real RreaInteractionTables::SamplePositronBhabhaSecondaryEnergy(
    amrex::Real primary_energy_eV,
    amrex::Real cdf_u) const
{
    RequireSchema6SamplerEnergy(
        "SamplePositronBhabhaSecondaryEnergy",
        "positron",
        m_charged_certified_range,
        primary_energy_eV);
    auto const it = m_conditional_cdfs.find("positron_hard_bhabha_secondary_cdf_air");
    if (it == m_conditional_cdfs.end()) {
        amrex::Abort("positron_hard_bhabha_secondary_cdf_air is not loaded");
    }
    return sample_conditional_cdf(it->second, primary_energy_eV, cdf_u, true);
}

amrex::Real RreaInteractionTables::PositronAnnihilationMeanFreePath(
    amrex::Real energy_eV,
    amrex::Real density_ratio) const
{
    if (density_ratio <= amrex::Real(0.0)) {
        amrex::Abort("density_ratio must be positive");
    }
    return mean_free_path_from_inverse_length(
        Require(RreaTable::PositronAnnihilationInverseLength).Interpolate(energy_eV)
            * density_ratio);
}

RreaPositronAnnihilationFinalState RreaInteractionTables::SamplePositronAnnihilationFinalState(
    amrex::Real positron_kinetic_energy_eV,
    amrex::Real draw_energy,
    amrex::Real draw_azimuth) const
{
    RequireSchema6SamplerEnergy(
        "SamplePositronAnnihilationFinalState",
        "positron",
        m_charged_certified_range,
        positron_kinetic_energy_eV,
        /*allow_exact_zero=*/true);
    RreaPositronAnnihilationFinalState state;
    positron_kinetic_energy_eV = std::max(amrex::Real(0.0), positron_kinetic_energy_eV);
    amrex::Real const gamma = amrex::Real(1.0) + positron_kinetic_energy_eV / kElectronRestEnergyEv;
    amrex::Real const total_photon_energy_eV =
        amrex::Real(2.0) * kElectronRestEnergyEv + positron_kinetic_energy_eV;
    if (gamma <= amrex::Real(1.0 + 1.0e-10)) {
        state.photon1_energy_eV = kElectronRestEnergyEv;
        state.photon2_energy_eV = kElectronRestEnergyEv;
        state.photon1_cos_theta = amrex::Real(2.0) * clamp_unit(draw_energy) - amrex::Real(1.0);
        state.photon1_azimuth_u = wrap_unit_interval(draw_azimuth);
        state.valid = true;
        state.at_rest = true;
        return state;
    }
    amrex::Real const beta = std::sqrt(std::max(
        amrex::Real(0.0),
        amrex::Real(1.0) - amrex::Real(1.0) / (gamma * gamma)));
    amrex::Real const epsilon = sample_annihilation_photon_energy_fraction(gamma, draw_energy);
    state.photon1_energy_eV = epsilon * total_photon_energy_eV;
    state.photon2_energy_eV = std::max(amrex::Real(0.0), total_photon_energy_eV - state.photon1_energy_eV);
    if (state.photon1_energy_eV <= amrex::Real(0.0) || state.photon2_energy_eV <= amrex::Real(0.0)) {
        return state;
    }
    amrex::Real const denom = epsilon * gamma * beta;
    if (denom <= amrex::Real(0.0)) {
        return state;
    }
    state.photon1_cos_theta = std::max(
        amrex::Real(-1.0),
        std::min(amrex::Real(1.0), (epsilon * (gamma + amrex::Real(1.0)) - amrex::Real(1.0)) / denom));
    state.photon1_azimuth_u = wrap_unit_interval(draw_azimuth);
    state.valid = true;
    state.at_rest = false;
    return state;
}

amrex::Real RreaInteractionTables::PositronElasticTransportMeanFreePath(
    amrex::Real energy_eV,
    amrex::Real density_ratio) const
{
    if (density_ratio <= amrex::Real(0.0)) {
        amrex::Abort("density_ratio must be positive");
    }
    return mean_free_path_from_inverse_length(
        PositronTransportCoefficients(energy_eV, density_ratio)
            .elastic_transport_inverse_m);
}

LowEnergyElectronClosure LowEnergyElectronClosure::Load(
    std::string const& table_path)
{
    LowEnergyElectronClosure closure;
    closure.m_model_id = "dry_air_swarm_hybrid_v3_flux_attachment_diffusion";
    closure.m_interpolation_id =
        "piecewise_log_linear_positive_with_nonnegative_onset";

    constexpr char const* expected_header =
        "en_td,flux_drift_velocity_m_per_s,"
        "reduced_flux_mobility_ref_m2_per_v_s,"
        "longitudinal_diffusion_ref_m2_per_s,"
        "transverse_diffusion_ref_m2_per_s,eta_over_n2_cm5,"
        "three_body_rate_cm6_per_s,three_body_frequency_ref_s,"
        "eta_over_n_cm2,two_body_rate_cm3_per_s,"
        "two_body_frequency_ref_s,relative_uncertainty_mobility,"
        "relative_uncertainty_three_body,relative_uncertainty_two_body,"
        "source_row";
    std::istringstream csv(file_text(table_path));
    std::string line;
    while (std::getline(csv, line)) {
        std::string const cleaned = trim(line);
        if (!cleaned.empty() && cleaned.front() != '#') {
            break;
        }
    }
    if (trim(line) != expected_header) {
        throw std::runtime_error(
            "electron closure table header must exactly match schema 3");
    }
    double const n0_cm3 = n_loschmidt_m3 * 1.0e-6;
    while (std::getline(csv, line)) {
        if (trim(line).empty() || trim(line).front() == '#') {
            continue;
        }
        auto const cells = split_csv_line(line);
        if (cells.size() != 15U) {
            throw std::runtime_error(
                "electron closure table row must have exactly 15 columns");
        }
        double const en_td = parse_number<double>(cells[0], "en_td");
        double const w_m_s = parse_number<double>(
            cells[1], "flux_drift_velocity_m_per_s");
        double const k0 = parse_number<double>(
            cells[2], "reduced_flux_mobility_ref_m2_per_v_s");
        double const longitudinal_diffusion_ref = parse_number<double>(
            cells[3], "longitudinal_diffusion_ref_m2_per_s");
        double const transverse_diffusion_ref = parse_number<double>(
            cells[4], "transverse_diffusion_ref_m2_per_s");
        double const eta_cm5 =
            parse_number<double>(cells[5], "eta_over_n2_cm5");
        double const k3 = parse_number<double>(
            cells[6], "three_body_rate_cm6_per_s");
        double const nu3_ref = parse_number<double>(
            cells[7], "three_body_frequency_ref_s");
        double const eta2_cm2 = parse_number<double>(
            cells[8], "eta_over_n_cm2");
        double const k2 = parse_number<double>(
            cells[9], "two_body_rate_cm3_per_s");
        double const nu2_ref = parse_number<double>(
            cells[10], "two_body_frequency_ref_s");
        for (double value : {
                 en_td,
                 w_m_s,
                 k0,
                 longitudinal_diffusion_ref,
                 transverse_diffusion_ref,
                 eta_cm5,
                 k3,
                 nu3_ref}) {
            if (!std::isfinite(value) || !(value > 0.0)) {
                throw std::runtime_error(
                    "electron closure table values must be finite and "
                    "positive: " + line);
            }
        }
        for (double value : {eta2_cm2, k2, nu2_ref}) {
            if (!std::isfinite(value) || value < 0.0) {
                throw std::runtime_error(
                    "electron closure two-body values must be finite and "
                    "non-negative: " + line);
            }
        }
        if ((eta2_cm2 == 0.0) != (k2 == 0.0)
            || (k2 == 0.0) != (nu2_ref == 0.0)) {
            throw std::runtime_error(
                "electron closure two-body columns must become positive together");
        }
        if (!closure.m_en_td.empty() && !(en_td > closure.m_en_td.back())) {
            throw std::runtime_error(
                "electron closure E/N nodes must be strictly increasing");
        }
        if (cells[14].empty()) {
            throw std::runtime_error(
                "electron closure table rows must carry a source_row id");
        }
        // Re-derive the physics chain so a hand-edited coefficient cannot
        // load: k2 = W_F*eta/N, nu2_ref = k2*N0,
        // k3 = W_F*eta/N^2, nu3_ref = k3*N0^2, K0 = W_F/(E/N*N0).
        double const k2_expected = (w_m_s * 100.0) * eta2_cm2;
        double const nu2_expected = k2_expected * n0_cm3;
        double const k3_expected = (w_m_s * 100.0) * eta_cm5;
        double const nu3_expected = k3_expected * n0_cm3 * n0_cm3;
        double const k0_expected =
            w_m_s / (en_td * townsend_v_m2 * n_loschmidt_m3);
        if (std::abs(k2 - k2_expected) > 1.0e-9 * std::max(k2_expected, 1.0e-300)
            || std::abs(nu2_ref - nu2_expected)
                > 1.0e-9 * std::max(nu2_expected, 1.0e-300)
            || std::abs(k3 - k3_expected) > 1.0e-9 * k3_expected
            || std::abs(nu3_ref - nu3_expected) > 1.0e-9 * nu3_expected
            || std::abs(k0 - k0_expected) > 1.0e-9 * k0_expected) {
            throw std::runtime_error(
                "electron closure row fails its derivation identity "
                "(two-/three-body rates, frequencies, and K0): "
                + line);
        }
        closure.m_en_td.push_back(en_td);
        closure.m_log_en_td.push_back(std::log(en_td));
        closure.m_k0_flux_ref.push_back(k0);
        closure.m_log_k0_flux_ref.push_back(std::log(k0));
        closure.m_nu2_ref.push_back(nu2_ref);
        closure.m_nu3_ref.push_back(nu3_ref);
        closure.m_log_nu3_ref.push_back(std::log(nu3_ref));
        closure.m_longitudinal_diffusion_ref.push_back(
            longitudinal_diffusion_ref);
        closure.m_log_longitudinal_diffusion_ref.push_back(
            std::log(longitudinal_diffusion_ref));
        closure.m_transverse_diffusion_ref.push_back(
            transverse_diffusion_ref);
        closure.m_log_transverse_diffusion_ref.push_back(
            std::log(transverse_diffusion_ref));
    }
    if (closure.m_en_td.size() < 2U) {
        throw std::runtime_error(
            "electron closure table needs at least two E/N nodes");
    }
    return closure;
}

RreaElectronClosureCoefficients LowEnergyElectronClosure::Evaluate(
    double en_td, RreaClosureRangeStatus& status) const noexcept
{
    RreaElectronClosureCoefficients out;
    if (m_en_td.empty() || !std::isfinite(en_td) || en_td < 0.0) {
        status = RreaClosureRangeStatus::Invalid;
        return out;
    }
    std::size_t node = m_en_td.size();
    if (en_td <= m_en_td.front()) {
        status = en_td < m_en_td.front()
            ? RreaClosureRangeStatus::HeldBelow
            : RreaClosureRangeStatus::InRange;
        node = 0;
    } else if (en_td >= m_en_td.back()) {
        status = en_td > m_en_td.back()
            ? RreaClosureRangeStatus::AboveRange
            : RreaClosureRangeStatus::InRange;
        node = m_en_td.size() - 1U;
    }
    if (node < m_en_td.size()) {
        out.k0_flux_ref_m2_per_vs =
            static_cast<amrex::Real>(m_k0_flux_ref[node]);
        out.nu2_ref_per_s = static_cast<amrex::Real>(m_nu2_ref[node]);
        out.nu3_ref_per_s = static_cast<amrex::Real>(m_nu3_ref[node]);
        out.longitudinal_diffusion_ref_m2_per_s = static_cast<amrex::Real>(
            m_longitudinal_diffusion_ref[node]);
        out.transverse_diffusion_ref_m2_per_s = static_cast<amrex::Real>(
            m_transverse_diffusion_ref[node]);
        return out;
    }
    status = RreaClosureRangeStatus::InRange;
    std::size_t const hi = static_cast<std::size_t>(
        std::upper_bound(m_en_td.begin(), m_en_td.end(), en_td)
        - m_en_td.begin());
    std::size_t const lo = hi - 1U;
    if (en_td == m_en_td[lo]) {
        // Exact interior node: return the stored value, not exp(log(value)).
        out.k0_flux_ref_m2_per_vs =
            static_cast<amrex::Real>(m_k0_flux_ref[lo]);
        out.nu2_ref_per_s = static_cast<amrex::Real>(m_nu2_ref[lo]);
        out.nu3_ref_per_s = static_cast<amrex::Real>(m_nu3_ref[lo]);
        out.longitudinal_diffusion_ref_m2_per_s = static_cast<amrex::Real>(
            m_longitudinal_diffusion_ref[lo]);
        out.transverse_diffusion_ref_m2_per_s = static_cast<amrex::Real>(
            m_transverse_diffusion_ref[lo]);
        return out;
    }
    double const f = (std::log(en_td) - m_log_en_td[lo])
        / (m_log_en_td[hi] - m_log_en_td[lo]);
    out.k0_flux_ref_m2_per_vs = static_cast<amrex::Real>(std::exp(
        m_log_k0_flux_ref[lo]
        + f * (m_log_k0_flux_ref[hi] - m_log_k0_flux_ref[lo])));
    double const nu2_lo = m_nu2_ref[lo];
    double const nu2_hi = m_nu2_ref[hi];
    out.nu2_ref_per_s = static_cast<amrex::Real>(
        nu2_lo > 0.0 && nu2_hi > 0.0
            ? std::exp(std::log(nu2_lo) + f * std::log(nu2_hi / nu2_lo))
            : nu2_lo + f * (nu2_hi - nu2_lo));
    out.nu3_ref_per_s = static_cast<amrex::Real>(std::exp(
        m_log_nu3_ref[lo] + f * (m_log_nu3_ref[hi] - m_log_nu3_ref[lo])));
    out.longitudinal_diffusion_ref_m2_per_s = static_cast<amrex::Real>(
        std::exp(
            m_log_longitudinal_diffusion_ref[lo]
            + f * (m_log_longitudinal_diffusion_ref[hi]
                   - m_log_longitudinal_diffusion_ref[lo])));
    out.transverse_diffusion_ref_m2_per_s = static_cast<amrex::Real>(
        std::exp(
            m_log_transverse_diffusion_ref[lo]
            + f * (m_log_transverse_diffusion_ref[hi]
                   - m_log_transverse_diffusion_ref[lo])));
    return out;
}

RreaElectronClosureCoefficients LowEnergyElectronClosure::EvaluateHost(
    double en_td) const
{
    RreaClosureRangeStatus status = RreaClosureRangeStatus::Invalid;
    auto const out = Evaluate(en_td, status);
    if (status == RreaClosureRangeStatus::AboveRange) {
        std::ostringstream message;
        message << std::setprecision(17)
                << "electron closure evaluated above its valid range: "
                << en_td << " Td > " << MaxEnTd() << " Td";
        throw std::runtime_error(message.str());
    }
    if (status == RreaClosureRangeStatus::Invalid) {
        throw std::runtime_error(
            "electron closure evaluated on an invalid E/N input");
    }
    return out;
}

}  // namespace rrea
