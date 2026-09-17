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
using namespace kinetic_tables;

constexpr char kOption4GsBranch[] = "goudsmit_saunderson_option4";
constexpr char kOption4GsModel[] = "G4GoudsmitSaundersonMscModel";
constexpr char kOption4WentzelBranch[] = "wentzel_vi_option4";
constexpr char kOption4WentzelModel[] = "G4WentzelVIModel";











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
    // amrex::Vector::size() is a signed amrex::Long; KineticSpan stores a
    // std::size_t. Narrowing inside aggregate init is a hard error for nvcc.
    auto const n = [](auto const& v) { return static_cast<std::size_t>(v.size()); };
    return RreaInteractionTableView{mode,
        {energy_eV.data(),n(energy_eV)}, {values.data(),n(values)},
        {log_energy.data(),n(log_energy)}, {log_values.data(),n(log_values)}}
        .Interpolate(energy_eV_query);
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
    BuildKineticStorage();
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
        // Spell the element type once: a braced-init-list range needs a
        // deducible type, and EDG will not take it from a leading explicit
        // entry the way GCC does.
        using ScaledColumn = std::pair<char const*, amrex::Real>;
        ScaledColumn const scaled_columns[] = {
            {"electron_collision_stopping_air",
             sensitivity.electron_collision_loss},
            {"electron_hard_moller_inverse_length_air",
             amrex::Real(1.0) / sensitivity.hard_moller_mfp},
            {"electron_soft_radiative_stopping_air",
             sensitivity.soft_radiative_drag},
            {"positron_soft_radiative_stopping_air",
             sensitivity.soft_radiative_drag}};
        for (auto const& [name, factor] : scaled_columns) {
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
    BuildKineticStorage();
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
    return View().RequireSchema6SamplerEnergy(sampler, species, certified_range, energy_eV, allow_exact_zero);
}

amrex::Real RreaInteractionTables::ProbeTableValue(
    std::string const& name, amrex::Real energy_eV) const
{
    return Require(name).Interpolate(energy_eV);
}

RreaPhotonProcessInverseLengths RreaInteractionTables::PhotonProcessInverseLengths(
    amrex::Real energy_eV, amrex::Real density_ratio) const
{
    return View().PhotonProcessInverseLengths(energy_eV, density_ratio);
}

RreaElectronTransportCoefficients RreaInteractionTables::ElectronTransportCoefficients(
    amrex::Real energy_eV, amrex::Real density_ratio) const
{
    return View().ElectronTransportCoefficients(energy_eV, density_ratio);
}

RreaPositronTransportCoefficients RreaInteractionTables::PositronTransportCoefficients(
    amrex::Real energy_eV, amrex::Real density_ratio) const
{
    return View().PositronTransportCoefficients(energy_eV, density_ratio);
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
    return View().ElectronCollisionLoss(energy_eV, density_ratio);
}

amrex::Real RreaInteractionTables::ElectronHardMollerMeanFreePath(
    amrex::Real energy_eV,
    amrex::Real density_ratio) const
{
    return View().ElectronHardMollerMeanFreePath(energy_eV, density_ratio);
}

amrex::Real RreaInteractionTables::ElectronHardMollerEnergyLoss(
    amrex::Real energy_eV,
    amrex::Real density_ratio) const
{
    return View().ElectronHardMollerEnergyLoss(energy_eV, density_ratio);
}

amrex::Real RreaInteractionTables::ElectronElasticTransportMeanFreePath(
    amrex::Real energy_eV,
    amrex::Real density_ratio) const
{
    return View().ElectronElasticTransportMeanFreePath(energy_eV, density_ratio);
}

// --- GS / Wentzel elastic runtime accessors ----------------------------------

amrex::Real RreaInteractionTables::ElasticMeanFreePath(
    bool is_positron, amrex::Real energy_eV, amrex::Real density_ratio) const
{
    return View().ElasticMeanFreePath(is_positron, energy_eV, density_ratio);
}

amrex::Real RreaInteractionTables::ScreeningEta(bool is_positron, amrex::Real energy_eV) const
{
    return View().ScreeningEta(is_positron, energy_eV);
}

amrex::Real RreaInteractionTables::SampleGsCosTheta(
    bool is_positron, amrex::Real energy_eV, amrex::Real path_n, amrex::Real cdf_u) const
{
    return View().SampleGsCosTheta(is_positron, energy_eV, path_n, cdf_u);
}

amrex::Real RreaInteractionTables::SampleSingleElasticCosTheta(
    bool is_positron, amrex::Real energy_eV, amrex::Real cdf_u) const
{
    return View().SampleSingleElasticCosTheta(is_positron, energy_eV, cdf_u);
}

amrex::Real RreaInteractionTables::SamplePositronBremsstrahlungPhotonEnergy(
    amrex::Real primary_energy_eV, amrex::Real cdf_u) const
{
    return View().SamplePositronBremsstrahlungPhotonEnergy(primary_energy_eV, cdf_u);
}

amrex::Real RreaInteractionTables::SampleMollerSecondaryEnergy(
    amrex::Real primary_energy_eV,
    amrex::Real cdf_u) const
{
    return View().SampleMollerSecondaryEnergy(primary_energy_eV, cdf_u);
}

amrex::Real RreaInteractionTables::ElectronSoftRadiativeDrag(
    amrex::Real energy_eV,
    amrex::Real density_ratio) const
{
    return View().ElectronSoftRadiativeDrag(energy_eV, density_ratio);
}

amrex::Real RreaInteractionTables::ElectronBremsstrahlungMeanFreePath(
    amrex::Real energy_eV,
    amrex::Real density_ratio) const
{
    return View().ElectronBremsstrahlungMeanFreePath(energy_eV, density_ratio);
}

amrex::Real RreaInteractionTables::ElectronTrackedBremsEnergyLoss(
    amrex::Real energy_eV,
    amrex::Real density_ratio) const
{
    return View().ElectronTrackedBremsEnergyLoss(energy_eV, density_ratio);
}

amrex::Real RreaInteractionTables::SampleBremsstrahlungPhotonEnergy(
    amrex::Real primary_energy_eV,
    amrex::Real cdf_u) const
{
    return View().SampleBremsstrahlungPhotonEnergy(primary_energy_eV, cdf_u);
}

amrex::Real RreaInteractionTables::SampleBremsstrahlungCosTheta(
    amrex::Real primary_energy_eV,
    amrex::Real cdf_u) const
{
    return View().SampleBremsstrahlungCosTheta(primary_energy_eV, cdf_u);
}

amrex::Real RreaInteractionTables::PhotonFeedbackMeanFreePath(
    amrex::Real energy_eV,
    amrex::Real density_ratio) const
{
    return View().PhotonFeedbackMeanFreePath(energy_eV, density_ratio);
}

RreaPhotonChannel RreaInteractionTables::SamplePhotonChannel(
    amrex::Real energy_eV,
    amrex::Real cdf_u) const
{
    return View().SamplePhotonChannel(energy_eV, cdf_u);
}

RreaComptonFinalState RreaInteractionTables::SampleComptonKleinNishinaFinalState(
    amrex::Real primary_energy_eV,
    amrex::Real draw_mu,
    amrex::Real draw_accept) const
{
    return View().SampleComptonKleinNishinaFinalState(primary_energy_eV, draw_mu, draw_accept);
}

RreaPhotoelectricFinalState RreaInteractionTables::SamplePhotoelectricFinalState(
    amrex::Real photon_energy_eV,
    amrex::Real cdf_u) const
{
    auto const value=View().SamplePhotoelectricFinalState(photon_energy_eV, cdf_u);
    auto const& row=m_photoelectric_subshell_cdf[value.group_index].rows[value.row_index];
    return {value.photoelectron_energy_eV,value.binding_energy_eV,value.relaxation_energy_eV,
            value.element_Z,row.element_symbol,row.shell_id};
}

amrex::Real RreaInteractionTables::PhotoelectricElectronEnergy(
    amrex::Real photon_energy_eV) const
{
    return View().PhotoelectricElectronEnergy(photon_energy_eV);
}

amrex::Real RreaInteractionTables::SamplePhotoelectronCosTheta(
    amrex::Real photoelectron_energy_eV,
    amrex::Real draw_polar) const
{
    return View().SamplePhotoelectronCosTheta(photoelectron_energy_eV, draw_polar);
}

RreaPairFinalState RreaInteractionTables::SamplePairFinalState(
    amrex::Real photon_energy_eV,
    RreaPhotonChannel channel,
    amrex::Real draw_invariant,
    amrex::Real draw_pair_polar,
    amrex::Real draw_pair_azimuth,
    amrex::Real draw_lepton_polar,
    amrex::Real draw_lepton_azimuth) const
{
    return View().SamplePairFinalState(photon_energy_eV, channel, draw_invariant, draw_pair_polar, draw_pair_azimuth, draw_lepton_polar, draw_lepton_azimuth);
}

amrex::Real RreaInteractionTables::PositronCollisionLoss(
    amrex::Real energy_eV,
    amrex::Real density_ratio) const
{
    return View().PositronCollisionLoss(energy_eV, density_ratio);
}

amrex::Real RreaInteractionTables::PositronSoftRadiativeDrag(
    amrex::Real energy_eV,
    amrex::Real density_ratio) const
{
    return View().PositronSoftRadiativeDrag(energy_eV, density_ratio);
}

amrex::Real RreaInteractionTables::PositronBremsstrahlungMeanFreePath(
    amrex::Real energy_eV,
    amrex::Real density_ratio) const
{
    return View().PositronBremsstrahlungMeanFreePath(energy_eV, density_ratio);
}

amrex::Real RreaInteractionTables::PositronHardBhabhaMeanFreePath(
    amrex::Real energy_eV,
    amrex::Real density_ratio) const
{
    return View().PositronHardBhabhaMeanFreePath(energy_eV, density_ratio);
}

amrex::Real RreaInteractionTables::SamplePositronBhabhaSecondaryEnergy(
    amrex::Real primary_energy_eV,
    amrex::Real cdf_u) const
{
    return View().SamplePositronBhabhaSecondaryEnergy(primary_energy_eV, cdf_u);
}

amrex::Real RreaInteractionTables::PositronAnnihilationMeanFreePath(
    amrex::Real energy_eV,
    amrex::Real density_ratio) const
{
    return View().PositronAnnihilationMeanFreePath(energy_eV, density_ratio);
}

RreaPositronAnnihilationFinalState RreaInteractionTables::SamplePositronAnnihilationFinalState(
    amrex::Real positron_kinetic_energy_eV,
    amrex::Real draw_energy,
    amrex::Real draw_azimuth) const
{
    return View().SamplePositronAnnihilationFinalState(positron_kinetic_energy_eV, draw_energy, draw_azimuth);
}

amrex::Real RreaInteractionTables::PositronElasticTransportMeanFreePath(
    amrex::Real energy_eV,
    amrex::Real density_ratio) const
{
    return View().PositronElasticTransportMeanFreePath(energy_eV, density_ratio);
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
    return View().Evaluate(en_td, status);
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


void RreaInteractionTables::BuildKineticStorage()
{
    m_kinetic_storage=RreaInteractionTableStorage{};
    auto& s=m_kinetic_storage; auto& v=s.view;
    v.m_loaded=m_loaded;
    v.m_hard_moller_secondary_threshold_eV=m_hard_moller_secondary_threshold_eV;
    v.m_brems_absolute_cutoff_eV=m_brems_absolute_cutoff_eV;
    v.m_reference_density_kg_m3=m_reference_density_kg_m3;
    v.m_charged_certified_range=m_charged_certified_range;
    v.m_photon_certified_range=m_photon_certified_range;
    v.m_gs_few_collision_threshold=m_gs_few_collision_threshold;
    v.m_gs_isotropic_above_n=m_gs_isotropic_above_n;
    v.m_gs_wentzel_transition_energy_eV=m_gs_wentzel_transition_energy_eV;
    v.m_debug=DebugTransportSensitivityOptions();
    v.m_hard_moller_mean_secondary_eV=s.Keep(m_hard_moller_mean_secondary_eV);
    std::size_t hot_index=0;
#define RREA_BUILD_VIEW(id,name) { auto const& t=Require(name); v.m_hot[hot_index++]={t.mode,s.Keep(t.energy_eV),s.Keep(t.values),s.Keep(t.log_energy),s.Keep(t.log_values)}; }
    RREA_HOT_TABLES(RREA_BUILD_VIEW)
#undef RREA_BUILD_VIEW
    s.moller.reserve(m_moller_cdf.size());
    for(auto const& g:m_moller_cdf) s.moller.push_back({g.primary_energy_eV,s.Keep(g.cdf_u),s.Keep(g.secondary_energy_eV)});
    v.m_moller_cdf={s.moller.data(),s.moller.size()};
    if(auto it=m_conditional_cdfs.find("brems_photon_energy_cdf_air");it!=m_conditional_cdfs.end()) {
        for(auto const& g:it->second) s.conditional[0].push_back({g.primary_energy_eV,s.Keep(g.cdf_u),s.Keep(g.values)});
    }
    v.m_conditional_cdfs[0]={s.conditional[0].data(),s.conditional[0].size()};
    if(auto it=m_conditional_cdfs.find("positron_brems_photon_energy_cdf_air");it!=m_conditional_cdfs.end()) {
        for(auto const& g:it->second) s.conditional[1].push_back({g.primary_energy_eV,s.Keep(g.cdf_u),s.Keep(g.values)});
    }
    v.m_conditional_cdfs[1]={s.conditional[1].data(),s.conditional[1].size()};
    if(auto it=m_conditional_cdfs.find("positron_hard_bhabha_secondary_cdf_air");it!=m_conditional_cdfs.end()) {
        for(auto const& g:it->second) s.conditional[2].push_back({g.primary_energy_eV,s.Keep(g.cdf_u),s.Keep(g.values)});
    }
    v.m_conditional_cdfs[2]={s.conditional[2].data(),s.conditional[2].size()};
    v.m_electron_elastic_mfp={s.Keep(m_electron_elastic_mfp.energy_eV),s.Keep(m_electron_elastic_mfp.elastic_mfp_m_stp),s.Keep(m_electron_elastic_mfp.screening_eta),s.Keep(m_electron_elastic_mfp.transport_mfp_m_stp)};
    v.m_positron_elastic_mfp={s.Keep(m_positron_elastic_mfp.energy_eV),s.Keep(m_positron_elastic_mfp.elastic_mfp_m_stp),s.Keep(m_positron_elastic_mfp.screening_eta),s.Keep(m_positron_elastic_mfp.transport_mfp_m_stp)};
    v.m_electron_gs_angle={s.Keep(m_electron_gs_angle.energy_eV),s.Keep(m_electron_gs_angle.path_n),s.Keep(m_electron_gs_angle.cdf_u),s.Keep(m_electron_gs_angle.one_minus_cos)};
    v.m_positron_gs_angle={s.Keep(m_positron_gs_angle.energy_eV),s.Keep(m_positron_gs_angle.path_n),s.Keep(m_positron_gs_angle.cdf_u),s.Keep(m_positron_gs_angle.one_minus_cos)};
    v.m_electron_wentzel_angle={s.Keep(m_electron_wentzel_angle.energy_eV),s.Keep(m_electron_wentzel_angle.path_n),s.Keep(m_electron_wentzel_angle.cdf_u),s.Keep(m_electron_wentzel_angle.one_minus_cos)};
    v.m_positron_wentzel_angle={s.Keep(m_positron_wentzel_angle.energy_eV),s.Keep(m_positron_wentzel_angle.path_n),s.Keep(m_positron_wentzel_angle.cdf_u),s.Keep(m_positron_wentzel_angle.one_minus_cos)};
    for(auto const& g:m_photoelectric_subshell_cdf) {
        s.photo_rows.emplace_back(); auto& rows=s.photo_rows.back();
        for(auto const& r:g.rows) rows.push_back({r.cdf_u,r.subshell_probability,r.element_Z,r.binding_energy_eV,r.relaxation_energy_eV});
        s.photo.push_back({g.primary_energy_eV,{rows.data(),rows.size()}});
    }
    v.m_photoelectric_subshell_cdf={s.photo.data(),s.photo.size()};
    for(auto const& g:m_pair_empirical_blocks) {
        s.pair_rows.emplace_back(g.training_records.begin(),g.training_records.end());
        auto const& rows=s.pair_rows.back();
        s.pair.push_back({g.channel,g.primary_energy_eV,{rows.data(),rows.size()}});
    }
    v.m_pair_empirical_blocks={s.pair.data(),s.pair.size()};
}

}  // namespace rrea
