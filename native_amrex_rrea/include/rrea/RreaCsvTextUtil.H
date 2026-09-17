#ifndef RREA_CSV_TEXT_UTIL_H
#define RREA_CSV_TEXT_UTIL_H

// Shared CSV text convention for tables, profiles, schedules and diagnostics:
// cells are whitespace-trimmed, quoted commas are protected, and a doubled
// quote inside a quoted cell decodes to one literal quote.

#include <AMReX_Vector.H>

#include <algorithm>
#include <cctype>
#include <string>

namespace rrea::csv_text {

inline std::string trim(std::string value)
{
    auto const first = std::find_if_not(value.begin(), value.end(), [](unsigned char c) {
        return std::isspace(c) != 0;
    });
    auto const last = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char c) {
        return std::isspace(c) != 0;
    }).base();
    if (first >= last) {
        return {};
    }
    return std::string(first, last);
}

inline amrex::Vector<std::string> split_csv_line(std::string const& line)
{
    amrex::Vector<std::string> cells;
    std::string cell;
    bool in_quotes = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        char const c = line[i];
        if (c == '"') {
            if (in_quotes && i + 1 < line.size() && line[i + 1] == '"') {
                cell.push_back('"');
                ++i;
            } else {
                in_quotes = !in_quotes;
            }
        } else if (c == ',' && !in_quotes) {
            cells.push_back(trim(cell));
            cell.clear();
        } else {
            cell.push_back(c);
        }
    }
    cells.push_back(trim(cell));
    return cells;
}

}  // namespace rrea::csv_text

#endif  // RREA_CSV_TEXT_UTIL_H
