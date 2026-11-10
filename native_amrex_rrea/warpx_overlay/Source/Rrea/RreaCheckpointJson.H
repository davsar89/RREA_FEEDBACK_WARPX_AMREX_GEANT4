#ifndef RREA_CHECKPOINT_JSON_H
#define RREA_CHECKPOINT_JSON_H

// Shared bounded scanners for the flat checkpoint-sidecar JSON written by the
// engine. They are deliberately not general JSON parsers: no nesting or object
// traversal is supported, and malformed or missing scalar values return an
// empty result that the checkpoint reader rejects. The native smoke pins the
// accepted flat syntax.

#include "rrea/RreaJsonTextUtil.H"

#include <cctype>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace rrea::warpx {

inline std::string json_escape(std::string const& value)
{
    std::string out;
    out.reserve(value.size());
    for (char c : value) {
        if (c == '\\' || c == '"') {
            out.push_back('\\');
        }
        out.push_back(c);
    }
    return out;
}

inline std::string json_string_value(
    std::string const& text, std::string const& key)
{
    try {
        return rrea::json_text::value(text, key, "RREA checkpoint metadata");
    } catch (std::runtime_error const&) {
        return {};
    }
}

}  // namespace rrea::warpx

#endif
