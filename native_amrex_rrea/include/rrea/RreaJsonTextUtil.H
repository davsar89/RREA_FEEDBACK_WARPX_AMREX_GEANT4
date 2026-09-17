#ifndef RREA_JSON_TEXT_UTIL_H
#define RREA_JSON_TEXT_UTIL_H

// One reader for a JSON string value, shared by the transport configuration
// and the RREATBL container header.  Deliberately not a JSON parser: it finds
// "key" : "value" and decodes the escapes, which is the whole of what either
// caller needs.  Kept beside RreaCsvTextUtil.H for the same reason -- a second
// copy of this loop is a second convention that can disagree with the first.

#include <cstddef>
#include <stdexcept>
#include <string>

namespace rrea::json_text {

/*! Value of the first string-valued "key" in one JSON object text.
 *
 * Throws if the key is absent, unterminated, or not string-valued.
 */
[[nodiscard]] inline std::string value(
    std::string const& object,
    std::string const& key,
    char const* context = "JSON")
{
    std::string const quoted_key = "\"" + key + "\"";
    std::size_t pos = object.find(quoted_key);
    if (pos == std::string::npos) {
        throw std::runtime_error(
            std::string(context) + ": missing string key: " + key);
    }
    pos = object.find(':', pos + quoted_key.size());
    if (pos == std::string::npos) {
        throw std::runtime_error(
            std::string(context) + ": malformed key: " + key);
    }
    pos = object.find('"', pos);
    if (pos == std::string::npos) {
        throw std::runtime_error(
            std::string(context) + ": malformed string value: " + key);
    }
    ++pos;
    std::string decoded;
    bool escaped = false;
    for (; pos < object.size(); ++pos) {
        char const c = object[pos];
        if (escaped) {
            decoded.push_back(c);
            escaped = false;
        } else if (c == '\\') {
            escaped = true;
        } else if (c == '"') {
            return decoded;
        } else {
            decoded.push_back(c);
        }
    }
    throw std::runtime_error(
        std::string(context) + ": unterminated string value: " + key);
}

}  // namespace rrea::json_text

#endif  // RREA_JSON_TEXT_UTIL_H
