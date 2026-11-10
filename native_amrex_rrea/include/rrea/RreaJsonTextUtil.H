#ifndef RREA_JSON_TEXT_UTIL_H
#define RREA_JSON_TEXT_UTIL_H

// One reader for a JSON string value, shared by the transport configuration
// and the RREATBL container header.  Deliberately not a JSON parser: it finds
// "key" : "value" and decodes the escapes, which is the whole of what either
// caller needs.  Kept beside RreaCsvTextUtil.H for the same reason -- a second
// copy of this loop is a second convention that can disagree with the first.

#include <cctype>
#include <cstddef>
#include <cstdlib>
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

/*! Position just past "key": and its whitespace, or npos when the key is
 *  absent.  Throws when the key has no colon.
 */
[[nodiscard]] inline std::size_t after_key(
    std::string const& object,
    std::string const& key,
    char const* context = "JSON")
{
    std::string const quoted_key = "\"" + key + "\"";
    std::size_t pos = object.find(quoted_key);
    if (pos == std::string::npos) {
        return pos;
    }
    pos = object.find(':', pos + quoted_key.size());
    if (pos == std::string::npos) {
        throw std::runtime_error(std::string(context) + ": malformed key: " + key);
    }
    for (++pos; pos < object.size()
         && std::isspace(static_cast<unsigned char>(object[pos])) != 0; ++pos) {
    }
    return pos;
}

/*! Numeric value of "key", or fallback when the key is absent.  Throws when
 *  the value is not a number.
 */
[[nodiscard]] inline double number_or(
    std::string const& object,
    std::string const& key,
    double fallback,
    char const* context = "JSON")
{
    std::size_t const pos = after_key(object, key, context);
    if (pos == std::string::npos) {
        return fallback;
    }
    char const* const begin = object.c_str() + pos;
    char* end = nullptr;
    double const value = std::strtod(begin, &end);
    if (end == begin) {
        throw std::runtime_error(std::string(context) + ": malformed number: " + key);
    }
    return value;
}

/*! Index of the bracket closing the one at open_pos ('{' or '['), skipping
 *  quoted text; npos when unterminated.
 */
[[nodiscard]] inline std::size_t matching_close(std::string const& text, std::size_t open_pos)
{
    char const open = text[open_pos];
    char const close = open == '{' ? '}' : ']';
    int depth = 0;
    bool in_quotes = false;
    bool escaped = false;
    for (std::size_t i = open_pos; i < text.size(); ++i) {
        char const c = text[i];
        if (escaped) {
            escaped = false;
        } else if (in_quotes && c == '\\') {
            escaped = true;
        } else if (c == '"') {
            in_quotes = !in_quotes;
        } else if (!in_quotes && c == open) {
            ++depth;
        } else if (!in_quotes && c == close && --depth == 0) {
            return i;
        }
    }
    return std::string::npos;
}

}  // namespace rrea::json_text

#endif  // RREA_JSON_TEXT_UTIL_H
