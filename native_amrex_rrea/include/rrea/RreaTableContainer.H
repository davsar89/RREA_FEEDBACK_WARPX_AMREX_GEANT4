#ifndef RREA_TABLE_CONTAINER_H
#define RREA_TABLE_CONTAINER_H

// RREATBL v1 -- production schema-6 tables in a self-describing container.
//
//    0   8  magic "RREATBL\0"
//    8   4  version                     u32 le
//   12   4  endian marker 0x01020304    u32 le
//   16   4  header_bytes                u32 le
//   20   4  reserved, must be 0
//   24   8  payload_bytes               u64 le
//   32  32  reserved, must be zero
//   64   N  header: UTF-8 JSON, every value a string
//       ..  payload: one deflate stream of the columns, in header order
//
// The JSON names every column with its dtype and byte length, so a replacement
// table set supplies a new header rather than requiring reader offset changes.
// scripts/rrea_table_pack.py writes the format and can dump columns for review.
// Numeric columns decode to amrex::Real (double); narrowing can merge adjacent
// energy nodes and violate final-state direction or energy closure.

#include "rrea/RreaCsvTextUtil.H"
#include "rrea/RreaJsonTextUtil.H"

#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <zlib.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace rrea::table_container {

inline constexpr char const* magic = "RREATBL";
inline constexpr std::uint32_t version = 1U;
inline constexpr std::uint32_t endian_marker = 0x01020304U;
inline constexpr std::size_t prefix_bytes = 64U;

struct Container {
    std::string header;   //!< the JSON text, for json_text::value
    std::string payload;  //!< inflated columns, concatenated in header order
    std::string path;     //!< for diagnostics
};

namespace detail {

[[nodiscard]] inline std::uint64_t read_le(
    std::string const& bytes, std::size_t offset, std::size_t width)
{
    std::uint64_t value = 0U;
    for (std::size_t i = 0U; i < width; ++i) {
        value |= static_cast<std::uint64_t>(
                     static_cast<unsigned char>(bytes[offset + i]))
            << (8U * i);
    }
    return value;
}

[[nodiscard]] inline std::size_t item_bytes(std::string const& dtype)
{
    if (dtype == "f8") { return 8U; }
    if (dtype == "i4" || dtype == "u4") { return 4U; }
    if (dtype == "u2") { return 2U; }
    if (dtype == "u1") { return 1U; }
    if (dtype == "str") { return 0U; }
    throw std::runtime_error("RREATBL: unsupported column dtype: " + dtype);
}

//! Inverse of the packer's permutation: byte k of every item was written
//! contiguously, so gather one item at a time back out of the planes.
inline void unshuffle(
    std::string const& planes, std::size_t width, std::size_t count, char* out)
{
    for (std::size_t k = 0U; k < width; ++k) {
        char const* plane = planes.data() + k * count;
        for (std::size_t i = 0U; i < count; ++i) {
            out[i * width + k] = plane[i];
        }
    }
}

}  // namespace detail

/*! Read and inflate one container, refusing anything incompatible. */
[[nodiscard]] inline Container Read(std::string const& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("RREATBL: could not open " + path);
    }
    std::string const bytes(
        (std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    if (bytes.size() < prefix_bytes || bytes.compare(0U, 7U, magic) != 0
        || bytes[7] != '\0') {
        throw std::runtime_error("RREATBL: bad magic in " + path);
    }
    if (detail::read_le(bytes, 8U, 4U) != version
        || detail::read_le(bytes, 12U, 4U) != endian_marker
        || detail::read_le(bytes, 20U, 4U) != 0U) {
        throw std::runtime_error(
            "RREATBL: incompatible version/endianness in " + path);
    }
    std::size_t const header_bytes =
        static_cast<std::size_t>(detail::read_le(bytes, 16U, 4U));
    std::uint64_t const payload_bytes = detail::read_le(bytes, 24U, 8U);
    if (bytes.size() != prefix_bytes + header_bytes + payload_bytes) {
        throw std::runtime_error(
            "RREATBL: declared header/payload size does not match " + path);
    }
    Container container;
    container.path = path;
    container.header = bytes.substr(prefix_bytes, header_bytes);

    // Inflate in one shot; the header states the uncompressed extent through
    // the per-column byte lengths, so grow until zlib is satisfied.
    std::size_t capacity = static_cast<std::size_t>(payload_bytes) * 4U + 1024U;
    for (int attempt = 0; attempt < 24; ++attempt) {
        container.payload.assign(capacity, '\0');
        uLongf produced = static_cast<uLongf>(capacity);
        int const status = uncompress(
            reinterpret_cast<Bytef*>(container.payload.data()),
            &produced,
            reinterpret_cast<Bytef const*>(bytes.data() + prefix_bytes + header_bytes),
            static_cast<uLong>(payload_bytes));
        if (status == Z_OK) {
            container.payload.resize(static_cast<std::size_t>(produced));
            return container;
        }
        if (status != Z_BUF_ERROR) {
            throw std::runtime_error(
                "RREATBL: zlib inflate failed with status "
                + std::to_string(status) + " on " + path);
        }
        capacity *= 2U;
    }
    throw std::runtime_error("RREATBL: payload did not inflate: " + path);
}

/*! Declared row count. */
[[nodiscard]] inline std::size_t Rows(Container const& container)
{
    return static_cast<std::size_t>(std::stoull(
        rrea::json_text::value(container.header, "rows", "RREATBL")));
}

/*! Byte range and dtype of one named column, located by name as the CSV
 *  loader located it by header name -- so column order stays irrelevant. */
[[nodiscard]] inline bool FindColumn(
    Container const& container,
    std::string const& name,
    std::size_t& offset,
    std::size_t& length,
    std::string& dtype)
{
    using rrea::csv_text::split_csv_line;
    auto const names = split_csv_line(
        rrea::json_text::value(container.header, "column_names", "RREATBL"));
    auto const dtypes = split_csv_line(
        rrea::json_text::value(container.header, "column_dtypes", "RREATBL"));
    auto const sizes = split_csv_line(
        rrea::json_text::value(container.header, "column_bytes", "RREATBL"));
    if (names.size() != dtypes.size() || names.size() != sizes.size()) {
        throw std::runtime_error(
            "RREATBL: column name/dtype/size lists disagree in " + container.path);
    }
    offset = 0U;
    for (std::size_t i = 0U; i < names.size(); ++i) {
        std::size_t const bytes = static_cast<std::size_t>(std::stoull(sizes[i]));
        if (names[i] == name) {
            length = bytes;
            dtype = dtypes[i];
            if (offset + length > container.payload.size()) {
                throw std::runtime_error(
                    "RREATBL: column runs past the payload: " + name);
            }
            return true;
        }
        offset += bytes;
    }
    return false;
}

/*! One numeric column as doubles, in row order. */
[[nodiscard]] inline amrex::Vector<amrex::Real> Column(
    Container const& container, std::string const& name)
{
    std::size_t offset = 0U;
    std::size_t length = 0U;
    std::string dtype;
    if (!FindColumn(container, name, offset, length, dtype)) {
        throw std::runtime_error(
            "RREATBL: " + container.path + " has no column named " + name);
    }
    std::size_t const width = detail::item_bytes(dtype);
    if (width == 0U) {
        throw std::runtime_error(
            "RREATBL: column " + name + " is text, not numeric");
    }
    std::size_t const count = length / width;
    std::string plain(length, '\0');
    detail::unshuffle(
        container.payload.substr(offset, length), width, count, plain.data());

    amrex::Vector<amrex::Real> values;
    values.reserve(count);
    for (std::size_t i = 0U; i < count; ++i) {
        char const* item = plain.data() + i * width;
        if (dtype == "f8") {
            double decoded = 0.0;
            std::memcpy(&decoded, item, sizeof(double));
            // No table column may carry NaN or Inf: every loader that consumes
            // one asserts finiteness anyway, so fail here with the column name.
            if (!std::isfinite(decoded)) {
                throw std::runtime_error(
                    "RREATBL: non-finite value in column " + name + " of "
                    + container.path);
            }
            values.push_back(static_cast<amrex::Real>(decoded));
        } else if (dtype == "i4") {
            std::int32_t decoded = 0;
            std::memcpy(&decoded, item, sizeof(std::int32_t));
            values.push_back(static_cast<amrex::Real>(decoded));
        } else {
            std::uint64_t decoded = 0U;
            std::memcpy(&decoded, item, width);
            values.push_back(static_cast<amrex::Real>(decoded));
        }
    }
    return values;
}

/*! One text column, newline-separated in the payload. */
[[nodiscard]] inline std::vector<std::string> TextColumn(
    Container const& container, std::string const& name)
{
    std::size_t offset = 0U;
    std::size_t length = 0U;
    std::string dtype;
    if (!FindColumn(container, name, offset, length, dtype) || dtype != "str") {
        throw std::runtime_error(
            "RREATBL: " + container.path + " has no text column named " + name);
    }
    std::vector<std::string> out;
    std::string const blob = container.payload.substr(offset, length);
    std::size_t start = 0U;
    while (start <= blob.size()) {
        std::size_t const stop = blob.find('\n', start);
        out.push_back(blob.substr(start, stop - start));
        if (stop == std::string::npos) { break; }
        start = stop + 1U;
    }
    return out;
}

/*! The undecoded payload of an opaque container. */
[[nodiscard]] inline std::string const& Opaque(Container const& container)
{
    if (rrea::json_text::value(container.header, "kind", "RREATBL") != "opaque") {
        throw std::runtime_error("RREATBL: not an opaque container: " + container.path);
    }
    return container.payload;
}

}  // namespace rrea::table_container

#endif  // RREA_TABLE_CONTAINER_H
