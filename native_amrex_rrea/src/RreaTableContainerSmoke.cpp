#include "rrea/RreaGpuSmoke.H"
// RREATBL container reader: build a container in the build tree, read it back.
//
// The packer (scripts/rrea_table_pack.py) is the writer; this pins the reader
// against a byte layout written here by hand, so the two cannot drift apart
// without one of them failing.  Covers every dtype the bundle uses, the
// name-based column lookup, the un-shuffle, and every header refusal.

#include "rrea/RreaTableContainer.H"
#include "RreaCheckpointJson.H"

#include <AMReX.H>
#include <AMReX_Print.H>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

void require(bool condition, char const* message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_throws(std::function<void()> const& action, char const* message)
{
    try {
        action();
    } catch (std::exception const&) {
        return;
    }
    throw std::runtime_error(message);
}

//! The packer's permutation: byte k of every item, contiguously.
std::string shuffle(std::string const& plain, std::size_t width)
{
    if (width == 1U) { return plain; }
    std::size_t const count = plain.size() / width;
    std::string out(plain.size(), '\0');
    for (std::size_t k = 0U; k < width; ++k) {
        for (std::size_t i = 0U; i < count; ++i) {
            out[k * count + i] = plain[i * width + k];
        }
    }
    return out;
}

template <typename T>
std::string raw_of(std::vector<T> const& values)
{
    std::string out(values.size() * sizeof(T), '\0');
    for (std::size_t i = 0U; i < values.size(); ++i) {
        std::memcpy(out.data() + i * sizeof(T), &values[i], sizeof(T));
    }
    return out;
}

void put_le(std::string& out, std::uint64_t value, std::size_t width)
{
    for (std::size_t i = 0U; i < width; ++i) {
        out.push_back(static_cast<char>((value >> (8U * i)) & 0xFFU));
    }
}

std::string build(std::string const& header_json, std::string const& payload_plain)
{
    uLongf bound = compressBound(static_cast<uLong>(payload_plain.size()));
    std::string deflated(bound, '\0');
    int const status = compress2(
        reinterpret_cast<Bytef*>(deflated.data()), &bound,
        reinterpret_cast<Bytef const*>(payload_plain.data()),
        static_cast<uLong>(payload_plain.size()), 9);
    require(status == Z_OK, "smoke could not deflate its own payload");
    deflated.resize(static_cast<std::size_t>(bound));

    std::string out = "RREATBL";
    out.push_back('\0');
    put_le(out, 1U, 4U);
    put_le(out, 0x01020304U, 4U);
    put_le(out, header_json.size(), 4U);
    put_le(out, 0U, 4U);
    put_le(out, deflated.size(), 8U);
    out.append(32U, '\0');
    return out + header_json + deflated;
}

void run_smoke()
{
    std::string const json = R"({"value":"a\"b\\c","empty":""})";
    require(
        rrea::warpx::json_string_value(json, "value") == "a\"b\\c",
        "shared JSON reader decoded escapes incorrectly");
    require(
        rrea::warpx::json_has_key(json, "empty")
            && rrea::warpx::json_string_value(json, "empty").empty(),
        "shared JSON reader rejected an empty checkpoint string");
    require(
        rrea::warpx::json_string_value(json, "missing").empty(),
        "checkpoint JSON adapter accepted a missing string");
    require(
        rrea::warpx::json_string_value(R"({"value":42})", "value").empty(),
        "checkpoint JSON adapter accepted a non-string value");
    require(
        rrea::warpx::json_string_value(R"({"value":"unterminated})", "value").empty(),
        "checkpoint JSON adapter accepted an unterminated string");
    require_throws(
        [&]() { (void)rrea::json_text::value(json, "missing", "smoke"); },
        "shared JSON reader did not report a missing string");

    namespace fs = std::filesystem;
    fs::path const dir = fs::temp_directory_path() / "rrea_table_container_smoke";
    fs::create_directories(dir);
    fs::path const path = dir / "t.rtb";

    // Two sub-ulp-adjacent anchors: the loaders require these to stay
    // distinct, and they are exactly what a narrowed store would collapse.
    std::vector<double> const energies{
        1000.0, 50000.00000000001, 50000.000000000015, 1.0e11};
    std::vector<std::int32_t> const pdgs{11, -11, 0, 11};
    std::vector<std::uint16_t> const z{6, 7, 8, 18};
    std::string const symbols = "C\nN\nO\nAr";

    std::string const e_blob = shuffle(raw_of(energies), 8U);
    std::string const p_blob = shuffle(raw_of(pdgs), 4U);
    std::string const z_blob = shuffle(raw_of(z), 2U);
    std::string const header =
        std::string(R"({"column_bytes":")") + std::to_string(e_blob.size()) + ","
        + std::to_string(p_blob.size()) + "," + std::to_string(z_blob.size()) + ","
        + std::to_string(symbols.size())
        + R"(","column_dtypes":"f8,i4,u2,str",)"
        + R"("column_names":"energy_eV,pdg,element_Z,element_symbol",)"
        + R"("kind":"columns","origin":"smoke","rows":"4"})";
    {
        std::ofstream out(path, std::ios::binary);
        std::string const blob = build(header, e_blob + p_blob + z_blob + symbols);
        out.write(blob.data(), static_cast<std::streamsize>(blob.size()));
    }

    auto const container = rrea::table_container::Read(path.string());
    require(rrea::table_container::Rows(container) == 4U, "row count wrong");

    auto const read_energies = rrea::table_container::Column(container, "energy_eV");
    require(read_energies.size() == 4U, "energy column length wrong");
    for (std::size_t i = 0U; i < energies.size(); ++i) {
        require(static_cast<double>(read_energies[i]) == energies[i],
                "energy column did not round trip bit-exactly");
    }
    require(read_energies[1] != read_energies[2],
            "the sub-ulp anchor pair collapsed");
    require(static_cast<double>(read_energies[3]) == 1.0e11,
            "1e11 did not survive; the certified range check would fail");

    // Located by name, so declaration order is irrelevant -- element_Z sits
    // after pdg in the payload and must still decode from its own bytes.
    auto const read_pdg = rrea::table_container::Column(container, "pdg");
    auto const read_z = rrea::table_container::Column(container, "element_Z");
    for (std::size_t i = 0U; i < pdgs.size(); ++i) {
        require(static_cast<int>(read_pdg[i]) == pdgs[i], "signed i4 column wrong");
        require(static_cast<int>(read_z[i]) == static_cast<int>(z[i]), "u2 column wrong");
    }

    auto const read_symbols =
        rrea::table_container::TextColumn(container, "element_symbol");
    require(read_symbols.size() == 4U && read_symbols[0] == "C"
                && read_symbols[3] == "Ar",
            "text column wrong");

    require_throws(
        [&]() { (void)rrea::table_container::Column(container, "absent"); },
        "an absent column was not refused");
    require_throws(
        [&]() { (void)rrea::table_container::Column(container, "element_symbol"); },
        "a text column was returned as numeric");

    // Every header refusal, each from a single mutated byte or a truncation.
    std::string const good = [&]() {
        std::ifstream in(path, std::ios::binary);
        return std::string((std::istreambuf_iterator<char>(in)),
                           std::istreambuf_iterator<char>());
    }();
    auto write_and_expect_throw = [&](std::string const& blob, char const* message) {
        fs::path const bad = dir / "bad.rtb";
        {
            std::ofstream out(bad, std::ios::binary);
            out.write(blob.data(), static_cast<std::streamsize>(blob.size()));
        }
        require_throws(
            [&]() { (void)rrea::table_container::Read(bad.string()); }, message);
    };
    std::string mutated = good;
    mutated[0] = 'X';
    write_and_expect_throw(mutated, "a bad magic was accepted");
    mutated = good;
    mutated[8] = static_cast<char>(9);
    write_and_expect_throw(mutated, "an unknown version was accepted");
    mutated = good;
    mutated[12] = static_cast<char>(0x99);
    write_and_expect_throw(mutated, "a foreign endianness was accepted");
    write_and_expect_throw(good.substr(0, good.size() - 1),
                           "a truncated payload was accepted");

    // A non-finite double must be refused by column name.
    std::vector<double> const with_nan{1.0, std::nan("")};
    std::string const nan_blob = shuffle(raw_of(with_nan), 8U);
    std::string const nan_header =
        std::string(R"({"column_bytes":")") + std::to_string(nan_blob.size())
        + R"(","column_dtypes":"f8","column_names":"x","kind":"columns",)"
        + R"("origin":"smoke","rows":"2"})";
    {
        std::ofstream out(dir / "nan.rtb", std::ios::binary);
        std::string const blob = build(nan_header, nan_blob);
        out.write(blob.data(), static_cast<std::streamsize>(blob.size()));
    }
    auto const nan_container = rrea::table_container::Read((dir / "nan.rtb").string());
    require_throws(
        [&]() { (void)rrea::table_container::Column(nan_container, "x"); },
        "a non-finite column value was accepted");

    fs::remove_all(dir);
}

}  // namespace

int main(int argc, char** argv)
{
    rrea::smoke::InitializeAmrexSmoke(argc, argv);
    int result = 0;
    try {
        run_smoke();
        amrex::Print() << "rrea_table_container_smoke passed\n";
    } catch (std::exception const& exc) {
        result = 1;
        amrex::AllPrint() << "rrea_table_container_smoke failed: " << exc.what() << "\n";
    }
    amrex::Finalize();
    return result;
}
