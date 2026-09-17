#!/usr/bin/env python3
"""Pack schema-6 transport data into the RREATBL container.

``native_amrex_rrea/include/rrea/RreaTableContainer.H`` defines the binary
format consumed by C++; this module is its writer and Python reader. The JSON
header names every column with its dtype and byte length. Three payload kinds
are supported:

  columns  a CSV: named columns, each byte-shuffled then deflated together.
           Shuffle groups like-significance bytes before deflate.
  records  a fixed-stride binary: the same column payload, plus the original
           file header verbatim so the stride layout can be rebuilt.
  opaque   anything else: the original bytes, deflated.

Numeric columns remain float64 because narrowing can merge adjacent energy
nodes and violate final-state norm or energy closure. The CLI exposes
read-only ``dump`` and ``verify`` commands as well as replacement-bundle
``pack``/``unpack`` operations.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import struct
import sys
import zlib
from pathlib import Path

# Two files stay plain text: the transport configuration selects a table set,
# and the pair index carries the block tiling used for inspection.
KEEP_PLAIN = ("transport_physics.json", ".index.json")

MAGIC = b"RREATBL\0"
VERSION = 1
ENDIAN_MARKER = 0x01020304
HEADER_PREFIX_BYTES = 64
SUFFIX = ".rtb"

# The one fixed-stride binary in the bundle.  Its 144-byte record is decoded by
# load_pair_empirical_blocks; the field list is stated once, here, instead of
# being implicit in offset arithmetic on both sides.
PAIR_RECORD_BYTES = 144
PAIR_FIELDS: tuple[tuple[str, int, str], ...] = (
    ("target_Z", 0, "u2"),
    ("channel_code", 2, "u1"),
    ("lepton_count", 3, "u1"),
    ("flags", 4, "u4"),
    ("primary_energy_eV", 8, "f8"),
    ("local_recoil_fraction", 16, "f8"),
    *[
        (f"p{s}_{name}", 24 + 40 * s + off, dtype)
        for s in range(3)
        for name, off, dtype in (
            ("pdg", 0, "i4"),
            ("reserved", 4, "u4"),
            ("kinetic_fraction", 8, "f8"),
            ("dir_x", 16, "f8"),
            ("dir_y", 24, "f8"),
            ("dir_z", 32, "f8"),
        )
    ],
)
_ITEM_BYTES = {"f8": 8, "i4": 4, "u4": 4, "u2": 2, "u1": 1}
_STRUCT = {"f8": "<d", "i4": "<i", "u4": "<I", "u2": "<H", "u1": "<B"}


def _shuffle(blob: bytes, item_bytes: int) -> bytes:
    """Byte k of every item, then byte k+1 of every item, ..."""
    if item_bytes == 1:
        return blob
    return b"".join(blob[k::item_bytes] for k in range(item_bytes))


def _unshuffle(blob: bytes, item_bytes: int) -> bytes:
    if item_bytes == 1:
        return blob
    count = len(blob) // item_bytes
    planes = [blob[k * count:(k + 1) * count] for k in range(item_bytes)]
    return b"".join(bytes(plane[i] for plane in planes) for i in range(count))


def _encode(values: list, dtype: str) -> bytes:
    if dtype == "str":
        return "\n".join(values).encode("utf-8")
    fmt = _STRUCT[dtype]
    cast = float if dtype == "f8" else int
    return b"".join(struct.pack(fmt, cast(v)) for v in values)


def _decode(blob: bytes, dtype: str) -> list:
    if dtype == "str":
        return blob.decode("utf-8").split("\n") if blob else []
    fmt = _STRUCT[dtype]
    size = _ITEM_BYTES[dtype]
    return [struct.unpack_from(fmt, blob, i)[0] for i in range(0, len(blob), size)]


def _build(header: dict, blobs: list[bytes]) -> bytes:
    payload = zlib.compress(b"".join(blobs), 9)
    text = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    prefix = (
        MAGIC
        + struct.pack("<III", VERSION, ENDIAN_MARKER, len(text))
        + struct.pack("<I", 0)
        + struct.pack("<Q", len(payload))
        + bytes(32)
    )
    assert len(prefix) == HEADER_PREFIX_BYTES
    return prefix + text + payload


def read_container(blob: bytes) -> tuple[dict, bytes]:
    """Return (header, inflated payload), refusing anything incompatible."""
    if len(blob) < HEADER_PREFIX_BYTES or blob[:8] != MAGIC:
        raise ValueError("not an RREATBL container")
    version, marker, header_bytes = struct.unpack_from("<III", blob, 8)
    payload_bytes = struct.unpack_from("<Q", blob, 24)[0]
    if version != VERSION or marker != ENDIAN_MARKER:
        raise ValueError(f"RREATBL v{version} / endian {marker:#x} is not supported")
    if len(blob) != HEADER_PREFIX_BYTES + header_bytes + payload_bytes:
        raise ValueError("RREATBL size does not match its declared header/payload")
    start = HEADER_PREFIX_BYTES + header_bytes
    header = json.loads(blob[HEADER_PREFIX_BYTES:start].decode("utf-8"))
    return header, zlib.decompress(blob[start:])


def header_columns(header: dict) -> list[tuple[str, str, int]]:
    """(name, dtype, bytes) per column.

    Three comma-joined strings rather than an array of objects, so the C++
    reader needs only json_string plus the split_csv_line it already has --
    no JSON array walker enters the engine for this.
    """
    return list(zip(
        header["column_names"].split(","),
        header["column_dtypes"].split(","),
        (int(v) for v in header["column_bytes"].split(",")),
        strict=True))


def _columns_from(header: dict, payload: bytes) -> dict[str, list]:
    out, offset = {}, 0
    for name, dtype, size in header_columns(header):
        blob = payload[offset:offset + size]
        offset += size
        if dtype != "str":
            blob = _unshuffle(blob, _ITEM_BYTES[dtype])
        out[name] = _decode(blob, dtype)
    return out


def load_columns(path) -> dict[str, list]:
    """Named columns of a packed table, for Python consumers."""
    header, payload = read_container(pathlib.Path(path).read_bytes())
    return _columns_from(header, payload)


def load_bytes(path) -> bytes:
    """The original bytes of an opaque container."""
    return unpack_bytes(pathlib.Path(path).read_bytes())


def _pack_columns(names: list[str], values: dict[str, list], kind: str, extra: dict) -> bytes:
    dtypes, sizes, blobs = [], [], []
    for name in names:
        column = values[name]
        dtype = "f8"
        if any(isinstance(v, str) for v in column):
            dtype = "str"
        elif extra.get("dtypes", {}).get(name):
            dtype = extra["dtypes"][name]
        blob = _encode(column, dtype)
        if dtype != "str":
            blob = _shuffle(blob, _ITEM_BYTES[dtype])
        blobs.append(blob)
        dtypes.append(dtype)
        sizes.append(str(len(blob)))
    header = {
        "kind": kind,
        "rows": str(len(values[names[0]]) if names else 0),
        "column_names": ",".join(names),
        "column_dtypes": ",".join(dtypes),
        "column_bytes": ",".join(sizes),
    }
    header.update({k: v for k, v in extra.items() if k != "dtypes"})
    return _build(header, blobs)


def pack_bytes(relative: str, raw: bytes) -> bytes:
    """Pack one bundle file by what it is, not by its extension."""
    if raw[:8] == b"RREAPAIR":
        body = raw[HEADER_PREFIX_BYTES:]
        count = len(body) // PAIR_RECORD_BYTES
        values = {
            name: [
                struct.unpack_from(_STRUCT[dtype], body, r * PAIR_RECORD_BYTES + off)[0]
                for r in range(count)
            ]
            for name, off, dtype in PAIR_FIELDS
        }
        return _pack_columns(
            [f[0] for f in PAIR_FIELDS], values, "records",
            {"origin": relative, "record_bytes": str(PAIR_RECORD_BYTES),
             "file_header_hex": raw[:HEADER_PREFIX_BYTES].hex(),
             "dtypes": {f[0]: f[2] for f in PAIR_FIELDS}})
    if relative.endswith(".csv"):
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = None
        if text is not None:
            head, *rows = text.splitlines()
            names = head.split(",")
            values: dict[str, list] = {name: [] for name in names}
            for row in rows:
                fields = row.split(",")
                if len(fields) != len(names):
                    values = {}
                    break
                for name, field in zip(names, fields):
                    try:
                        values[name].append(float(field))
                    except ValueError:
                        values[name].append(field)
            if values:
                for name in names:
                    if any(isinstance(v, str) for v in values[name]):
                        values[name] = [str(v) for v in values[name]]
                return _pack_columns(names, values, "columns", {"origin": relative})
    return _build({"kind": "opaque", "origin": relative}, [raw])


def unpack_bytes(blob: bytes) -> bytes:
    """Rebuild the original file: exact for opaque, value-exact otherwise."""
    header, payload = read_container(blob)
    kind = header["kind"]
    if kind == "opaque":
        return payload
    columns = _columns_from(header, payload)
    names = [c[0] for c in header_columns(header)]
    if kind == "records":
        out = bytearray(bytes.fromhex(header["file_header_hex"]))
        stride = int(header["record_bytes"])
        dtypes = {name: dtype for name, dtype, _ in header_columns(header)}
        offsets = {name: off for name, off, _ in PAIR_FIELDS}
        for r in range(int(header["rows"])):
            record = bytearray(stride)
            for name in names:
                struct.pack_into(_STRUCT[dtypes[name]], record, offsets[name], columns[name][r])
            out += record
        return bytes(out)
    lines = [",".join(names)]
    for r in range(int(header["rows"])):
        lines.append(",".join(
            v if isinstance(v := columns[name][r], str) else repr(v) for name in names))
    return ("\n".join(lines) + "\n").encode("utf-8")


def _same_values(original: bytes, restored: bytes) -> bool:
    """Value equality, since a float is rewritten in its shortest exact form."""
    def norm(blob: bytes) -> list:
        out = []
        for line in blob.decode("utf-8").strip("\n").split("\n"):
            row = []
            for field in line.split(","):
                try:
                    row.append(float(field))
                except ValueError:
                    row.append(field)
            out.append(row)
        return out
    return norm(original) == norm(restored)


def stays_plain(relative: str) -> bool:
    return any(relative.endswith(marker) for marker in KEEP_PLAIN)


def _packable(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix != SUFFIX
        and not stays_plain(p.relative_to(root).as_posix()))


def _packed(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix == SUFFIX)


def retarget_config(text: str, to_packed: bool) -> str:
    """Point every table "path" at the packed or the plain file.

    The engine resolves each table from this configuration; container names
    are stated here rather than inferred by the reader.
    """
    if to_packed:
        return text.replace('.csv"', '.csv' + SUFFIX + '"').replace(
            '.bin"', '.bin' + SUFFIX + '"')
    return text.replace('.csv' + SUFFIX + '"', '.csv"').replace(
        '.bin' + SUFFIX + '"', '.bin"')


HELP = {
    "pack": "CSV/bin -> .rtb, and DELETE each source file; retargets the "
            "transport configuration at the packed names. The tables ship packed, so this is "
            "for a replacement table set, not for the shipped one.",
    "unpack": "restore the original files beside their .rtb. Byte-exact for "
              "opaque and records; value-exact for columns, since a float is "
              "rewritten in its shortest exact form. Leaves the .rtb in place.",
    "verify": "pack and unpack the whole bundle in memory and check the round "
              "trip. Reads only.",
    "dump": "print one container's header, or one column, without unpacking "
            "anything. This is how you read a shipped table.",
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        epilog="The shipped bundle is already packed: use dump to read a table "
               "and unpack only if you need the CSV text back.")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("pack", "unpack", "verify"):
        sub.add_parser(name, help=HELP[name]).add_argument("directory", type=Path)
    dump = sub.add_parser("dump", help=HELP["dump"])
    dump.add_argument("file", type=Path)
    dump.add_argument("--column")
    dump.add_argument("--rows", type=int, default=10)
    args = p.parse_args(argv)

    if args.command == "dump":
        header, payload = read_container(args.file.read_bytes())
        if args.column is None:
            print(json.dumps(header, indent=2, sort_keys=True))
            if header["kind"] != "opaque":
                print("columns:")
                for name, dtype, size in header_columns(header):
                    print(f"  {name:34s} {dtype:4s} {size:12d} B")
            return 0
        column = _columns_from(header, payload)[args.column]
        print(args.column)
        for value in column[:args.rows]:
            print(value)
        if len(column) > args.rows:
            print(f"... {len(column) - args.rows} more of {len(column)}")
        return 0

    root: Path = args.directory
    config = root / "transport_physics.json"
    if args.command == "pack":
        for path in _packable(root):
            relative = path.relative_to(root).as_posix()
            out = path.with_suffix(path.suffix + SUFFIX)
            out.write_bytes(pack_bytes(relative, path.read_bytes()))
            path.unlink()
            print(f"packed {relative} -> {out.relative_to(root).as_posix()}")
        config.write_text(
            retarget_config(config.read_text(encoding="utf-8"), True), encoding="utf-8")
        print("retargeted transport_physics.json at the packed tables")
        return 0

    if args.command == "unpack":
        for path in _packed(root):
            target = path.with_suffix("")
            target.write_bytes(unpack_bytes(path.read_bytes()))
            print(f"unpacked {target.relative_to(root).as_posix()}")
        print("(the transport configuration still points at the packed tables)")
        return 0

    # Works on the bundle in whichever state it is: a packed one is inflated
    # and re-packed, a loose one is packed and restored.  Either way the check
    # is the same round trip, and finding nothing to check is a failure --
    # a verify that silently examines zero files reads exactly like a pass.
    failures, packed_total, raw_total = [], 0, 0
    packed = _packed(root)
    for path in packed or _packable(root):
        relative = path.relative_to(root).as_posix()
        if path in packed:
            blob = path.read_bytes()
            header = read_container(blob)[0]
            kind = header["kind"]
            raw = unpack_bytes(blob)
            again = unpack_bytes(pack_bytes(header["origin"], raw))
            ok = again == raw if kind != "columns" else _same_values(raw, again)
        else:
            raw = path.read_bytes()
            blob = pack_bytes(relative, raw)
            restored = unpack_bytes(blob)
            kind = read_container(blob)[0]["kind"]
            ok = restored == raw if kind != "columns" else _same_values(raw, restored)
        raw_total += len(raw)
        packed_total += len(blob)
        if not ok:
            failures.append(f"{relative} ({kind})")
        print(f"  {'ok ' if ok else 'FAIL'} {kind:8s} {len(raw) / 1048576:8.2f} -> "
              f"{len(blob) / 1048576:6.2f} MB  {relative}")
    print(f"total {raw_total / 1048576:.2f} MB -> {packed_total / 1048576:.2f} MB")
    if not (packed or _packable(root)):
        print(f"nothing to verify under {root}", file=sys.stderr)
        return 1
    if failures:
        print("round trip failed: " + ", ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
