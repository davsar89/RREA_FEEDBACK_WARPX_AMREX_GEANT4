#!/usr/bin/env python3
"""Answer remote HPC SSH password and OATH prompts without logging credentials.

The password comes from a local environment variable or private local file. The
TOTP normally comes from ``HPC_TOTP_CODE``; when the explicitly authorized
``HPC_TOTP_AUTOMATED=1`` mode is selected, it is derived in memory from a
Base32 ``HPC_TOTP_SECRET`` or the private ``authenticator_qr.jpg``.
Both standard ``otpauth://totp`` and Google Authenticator migration QR records
are supported.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


def _local_file(
    env_name: str,
    state_filename: str,
    project_filename: str,
) -> Path | None:
    explicit = os.environ.get(env_name, "").strip()
    if explicit:
        return Path(explicit)
    state_root = os.environ.get("HPC_STATE_DIR", "").strip()
    if not state_root:
        state_root = str(
            Path.home() / ".local" / "share" / "rrea_warpx_amrex" / "hpc"
        )
    state_path = Path(state_root) / state_filename
    if state_path.is_file():
        return state_path
    project = os.environ.get("HPC_LOCAL_PROJECT", "").strip()
    project_path = Path(project) / project_filename if project else None
    if project_path is not None and project_path.is_file():
        return project_path
    return state_path


def _value_from_env_file(path: Path | None, wanted_key: str) -> str | None:
    if path is None or not path.is_file():
        return None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if separator and key.strip().upper() == wanted_key and value.strip():
            return value.strip().strip("`\"'")
    return None


def _temporary_credential(wanted_key: str) -> str | None:
    path_text = os.environ.get("HPC_ASKPASS_CREDENTIAL_FILE", "").strip()
    return _value_from_env_file(Path(path_text) if path_text else None, wanted_key)


def _password_from_local_note(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        normalized = line.lstrip("-* ").strip().lower()
        if not normalized.startswith("password"):
            continue
        _key, separator, value = line.partition(":")
        if not separator:
            _key, separator, value = line.partition("=")
        if separator and value.strip():
            return value.strip().strip("`\"'")
    return None


def read_password() -> str:
    env_file = _local_file(
        "HPC_ENV_FILE", "credentials.env", "hpc.env"
    )
    local_note = _local_file(
        "HPC_PASSWORD_FILE", "password_note.txt", "REMOTE_HPC.md"
    )
    for source in (
        os.environ.get("HPC_PASSWORD", "").strip(),
        _temporary_credential("HPC_PASSWORD"),
        _value_from_env_file(env_file, "HPC_PASSWORD"),
        _password_from_local_note(local_note),
    ):
        if source:
            return source
    raise SystemExit(
        "No HPC password found. Set HPC_PASSWORD or configure "
        "HPC_STATE_DIR with a mode-0600 credentials.env file."
    )


def _read_varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    for shift in range(0, 70, 7):
        if offset >= len(data):
            raise ValueError("truncated varint in authenticator migration record")
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
    raise ValueError("oversized varint in authenticator migration record")


def _read_fields(data: bytes) -> list[tuple[int, int, Any]]:
    fields: list[tuple[int, int, Any]] = []
    offset = 0
    while offset < len(data):
        key, offset = _read_varint(data, offset)
        field_number = key >> 3
        wire_type = key & 7
        if field_number == 0:
            raise ValueError("invalid field in authenticator migration record")
        if wire_type == 0:
            value, offset = _read_varint(data, offset)
        elif wire_type in (1, 5):
            size = 8 if wire_type == 1 else 4
            end = offset + size
            if end > len(data):
                raise ValueError("truncated fixed field in authenticator migration record")
            value = data[offset:end]
            offset = end
        elif wire_type == 2:
            size, offset = _read_varint(data, offset)
            end = offset + size
            if end > len(data):
                raise ValueError("truncated field in authenticator migration record")
            value = data[offset:end]
            offset = end
        else:
            raise ValueError(
                f"unsupported wire type {wire_type} in authenticator migration record"
            )
        fields.append((field_number, wire_type, value))
    return fields


def _parse_otp_parameters(data: bytes) -> dict[str, Any]:
    item: dict[str, Any] = {
        "secret": None,
        "name": "",
        "issuer": "",
        "algorithm": 1,
        "digits": 1,
        "type": 2,
    }
    for field_number, _wire_type, value in _read_fields(data):
        if field_number == 1:
            item["secret"] = value
        elif field_number == 2:
            item["name"] = value.decode("utf-8", errors="replace")
        elif field_number == 3:
            item["issuer"] = value.decode("utf-8", errors="replace")
        elif field_number == 4:
            item["algorithm"] = value
        elif field_number == 5:
            item["digits"] = value
        elif field_number == 6:
            item["type"] = value
    return item


def _qr_text() -> str:
    qr_path = _local_file(
        "HPC_QR_FILE",
        "authenticator_qr.jpg",
        "QR_hpc_ssh.jpg",
    )
    if qr_path is None or not qr_path.is_file():
        raise SystemExit(
            "No local HPC QR image found; use interactive TOTP or set "
            "HPC_QR_FILE to the private authenticator export."
        )
    if shutil.which("zbarimg"):
        try:
            decoded = subprocess.run(
                ["zbarimg", "--quiet", "--raw", str(qr_path)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            ).stdout.strip()
        except subprocess.CalledProcessError as exc:
            raise SystemExit("The local HPC QR image could not be decoded.") from exc
    else:
        try:
            import cv2
        except ImportError as exc:
            raise SystemExit("Install zbar-tools or OpenCV to decode the local QR.") from exc
        image = cv2.imread(str(qr_path))
        if image is None:
            raise SystemExit("The local HPC QR image could not be read.")
        decoded, _points, _straight = cv2.QRCodeDetector().detectAndDecode(image)
        decoded = decoded.strip()
    if not decoded:
        raise SystemExit("The local HPC QR image could not be decoded.")
    return decoded


def _base32_secret(text: str) -> bytes:
    clean = re.sub(r"\s+", "", text).upper()
    try:
        return base64.b32decode(clean + "=" * (-len(clean) % 8), casefold=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise SystemExit("The HPC TOTP credential has an invalid Base32 secret.") from exc


def _digest(name: str):
    normalized = name.strip().upper().replace("-", "")
    algorithms = {
        "SHA1": hashlib.sha1,
        "SHA256": hashlib.sha256,
        "SHA512": hashlib.sha512,
    }
    if normalized not in algorithms:
        raise SystemExit(f"Unsupported TOTP algorithm: {normalized or '<empty>'}")
    return algorithms[normalized]


def _first_query_value(query: dict[str, list[str]], key: str) -> str:
    values = query.get(key, [])
    if not values or not values[0]:
        raise SystemExit(f"The HPC QR is missing required TOTP field {key}.")
    return values[0]


def _totp_parameters_from_qr() -> tuple[bytes, int, Any, int]:
    decoded = _qr_text()
    start = decoded.find("otpauth")
    if start < 0:
        raise SystemExit("The HPC QR does not contain an authenticator record.")
    parsed = urlparse(decoded[start:].split()[0])

    if parsed.scheme == "otpauth" and parsed.netloc == "totp":
        query = parse_qs(parsed.query)
        secret = _base32_secret(_first_query_value(query, "secret"))
        try:
            digits = int(query.get("digits", ["6"])[0])
            period = int(query.get("period", ["30"])[0])
        except ValueError as exc:
            raise SystemExit("The HPC QR has invalid TOTP numeric parameters.") from exc
        digest = _digest(query.get("algorithm", ["SHA1"])[0])
        if digits not in (6, 8) or period <= 0:
            raise SystemExit("The HPC QR has unsupported TOTP digits or period.")
        return secret, digits, digest, period

    if parsed.scheme != "otpauth-migration":
        raise SystemExit("The HPC QR uses an unsupported authenticator format.")

    query = parse_qs(parsed.query)
    encoded = _first_query_value(query, "data")
    try:
        payload = base64.b64decode(
            encoded + "=" * (-len(encoded) % 4),
            altchars=b"-_",
            validate=True,
        )
        fields = _read_fields(payload)
        scalar_fields = {
            field_number: value
            for field_number, wire_type, value in fields
            if wire_type == 0
        }
        if scalar_fields.get(3, 1) != 1 or scalar_fields.get(4, 0) != 0:
            raise ValueError("incomplete multi-image authenticator migration record")
        candidates = [
            _parse_otp_parameters(value)
            for field_number, _wire_type, value in fields
            if field_number == 1
        ]
    except (ValueError, base64.binascii.Error) as exc:
        raise SystemExit("The HPC authenticator migration record is invalid.") from exc
    candidates = [
        item for item in candidates if item["type"] == 2 and item["secret"]
    ]
    if not candidates:
        raise SystemExit("The HPC QR contains no TOTP account.")

    hint = os.environ.get("HPC_TOTP_LABEL_CONTAINS", "").strip().lower()
    matches = [
        item
        for item in candidates
        if hint
        and hint
        in f"{item.get('issuer', '')} {item.get('name', '')}".lower()
    ]
    if len(matches) == 1:
        selected = matches[0]
    elif len(candidates) == 1:
        selected = candidates[0]
    else:
        raise SystemExit(
            "The HPC QR contains multiple TOTP accounts and the configured "
            "HPC_TOTP_LABEL_CONTAINS hint did not select exactly one."
        )

    digest_by_id = {
        0: hashlib.sha1,
        1: hashlib.sha1,
        2: hashlib.sha256,
        3: hashlib.sha512,
    }
    digits_by_id = {0: 6, 1: 6, 2: 8}
    algorithm = digest_by_id.get(selected["algorithm"])
    digits = digits_by_id.get(selected["digits"])
    if algorithm is None or digits is None:
        raise SystemExit("The HPC QR uses unsupported TOTP parameters.")
    return selected["secret"], digits, algorithm, 30


def _hotp(secret: bytes, counter: int, digits: int, digest) -> str:
    result = hmac.new(secret, struct.pack(">Q", counter), digest).digest()
    offset = result[-1] & 0x0F
    value = int.from_bytes(result[offset : offset + 4], "big") & 0x7FFFFFFF
    return str(value % (10**digits)).zfill(digits)


def _current_totp(secret: bytes, digits: int, digest, period: int) -> str:
    now = time.time()
    remaining = period - (now % period)
    if remaining < 5.0:
        time.sleep(remaining + 0.25)
        now = time.time()
    return _hotp(secret, int(now) // period, digits, digest)


def _totp_from_local_credential() -> str:
    env_file = _local_file(
        "HPC_ENV_FILE", "credentials.env", "hpc.env"
    )
    setup_key = (
        os.environ.get("HPC_TOTP_SECRET", "").strip()
        or _temporary_credential("HPC_TOTP_SECRET")
        or _value_from_env_file(env_file, "HPC_TOTP_SECRET")
    )
    if setup_key:
        return _current_totp(_base32_secret(setup_key), 6, hashlib.sha1, 30)
    return _current_totp(*_totp_parameters_from_qr())


def read_totp() -> str:
    code = (
        os.environ.get("HPC_TOTP_CODE", "").strip()
        or (_temporary_credential("HPC_TOTP_CODE") or "").strip()
    )
    if code:
        if not re.fullmatch(r"[0-9]{6}|[0-9]{8}", code):
            raise SystemExit("HPC_TOTP_CODE must contain six or eight digits.")
        return code
    if os.environ.get("HPC_TOTP_AUTOMATED", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }:
        return _totp_from_local_credential()
    raise SystemExit(
        "No transient HPC_TOTP_CODE is available. Run the socket opener "
        "interactively or explicitly authorize HPC_TOTP_AUTOMATED=1."
    )


def main() -> None:
    prompt = " ".join(sys.argv[1:]).lower()
    if any(
        word in prompt
        for word in (
            "one-time",
            "oath",
            "otp",
            "totp",
            "verification",
            "authenticator",
            "token",
        )
    ):
        print(read_totp())
    elif "password" in prompt:
        print(read_password())
    else:
        print("")


if __name__ == "__main__":
    main()
