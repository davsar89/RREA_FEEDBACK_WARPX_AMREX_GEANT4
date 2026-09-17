#!/usr/bin/env bash
# One-time: pin the HPC system's SSH host key so hpc_open_socket.sh can require
# StrictHostKeyChecking=yes without accept-new.
#
# Run this ONCE from a trusted network. VERIFY the printed fingerprints
# out-of-band against your HPC provider's official documentation before
# relying on the pin -- ssh-keyscan itself is TOFU, so this only moves the
# trust decision to a single supervised moment instead of every first connect.
#
#   wsl -e bash /mnt/c/Users/david/Desktop/rrea_warpx_amrex/scripts/hpc_pin_hostkey.sh
set -euo pipefail

STATE_DIR="${HPC_STATE_DIR:-${HOME}/.local/share/rrea_warpx_amrex/hpc}"
if [[ -z "${HPC_LOGIN:-}" && -s "${STATE_DIR}/credentials.env" ]]; then
  HPC_LOGIN="$(grep -E '^HPC_LOGIN=' "${STATE_DIR}/credentials.env" | tail -n1 | cut -d= -f2-)"
fi
: "${HPC_LOGIN:?no HPC_LOGIN found; set HPC_LOGIN=user@host in the environment or in ${STATE_DIR}/credentials.env (see docs/HPC_CONNECTION_AND_2FA.md)}"
HOST="${HPC_LOGIN#*@}"
KNOWN_HOSTS="${HPC_KNOWN_HOSTS:-${STATE_DIR}/known_hosts}"

echo "scanning host key(s) for ${HOST} ..."
tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT
ssh-keyscan -T 15 "$HOST" > "$tmp" 2>/dev/null || true
if [[ ! -s "$tmp" ]]; then
  echo "ssh-keyscan returned nothing for ${HOST} (network/DNS?)" >&2
  exit 1
fi

echo "=== fingerprints -- VERIFY these against your HPC provider's official"
echo "=== host-key documentation before trusting ==="
ssh-keygen -lf "$tmp"
echo "==========================================================================="
if ! { exec 3<>/dev/tty; } 2>/dev/null; then
  echo "refusing to pin without an interactive out-of-band verification" >&2
  exit 2
fi
printf 'After comparing the fingerprints, type VERIFIED to pin them: ' >&3
read -r confirmation <&3
exec 3>&-
if [[ "$confirmation" != "VERIFIED" ]]; then
  echo "host key was not pinned" >&2
  exit 2
fi

mkdir -p -- "$(dirname -- "$KNOWN_HOSTS")"
chmod 700 "$(dirname -- "$KNOWN_HOSTS")"
mv "$tmp" "$KNOWN_HOSTS"
trap - EXIT
chmod 600 "$KNOWN_HOSTS" 2>/dev/null || true
echo "pinned host key(s) -> ${KNOWN_HOSTS}"
echo "hpc_open_socket.sh will now use StrictHostKeyChecking=yes."
