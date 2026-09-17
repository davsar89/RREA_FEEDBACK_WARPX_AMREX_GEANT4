#!/usr/bin/env bash
# One-time: pin OLIVIA's SSH host key so olivia_open_socket.sh can require
# StrictHostKeyChecking=yes without accept-new.
#
# Run this ONCE from a trusted network. VERIFY the printed fingerprints
# out-of-band against Sigma2/NRIS documentation before relying on the pin --
# ssh-keyscan itself is TOFU, so this only moves the trust decision to a
# single supervised moment instead of every first connect.
#
#   wsl -e bash /mnt/c/Users/david/Desktop/rrea_warpx_amrex/scripts/olivia_pin_hostkey.sh
set -euo pipefail

LOGIN="${OLIVIA_LOGIN:-dsarria@olivia.sigma2.no}"
HOST="${LOGIN#*@}"
STATE_DIR="${OLIVIA_STATE_DIR:-${HOME}/.local/share/rrea_warpx_amrex/olivia}"
KNOWN_HOSTS="${OLIVIA_KNOWN_HOSTS:-${STATE_DIR}/known_hosts}"

echo "scanning host key(s) for ${HOST} ..."
tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT
ssh-keyscan -T 15 "$HOST" > "$tmp" 2>/dev/null || true
if [[ ! -s "$tmp" ]]; then
  echo "ssh-keyscan returned nothing for ${HOST} (network/DNS?)" >&2
  exit 1
fi

echo "=== fingerprints -- VERIFY these against Sigma2/NRIS docs before trusting ==="
ssh-keygen -lf "$tmp"
echo "==========================================================================="
echo "Official reference: https://documentation.sigma2.no/getting_started/fingerprints.html"
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
echo "olivia_open_socket.sh will now use StrictHostKeyChecking=yes."
