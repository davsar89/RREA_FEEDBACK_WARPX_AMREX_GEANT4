#!/usr/bin/env bash
# Open (or reuse) a reusable SSH ControlMaster socket to OLIVIA noninteractively.
#
# Normally invoked automatically by scripts/olivia.py (ensure_socket). To run
# by hand, use a Windows-side shell (NOT git-bash — `wsl ssh` exits 255 there)
# and set the env var INSIDE WSL (Windows env does not cross the boundary):
#   wsl -e env OLIVIA_TOTP_AUTOMATED=1 bash scripts/olivia_open_socket.sh
#
# Requires scripts/olivia_askpass.py plus a private credentials.env (password).
# TOTP is interactive by default. OLIVIA_TOTP_AUTOMATED=1 uses the local
# gitignored setup key or QR only when the user explicitly authorizes that mode.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT="${OLIVIA_LOCAL_PROJECT:-$(cd -- "${SCRIPT_DIR}/.." && pwd -P)}"
export OLIVIA_LOCAL_PROJECT="${PROJECT}"
STATE_DIR="${OLIVIA_STATE_DIR:-${HOME}/.local/share/rrea_warpx_amrex/olivia}"
export OLIVIA_STATE_DIR="${STATE_DIR}"
SOCK="${OLIVIA_SOCK:-/tmp/codex_olivia_dsarria_mux}"
LOGIN="${OLIVIA_LOGIN:-dsarria@olivia.sigma2.no}"
# Host-key pinning: refuse anything but the out-of-band-verified local pin so
# a first-connect MITM cannot harvest a live password + TOTP. There is
# intentionally no accept-new/TOFU fallback.
if [[ -n "${OLIVIA_KNOWN_HOSTS:-}" ]]; then
  KNOWN_HOSTS="${OLIVIA_KNOWN_HOSTS}"
elif [[ -s "${STATE_DIR}/known_hosts" ]]; then
  KNOWN_HOSTS="${STATE_DIR}/known_hosts"
else
  # Compatibility fallback for existing workspaces. New installations pin
  # into STATE_DIR so reusable authentication material is not kept on drvfs.
  KNOWN_HOSTS="${PROJECT}/.olivia_known_hosts"
fi
if [[ -s "$KNOWN_HOSTS" ]]; then
  HK_OPTS=(-o "UserKnownHostsFile=$KNOWN_HOSTS" -o StrictHostKeyChecking=yes)
else
  echo "refusing OLIVIA connection: no verified host-key pin at $KNOWN_HOSTS." >&2
  echo "Run scripts/olivia_pin_hostkey.sh interactively and compare against" \
       "the official Sigma2 fingerprint page first." >&2
  exit 2
fi

# Require both a live local master process and a successful remote round trip.
# The second command is BatchMode-only and pinned, so it cannot fall back to
# an unexpected password/TOTP prompt if the socket became stale.
socket_is_usable() {
  ssh -S "$SOCK" -O check "$LOGIN" >/dev/null 2>&1 \
    && ssh -S "$SOCK" \
      -o BatchMode=yes \
      -o ConnectTimeout=10 \
      "${HK_OPTS[@]}" \
      "$LOGIN" true >/dev/null 2>&1
}
if socket_is_usable; then
  echo "socket already alive: $SOCK"
  exit 0
fi

# Install short-lived askpass material in mode-0600/0700 files. Secret-bearing
# environment overrides are copied to this handoff and scrubbed before the
# long-lived SSH master starts; the files are removed as this opener exits.
ASKPASS="$(mktemp /tmp/olivia_askpass.XXXXXX.py)"
CREDENTIAL_FILE="$(mktemp /tmp/olivia_askpass_credentials.XXXXXX)"
cleanup() {
  rm -f -- "$ASKPASS" "$CREDENTIAL_FILE"
}
trap cleanup EXIT

cp "$PROJECT/scripts/olivia_askpass.py" "$ASKPASS"
sed -i 's/\r$//' "$ASKPASS"
chmod 700 "$ASKPASS"
chmod 600 "$CREDENTIAL_FILE"

write_credential() {
  local key="$1"
  local value="$2"
  if [[ "$value" == *$'\n'* || "$value" == *$'\r'* ]]; then
    echo "refusing multiline OLIVIA credential value for $key." >&2
    exit 3
  fi
  printf '%s=%s\n' "$key" "$value" >>"$CREDENTIAL_FILE"
}

if [[ -n "${OLIVIA_PASSWORD:-}" ]]; then
  write_credential OLIVIA_PASSWORD "$OLIVIA_PASSWORD"
fi
if [[ -n "${OLIVIA_TOTP_SECRET:-}" ]]; then
  write_credential OLIVIA_TOTP_SECRET "$OLIVIA_TOTP_SECRET"
fi

rm -f "$SOCK"
export SSH_ASKPASS="$ASKPASS"
export SSH_ASKPASS_REQUIRE=force
export DISPLAY=:0
export OLIVIA_ASKPASS_CREDENTIAL_FILE="$CREDENTIAL_FILE"

# TOTP is prompted once per ControlPersist window unless the explicitly
# authorized local setup-key/QR mode is selected.
if [[ "${OLIVIA_TOTP_AUTOMATED:-}" == "1" ]]; then
  export OLIVIA_TOTP_AUTOMATED
elif [[ -n "${OLIVIA_TOTP_CODE:-}" ]]; then
  if [[ ! "$OLIVIA_TOTP_CODE" =~ ^([0-9]{6}|[0-9]{8})$ ]]; then
    echo "OLIVIA_TOTP_CODE must contain six or eight digits." >&2
    exit 3
  fi
  write_credential OLIVIA_TOTP_CODE "$OLIVIA_TOTP_CODE"
elif { exec 3<>/dev/tty; } 2>/dev/null; then
  printf 'OLIVIA 6-digit TOTP code (from your authenticator app): ' >&3
  read -r -s OLIVIA_TOTP_CODE <&3
  printf '\n' >&3
  exec 3>&-
  if [[ ! "$OLIVIA_TOTP_CODE" =~ ^([0-9]{6}|[0-9]{8})$ ]]; then
    echo "OLIVIA TOTP code must contain six or eight digits." >&2
    exit 3
  fi
  write_credential OLIVIA_TOTP_CODE "$OLIVIA_TOTP_CODE"
else
  echo "No TOTP available: run attached to a terminal, supply transient" \
       "OLIVIA_TOTP_CODE, or explicitly authorize OLIVIA_TOTP_AUTOMATED=1." >&2
  exit 3
fi

# Askpass recovers one-shot overrides from the mode-0600 handoff.
# Do not let the daemonized ControlMaster inherit raw credentials.
unset OLIVIA_PASSWORD OLIVIA_TOTP_SECRET OLIVIA_TOTP_CODE

setsid ssh -fN -M -S "$SOCK" \
  -o ControlPersist=8h \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  "${HK_OPTS[@]}" \
  -o PreferredAuthentications=keyboard-interactive,password \
  -o PubkeyAuthentication=no \
  "$LOGIN"

unset OLIVIA_TOTP_AUTOMATED
sleep 6
if ssh -S "$SOCK" \
    -o BatchMode=yes \
    -o ConnectTimeout=10 \
    "${HK_OPTS[@]}" \
    "$LOGIN" hostname 2>/dev/null; then
  echo "socket opened: $SOCK"
else
  echo "FAILED to open socket. If auth was refused, check: (1) credentials.env password," \
       "(2) WSL clock skew vs real UTC, (3) wait for the next 30s TOTP window." >&2
  exit 1
fi
