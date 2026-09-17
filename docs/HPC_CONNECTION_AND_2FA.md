# HPC Connection and 2FA

This guide covers host-key pinning, password + TOTP, and the reusable SSH
socket for the remote HPC system used for production captures. Post-connection
commands are in `docs/HPC_COMMAND_WORKFLOW_NOTES.md`; sensitive local notes
stay in `REMOTE_HPC.md`.

No password, TOTP seed, live six-digit code, private key, or
credential-bearing URL belongs in this file, Git, a source archive, a terminal
transcript, or a Slurm log.

The repository is deliberately site-agnostic: it names no cluster, login, or
project. Everything site-specific lives in the private local state directory
or `HPC_*` environment variables.

## Authentication method at a glance

Direct interactive HPC login is **account password + a TOTP 2FA code** over
keyboard-interactive SSH. This helper does not use private-key login: the
opener forces `PubkeyAuthentication=no` and
`PreferredAuthentications=keyboard-interactive,password`, so no key pair is used,
generated, or needed for this workflow. The only "key" in the login itself is the **TOTP seed**
inside your authenticator app.

The TOTP can be supplied three ways; the first is the default:

1. **Interactive code (default).** You type the current 6-digit code at the
   opener's prompt. Nothing secret is stored on disk.
2. **Base32 setup key** in the private `credentials.env` as
   `HPC_TOTP_SECRET` (automated mode).
3. **Authenticator QR export** in the private `authenticator_qr.jpg`
   (automated mode).

The QR image and the Base32 setup key are two encodings of the *same* seed, not
separate methods; the setup key wins if both are present. Automated codes (2 and
3) run only with explicit authorization via `HPC_TOTP_AUTOMATED=1`, because
holding both factors on one machine weakens 2FA separation. A human driving
interactively uses method 1; unattended sessions use the authorized automated
path.

Do not confuse this with the **host-key pin** (`known_hosts`, below):
that authenticates *the server to you* and is not one of your login credentials.

## Local paths and ignored secret files

```text
SSH socket:    /tmp/rrea_hpc_mux
login:         HPC_LOGIN (from the environment or credentials.env)
private state: ~/.local/share/rrea_warpx_amrex/hpc
```

Reusable credentials and the host-key pin should live in that WSL-native
private state directory:

```text
credentials.env       site configuration: login, work root, Slurm account,
                      module stack, account password, optional TOTP setup key
authenticator_qr.jpg  alternative authenticator export QR
known_hosts           out-of-band-verified server key
```

Use mode `0700` for the directory and `0600` for the files. The helpers prefer
these paths. Existing ignored checkout files (`hpc.env`,
`QR_hpc_ssh.jpg`, `REMOTE_HPC.md`, and
`.hpc_known_hosts`) remain compatibility inputs, but Windows drvfs mode bits
are not an access-control boundary. None may be archived or committed.

`credentials.env` holds both the non-secret site configuration the helpers
forward to the cluster and the secrets the askpass reads:

```text
HPC_LOGIN=user@hpc.example.org
REMOTE_ROOT=/absolute/work/root/on/the/cluster
HPC_SLURM_ACCOUNT=<slurm account/project, optional>
HPC_SLURM_MODULES=<space-separated module stack, optional>
HPC_PASSWORD=<account password>
HPC_TOTP_SECRET=<optional Base32 authenticator setup key>
```

`REMOTE_ROOT` is the absolute remote directory that holds `repo_amrex_latest/`,
`runs/`, `logs/`, and the engine builds; `HPC_SLURM_ACCOUNT` and
`HPC_SLURM_MODULES` configure the sbatch account and the module stack the
cluster jobs load. None of these names a specific resource in the repository.

## Authentication components (what the scripts do)

- `scripts/hpc_pin_hostkey.sh` — scans and prints candidate host keys for
  out-of-band comparison; writes the private `known_hosts` only after you
  accept.
- `scripts/hpc_open_socket.sh` — opens (or reuses) the ControlMaster socket
  with `StrictHostKeyChecking=yes` against the pin (no `accept-new`/TOFU
  fallback), password via the askpass, and the TOTP path selected as described
  under "Authentication method at a glance". It verifies a remote round trip
  before declaring the socket usable.
- `scripts/hpc_askpass.py` — `SSH_ASKPASS` helper: answers the password and
  OATH/OTP prompts from local ignored/private files only, never from a
  command line or transcript. The script is the authority on credential
  precedence, and the opener on how one-shot material is handed over, scrubbed,
  and removed after authentication.
- `scripts/hpc.py` — the one remote-work helper; reads only the non-secret
  site values (`HPC_LOGIN`, `REMOTE_ROOT`, `HPC_SLURM_ACCOUNT`,
  `HPC_SLURM_MODULES`) and forwards them to remote scripts and sbatch jobs.

## Authentication policy

1. Verify the HPC host's key fingerprints out of band against your HPC
   provider's current official host-key fingerprint documentation.
   Do not rely on `ssh-keyscan` alone; stop on any mismatch.
2. Pin only the verified key with `scripts/hpc_pin_hostkey.sh`.
3. Require `StrictHostKeyChecking=yes`; the opener intentionally has no
   `accept-new` fallback.
4. Open exactly one ControlMaster socket with `scripts/hpc_open_socket.sh`.
5. No factor — seed, setup key, QR payload, or live code — is ever printed,
   logged, archived, or committed: interactive input stays silent and automated
   codes are computed in memory.

## Private WSL state for automated TOTP

Prepare the canonical state once from WSL. Do not paste real values into a
tracked file or a command transcript:

```bash
install -d -m 700 ~/.local/share/rrea_warpx_amrex/hpc
install -m 600 /path/to/local/credentials.env \
  ~/.local/share/rrea_warpx_amrex/hpc/credentials.env
install -m 600 /path/to/local/authenticator_qr.jpg \
  ~/.local/share/rrea_warpx_amrex/hpc/authenticator_qr.jpg
```

If `HPC_TOTP_SECRET` is omitted, the opener falls back to the authenticator
export `authenticator_qr.jpg`. The real setup key and
QR payload never belong in documentation. QR automation requires `zbarimg`
(package `zbar-tools`, preferred) or Python OpenCV. Check the dependency without
decoding or printing the QR payload:

```powershell
wsl -e zbarimg --version
# If zbarimg is unavailable, this is the supported fallback:
wsl -e python3 -c "import cv2"
```

If its socket has dropped, `scripts/hpc.py` invokes the opener with
`HPC_TOTP_AUTOMATED=1`; using that helper with stored factors therefore
authorizes an automated reopen. Direct opener calls remain interactive unless
the flag is supplied explicitly.

If your authenticator holds several TOTP accounts, set `HPC_TOTP_LABEL_CONTAINS`
to a substring that selects the right one.

Explicit `HPC_ENV_FILE`, `HPC_QR_FILE`, `HPC_KNOWN_HOSTS`, and
`HPC_STATE_DIR` overrides remain available.

## Connect from PowerShell

Set the checkout path once, then invoke the helpers from PowerShell (Git Bash
path-mangles the WSL script path):

```powershell
$project = "/mnt/c/Users/david/Desktop/rrea_warpx_amrex"

# 1. Pin the host key (scan + print only; compare with your provider's
#    official fingerprint page, stop on mismatch)
wsl -e bash "$project/scripts/hpc_pin_hostkey.sh"

# 2a. Default: open one reusable socket and prompt for the TOTP
wsl -e bash "$project/scripts/hpc_open_socket.sh"

# 2b. Only after explicit authorization: use the ignored setup key or QR
wsl -e env HPC_TOTP_AUTOMATED=1 bash "$project/scripts/hpc_open_socket.sh"
```

Authentication runs only when no usable ControlMaster socket exists. Raw
credential environment variables are scrubbed
before the daemonized master starts; only the requested askpass answer reaches
SSH. WSL clock drift breaks TOTP — if login fails, check WSL clock skew vs real UTC
(verify with .NET `UtcNow`, not PowerShell 5.1 `Get-Date -UFormat %s`, which is
off by the timezone offset), the `credentials.env` password, and the 30 s TOTP
window.

## Is the socket and remote session alive? / reopen

Run both checks. `ProxyCommand=false` makes the second command fail instead of
opening a fresh connection if the master disappears:

```powershell
wsl -e ssh -S /tmp/rrea_hpc_mux -O check "$HPC_LOGIN"
wsl -e ssh -S /tmp/rrea_hpc_mux -o ProxyCommand=false -o BatchMode=yes "$HPC_LOGIN" hostname
```

Replace `$HPC_LOGIN` with the configured `user@host`. Reopen with step 2 above
(a `wsl --shutdown`, laptop suspend, or the 8 h
`ControlPersist` expiry drops the socket; WSL `/tmp` is also wiped on
`wsl --shutdown`, so the askpass is reinstalled by the opener).

## Close the socket (optional)

```powershell
wsl -e ssh -S /tmp/rrea_hpc_mux -O exit "$HPC_LOGIN"
```

`ControlPersist` closes it after 8 h on its own; leaving it open avoids
consuming another TOTP. The opener removes its temporary askpass material
automatically after authentication.