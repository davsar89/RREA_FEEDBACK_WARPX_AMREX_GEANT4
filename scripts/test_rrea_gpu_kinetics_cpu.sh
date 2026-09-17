#!/usr/bin/env bash
# Actual scalar physics/transaction tests. This cannot certify a CUDA build:
# no fake ParallelFor, mesh, particle container or GPU runtime is provided.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
CXX=${CXX:-c++}
OUT=${OUT:-"$ROOT/tmp/kinetic_cpu_tests"}
SANITIZE=${SANITIZE:-0}
BINARY64=${BINARY64:-0}
for option in "$SANITIZE" "$BINARY64"; do
    [[ "$option" == 0 || "$option" == 1 ]] || { echo 'SANITIZE and BINARY64 must be 0 or 1' >&2; exit 2; }
done
command -v "$CXX" >/dev/null || { echo "Compiler not found: $CXX" >&2; exit 2; }
mkdir -p -- "$OUT"; OUT=$(cd -- "$OUT" && pwd)
N="$ROOT/native_amrex_rrea"
CONFIG=${CONFIG:-"$ROOT/rrea_transport_tables/schema6/production/transport_physics.json"}
[[ -f "$CONFIG" ]] || { echo "Missing table config: $CONFIG" >&2; exit 2; }
FLAGS=(-std=c++20 -O2 -g -Wall -Wextra -Wpedantic -DRREA_SCALAR_TEST_ADAPTERS=1
    -I"$N/tests/scalar_types" -I"$N/include" -I"$N/warpx_overlay/Source/Rrea")
if [[ "$SANITIZE" == 1 ]]; then FLAGS+=(-O1 -fsanitize=address,undefined -fno-omit-frame-pointer); fi
if [[ "$BINARY64" == 1 ]]; then FLAGS+=(-DRREA_KINETIC_TEST_BINARY64=1); fi
ulimit -c 0
"$CXX" --version | head -n1
printf 'Scalar adapters only; sanitizers=%s; binary64 working arithmetic=%s; NO CUDA execution\n' "$SANITIZE" "$BINARY64"
"$CXX" "${FLAGS[@]}" -c "$N/src/RreaInteractionTables.cpp" -o "$OUT/interaction_tables.o"
"$CXX" "${FLAGS[@]}" -c "$N/tests/scalar_types/DefaultDebugOptions.cpp" -o "$OUT/default_debug.o"
run() {
    local name=$1 source=$2 tables=$3; shift 3
    printf '\n=== %s ===\n' "$name"
    local extra=()
    if [[ "$tables" == 1 ]]; then extra=("$OUT/interaction_tables.o" "$OUT/default_debug.o" -lz); fi
    "$CXX" "${FLAGS[@]}" "$N/$source" "${extra[@]}" -o "$OUT/$name"
    (cd -- "$ROOT"; "$OUT/$name" "$@")
}
run interaction_tables src/RreaInteractionTablesSmoke.cpp 1 "$CONFIG"
run scattering_sampler src/RreaScatteringSamplerSmoke.cpp 1 "$CONFIG"
run kinetic_engine src/RreaKineticEngineSmoke.cpp 1 "$CONFIG"
run population_gpu src/RreaPopulationGpuSmoke.cpp 0
run kinetic_storage src/RreaKineticStorageSmoke.cpp 0
run transport_path src/RreaTransportPathSmoke.cpp 0
run secondary_groups src/RreaSecondaryGroupSmoke.cpp 0
run charged_interleave warpx_overlay/Source/Rrea/RreaChargedInterleaveSmoke.cpp 0
run cic_resample src/RreaCicResampleSmoke.cpp 0
printf '\nAll nine CPU scalar/table/transport/population executables passed.\n'
