#!/usr/bin/env bash
# Real CPU numerical tests of the shared primitives, without AMReX/CUDA.
# This does NOT substitute for the production CUDA build and native tests.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
CXX=${CXX:-c++}
OUT=${OUT:-"$ROOT/tmp/gpu_primitives_cpu"}
SANITIZE=${SANITIZE:-0}
command -v "$CXX" >/dev/null || { echo "Compiler not found: $CXX" >&2; exit 2; }
if [[ "$SANITIZE" != 0 && "$SANITIZE" != 1 ]]; then
    echo 'SANITIZE must be 0 or 1' >&2; exit 2
fi
mkdir -p -- "$OUT"
OUT=$(cd -- "$OUT" && pwd)
NATIVE="$ROOT/native_amrex_rrea"
FLAGS=(-std=c++17 -O2 -g -Wall -Wextra -Wpedantic
    -I"$NATIVE/include" -I"$NATIVE/warpx_overlay/Source/Rrea")
if [[ "$SANITIZE" == 1 ]]; then
    FLAGS+=(-O1 -fsanitize=address,undefined -fno-omit-frame-pointer)
fi
run_test() {
    local name=$1
    shift
    printf '\n=== %s (%s; sanitizers=%s) ===\n' "$name" "$CXX" "$SANITIZE"
    "$CXX" "${FLAGS[@]}" "$@" -o "$OUT/$name"
    (cd -- "$OUT" && "./$name")
}
"$CXX" --version | head -n 1
run_test gpu_primitives "$NATIVE/src/RreaGpuPrimitivesSmoke.cpp"
run_test maxwell "$NATIVE/src/RreaMaxwellTMRZSmoke.cpp"
run_test segment_deposition "$NATIVE/src/RreaSegmentDepositionSmoke.cpp"
run_test gauss_legendre8 "$NATIVE/src/RreaGaussLegendre8Smoke.cpp"
run_test rng_endpoint "$NATIVE/src/RreaRngEndpointSmoke.cpp" \
    "$NATIVE/warpx_overlay/Source/Rrea/RreaRng.cpp"
run_test cic_resample "$NATIVE/src/RreaCicResampleSmoke.cpp" \
    "$NATIVE/warpx_overlay/Source/Rrea/RreaRng.cpp"
printf '\nAll six standalone CPU numerical executables passed.\n'
