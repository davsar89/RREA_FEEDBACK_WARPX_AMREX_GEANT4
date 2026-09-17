#!/usr/bin/env bash
#SBATCH --job-name=rrea_cd_build
#SBATCH --account=nn9526k
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Four CPUs schedule readily on the small partition. The LTO link sets peak
# memory; -flto=${PARALLEL} below keeps its workers tied to
# SLURM_CPUS_PER_TASK.
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=rrea_cd_build_%j.out
#SBATCH --error=rrea_cd_build_%j.err

# WarpX 26.06 + AMReX 26.06 production build and physics smokes.

set -euo pipefail
umask 027

die()
{
    echo "ERROR: $*" >&2
    exit 2
}

require_absolute_file()
{
    local label="$1"
    local value="$2"
    [[ "${value}" == /* ]] || die "${label} must be an absolute path"
    [[ -f "${value}" ]] || die "${label} is not a file: ${value}"
}

require_absolute_directory()
{
    local label="$1"
    local value="$2"
    [[ "${value}" == /* ]] || die "${label} must be an absolute path"
    [[ -d "${value}" ]] || die "${label} is not a directory: ${value}"
}

module --force purge >/dev/null 2>&1 || true
module load NRIS/CPU
module load foss/2025a
module load Python/3.13.1-GCCcore-14.2.0

for command_name in cmake git mpicc mpicxx python3 realpath rsync tar; do
    command -v "${command_name}" >/dev/null 2>&1 \
        || die "required build command is unavailable: ${command_name}"
done
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
"${PYTHON_BIN}" -c \
    'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 2)' \
    || die "Python 3.10 or newer is required for source preparation"

: "${SLURM_JOB_ID:?This launcher must run through Slurm}"
: "${RREA_SOURCE_ARCHIVE:?Set RREA_SOURCE_ARCHIVE to the project source archive}"
: "${WARPX_SRC:?Set WARPX_SRC to the clean pinned WarpX checkout}"
: "${AMREX_SRC:?Set AMREX_SRC to the clean pinned AMReX checkout}"
: "${EXPECTED_WARPX_COMMIT:?Set EXPECTED_WARPX_COMMIT from the project gitlink}"
: "${EXPECTED_AMREX_COMMIT:?Set EXPECTED_AMREX_COMMIT from the project gitlink}"
: "${RREA_INSTALL_PARENT:?Set RREA_INSTALL_PARENT to an absolute project-filesystem directory}"
: "${RREA_SMOKE_TABLE_CONFIG:?Set RREA_SMOKE_TABLE_CONFIG to transport_physics.json}"

require_absolute_file "RREA_SOURCE_ARCHIVE" "${RREA_SOURCE_ARCHIVE}"
require_absolute_file "RREA_SMOKE_TABLE_CONFIG" "${RREA_SMOKE_TABLE_CONFIG}"
require_absolute_directory "WARPX_SRC" "${WARPX_SRC}"
require_absolute_directory "AMREX_SRC" "${AMREX_SRC}"
[[ "${RREA_INSTALL_PARENT}" == /* ]] \
    || die "RREA_INSTALL_PARENT must be an absolute path"

RREA_SOURCE_ARCHIVE="$(realpath -e "${RREA_SOURCE_ARCHIVE}")"
WARPX_SRC="$(realpath -e "${WARPX_SRC}")"
AMREX_SRC="$(realpath -e "${AMREX_SRC}")"
RREA_SMOKE_TABLE_CONFIG="$(realpath -e "${RREA_SMOKE_TABLE_CONFIG}")"
mkdir -p "${RREA_INSTALL_PARENT}"
RREA_INSTALL_PARENT="$(realpath -e "${RREA_INSTALL_PARENT}")"


mapfile -t source_commit_lines < <(
    tar -xOf "${RREA_SOURCE_ARCHIVE}" CODEX_SOURCE_COMMIT
)
[[ "${#source_commit_lines[@]}" -eq 1 ]] \
    || die "archive CODEX_SOURCE_COMMIT must contain exactly one line"
SOURCE_COMMIT="${source_commit_lines[0]%$'\r'}"
[[ "${SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
    || die "archive CODEX_SOURCE_COMMIT must be exactly 40 lowercase hexadecimal characters"


[[ "${EXPECTED_WARPX_COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
    || die "EXPECTED_WARPX_COMMIT must be a full lowercase Git commit"
[[ "${EXPECTED_AMREX_COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
    || die "EXPECTED_AMREX_COMMIT must be a full lowercase Git commit"
[[ "$(git -C "${WARPX_SRC}" rev-parse HEAD)" == "${EXPECTED_WARPX_COMMIT}" ]] \
    || die "WARPX_SRC is not the project-pinned WarpX commit"
[[ "$(git -C "${AMREX_SRC}" rev-parse HEAD)" == "${EXPECTED_AMREX_COMMIT}" ]] \
    || die "AMREX_SRC is not the project-pinned AMReX commit"
[[ -z "$(git -C "${WARPX_SRC}" status --porcelain --untracked-files=all)" ]] \
    || die "WARPX_SRC must be a clean checkout"
[[ -z "$(git -C "${AMREX_SRC}" status --porcelain --untracked-files=all)" ]] \
    || die "AMREX_SRC must be a clean checkout"

SCRATCH_BASE="${LOCALSCRATCH:-${SCRATCH:-/localscratch/${SLURM_JOB_ID}}}"
[[ "${SCRATCH_BASE}" == /* ]] || die "Slurm scratch path must be absolute"
BUILD_ROOT="${SCRATCH_BASE}/rrea_cd_build_${SLURM_JOB_ID}"
[[ ! -e "${BUILD_ROOT}" ]] || die "refusing to reuse build root: ${BUILD_ROOT}"
mkdir -p "${BUILD_ROOT}"

PREFIX_ROOT="${RREA_INSTALL_PARENT}/rrea_cd_26_06_${SOURCE_COMMIT:0:12}_job${SLURM_JOB_ID}"
[[ ! -e "${PREFIX_ROOT}" ]] || die "refusing to reuse install prefix: ${PREFIX_ROOT}"
mkdir -p "${PREFIX_ROOT}"

ARCHIVE_CHECKOUT="${BUILD_ROOT}/source_archive_checkout"
mkdir -p "${ARCHIVE_CHECKOUT}"
tar -xf "${RREA_SOURCE_ARCHIVE}" -C "${ARCHIVE_CHECKOUT}"
REPO_ROOT="${ARCHIVE_CHECKOUT}"
[[ "$(<"${REPO_ROOT}/CODEX_SOURCE_COMMIT")" == "${SOURCE_COMMIT}" ]] \
    || die "extracted CODEX_SOURCE_COMMIT differs from archive marker"

PARALLEL="${SLURM_CPUS_PER_TASK:-16}"
OPT_FLAGS="${RREA_BUILD_OPT_FLAGS:--O3 -g -DNDEBUG -march=native}"
CMAKE_GENERATOR="${RREA_CMAKE_GENERATOR:-Unix Makefiles}"

PREPARED_SOURCE_ROOT="${PREFIX_ROOT}/prepared_sources"
RREA_WARPX_SRC="${PREPARED_SOURCE_ROOT}/WarpX-26.06"
PREPARED_AMREX_SRC="${PREPARED_SOURCE_ROOT}/AMReX-26.06"
EVIDENCE_DIR="${PREFIX_ROOT}/evidence"
WARPX_PREFIX="${PREFIX_ROOT}/warpx-26.06-rz-mpi"
SMOKE_DIR="${PREFIX_ROOT}/smoke"
mkdir -p "${PREPARED_SOURCE_ROOT}" "${EVIDENCE_DIR}" "${SMOKE_DIR}"

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/prepare_rrea_warpx_source.py" \
    --warpx-source "${WARPX_SRC}" \
    --output-dir "${RREA_WARPX_SRC}" \
    --overlay-dir "${REPO_ROOT}/native_amrex_rrea" \
    --force
rsync -a --delete --exclude='.git/' "${AMREX_SRC}/" "${PREPARED_AMREX_SRC}/"

MPI_CXX_COMMAND="$(command -v mpicxx)"
COMPILER_IDENTITY="${EVIDENCE_DIR}/compiler_identity.txt"
MODULE_ENVIRONMENT="${EVIDENCE_DIR}/module_environment.txt"
{
    "${MPI_CXX_COMMAND}" --version
    "${PYTHON_BIN}" --version
    cmake --version
    git --version
} > "${COMPILER_IDENTITY}"
{
    echo "required_module_stack=NRIS/CPU,foss/2025a"
    echo "loaded_modules_begin"
    module -t list 2>&1 || true
    echo "loaded_modules_end"
    echo "SOURCE_COMMIT=${SOURCE_COMMIT}"
    echo "WARPX_UPSTREAM_COMMIT=${EXPECTED_WARPX_COMMIT}"
    echo "AMREX_UPSTREAM_COMMIT=${EXPECTED_AMREX_COMMIT}"
    echo "CMAKE_GENERATOR=${CMAKE_GENERATOR}"
    echo "OPT_FLAGS=${OPT_FLAGS}"
    echo "PARALLEL=${PARALLEL}"
} > "${MODULE_ENVIRONMENT}"

BUILD_LOG="${EVIDENCE_DIR}/build.log"
{
    echo "Job ${SLURM_JOB_ID} on $(hostname)"
    date --iso-8601=seconds
    echo "SOURCE_COMMIT=${SOURCE_COMMIT}"
    echo "PREFIX_ROOT=${PREFIX_ROOT}"
    echo "BUILD_ROOT=${BUILD_ROOT}"

    cmake -S "${RREA_WARPX_SRC}" -B "${BUILD_ROOT}/warpx" \
        -G "${CMAKE_GENERATOR}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${WARPX_PREFIX}" \
        -DCMAKE_C_COMPILER=mpicc \
        -DCMAKE_CXX_COMPILER=mpicxx \
        -DCMAKE_C_FLAGS="-pthread" \
        -DCMAKE_CXX_FLAGS="-pthread" \
        -DCMAKE_C_FLAGS_RELEASE="${OPT_FLAGS}" \
        -DCMAKE_CXX_FLAGS_RELEASE="${OPT_FLAGS}" \
        -DCMAKE_EXE_LINKER_FLAGS="-pthread -flto=${PARALLEL}" \
        -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON \
        -DWarpX_DIMS=RZ \
        -DWarpX_PRECISION=DOUBLE \
        -DWarpX_PARTICLE_PRECISION=DOUBLE \
        -DWarpX_COMPUTE=OMP \
        -DWarpX_MPI=ON \
        -DWarpX_MPI_THREAD_MULTIPLE=ON \
        -DWarpX_OPENPMD=OFF \
        -DWarpX_QED=OFF \
        -DWarpX_FFT=OFF \
        -DWarpX_PYTHON=OFF \
        -DWarpX_EB=ON \
        -DWarpX_APP=ON \
        -DWarpX_LIB=ON \
        -DBUILD_TESTING=OFF \
        -DWarpX_amrex_internal=ON \
        -DWarpX_amrex_src="${PREPARED_AMREX_SRC}" \
        -DRREA_AMREX_OVERLAY="${REPO_ROOT}/native_amrex_rrea" \
        -DRREA_AMREX_BUILD_TESTS=ON \
        -DRREA_SMOKE_CONFIG="${RREA_SMOKE_TABLE_CONFIG}" \
        -DRREA_PRODUCTION_TRANSPORT_CONFIG="${RREA_SMOKE_TABLE_CONFIG}" \
        -DRREA_AMREX_ENABLE_IPO=ON \
        -DRREA_AMREX_INSTALL=OFF

    cmake --build "${BUILD_ROOT}/warpx" --parallel "${PARALLEL}"
    cmake --install "${BUILD_ROOT}/warpx"
} 2>&1 | tee "${BUILD_LOG}"

WARPX_EXE="$(
    find "${WARPX_PREFIX}/bin" -maxdepth 1 -type f -name 'warpx.rz*' -perm -u+x \
        | LC_ALL=C sort \
        | head -1
)"
[[ -n "${WARPX_EXE}" ]] || die "installed WarpX RZ executable was not found"
ln -sfn "${WARPX_EXE}" "${WARPX_PREFIX}/bin/warpx.rz"

export OMP_NUM_THREADS="${SMOKE_OMP_THREADS:-2}"
ctest --test-dir "${BUILD_ROOT}/warpx" --output-on-failure --no-tests=error -j1

{
    echo "BUILD_AND_SMOKE_STATUS=PASS"
    date --iso-8601=seconds
} >> "${BUILD_LOG}"

echo "SOURCE_COMMIT=${SOURCE_COMMIT}"
echo "PREPARED_SOURCE_ROOT=${PREPARED_SOURCE_ROOT}"
echo "WARPX_EXE=${WARPX_EXE}"
echo "PREFIX_ROOT=${PREFIX_ROOT}"
