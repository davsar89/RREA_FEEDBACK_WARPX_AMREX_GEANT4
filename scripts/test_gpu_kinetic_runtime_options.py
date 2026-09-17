"""Execution controls must work without opening protected physics overrides."""
import pytest
from rrea_run_support import validate_user_warpx_overrides


def test_gpu_execution_controls_allowed():
    options = [
        "amrex.the_arena_is_managed=1",
        "rrea.kinetic_backend=gpu",
        "rrea.gpu_kinetic_batch_size=8192",
        "rrea.gpu_kinetic_heap_mib=1024",
        "rrea.gpu_population_batch_size=4096",
        "rrea.gpu_population_cpu_fallback=0",
        "rrea.gpu_maxwell=0",
        "rrea.gpu_fluid=1",
    ]
    assert validate_user_warpx_overrides(options) == options
    assert validate_user_warpx_overrides(["rrea.kinetic_backend=cpu"]) == [
        "rrea.kinetic_backend=cpu"
    ]


@pytest.mark.parametrize("option", [
    "rrea.kinetic_backend=automatic",
    "rrea.kinetic_backend=gpu rrea.low_energy_cutoff_eV=0",
    "rrea.gpu_kinetic_batch_size=0",
    "rrea.gpu_kinetic_batch_size=1048577",
    "rrea.gpu_kinetic_heap_mib=7",
    "rrea.gpu_kinetic_heap_mib=2147483648",
    "rrea.gpu_kinetic_heap_mib=1.5",
    "rrea.gpu_population_batch_size=65537",
    "rrea.gpu_population_batch_size=-1",
    "rrea.gpu_maxwell=2",
    "rrea.gpu_fluid=-1",
    "rrea.gpu_population_cpu_fallback=2",
    "rrea.gpu_population_cpu_fallback=1 particles.nspecies=0",
    "rrea.gpu_kinetic_batch_size=١٢",
])
def test_invalid_gpu_controls_rejected(option):
    with pytest.raises(SystemExit):
        validate_user_warpx_overrides([option])


@pytest.mark.parametrize("option", [
    "rrea.low_energy_cutoff_eV=0",
    "rrea.strict_transport_guards=0",
    "rrea.population_ceiling_policy=strict_abort_v1",
    "rrea.gpu_unknown_physics_override=1",
    "geometry.prob_hi=100 100",
])
def test_physics_and_unknown_rrea_controls_still_protected(option):
    with pytest.raises(SystemExit, match="protected production key"):
        validate_user_warpx_overrides([option])
