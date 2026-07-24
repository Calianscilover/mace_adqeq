import os
from pathlib import Path

# This must be set before importing JAX or mace_adqeq.
os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import numpy as np
import pytest
import torch
from ase.io import read

from mace_adqeq import MACEJAXQEqCalculator


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
STRUCTURE = DATA / "STRU-2.pdb"
MACE_MODEL = DATA / "interface.model"
QEQ_PARAMS = DATA / "qeq_multihead_params.msgpack"
QEQ_CONFIG = DATA / "qeq_multihead_config.json"

TOTAL_CHARGE = float(os.environ.get("MACE_ADQEQ_TOTAL_CHARGE", "0.0"))
MAX_PAIRS = int(os.environ.get("MACE_ADQEQ_MAX_PAIRS", "0"))


def _assert_finite(name: str, value) -> None:
    array = np.asarray(value)
    assert np.all(np.isfinite(array)), f"{name} contains NaN or Inf"


def _assert_cuda_backends() -> None:
    assert torch.cuda.is_available(), (
        "PyTorch cannot access CUDA. Run this test on a GPU node with a "
        "CUDA-enabled PyTorch installation."
    )
    assert jax.default_backend() == "gpu", (
        f"JAX backend is {jax.default_backend()!r}, expected 'gpu'. "
        "Install a CUDA-enabled jaxlib/JAX plugin and request a GPU node."
    )
    assert any(device.platform == "gpu" for device in jax.devices()), (
        f"JAX did not report a GPU device: {jax.devices()}"
    )


def _assert_data_files() -> None:
    for path in (STRUCTURE, MACE_MODEL, QEQ_PARAMS, QEQ_CONFIG):
        assert path.is_file(), f"Required integration-test file is missing: {path}"
        assert path.stat().st_size > 0, f"Integration-test file is empty: {path}"


def _assert_result_shapes_and_values(atoms) -> None:
    results = atoms.calc.results
    n_atoms = len(atoms)

    assert np.asarray(results["forces"]).shape == (n_atoms, 3)
    assert np.asarray(results["charges"]).shape == (n_atoms,)
    assert np.asarray(results["chi"]).shape == (n_atoms,)
    assert np.asarray(results["hardness"]).shape == (n_atoms,)
    assert np.asarray(results["eta"]).shape == (n_atoms,)

    for name in (
        "energy",
        "forces",
        "charges",
        "short_range_energy",
        "long_range_energy",
        "short_range_forces",
        "long_range_forces",
        "chi",
        "hardness",
        "eta",
    ):
        _assert_finite(name, results[name])

    np.testing.assert_allclose(
        results["energy"],
        results["short_range_energy"] + results["long_range_energy"],
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        results["forces"],
        results["short_range_forces"] + results["long_range_forces"],
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        np.sum(results["charges"]),
        TOTAL_CHARGE,
        rtol=0.0,
        atol=5.0e-4,
    )


def _print_summary(frame: str, atoms) -> None:
    results = atoms.calc.results
    forces = np.asarray(results["forces"])
    charges = np.asarray(results["charges"])
    print(
        f"{frame}: "
        f"energy={results['energy']:.12g} eV, "
        f"max_force={np.linalg.norm(forces, axis=1).max():.12g} eV/A, "
        f"charge_sum={charges.sum():.12g} e, "
        f"solver={results['qeq_solver']}, "
        f"pg_iterations={results['qeq_pg_iterations']}, "
        f"pg_error={results['qeq_pg_error']}"
    )


@pytest.mark.integration
@pytest.mark.cuda
def test_real_cuda_calculator_and_hybrid_charge_solver():
    """Validate real models and the matrix -> projected-gradient MD flow."""
    _assert_cuda_backends()
    _assert_data_files()
    print(f"PyTorch CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"JAX devices: {jax.devices()}")

    atoms = read(STRUCTURE)
    assert len(atoms) == 288
    assert np.any(atoms.get_pbc()), "The QEq PME test structure must be periodic"
    atoms.info["total_charge"] = TOTAL_CHARGE

    atoms.calc = MACEJAXQEqCalculator(
        mace_model_path=MACE_MODEL,
        qeq_params_path=QEQ_PARAMS,
        qeq_config_path=QEQ_CONFIG,
        mace_device="cuda",
        mace_default_dtype="float32",
        qeq_options={
            "solver_mode": "hybrid",
            "max_pairs": MAX_PAIRS,
            "pg_tolerance": 1.0e-6,
            "pg_max_iterations": 500,
            "matrix_tolerance": 1.0e-8,
            "matrix_max_iterations": 8,
        },
    )

    # First frame: constrained Newton-KKT matrix solve.
    first_energy = atoms.get_potential_energy()
    first_forces = atoms.get_forces()
    _assert_finite("first_energy", first_energy)
    _assert_finite("first_forces", first_forces)
    _assert_result_shapes_and_values(atoms)
    _print_summary("first_frame", atoms)
    assert atoms.calc.results["qeq_solver"] == "matrix"
    assert len(atoms.calc.qeq_model.charge_list) == len(atoms)

    charge_list_id = id(atoms.calc.qeq_model.charge_list)

    # Next MD-like frame: reuse the previous charges as the projected-LBFGS
    # initial state. The small displacement forces ASE to recalculate.
    atoms.positions[0, 0] += 1.0e-3
    second_energy = atoms.get_potential_energy()
    second_forces = atoms.get_forces()
    _assert_finite("second_energy", second_energy)
    _assert_finite("second_forces", second_forces)
    _assert_result_shapes_and_values(atoms)
    _print_summary("second_frame", atoms)

    results = atoms.calc.results
    assert results["qeq_solver"] == "projected_gradient", (
        "The second frame did not converge with projected LBFGS; "
        f"solver={results['qeq_solver']!r}, "
        f"iterations={results['qeq_pg_iterations']!r}, "
        f"error={results['qeq_pg_error']!r}"
    )
    assert 0 < results["qeq_pg_iterations"] <= 500
    assert results["qeq_pg_error"] <= 1.0e-5

    second_charges = np.asarray(results["charges"])
    np.testing.assert_allclose(
        np.asarray(atoms.calc.qeq_model.charge_list),
        second_charges,
        rtol=0.0,
        atol=0.0,
    )
    assert id(atoms.calc.qeq_model.charge_list) == charge_list_id
