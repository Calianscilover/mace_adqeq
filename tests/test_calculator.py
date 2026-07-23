from types import SimpleNamespace

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator

import mace_adqeq.calculator as calculator_module
from mace_adqeq import MACEJAXQEqCalculator


class FakeMACECalculator(Calculator):
    implemented_properties = ["energy", "forces"]
    init_kwargs = None

    def __init__(self, **kwargs):
        super().__init__()
        type(self).init_kwargs = kwargs

    def calculate(self, atoms=None, properties=None, system_changes=None):
        self.results = {
            "energy": 2.5,
            "forces": np.full((len(atoms), 3), 0.25),
        }


class FakeQEqModel:
    init_kwargs = None

    def __init__(self, **kwargs):
        type(self).init_kwargs = kwargs

    def calculate(self, atoms, total_charge):
        n_atoms = len(atoms)
        charges = np.full(n_atoms, total_charge / n_atoms)
        return SimpleNamespace(
            energy=1.5,
            forces=np.full((n_atoms, 3), -0.1),
            charges=charges,
            chi=np.arange(n_atoms, dtype=float),
            hardness=np.full(n_atoms, 2.0),
            eta=np.full(n_atoms, 1.0),
        )


def make_calculator(monkeypatch, **kwargs):
    monkeypatch.setattr(
        calculator_module, "MACECalculator", FakeMACECalculator
    )
    monkeypatch.setattr(calculator_module, "JAXQEqModel", FakeQEqModel)
    return MACEJAXQEqCalculator(
        mace_model_path="mace.model",
        qeq_params_path="qeq.msgpack",
        qeq_config_path="qeq.json",
        **kwargs,
    )


def test_combines_mace_and_qeq_results(monkeypatch):
    atoms = Atoms("HO", positions=[[0, 0, 0], [0, 0, 1]], cell=[10, 10, 10])
    atoms.info["total_charge"] = 1.0
    atoms.calc = make_calculator(monkeypatch)

    assert atoms.get_potential_energy() == 4.0
    np.testing.assert_allclose(atoms.get_forces(), 0.15)
    np.testing.assert_allclose(atoms.calc.results["partial_charges"], [0.5, 0.5])
    assert atoms.calc.results["short_range_energy"] == 2.5
    assert atoms.calc.results["long_range_energy"] == 1.5


def test_constructs_official_mace_and_qeq_models(monkeypatch):
    calculator = make_calculator(
        monkeypatch,
        mace_device="cpu",
        mace_default_dtype="float64",
        qeq_options={"dipole_axis": 2},
    )

    assert isinstance(calculator.mace_calculator, FakeMACECalculator)
    assert isinstance(calculator.qeq_model, FakeQEqModel)
    assert FakeMACECalculator.init_kwargs["device"] == "cpu"
    assert FakeMACECalculator.init_kwargs["default_dtype"] == "float64"
    assert FakeQEqModel.init_kwargs["dipole_axis"] == 2


def test_requires_explicit_total_charge(monkeypatch):
    atoms = Atoms("H", positions=[[0, 0, 0]], cell=[10, 10, 10])
    atoms.calc = make_calculator(monkeypatch)

    try:
        atoms.get_potential_energy()
    except ValueError as exc:
        assert "total_charge" in str(exc)
    else:
        raise AssertionError("Expected missing total_charge to raise ValueError")


def test_default_total_charge(monkeypatch):
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]], cell=[10, 10, 10])
    atoms.calc = make_calculator(monkeypatch, default_total_charge=-1.0)

    assert atoms.get_potential_energy() == 4.0
    np.testing.assert_allclose(atoms.calc.results["charges"], [-0.5, -0.5])
