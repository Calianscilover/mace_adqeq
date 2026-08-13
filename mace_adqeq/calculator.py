from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from mace.calculators import MACECalculator

from .qeq import JAXQEqModel, QEqParameterPredictor


class MACEJAXQEqCalculator(Calculator):

    name = "MACEJAXQEq"
    implemented_properties = [
        "energy",
        "free_energy",
        "forces",
        "charges",
        "partial_charges",
        "short_range_energy",
        "long_range_energy",
        "short_range_forces",
        "long_range_forces",
        "chi",
        "hardness",
        "eta",
    ]

    def __init__(
        self,
        mace_model_path: str | Path,
        qeq_params_path: str | Path,
        qeq_config_path: str | Path,
        *,
        mace_device: str = "cuda",
        mace_default_dtype: str = "float32",
        mace_compile_mode=None,
        qeq_options: Optional[dict] = None,
        mode : int = 0,
        const_potential : bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.mode = mode
        self.const_potential = const_potential

        self.mace_calculator = MACECalculator(
            model_paths=str(Path(mace_model_path).expanduser().resolve()),
            device=mace_device,
            default_dtype=mace_default_dtype,
            compile_mode=mace_compile_mode,
        )
        qeq_options = dict(qeq_options or {})
        self.parameter_predictor = QEqParameterPredictor(
            params_path=qeq_params_path,
            config_path=qeq_config_path,
            n_jobs=qeq_options.pop("n_jobs", 1),
        )
        qeq_options.setdefault("cutoff", self.parameter_predictor.cutoff)
        self.qeq_model = JAXQEqModel(**qeq_options)

    def reset_charge_state(self) -> None:
        self.qeq_model.reset_charge_state()

    def calculate(self, atoms=None, properties=None, system_changes=all_changes) -> None:
        super().calculate(atoms, properties, system_changes)

        self.mace_calculator.calculate(
            atoms,
            properties=["energy", "forces"],
            system_changes=system_changes,
        )
        mace_results = self.mace_calculator.results
        mace_energy = float(mace_results["energy"])
        mace_forces = np.asarray(mace_results["forces"], dtype=float)

        parameters = self.parameter_predictor.predict(atoms)
        qeq_result = self.qeq_model.calculate(
            atoms,
            chi=parameters.chi,
            hardness=parameters.hardness,
            eta=parameters.eta,
            total_charge=0.0,
        )
        qeq_energy = float(qeq_result.energy)
        qeq_forces = np.asarray(qeq_result.forces, dtype=float)

        expected_force_shape = (len(atoms), 3)
        if mace_forces.shape != expected_force_shape:
            raise ValueError(
                f"MACE force shape is {mace_forces.shape}, "
                f"expected {expected_force_shape}"
            )
        if qeq_forces.shape != expected_force_shape:
            raise ValueError(
                f"QEq force shape is {qeq_forces.shape}, "
                f"expected {expected_force_shape}"
            )

        total_energy = mace_energy + qeq_energy
        total_forces = mace_forces + qeq_forces
        charges = np.asarray(qeq_result.charges, dtype=float)

        self.results = {
            "energy": total_energy,
            "free_energy": total_energy,
            "forces": total_forces,
            "charges": charges,
            "partial_charges": charges,
            "short_range_energy": mace_energy,
            "long_range_energy": qeq_energy,
            "short_range_forces": mace_forces,
            "long_range_forces": qeq_forces,
            "chi": np.asarray(qeq_result.chi, dtype=float),
            "hardness": np.asarray(qeq_result.hardness, dtype=float),
            "eta": np.asarray(qeq_result.eta, dtype=float),
            "qeq_solver": getattr(
                self.qeq_model, "last_charge_solver", None
            ),
            "qeq_pg_iterations": getattr(
                self.qeq_model, "last_pg_iterations", None
            ),
            "qeq_pg_error": getattr(
                self.qeq_model, "last_pg_error", None
            ),
        }
