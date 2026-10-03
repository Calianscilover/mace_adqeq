#!/usr/bin/env python3
"""Run one combined MACE + JAX-QEq ASE single-point calculation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase.io import read, write

from mace_adqeq import MACEJAXQEqCalculator


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", default="../STRU-2.pdb")
    parser.add_argument("--mace-model", default="../interface.model")
    parser.add_argument(
        "--qeq-params",
        required=True,
    )
    parser.add_argument(
        "--qeq-config",
        required=True,
    )
    parser.add_argument("--total-charge", type=float, required=True)
    parser.add_argument("--mace-device", default="cuda")
    parser.add_argument("--mace-dtype", default="float32")
    parser.add_argument(
        "--dipole-axis",
        type=int,
        choices=(0, 1, 2),
        default=2,
        help="Axis of the slab dipole correction (the non-periodic electrode normal, z).",
    )
    parser.add_argument("--output", default="combined_singlepoint.extxyz")
    return parser.parse_args()


def main():
    args = parse_args()
    atoms = read(args.structure)
    atoms.info["total_charge"] = args.total_charge
    atoms.calc = MACEJAXQEqCalculator(
        mace_model_path=args.mace_model,
        qeq_params_path=args.qeq_params,
        qeq_config_path=args.qeq_config,
        mace_device=args.mace_device,
        mace_default_dtype=args.mace_dtype,
        qeq_options={"dipole_axis": args.dipole_axis},
    )

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    charges = atoms.calc.results["partial_charges"].copy()
    atoms.arrays["qeq_charges"] = charges
    atoms.arrays["qeq_chi"] = atoms.calc.results["chi"].copy()
    atoms.arrays["qeq_hardness"] = atoms.calc.results["hardness"].copy()
    atoms.arrays["qeq_eta"] = atoms.calc.results["eta"].copy()
    atoms.info["short_range_energy"] = atoms.calc.results["short_range_energy"]
    atoms.info["long_range_energy"] = atoms.calc.results["long_range_energy"]
    write(Path(args.output), atoms, format="extxyz")

    print(f"total_energy_eV={energy:.12g}")
    print(f"short_range_energy_eV={atoms.calc.results['short_range_energy']:.12g}")
    print(f"long_range_energy_eV={atoms.calc.results['long_range_energy']:.12g}")
    print(f"max_force_eV_per_A={np.linalg.norm(forces, axis=1).max():.12g}")
    print(f"charge_sum_e={charges.sum():.12g}")
    print(f"charge_min_e={charges.min():.12g}")
    print(f"charge_max_e={charges.max():.12g}")
    print(f"output={Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
