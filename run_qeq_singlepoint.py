#!/usr/bin/env python3
"""Run the independent JAX-QEq model without loading MACE."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase.io import read, write

from mace_adqeq import JAXQEqModel


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", default="../STRU-2.pdb")
    parser.add_argument(
        "--qeq-params",
        required=True,
    )
    parser.add_argument(
        "--qeq-config",
        required=True,
    )
    parser.add_argument("--total-charge", type=float, required=True)
    parser.add_argument("--dipole-axis", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--report-first", type=int, default=93)
    parser.add_argument("--output", default="qeq_singlepoint.extxyz")
    return parser.parse_args()


def main():
    args = parse_args()
    atoms = read(args.structure)
    model = JAXQEqModel(
        args.qeq_params,
        args.qeq_config,
        dipole_axis=args.dipole_axis,
    )
    result = model.calculate(atoms, total_charge=args.total_charge)

    atoms.arrays["qeq_charges"] = result.charges
    atoms.arrays["qeq_forces"] = result.forces
    atoms.arrays["qeq_chi"] = result.chi
    atoms.arrays["qeq_hardness"] = result.hardness
    atoms.arrays["qeq_eta"] = result.eta
    atoms.info["qeq_energy"] = result.energy
    atoms.info["total_charge"] = args.total_charge
    write(Path(args.output), atoms, format="extxyz")

    count = min(max(args.report_first, 0), len(atoms))
    first = result.charges[:count]
    negative_indices = np.flatnonzero(first < 0.0) + 1
    print(f"qeq_energy_eV={result.energy:.12g}")
    print(f"charge_sum_e={result.charges.sum():.12g}")
    print(f"charge_min_e={result.charges.min():.12g}")
    print(f"charge_max_e={result.charges.max():.12g}")
    print(f"first_group_size={count}")
    print(f"first_group_negative_count={len(negative_indices)}")
    print("first_group_negative_indices_1based=" + ",".join(map(str, negative_indices)))
    print(f"output={Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
