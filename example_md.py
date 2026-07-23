#!/usr/bin/env python3
"""Minimal ASE Langevin MD example using MACEJAXQEqCalculator."""

from __future__ import annotations

import argparse
from pathlib import Path

from ase import units
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

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
    parser.add_argument("--temperature-K", type=float, default=300.0)
    parser.add_argument("--timestep-fs", type=float, default=0.1)
    parser.add_argument("--friction-per-ps", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--trajectory", default="combined_md.extxyz")
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
    )

    trajectory = Path(args.trajectory)
    if trajectory.exists():
        raise FileExistsError(
            f"Refusing to append to existing trajectory: {trajectory.resolve()}"
        )

    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature_K)
    dynamics = Langevin(
        atoms,
        timestep=args.timestep_fs * units.fs,
        temperature_K=args.temperature_K,
        friction=args.friction_per_ps / (1000.0 * units.fs),
    )

    def save_frame():
        atoms.get_forces()
        atoms.arrays["qeq_charges"] = atoms.calc.results["partial_charges"].copy()
        atoms.info["short_range_energy"] = atoms.calc.results["short_range_energy"]
        atoms.info["long_range_energy"] = atoms.calc.results["long_range_energy"]
        write(trajectory, atoms, append=True, format="extxyz")

    dynamics.attach(save_frame, interval=args.interval)
    dynamics.run(args.steps)


if __name__ == "__main__":
    main()
