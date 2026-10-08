#!/usr/bin/env python3
"""Plain MACE + QEq molecular dynamics of the Zn | ZnSO4(aq) | Zn slab.

The outermost Zn layer(s) of the bottom and the upper electrode are fixed; every other
atom moves under MACE (short range) + QEq (long range) forces from MACEJAXQEqCalculator.

    nve   velocity Verlet; checks energy conservation (drift and fluctuation of E_tot)
    nvt   Langevin thermostat; production-style run

Every step goes to md.csv (energies, temperature, the largest MACE / QEq / total force on
a mobile atom, charges, QEq solver statistics and, with --const-potential, the number of
electrode atoms); frames go to traj.extxyz with momenta, so a run can be continued from
its last frame:

    python md_run.py --ensemble nvt --steps 4000 --output runs/nvt_eq
    python md_run.py --ensemble nve --steps 10000 --structure runs/nvt_eq/traj.extxyz --frame -1 \\
        --keep-momenta --output runs/nve

The run stops when a mobile atom feels more than --max-force, the temperature exceeds
--max-temperature, or anything becomes NaN. It then writes the current frame to
failure_step_<n>.extxyz and the last --keep-last steps to pre_failure.extxyz, with the
model's charges and MACE / QEq forces as model_* arrays (not as forces, so they are never
mistaken for reference labels).
"""

import os

# Must be set before JAX is imported: the default allocator keeps freed GPU memory
# (up to 75% of the card), which starves MACE/PyTorch running on the same GPU.
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import argparse
from collections import deque
import json
from datetime import datetime
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np
from ase import units
from ase.constraints import FixAtoms
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet

from mace_adqeq import MACEJAXQEqCalculator
from mace_adqeq.const_potential import identify_electrode_atoms

STRUCTURE = HERE / "stru" / "pair_revised_equilibrated.extxyz"
MACE_MODEL = HERE / "data" / "mace_short_stagetwo.model"
QEQ_PARAMS = HERE / "jax_model" / "qeq_mace_chi_hardness_eta_params.msgpack"
QEQ_CONFIG = HERE / "jax_model" / "qeq_mace_chi_hardness_eta_config.json"

COLUMNS = (
    "step", "time_fs", "temperature_K", "epot_eV", "ekin_eV", "etot_eV", "mace_eV", "qeq_eV",
    "max_force_eV_A", "max_mace_force_eV_A", "max_qeq_force_eV_A", "max_force_atom",
    "charge_sum_e", "max_abs_charge_e", "qeq_solver", "qeq_iterations", "qeq_error",
    "n_electrode_bottom", "n_electrode_upper",
)


class Failure(RuntimeError):
    pass


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ensemble", choices=["nve", "nvt"], required=True)
    parser.add_argument("--structure", type=Path, default=STRUCTURE)
    parser.add_argument("--frame", type=int, default=0, help="frame of --structure; -1 for the last")
    parser.add_argument("--mace-model", type=Path, default=MACE_MODEL)
    parser.add_argument("--params", type=Path, default=QEQ_PARAMS)
    parser.add_argument("--config", type=Path, default=QEQ_CONFIG)
    parser.add_argument("--output", type=Path, default=None, help="default: runs/<ensemble>_<timestamp>")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mace-dtype", default="float32")
    parser.add_argument("--const-potential", action="store_true")
    parser.add_argument("--pg-tol", type=float, default=1.0e-5, help="CG tolerance of the QEq solver (eV/e)")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--timestep", type=float, default=0.5, help="fs")
    parser.add_argument("--temperature", type=float, default=300.0, help="K; initial velocities and NVT target")
    parser.add_argument("--friction", type=float, default=10.0, help="Langevin friction (1/ps)")
    parser.add_argument("--keep-momenta", action="store_true", help="start from the momenta stored in --structure")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--fix-layers", type=int, default=1, help="outermost Zn layers fixed on each electrode")
    parser.add_argument("--layer-gap", type=float, default=1.0, help="z gap (A) that separates two Zn layers")
    parser.add_argument("--traj-interval", type=int, default=100)
    parser.add_argument("--max-temperature", type=float, default=5000.0, help="K; stop above this")
    parser.add_argument("--max-force", type=float, default=50.0, help="eV/A on a mobile atom; stop above this")
    parser.add_argument("--keep-last", type=int, default=50, help="steps written to pre_failure.extxyz on a stop")
    return parser.parse_args()


def outer_zn_layers(atoms, n_layers, gap):
    """Indices of the n outermost Zn layers at the bottom and at the top of the slab."""
    zinc = np.flatnonzero(np.asarray(atoms.get_chemical_symbols()) == "Zn")
    z = atoms.positions[zinc, 2]
    order = np.argsort(z)
    layer_id = np.concatenate(([0], np.cumsum(np.diff(z[order]) > gap)))
    bottom = zinc[order[layer_id < n_layers]]
    top = zinc[order[layer_id > layer_id[-1] - n_layers]]
    return bottom, top


def build_calculator(args):
    return MACEJAXQEqCalculator(
        mace_model_path=args.mace_model,
        qeq_params_path=args.params,
        qeq_config_path=args.config,
        mace_device=args.device,
        mace_default_dtype=args.mace_dtype,
        qeq_options={"pg_tolerance": args.pg_tol},
        const_potential=args.const_potential,
    )


def step_state(atoms, mobile):
    """Model results of the current step and the quantities the stop criteria look at."""
    results = atoms.calc.results
    forces = np.asarray(results["forces"])
    mace_forces = np.asarray(results["short_range_forces"])
    qeq_forces = np.asarray(results["long_range_forces"])
    norms = np.linalg.norm(forces[mobile], axis=1)
    state = {
        "epot": float(results["energy"]),
        "ekin": float(atoms.get_kinetic_energy()),
        "temperature": float(atoms.get_temperature()),
        "mace_energy": float(results["short_range_energy"]),
        "qeq_energy": float(results["long_range_energy"]),
        "max_force": float(norms.max()),
        "max_mace_force": float(np.linalg.norm(mace_forces[mobile], axis=1).max()),
        "max_qeq_force": float(np.linalg.norm(qeq_forces[mobile], axis=1).max()),
        "max_force_atom": int(mobile[np.argmax(norms)]),
        "charges": np.asarray(results["charges"]),
        "mace_forces": mace_forces,
        "qeq_forces": qeq_forces,
    }
    state["finite"] = bool(
        np.isfinite(state["epot"]) and np.all(np.isfinite(forces)) and np.all(np.isfinite(state["charges"]))
    )
    return state


def stop_reason(state, max_force, max_temperature):
    if not state["finite"]:
        return "NaN or Inf"
    if state["max_force"] > max_force:
        return f"max force {state['max_force']:.1f} eV/A on atom {state['max_force_atom']}"
    if state["temperature"] > max_temperature:
        return f"temperature {state['temperature']:.0f} K"
    return None


def capture_frame(atoms, state, **info):
    """Copy of the current frame with the model's charges and forces as model_* arrays."""
    frame = atoms.copy()
    frame.calc = None
    frame.set_constraint()
    frame.arrays["model_charges"] = state["charges"].copy()
    frame.arrays["model_mace_forces"] = state["mace_forces"].copy()
    frame.arrays["model_qeq_forces"] = state["qeq_forces"].copy()
    frame.info.update(
        {
            "model_energy": state["epot"],
            "model_mace_energy": state["mace_energy"],
            "model_qeq_energy": state["qeq_energy"],
            "model_max_force": state["max_force"],
            "model_max_mace_force": state["max_mace_force"],
            "model_max_qeq_force": state["max_qeq_force"],
            "temperature_K": state["temperature"],
            **info,
        }
    )
    return frame


def main():
    args = parse_args()
    output = args.output or HERE / "runs" / f"{args.ensemble}_{datetime.now():%Y%m%d_%H%M%S}"
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.mkdir(parents=True)

    atoms = read(args.structure, index=args.frame)
    momenta = atoms.get_momenta()
    bottom, top = outer_zn_layers(atoms, args.fix_layers, args.layer_gap)
    fixed = np.concatenate((bottom, top))
    mobile = np.setdiff1d(np.arange(len(atoms)), fixed)
    atoms.set_constraint(FixAtoms(indices=fixed))
    atoms.calc = build_calculator(args)

    rng = np.random.default_rng(args.seed)
    if args.keep_momenta:
        if not np.any(momenta):
            raise ValueError(f"{args.structure} has no momenta; drop --keep-momenta")
        atoms.set_momenta(momenta)
    else:
        MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature, rng=rng, force_temp=True)

    timestep = args.timestep * units.fs
    if args.ensemble == "nve":
        dynamics = VelocityVerlet(atoms, timestep=timestep)
    else:
        # fixcm would move momentum into the fixed electrode layers; they already anchor the slab.
        dynamics = Langevin(
            atoms,
            timestep=timestep,
            temperature_K=args.temperature,
            friction=args.friction / (1000.0 * units.fs),
            fixcm=False,
            rng=rng,
        )

    symbols = atoms.get_chemical_symbols()
    print(f"structure={args.structure}[{args.frame}]  natoms={len(atoms)}  output={output}", flush=True)
    print(
        f"ensemble={args.ensemble}  steps={args.steps}  timestep={args.timestep} fs  T={args.temperature} K  "
        f"device={args.device}  mace_dtype={args.mace_dtype}  const_potential={args.const_potential}  pg_tol={args.pg_tol}",
        flush=True,
    )
    print(
        f"fixed {len(fixed)} Zn: bottom z={atoms.positions[bottom, 2].min():.2f}-{atoms.positions[bottom, 2].max():.2f} A ({len(bottom)}), "
        f"top z={atoms.positions[top, 2].min():.2f}-{atoms.positions[top, 2].max():.2f} A ({len(top)})",
        flush=True,
    )
    (output / "run.json").write_text(
        json.dumps({**{key: str(value) for key, value in vars(args).items()}, "fixed_atoms": fixed.tolist()}, indent=2),
        encoding="utf-8",
    )

    log = (output / "md.csv").open("w", encoding="utf-8", buffering=1)
    log.write(",".join(COLUMNS) + "\n")
    trajectory = output / "traj.extxyz"
    records = []
    recent = deque(maxlen=args.keep_last)
    start = time.perf_counter()

    def log_step():
        state = step_state(atoms, mobile)
        frame = capture_frame(atoms, state, step=dynamics.nsteps)
        recent.append(frame)
        reason = stop_reason(state, args.max_force, args.max_temperature)
        if reason is not None:
            failure_path = output / f"failure_step_{dynamics.nsteps}.extxyz"
            write(failure_path, frame, format="extxyz")
            write(output / "pre_failure.extxyz", list(recent), format="extxyz")
            raise Failure(f"{reason} at step {dynamics.nsteps}; frames: {failure_path}, {output / 'pre_failure.extxyz'}")

        if args.const_potential:
            electrode_bottom, electrode_upper, _ = identify_electrode_atoms(np.asarray(atoms.get_cell()), atoms.positions, symbols)
            n_bottom, n_upper = len(electrode_bottom), len(electrode_upper)
        else:
            n_bottom = n_upper = ""
        results = atoms.calc.results
        iterations, error = results.get("qeq_pg_iterations"), results.get("qeq_pg_error")
        record = {
            "step": dynamics.nsteps,
            "time_fs": dynamics.get_time() / units.fs,
            "temperature_K": state["temperature"],
            "epot_eV": state["epot"],
            "ekin_eV": state["ekin"],
            "etot_eV": state["epot"] + state["ekin"],
            "mace_eV": state["mace_energy"],
            "qeq_eV": state["qeq_energy"],
            "max_force_eV_A": state["max_force"],
            "max_mace_force_eV_A": state["max_mace_force"],
            "max_qeq_force_eV_A": state["max_qeq_force"],
            "max_force_atom": f"{symbols[state['max_force_atom']]}{state['max_force_atom']}",
            "charge_sum_e": float(state["charges"].sum()),
            "max_abs_charge_e": float(np.max(np.abs(state["charges"]))),
            "qeq_solver": results.get("qeq_solver"),
            "qeq_iterations": "" if iterations is None else iterations,
            "qeq_error": "" if error is None else error,
            "n_electrode_bottom": n_bottom,
            "n_electrode_upper": n_upper,
        }
        records.append(record)
        log.write(",".join(f"{value:.12g}" if isinstance(value, float) else str(value) for value in record.values()) + "\n")
        if dynamics.nsteps % 100 == 0:
            elapsed = time.perf_counter() - start
            print(
                f"step {dynamics.nsteps:7d}  T={state['temperature']:7.1f} K  Etot={record['etot_eV']:.6f} eV  "
                f"maxF={state['max_force']:.2f} eV/A  solver={record['qeq_solver']}  iters={record['qeq_iterations']}  "
                f"{elapsed / max(dynamics.nsteps, 1):.3f} s/step",
                flush=True,
            )

    def write_frame():
        frame = atoms.copy()
        frame.arrays["qeq_charges"] = np.asarray(atoms.calc.results["charges"]).copy()
        frame.calc = None
        write(trajectory, frame, format="extxyz", append=trajectory.exists())

    dynamics.attach(log_step, interval=1)
    dynamics.attach(write_frame, interval=args.traj_interval)
    try:
        dynamics.run(args.steps)
    finally:
        log.close()
        if records:
            write_summary(args, output, records, time.perf_counter() - start, len(mobile))


def write_summary(args, output, records, seconds, n_mobile):
    time_ps = np.array([record["time_fs"] for record in records]) / 1000.0
    etot = np.array([record["etot_eV"] for record in records])
    temperature = np.array([record["temperature_K"] for record in records])
    warm = records[1:]
    iterations = [record["qeq_iterations"] for record in warm if record["qeq_iterations"] != ""]
    electrode = {(record["n_electrode_bottom"], record["n_electrode_upper"]) for record in records}
    summary = {
        "ensemble": args.ensemble,
        "steps": records[-1]["step"],
        "time_ps": float(time_ps[-1]),
        "seconds_per_step": seconds / max(records[-1]["step"], 1),
        "temperature_mean_K": float(temperature.mean()),
        "temperature_std_K": float(temperature.std()),
        "max_force_eV_A": max(record["max_force_eV_A"] for record in records),
        "matrix_fallbacks": sum(record["qeq_solver"] == "matrix_fallback" for record in warm),
        "qeq_iterations_mean": float(np.mean(iterations)) if iterations else None,
        "qeq_iterations_max": int(np.max(iterations)) if iterations else None,
        "max_abs_charge_e": max(record["max_abs_charge_e"] for record in records),
    }
    if args.const_potential:
        summary["electrode_counts_seen"] = sorted(electrode)
    if args.ensemble == "nve" and len(records) > 2:
        slope, intercept = np.polyfit(time_ps, etot, 1)
        residual = etot - (slope * time_ps + intercept)
        summary.update(
            {
                "etot_drift_eV_per_ps": float(slope),
                "etot_drift_meV_per_mobile_atom_per_ps": float(1000.0 * slope / n_mobile),
                "etot_fluctuation_std_eV": float(residual.std()),
                "etot_max_minus_min_eV": float(etot.max() - etot.min()),
            }
        )
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== summary ===", flush=True)
    for key, value in summary.items():
        print(f"  {key:<40s} {value}", flush=True)


if __name__ == "__main__":
    main()
