#!/usr/bin/env python3
"""Accuracy of the hybrid QEq solver (matrix on the first frame, warm-started
projected-gradient CG or LBFGS afterwards) in float32 and float64.

QEq parameters are predicted once and then held fixed, so only the charge solver
and the QEq energy/force evaluation are tested. A short random-walk trajectory
emulates MD steps. Every frame is compared with a float64 matrix-solver
reference; the stationarity of each solution is re-evaluated in float64 so that
float32 rounding in the solver's own error estimate does not hide anything.
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import argparse
import json
from pathlib import Path
import sys
import time


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from ase.io import read
import jax
import jax.numpy as jnp
from mace_adqeq.qeq import JAXQEqModel, QEqParameterPredictor

STRUCTURE = HERE / "pair_revised_equilibrated.extxyz"
PARAMS = HERE / "jax_model" / "qeq_mace_chi_hardness_eta_params.msgpack"
CONFIG = HERE / "jax_model" / "qeq_mace_chi_hardness_eta_config.json"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--structure", type=Path, default=STRUCTURE)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--params", type=Path, default=PARAMS)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, default=HERE / "pg_solver_check.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--sigma", type=float, default=0.01, help="per-step random displacement (Angstrom)")
    parser.add_argument("--pg-method", choices=["lbfgs", "cg"], default="cg")
    parser.add_argument("--pg-tol", type=float, default=1.0e-3)
    parser.add_argument("--pg-maxiter", type=int, default=200)
    parser.add_argument("--max-pairs", type=int, default=30000)
    parser.add_argument("--dipole-axis", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--const-potential", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    qeq_config = config["qeq"]

    def build(dtype, solver_mode):
        return JAXQEqModel(
            cutoff=float(qeq_config["cutoff"]),
            pme_grid=tuple(qeq_config["pme_grid"]),
            jitter=float(qeq_config["jitter"]),
            dtype=dtype,
            max_pairs=args.max_pairs,
            solver_mode=solver_mode,
            pg_method=args.pg_method,
            pg_tolerance=args.pg_tol,
            pg_max_iterations=args.pg_maxiter,
            dipole_axis=args.dipole_axis,
            const_potential=args.const_potential,
        )

    predictor = QEqParameterPredictor(args.params, args.config, mace_device=args.device, mace_default_dtype="float32")
    atoms0 = read(args.structure, index=args.frame)
    parameters = predictor.predict(atoms0)
    fixed = {"chi": parameters.chi, "hardness": parameters.hardness, "eta": parameters.eta}

    rng = np.random.default_rng(0)
    trajectory = [atoms0]
    for _ in range(args.frames - 1):
        trial = trajectory[-1].copy()
        trial.set_positions(trial.get_positions() + rng.normal(scale=args.sigma, size=(len(trial), 3)))
        trajectory.append(trial)

    reference_model = build("float64", "matrix")
    references = [reference_model.calculate(atoms, **fixed, compute_forces=True) for atoms in trajectory]

    def projected_gradient_f64(atoms, charges):
        positions_np = np.asarray(atoms.get_positions(), dtype=np.float64)
        box_np = np.asarray(atoms.get_cell(), dtype=np.float64)
        pairs = reference_model.neighbor_pairs(positions_np, box_np)
        chi = jnp.asarray(fixed["chi"], dtype=jnp.float64)
        gradient = np.asarray(
            jax.grad(reference_model.energy_fn, argnums=0)(
                jnp.asarray(charges, dtype=jnp.float64),
                jnp.asarray(positions_np),
                jnp.asarray(box_np),
                pairs,
                jnp.asarray(fixed["eta"], dtype=jnp.float64),
                chi,
                jnp.asarray(fixed["hardness"], dtype=jnp.float64),
            )
        )
        projected = gradient - gradient.mean()
        return float(np.linalg.norm(projected)), float(np.max(np.abs(projected)))

    report = {}
    for dtype in ("float64", "float32"):
        model = build(dtype, "hybrid")
        rows = []
        print(f"\n=== hybrid solver, dtype={dtype}, pg_method={args.pg_method}, pg_tol={args.pg_tol}, pg_maxiter={args.pg_maxiter} ===", flush=True)
        print(
            f"{'frame':>5} {'solver':<19} {'iters':>5} {'pg_err':>9} {'|Pg|_2(f64)':>11} {'max|dq|':>9} "
            f"{'|dE| eV':>9} {'max|dF|':>9} {'rms dF':>9} {'time s':>7}",
            flush=True,
        )
        for frame, (atoms, reference) in enumerate(zip(trajectory, references)):
            start = time.perf_counter()
            result = model.calculate(atoms, **fixed, compute_forces=True)
            elapsed = time.perf_counter() - start
            # With const_potential the solver sees biased chi, so the unbiased-chi stationarity check does not apply.
            l2, max_abs = (float("nan"), float("nan")) if args.const_potential else projected_gradient_f64(atoms, result.charges)
            force_error = result.forces - reference.forces
            row = {
                "frame": frame,
                "solver": model.last_charge_solver,
                "pg_iterations": model.last_pg_iterations,
                "pg_error": model.last_pg_error,
                "projected_grad_l2_f64": l2,
                "projected_grad_max_f64": max_abs,
                "max_charge_error": float(np.max(np.abs(result.charges - reference.charges))),
                "energy_error": float(result.energy - reference.energy),
                "max_force_error": float(np.max(np.abs(force_error))),
                "rms_force_error": float(np.sqrt(np.mean(force_error**2))),
                "seconds": elapsed,
            }
            rows.append(row)
            iterations = "-" if row["pg_iterations"] is None else str(row["pg_iterations"])
            pg_error = "-" if row["pg_error"] is None else f"{row['pg_error']:.2e}"
            print(
                f"{frame:5d} {row['solver']:<19} {iterations:>5} {pg_error:>9} {l2:11.2e} "
                f"{row['max_charge_error']:9.2e} {abs(row['energy_error']):9.2e} {row['max_force_error']:9.2e} "
                f"{row['rms_force_error']:9.2e} {elapsed:7.2f}",
                flush=True,
            )
        report[dtype] = rows

    rms_reference_force = float(np.sqrt(np.mean(references[0].forces ** 2)))
    print(f"\nreference RMS QEq force component: {rms_reference_force:.4e} eV/A", flush=True)
    args.output.write_text(
        json.dumps(
            {
                "structure": str(args.structure),
                "frames": args.frames,
                "sigma": args.sigma,
                "pg_method": args.pg_method,
                "pg_tol": args.pg_tol,
                "pg_maxiter": args.pg_maxiter,
                "dipole_axis": args.dipole_axis,
                "reference_rms_force": rms_reference_force,
                "results": report,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
