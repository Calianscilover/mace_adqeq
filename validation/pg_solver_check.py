#!/usr/bin/env python3
"""Check that the production charge solver gives the same results as the matrix solver.

Production MD uses solver_mode="hybrid": the matrix solver on the first frame, then a
projected-gradient solver (CG by default) warm-started from the previous frame's charges,
falling back to the matrix solver only when the solver error exceeds max(10 * pg_tol, 1e-5).
This script walks through a trajectory (a random walk from --structure, or the frames of
--trajectory) with one hybrid model, exactly as calculator.py calls it, and compares every
frame with a float64 matrix-solver reference evaluated on the same geometry:

    calculate(atoms, predictor=predictor, compute_forces=True)

so the forces compared include the parameter response through MACE and the MLP, and the
QEq parameters (with the constant-potential chi bias) are re-predicted on every frame.

Verdict (exit status 1 if any fails; frame 0 is a matrix solve in both and is not counted):
    identical inputs     reference and hybrid see the same chi/hardness/eta; otherwise the
                         comparison measures predictor noise, not the solver (use --device cpu)
    no matrix fallback   every frame after the first is solved by the projected-gradient solver
    forces               max |F_hybrid - F_matrix| <= --force-tol on every frame
"""

import os

# Must be set before JAX is imported: the default allocator keeps freed GPU memory
# (up to 75% of the card), which starves MACE/PyTorch running on the same GPU.
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
from mace_adqeq.const_potential import identify_electrode_atoms
from mace_adqeq.qeq import JAXQEqModel, QEqParameterPredictor

STRUCTURE = HERE / "pair_revised_equilibrated.extxyz"
PARAMS = HERE / "jax_model" / "qeq_mace_chi_hardness_eta_params.msgpack"
CONFIG = HERE / "jax_model" / "qeq_mace_chi_hardness_eta_config.json"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--structure", type=Path, default=STRUCTURE)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--trajectory", type=Path, default=None, help="use these frames instead of a random walk")
    parser.add_argument("--params", type=Path, default=PARAMS)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, default=HERE / "pg_solver_check.json")
    parser.add_argument("--const-potential", action="store_true")
    parser.add_argument("--device", default="cpu", help="MACE device; cuda is not deterministic")
    parser.add_argument("--mace-dtype", default="float64", help="MACE default_dtype; empty string uses the model dtype")
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--sigma", type=float, default=0.01, help="per-step random displacement (Angstrom)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pg-method", choices=["lbfgs", "cg"], default="cg")
    parser.add_argument("--pg-tol", type=float, default=1.0e-3)
    parser.add_argument("--pg-maxiter", type=int, default=200)
    parser.add_argument("--max-pairs", type=int, default=30000)
    parser.add_argument("--dipole-axis", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--total-charge", type=float, default=0.0)
    parser.add_argument("--force-tol", type=float, default=1.0e-4, help="eV/A")
    parser.add_argument("--parameter-tol", type=float, default=1.0e-10, help="max |difference| of chi/hardness/eta between the two calls")
    return parser.parse_args()


def build_trajectory(args):
    if args.trajectory is not None:
        return read(args.trajectory, index=f":{args.frames}")
    rng = np.random.default_rng(args.seed)
    trajectory = [read(args.structure, index=args.frame)]
    for _ in range(args.frames - 1):
        trial = trajectory[-1].copy()
        trial.set_positions(trial.get_positions() + rng.normal(scale=args.sigma, size=(len(trial), 3)))
        trajectory.append(trial)
    return trajectory


def electrode_signature(atoms):
    bottom, upper, _ = identify_electrode_atoms(
        np.asarray(atoms.get_cell()), atoms.get_positions(), atoms.get_chemical_symbols()
    )
    return tuple(bottom.tolist()), tuple(upper.tolist())


def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    qeq_config = json.loads(args.config.read_text(encoding="utf-8"))["qeq"]

    def build(solver_mode):
        return JAXQEqModel(
            cutoff=float(qeq_config["cutoff"]),
            pme_grid=tuple(qeq_config["pme_grid"]),
            jitter=float(qeq_config["jitter"]),
            dtype="float64",
            max_pairs=args.max_pairs,
            solver_mode=solver_mode,
            pg_method=args.pg_method,
            pg_tolerance=args.pg_tol,
            pg_max_iterations=args.pg_maxiter,
            dipole_axis=args.dipole_axis,
            const_potential=args.const_potential,
        )

    predictor = QEqParameterPredictor(
        args.params, args.config, mace_device=args.device, mace_default_dtype=args.mace_dtype
    )
    reference_model = build("matrix")
    model = build("hybrid")
    trajectory = build_trajectory(args)
    source = args.trajectory if args.trajectory is not None else f"random walk from {args.structure}[{args.frame}], sigma={args.sigma}"
    print(f"trajectory: {source}, {len(trajectory)} frames, natoms={len(trajectory[0])}", flush=True)
    print(
        f"const_potential={args.const_potential}  device={args.device}  mace_dtype={args.mace_dtype!r}  "
        f"pg_method={args.pg_method}  pg_tol={args.pg_tol}  pg_maxiter={args.pg_maxiter}  "
        f"fallback above {max(10.0 * args.pg_tol, 1.0e-5):.0e}",
        flush=True,
    )

    def max_projected_gradient(atoms, result):
        """Stationarity of the hybrid charges, re-evaluated in float64 with the chi the solver saw."""
        positions_np = np.asarray(atoms.get_positions(), dtype=np.float64)
        box_np = np.asarray(atoms.get_cell(), dtype=np.float64)
        gradient = np.asarray(
            jax.grad(reference_model.energy_fn, argnums=0)(
                jnp.asarray(result.charges),
                jnp.asarray(positions_np),
                jnp.asarray(box_np),
                reference_model.neighbor_pairs(positions_np, box_np),
                jnp.asarray(result.eta),
                jnp.asarray(result.chi),
                jnp.asarray(result.hardness),
            )
        )
        return float(np.max(np.abs(gradient - gradient.mean())))

    print(
        f"\n{'frame':>5} {'solver':<19} {'iters':>5} {'pg_err':>9} {'max|Pg|':>9} {'max|dq|':>9} "
        f"{'|dE| eV':>9} {'max|dF|':>9} {'rms dF':>9} {'t_ref s':>7} {'t_hyb s':>7}",
        flush=True,
    )
    rows = []
    previous_signature = None
    for frame, atoms in enumerate(trajectory):
        start = time.perf_counter()
        reference = reference_model.calculate(atoms, predictor=predictor, total_charge=args.total_charge, compute_forces=True)
        reference_seconds = time.perf_counter() - start
        start = time.perf_counter()
        result = model.calculate(atoms, predictor=predictor, total_charge=args.total_charge, compute_forces=True)
        hybrid_seconds = time.perf_counter() - start

        signature = electrode_signature(atoms) if args.const_potential else None
        electrode_changed = previous_signature is not None and signature != previous_signature
        previous_signature = signature
        force_error = result.forces - reference.forces
        row = {
            "frame": frame,
            "solver": model.last_charge_solver,
            "pg_iterations": model.last_pg_iterations,
            "pg_error": model.last_pg_error,
            "max_projected_grad_f64": max_projected_gradient(atoms, result),
            "max_parameter_difference": float(
                max(np.max(np.abs(getattr(result, name) - getattr(reference, name))) for name in ("chi", "hardness", "eta"))
            ),
            "max_charge_error": float(np.max(np.abs(result.charges - reference.charges))),
            "energy_error": float(result.energy - reference.energy),
            "max_force_error": float(np.max(np.abs(force_error))),
            "rms_force_error": float(np.sqrt(np.mean(force_error**2))),
            "electrode_changed": electrode_changed,
            "reference_seconds": reference_seconds,
            "hybrid_seconds": hybrid_seconds,
        }
        rows.append(row)
        iterations = "-" if row["pg_iterations"] is None else str(row["pg_iterations"])
        pg_error = "-" if row["pg_error"] is None else f"{row['pg_error']:.2e}"
        note = "  electrode reassigned" if electrode_changed else ""
        print(
            f"{frame:5d} {row['solver']:<19} {iterations:>5} {pg_error:>9} {row['max_projected_grad_f64']:9.2e} "
            f"{row['max_charge_error']:9.2e} {abs(row['energy_error']):9.2e} {row['max_force_error']:9.2e} "
            f"{row['rms_force_error']:9.2e} {reference_seconds:7.2f} {hybrid_seconds:7.2f}{note}",
            flush=True,
        )

    warm = rows[1:]
    if not warm:
        raise ValueError("need at least two frames")
    fallbacks = [row["frame"] for row in warm if row["solver"] != "projected_gradient"]
    at_maxiter = [row["frame"] for row in warm if row["pg_iterations"] is not None and row["pg_iterations"] >= args.pg_maxiter]
    reassigned = [row["frame"] for row in warm if row["electrode_changed"]]
    parameter_difference = max(row["max_parameter_difference"] for row in warm)
    force_error = max(row["max_force_error"] for row in warm)
    iterations = [row["pg_iterations"] for row in warm if row["pg_iterations"] is not None]

    print("\n=== summary over frames 1.. ===", flush=True)
    print(f"  pg iterations          {min(iterations, default=0)}-{max(iterations, default=0)}, mean {np.mean(iterations) if iterations else 0:.1f}", flush=True)
    print(f"  at --pg-maxiter        {len(at_maxiter)} frames {at_maxiter}", flush=True)
    print(f"  max |dq|               {max(row['max_charge_error'] for row in warm):.2e} e", flush=True)
    print(f"  max |dE|               {max(abs(row['energy_error']) for row in warm):.2e} eV", flush=True)
    print(f"  max rms dF             {max(row['rms_force_error'] for row in warm):.2e} eV/A", flush=True)
    print(f"  mean time ref / hybrid {np.mean([r['reference_seconds'] for r in warm]):.2f} / {np.mean([r['hybrid_seconds'] for r in warm]):.2f} s", flush=True)
    if args.const_potential:
        print(f"  electrode reassigned   {len(reassigned)} frames {reassigned}", flush=True)

    checks = [
        ("identical inputs", parameter_difference <= args.parameter_tol, f"max |d param| {parameter_difference:.1e} <= {args.parameter_tol:.0e}"),
        ("no matrix fallback", not fallbacks, f"{len(fallbacks)} of {len(warm)} frames {fallbacks}"),
        ("forces", force_error <= args.force_tol, f"max |dF| {force_error:.2e} <= {args.force_tol:.0e} eV/A"),
    ]
    passed = all(ok for _, ok, _ in checks)
    print("\n=== verdict ===", flush=True)
    for name, ok, detail in checks:
        print(f"  {'OK  ' if ok else 'FAIL'}  {name:<20s} {detail}", flush=True)
    print(f"production solver matches the matrix solver: {'YES' if passed else 'NO'}", flush=True)

    args.output.write_text(
        json.dumps(
            {
                "trajectory": str(source),
                "frames": len(trajectory),
                "const_potential": args.const_potential,
                "device": args.device,
                "mace_dtype": args.mace_dtype,
                "pg_method": args.pg_method,
                "pg_tol": args.pg_tol,
                "pg_maxiter": args.pg_maxiter,
                "dipole_axis": args.dipole_axis,
                "reference_rms_force": float(np.sqrt(np.mean(reference.forces**2))),
                "frames_at_maxiter": at_maxiter,
                "electrode_reassigned": reassigned,
                "rows": rows,
                "checks": [{"name": name, "ok": ok, "detail": detail} for name, ok, detail in checks],
                "passed": passed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {args.output}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
