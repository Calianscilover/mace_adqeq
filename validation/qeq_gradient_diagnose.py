#!/usr/bin/env python3
"""Locate the mismatch between the frozen-parameter QEq force and finite differences.

Parameters (chi, hardness, eta) and the neighbor list are fixed at the reference
geometry throughout, so only the QEq energy function and the charge solver are tested.

T1  stationarity: grad_q E(q*) of the matrix-solved charges must be mu * 1
T2  fixed-charge FD of each energy term vs jax.grad w.r.t. positions
T3  charge response: FD with re-solved charges minus FD with fixed charges
    (zero by the envelope theorem if T1 passes)
T4  curvature of each term at q = 0 vs at q*; must agree because E(q) is quadratic
T5  eigenvalues of the Hessian on the neutral subspace; q* is a minimum only if all are positive
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import argparse
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from ase.io import read
import jax
import jax.numpy as jnp
from jax.scipy.special import erfc
from dmff.utils import pair_buffer_scales
from mace_adqeq import qeq as qeq_module
from mace_adqeq.qeq import JAXQEqModel, QEqParameterPredictor, ds_pairs

STRUCTURE = HERE / "pair_revised_equilibrated.extxyz"
PARAMS = HERE / "jax_model" / "qeq_mace_chi_hardness_eta_params.msgpack"
CONFIG = HERE / "jax_model" / "qeq_mace_chi_hardness_eta_config.json"
ATOMS = [587, 445, 458, 461, 261, 263, 262, 371, 345]
AXES = "xyz"
COULOMB = 1389.35455846
KJMOL_PER_EV = 96.4869


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--structure", type=Path, default=STRUCTURE)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--params", type=Path, default=PARAMS)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, default=HERE / "qeq_gradient_diagnose.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--atoms", type=int, nargs="+", default=ATOMS)
    parser.add_argument("--step", type=float, default=1.0e-3)
    parser.add_argument("--total-charge", type=float, default=0.0)
    parser.add_argument("--max-pairs", type=int, default=50000)
    parser.add_argument("--kappa", type=float, default=4.3804348)
    parser.add_argument("--dipole-axis", type=int, choices=(0, 1, 2), default=2)
    return parser.parse_args()


def build_terms(kappa, grid, dipole_axis):
    """Re-implementation of the four terms of qeq.generate_get_Energy_Qeq, in eV."""
    pme = qeq_module.generate_get_energy(kappa, *grid)

    def e_pme(q, positions, box, pairs, eta, chi, hardness):
        return pme(positions / 10.0, box / 10.0, pairs, q, mscales=jnp.ones(6, dtype=positions.dtype)) / KJMOL_PER_EV

    def e_correction(q, positions, box, pairs, eta, chi, hardness):
        distances = ds_pairs(positions, box, pairs)
        buffer_scales = pair_buffer_scales(pairs)
        pair_eta = jnp.sqrt(eta[pairs[:, 0]] ** 2 + eta[pairs[:, 1]] ** 2)
        pair = (
            q[pairs[:, 0]] * q[pairs[:, 1]]
            * erfc(distances / (jnp.sqrt(2.0) * pair_eta))
            * COULOMB / distances * buffer_scales
        )
        self_energy = q**2 * COULOMB / (2.0 * jnp.sqrt(jnp.pi) * eta)
        return (-jnp.sum(pair) + jnp.sum(self_energy)) / KJMOL_PER_EV

    def e_onsite(q, positions, box, pairs, eta, chi, hardness):
        return jnp.sum(chi * q + 0.5 * hardness * q * q)

    def e_dipole(q, positions, box, pairs, eta, chi, hardness):
        volume = jnp.linalg.det(box)
        moment = jnp.sum(q * positions[:, dipole_axis])
        return 2.0 * jnp.pi / volume * COULOMB * moment**2 / KJMOL_PER_EV

    return {
        "pme": jax.jit(e_pme),
        "correction": jax.jit(e_correction),
        "onsite": jax.jit(e_onsite),
        "dipole": jax.jit(e_dipole),
    }


def main():
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    qeq_config = config["qeq"]
    grid = tuple(int(value) for value in qeq_config["pme_grid"])

    predictor = QEqParameterPredictor(args.params, args.config, mace_device=args.device, mace_default_dtype="float32")
    qeq = JAXQEqModel(
        cutoff=float(qeq_config["cutoff"]),
        pme_grid=grid,
        kappa=args.kappa,
        dipole_axis=args.dipole_axis,
        jitter=float(qeq_config["jitter"]),
        dtype="float64",
        max_pairs=args.max_pairs,
        solver_mode="matrix",
        const_potential=False,
    )
    terms = build_terms(args.kappa, grid, args.dipole_axis)

    atoms = read(args.structure, index=args.frame)
    symbols = atoms.get_chemical_symbols()
    parameters = predictor.predict(atoms)
    chi = jnp.asarray(parameters.chi, dtype=jnp.float64)
    hardness = jnp.asarray(parameters.hardness, dtype=jnp.float64)
    eta = jnp.asarray(parameters.eta, dtype=jnp.float64)

    positions_np = np.asarray(atoms.get_positions(), dtype=np.float64)
    box_np = np.asarray(atoms.get_cell(), dtype=np.float64)
    positions = jnp.asarray(positions_np)
    box = jnp.asarray(box_np)
    pairs = qeq.neighbor_pairs(positions_np, box_np)
    real_pairs = int(np.sum(np.asarray(pair_buffer_scales(pairs))))
    print(f"natoms={len(atoms)}  pair rows={pairs.shape[0]}  real pairs (buffer_scale=1)={real_pairs}", flush=True)

    def solve(trial_positions):
        return qeq.solve_charges_matrix(args.total_charge, trial_positions, box, pairs, eta, chi, hardness)

    def energy(q, trial_positions):
        return float(qeq.energy_fn(q, trial_positions, box, pairs, eta, chi, hardness))

    def term_energies(q, trial_positions):
        return {name: float(fn(q, trial_positions, box, pairs, eta, chi, hardness)) for name, fn in terms.items()}

    charges = solve(positions)
    reference_terms = term_energies(charges, positions)
    reference_energy = energy(charges, positions)
    print(
        f"E(q*)={reference_energy:.10f} eV  sum of re-implemented terms={sum(reference_terms.values()):.10f} eV  "
        f"terms={ {name: round(value, 6) for name, value in reference_terms.items()} }",
        flush=True,
    )
    print(f"sum(q*)={float(jnp.sum(charges)):.3e}", flush=True)

    # T1: stationarity and quadratic consistency
    grad_q = np.asarray(jax.grad(qeq.energy_fn, argnums=0)(charges, positions, box, pairs, eta, chi, hardness))
    grad_q0 = np.asarray(jax.grad(qeq.energy_fn, argnums=0)(jnp.zeros_like(charges), positions, box, pairs, eta, chi, hardness))
    deviation = grad_q - grad_q.mean()
    rng = np.random.default_rng(0)
    direction = rng.normal(size=len(atoms))
    direction -= direction.mean()
    direction /= np.linalg.norm(direction)
    direction = jnp.asarray(direction)
    epsilon = 1.0e-2
    e_plus = energy(charges + epsilon * direction, positions)
    e_minus = energy(charges - epsilon * direction, positions)
    hvp = jax.jvp(
        lambda q: jax.grad(qeq.energy_fn, argnums=0)(q, positions, box, pairs, eta, chi, hardness),
        (charges,),
        (direction,),
    )[1]
    t1 = {
        "mu": float(grad_q.mean()),
        "rms_projected_grad": float(np.sqrt(np.mean(deviation**2))),
        "max_projected_grad": float(np.max(np.abs(deviation))),
        "rms_projected_grad_at_q0": float(np.std(grad_q0)),
        "fd_directional_grad": (e_plus - e_minus) / (2.0 * epsilon),
        "fd_curvature": (e_plus + e_minus - 2.0 * reference_energy) / epsilon**2,
        "hvp_curvature": float(jnp.dot(direction, hvp)),
    }
    print("\n=== T1: charge stationarity (eV/e) ===", flush=True)
    for key, value in t1.items():
        print(f"  {key:<26s} {value: .6e}", flush=True)
    print("  expected: projected grad << grad at q=0; fd_directional_grad ~ 0; fd_curvature == hvp_curvature", flush=True)
    t1["solver_residual"] = getattr(qeq, "last_matrix_residual", None)
    print(f"  solver_residual            {t1['solver_residual']}", flush=True)

    # T4: E(q) is quadratic, so the curvature must not depend on where it is evaluated
    def curvature(fn, q):
        hvp_q = jax.jvp(
            lambda x: jax.grad(fn, argnums=0)(x, positions, box, pairs, eta, chi, hardness),
            (q,),
            (direction,),
        )[1]
        return float(jnp.dot(direction, hvp_q))

    zero_charges = jnp.zeros_like(charges)
    t4 = {
        name: {"at_q0": curvature(fn, zero_charges), "at_qstar": curvature(fn, charges)}
        for name, fn in {**terms, "total": qeq.energy_fn}.items()
    }
    print("\n=== T4: curvature v.H.v at q=0 vs q* (eV/e^2) ===", flush=True)
    for name, item in t4.items():
        print(f"  {name:<12s} at_q0={item['at_q0']: .8e}  at_qstar={item['at_qstar']: .8e}", flush=True)
    print("  expected: identical for every term (a mismatch means jax.hessian at q=0 is wrong)", flush=True)

    # T5: q* is a minimum only if H is positive definite on the neutral subspace sum(dq) = 0
    hessian = np.asarray(jax.hessian(qeq.energy_fn, argnums=0)(charges, positions, box, pairs, eta, chi, hardness))
    hessian = 0.5 * (hessian + hessian.T)
    projector = np.eye(len(atoms)) - 1.0 / len(atoms)
    eigenvalues = np.linalg.eigvalsh(projector @ hessian @ projector)
    neutral = np.delete(eigenvalues, np.argmin(np.abs(eigenvalues)))
    t5 = {
        "min_eigenvalue": float(neutral.min()),
        "max_eigenvalue": float(neutral.max()),
        "n_negative": int(np.sum(neutral < 0.0)),
    }
    print(
        f"\n=== T5: neutral-subspace Hessian at q* (eV/e^2) ===\n"
        f"  min={t5['min_eigenvalue']:.4e}  max={t5['max_eigenvalue']:.4e}  negative={t5['n_negative']}  (expected: none negative)",
        flush=True,
    )

    # T2 and T3
    analytic = {
        name: np.asarray(jax.grad(fn, argnums=1)(charges, positions, box, pairs, eta, chi, hardness))
        for name, fn in terms.items()
    }
    analytic_total = np.asarray(jax.grad(qeq.energy_fn, argnums=1)(charges, positions, box, pairs, eta, chi, hardness))
    step = args.step
    rows = []
    print(f"\n=== T2/T3: forces (eV/A), step={step} A ===", flush=True)
    header = f"{'atom':>4} {'el':>2} ax {'analytic':>10} {'fd_fixed_q':>10} {'fd_resolved':>11} {'q_response':>10} | analytic - fd per term: " + " ".join(f"{name:>10s}" for name in terms)
    print(header, flush=True)
    for index in args.atoms:
        for axis in range(3):
            shift = np.zeros_like(positions_np)
            shift[index, axis] = step
            plus = jnp.asarray(positions_np + shift)
            minus = jnp.asarray(positions_np - shift)
            terms_plus = term_energies(charges, plus)
            terms_minus = term_energies(charges, minus)
            fd_terms = {name: -(terms_plus[name] - terms_minus[name]) / (2.0 * step) for name in terms}
            fd_fixed_q = -(energy(charges, plus) - energy(charges, minus)) / (2.0 * step)
            fd_resolved = -(energy(solve(plus), plus) - energy(solve(minus), minus)) / (2.0 * step)
            force = -float(analytic_total[index, axis])
            term_errors = {name: -float(analytic[name][index, axis]) - fd_terms[name] for name in terms}
            rows.append(
                {
                    "atom": index,
                    "element": symbols[index],
                    "axis": AXES[axis],
                    "analytic": force,
                    "fd_fixed_q": fd_fixed_q,
                    "fd_resolved": fd_resolved,
                    "charge_response": fd_resolved - fd_fixed_q,
                    "term_errors": term_errors,
                    "term_fd": fd_terms,
                }
            )
            print(
                f"{index:4d} {symbols[index]:>2s} {AXES[axis]}  {force:10.5f} {fd_fixed_q:10.5f} {fd_resolved:11.5f} "
                f"{fd_resolved - fd_fixed_q:10.5f} | " + " ".join(f"{term_errors[name]:10.2e}" for name in terms),
                flush=True,
            )

    def mean_abs(values):
        return float(np.mean(np.abs(values)))

    summary = {
        "analytic_vs_fd_fixed_q": mean_abs([row["analytic"] - row["fd_fixed_q"] for row in rows]),
        "analytic_vs_fd_resolved": mean_abs([row["analytic"] - row["fd_resolved"] for row in rows]),
        "charge_response": mean_abs([row["charge_response"] for row in rows]),
        "term_errors": {name: mean_abs([row["term_errors"][name] for row in rows]) for name in terms},
    }
    print("\n=== summary, mean |difference| (eV/A) ===", flush=True)
    print(f"  T2 analytic vs fd_fixed_q   {summary['analytic_vs_fd_fixed_q']:.3e}   (position gradient of energy_fn)", flush=True)
    print(f"  T3 charge response          {summary['charge_response']:.3e}   (envelope theorem, should be ~0)", flush=True)
    print(f"     analytic vs fd_resolved  {summary['analytic_vs_fd_resolved']:.3e}   (= frozen-force error of fd_force_test)", flush=True)
    for name, value in summary["term_errors"].items():
        print(f"  T2 term {name:<12s}        {value:.3e}", flush=True)

    args.output.write_text(
        json.dumps(
            {
                "structure": str(args.structure),
                "step": step,
                "real_pairs": real_pairs,
                "reference_energy": reference_energy,
                "reference_terms": reference_terms,
                "t1": t1,
                "t4": t4,
                "t5": t5,
                "summary": summary,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
