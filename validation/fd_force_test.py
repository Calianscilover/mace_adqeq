#!/usr/bin/env python3
"""Finite-difference check that MACE-adQEq forces equal -dE/dR (constQ or constant potential).

For selected atoms (one per chemical group plus the atoms with the largest parameter
response) and two global 3N directions, central differences of the energy are compared
with two analytic forces:

    frozen  chi/hardness/eta fixed at the reference geometry; FD keeps the same parameters
    whole   including the parameter response through MACE and the MLP; FD re-predicts them

The QEq neighbor list is fixed to the reference pairs on every displaced geometry: the
real-space terms are truncated at the cutoff without smoothing, so a pair crossing it
would add an energy step unrelated to the force. MACE rebuilds its own graph, which is
smooth through its radial envelope.

Verdict (exit status 1 if any fails):
    frozen force         best-step mean |frozen - fd_frozen| <= --frozen-tol
    parameter response   including it removes >= --explained of the frozen-force error,
                         1 - |whole - fd_full| / |frozen - fd_full| at the best step; the
                         float32 parameter MLP leaves ~1e-3 eV of energy noise, so
                         |whole - fd_full| itself only falls as 1/step
    electrodes           with --const-potential, no reassignment in any displaced geometry

有限差分测试通过是必要条件，但不是充分条件。
它只能证明一件事：在这次测试用的配置下，解析力等于当前实现的能量函数的负梯度。
能量函数本身对不对、生产环境下用的求解路径对不对、MD 中会不会出现能量跳变，
这些它都验证不了。
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Must be set before JAX is imported: the default allocator keeps freed GPU memory
# (up to 75% of the card), which starves MACE/PyTorch running on the same GPU.
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from ase.io import read
from mace_adqeq.const_potential import identify_electrode_atoms
from mace_adqeq.qeq import JAXQEqModel, QEqParameterPredictor

STRUCTURE = HERE / "pair_revised_equilibrated.extxyz"
PARAMS = HERE / "jax_model" / "qeq_mace_chi_hardness_eta_params.msgpack"
CONFIG = HERE / "jax_model" / "qeq_mace_chi_hardness_eta_config.json"
AXES = "xyz"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--structure", type=Path, default=STRUCTURE)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--params", type=Path, default=PARAMS)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, default=None, help="default: HERE/fd_results/<timestamp>")
    parser.add_argument("--const-potential", action="store_true")
    parser.add_argument("--device", default="cpu", help="MACE device; cuda is not deterministic")
    parser.add_argument("--mace-dtype", default="float64", help="MACE default_dtype; empty string uses the model dtype")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64", help="JAX QEq precision")
    parser.add_argument("--steps", type=float, nargs="+", default=[0.001, 0.002, 0.005, 0.01], help="displacements in Angstrom")
    parser.add_argument("--atoms", type=int, nargs="+", default=None, help="explicit atom indices; default selects automatically")
    parser.add_argument("--top-response", type=int, default=3, help="extra atoms with the largest |F_whole - F_frozen|")
    parser.add_argument("--total-charge", type=float, default=0.0)
    parser.add_argument("--max-pairs", type=int, default=50000)
    parser.add_argument("--frozen-tol", type=float, default=1.0e-5, help="eV/A")
    parser.add_argument("--explained", type=float, default=0.9, help="required fraction of the frozen-force error removed by the parameter response")
    return parser.parse_args()


def displaced(atoms, displacement):
    trial = atoms.copy()
    trial.set_positions(atoms.get_positions() + displacement)
    return trial


def electrode_signature(atoms):
    bottom, upper, _ = identify_electrode_atoms(
        np.asarray(atoms.get_cell()), atoms.get_positions(), atoms.get_chemical_symbols()
    )
    return tuple(bottom.tolist()), tuple(upper.tolist())


def select_atoms(atoms, response_norm, top_response):
    symbols = np.asarray(atoms.get_chemical_symbols())
    bottom, upper = electrode_signature(atoms)
    electrode = np.zeros(len(atoms), dtype=bool)
    electrode[list(bottom) + list(upper)] = True

    sulfur = np.flatnonzero(symbols == "S")
    oxygen = np.flatnonzero(symbols == "O")
    sulfate_oxygen = np.zeros(len(atoms), dtype=bool)
    for s_index in sulfur:
        distances = atoms.get_distances(s_index, oxygen, mic=True)
        sulfate_oxygen[oxygen[distances < 1.8]] = True

    groups = {
        "Zn_electrode": (symbols == "Zn") & electrode,
        "Zn_ion": (symbols == "Zn") & ~electrode,
        "S": symbols == "S",
        "O_sulfate": (symbols == "O") & sulfate_oxygen,
        "O_water": (symbols == "O") & ~sulfate_oxygen,
        "H": symbols == "H",
    }
    # Within each group, take the atom whose force is most sensitive to the parameter response.
    selected = {}
    for label, mask in groups.items():
        indices = np.flatnonzero(mask)
        if len(indices):
            selected[int(indices[np.argmax(response_norm[indices])])] = label

    extra = 0
    for index in np.argsort(-response_norm):
        if extra >= top_response:
            break
        if int(index) not in selected:
            selected[int(index)] = "top_response"
            extra += 1
    return selected


def step_errors(rows, steps):
    """Mean absolute differences per step (eV/A)."""
    table = {}
    for step in steps:
        subset = [row for row in rows if row["step"] == step]
        frozen_vs_full = float(np.mean([abs(row["frozen"] - row["fd_full"]) for row in subset]))
        whole_err = float(np.mean([abs(row["whole"] - row["fd_full"]) for row in subset]))
        table[str(step)] = {
            "frozen_vs_fd_frozen": float(np.mean([abs(row["frozen"] - row["fd_frozen"]) for row in subset])),
            "whole_vs_fd_full": whole_err,
            "frozen_vs_fd_full": frozen_vs_full,
            "explained": 1.0 - whole_err / frozen_vs_full if frozen_vs_full > 0.0 else None,
        }
    return table


def print_table(title, table, n):
    print(f"\n=== {title} ({n} per step), mean |difference| (eV/A) ===", flush=True)
    print(f"{'step':>8} {'frozen-fd_frozen':>17} {'whole-fd_full':>14} {'frozen-fd_full':>15} {'explained':>10}", flush=True)
    for step, item in table.items():
        explained = "-" if item["explained"] is None else f"{100.0 * item['explained']:.1f}%"
        print(
            f"{float(step):8.4f} {item['frozen_vs_fd_frozen']:17.3e} {item['whole_vs_fd_full']:14.3e} "
            f"{item['frozen_vs_fd_full']:15.3e} {explained:>10}",
            flush=True,
        )


def verdict(label, table, args):
    checks = []
    frozen = min(item["frozen_vs_fd_frozen"] for item in table.values())
    checks.append((f"frozen force, {label}", frozen <= args.frozen_tol, f"best {frozen:.2e} <= {args.frozen_tol:.0e} eV/A"))
    explained = [item["explained"] for item in table.values() if item["explained"] is not None]
    if explained:
        best = max(explained)
        checks.append((f"parameter response, {label}", best >= args.explained, f"explains {100.0 * best:.1f}% >= {100.0 * args.explained:.0f}%"))
    return checks


def main():
    args = parse_args()
    output = args.output or HERE / "fd_results" / datetime.now().strftime("%Y%m%d_%H%M%S")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.mkdir(parents=True)

    qeq_config = json.loads(args.config.read_text(encoding="utf-8"))["qeq"]
    predictor = QEqParameterPredictor(
        args.params, args.config, mace_device=args.device, mace_default_dtype=args.mace_dtype
    )
    qeq = JAXQEqModel(
        cutoff=float(qeq_config["cutoff"]),
        pme_grid=tuple(qeq_config["pme_grid"]),
        jitter=float(qeq_config["jitter"]),
        dtype=args.dtype,
        max_pairs=args.max_pairs,
        solver_mode="matrix",
        const_potential=args.const_potential,
    )

    atoms = read(args.structure, index=args.frame)
    symbols = atoms.get_chemical_symbols()
    print(f"structure={args.structure}  frame={args.frame}  natoms={len(atoms)}  pbc={atoms.pbc.tolist()}", flush=True)
    print(
        f"const_potential={args.const_potential}  device={args.device}  mace_dtype={args.mace_dtype!r}  "
        f"dtype={args.dtype}  steps={args.steps}",
        flush=True,
    )

    reference_pairs = qeq.neighbor_pairs(
        np.asarray(atoms.get_positions(), dtype=qeq.np_dtype),
        np.asarray(atoms.get_cell(), dtype=qeq.np_dtype),
    )
    qeq.neighbor_pairs = lambda positions, box: reference_pairs

    start = time.perf_counter()
    parameters = predictor.predict(atoms)
    fixed = {"chi": parameters.chi, "hardness": parameters.hardness, "eta": parameters.eta}
    ref_whole = qeq.calculate(atoms, predictor=predictor, total_charge=args.total_charge, compute_forces=True)
    ref_frozen = qeq.calculate(atoms, **fixed, total_charge=args.total_charge, compute_forces=True)
    response = ref_whole.forces - ref_frozen.forces
    response_norm = np.linalg.norm(response, axis=1)
    print(f"reference evaluation: {time.perf_counter() - start:.1f} s", flush=True)
    print(
        f"E={ref_whole.energy:.8f} eV  RMS |F_whole|={np.sqrt(np.mean(ref_whole.forces**2)):.3e}  "
        f"RMS |F_frozen|={np.sqrt(np.mean(ref_frozen.forces**2)):.3e}  "
        f"RMS |F_whole-F_frozen|={np.sqrt(np.mean(response**2)):.3e} eV/A",
        flush=True,
    )

    reference_signature = electrode_signature(atoms) if args.const_potential else None
    electrode_changes = []

    def energies(trial, tag):
        if reference_signature is not None and electrode_signature(trial) != reference_signature:
            electrode_changes.append(tag)
            print(f"WARNING: electrode assignment changed at {tag}", flush=True)
        full = qeq.calculate(trial, predictor=predictor, total_charge=args.total_charge, compute_forces=False).energy
        frozen = qeq.calculate(trial, **fixed, total_charge=args.total_charge, compute_forces=False).energy
        return full, frozen

    def fd_row(direction, step, whole, frozen, tag):
        full_plus, frozen_plus = energies(displaced(atoms, step * direction), f"{tag}+")
        full_minus, frozen_minus = energies(displaced(atoms, -step * direction), f"{tag}-")
        return {
            "step": step,
            "whole": whole,
            "frozen": frozen,
            "fd_full": -(full_plus - full_minus) / (2.0 * step),
            "fd_frozen": -(frozen_plus - frozen_minus) / (2.0 * step),
        }

    if args.atoms is None:
        selected = select_atoms(atoms, response_norm, args.top_response)
    else:
        selected = {int(index): "user" for index in args.atoms}

    start = time.perf_counter()
    component_rows = []
    for index, label in selected.items():
        for axis in range(3):
            direction = np.zeros_like(atoms.get_positions())
            direction[index, axis] = 1.0
            for step in args.steps:
                row = {
                    "atom": index,
                    "element": symbols[index],
                    "label": label,
                    "axis": AXES[axis],
                    **fd_row(
                        direction,
                        step,
                        float(ref_whole.forces[index, axis]),
                        float(ref_frozen.forces[index, axis]),
                        f"atom{index}{AXES[axis]}@{step}",
                    ),
                }
                component_rows.append(row)
                print(
                    f"atom={index:5d} {symbols[index]:>2s} {label:<13s} {AXES[axis]} step={step:.4f}  "
                    f"whole={row['whole']: .6e}  fd_full={row['fd_full']: .6e}  "
                    f"frozen={row['frozen']: .6e}  fd_frozen={row['fd_frozen']: .6e}",
                    flush=True,
                )

    directional_rows = []
    for name, vector in (("along_whole_force", ref_whole.forces), ("along_parameter_response", response)):
        norm = np.linalg.norm(vector)
        if norm == 0.0:
            continue
        direction = vector / norm
        for step in args.steps:
            row = {
                "direction": name,
                **fd_row(
                    direction,
                    step,
                    float(np.sum(ref_whole.forces * direction)),
                    float(np.sum(ref_frozen.forces * direction)),
                    f"{name}@{step}",
                ),
            }
            directional_rows.append(row)
            print(
                f"{name:<26s} step={step:.4f}  whole={row['whole']: .6e}  fd_full={row['fd_full']: .6e}  "
                f"frozen={row['frozen']: .6e}  fd_frozen={row['fd_frozen']: .6e}",
                flush=True,
            )
    print(f"finite differences: {time.perf_counter() - start:.1f} s", flush=True)

    component_table = step_errors(component_rows, args.steps)
    directional_table = step_errors(directional_rows, args.steps)
    print_table("components", component_table, len(selected) * 3)
    print_table("global directions", directional_table, len(directional_rows) // len(args.steps))

    checks = verdict("components", component_table, args) + verdict("global directions", directional_table, args)
    if args.const_potential:
        checks.append(("electrode assignment", not electrode_changes, f"{len(electrode_changes)} changes"))
    passed = all(ok for _, ok, _ in checks)
    print("\n=== verdict ===", flush=True)
    for name, ok, detail in checks:
        print(f"  {'OK  ' if ok else 'FAIL'}  {name:<36s} {detail}", flush=True)
    print(f"forces consistent with the energy: {'YES' if passed else 'NO'}", flush=True)

    for name, rows in (("fd_components.csv", component_rows), ("fd_directional.csv", directional_rows)):
        with (output / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    (output / "fd_summary.json").write_text(
        json.dumps(
            {
                "structure": str(args.structure),
                "frame": args.frame,
                "const_potential": args.const_potential,
                "device": args.device,
                "mace_dtype": args.mace_dtype,
                "dtype": args.dtype,
                "steps": args.steps,
                "energy": ref_whole.energy,
                "selected_atoms": {str(index): label for index, label in selected.items()},
                "components": component_table,
                "global_directions": directional_table,
                "electrode_changes": electrode_changes,
                "checks": [{"name": name, "ok": ok, "detail": detail} for name, ok, detail in checks],
                "passed": passed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote results to {output}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
