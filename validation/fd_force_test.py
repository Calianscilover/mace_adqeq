#!/usr/bin/env python3
"""Finite-difference check of MACE-adQEq forces.

Four force estimates are compared on selected Cartesian components and along
two global 3N directions:

    whole      analytic force including the parameter response (predictor path)
    frozen     analytic force with chi/hardness/eta fixed at the reference geometry
    fd_full    -dE/dR by central differences, parameters re-predicted at each geometry
    fd_frozen  -dE/dR by central differences, parameters fixed

Expected: whole == fd_full and frozen == fd_frozen up to finite-difference and
rounding noise; (fd_full - fd_frozen) == (whole - frozen) is the parameter response.

The energy noise floor is measured by repeated reference evaluations and by a
1D energy scan along each global direction (polynomial fit: slope -> force
projection, RMS residual -> noise). The defaults (CPU, float64 MACE, float64
QEq, matrix solver) are the validation setting: float32 MACE on GPU is not
deterministic and its energy noise swamps the parameter response. Use
--device cuda --mace-dtype float32 or --solver-mode hybrid to quantify the
production noise; the PASS/FAIL thresholds below are meant for the defaults.

The real-space QEq terms are truncated at the cutoff without smoothing, so a pair
crossing it between the +/- geometries adds an energy step that does not vanish
with the step size. Such rows are flagged, reported with their implied energy
jump, and excluded from the error statistics; the energy-scan fit gets one step
term per interval in which the QEq neighbor list changes.

The run ends with PASS/FAIL checks (determinism, frozen and whole force vs
finite differences at the best step, parameter response, energy scan, net
force, electrode assignment) and exits with status 1 if any check fails. A
failure within 3x the measured energy-noise floor is marked noise-limited.
See validation/fd_test.md for the full pre-production checklist.

有限差分测试通过是必要条件，但不是充分条件。
它只能证明一件事：在这次测试用的配置下，解析力等于当前实现的能量函数的负梯度。
能量函数本身对不对、生产环境下用的求解路径对不对、MD 中会不会出现能量跳变，
这些它都验证不了。
"""

import argparse
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path
import sys

# Must be set before JAX is imported: the default allocator keeps freed GPU memory
# (up to 75% of the card), which starves MACE/PyTorch running on the same GPU.
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from ase.io import read
from dmff.utils import pair_buffer_scales
from mace_adqeq.const_potential import identify_electrode_atoms
from mace_adqeq.qeq import JAXQEqModel, QEqParameterPredictor

STRUCTURE = HERE / "pair_revised_equilibrated.extxyz"
PARAMS = HERE / "jax_model" / "qeq_mace_chi_hardness_eta_params.msgpack"
CONFIG = HERE / "jax_model" / "qeq_mace_chi_hardness_eta_config.json"
AXES = "xyz"
SUMMARY_KEYS = (
    "whole_vs_fd_full_mae",
    "whole_vs_fd_full_max",
    "frozen_vs_fd_full_mae",
    "frozen_vs_fd_full_max",
    "frozen_vs_fd_frozen_mae",
    "frozen_vs_fd_frozen_max",
    "response_analytic_mean_abs",
    "response_fd_minus_analytic_mae",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--structure", type=Path, default=STRUCTURE)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--params", type=Path, default=PARAMS)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, default=None, help="default: HERE/fd_results/<timestamp>")
    parser.add_argument("--device", default="cpu", help="MACE device; cuda is not deterministic")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64", help="JAX QEq precision")
    parser.add_argument("--mace-dtype", default="float64", help="MACE default_dtype; features enter the float32 MLP anyway, empty string uses the model dtype")
    parser.add_argument("--solver-mode", choices=["matrix", "hybrid"], default="matrix", help="hybrid warm-starts the projected-gradient solver from the previous call")
    parser.add_argument("--pg-method", choices=["cg", "lbfgs"], default="cg")
    parser.add_argument("--pg-tol", type=float, default=1.0e-3)
    parser.add_argument("--dipole-axis", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--steps", type=float, nargs="+", default=[0.001, 0.002, 0.005, 0.01], help="displacements in Angstrom")
    parser.add_argument("--scan-points", type=int, default=21, help="energy scan points per direction; 0 disables the scan")
    parser.add_argument("--scan-range", type=float, default=0.02, help="energy scan half-range in Angstrom (3N norm)")
    parser.add_argument("--repeats", type=int, default=3, help="repeated reference energy evaluations")
    parser.add_argument("--atoms", type=int, nargs="+", default=None, help="explicit atom indices; default selects automatically")
    parser.add_argument("--top-response", type=int, default=3, help="extra atoms with the largest |F_whole - F_frozen|")
    parser.add_argument("--const-potential", action="store_true")
    parser.add_argument("--total-charge", type=float, default=0.0)
    parser.add_argument("--max-pairs", type=int, default=50000)
    parser.add_argument("--repeat-tol", type=float, default=1.0e-8, help="PASS threshold on the repeated reference energy spread (eV)")
    parser.add_argument("--frozen-tol", type=float, default=1.0e-5, help="PASS threshold on frozen vs fd_frozen at the best step (eV/A)")
    parser.add_argument("--whole-tol", type=float, default=1.0e-3, help="PASS threshold on whole vs fd_full at the best step and on the scan fits (eV/A)")
    parser.add_argument("--response-ratio", type=float, default=1.0e-2, help="PASS threshold on |FD response - analytic response| / |analytic response|")
    parser.add_argument("--net-force-tol", type=float, default=1.0e-2, help="PASS threshold on |sum of forces| (eV/A); PME breaks translation invariance at the grid-discretization level")
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


def fit_scan(alphas, energies, jump_positions=(), degree=4):
    """Fit a 1D energy scan with a polynomial plus one step per QEq neighbor-list change.

    Returns (-dE/dalpha at 0, RMS residual, standard error of the slope); all None when
    there are too many neighbor-list changes for the number of scan points.
    """
    shifted = energies - energies[len(energies) // 2]
    columns = [alphas**power for power in range(degree + 1)]
    columns += [(alphas > position).astype(float) for position in jump_positions]
    design = np.stack(columns, axis=1)
    dof = len(alphas) - design.shape[1]
    if dof <= 0:
        return None, None, None
    coefficients = np.linalg.lstsq(design, shifted, rcond=None)[0]
    residual = shifted - design @ coefficients
    rms = float(np.sqrt(np.mean(residual**2)))
    sigma2 = float(residual @ residual) / dof
    slope_std = float(np.sqrt(sigma2 * np.linalg.pinv(design.T @ design)[1, 1]))
    return -float(coefficients[1]), rms, slope_std


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, steps):
    """Error statistics per step over the rows whose ± geometries share the QEq neighbor list."""
    summary = {}
    for step in steps:
        all_rows = [row for row in rows if row["step"] == step]
        if not all_rows:
            continue
        subset = [row for row in all_rows if not row["cutoff_crossed"]]
        counts = {"n": len(all_rows), "n_clean": len(subset), "n_cutoff_crossed": len(all_rows) - len(subset)}
        if not subset:
            summary[str(step)] = {**counts, **dict.fromkeys(SUMMARY_KEYS)}
            continue
        err_whole = np.array([row["whole"] - row["fd_full"] for row in subset])
        err_frozen_full = np.array([row["frozen"] - row["fd_full"] for row in subset])
        err_frozen = np.array([row["frozen"] - row["fd_frozen"] for row in subset])
        analytic_response = np.array([row["whole"] - row["frozen"] for row in subset])
        fd_response = np.array([row["fd_full"] - row["fd_frozen"] for row in subset])
        summary[str(step)] = {
            **counts,
            "whole_vs_fd_full_mae": float(np.mean(np.abs(err_whole))),
            "whole_vs_fd_full_max": float(np.max(np.abs(err_whole))),
            "frozen_vs_fd_full_mae": float(np.mean(np.abs(err_frozen_full))),
            "frozen_vs_fd_full_max": float(np.max(np.abs(err_frozen_full))),
            "frozen_vs_fd_frozen_mae": float(np.mean(np.abs(err_frozen))),
            "frozen_vs_fd_frozen_max": float(np.max(np.abs(err_frozen))),
            "response_analytic_mean_abs": float(np.mean(np.abs(analytic_response))),
            "response_fd_minus_analytic_mae": float(np.mean(np.abs(fd_response - analytic_response))),
        }
    return summary


def print_summary(title, summary):
    def fmt(value, width):
        return f"{'-':>{width}}" if value is None else f"{value:{width}.3e}"

    print(f"\n=== {title}, rows crossing the QEq cutoff excluded (eV/A) ===")
    print(f"{'step':>8} {'n':>4} {'clean':>5} {'whole-fd':>12} {'frozen-fd':>12} {'frozen-fdfrz':>13} {'|resp|':>10} {'resp err':>10}")
    for step, item in summary.items():
        print(
            f"{float(step):8.4f} {item['n']:4d} {item['n_clean']:5d} "
            f"{fmt(item['whole_vs_fd_full_mae'], 12)} {fmt(item['frozen_vs_fd_full_mae'], 12)} "
            f"{fmt(item['frozen_vs_fd_frozen_mae'], 13)} {fmt(item['response_analytic_mean_abs'], 10)} "
            f"{fmt(item['response_fd_minus_analytic_mae'], 10)}",
            flush=True,
        )


def print_crossings(title, rows, label):
    crossed = [row for row in rows if row["cutoff_crossed"]]
    print(f"\n=== {title}: {len(crossed)} of {len(rows)} rows cross the QEq cutoff (excluded above) ===", flush=True)
    if crossed:
        print(f"{'row':<34s} {'step':>7} {'pairs':>5} {'jump_frozen eV':>15} {'jump_full eV':>13}", flush=True)
    for row in crossed:
        print(
            f"{label(row):<34s} {row['step']:7.4f} {row['pairs_changed']:5d} "
            f"{row['implied_jump_frozen']:15.3e} {row['implied_jump_full']:13.3e}",
            flush=True,
        )


def best_step(summary, key):
    """(smallest value of key over the steps, its step); (None, None) if no step has clean rows."""
    values = [(item[key], float(step)) for step, item in summary.items() if item[key] is not None]
    return min(values) if values else (None, None)


def make_check(name, value, threshold, floor=None, note=""):
    """A check is SKIP without a value and FAIL above threshold; a FAIL within 3x the
    energy-noise floor is marked noise-limited, i.e. not resolvable with this noise."""
    if value is None:
        status, note = "SKIP", note or "no rows without a QEq cutoff crossing"
    elif value <= threshold:
        status = "PASS"
    else:
        status = "FAIL"
        if floor is not None and value <= 3.0 * floor:
            note = f"noise-limited: {value / floor:.1f}x the noise floor {floor:.1e}"
    return {"name": name, "value": value, "threshold": threshold, "status": status, "note": note}


def evaluate_checks(args, repeat_spread, component_summary, directional_summary, scan_summary, net_force, electrode_changes):
    """Validation criteria; FD statistics exclude rows whose ± geometries differ in the QEq neighbor list."""
    noise_full = max((item["noise_rms_full"] for item in scan_summary.values() if item["noise_rms_full"] is not None), default=None)

    def fd_floor(step):
        # Two independent energy errors of RMS noise_full in a central difference of width 2*step.
        return None if noise_full is None or step is None else np.sqrt(2.0) * noise_full / (2.0 * step)

    checks = [
        make_check(f"repeat spread {name} (eV)", spread, args.repeat_tol)
        for name, spread in repeat_spread.items()
        if spread is not None
    ]
    for label, summary in (("components", component_summary), ("directional", directional_summary)):
        value, _ = best_step(summary, "frozen_vs_fd_frozen_mae")
        checks.append(make_check(f"{label}: frozen vs fd_frozen, best step (eV/A)", value, args.frozen_tol))
        value, step = best_step(summary, "whole_vs_fd_full_mae")
        checks.append(make_check(f"{label}: whole vs fd_full, best step (eV/A)", value, args.whole_tol, fd_floor(step)))
        responses = [item["response_analytic_mean_abs"] for item in summary.values() if item["response_analytic_mean_abs"] is not None]
        error, step = best_step(summary, "response_fd_minus_analytic_mae")
        if responses and max(responses) > 0.0:
            response = max(responses)
            floor = fd_floor(step)
            checks.append(
                make_check(
                    f"{label}: response error / |response|, best step",
                    None if error is None else error / response,
                    args.response_ratio,
                    None if floor is None else floor / response,
                )
            )
    for name, item in scan_summary.items():
        for kind, analytic in (("full", "whole"), ("frozen", "frozen")):
            fit = item[f"fit_{kind}"]
            checks.append(
                make_check(
                    f"scan {name}: |fit_{kind} - {analytic}| (eV/A)",
                    None if fit is None else abs(fit - item[analytic]),
                    args.whole_tol,
                    item[f"slope_std_{kind}"],
                    "" if fit is not None else "too many QEq neighbor-list changes for the scan fit",
                )
            )
    for name, vector in net_force.items():
        checks.append(make_check(f"|net force {name}| (eV/A)", float(np.linalg.norm(vector)), args.net_force_tol))
    if args.const_potential:
        checks.append(make_check("electrode assignment changes", float(len(electrode_changes)), 0.0))
    return checks


def print_checks(checks):
    print("\n=== PASS/FAIL ===", flush=True)
    for check in checks:
        value = "-" if check["value"] is None else f"{check['value']:.3e}"
        note = f"  ({check['note']})" if check["note"] else ""
        print(f"  {check['status']:<4s}  {check['name']:<62s} {value:>11s}  <= {check['threshold']:.1e}{note}", flush=True)
    failed = [check for check in checks if check["status"] == "FAIL"]
    skipped = sum(check["status"] == "SKIP" for check in checks)
    noise_limited = sum(check["note"].startswith("noise-limited") for check in failed)
    if failed:
        overall = f"FAIL ({len(failed)} of {len(checks)} checks, {noise_limited} of them noise-limited)"
    else:
        overall = "PASS"
    if skipped:
        overall += f", {skipped} skipped"
    print(f"overall: {overall}", flush=True)
    return not failed


def main():
    args = parse_args()
    output = args.output or HERE / "fd_results" / datetime.now().strftime("%Y%m%d_%H%M%S")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.mkdir(parents=True)

    config = json.loads(args.config.read_text(encoding="utf-8"))
    qeq_config = config["qeq"]
    mace_dtype = args.mace_dtype
    predictor = QEqParameterPredictor(
        args.params, args.config, mace_device=args.device, mace_default_dtype=mace_dtype
    )
    qeq = JAXQEqModel(
        cutoff=float(qeq_config["cutoff"]),
        pme_grid=tuple(qeq_config["pme_grid"]),
        jitter=float(qeq_config["jitter"]),
        dtype=args.dtype,
        max_pairs=args.max_pairs,
        solver_mode=args.solver_mode,
        pg_method=args.pg_method,
        pg_tolerance=args.pg_tol,
        dipole_axis=args.dipole_axis,
        const_potential=args.const_potential,
    )

    atoms = read(args.structure, index=args.frame)
    print(f"structure={args.structure}  frame={args.frame}  natoms={len(atoms)}  pbc={atoms.pbc.tolist()}", flush=True)
    print(
        f"device={args.device}  dtype={args.dtype}  mace_dtype={mace_dtype!r}  solver_mode={args.solver_mode}  "
        f"pg_method={args.pg_method}  dipole_axis={args.dipole_axis}  const_potential={args.const_potential}  steps={args.steps}",
        flush=True,
    )
    if args.device != "cpu" or mace_dtype != "float64" or args.dtype != "float64" or args.solver_mode != "matrix":
        print("NOTE: not the validation setting (cpu, float64 MACE/QEq, matrix); PASS/FAIL thresholds may not apply", flush=True)

    start = time.perf_counter()
    reference_parameters = predictor.predict(atoms)
    ref_whole = qeq.calculate(atoms, predictor=predictor, total_charge=args.total_charge, compute_forces=True)
    ref_frozen = qeq.calculate(
        atoms,
        chi=reference_parameters.chi,
        hardness=reference_parameters.hardness,
        eta=reference_parameters.eta,
        total_charge=args.total_charge,
        compute_forces=True,
    )
    print(f"reference evaluation: {time.perf_counter() - start:.1f} s", flush=True)
    print(
        f"E_whole={ref_whole.energy:.8f}  E_frozen={ref_frozen.energy:.8f}  "
        f"diff={ref_whole.energy - ref_frozen.energy:.3e} eV (should be ~0)",
        flush=True,
    )

    response = ref_whole.forces - ref_frozen.forces
    response_norm = np.linalg.norm(response, axis=1)
    print(
        f"RMS |F_whole|={np.sqrt(np.mean(ref_whole.forces**2)):.4e}  "
        f"RMS |F_frozen|={np.sqrt(np.mean(ref_frozen.forces**2)):.4e}  "
        f"RMS |F_whole-F_frozen|={np.sqrt(np.mean(response**2)):.4e}  "
        f"max atom |F_whole-F_frozen|={response_norm.max():.4e} eV/A",
        flush=True,
    )
    net_force = {
        "whole": ref_whole.forces.sum(axis=0).tolist(),
        "frozen": ref_frozen.forces.sum(axis=0).tolist(),
    }
    print(
        f"net force (should be ~0): whole={np.round(net_force['whole'], 6).tolist()}  "
        f"frozen={np.round(net_force['frozen'], 6).tolist()} eV/A",
        flush=True,
    )

    reference_signature = electrode_signature(atoms) if args.const_potential else None
    electrode_changes = []

    def check_electrodes(trial, tag):
        if reference_signature is not None and electrode_signature(trial) != reference_signature:
            electrode_changes.append(tag)
            print(f"WARNING: electrode assignment changed at {tag}", flush=True)

    def energy_full(trial):
        return qeq.calculate(trial, predictor=predictor, total_charge=args.total_charge, compute_forces=False).energy

    def energy_frozen(trial):
        return qeq.calculate(
            trial,
            chi=reference_parameters.chi,
            hardness=reference_parameters.hardness,
            eta=reference_parameters.eta,
            total_charge=args.total_charge,
            compute_forces=False,
        ).energy

    def pair_keys(trial):
        """Sorted i*N+j keys of the real (unpadded) QEq pairs, as built inside qeq.calculate."""
        pairs = np.asarray(
            qeq.neighbor_pairs(
                np.asarray(trial.get_positions(), dtype=qeq.np_dtype),
                np.asarray(trial.get_cell(), dtype=qeq.np_dtype),
            )
        )
        real = pairs[np.asarray(pair_buffer_scales(pairs)) > 0, :2].astype(np.int64)
        return np.unique(real.min(axis=1) * len(trial) + real.max(axis=1))

    def pairs_changed(first, second):
        return int(len(np.setxor1d(pair_keys(first), pair_keys(second), assume_unique=True)))

    def fd_projection(direction, step, tag):
        """Central differences; also the number of QEq pairs that differ between the ± geometries.

        The real-space PME and Gaussian-correction terms are truncated at the cutoff without
        smoothing, so a pair crossing it adds an energy step that does not vanish with the
        step size and must not be read as a force error.
        """
        plus = displaced(atoms, step * direction)
        minus = displaced(atoms, -step * direction)
        check_electrodes(plus, f"{tag}+")
        check_electrodes(minus, f"{tag}-")
        fd_full = -(energy_full(plus) - energy_full(minus)) / (2.0 * step)
        fd_frozen = -(energy_frozen(plus) - energy_frozen(minus)) / (2.0 * step)
        return fd_full, fd_frozen, pairs_changed(plus, minus)

    def fd_row(step, whole, frozen, fd_full, fd_frozen, changed):
        return {
            "step": step,
            "whole": whole,
            "frozen": frozen,
            "fd_full": fd_full,
            "fd_frozen": fd_frozen,
            "whole_minus_fd_full": whole - fd_full,
            "frozen_minus_fd_frozen": frozen - fd_frozen,
            "pairs_changed": changed,
            "cutoff_crossed": changed > 0,
            # E(+) - E(-) minus the analytic prediction: the energy step when a pair crossed the cutoff.
            "implied_jump_frozen": (frozen - fd_frozen) * 2.0 * step,
            "implied_jump_full": (whole - fd_full) * 2.0 * step,
        }

    repeats = {"full": [], "frozen": []}
    for _ in range(args.repeats):
        repeats["full"].append(energy_full(atoms))
        repeats["frozen"].append(energy_frozen(atoms))
    repeat_spread = {name: float(np.ptp(values)) if values else None for name, values in repeats.items()}
    print(
        f"\nrepeat energies at reference ({args.repeats}x): "
        f"spread full={repeat_spread['full']}  frozen={repeat_spread['frozen']} eV",
        flush=True,
    )

    directions = {"along_whole_force": ref_whole.forces}
    if np.linalg.norm(response) > 0.0:
        directions["along_parameter_response"] = response
    directions = {name: vector / np.linalg.norm(vector) for name, vector in directions.items()}

    scan_rows = []
    scan_summary = {}
    if args.scan_points > 0:
        alphas = np.linspace(-args.scan_range, args.scan_range, 2 * (args.scan_points // 2) + 1)
        start = time.perf_counter()
        for name, direction in directions.items():
            geometries = [displaced(atoms, alpha * direction) for alpha in alphas]
            e_full = np.array([energy_full(trial) for trial in geometries])
            e_frozen = np.array([energy_frozen(trial) for trial in geometries])
            keys = [pair_keys(trial) for trial in geometries]
            # One step regressor per scan interval in which the QEq neighbor list changes.
            jumps = [
                0.5 * (alphas[k] + alphas[k + 1])
                for k in range(len(alphas) - 1)
                if not np.array_equal(keys[k], keys[k + 1])
            ]
            slope_full, noise_full, std_full = fit_scan(alphas, e_full, jumps)
            slope_frozen, noise_frozen, std_frozen = fit_scan(alphas, e_frozen, jumps)
            scan_summary[name] = {
                "whole": float(np.sum(ref_whole.forces * direction)),
                "frozen": float(np.sum(ref_frozen.forces * direction)),
                "fit_full": slope_full,
                "fit_frozen": slope_frozen,
                "slope_std_full": std_full,
                "slope_std_frozen": std_frozen,
                "noise_rms_full": noise_full,
                "noise_rms_frozen": noise_frozen,
                "neighbor_list_changes": len(jumps),
            }
            scan_rows.extend(
                {"direction": name, "alpha": float(alpha), "energy_full": float(a), "energy_frozen": float(b)}
                for alpha, a, b in zip(alphas, e_full, e_frozen)
            )

        def fmt(value, spec):
            return f"{'-':>{spec.split('.')[0]}}" if value is None else f"{value:{spec}}"

        print(f"\n=== energy scan, {len(alphas)} points in +/-{args.scan_range} A ({time.perf_counter() - start:.1f} s) ===", flush=True)
        print("fit: degree-4 polynomial plus one step per interval where the QEq neighbor list changes", flush=True)
        print(f"{'direction':<26s} {'whole':>11} {'fit_full':>11} {'frozen':>11} {'fit_frozen':>11} {'noise_full':>11} {'noise_frz':>11} {'nl_changes':>10}")
        for name, item in scan_summary.items():
            print(
                f"{name:<26s} {item['whole']:11.5f} {fmt(item['fit_full'], '11.5f')} {item['frozen']:11.5f} "
                f"{fmt(item['fit_frozen'], '11.5f')} {fmt(item['noise_rms_full'], '11.3e')} "
                f"{fmt(item['noise_rms_frozen'], '11.3e')} {item['neighbor_list_changes']:10d}",
                flush=True,
            )

    if args.atoms is None:
        selected = select_atoms(atoms, response_norm, args.top_response)
    else:
        selected = {int(index): "user" for index in args.atoms}
    symbols = atoms.get_chemical_symbols()
    print("\nselected atoms:", flush=True)
    for index, label in selected.items():
        print(f"  {index:5d} {symbols[index]:>2s} {label:<14s} |F_whole-F_frozen|={response_norm[index]:.4e}", flush=True)

    component_rows = []
    start = time.perf_counter()
    for index, label in selected.items():
        for axis in range(3):
            direction = np.zeros_like(atoms.get_positions())
            direction[index, axis] = 1.0
            whole = float(ref_whole.forces[index, axis])
            frozen = float(ref_frozen.forces[index, axis])
            for step in args.steps:
                fd_full, fd_frozen, changed = fd_projection(direction, step, f"atom{index}{AXES[axis]}@{step}")
                component_rows.append(
                    {
                        "atom": index,
                        "element": symbols[index],
                        "label": label,
                        "axis": AXES[axis],
                        **fd_row(step, whole, frozen, fd_full, fd_frozen, changed),
                    }
                )
                print(
                    f"atom={index:5d} {symbols[index]:>2s} {AXES[axis]} step={step:.4f}  "
                    f"whole={whole: .6e}  fd_full={fd_full: .6e}  "
                    f"frozen={frozen: .6e}  fd_frozen={fd_frozen: .6e}"
                    + (f"  [cutoff crossed: {changed} pairs]" if changed else ""),
                    flush=True,
                )
    print(f"component tests: {time.perf_counter() - start:.1f} s", flush=True)

    directional_rows = []
    for name, direction in directions.items():
        whole = float(np.sum(ref_whole.forces * direction))
        frozen = float(np.sum(ref_frozen.forces * direction))
        for step in args.steps:
            fd_full, fd_frozen, changed = fd_projection(direction, step, f"{name}@{step}")
            directional_rows.append({"direction": name, **fd_row(step, whole, frozen, fd_full, fd_frozen, changed)})
            print(
                f"{name:<26s} step={step:.4f}  whole={whole: .6e}  fd_full={fd_full: .6e}  "
                f"frozen={frozen: .6e}  fd_frozen={fd_frozen: .6e}"
                + (f"  [cutoff crossed: {changed} pairs]" if changed else ""),
                flush=True,
            )

    print_crossings("component tests", component_rows, lambda row: f"atom {row['atom']} {row['element']} {row['axis']}")
    print_crossings("directional tests", directional_rows, lambda row: row["direction"])
    component_summary = summarize(component_rows, args.steps)
    directional_summary = summarize(directional_rows, args.steps)
    print_summary("component tests, mean |error|", component_summary)
    print_summary("directional tests, mean |error|", directional_summary)
    if electrode_changes:
        print(f"\nWARNING: electrode assignment changed in {len(electrode_changes)} displaced geometries", flush=True)
    checks = evaluate_checks(
        args, repeat_spread, component_summary, directional_summary, scan_summary, net_force, electrode_changes
    )
    passed = print_checks(checks)

    write_csv(output / "fd_components.csv", component_rows)
    write_csv(output / "fd_directional.csv", directional_rows)
    write_csv(output / "fd_scan.csv", scan_rows)
    np.savez(
        output / "fd_reference.npz",
        positions=atoms.get_positions(),
        symbols=np.asarray(symbols),
        forces_whole=ref_whole.forces,
        forces_frozen=ref_frozen.forces,
        charges=ref_whole.charges,
        chi=ref_whole.chi,
        hardness=ref_whole.hardness,
        eta=ref_whole.eta,
        selected_atoms=np.asarray(list(selected.keys()), dtype=int),
    )
    (output / "fd_summary.json").write_text(
        json.dumps(
            {
                "structure": str(args.structure),
                "frame": args.frame,
                "natoms": len(atoms),
                "device": args.device,
                "dtype": args.dtype,
                "mace_dtype": mace_dtype,
                "solver_mode": args.solver_mode,
                "pg_method": args.pg_method,
                "pg_tol": args.pg_tol,
                "dipole_axis": args.dipole_axis,
                "const_potential": args.const_potential,
                "total_charge": args.total_charge,
                "steps": args.steps,
                "energy_whole": ref_whole.energy,
                "energy_frozen": ref_frozen.energy,
                "repeat_energies": repeats,
                "repeat_spread": repeat_spread,
                "scan": scan_summary,
                "selected_atoms": {str(index): label for index, label in selected.items()},
                "electrode_changes": electrode_changes,
                "components": component_summary,
                "directional": directional_summary,
                "net_force": net_force,
                "cutoff_crossings": {
                    "components": sum(row["cutoff_crossed"] for row in component_rows),
                    "directional": sum(row["cutoff_crossed"] for row in directional_rows),
                },
                "checks": checks,
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
