#!/usr/bin/env python3
"""Committee-driven collection of active-learning candidates with MACE + QEq MD.

Walkers run Langevin NVT with the production MACE + QEq calculator (outermost Zn layers
fixed, as in md_run.py), cycling over --temperatures. Every --check-interval steps the frame
is scored by a committee of short-range MACE models trained on the same residual labels
with different seeds. Per atom,

    relative std = |std of committee forces| / (|F_short,mean + F_QEq| + --force-regularizer)

and the frame's uncertainty is its maximum over mobile atoms (QEq is shared by all members,
so the disagreement is that of the short-range part, measured against the total force).

    > --error-threshold        saved as high_uncertainty, at most every --candidate-interval steps
    >= --hard-error-threshold  saved as hard_uncertainty; the walker rewinds --rewind steps,
                               draws new velocities and continues
    crash                      force > --max-force, temperature > --max-temperature or NaN:
                               every --failure-stride-th of the last --keep-last steps is scored
                               and those above --error-threshold are saved as failure (the
                               crash frame always); then the walker rewinds as above

After --max-hard-events rewinds a walker gives up. --periodic-interval > 0 additionally saves
unconditional frames (source=periodic) as a control. Every score goes to scores.csv, which
is what the two thresholds should be calibrated against. Model and committee results are
stored as model_* / committee_* keys, never as forces/energy.

    python al_collect.py --committee data/committee/short_*.model --output al/round1_s1 --seed 1
"""

import os

# Must be set before JAX is imported: the default allocator keeps freed GPU memory
# (up to 75% of the card), which starves MACE/PyTorch running on the same GPU.
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import argparse
from collections import deque
import csv
import json
from pathlib import Path
import time

import numpy as np
from ase import units
from ase.constraints import FixAtoms
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from mace.calculators import MACECalculator

from md_run import (
    MACE_MODEL, QEQ_CONFIG, QEQ_PARAMS, STRUCTURE, Failure,
    build_calculator, capture_frame, outer_zn_layers, step_state, stop_reason,
)


class HardUncertainty(RuntimeError):
    pass


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--committee", type=Path, nargs="+", required=True, help="short-range MACE models (residual labels), or a directory of *.model")
    parser.add_argument("--start", type=Path, nargs="+", default=[STRUCTURE], help="start frames (all frames of every file)")
    parser.add_argument("--mace-model", type=Path, default=MACE_MODEL, help="short-range MACE that drives the MD")
    parser.add_argument("--params", type=Path, default=QEQ_PARAMS)
    parser.add_argument("--config", type=Path, default=QEQ_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mace-dtype", default="float32")
    parser.add_argument("--const-potential", action="store_true")
    parser.add_argument("--pg-tol", type=float, default=1.0e-5)
    parser.add_argument("--walkers", type=int, default=6)
    parser.add_argument("--temperatures", type=float, nargs="+", default=[300.0, 400.0, 500.0], help="K; cycled over walkers")
    parser.add_argument("--steps", type=int, default=10000, help="MD steps per walker, including rewound segments")
    parser.add_argument("--timestep", type=float, default=0.5, help="fs")
    parser.add_argument("--friction", type=float, default=10.0, help="Langevin friction (1/ps)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fix-layers", type=int, default=1)
    parser.add_argument("--layer-gap", type=float, default=1.0)
    parser.add_argument("--check-interval", type=int, default=10, help="steps between committee scores")
    parser.add_argument("--error-threshold", type=float, default=0.4, help="soft threshold on the max relative force std")
    parser.add_argument("--hard-error-threshold", type=float, default=0.8, help="hard threshold: save, rewind, redraw velocities")
    parser.add_argument("--candidate-interval", type=int, default=100, help="min steps between high_uncertainty candidates")
    parser.add_argument("--force-regularizer", type=float, default=0.2, help="eV/A added to |F| in the relative std")
    parser.add_argument("--periodic-interval", type=int, default=0, help="steps between unconditional frames; 0 = off")
    parser.add_argument("--keep-last", type=int, default=60, help="steps kept for a crash")
    parser.add_argument("--failure-stride", type=int, default=2, help="score every n-th of the kept steps on a crash")
    parser.add_argument("--rewind", type=int, default=40, help="steps to go back after a hard event")
    parser.add_argument("--max-hard-events", type=int, default=5, help="per walker; hard thresholds and crashes")
    parser.add_argument("--max-force", type=float, default=50.0, help="eV/A on a mobile atom")
    parser.add_argument("--max-temperature", type=float, default=2000.0, help="K")
    return parser.parse_args()


class Committee:
    def __init__(self, models, device, dtype, regularizer, mobile):
        models = [model for path in models for model in (sorted(path.glob("*.model")) if path.is_dir() else [path])]
        if len(models) < 2:
            raise ValueError(f"a committee needs at least two models, got {[str(model) for model in models]}")
        print("committee: " + ", ".join(model.name for model in models), flush=True)
        self.models = models
        self.calculator = MACECalculator(model_paths=[str(path) for path in models], device=device, default_dtype=dtype)
        self.regularizer = regularizer
        self.mobile = mobile

    def score(self, frame):
        """Annotate frame with committee disagreement; return its max relative force std."""
        atoms = frame.copy()
        atoms.calc = self.calculator
        atoms.get_potential_energy()
        forces = np.asarray(self.calculator.results["forces_comm"], dtype=float)
        energies = np.asarray(self.calculator.results["energy_comm"], dtype=float)
        std = np.sqrt(np.sum(np.var(forces, axis=0), axis=1))
        total = forces.mean(axis=0) + frame.arrays["model_qeq_forces"]
        relative = std / (np.linalg.norm(total, axis=1) + self.regularizer)
        fixed = np.setdiff1d(np.arange(len(frame)), self.mobile)
        std[fixed] = 0.0
        relative[fixed] = 0.0
        worst = int(np.argmax(relative))
        frame.arrays["committee_force_std"] = std
        frame.arrays["committee_force_relative_std"] = relative
        frame.info.update(
            committee_force_max_std=float(std.max()),
            committee_force_max_relative_std=float(relative[worst]),
            committee_max_atom=worst,
            committee_energy_std_per_atom=float(np.std(energies) / len(frame)),
        )
        return float(relative[worst])


class CandidateWriter:
    def __init__(self, output):
        self.path = output / "candidates.extxyz"
        self.counts = {"high_uncertainty": 0, "hard_uncertainty": 0, "failure": 0, "periodic": 0}
        self.events = (output / "events.csv").open("w", newline="", encoding="utf-8")
        self.event_log = csv.writer(self.events)
        self.event_log.writerow(["event_id", "walker", "walker_step", "target_temperature_K", "kind", "detail", "frames_written"])
        self.scores = (output / "scores.csv").open("w", newline="", encoding="utf-8")
        self.score_log = csv.writer(self.scores)
        self.score_log.writerow(["walker", "walker_step", "max_relative_std", "max_std_eV_A", "atom", "energy_std_eV_atom"])
        self.event_id = 0

    def log_score(self, frame):
        info = frame.info
        self.score_log.writerow([
            info["walker"], info["walker_step"], f"{info['committee_force_max_relative_std']:.5f}",
            f"{info['committee_force_max_std']:.5f}", info["committee_max_atom"], f"{info['committee_energy_std_per_atom']:.3e}",
        ])

    def add(self, frames, source, **info):
        frames = [frame.copy() for frame in frames]
        for frame in frames:
            frame.info.update(source=source, **info)
        write(self.path, frames, format="extxyz", append=self.path.exists())
        self.counts[source] += len(frames)

    def add_event(self, frames, kind, walker, walker_step, temperature, detail):
        for frame in frames:
            frame.info["steps_before_event"] = walker_step - frame.info["walker_step"]
        if frames:
            self.add(frames, kind, event_id=self.event_id)
        self.event_log.writerow([self.event_id, walker, walker_step, temperature, kind, detail, len(frames)])
        self.events.flush()
        self.scores.flush()
        self.event_id += 1

    def close(self):
        self.events.close()
        self.scores.close()


def run_walker(walker, start, temperature, rng, args, calc, committee, mobile, fixed, writer):
    atoms = start.copy()
    atoms.set_constraint(FixAtoms(indices=fixed))
    atoms.calc = calc
    calc.reset_charge_state()
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature, rng=rng, force_temp=True)
    recent = deque(maxlen=args.keep_last)
    done = hard_events = 0
    last_candidate = -args.candidate_interval
    scores = []
    start_time = time.perf_counter()

    while done < args.steps:
        dynamics = Langevin(
            atoms,
            timestep=args.timestep * units.fs,
            temperature_K=temperature,
            friction=args.friction / (1000.0 * units.fs),
            fixcm=False,
            rng=rng,
        )
        base = done

        def observe():
            nonlocal last_candidate
            step = base + dynamics.nsteps
            state = step_state(atoms, mobile)
            frame = capture_frame(atoms, state, walker=walker, walker_step=step, target_temperature_K=temperature)
            recent.append(frame)
            reason = stop_reason(state, args.max_force, args.max_temperature)
            if reason is not None:
                raise Failure(reason)
            if dynamics.nsteps == 0:
                return
            if step % args.check_interval == 0:
                uncertainty = committee.score(frame)
                writer.log_score(frame)
                scores.append(uncertainty)
                if uncertainty >= args.hard_error_threshold:
                    raise HardUncertainty(f"relative std {uncertainty:.3f} on atom {frame.info['committee_max_atom']}")
                if uncertainty > args.error_threshold and step - last_candidate >= args.candidate_interval:
                    writer.add([frame], "high_uncertainty")
                    last_candidate = step
            if args.periodic_interval and step % args.periodic_interval == 0:
                if "committee_force_max_relative_std" not in frame.info:
                    committee.score(frame)
                writer.add([frame], "periodic")
            if step % 1000 == 0:
                elapsed = time.perf_counter() - start_time
                recent_scores = scores[-100:]
                print(
                    f"  walker {walker}  step {step:6d}  T={state['temperature']:6.1f} K  maxF={state['max_force']:6.2f} eV/A  "
                    f"rel_std median/max (last 100 checks) {np.median(recent_scores):.3f}/{np.max(recent_scores):.3f}  "
                    f"hard={hard_events}  candidates={writer.counts}  {elapsed / step:.3f} s/step",
                    flush=True,
                )

        dynamics.attach(observe, interval=1)
        try:
            dynamics.run(args.steps - done)
            done = args.steps
        except HardUncertainty as event:
            done = base + dynamics.nsteps
            detail = f"hard_uncertainty, {event}"
            writer.add_event([recent[-1]], "hard_uncertainty", walker, done, temperature, str(event))
        except Failure as event:
            done = base + dynamics.nsteps
            detail = f"failure, {event}"
            frames = list(recent)
            kept = []
            for frame in frames[::-1][::args.failure_stride][::-1]:
                uncertainty = committee.score(frame)
                writer.log_score(frame)
                if uncertainty > args.error_threshold or frame is frames[-1]:
                    kept.append(frame)
            writer.add_event(kept, "failure", walker, done, temperature, str(event))
        else:
            break

        hard_events += 1
        print(f"  walker {walker}  hard event {hard_events} at step {done}: {detail}", flush=True)
        if hard_events >= args.max_hard_events:
            print(f"  walker {walker}  stopped after {hard_events} hard events", flush=True)
            break
        frames = list(recent)
        restart = frames[max(0, len(frames) - 1 - args.rewind)]
        atoms.set_positions(restart.positions, apply_constraint=False)
        calc.reset_charge_state()
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature, rng=rng, force_temp=True)
        recent.clear()

    return {
        "walker": walker,
        "temperature_K": temperature,
        "steps": done,
        "hard_events": hard_events,
        "relative_std_quantiles": {q: float(np.quantile(scores, q)) for q in (0.5, 0.9, 0.99)} if scores else None,
        "seconds": time.perf_counter() - start_time,
    }


def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    args.output.mkdir(parents=True)
    (args.output / "collect.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2), encoding="utf-8")

    pool = [frame for path in args.start for frame in read(path, index=":")]
    bottom, top = outer_zn_layers(pool[0], args.fix_layers, args.layer_gap)
    fixed = np.concatenate((bottom, top))
    mobile = np.setdiff1d(np.arange(len(pool[0])), fixed)
    calc = build_calculator(args)
    committee = Committee(args.committee, args.device, args.mace_dtype, args.force_regularizer, mobile)
    rng = np.random.default_rng(args.seed)
    writer = CandidateWriter(args.output)
    print(
        f"{len(pool)} start frames, {args.walkers} walkers x {args.steps} steps, T={args.temperatures} K, "
        f"committee of {len(committee.models)}, thresholds {args.error_threshold}/{args.hard_error_threshold}, "
        f"fixed {len(fixed)} Zn, output={args.output}",
        flush=True,
    )

    walkers = []
    try:
        for walker in range(args.walkers):
            start = pool[walker % len(pool)]
            temperature = args.temperatures[walker % len(args.temperatures)]
            print(f"walker {walker}: T={temperature} K, start frame {walker % len(pool)}", flush=True)
            walkers.append(run_walker(walker, start, temperature, rng, args, calc, committee, mobile, fixed, writer))
    finally:
        writer.close()
        summary = {"candidates": writer.counts, "hard_events": writer.event_id, "walkers": walkers}
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\ncandidates: {writer.counts}  hard events: {writer.event_id}  -> {writer.path}", flush=True)


if __name__ == "__main__":
    main()
