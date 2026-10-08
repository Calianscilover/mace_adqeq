#!/usr/bin/env python3
"""Pick a diverse subset of committee-flagged candidates for DFT labelling.

Candidates come from al_collect.py and carry the committee's per-atom relative force std.
Each frame is described by MACE invariant descriptors (per atom, from the short-range MACE
model) pooled in two ways:

    local    mean and max over the --top-atoms atoms with the largest committee std,
             i.e. the environments the committee disagrees on
    global   mean over all atoms of each element, i.e. the state of the whole slab

Columns are standardised (the local block weighted by --local-weight) and frames are chosen
by farthest-point sampling, starting from the most uncertain frame: each further pick is the
candidate least like everything already chosen, so a dozen near-identical frames from one
unstable event cost one DFT calculation, not twelve.

    python al_select.py --candidates al/round1_*/candidates.extxyz --n 1000 --output al/round1_selected.extxyz

selection.csv lists the picks in order with their committee uncertainty and their distance
at the time of picking; where the distance levels off, further frames add little.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from ase.io import read, write
from ase.neighborlist import neighbor_list
from mace.calculators import MACECalculator

MACE_MODEL = Path(__file__).resolve().parent / "data" / "mace_short_stagetwo.model"
KEEP_INFO = (
    "source", "walker", "walker_step", "target_temperature_K", "temperature_K", "event_id", "steps_before_event",
    "committee_force_max_relative_std", "committee_force_max_std", "committee_max_atom", "committee_energy_std_per_atom",
    "model_energy", "model_max_force", "model_max_mace_force", "model_max_qeq_force",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", type=Path, nargs="+", required=True)
    parser.add_argument("--n", type=int, required=True, help="frames to select")
    parser.add_argument("--output", type=Path, required=True, help="selected frames (.extxyz)")
    parser.add_argument("--sources", nargs="+", default=["high_uncertainty", "hard_uncertainty", "failure"])
    parser.add_argument("--min-uncertainty", type=float, default=None, help="drop frames with a lower max relative std")
    parser.add_argument("--top-atoms", type=int, default=8, help="most uncertain atoms pooled into the local descriptor")
    parser.add_argument("--local-weight", type=float, default=2.0, help="weight of the local block in the distance")
    parser.add_argument("--mace-model", type=Path, default=MACE_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mace-dtype", default="float32")
    parser.add_argument("--min-distance", type=float, default=0.6, help="A; drop frames with a closer pair")
    parser.add_argument("--max-model-force", type=float, default=None, help="eV/A; drop frames above this")
    return parser.parse_args()


def frame_descriptor(calculator, atoms, elements, top_atoms):
    descriptors = calculator.get_descriptors(atoms, invariants_only=True)
    if isinstance(descriptors, list):
        descriptors = descriptors[0]
    uncertain = np.argsort(-atoms.arrays["committee_force_relative_std"])[:top_atoms]
    local = np.concatenate((descriptors[uncertain].mean(axis=0), descriptors[uncertain].max(axis=0)))
    symbols = np.asarray(atoms.get_chemical_symbols())
    pooled = [
        descriptors[symbols == element].mean(axis=0) if np.any(symbols == element) else np.zeros(descriptors.shape[1])
        for element in elements
    ]
    return local, np.concatenate(pooled)


def main():
    args = parse_args()
    candidates = [frame for path in args.candidates for frame in read(path, index=":")]
    print(f"{len(candidates)} candidates from {len(args.candidates)} file(s)", flush=True)

    keep = []
    for index, atoms in enumerate(candidates):
        info = atoms.info
        if info.get("source") not in args.sources or "committee_force_relative_std" not in atoms.arrays:
            continue
        if args.min_uncertainty is not None and info["committee_force_max_relative_std"] < args.min_uncertainty:
            continue
        if args.max_model_force is not None and info.get("model_max_force", 0.0) > args.max_model_force:
            continue
        if len(neighbor_list("d", atoms, args.min_distance)) == 0:
            keep.append(index)
    print(
        f"{len(keep)} pass the filters (sources {args.sources}, min uncertainty {args.min_uncertainty}, "
        f"min distance {args.min_distance} A, max model force {args.max_model_force})",
        flush=True,
    )
    if not keep:
        raise SystemExit("nothing to select")
    candidates = [candidates[index] for index in keep]
    uncertainty = np.array([atoms.info["committee_force_max_relative_std"] for atoms in candidates])
    elements = sorted({symbol for atoms in candidates for symbol in atoms.get_chemical_symbols()})

    calculator = MACECalculator(model_paths=str(args.mace_model), device=args.device, default_dtype=args.mace_dtype)
    local, pooled = [], []
    for index, atoms in enumerate(candidates):
        frame_local, frame_pooled = frame_descriptor(calculator, atoms, elements, args.top_atoms)
        local.append(frame_local)
        pooled.append(frame_pooled)
        if (index + 1) % 200 == 0:
            print(f"  descriptors {index + 1}/{len(candidates)}", flush=True)

    def standardise(block):
        block = np.asarray(block)
        scale = block.std(axis=0)
        scale[scale == 0.0] = 1.0
        return (block - block.mean(axis=0)) / scale / np.sqrt(block.shape[1])

    features = np.hstack((args.local_weight * standardise(local), standardise(pooled)))

    n = min(args.n, len(candidates))
    first = int(np.argmax(uncertainty))
    picks, picked_distance = [first], [float("inf")]
    distance = np.linalg.norm(features - features[first], axis=1)
    distance[first] = -1.0
    for _ in range(n - 1):
        index = int(np.argmax(distance))
        picks.append(index)
        picked_distance.append(float(distance[index]))
        distance = np.minimum(distance, np.linalg.norm(features - features[index], axis=1))
        distance[index] = -1.0

    selected = []
    for index in picks:
        source = candidates[index]
        frame = source.copy()
        frame.calc = None
        for name in list(frame.arrays):
            if name not in ("numbers", "positions"):
                del frame.arrays[name]
        frame.info = {key: source.info[key] for key in KEEP_INFO if key in source.info}
        frame.info["candidate_index"] = keep[index]
        selected.append(frame)
    write(args.output, selected, format="extxyz")

    with args.output.with_name("selection.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "candidate_index", "source", "event_id", "walker", "walker_step", "max_relative_std", "fps_distance"])
        for rank, (index, value) in enumerate(zip(picks, picked_distance)):
            info = candidates[index].info
            writer.writerow([rank, keep[index], info.get("source"), info.get("event_id", ""), info.get("walker"),
                             info.get("walker_step"), f"{uncertainty[index]:.4f}", f"{value:.4f}"])

    sources = [candidates[index].info.get("source") for index in picks]
    print(f"selected {n} of {len(candidates)}: " + ", ".join(f"{name} {sources.count(name)}" for name in sorted(set(sources))), flush=True)
    print(f"  max relative std of the picks: median {np.median(uncertainty[picks]):.3f}, min {np.min(uncertainty[picks]):.3f}", flush=True)
    for rank in sorted({1, n // 4, n // 2, 3 * n // 4, n - 1}):
        if 0 < rank < n:
            print(f"  fps distance at rank {rank:5d}: {picked_distance[rank]:.3f}", flush=True)
    print(f"wrote {args.output} and {args.output.with_name('selection.csv')}", flush=True)


if __name__ == "__main__":
    main()
