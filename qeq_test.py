from collections import Counter
from pathlib import Path

import numpy as np
from ase.io import read, write

from mace_adqeq import JAXQEqModel, QEqParameterPredictor, QEqResult


ROOT = Path(__file__).resolve().parent
STRUCTURE = ROOT / "stru" / "pair_revised.pdb"
PARAMS = ROOT / "data" / "qeq_multihead_params.msgpack"
CONFIG = ROOT / "data" / "qeq_multihead_config.json"
RESULTS = ROOT / "qeq_result.npz"
STRUCTURE_WITH_CHARGES = ROOT / "STRU-10-qeq.extxyz"


def correct_pdb_symbols(atoms):
    """Correct Zn atoms whose PDB atom name is Zn but element field is N."""
    symbols = atoms.get_chemical_symbols()
    atom_types = atoms.arrays.get("atomtypes")
    if atom_types is None:
        return []

    corrected_indices = []
    for atom_index, (symbol, atom_type) in enumerate(zip(symbols, atom_types)):
        if symbol == "N" and str(atom_type).upper().startswith("ZN"):
            symbols[atom_index] = "Zn"
            corrected_indices.append(atom_index)
    atoms.set_chemical_symbols(symbols)
    return corrected_indices


def print_input_diagnostics(atoms, predictor):
    symbols = atoms.get_chemical_symbols()
    features = predictor.build_features(atoms)

    print(f"chemical symbols: {dict(Counter(symbols))}")
    print(f"first 20 symbols: {symbols[:20]}")
    print(f"last 20 symbols:  {symbols[-20:]}")
    print(
        f"descriptor: shape={features.shape}, dtype={features.dtype}, "
        f"finite={np.all(np.isfinite(features))}, "
        f"range=[{features.min():.6g}, {features.max():.6g}]"
    )

    if predictor.append_element_one_hot:
        one_hot = features[:, -len(predictor.element_map):]
        expected = np.asarray([predictor.element_map[symbol] for symbol in symbols])
        one_hot_is_correct = bool(
            np.allclose(one_hot.sum(axis=1), 1.0)
            and np.array_equal(one_hot.argmax(axis=1), expected)
        )
        print(f"element one-hot correct: {one_hot_is_correct}")


def main():
    atoms = read(STRUCTURE)
    raw_symbols = atoms.get_chemical_symbols()
    print(f"raw PDB symbols: {dict(Counter(raw_symbols))}")
    corrected_indices = correct_pdb_symbols(atoms)
    if corrected_indices:
        print(
            "corrected N -> Zn from PDB atom names at 1-based indices: "
            f"{[index + 1 for index in corrected_indices]}"
        )

    predictor = QEqParameterPredictor(PARAMS, CONFIG)
    print_input_diagnostics(atoms, predictor)
    parameters = predictor.predict(atoms)

    qeq_model = JAXQEqModel(cutoff=predictor.cutoff)
    result = qeq_model.calculate(
        atoms,
        chi=parameters.chi,
        hardness=parameters.hardness,
        eta=parameters.eta,
        total_charge=0.0,
    )

    np.savez(
        RESULTS,
        energy=result.energy,
        forces=result.forces,
        charges=result.charges,
        chi=result.chi,
        hardness=result.hardness,
        eta=result.eta,
    )

    atoms.arrays["qeq_charges"] = result.charges
    atoms.arrays["qeq_chi"] = result.chi
    atoms.arrays["qeq_hardness"] = result.hardness
    atoms.arrays["qeq_eta"] = result.eta
    atoms.arrays["qeq_forces"] = result.forces
    atoms.info["qeq_energy_eV"] = result.energy
    write(STRUCTURE_WITH_CHARGES, atoms, format="extxyz")

    print(f"QEqResult: energy={result.energy:.8f} eV")
    print(f"atoms={len(atoms)}, total_charge={result.charges.sum():.8e} e")
    print(f"numerical results: {RESULTS}")
    print(f"structure for visualization: {STRUCTURE_WITH_CHARGES}")
    return result


if __name__ == "__main__":
    qeq_result = main()
