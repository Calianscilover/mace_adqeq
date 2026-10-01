from __future__ import annotations

from typing import Sequence

import freud
import numpy as np


class NeighborListFreud_numpy:
    def __init__(self, box, rcut, cov_map, padding=True, max_shape=0):
        self.fbox = freud.box.Box.from_matrix(box)
        self.rcut = rcut
        self.capacity_multiplier = None
        self.padding = padding
        self.cov_map = cov_map
        self.max_shape = max_shape

    def do_cov_map(self, pairs):
        nbond = self.cov_map[pairs[:, 0], pairs[:, 1]]
        return np.concatenate([pairs, nbond[:, None]], axis=1)

    def allocate(self, coords, box=None):
        self.positions_array = coords
        fbox = freud.box.Box.from_matrix(box) if box is not None else self.fbox
        query = freud.locality.AABBQuery(fbox, coords)
        result = query.query(
            coords,
            {"r_max": self.rcut, "exclude_ii": True},
        )
        neighbor_list = result.toNeighborList()
        neighbor_list = np.vstack(
            (neighbor_list[:, 0], neighbor_list[:, 1])
        ).T.astype(np.int32)
        neighbor_list = neighbor_list[
            neighbor_list[:, 0] < neighbor_list[:, 1]
        ]

        if self.capacity_multiplier is None:
            if self.max_shape == 0:
                self.capacity_multiplier = int(neighbor_list.shape[0] * 1.5)
            else:
                self.capacity_multiplier = self.max_shape

        if not self.padding:
            self.pair_array = self.do_cov_map(neighbor_list)
            return self.pair_array

        if self.max_shape == 0:
            self.capacity_multiplier = max(
                self.capacity_multiplier,
                neighbor_list.shape[0],
            )
        else:
            self.capacity_multiplier = self.max_shape

        padding_width = self.capacity_multiplier - neighbor_list.shape[0]
        if padding_width < 0:
            raise ValueError(
                f"Neighbor-list capacity {self.capacity_multiplier} is smaller "
                f"than the required pair count {neighbor_list.shape[0]}"
            )
        if padding_width > 0:
            padding_pairs = np.full(
                (padding_width, 2),
                coords.shape[0],
                dtype=np.int32,
            )
            neighbor_list = np.vstack((neighbor_list, padding_pairs))

        self.pair_array = self.do_cov_map(neighbor_list)
        return self.pair_array

    def update(self, positions, box=None):
        return self.allocate(positions, box)

    @property
    def pairs(self):
        return self.pair_array

    @property
    def scaled_pairs(self):
        return self.pair_array

    @property
    def positions(self):
        return self.positions_array


def get_neighbor_list_numpy(box,rc,positions,natoms,padding=True,max_shape=0):
    neighbor_list = NeighborListFreud_numpy(
        box,
        rc,
        np.zeros((natoms, natoms), dtype=np.int32),
        padding=padding,
        max_shape=max_shape,
    )
    return neighbor_list.allocate(positions)


def identify_electrode_atoms(
    box: np.ndarray,
    positions: np.ndarray,
    symbols: Sequence[str],
    electrode_element: str = "Zn",
    coordination_cutoff: float = 3.5,
    minimum_coordination: int = 8,
    electrode_axis: int = 2,
    forced_bottom_indices: Sequence[int] = (),
    forced_upper_indices: Sequence[int] = (),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    box = np.asarray(box, dtype=float)
    positions = np.asarray(positions, dtype=float)
    symbols = tuple(symbols)
    atom_count = len(symbols)
    pairs = get_neighbor_list_numpy(box,coordination_cutoff,positions,atom_count,padding=False)
    coordination_numbers = np.zeros(atom_count, dtype=np.int32)
    for atom_i, atom_j, pair_scale in pairs:
        if (symbols[atom_i] == electrode_element and symbols[atom_j] == electrode_element):
            coordination_numbers[atom_i] += 1
            coordination_numbers[atom_j] += 1

    element_mask = np.asarray([symbol == electrode_element for symbol in symbols],dtype=bool)
    electrode_mask = element_mask & (coordination_numbers >= minimum_coordination)

    fractional_positions = positions @ np.linalg.inv(box)
    axis_coordinates = np.mod(fractional_positions[:, electrode_axis], 1.0)
    bottom_mask = electrode_mask & (axis_coordinates < 0.5)
    upper_mask = electrode_mask & (axis_coordinates >= 0.5)

    validate_forced_indices(
        forced_bottom_indices,
        atom_count,
        symbols,
        electrode_element,
        "bottom",
    )
    validate_forced_indices(
        forced_upper_indices,
        atom_count,
        symbols,
        electrode_element,
        "upper",
    )
    bottom_mask[np.asarray(forced_bottom_indices, dtype=np.int32)] = True
    upper_mask[np.asarray(forced_bottom_indices, dtype=np.int32)] = False
    upper_mask[np.asarray(forced_upper_indices, dtype=np.int32)] = True
    bottom_mask[np.asarray(forced_upper_indices, dtype=np.int32)] = False

    return (
        np.flatnonzero(bottom_mask),
        np.flatnonzero(upper_mask),
        coordination_numbers,
    )


def validate_forced_indices(
    indices: Sequence[int],
    atom_count: int,
    symbols: Sequence[str],
    electrode_element: str,
    electrode_name: str,
) -> None:
    for atom_index in indices:
        if atom_index < 0 or atom_index >= atom_count:
            raise IndexError(
                f"forced {electrode_name} index {atom_index} is outside "
                f"the valid range [0, {atom_count})"
            )
        if symbols[atom_index] != electrode_element:
            raise ValueError(
                f"forced {electrode_name} atom {atom_index} is "
                f"{symbols[atom_index]}, expected {electrode_element}"
            )


def apply_chi_bias(chi,bottom_indices,upper_indices,bottom_chi_bias: float = 10.0,upper_chi_bias: float = -10.0) -> np.ndarray:
    biased_chi = np.asarray(chi, dtype=float).copy()
    biased_chi[np.asarray(bottom_indices, dtype=np.int32)] += float(bottom_chi_bias)
    biased_chi[np.asarray(upper_indices, dtype=np.int32)] += float(upper_chi_bias)

    return biased_chi


def determine_chi(
    box: np.ndarray,
    positions: np.ndarray,
    symbols: Sequence[str],
    chi: np.ndarray,
    electrode_element: str = "Zn",
    coordination_cutoff: float = 3.5,
    minimum_coordination: int = 8,
    electrode_axis: int = 2,
    bottom_chi_bias: float = 10.0,
    upper_chi_bias: float = -2.0,
    forced_bottom_indices: Sequence[int] = (),
    forced_upper_indices: Sequence[int] = (),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    chi = np.asarray(chi, dtype=float)
    if chi.shape != (len(symbols),):
        raise ValueError(
            f"chi shape is {chi.shape}, expected {(len(symbols),)}"
        )

    bottom_indices, upper_indices, coordination_numbers = (
        identify_electrode_atoms(
            box,
            positions,
            symbols,
            electrode_element=electrode_element,
            coordination_cutoff=coordination_cutoff,
            minimum_coordination=minimum_coordination,
            electrode_axis=electrode_axis,
            forced_bottom_indices=forced_bottom_indices,
            forced_upper_indices=forced_upper_indices,
        )
    )
    biased_chi = apply_chi_bias(
        chi,
        bottom_indices,
        upper_indices,
        bottom_chi_bias=bottom_chi_bias,
        upper_chi_bias=upper_chi_bias,
    )
    return (
        biased_chi,
        bottom_indices,
        upper_indices,
        coordination_numbers,
    )
