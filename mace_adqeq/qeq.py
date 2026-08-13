from __future__ import annotations
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLUGINS"] = ""  # 禁用所有插件
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 让 JAX 看imp不到 GPU
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
import jaxopt
from dscribe.descriptors import ACSF
import freud
from flax import linen as nn
from flax import serialization
from dmff.admp.pme import energy_pme
from dmff.admp.recip import Ck_1, generate_pme_recip
from dmff.utils import pair_buffer_scales, regularize_pairs
from jax.scipy.special import erfc

from .const_potential import determine_chi


def build_acsf_descriptor(atoms, species, r_cut, g2_params, g4_params, n_jobs) -> np.ndarray:
    acsf = ACSF(
        species=list(species),
        r_cut=r_cut,
        g2_params=np.asarray(g2_params, dtype=float),
        g4_params=np.asarray(g4_params, dtype=float),
        periodic=bool(np.any(atoms.get_pbc())),
        sparse=False,
        dtype="float64",
    )
    return np.asarray(acsf.create(atoms, centers=None, n_jobs=n_jobs),dtype=np.float32)


def one_hot_encode(descriptor,symbols,element_map) -> np.ndarray:
    one_hot = np.zeros((descriptor.shape[0], len(element_map)), dtype=np.float32)
    for atom_index, symbol in enumerate(symbols):
        one_hot[atom_index, element_map[symbol]] = 1.0
    return np.concatenate((descriptor.astype(np.float32), one_hot), axis=1)

class NeighborListFreud:
    def __init__(self, box, rcut, cov_map, padding=True, max_shape=0):
        self.fbox = freud.box.Box.from_matrix(box)
        self.rcut = rcut
        self.capacity_multiplier = None
        self.padding = padding
        self.cov_map = cov_map
        self.max_shape = max_shape

    def do_cov_map(self, pairs):
        nbond = self.cov_map[pairs[:, 0], pairs[:, 1]]
        pairs = jnp.concatenate([pairs, nbond[:, None]], axis=1)
        return pairs

    def allocate(self, coords, box=None):
        self.stored_positions = coords
        fbox = freud.box.Box.from_matrix(box) if box is not None else self.fbox
        query = freud.locality.AABBQuery(fbox, coords)
        result = query.query(coords, dict(r_max=self.rcut, exclude_ii=True))
        neighbor_list = result.toNeighborList()
        neighbor_list = np.vstack((neighbor_list[:, 0], neighbor_list[:, 1])).T
        neighbor_list = neighbor_list.astype(np.int32)
        neighbor_list = neighbor_list[neighbor_list[:, 0] < neighbor_list[:, 1]]

        if self.capacity_multiplier is None:
            if self.max_shape == 0:
                self.capacity_multiplier = int(neighbor_list.shape[0] * 1.3)
            else:
                self.capacity_multiplier = self.max_shape

        if not self.padding:
            self.pair_array = self.do_cov_map(neighbor_list)
            return self.pair_array

        if self.max_shape == 0:
            self.capacity_multiplier = max(
                self.capacity_multiplier, neighbor_list.shape[0]
            )
        else:
            self.capacity_multiplier = self.max_shape

        padding_width = self.capacity_multiplier - neighbor_list.shape[0]
        if padding_width == 0:
            self.pair_array = self.do_cov_map(neighbor_list)
        elif padding_width > 0:
            padding = np.full((padding_width, 2), coords.shape[0], dtype=np.int32)
            neighbor_list = np.vstack((neighbor_list, padding))
            self.pair_array = self.do_cov_map(neighbor_list)
        else:
            raise ValueError(
                f"Neighbor-list capacity {self.capacity_multiplier} is smaller "
                f"than the required pair count {neighbor_list.shape[0]}"
            )
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
        return self.stored_positions


def get_neighbor_list(box, rc, positions, natoms, padding=True, max_shape=0):
    neighbor_list = NeighborListFreud(box,rc,jnp.zeros((natoms, natoms), dtype=jnp.int32),padding=padding,max_shape=max_shape)
    neighbor_list.allocate(positions)
    pairs = neighbor_list.pairs
    pairs = pairs.at[:, :2].set(regularize_pairs(pairs[:, :2]))
    return pairs

#@jax.jit
def ds_pairs(positions, box, pairs):
    pos1 = positions[pairs[:, 0].astype(int)]
    pos2 = positions[pairs[:, 1].astype(int)]
    box_inv = jnp.linalg.inv(box)
    displacement = (pos1 - pos2).dot(box_inv)
    displacement -= jnp.floor(displacement + 0.5)
    displacement = displacement.dot(box)
    return jnp.linalg.norm(displacement, axis=1)


def generate_get_energy(kappa, K1, K2, K3, pme_order=6):
    pme_recip_fn = generate_pme_recip(
        Ck_fn=Ck_1,
        kappa=kappa / 10.0,
        gamma=False,
        pme_order=pme_order,
        K1=K1,
        K2=K2,
        K3=K3,
        lmax=0,
    )

    def get_energy_kernel(positions, box, pairs, charges, mscales):
        atom_charges = jnp.reshape(charges, (-1, 1))
        return energy_pme(
            positions * 10.0,
            box * 10.0,
            pairs,
            atom_charges,
            None,
            None,
            None,
            mscales,
            None,
            None,
            None,
            pme_recip_fn,
            kappa / 10.0,
            K1,
            K2,
            K3,
            0,
            False,
        )

    def get_energy(positions, box, pairs, charges, mscales):
        return get_energy_kernel(positions, box, pairs, charges, mscales)

    return get_energy

#@jax.jit
def generate_get_Energy_Qeq(
    kappa=4.3804348,
    K1=45,
    K2=123,
    K3=22,
    pme_order=6,
    dipole_axis=2,
    include_dipole_correction=True,
):
    pme = generate_get_energy(kappa, K1, K2, K3, pme_order=pme_order)

    def get_Energy_Qeq(charges, positions, box, pairs, eta, chi, hardness):
        def get_Energy_PME():
            return pme(
                positions / 10.0,
                box / 10.0,
                pairs,
                charges,
                mscales=jnp.ones(6, dtype=positions.dtype),
            )

        def get_Energy_Correction():
            distances = ds_pairs(positions, box, pairs)
            buffer_scales = pair_buffer_scales(pairs)
            pair_eta = jnp.sqrt(eta[pairs[:, 0]] ** 2 + eta[pairs[:, 1]] ** 2)
            correction_pair = (
                charges[pairs[:, 0]]
                * charges[pairs[:, 1]]
                * erfc(distances / (jnp.sqrt(2.0) * pair_eta))
                * 1389.35455846
                / distances
                * buffer_scales
            )
            correction_self = (
                charges**2
                * 1389.35455846
                / (2.0 * jnp.sqrt(jnp.pi) * eta)
            )
            return -jnp.sum(correction_pair) + jnp.sum(correction_self)

        def get_Energy_Onsite():
            onsite = (chi * charges + 0.5 * hardness * charges * charges) * 96.4869
            return jnp.sum(onsite)

        def get_dipole_correction():
            if not include_dipole_correction:
                return jnp.asarray(0.0, dtype=positions.dtype)
            volume = jnp.linalg.det(box)
            prefactor = 2.0 * jnp.pi / volume * 1389.35455846
            moment = jnp.sum(charges * positions[:, dipole_axis])
            return prefactor * moment**2

        return (
            get_Energy_PME()
            + get_Energy_Correction()
            + get_Energy_Onsite()
            + get_dipole_correction()
        ) / 96.4869 #eV

    return get_Energy_Qeq


def generate_solve_q_pg(energy_fn, tol=1.0e-3, maxiter=500):
    def projected_energy(charges, *energy_args):
        value, gradient = jax.value_and_grad(energy_fn)(charges, *energy_args)
        projected_gradient = gradient - jnp.mean(gradient)
        return value, projected_gradient

    solver = jaxopt.LBFGS(
        fun=projected_energy,
        value_and_grad=True,
        tol=tol,
        maxiter=maxiter,
    )

    def solve_q_pg(
        charges,
        positions,
        box,
        pairs,
        eta,
        chi,
        hardness,
        return_state=False,
    ):
        total_charge = jnp.sum(charges)
        result = solver.run(
            charges,
            positions,
            box,
            pairs,
            eta,
            chi,
            hardness,
        )
        optimized_charges = result.params
        optimized_charges += (
            total_charge - jnp.sum(optimized_charges)
        ) / optimized_charges.shape[0]
        if return_state:
            return optimized_charges, result.state
        return optimized_charges

    return solve_q_pg


@dataclass(frozen=True)
class QEqParameters:
    chi: Any
    hardness: Any
    eta: Any


@dataclass(frozen=True)
class QEqResult:
    energy: float
    forces: np.ndarray
    charges: np.ndarray
    chi: np.ndarray
    hardness: np.ndarray
    eta: np.ndarray


class QEqParameterPredictor:
    def __init__(self,params_path,config_path,*,n_jobs=1):
        config = json.loads(Path(config_path).expanduser().resolve().read_text(encoding="utf-8"))
        self.n_jobs = int(n_jobs)
        #feature detail
        feature_config = config["features"]
        self.species = tuple(feature_config["species"])
        self.element_map = {
            str(key): int(value)
            for key, value in feature_config["element_map"].items()
        }
        self.cutoff = float(feature_config["r_cut"])
        self.g2_params = np.asarray(feature_config["g2_params"], dtype=float)
        self.g4_params = np.asarray(feature_config["g4_params"], dtype=float)
        self.append_element_one_hot = bool(feature_config.get("append_element_one_hot", True))

        parameterization = config["parameterization"]
        self.chi_scale = float(parameterization.get("chi_scale", 1.0))
        self.hardness_log_range = float(parameterization["hardness_log_range"])
        self.eta_log_range = float(parameterization["eta_log_range"])
        self.center_chi = bool(parameterization.get("chi_centered_per_structure", True))

        baselines = config["element_baselines"]
        self.chi_baseline = self.float_mapping(baselines["chi"])
        self.hardness_baseline = self.float_mapping(baselines["hardness"])
        self.eta_baseline = self.float_mapping(baselines["eta"])

        #model detail
        model_config = config["model"]
        self.input_dim = int(model_config["input_dim"])
        self.model = self.build_model(
            hidden_dim=int(model_config["hidden_dim"]),
            num_hidden_layers=int(model_config["num_hidden_layers"]),
            use_layer_norm=bool(model_config["use_layer_norm"]),
        )
        dummy_features = jnp.zeros((1, self.input_dim), dtype=jnp.float32)
        params_template = self.model.init(jax.random.PRNGKey(0),dummy_features)["params"]
        self.params = serialization.from_bytes(
            params_template,
            Path(params_path).expanduser().resolve().read_bytes(),
        )

    @staticmethod
    def float_mapping(values: Mapping[str, Any]) -> dict[str, float]:
        return {str(key): float(value) for key, value in values.items()}

    def build_model(self,hidden_dim,num_hidden_layers,use_layer_norm):
        class QEqMultiHeadMLP(nn.Module):
            hidden_dim: int
            num_hidden_layers: int
            use_layer_norm: bool

            @nn.compact
            def __call__(self, x):
                h = x
                for layer_idx in range(self.num_hidden_layers):
                    h = nn.Dense(
                        self.hidden_dim, name=f"trunk_dense_{layer_idx}"
                    )(h)
                    if self.use_layer_norm:
                        h = nn.LayerNorm(name=f"trunk_norm_{layer_idx}")(h)
                    h = nn.silu(h)

                zero_init = nn.initializers.zeros
                delta_chi = nn.Dense(
                    1,
                    kernel_init=zero_init,
                    bias_init=zero_init,
                    name="chi_head",
                )(h)
                delta_hardness = nn.Dense(
                    1,
                    kernel_init=zero_init,
                    bias_init=zero_init,
                    name="hardness_head",
                )(h)
                delta_eta = nn.Dense(
                    1,
                    kernel_init=zero_init,
                    bias_init=zero_init,
                    name="eta_head",
                )(h)
                return {
                    "delta_chi_raw": jnp.squeeze(delta_chi, axis=-1),
                    "delta_hardness_raw": jnp.squeeze(delta_hardness, axis=-1),
                    "delta_eta_raw": jnp.squeeze(delta_eta, axis=-1),
                }

        return QEqMultiHeadMLP(
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            use_layer_norm=use_layer_norm,
        )

    def validate_symbols(self, symbols: Sequence[str]) -> None:
        unsupported = sorted(set(symbols).difference(self.species))
        if unsupported:
            raise ValueError(
                f"QEq checkpoint does not support elements {unsupported}; "
                f"trained species are {list(self.species)}"
            )

    def build_features(self, atoms) -> np.ndarray:
        symbols = atoms.get_chemical_symbols()
        self.validate_symbols(symbols)
        descriptor = build_acsf_descriptor(
            atoms,
            self.species,
            self.cutoff,
            self.g2_params,
            self.g4_params,
            self.n_jobs,
        )
        if self.append_element_one_hot:
            descriptor = one_hot_encode(descriptor, symbols, self.element_map)
        if descriptor.shape != (len(symbols), self.input_dim):
            raise ValueError(
                "QEq feature shape mismatch: expected "
                f"{(len(symbols), self.input_dim)}, "
                f"got {descriptor.shape}"
            )
        return descriptor

    #core step
    def predict(self, atoms) -> QEqParameters:
        symbols = atoms.get_chemical_symbols()
        features = jnp.asarray(self.build_features(atoms), dtype=jnp.float32)
        raw = self.model.apply({"params": self.params}, features) #forward inference

        chi_base = jnp.asarray([self.chi_baseline[symbol] for symbol in symbols], dtype=jnp.float32)
        hardness_base = jnp.asarray([self.hardness_baseline[symbol] for symbol in symbols], dtype=jnp.float32)
        eta_base = jnp.asarray([self.eta_baseline[symbol] for symbol in symbols], dtype=jnp.float32)

        delta_chi = raw["delta_chi_raw"]
        delta_chi -= jnp.mean(delta_chi)
        delta_log_hardness = self.hardness_log_range * jnp.tanh(raw["delta_hardness_raw"])
        delta_log_eta = self.eta_log_range * jnp.tanh(raw["delta_eta_raw"])

        return QEqParameters(
            chi=chi_base + delta_chi,
            hardness=hardness_base * jnp.exp(delta_log_hardness),
            eta=eta_base * jnp.exp(delta_log_eta),
        )


class JAXQEqModel:

    def __init__(
        self,
        *,
        cutoff: float = 6.0,
        pme_grid: Sequence[int] = (45, 123, 22),
        kappa: float = 4.3804348,
        pme_order: int = 6,
        dipole_axis: int = 2,
        include_dipole_correction: bool = True,
        jitter: float = 1.0e-8,
        dtype: str = "float32",
        max_pairs: int = 0,
        solver_mode: str = "hybrid",
        pg_tolerance: float = 1.0e-3,
        pg_max_iterations: int = 200,
        matrix_tolerance: float = 1.0e-2,
        matrix_max_iterations: int = 10,
        const_potential: bool = True,
    ):
        self.cutoff = float(cutoff)
        self.max_pairs = int(max_pairs)
        self.solver_mode = str(solver_mode).lower()
        self.pg_tolerance = float(pg_tolerance)
        self.pg_max_iterations = int(pg_max_iterations)
        self.matrix_tolerance = float(matrix_tolerance)
        self.matrix_max_iterations = int(matrix_max_iterations)
        self.jitter = float(jitter)
        self.const_potential = bool(const_potential)
        self.np_dtype = np.dtype(dtype)
        self.jax_dtype = jnp.float32

        if self.cutoff <= 0.0:
            raise ValueError("cutoff must be positive")
        if self.max_pairs < 0:
            raise ValueError("max_pairs must be non-negative")
        if self.solver_mode not in {"matrix", "hybrid"}:
            raise ValueError("solver_mode must be either 'matrix' or 'hybrid'")
        if self.pg_tolerance <= 0.0 or self.matrix_tolerance <= 0.0:
            raise ValueError("solver tolerances must be positive")
        if self.pg_max_iterations <= 0 or self.matrix_max_iterations <= 0:
            raise ValueError("solver iteration limits must be positive")

        grid = tuple(int(value) for value in pme_grid)
        if len(grid) != 3:
            raise ValueError("pme_grid must contain exactly three integers")
        self.energy_fn = generate_get_Energy_Qeq(
            kappa=float(kappa),
            K1=grid[0],
            K2=grid[1],
            K3=grid[2],
            pme_order=int(pme_order),
            dipole_axis=int(dipole_axis),
            include_dipole_correction=bool(include_dipole_correction),
        )
        self.solve_q_pg = generate_solve_q_pg(
            self.energy_fn,
            tol=self.pg_tolerance,
            maxiter=self.pg_max_iterations,
        )
        self.charge_list: list[float] = []
        self.last_charge_solver: str | None = None
        self.last_pg_iterations: int | None = None
        self.last_pg_error: float | None = None
        self.charge_symbols: tuple[str, ...] | None = None
        self.charge_total: float | None = None

    def neighbor_pairs(self, positions: np.ndarray, box: np.ndarray):
        return get_neighbor_list(
            box,
            self.cutoff,
            positions,
            len(positions),
            padding=self.max_pairs > 0,
            max_shape=self.max_pairs,
        )

    def reset_charge_state(self):
        self.charge_list.clear()
        self.last_charge_solver = None
        self.last_pg_iterations = None
        self.last_pg_error = None
        self.charge_symbols = None
        self.charge_total = None

    def has_compatible_charge_state(
        self,
        symbols: Sequence[str],
        total_charge: float,
    ):
        return (
            len(self.charge_list) == len(symbols)
            and self.charge_symbols == tuple(symbols)
            and self.charge_total is not None
            and abs(self.charge_total - total_charge) <= 1.0e-8
        )

    def solve_charges_matrix(
        self,
        total_charge,
        positions,
        box,
        pairs,
        eta,
        chi,
        hardness,
    ):
        n_atoms = positions.shape[0]
        charges = jnp.full(n_atoms, total_charge / n_atoms, dtype=positions.dtype)

        def energy_for_q(q):
            return self.energy_fn(q, positions, box, pairs, eta, chi, hardness)

        ones = jnp.ones((n_atoms, 1), dtype=positions.dtype)
        for iteration_index in range(self.matrix_max_iterations):
            gradient = jax.grad(energy_for_q)(charges)
            projected_gradient = gradient - jnp.mean(gradient)
            residual = float(np.asarray(jnp.max(jnp.abs(projected_gradient))))
            if residual <= self.matrix_tolerance:
                break

            qeq_matrix = jax.hessian(energy_for_q)(charges)
            qeq_matrix = 0.5 * (qeq_matrix + qeq_matrix.T)
            qeq_matrix += self.jitter * jnp.eye(n_atoms, dtype=positions.dtype)
            kkt_matrix = jnp.block(
                [
                    [qeq_matrix, ones],
                    [
                        ones.T,
                        jnp.zeros((1, 1), dtype=positions.dtype),
                    ],
                ]
            )
            charge_error = total_charge - jnp.sum(charges)
            rhs = jnp.concatenate((-gradient, jnp.asarray([charge_error])))
            update = jnp.linalg.solve(kkt_matrix, rhs)[:n_atoms]
            charges += update

        charges += (total_charge - jnp.sum(charges)) / n_atoms
        return charges

    def solve_charges(
        self,
        total_charge,
        symbols,
        positions,
        box,
        pairs,
        eta,
        chi,
        hardness,
    ):
        use_projected_gradient = (
            self.solver_mode == "hybrid"
            and self.has_compatible_charge_state(symbols, total_charge)
        )
        if not use_projected_gradient:
            charges = self.solve_charges_matrix(
                total_charge,
                positions,
                box,
                pairs,
                eta,
                chi,
                hardness,
            )
            return charges, "matrix", None, None

        initial_charges = jnp.asarray(self.charge_list, dtype=positions.dtype)
        charges, state = self.solve_q_pg(
            initial_charges,
            positions,
            box,
            pairs,
            eta,
            chi,
            hardness,
            return_state=True,
        )
        iterations = int(np.asarray(state.iter_num))
        error = float(np.asarray(state.error))
        acceptable_error = max(10.0 * self.pg_tolerance, 1.0e-5)
        if not np.isfinite(error) or error > acceptable_error:
            charges = self.solve_charges_matrix(
                total_charge, positions, box, pairs, eta, chi, hardness
            )
            return charges, "matrix_fallback", iterations, error
        return charges, "projected_gradient", iterations, error

    def update_charge_state(
        self,
        charges: np.ndarray,
        symbols: Sequence[str],
        total_charge: float,
        solver: str,
        pg_iterations: int | None,
        pg_error: float | None,
    ):
        self.charge_list[:] = np.asarray(charges, dtype=float).tolist()
        self.charge_symbols = tuple(symbols)
        self.charge_total = float(total_charge)
        self.last_charge_solver = solver
        self.last_pg_iterations = pg_iterations
        self.last_pg_error = pg_error

    def calculate(self,atoms,chi,hardness,eta,total_charge=0.0):
        symbols = atoms.get_chemical_symbols()
        positions_np = np.asarray(atoms.get_positions(), dtype=self.np_dtype)
        box_np = np.asarray(atoms.get_cell(), dtype=self.np_dtype)
        positions = jnp.asarray(positions_np, dtype=self.jax_dtype)
        box = jnp.asarray(box_np, dtype=self.jax_dtype)
        pairs = self.neighbor_pairs(positions_np, box_np)
        chi = self.parameter_array("chi", chi, len(atoms))
        if self.const_potential:
            chi = determine_chi(box_np,positions_np,symbols,np.asarray(chi))[0]
            chi = jnp.asarray(chi, dtype=self.jax_dtype)
        hardness = self.parameter_array("hardness", hardness, len(atoms))
        eta = self.parameter_array("eta", eta, len(atoms))
        charges, solver, pg_iterations, pg_error = self.solve_charges(
            float(total_charge),
            symbols,
            positions,
            box,
            pairs,
            eta,
            chi,
            hardness,
        )

        energy, gradient = jax.value_and_grad(self.energy_fn, argnums=1)(
            charges, positions, box, pairs, eta, chi, hardness
        )
        forces = -gradient

        result = QEqResult(
            energy=float(np.asarray(energy)),
            forces=np.asarray(forces, dtype=float),
            charges=np.asarray(charges, dtype=float),
            chi=np.asarray(chi, dtype=float),
            hardness=np.asarray(hardness, dtype=float),
            eta=np.asarray(eta, dtype=float),
        )

        self.update_charge_state(
            result.charges,
            symbols,
            float(total_charge),
            solver,
            pg_iterations,
            pg_error,
        )
        return result

    def parameter_array(self, name: str, values, n_atoms: int):
        array = jnp.asarray(values, dtype=self.jax_dtype)
        if array.shape != (n_atoms,):
            raise ValueError(
                f"{name} shape is {array.shape}, expected {(n_atoms,)}"
            )
        return array
