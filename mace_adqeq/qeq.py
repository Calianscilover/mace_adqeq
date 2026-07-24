from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import jax
import jax.numpy as jnp
import jaxopt
from jax import jit, grad, jacfwd, jacrev, value_and_grad
from dscribe.descriptors import ACSF
import freud
from flax import linen as nn
from flax import serialization
from dmff.admp.pme import energy_pme
from dmff.admp.recip import Ck_1, generate_pme_recip
from dmff.utils import pair_buffer_scales, regularize_pairs
from jax.scipy.special import erfc


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
    acsf_str = acsf.create(atoms)
    # print(acsf_str.shape)
    return np.asarray(acsf.create(atoms, centers=None, n_jobs=n_jobs), dtype=np.float32)


def one_hot_encode(
    descriptor: np.ndarray,
    symbols: Sequence[str],
    element_map: Mapping[str, int],
) -> np.ndarray:
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

    def _do_cov_map(self, pairs):
        nbond = self.cov_map[pairs[:, 0], pairs[:, 1]]
        pairs = jnp.concatenate([pairs, nbond[:, None]], axis=1)
        return pairs

    def allocate(self, coords, box=None):
        self._positions = coords
        fbox = freud.box.Box.from_matrix(box) if box is not None else self.fbox
        query = freud.locality.AABBQuery(fbox, coords)
        result = query.query(coords, {"r_max": self.rcut, "exclude_ii": True})
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
            self._pairs = self._do_cov_map(neighbor_list)
            return self._pairs

        if self.max_shape == 0:
            self.capacity_multiplier = max(
                self.capacity_multiplier, neighbor_list.shape[0]
            )
        else:
            self.capacity_multiplier = self.max_shape

        padding_width = self.capacity_multiplier - neighbor_list.shape[0]
        if padding_width == 0:
            self._pairs = self._do_cov_map(neighbor_list)
        elif padding_width > 0:
            padding = np.full((padding_width, 2), coords.shape[0], dtype=np.int32)
            neighbor_list = np.vstack((neighbor_list, padding))
            self._pairs = self._do_cov_map(neighbor_list)
        else:
            raise ValueError(
                f"Neighbor-list capacity {self.capacity_multiplier} is smaller "
                f"than the required pair count {neighbor_list.shape[0]}"
            )
        return self._pairs

    def update(self, positions, box=None):
        return self.allocate(positions, box)

    @property
    def pairs(self):
        return self._pairs

    @property
    def scaled_pairs(self):
        return self._pairs

    @property
    def positions(self):
        return self._positions


def get_neighbor_list(
    box, rc, positions, natoms, padding=True, max_shape=0
):
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
def generate_get_Energy_Qeq_2(
    kappa=4.3804348,
    K1=45,
    K2=123,
    K3=22,
    pme_order=6,
    dipole_axis=1,
    include_dipole_correction=True,
):
    pme = generate_get_energy(kappa, K1, K2, K3, pme_order=pme_order)

    def get_Energy_Qeq_2(charges, positions, box, pairs, eta, chi, hardness):
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
            correction_pair = (charges[pairs[:, 0]]* charges[pairs[:, 1]]* erfc(distances / (jnp.sqrt(2.0) * pair_eta))* 1389.35455846/ distances* buffer_scales)
            correction_self = (charges**2* 1389.35455846/ (2.0 * jnp.sqrt(jnp.pi) * eta))
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
        ) / 96.4869

    return get_Energy_Qeq_2


get_Energy_Qeq = generate_get_Energy_Qeq_2()


def fn_value_and_proj_grad(func, constraint_matrix, has_aux=False):
    def value_and_proj_grad(*args, **kwargs):
        value, gradient = jax.value_and_grad(func, has_aux=has_aux)(*args, **kwargs)
        #n * 1
        constraint_gradient = jnp.matmul(constraint_matrix, gradient.reshape(-1, 1))
        #n * 1
        constraint_norm = jnp.sum(constraint_matrix * constraint_matrix,axis=1,keepdims=True)
        #1*N
        removed_gradient = jnp.matmul((constraint_gradient / constraint_norm).T,constraint_matrix)
        #N
        projected_gradient = gradient - removed_gradient.reshape(-1)
        return value, projected_gradient

    return value_and_proj_grad


def generate_solve_q_pg(energy_fn, tol=1.0e-6, maxiter=500):
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

    def solve_q_pg(charges,positions,box,pairs,eta,chi,hardness,return_state=False):
        total_charge = jnp.sum(charges)
        result = solver.run(charges,positions,box,pairs,eta,chi,hardness,)
        optimized_charges = result.params
        optimized_charges += ( total_charge - jnp.sum(optimized_charges)) / optimized_charges.shape[0]
        if return_state:
            return optimized_charges, result.state
        return optimized_charges

    return solve_q_pg


# Original main_multi.py-compatible projected-gradient charge solver.
solve_q_pg = generate_solve_q_pg(get_Energy_Qeq)


@dataclass(frozen=True)
class QEqResult:
    energy: float
    forces: np.ndarray
    charges: np.ndarray
    chi: np.ndarray
    hardness: np.ndarray
    eta: np.ndarray

class JAXQEqModel:
    def __init__(self,params_path: str | Path, config_path: str | Path,*,
        pme_grid: Sequence[int] = (45, 123, 22),kappa: float = 4.3804348,pme_order: int = 6,
        dipole_axis: int = 1,include_dipole_correction: bool = True,
        jitter: float = 1.0e-8,n_jobs: int = 1,dtype: str = "float32",
        max_pairs: int = 0,solver_mode: str = "hybrid",
        pg_tolerance: float = 1.0e-6,pg_max_iterations: int = 500,
        matrix_tolerance: float = 1.0e-8,matrix_max_iterations: int = 8):

        self.jax = jax
        self.jnp = jnp
        self.nn = nn
        self.serialization = serialization

        self.params_path = Path(params_path).expanduser().resolve()
        self.config_path = Path(config_path).expanduser().resolve()
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))

        self.pme_grid = tuple(int(value) for value in pme_grid)
        self.kappa = float(kappa)
        self.pme_order = int(pme_order)
        self.dipole_axis = int(dipole_axis)

        self.include_dipole_correction = bool(include_dipole_correction)
        self.jitter = float(jitter)
        self.n_jobs = int(n_jobs)
        self.max_pairs = int(max_pairs)
        if self.max_pairs < 0:
            raise ValueError("max_pairs must be non-negative")
        self.solver_mode = str(solver_mode).lower()
        if self.solver_mode not in {"matrix", "hybrid"}:
            raise ValueError(
                "solver_mode must be either 'matrix' or 'hybrid'"
            )
        self.pg_tolerance = float(pg_tolerance)
        self.pg_max_iterations = int(pg_max_iterations)
        self.matrix_tolerance = float(matrix_tolerance)
        self.matrix_max_iterations = int(matrix_max_iterations)
        if self.pg_tolerance <= 0.0:
            raise ValueError("pg_tolerance must be positive")
        if self.pg_max_iterations <= 0:
            raise ValueError("pg_max_iterations must be positive")
        if self.matrix_tolerance <= 0.0:
            raise ValueError("matrix_tolerance must be positive")
        if self.matrix_max_iterations <= 0:
            raise ValueError("matrix_max_iterations must be positive")
        self.np_dtype = np.dtype(dtype)
        self.jax_dtype = jnp.float32

        feature_config = self.config["features"]
        if feature_config.get("descriptor") != "ACSF":
            raise ValueError(
                "This loader currently supports ACSF checkpoints only; got "
                f"{feature_config.get('descriptor')!r}"
            )
        self.species = tuple(feature_config["species"])
        self.element_map = {
            str(key): int(value) for key, value in feature_config["element_map"].items()
        }
        self.r_cut = float(feature_config["r_cut"])
        self.g2_params = np.asarray(feature_config["g2_params"], dtype=float)
        self.g4_params = np.asarray(feature_config["g4_params"], dtype=float)
        self.append_element_one_hot = bool(
            feature_config.get("append_element_one_hot", True)
        )

        parameterization = self.config["parameterization"]
        self.chi_scale = float(parameterization.get("chi_scale", 1.0))
        self.hardness_log_range = float(parameterization["hardness_log_range"])
        self.eta_log_range = float(parameterization["eta_log_range"])
        self.center_chi = bool(parameterization.get("chi_centered_per_structure", True))

        baselines = self.config["element_baselines"]
        self.chi_baseline = self._float_mapping(baselines["chi"])
        self.hardness_baseline = self._float_mapping(baselines["hardness"])
        self.eta_baseline = self._float_mapping(baselines["eta"])

        model_config = self.config["model"]
        self.input_dim = int(model_config["input_dim"])
        self.model = self._build_model(
            hidden_dim=int(model_config["hidden_dim"]),
            num_hidden_layers=int(model_config["num_hidden_layers"]),
            use_layer_norm=bool(model_config["use_layer_norm"]),
        )
        dummy_features = jnp.zeros((1, self.input_dim), dtype=self.jax_dtype)
        params_template = self.model.init(jax.random.PRNGKey(0), dummy_features)["params"]
        self.params = serialization.from_bytes(
            params_template, self.params_path.read_bytes()
        )

        self.get_Energy_Qeq_2 = generate_get_Energy_Qeq_2(
            kappa=self.kappa,
            K1=self.pme_grid[0],
            K2=self.pme_grid[1],
            K3=self.pme_grid[2],
            pme_order=self.pme_order,
            dipole_axis=self.dipole_axis,
            include_dipole_correction=self.include_dipole_correction,
        )
        self.solve_q_pg = generate_solve_q_pg(
            self.get_Energy_Qeq_2,
            tol=self.pg_tolerance,
            maxiter=self.pg_max_iterations,
        )
        self.charge_list: list[float] = []
        self.last_charge_solver: str | None = None
        self.last_pg_iterations: int | None = None
        self.last_pg_error: float | None = None
        self._charge_symbols: tuple[str, ...] | None = None
        self._charge_total: float | None = None

    @staticmethod
    def _float_mapping(values: Mapping[str, Any]) -> dict[str, float]:
        return {str(key): float(value) for key, value in values.items()}

    def _build_model(self, hidden_dim: int, num_hidden_layers: int, use_layer_norm: bool):
        nn = self.nn
        jnp = self.jnp

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

        return QEqMultiHeadMLP(hidden_dim=hidden_dim,num_hidden_layers=num_hidden_layers,use_layer_norm=use_layer_norm)

    def _validate_symbols(self, symbols: Sequence[str]) -> None:
        unsupported = sorted(set(symbols).difference(self.species))
        if unsupported:
            raise ValueError(
                f"QEq checkpoint does not support elements {unsupported}; "
                f"trained species are {list(self.species)}"
            )

    def build_features(self, atoms) -> np.ndarray:
        symbols = atoms.get_chemical_symbols()
        self._validate_symbols(symbols)
        descriptor = build_acsf_descriptor(
            atoms,
            self.species,
            self.r_cut,
            self.g2_params,
            self.g4_params,
            self.n_jobs,
        )
        if self.append_element_one_hot:
            descriptor = one_hot_encode(descriptor, symbols, self.element_map)
        if descriptor.shape != (len(symbols), self.input_dim):
            raise ValueError(
                f"QEq feature shape mismatch: expected {(len(symbols), self.input_dim)}, "
                f"got {descriptor.shape}"
            )
        return descriptor

    def predict_parameters(self, atoms) -> dict[str, np.ndarray]:
        symbols = atoms.get_chemical_symbols()
        features = self.jnp.asarray(self.build_features(atoms), dtype=self.jax_dtype)
        raw = self.model.apply({"params": self.params}, features)

        chi_base = self.jnp.asarray([self.chi_baseline[symbol] for symbol in symbols], dtype=self.jax_dtype)
        hardness_base = self.jnp.asarray([self.hardness_baseline[symbol] for symbol in symbols],dtype=self.jax_dtype)
        eta_base = self.jnp.asarray([self.eta_baseline[symbol] for symbol in symbols], dtype=self.jax_dtype)

        delta_chi = self.chi_scale * raw["delta_chi_raw"]
        if self.center_chi:
            delta_chi = delta_chi - self.jnp.mean(delta_chi)
        delta_log_hardness = self.hardness_log_range * self.jnp.tanh(
            raw["delta_hardness_raw"]
        )
        delta_log_eta = self.eta_log_range * self.jnp.tanh(raw["delta_eta_raw"])
        chi = chi_base + delta_chi
        hardness = hardness_base * self.jnp.exp(delta_log_hardness)
        eta = eta_base
        return {
            "chi": chi,
            "hardness": hardness,
            "eta": eta,
            "delta_chi": delta_chi,
            "delta_log_hardness": delta_log_hardness,
            "delta_log_eta": delta_log_eta,
        }

    def _neighbor_pairs(self, positions: np.ndarray, box: np.ndarray):
        return get_neighbor_list(box,self.r_cut,positions,len(positions),padding=self.max_pairs > 0,max_shape=self.max_pairs)

    def reset_charge_state(self) -> None:
        self.charge_list.clear()
        self.last_charge_solver = None
        self.last_pg_iterations = None
        self.last_pg_error = None
        self._charge_symbols = None
        self._charge_total = None

    def _has_compatible_charge_state(self, symbols: Sequence[str], total_charge: float) -> bool:
        return (
            len(self.charge_list) == len(symbols)
            and self._charge_symbols == tuple(symbols)
            and self._charge_total is not None
            and abs(self._charge_total - total_charge) <= 1.0e-8
        )

    def _solve_charges_matrix(self, total_charge, positions, box, pairs, eta, chi, hardness):
        n_atoms = positions.shape[0]
        charges = self.jnp.full(n_atoms,total_charge / n_atoms,dtype=positions.dtype)

        def energy_for_q(q):
            return self.get_Energy_Qeq_2(
                q, positions, box, pairs, eta, chi, hardness
            )

        ones = self.jnp.ones((n_atoms, 1), dtype=positions.dtype)
        for _ in range(self.matrix_max_iterations):
            gradient = self.jax.grad(energy_for_q)(charges)
            projected_gradient = gradient - self.jnp.mean(gradient)
            residual = float(
                np.asarray(self.jnp.max(self.jnp.abs(projected_gradient)))
            )
            if residual <= self.matrix_tolerance:
                break

            qeq_matrix = self.jax.hessian(energy_for_q)(charges)
            qeq_matrix = 0.5 * (qeq_matrix + qeq_matrix.T)
            qeq_matrix += self.jitter * self.jnp.eye(
                n_atoms, dtype=positions.dtype
            )
            kkt_matrix = self.jnp.block(
                [
                    [qeq_matrix, ones],
                    [
                        ones.T,
                        self.jnp.zeros((1, 1), dtype=positions.dtype),
                    ],
                ]
            )
            charge_error = total_charge - self.jnp.sum(charges)
            rhs = self.jnp.concatenate((-gradient, self.jnp.asarray([charge_error])))
            update = self.jnp.linalg.solve(kkt_matrix, rhs)[:n_atoms]
            charges += update

        charges += (total_charge - self.jnp.sum(charges)) / n_atoms
        return charges

    def _solve_charges(self,total_charge,symbols,positions,box,pairs,eta,chi,hardness):
        use_projected_gradient = (self.solver_mode == "hybrid"and self._has_compatible_charge_state(symbols, total_charge))
        if not use_projected_gradient:
            charges = self._solve_charges_matrix(total_charge, positions, box, pairs, eta, chi, hardness)
            return charges, "matrix", None, None

        initial_charges = self.jnp.asarray(
            self.charge_list, dtype=positions.dtype
        )
        charges, state = self.solve_q_pg(initial_charges,positions,box,pairs,eta,chi,hardness,return_state=True)
        iterations = int(np.asarray(state.iter_num))
        error = float(np.asarray(state.error))
        acceptable_error = max(10.0 * self.pg_tolerance, 1.0e-5)
        if not np.isfinite(error) or error > acceptable_error:
            charges = self._solve_charges_matrix(
                total_charge, positions, box, pairs, eta, chi, hardness
            )
            return charges, "matrix_fallback", iterations, error
        return charges, "projected_gradient", iterations, error

    def _update_charge_state(
        self,
        charges: np.ndarray,
        symbols: Sequence[str],
        total_charge: float,
        solver: str,
        pg_iterations: int | None,
        pg_error: float | None,
    ) -> None:
        self.charge_list[:] = np.asarray(charges, dtype=float).tolist()
        self._charge_symbols = tuple(symbols)
        self._charge_total = float(total_charge)
        self.last_charge_solver = solver
        self.last_pg_iterations = pg_iterations
        self.last_pg_error = pg_error

    def calculate(self, atoms, total_charge: float) -> QEqResult:
        symbols = atoms.get_chemical_symbols()
        parameters = self.predict_parameters(atoms)
        positions_np = np.asarray(atoms.get_positions(), dtype=self.np_dtype)
        box_np = np.asarray(atoms.get_cell(), dtype=self.np_dtype)
        positions = self.jnp.asarray(positions_np, dtype=self.jax_dtype)
        box = self.jnp.asarray(box_np, dtype=self.jax_dtype)
        pairs = self._neighbor_pairs(positions_np, box_np)
        chi = parameters["chi"]
        hardness = parameters["hardness"]
        eta = parameters["eta"]
        charges, solver, pg_iterations, pg_error = self._solve_charges(
            float(total_charge),
            symbols,
            positions,
            box,
            pairs,
            eta,
            chi,
            hardness,
        )

        energy, gradient = self.jax.value_and_grad(self.get_Energy_Qeq_2, argnums=1)(charges, positions, box, pairs, eta, chi, hardness)
        forces = -gradient

        result = QEqResult(
            energy=float(np.asarray(energy)),
            forces=np.asarray(forces, dtype=float),
            charges=np.asarray(charges, dtype=float),
            chi=np.asarray(chi, dtype=float),
            hardness=np.asarray(hardness, dtype=float),
            eta=np.asarray(eta, dtype=float),
        )
        self._validate_result(result, float(total_charge), len(atoms))
        self._update_charge_state(
            result.charges,
            symbols,
            float(total_charge),
            solver,
            pg_iterations,
            pg_error,
        )
        return result

    @staticmethod
    def _validate_result(result: QEqResult, total_charge: float, n_atoms: int) -> None:
        arrays = (result.forces, result.charges, result.chi, result.hardness, result.eta)
        if not np.isfinite(result.energy) or any(not np.all(np.isfinite(x)) for x in arrays):
            raise FloatingPointError("JAX QEq returned a non-finite result")
        if result.forces.shape != (n_atoms, 3):
            raise ValueError(f"QEq force shape is {result.forces.shape}, expected {(n_atoms, 3)}")
        if result.charges.shape != (n_atoms,):
            raise ValueError(f"QEq charge shape is {result.charges.shape}, expected {(n_atoms,)}")
        charge_error = float(result.charges.sum() - total_charge)
        if abs(charge_error) > 5.0e-4:
            raise ValueError(
                f"QEq total charge error is {charge_error:+.6e} e; "
                f"target={total_charge:+.6f} e"
            )
