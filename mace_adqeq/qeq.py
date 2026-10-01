from __future__ import annotations
import os
os.environ["JAX_PLATFORM_NAME"] = "gpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # 禁用所有插件
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Sequence
import warnings
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit
import jaxopt
import torch
from ase.data import chemical_symbols
from e3nn import o3
from mace.calculators.mace import MACECalculator
from mace.modules.utils import extract_invariant
import freud
from flax import linen as nn
from flax import serialization
from dmff.admp.pme import energy_pme
from dmff.admp.recip import Ck_1, generate_pme_recip
from dmff.utils import pair_buffer_scales, regularize_pairs
from jax.scipy.linalg import lu_factor, lu_solve
from jax.scipy.special import erfc

if __package__:
    from .const_potential import determine_chi
else:
    try:
        from const_potential import determine_chi
    except ImportError:
        determine_chi = None


@dataclass(frozen=True)
class MACEEmbeddingContext:
    features: Any
    positions: Any


class MACEEmbedding:
    def __init__(self,model_path,*,device="cpu",default_dtype="",num_layers=-1):
        self.calculator = MACECalculator(
            model_paths=str(Path(model_path).expanduser().resolve()),
            device=device,
            default_dtype=default_dtype,
            model_type="MACE",
        )

        self.model = self.calculator.models[0]
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False) 

        self.num_interactions = int(self.model.num_interactions)
        self.num_layers = (
            self.num_interactions if int(num_layers) == -1 else int(num_layers)
        )
        if not 1 <= self.num_layers <= self.num_interactions:
            raise ValueError(
                "num_layers must be -1 or between 1 and "
                f"{self.num_interactions}, got {num_layers}"
            )

        irreps_out = o3.Irreps(str(self.model.products[0].linear.irreps_out))
        self.l_max = irreps_out.lmax
        self.num_invariant_features = irreps_out.dim // (self.l_max + 1) ** 2
        self.output_dim = self.num_layers * self.num_invariant_features
        #one-hot for element
        self.species = tuple(
            chemical_symbols[int(atomic_number)]
            for atomic_number in np.asarray(
                self.model.atomic_numbers.detach().cpu(), dtype=int
            )
        )

    def _batch_for_atoms(self, atoms):
        batch = self.calculator._atoms_to_batch(atoms)
        model_dtype = next(self.model.parameters()).dtype
        for key in batch.keys:
            value = batch[key]
            if torch.is_tensor(value) and torch.is_floating_point(value):
                batch[key] = value.to(dtype=model_dtype)
        return batch

    def _invariant_features(self, atoms):
        batch = self._batch_for_atoms(atoms)
        batch_dict = batch.to_dict()
        output = self.model(batch_dict, compute_force=False)
        features = extract_invariant(
            output["node_feats"],
            num_layers=self.num_layers,
            num_features=self.num_invariant_features,
            l_max=self.l_max,
        )
        features = features[: len(atoms)]
        if features.shape != (len(atoms), self.output_dim):
            raise ValueError(
                "MACE feature shape mismatch: expected "
                f"{(len(atoms), self.output_dim)}, got {tuple(features.shape)}"
            )
        return features, batch_dict["positions"]
    #only for parameters
    #Solving for energy labels or charges does not require saving a large Torch autograd graph; 
    #only complete forces retain the graph.
    def features(self, atoms) -> np.ndarray:
        with torch.no_grad():
            features, _ = self._invariant_features(atoms)
        return np.asarray(features.detach().cpu(), dtype=np.float32)
    #for force calculation
    def features_with_context(self, atoms):
        features, positions = self._invariant_features(atoms)
        if not features.requires_grad or not positions.requires_grad:
            raise RuntimeError(
                "MACE features are not connected to atomic positions for autograd"
            )
        descriptor = np.asarray(features.detach().cpu(), dtype=np.float32)
        return descriptor, MACEEmbeddingContext(features=features, positions=positions)

    @staticmethod
    def position_vjp(context: MACEEmbeddingContext, grad_features) -> np.ndarray:
        cotangent = torch.as_tensor(
            np.asarray(grad_features),
            dtype=context.features.dtype,
            device=context.features.device,
        )
        grad_positions = torch.autograd.grad(
            outputs=context.features,
            inputs=context.positions,
            grad_outputs=cotangent,
            retain_graph=False,
            create_graph=False,
        )[0]
        return np.asarray(grad_positions.detach().cpu(), dtype=np.float32)


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

@jax.jit
def ds_pairs(positions, box, pairs):
    pos1 = positions[pairs[:, 0].astype(int)]
    pos2 = positions[pairs[:, 1].astype(int)]
    box_inv = jnp.linalg.inv(box)
    displacement = (pos1 - pos2).dot(box_inv)
    displacement -= jnp.floor(displacement + 0.5)
    displacement = displacement.dot(box)
    return jnp.linalg.norm(displacement, axis=1)


def generate_get_energy(kappa, K1, K2, K3):
    pme_recip_fn = generate_pme_recip(
        Ck_fn=Ck_1,
        kappa=kappa / 10.0,
        gamma=False,
        pme_order=6,
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


def generate_get_Energy_Qeq(
    kappa=4.3804348,
    K1=45,
    K2=22,
    K3=123,
    dipole_axis=2,
):
    pme = generate_get_energy(kappa, K1, K2, K3)

    @jax.jit
    def get_Energy_Qeq(charges, positions, box, pairs, eta, chi, hardness):

        @jax.jit
        def get_Energy_PME():

            return pme(
                positions / 10.0,
                box / 10.0,
                pairs,
                charges,
                mscales=jnp.ones(6, dtype=positions.dtype),
            )

        @jax.jit
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
                charges**2 * 1389.35455846 / (2.0 * jnp.sqrt(jnp.pi) * eta)
            )
            return -jnp.sum(correction_pair) + jnp.sum(correction_self)

        @jax.jit
        def get_Energy_Onsite():
            onsite = (chi * charges + 0.5 * hardness * charges * charges) * 96.4869
            return jnp.sum(onsite)

        @jax.jit
        def get_dipole_correction():
            volume = jnp.linalg.det(box)
            prefactor = 2.0 * jnp.pi / volume * 1389.35455846
            moment = jnp.sum(charges * positions[:, dipole_axis])
            return prefactor * moment**2

        return (
            get_Energy_PME()
            + get_Energy_Correction()
            + get_Energy_Onsite()
            + get_dipole_correction()
        ) / 96.4869  # eV

    return get_Energy_Qeq


def constrained_energy(energy_fn):
    """E evaluated on charges shifted back onto sum(q) = total_charge.

    Its exact gradient is the projected gradient of E, so energy values and gradients stay
    consistent even when float32 rounding lets sum(q) drift during an iterative solve; an
    unprojected value with a projected gradient breaks line searches, and a drifted sum
    shifted back afterwards lands on the stiff dipole-correction mode.
    """
    def energy(charges, total_charge, *energy_args):
        shifted = charges + (total_charge - jnp.sum(charges)) / charges.shape[0]
        return energy_fn(shifted, *energy_args)

    return energy


def generate_solve_q_pg(energy_fn, tol=1.0e-3, maxiter=500):
    solver = jaxopt.LBFGS(
        fun=jax.value_and_grad(constrained_energy(energy_fn), argnums=0),
        value_and_grad=True,
        tol=tol,
        maxiter=maxiter,
        implicit_diff=False,
    )

    def solve_q_pg(
        charges, positions, box, pairs, eta, chi, hardness, return_state=False
    ):
        total_charge = jnp.sum(charges)
        result = solver.run(charges, total_charge, positions, box, pairs, eta, chi, hardness)
        optimized_charges = result.params
        optimized_charges += (
            total_charge - jnp.sum(optimized_charges)
        ) / optimized_charges.shape[0]
        if return_state:
            return optimized_charges, result.state
        return optimized_charges

    return solve_q_pg


def charge_expansion_point(n_atoms, dtype):
    # Second derivatives must be taken away from q = 0: there every PME structure factor S(k)
    # vanishes and autodiff of |S(k)|^2 returns zero curvature for the reciprocal-space term.
    expansion = jax.random.normal(jax.random.PRNGKey(0), (n_atoms,), dtype=dtype)
    return 0.1 * (expansion - jnp.mean(expansion))


class CGState(NamedTuple):
    iter_num: Any
    error: Any


def generate_solve_q_cg(energy_fn, tol=1.0e-3, maxiter=200, patience=10):
    """Projected, Jacobi-preconditioned conjugate gradient for the quadratic QEq energy.

    Only gradients and Hessian-vector products are used: the step length along each
    direction is exact, so no energy values enter a line search. Every search direction
    sums to zero and the energy is evaluated through constrained_energy, which keeps
    sum(q) fixed. ``error`` is the L2 norm of the projected gradient, the same quantity
    jaxopt.LBFGS reports. The loop also stops after ``patience`` iterations without
    improvement, i.e. at the floating-point noise floor.
    """
    grad_q = jax.grad(constrained_energy(energy_fn), argnums=0)
    self_coulomb = 1389.35455846 / 96.4869

    @jax.jit
    def solve(charges, positions, box, pairs, eta, chi, hardness):
        args = (jnp.sum(charges), positions, box, pairs, eta, chi, hardness)
        expansion = charge_expansion_point(charges.shape[0], charges.dtype)

        def residual_of(q):
            return -grad_q(q, *args)

        def hvp(direction):
            return jax.jvp(lambda q: grad_q(q, *args), (expansion,), (direction,))[1]

        # Onsite hardness plus the Gaussian self-interaction of get_Energy_Correction.
        inverse_diagonal = 1.0 / (hardness + self_coulomb / (jnp.sqrt(jnp.pi) * eta))
        inverse_diagonal_sum = jnp.sum(inverse_diagonal)

        def precondition(r):
            z = inverse_diagonal * r
            return z - inverse_diagonal * jnp.sum(z) / inverse_diagonal_sum

        def error_of(r):
            return jnp.linalg.norm(r - jnp.mean(r))

        zero = jnp.asarray(0, dtype=jnp.int32)
        r = residual_of(charges)
        z = precondition(r)
        error = error_of(r)
        init = (zero, charges, r, z, z, error, error, zero)

        def cond(carry):
            k, _, _, _, _, error, _, stalled = carry
            return (k < maxiter) & (error > tol) & (stalled < patience)

        def body(carry):
            k, q, r, z, d, _, best, stalled = carry
            curvature = jnp.dot(d, hvp(d))
            rz = jnp.dot(r, z)
            alpha = jnp.where(curvature > 0.0, rz / jnp.where(curvature > 0.0, curvature, 1.0), 0.0)
            q = q + alpha * d
            r_new = residual_of(q)
            z_new = precondition(r_new)
            beta = jnp.maximum(0.0, jnp.dot(z_new, r_new - r) / jnp.where(rz > 0.0, rz, 1.0))
            d = z_new + beta * d
            d = d - jnp.mean(d)
            error = error_of(r_new)
            stalled = jnp.where(error < best, zero, stalled + 1)
            return (k + 1, q, r_new, z_new, d, error, jnp.minimum(best, error), stalled)

        k, q, _, _, _, error, _, _ = jax.lax.while_loop(cond, body, init)
        return q, CGState(iter_num=k, error=error)

    def solve_q_cg(
        charges, positions, box, pairs, eta, chi, hardness, return_state=False
    ):
        total_charge = jnp.sum(charges)
        optimized_charges, state = solve(charges, positions, box, pairs, eta, chi, hardness)
        optimized_charges += (
            total_charge - jnp.sum(optimized_charges)
        ) / optimized_charges.shape[0]
        if return_state:
            return optimized_charges, state
        return optimized_charges

    return solve_q_cg


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


class QEqMultiHeadMLP(nn.Module):
    hidden_dim: int = 256
    num_hidden_layers: int = 4
    use_layer_norm: bool = True

    @nn.compact
    def __call__(self, x):
        h = x
        for layer_idx in range(self.num_hidden_layers):
            h = nn.Dense(self.hidden_dim, name=f"trunk_dense_{layer_idx}")(h)
            if self.use_layer_norm:
                h = nn.LayerNorm(name=f"trunk_norm_{layer_idx}")(h)
            h = nn.silu(h)

        zero_init = nn.initializers.zeros
        delta_chi = nn.Dense(1,kernel_init=zero_init,bias_init=zero_init,name="chi_head")(h)
        delta_hardness = nn.Dense(1,kernel_init=zero_init,bias_init=zero_init,name="hardness_head")(h)
        delta_eta = nn.Dense(1,kernel_init=zero_init,bias_init=zero_init,name="eta_head")(h)
        return {
            "delta_chi_raw": jnp.squeeze(delta_chi, axis=-1),
            "delta_hardness_raw": jnp.squeeze(delta_hardness, axis=-1),
            "delta_eta_raw": jnp.squeeze(delta_eta, axis=-1),
        }


def predict_qeq_parameters(
    model,
    params,
    features,
    chi_base,
    hardness_base,
    eta_base,
    *,
    chi_scale=1.0,
    hardness_log_range=np.log(2.0),
    eta_log_range=np.log(1.5),
    center_chi=True,
):
    #forward
    raw = model.apply({"params": params}, features)
    delta_chi = chi_scale * raw["delta_chi_raw"]
    if center_chi:
        delta_chi = delta_chi - jnp.mean(delta_chi)
    delta_log_hardness = hardness_log_range * jnp.tanh(raw["delta_hardness_raw"])
    delta_log_eta = eta_log_range * jnp.tanh(raw["delta_eta_raw"])
    return {
        "chi": chi_base + delta_chi,
        "hardness": hardness_base * jnp.exp(delta_log_hardness),
        "eta": eta_base * jnp.exp(delta_log_eta),
        #"delta_chi": delta_chi,
        #"delta_log_hardness": delta_log_hardness,
        #"delta_log_eta": delta_log_eta,
    }


class QEqParameterPredictor:
    def __init__(
        self,
        params_path,
        config_path,
        *,
        mace_model_path=None,
        mace_device="cpu",
        mace_default_dtype="",
        mace_num_layers=None,
    ):
        config_path = Path(config_path).expanduser().resolve()
        config = json.loads(config_path.read_text(encoding="utf-8"))

        # feature detail
        feature_config = config["features"]
        descriptor_name = str(feature_config.get("descriptor", "MACE")).upper()
        if descriptor_name != "MACE":
            raise ValueError(
                f"Only MACE features are supported, got {descriptor_name!r}"
            )

        if mace_model_path is None:
            mace_model_path = feature_config.get("model_path")
        if mace_model_path is None:
            raise ValueError(
                "A MACE model is required. Pass mace_model_path or set "
                "features.model_path in the config."
            )
        mace_model_path = Path(mace_model_path).expanduser()
        if not mace_model_path.is_absolute():
            mace_model_path = config_path.parent / mace_model_path

        if mace_num_layers is None:
            mace_num_layers = int(feature_config.get("num_layers", -1))
        self.embedding = MACEEmbedding(
            mace_model_path,
            device=mace_device,
            default_dtype=mace_default_dtype,
            num_layers=mace_num_layers,
        )
        self.species = tuple(feature_config.get("species", self.embedding.species))
        unsupported_by_mace = sorted(set(self.species).difference(self.embedding.species))
        if unsupported_by_mace:
            raise ValueError(
                f"Configured species {unsupported_by_mace} are not supported by "
                f"the MACE model, which supports {list(self.embedding.species)}"
            )

        parameterization = config["parameterization"]
        self.chi_scale = float(parameterization.get("chi_scale", 1.0))
        self.hardness_log_range = float(parameterization["hardness_log_range"])
        self.eta_log_range = float(parameterization["eta_log_range"])
        self.center_chi = bool(parameterization.get("chi_centered_per_structure", True))

        baselines = config["element_baselines"]
        self.chi_baseline = self.float_mapping(baselines["chi"])
        self.hardness_baseline = self.float_mapping(baselines["hardness"])
        self.eta_baseline = self.float_mapping(baselines["eta"])

        # model detail
        model_config = config["model"]
        self.input_dim = int(model_config.get("input_dim", self.embedding.output_dim))
        if self.input_dim != self.embedding.output_dim:
            raise ValueError(
                "Flax input_dim does not match the selected MACE invariant "
                f"features: config has {self.input_dim}, MACE produces "
                f"{self.embedding.output_dim}"
            )
        self.model = self.build_model(
            hidden_dim=int(model_config["hidden_dim"]),
            num_hidden_layers=int(model_config["num_hidden_layers"]),
            use_layer_norm=bool(model_config["use_layer_norm"]),
        )
        dummy_features = jnp.zeros((1, self.input_dim), dtype=jnp.float32)
        params_template = self.model.init(jax.random.PRNGKey(0), dummy_features)[
            "params"
        ]
        self.params = serialization.from_bytes(
            params_template,
            Path(params_path).expanduser().resolve().read_bytes(),
        )

    @staticmethod
    def float_mapping(values: Mapping[str, Any]) -> dict[str, float]:
        return {str(key): float(value) for key, value in values.items()}

    @staticmethod
    def build_model(hidden_dim, num_hidden_layers, use_layer_norm):
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
        descriptor = self.embedding.features(atoms)
        if descriptor.shape != (len(symbols), self.input_dim):
            raise ValueError(
                "QEq feature shape mismatch: expected "
                f"{(len(symbols), self.input_dim)}, "
                f"got {descriptor.shape}"
            )
        return descriptor

    def parameters_from_features(self, features, symbols):
        dtype = features.dtype
        chi_base = jnp.asarray([self.chi_baseline[symbol] for symbol in symbols], dtype=dtype)
        hardness_base = jnp.asarray([self.hardness_baseline[symbol] for symbol in symbols], dtype=dtype)
        eta_base = jnp.asarray([self.eta_baseline[symbol] for symbol in symbols], dtype=dtype)

        predicted = predict_qeq_parameters(
            self.model,
            self.params,
            features,
            chi_base,
            hardness_base,
            eta_base,
            chi_scale=self.chi_scale,
            hardness_log_range=self.hardness_log_range,
            eta_log_range=self.eta_log_range,
            center_chi=self.center_chi,
        )
        return (
            predicted["chi"],
            predicted["hardness"],
            predicted["eta"],
        )

    def predict_with_derivatives(self, atoms):
        symbols = atoms.get_chemical_symbols()
        self.validate_symbols(symbols)
        descriptor, embedding_context = self.embedding.features_with_context(atoms)
        features = jnp.asarray(descriptor, dtype=jnp.float32)
        parameter_fn = lambda values: self.parameters_from_features(values, symbols)
        values, pullback = jax.vjp(parameter_fn, features)
        chi, hardness, eta = values
        return (
            QEqParameters(chi=chi, hardness=hardness, eta=eta),
            embedding_context,
            pullback,
        )

    def position_vjp(self, embedding_context, grad_features) -> np.ndarray:
        return self.embedding.position_vjp(embedding_context, grad_features)

    # core step
    def predict(self, atoms) -> QEqParameters:
        symbols = atoms.get_chemical_symbols()
        features = jnp.asarray(self.build_features(atoms), dtype=jnp.float32)
        chi, hardness, eta = self.parameters_from_features(features, symbols)
        return QEqParameters(chi=chi, hardness=hardness, eta=eta)


class JAXQEqModel:

    def __init__(
        self,
        *,
        cutoff: float = 6.0,
        pme_grid: Sequence[int] = (36, 40, 90),
        kappa: float = 4.3804348,
        dipole_axis: int = 2,
        jitter: float = 1.0e-8,
        dtype: str = "float64",
        max_pairs: int = 0,
        solver_mode: str = "hybrid",
        pg_method: str = "cg",
        pg_tolerance: float = 1.0e-3,
        pg_max_iterations: int = 200,
        matrix_tolerance: float = 1.0e-2,
        matrix_max_iterations: int = 10,
        const_potential: bool = True,
    ):
        self.cutoff = float(cutoff)
        self.max_pairs = int(max_pairs)
        self.solver_mode = str(solver_mode).lower()
        self.pg_method = str(pg_method).lower()
        self.pg_tolerance = float(pg_tolerance)
        self.pg_max_iterations = int(pg_max_iterations)
        self.matrix_tolerance = float(matrix_tolerance)
        self.matrix_max_iterations = int(matrix_max_iterations)
        self.jitter = float(jitter)
        self.const_potential = bool(const_potential)
        self.np_dtype = np.dtype(dtype)
        if self.np_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise ValueError(f"dtype must be 'float32' or 'float64', got {dtype!r}")
        self.jax_dtype = jnp.dtype(self.np_dtype)

        if self.cutoff <= 0.0:
            raise ValueError("cutoff must be positive")
        if self.max_pairs < 0:
            raise ValueError("max_pairs must be non-negative")
        if self.solver_mode not in {"matrix", "hybrid"}:
            raise ValueError("solver_mode must be either 'matrix' or 'hybrid'")
        if self.pg_method not in {"lbfgs", "cg"}:
            raise ValueError("pg_method must be either 'lbfgs' or 'cg'")
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
            dipole_axis=int(dipole_axis),
        )
        generate_solver = generate_solve_q_cg if self.pg_method == "cg" else generate_solve_q_pg
        self.solve_q_pg = generate_solver(
            self.energy_fn,
            tol=self.pg_tolerance,
            maxiter=self.pg_max_iterations,
        )
        self.charge_list: list[float] = []
        self.last_charge_solver: str | None = None
        self.last_matrix_residual: float | None = None
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
        self.last_matrix_residual = None
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

    def solve_charges_matrix(self,total_charge,positions,box,pairs,eta,chi,hardness):
        
        n_atoms = positions.shape[0]
        dtype = positions.dtype
        def energy_for_q(q):
            return self.energy_fn(q, positions, box, pairs, eta, chi, hardness)
        grad_for_q = jax.grad(energy_for_q)

        # E(q) is quadratic, so its Hessian is constant; see charge_expansion_point for why it is not taken at q = 0.
        qeq_matrix = jax.hessian(energy_for_q)(charge_expansion_point(n_atoms, dtype))

        qeq_matrix = 0.5 * (qeq_matrix + qeq_matrix.T)
        qeq_matrix += self.jitter * jnp.eye(n_atoms, dtype=dtype)

        ones = jnp.ones((n_atoms, 1), dtype=dtype)
        kkt_matrix = jnp.block(
            [
                [qeq_matrix, ones],
                [
                    ones.T,
                    jnp.zeros((1, 1), dtype=dtype),
                ],
            ]
        )
        kkt_factor = lu_factor(kkt_matrix)
        zero = jnp.zeros((1,), dtype=dtype)

        # Constrained Newton steps from the uniform charge distribution: one step is exact for a
        # quadratic E(q); later steps only run if the projected gradient is still above tolerance.
        charges = jnp.full(n_atoms, total_charge / n_atoms, dtype=dtype)
        gradient = grad_for_q(charges)
        residual = float(jnp.max(jnp.abs(gradient - jnp.mean(gradient))))
        for _ in range(self.matrix_max_iterations):
            if residual <= self.matrix_tolerance:
                break
            step = lu_solve(kkt_factor, jnp.concatenate((-gradient, zero)))[:n_atoms]
            charges = charges + step
            gradient = grad_for_q(charges)
            residual = float(jnp.max(jnp.abs(gradient - jnp.mean(gradient))))
        if residual > self.matrix_tolerance:
            warnings.warn(
                f"matrix charge solver did not reach tolerance: max projected gradient "
                f"{residual:.3e} > {self.matrix_tolerance:.3e} eV/e",
                RuntimeWarning,
            )
        self.last_matrix_residual = residual
        charges += (total_charge - jnp.sum(charges)) / n_atoms
        return charges

    def solve_charges(self,total_charge,symbols,positions,box,pairs,eta,chi,hardness):
        use_projected_gradient = (
            self.solver_mode == "hybrid"
            and self.has_compatible_charge_state(symbols, total_charge)
        )
        if not use_projected_gradient:
            charges = self.solve_charges_matrix(total_charge,positions,box,pairs,eta,chi,hardness)
            return charges, "matrix", None, None

        initial_charges = jnp.asarray(self.charge_list, dtype=positions.dtype)
        charges, state = self.solve_q_pg(initial_charges,positions,box,pairs,eta,chi,hardness,return_state=True)
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

    def calculate(
        self,
        atoms,
        chi=None,
        hardness=None,
        eta=None,
        total_charge=0.0,
        predictor=None,
        compute_forces=True,
    ):
        symbols = atoms.get_chemical_symbols()
        positions_np = np.asarray(atoms.get_positions(), dtype=self.np_dtype)
        box_np = np.asarray(atoms.get_cell(), dtype=self.np_dtype)
        positions = jnp.asarray(positions_np, dtype=self.jax_dtype)
        box = jnp.asarray(box_np, dtype=self.jax_dtype)
        pairs = self.neighbor_pairs(positions_np, box_np)
        parameter_pullback = None
        embedding_context = None
        if predictor is not None:
            if any(value is not None for value in (chi, hardness, eta)):
                raise ValueError("Pass either predictor or chi/hardness/eta, not both")
            if compute_forces:
                parameters, embedding_context, parameter_pullback = (
                    predictor.predict_with_derivatives(atoms)
                )
            else:
                parameters = predictor.predict(atoms)
            chi = parameters.chi
            hardness = parameters.hardness
            eta = parameters.eta
            parameter_dtype = parameters.chi.dtype
        elif any(value is None for value in (chi, hardness, eta)):
            raise ValueError("chi, hardness and eta are required without predictor")
        chi = self.parameter_array("chi", chi, len(atoms))
        if self.const_potential:
            if determine_chi is None:
                raise ImportError(
                    "const_potential.determine_chi is required when "
                    "const_potential=True"
                )
            chi = determine_chi(box_np, positions_np, symbols, np.asarray(chi))[0]
            chi = jnp.asarray(chi, dtype=self.jax_dtype)
        hardness = self.parameter_array("hardness", hardness, len(atoms))
        eta = self.parameter_array("eta", eta, len(atoms))
        charges, solver, pg_iterations, pg_error = self.solve_charges(float(total_charge),symbols,positions,box,pairs,eta,chi,hardness)

        if not compute_forces:
            #forzen_force
            energy = self.energy_fn(charges, positions, box, pairs, eta, chi, hardness)
            forces = jnp.zeros_like(positions)
        elif parameter_pullback is None:
            energy, grad_positions = jax.value_and_grad(self.energy_fn, argnums=1)(
                charges, positions, box, pairs, eta, chi, hardness
            )
            forces = -grad_positions
        else:
            #whole_force
            energy, gradients = jax.value_and_grad(
                self.energy_fn, argnums=(1, 4, 5, 6)
            )(charges, positions, box, pairs, eta, chi, hardness)
            grad_positions, grad_eta, grad_chi, grad_hardness = gradients
            # jax.vjp requires cotangents with the same dtype as the predictor outputs.
            (grad_features,) = parameter_pullback(tuple(gradient.astype(parameter_dtype) for gradient in (grad_chi, grad_hardness, grad_eta)))
            grad_parameter_response = jnp.asarray(
                predictor.position_vjp(embedding_context, grad_features),
                dtype=positions.dtype,
            )
            forces = -(grad_positions + grad_parameter_response)

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
            raise ValueError(f"{name} shape is {array.shape}, expected {(n_atoms,)}")
        return array
