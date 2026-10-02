"""Physical parameters for the bistable mass-in-mass chain."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import jax.numpy as jnp

import helpers_builders
import plot_funcs
from config import ExperimentConfig


class VariablesClass:
    """Relevant physical parameters used by the simulation.

    equilibria are ``delta=0`` and ``delta=2*a0``.
    ``L`` and ``theta0`` affect both the saved equilibria and the independently packed
    potential parameters.

    "Direct" below means :class:`EquilibriumClass` reads the attribute during
    integration. "Indirect" means it is copied into ``StateClass`` or a packed
    array that integration reads. "Metadata" is retained for configuration,
    inspection, validation, or plotting and is not itself read by integration.
    """

    # Sizes and model selection (metadata, except sizes are also used indirectly by StateClass).
    n_physical_units: int  # Number of physical trusses, n.
    n_units: int  # n + 2, including the driven and fixed boundary units.
    n_dofs: int  # 2*n_units mechanical coordinates (outer and inner mass); indirect.
    driven_node: str  # translating state. Validated during construction; metadata afterwards.
    truss_model: str  # 'trusses' or '4th order'
    setup: str  # Selects how per-truss parameters are constructed.
                # 'experiment_full' - 3 trusses from Audrey's data
                # 'experiment_1st_mass' - identical trusses as the 1st mass from Audrey's, 
                # 'experiment_single' - Don't even remember
                # 'increasing_b', 'increasing_theta0', or 'increasing_m' - 
                # different value for each truss

    # Scalar material parameters copied from config; their arrays below are used in integration.
    m1: float  # mass of truss outer frame
    k1: float  # Direct: coupling stiffness and reported endpoint forces.
    c1: float  # damping of oscillation - outer mass
    c2: float  # damping of oscillation - inner mass
    mu_k: float  # kinetic friction
    beta: float  # Direct: tanh smoothing of kinetic friction in EquilibriumClass.rhs.

    # Truss geometry (defined only for truss_model="Trusses").
    L: jnp.ndarray  # Spring length. Metadata copy; used indirectly through bistable_potential_params and a0.
    theta0: jnp.ndarray  # Rest angle. Metadata copy; used indirectly through bistable_potential_params and a0.
    # ``b`` cancels out since potential reduces to
    # ``k_eff/2 * (L/cos(theta0) - sqrt((a0-x)**2 + L**2))**2`` -> 
    b: jnp.ndarray  # End offset. Metadata copy; packed into the potential but cancels algebraically at present.
    a0: jnp.ndarray  # horizontal rest deflection from truss center

    # Fourth-order coefficients (defined only for truss_model="4th Order").
    k2: float
    k3: float  # 3*sqrt(k2*k4/2).
    k4: float
    k_fit4: jnp.ndarray  # [k2, k3, k4]; used indirectly in the packed potential parameters.
    k_c: float  # Contact stiffness for the truss model.
    eps: float  # Clearance beyond the second equilibrium before contact begins.

    # Potential and equilibria shared by both model choices.
    potential_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]  # Direct: evaluated for local energy at every integration step.
    bistable_potential_params: jnp.ndarray  # Indirect: truss rows are [2*k_truss, L, theta0, b], plus [k_c, eps] when contact is enabled.
    equilibrium1: jnp.ndarray  # Indirect: StateClass uses it only to initialize state-0 displacements.
    equilibrium2: jnp.ndarray  # Indirect: StateClass uses it only to initialize state-1 displacements; equals 2*a0 for trusses.
    stiffness_vals: jnp.ndarray  # Direct: rows contain [k1, *bistable_potential_params].

    # Boundary indices and full arrays; StateClass extracts free entries before integration.
    m1_constrained_unit_ids: jnp.ndarray  # Indirect: outer-mass boundary indices.
    m2_constrained_unit_ids: jnp.ndarray  # Indirect: inner-mass boundary indices.
    constrained_unit_ids: tuple[jnp.ndarray, jnp.ndarray]  # Convenience metadata; not currently read elsewhere.
    m1_arr: jnp.ndarray  # Indirect: outer masses passed to EquilibriumClass.rhs through StateClass.
    m2_arr: jnp.ndarray  # Indirect: inner masses passed to EquilibriumClass.rhs through StateClass.
    c1_arr: jnp.ndarray  # Indirect: outer damping passed through StateClass.
    c2_arr: jnp.ndarray  # Indirect: internal damping passed through StateClass.
    mu_k_arr: jnp.ndarray  # Indirect: kinetic-friction coefficients passed through StateClass.

    def __init__(self, cfg: ExperimentConfig, plot_potential: bool = True) -> None:
        if cfg.Variabs.n_units < 1:
            raise ValueError("n_units must be positive.")
        if cfg.Variabs.driven_node != "1st":
            raise NotImplementedError("Only driven_node='1st' is currently supported.")

        # read for CFG
        self.n_physical_units = cfg.Variabs.n_units  # what we actually call number of trusses
        self.n_units = cfg.Variabs.n_units + 2  # we simulate two extra units - boundaries
        self.n_dofs = 2 * self.n_units  # outer- and inner-mass coordinates; position and velocity are separate state rows
        self.driven_node = cfg.Variabs.driven_node
        self.truss_model = cfg.Variabs.truss_model
        self.setup = cfg.Variabs.setup
        self.m1 = cfg.Variabs.m1
        self.k1 = cfg.Variabs.k1
        self.c1 = cfg.Variabs.c1
        self.c2 = cfg.Variabs.c2
        self.mu_k = cfg.Variabs.mu_k
        self.beta = cfg.Variabs.beta
        self.k_c = cfg.Variabs.k_c
        self.eps = cfg.Variabs.eps

        # non-uniform mass
        if self.setup == "increasing_m":
            first_g, last_g = cfg.Variabs.increasing_m_range
            physical_m2 = jnp.linspace(first_g, last_g, self.n_physical_units) * 1e-3
        else:
            physical_m2 = cfg.Variabs.m2 * jnp.ones(self.n_physical_units)

        # get model info dedicated function
        self._get_model_info(cfg)
        
        # concatenate into array
        self.stiffness_vals = jnp.concatenate([self.k1 * jnp.ones((self.n_units, 1)), self.bistable_potential_params], axis=1)
        self.m1_constrained_unit_ids = jnp.array([0, self.n_units - 1])
        self.m2_constrained_unit_ids = jnp.array([0, self.n_units - 1])
        self.constrained_unit_ids = (self.m1_constrained_unit_ids, self.m2_constrained_unit_ids)
        self.m1_arr = self.m1 * jnp.ones(self.n_units)
        self.m2_arr = jnp.concatenate((physical_m2[:1], physical_m2, physical_m2[-1:]))
        self.c1_arr = self.c1 * jnp.ones(self.n_units)
        self.c2_arr = self.c2 * jnp.ones(self.n_units)
        self.mu_k_arr = self.mu_k * jnp.ones(self.n_units)

        if plot_potential:
            plot_funcs.plot_potential(self.bistable_potential_params[1:-1], self.truss_model)

    def _load_experimental_geometry(self, cfg: ExperimentConfig) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Load ``L``, ``theta0``, and ``b`` from the experimental image data."""
        # allocated data directory inside this directory
        data_dir = Path(cfg.Variabs.experimental_data_dir)

        # load experimental geometry and pixels to metres conversion
        pixels_per_metre = helpers_builders.pixel_per_meter(jnp.load(data_dir / "px_mm_conversion.npy"), cfg.Variabs.reference_length)
        length_points = jnp.load(data_dir / "data_for_L_values.npy") / pixels_per_metre
        theta_points = jnp.load(data_dir / "data_for_theta_values.npy") / pixels_per_metre
        b_points = jnp.load(data_dir / "data_for_b_value.npy") / pixels_per_metre

        # Got this chunk from Audrey's code, not sure what to make of it
        length = jnp.abs(jnp.diff(length_points[:, 1])[::2]) / 2
        pairs = theta_points.reshape(-1, 2, 2)
        theta0 = jnp.arctan2(jnp.abs(pairs[:, 0, 0] - pairs[:, 1, 0]), jnp.abs(pairs[:, 0, 1] - pairs[:, 1, 1]))
        b_value = jnp.linalg.norm(b_points[0] - b_points[1])
        b = jnp.full_like(length, b_value)

        if length.ndim != 1 or theta0.ndim != 1 or b.ndim != 1 or not length.size or length.shape != theta0.shape or length.shape != b.shape:
            raise ValueError("Experimental geometry must contain matching, nonempty L, theta0, and b arrays.")
        return length, theta0, b

    def _get_model_info(self, cfg: ExperimentConfig) -> None:
        """Select the potential and construct its per-unit parameters."""
        if self.truss_model == "4th Order":
            self.potential_fn = helpers_builders.potential_order4
            self.k2 = cfg.Variabs.k2
            self.k4 = cfg.Variabs.k4
            self.k3 = 3 * jnp.sqrt(self.k2 * self.k4 / 2)
            self.k_fit4 = jnp.array([self.k2, self.k3, self.k4])
            equilibrium2 = self.k3 / (2 * self.k4) + jnp.sqrt((self.k3 / (2 * self.k4)) ** 2 - self.k2 / self.k4)
            self.equilibrium1 = jnp.zeros(self.n_physical_units)
            self.equilibrium2 = jnp.full((self.n_physical_units,), equilibrium2)
            physical_params = jnp.tile(self.k_fit4, (self.n_physical_units, 1))
            self.bistable_potential_params = jnp.vstack((jnp.zeros((1, 3)), physical_params, jnp.zeros((1, 3))))
            return

        if self.truss_model != "Trusses":
            raise ValueError(f"Unknown potential model: {self.truss_model}")

        self.potential_fn = helpers_builders.bistable_potential

        # load from experimental Audrey data
        length, theta0, b = self._load_experimental_geometry(cfg)

        # the setup decides how to use the parameters
        if self.setup == "experiment_full":
            expected_shape = (self.n_physical_units,)
            if length.shape != expected_shape:
                raise ValueError("The experiment_full setup requires one experimental truss per physical unit.")
        elif self.setup == "experiment_1st_mass":
            length = jnp.full((self.n_physical_units,), length[0])
            theta0 = jnp.full((self.n_physical_units,), theta0[0])
            b = jnp.full((self.n_physical_units,), b[0])
        elif self.setup in {"experiment_single", "increasing_b", "increasing_m", "increasing_theta0"}:
            length = jnp.full_like(length, length[0])
            theta0 = jnp.full_like(theta0, theta0[0])
            if self.setup == "increasing_b":
                b = jnp.asarray(cfg.Variabs.increasing_b_mm) * 1e-3
            else:
                b = jnp.full_like(b, b[0])
            if self.setup == "increasing_theta0":
                rises = jnp.asarray(cfg.Variabs.increasing_theta0_rise_mm)
                theta0 = jnp.arctan(rises / cfg.Variabs.increasing_theta0_span_mm)
        else:
            raise ValueError(f"Unknown setup: {self.setup}")

        # save into class instance
        self.L = length
        self.theta0 = theta0
        self.b = b

        # calculate equilibria
        physical_params = jnp.column_stack((2 * cfg.Variabs.k_truss * jnp.ones(self.n_physical_units), length, theta0, b))
        if self.k_c != 0:
            physical_params = jnp.column_stack((physical_params, self.k_c * jnp.ones(self.n_physical_units), self.eps * jnp.ones(self.n_physical_units)))
            left_boundary = physical_params[0].at[jnp.array([0, 4])].set(0)
            right_boundary = physical_params[-1].at[jnp.array([0, 4])].set(0)
        else:
            left_boundary = physical_params[0].at[0].set(0)
            right_boundary = physical_params[-1].at[0].set(0)
        self.bistable_potential_params = jnp.vstack((left_boundary, physical_params, right_boundary))  # all k_trusses the same, but length, theta0, vary experimentally
        rest_length = length / jnp.cos(theta0) - 2 * b
        a0 = (rest_length + 2 * b) * jnp.sin(theta0)
        self.equilibrium1 = jnp.zeros_like(a0)
        self.equilibrium2 = 2 * a0
