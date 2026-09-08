"""Physical parameters for the bistable mass-in-mass chain."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp

import helpers_builders
import plot_funcs
from config import ExperimentConfig


class VariablesClass:
    """Build all per-unit physical arrays from :mod:`config`."""

    def __init__(self, cfg: ExperimentConfig, plot_potential: bool = True) -> None:
        values = cfg.Variabs
        if values.n_units < 1:
            raise ValueError("n_units must be positive.")
        if values.driven_node != "1st":
            raise NotImplementedError("Only driven_node='1st' is currently supported.")

        self.n_physical_units = values.n_units
        self.n_units = values.n_units + 2
        self.n_dofs = 2 * self.n_units
        self.driven_node = values.driven_node
        self.truss_model = values.truss_model
        self.setup = values.setup
        self.m1 = values.m1
        self.k1 = values.k1
        self.c1 = values.c1
        self.c2 = values.c2
        self.mu_k = values.mu_k
        self.beta = values.beta

        self._get_model_info(cfg)
        self.stiffness_vals = jnp.concatenate(
            [self.k1 * jnp.ones((self.n_units, 1)), self.bistable_potential_params],
            axis=1,
        )
        self.m1_constrained_unit_ids = jnp.array([0, self.n_units - 1])
        self.m2_constrained_unit_ids = jnp.array([0, self.n_units - 1])
        self.constrained_unit_ids = (
            self.m1_constrained_unit_ids,
            self.m2_constrained_unit_ids,
        )

        physical_m2 = self._physical_inner_masses(cfg)
        self.m1_arr = self.m1 * jnp.ones(self.n_units)
        self.m2_arr = jnp.concatenate((physical_m2[:1], physical_m2, physical_m2[-1:]))
        self.c1_arr = self.c1 * jnp.ones(self.n_units)
        self.c2_arr = self.c2 * jnp.ones(self.n_units)
        self.mu_k_arr = self.mu_k * jnp.ones(self.n_units)

        if plot_potential:
            plot_funcs.plot_potential(self)

    def _physical_inner_masses(self, cfg: ExperimentConfig) -> jnp.ndarray:
        """Return one inner mass per physical unit."""
        values = cfg.Variabs
        if self.setup == "increasing_m":
            first_g, last_g = values.increasing_m_range_g
            return jnp.linspace(first_g, last_g, self.n_physical_units) * 1e-3
        return values.m2 * jnp.ones(self.n_physical_units)

    def _load_experimental_geometry(
        self, cfg: ExperimentConfig
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Load ``L``, ``theta0``, and ``b`` from the experimental image data."""
        values = cfg.Variabs
        data_dir = Path(values.experimental_data_dir)
        pixels_per_metre = helpers_builders.pixel_per_meter(
            jnp.load(data_dir / "px_mm_conversion.npy"), values.reference_length
        )
        length_points = jnp.load(data_dir / "data_for_L_values.npy") / pixels_per_metre
        theta_points = jnp.load(data_dir / "data_for_theta_values.npy") / pixels_per_metre
        b_points = jnp.load(data_dir / "data_for_b_value.npy") / pixels_per_metre

        length = jnp.abs(jnp.diff(length_points[:, 1])[::2]) / 2
        pairs = theta_points.reshape(-1, 2, 2)
        theta0 = jnp.arctan2(
            jnp.abs(pairs[:, 0, 0] - pairs[:, 1, 0]),
            jnp.abs(pairs[:, 0, 1] - pairs[:, 1, 1]),
        )
        b_value = jnp.linalg.norm(b_points[0] - b_points[1])
        b = jnp.full_like(length, b_value)

        expected_shape = (self.n_physical_units,)
        if length.shape != expected_shape or theta0.shape != expected_shape:
            raise ValueError(
                "Experimental geometry must contain one L and theta0 value per physical unit."
            )
        return length, theta0, b

    @staticmethod
    def _configured_array(values: tuple[float, ...], count: int, name: str) -> jnp.ndarray:
        """Validate and convert a per-unit configuration tuple."""
        if len(values) != count:
            raise ValueError(f"{name} must contain one value per physical unit.")
        return jnp.asarray(values)

    def _get_model_info(self, cfg: ExperimentConfig) -> None:
        """Select the potential and construct its per-unit parameters."""
        values = cfg.Variabs
        if self.truss_model == "4th Order":
            self.potential_fn = helpers_builders.potential_order4
            self.k2 = values.k2
            self.k4 = values.k4
            self.k3 = 3 * jnp.sqrt(self.k2 * self.k4 / 2)
            self.k_fit4 = jnp.array([self.k2, self.k3, self.k4])
            equilibrium2 = self.k3 / (2 * self.k4) + jnp.sqrt(
                (self.k3 / (2 * self.k4)) ** 2 - self.k2 / self.k4
            )
            self.equilibrium1 = jnp.zeros(self.n_physical_units)
            self.equilibrium2 = jnp.full((self.n_physical_units,), equilibrium2)
            physical_params = jnp.tile(self.k_fit4, (self.n_physical_units, 1))
            self.bistable_potential_params = jnp.vstack(
                (jnp.zeros((1, 3)), physical_params, jnp.zeros((1, 3)))
            )
            return

        if self.truss_model != "Trusses":
            raise ValueError(f"Unknown potential model: {self.truss_model}")

        self.potential_fn = helpers_builders.bistable_potential
        length, theta0, b = self._load_experimental_geometry(cfg)
        if self.setup == "experiment_full":
            pass
        elif self.setup in {
            "experiment_single",
            "increasing_b",
            "increasing_m",
            "increasing_theta0",
        }:
            length = jnp.full_like(length, length[0])
            theta0 = jnp.full_like(theta0, theta0[0])
            b = jnp.full_like(b, b[0])
        else:
            raise ValueError(f"Unknown setup: {self.setup}")

        if self.setup == "increasing_b":
            b = self._configured_array(
                values.increasing_b_mm, self.n_physical_units, "increasing_b_mm"
            ) * 1e-3
        elif self.setup == "increasing_theta0":
            rises = self._configured_array(
                values.increasing_theta0_rise_mm,
                self.n_physical_units,
                "increasing_theta0_rise_mm",
            )
            theta0 = jnp.arctan(rises / values.increasing_theta0_span_mm)

        self.L = length
        self.theta0 = theta0
        self.b = b
        physical_params = jnp.column_stack(
            (2 * values.k_truss * jnp.ones(self.n_physical_units), length, theta0, b)
        )
        left_boundary = jnp.concatenate((jnp.zeros(1), physical_params[0, 1:]))
        right_boundary = jnp.concatenate((jnp.zeros(1), physical_params[-1, 1:]))
        self.bistable_potential_params = jnp.vstack(
            (left_boundary, physical_params, right_boundary)
        )

        rest_length = length / jnp.cos(theta0) - 2 * b
        a0 = (rest_length + 2 * b) * jnp.sin(theta0)
        self.equilibrium1 = jnp.zeros_like(a0)
        self.equilibrium2 = 2 * a0
