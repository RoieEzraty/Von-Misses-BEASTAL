"""Initial and simulated state storage."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import vmap

from config import ExperimentConfig

if TYPE_CHECKING:
    from SupervisorClass import SupervisorClass
    from VariablesClass import VariablesClass


class StateClass:
    """Construct local states and map solver results to global coordinates."""

    def __init__(self, Variabs: "VariablesClass") -> None:
        # inherit from Variabs
        self.m1_state0 = jnp.vstack((jnp.full(Variabs.n_units, 1e-11), jnp.zeros(Variabs.n_units)))  # outer masses, displacement and velocity
        self.m2_state0 = jnp.zeros((2, Variabs.n_units)) # inner masses, displacement and velocity
        self.state0_local: jnp.ndarray | None = None
        self.state0_global: jnp.ndarray | None = None
        self.global_solution: jnp.ndarray | None = None

    def get_free_dof_parameters(self, Variabs: "VariablesClass") -> None:
        """Store parameter arrays for unconstrained degrees of freedom.
        It is its own function just in case I want to change what's clamped and what's not"""
        self.m1_free_unit_ids = self._free_unit_ids(Variabs.n_units, Variabs.m1_constrained_unit_ids)
        self.m2_free_unit_ids = self._free_unit_ids(Variabs.n_units, Variabs.m2_constrained_unit_ids)
        if not jnp.array_equal(self.m1_free_unit_ids, self.m2_free_unit_ids):
            raise NotImplementedError("The two masses must currently share their constraints.")
        self.m1s_free = Variabs.m1_arr[self.m1_free_unit_ids]
        self.m2s_free = Variabs.m2_arr[self.m2_free_unit_ids]
        self.c1s_free = Variabs.c1_arr[self.m1_free_unit_ids]
        self.c2s_free = Variabs.c2_arr[self.m2_free_unit_ids]
        self.mu_ks_free = Variabs.mu_k_arr[self.m1_free_unit_ids]

    def get_local_global_states(self, Variabs: "VariablesClass") -> tuple[jnp.ndarray, jnp.ndarray]:
        """Build local and global initial states using the current mass states.
        global state indices as <u_left, v_left, u_0, v_0, u_1, v_1, .... u_n, v_n, u_right, v_right>
        (and their velocities a row under)
        local state indices as <u_0, v_0, u_1, v_1, .... u_n, v_n>
        (and their velocities a row under)"""
        if not hasattr(self, "m1_free_unit_ids"):
            self.get_free_dof_parameters(Variabs)

        global_state = jnp.stack((self.m1_state0, self.m2_state0), axis=-1).reshape(2, Variabs.n_dofs)
        free_dof_ids = jnp.sort(jnp.concatenate((2 * self.m1_free_unit_ids, 2 * self.m2_free_unit_ids + 1)))
        self.state0_global = global_state
        self.state0_local = global_state[:, free_dof_ids]
        return self.state0_local, self.state0_global

    def get_current_local_global_states(self, Variabs: "VariablesClass", phase_description: str) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Set physical units to the requested binary phase description."""
        if len(phase_description) != Variabs.n_physical_units or any(
            bit not in "01" for bit in phase_description
        ):
            raise ValueError(
                "phase_description must be a binary string with one bit per physical unit."
            )
        physical_displacements = jnp.where(
            jnp.asarray([bit == "1" for bit in phase_description]),
            Variabs.equilibrium2,
            Variabs.equilibrium1,
        )
        m2_displacement = jnp.concatenate(
            (jnp.zeros(1), physical_displacements, jnp.zeros(1))
        )
        self.m2_state0 = jnp.vstack((m2_displacement, jnp.zeros(Variabs.n_units)))
        return self.get_local_global_states(Variabs)

    def reshape_local_to_global(
        self,
        variables: "VariablesClass",
        supervisor: "SupervisorClass",
        free_dof_solution: jnp.ndarray,
    ) -> jnp.ndarray:
        """Map a free-DOF ODE solution into global mass coordinates."""
        n_times = supervisor.timepoints.shape[0]
        driven_disp = supervisor.impulse_fn(supervisor.timepoints)
        driven_vel = vmap(supervisor.dimpulse_fn)(supervisor.timepoints)
        zeros = jnp.zeros((n_times, 1))
        m1_displacement = jnp.concatenate(
            (driven_disp[:, None], free_dof_solution[:, 0, 0::2], zeros), axis=1
        )
        m1_velocity = jnp.concatenate(
            (driven_vel[:, None], free_dof_solution[:, 1, 0::2], zeros), axis=1
        )
        m2_displacement = jnp.concatenate(
            (driven_disp[:, None], free_dof_solution[:, 0, 1::2], zeros), axis=1
        )
        m2_velocity = jnp.concatenate(
            (driven_vel[:, None], free_dof_solution[:, 1, 1::2], zeros), axis=1
        )
        global_solution = jnp.stack(
            (m1_displacement, m1_velocity, m2_displacement, m2_velocity), axis=1
        )
        self.global_solution = global_solution
        return global_solution

    def get_system_state(
        self, vn_minus_un: jnp.ndarray, threshold: float = 15e-3
    ) -> str:
        """Return a binary phase string from physical relative displacements."""
        relative_displacements = jnp.asarray(vn_minus_un).reshape(-1)
        return "".join(
            "1" if float(value) > threshold else "0"
            for value in relative_displacements
        )

    @staticmethod
    def _free_unit_ids(n_units: int, constrained_ids: jnp.ndarray) -> jnp.ndarray:
        """Return unit indices not present in ``constrained_ids``."""
        constrained = {int(index) % n_units for index in constrained_ids.tolist()}
        return jnp.asarray([index for index in range(n_units) if index not in constrained])
