"""Initial and simulated state storage."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import jax.numpy as jnp

from config import ExperimentConfig

if TYPE_CHECKING:
    from SupervisorClass import SupervisorClass
    from VariablesClass import VariablesClass


class StateClass:
    """Store bitwise states and the current mechanical dynamics."""

    def __init__(self, cfg: ExperimentConfig, Variabs: "VariablesClass") -> None:
        # inherit from Variabs
        # outer masses, displacement and velocity
        self.m1_state = jnp.vstack((jnp.full(Variabs.n_units, 1e-11), jnp.zeros(Variabs.n_units)))
        self.m2_state = jnp.zeros((2, Variabs.n_units)) # inner masses, displacement and velocity
        self.dyn_local: jnp.ndarray | None = None
        self.dyn_global: jnp.ndarray | None = None
        self.u: jnp.ndarray | None = None
        self.delta: jnp.ndarray | None = None
        self.state_in_t = np.zeros((cfg.Sprvsr.T, Variabs.n_physical_units), dtype=np.float32)
        self.state_identification_threshold = cfg.Output.state_identification_threshold

        # Store parameter arrays for unconstrained degrees of freedom. I could later change what's clamped and what's not
        self.get_free_dof_parameters(Variabs)

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

    def set_state(self, Variabs: "VariablesClass", input_state: str) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Set physical units to the requested binary state description."""
        # state-defined dynamics
        if len(input_state) != Variabs.n_physical_units or any(bit not in "01" for bit in input_state):
            raise ValueError("input_state must be a binary string with one bit per physical unit.")
        state = np.fromiter((int(bit) for bit in input_state), dtype=np.int8)
        self.m1_state = jnp.vstack((jnp.full(Variabs.n_units, 1e-11), jnp.zeros(Variabs.n_units)))
        physical_displacements = jnp.where(jnp.asarray(state, dtype=bool), Variabs.equilibrium2, Variabs.equilibrium1)
        m2_displacement = jnp.concatenate((jnp.zeros(1), physical_displacements, jnp.zeros(1)))
        self.m2_state = jnp.vstack((m2_displacement, jnp.zeros(Variabs.n_units)))

        # local and global dynamics
        if not hasattr(self, "m1_free_unit_ids"):
            self.get_free_dof_parameters(Variabs)

        global_dyn = jnp.stack((self.m1_state, self.m2_state), axis=-1).reshape(2, Variabs.n_dofs)
        free_dof_ids = jnp.sort(jnp.concatenate((2 * self.m1_free_unit_ids, 2 * self.m2_free_unit_ids + 1)))
        self.dyn_global = global_dyn
        self.dyn_local = global_dyn[:, free_dof_ids]

        # u and delta
        self.u = self.dyn_global[0, 0::2][1:-1]
        self.delta = self.dyn_global[0, 1::2][1:-1] - self.u
        self.state = state
        # return self.dyn_local, self.dyn_global

    def update_from_dyn(self, final_free_dyn: jnp.ndarray, final_global_dyn: jnp.ndarray) -> None:
        """Update the current dynamics and bitwise state from the end of a simulation."""
        self.dyn_local = final_free_dyn
        self.m1_state = jnp.vstack((final_global_dyn[0], final_global_dyn[1]))
        self.m2_state = jnp.vstack((final_global_dyn[2], final_global_dyn[3]))
        self.dyn_global = jnp.stack((self.m1_state, self.m2_state), axis=-1).reshape(2, -1)
        self.u = final_global_dyn[0, 1:-1]
        self.delta = final_global_dyn[2, 1:-1] - self.u
        self.state = self.get_system_state(self.delta, self.state_identification_threshold)

    def get_system_state(self, vn_minus_un: jnp.ndarray, threshold: float = 15e-3) -> np.ndarray:
        """Return a binary integer array from physical relative displacements."""
        relative_displacements = jnp.asarray(vn_minus_un).reshape(-1)
        return np.asarray(relative_displacements > threshold, dtype=np.int8)

    def measure_state(self, t: int) -> np.ndarray:
        """Measure and store the system state at training step ``t``."""
        self.state = self.get_system_state(self.delta, self.state_identification_threshold)
        self.state_in_t[t] = self.state
        return self.state_in_t[t]

    @staticmethod
    def _free_unit_ids(n_units: int, constrained_ids: jnp.ndarray) -> jnp.ndarray:
        """Return unit indices not present in ``constrained_ids``."""
        constrained = {int(index) % n_units for index in constrained_ids.tolist()}
        return jnp.asarray([index for index in range(n_units) if index not in constrained])
