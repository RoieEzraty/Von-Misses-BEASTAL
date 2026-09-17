"""Time integration of the constrained bistable chain."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import grad, vmap
from jax.experimental.ode import odeint

if TYPE_CHECKING:
    from StateClass import StateClass
    from SupervisorClass import SupervisorClass
    from VariablesClass import VariablesClass


class EquilibriumClass:
    """Calculate the dynamic response for a configured chain."""

    def __init__(self, Variabs: "VariablesClass") -> None:
        # inherit from Variabs
        self.Variabs = Variabs
        self.force_fn = grad(self.potential_energy_constrained_ends, argnums=0)
        self.free_dyn: jnp.ndarray | None = None
        self.global_dyn: jnp.ndarray | None = None
        self.F_dyn: jnp.ndarray | None = None
        self.u_dyn: jnp.ndarray | None = None
        self.delta_dyn: jnp.ndarray | None = None

    def potential_energy_constrained_ends(self, free_dof_displacement: jnp.ndarray, time: float, 
                                          stiffness_vals: jnp.ndarray, timepoints: jnp.ndarray, 
                                          impulse_dyn: jnp.ndarray) -> jnp.ndarray:
        """Return total potential energy with driven-left and fixed-right ends."""
        Variabs = self.Variabs
        imposed_displacement = jnp.interp(time, timepoints, impulse_dyn)
        displacement1 = jnp.concatenate((jnp.atleast_1d(imposed_displacement), free_dof_displacement[0::2], jnp.zeros(1)))
        displacement2 = jnp.concatenate((jnp.atleast_1d(imposed_displacement), free_dof_displacement[1::2], jnp.zeros(1)))
        coupling_energy = 0.5 * jnp.sum(stiffness_vals[:-1, 0] * jnp.diff(displacement1) ** 2)
        local_energy = jnp.sum(Variabs.potential_fn(stiffness_vals[:, 1:].T, displacement2 - displacement1))
        return coupling_energy + local_energy

    def rhs(self, local_dyn: jnp.ndarray, time: float, m1s: jnp.ndarray, m2s: jnp.ndarray,
            c1s: jnp.ndarray, c2s: jnp.ndarray, mu_ks: jnp.ndarray, beta: float, 
            stiffness_vals: jnp.ndarray, timepoints: jnp.ndarray, 
            impulse_dyn: jnp.ndarray) -> jnp.ndarray:
        """Return displacement and velocity derivatives for free DOFs."""
        displacement, velocity = local_dyn
        mass_vals = jnp.stack((m1s, m2s), axis=1).reshape(-1)
        relative_velocity = velocity[1::2] - velocity[0::2]
        damping_un = (c1s * velocity[0::2] + mu_ks * 9.81 * m1s * jnp.tanh(velocity[0::2] * beta) 
                      - c2s * relative_velocity)
        damping_vn = c2s * relative_velocity
        damping = jnp.stack((damping_un, damping_vn), axis=1).reshape(-1)
        # force_fn differentiates potential energy, so mechanical force is -grad(U).
        force = -self.force_fn(displacement, time, stiffness_vals, timepoints, impulse_dyn)
        acceleration = (force - damping) / mass_vals
        return jnp.stack((velocity, acceleration))

    def solve(self, State: "StateClass", supervisor: "SupervisorClass") -> None:
        """Integrate the dynamics and store all simulation-time results."""
        if State.dyn_local is None:
            raise RuntimeError("Initialize the local state before calling solve().")
        if supervisor.impulse_dyn is None:
            raise RuntimeError("Program the impulse before calling solve().")
        self.free_dyn = odeint(self.rhs, State.dyn_local, supervisor.timepoints, State.m1s_free,
                               State.m2s_free, State.c1s_free, State.c2s_free, State.mu_ks_free,
                               self.Variabs.beta, self.Variabs.stiffness_vals, supervisor.timepoints,
                               supervisor.impulse_dyn, rtol=supervisor.rtol, atol=supervisor.atol,
                               mxstep=supervisor.mxstep)
        self.reshape_free_to_global_dyn(supervisor)
        State.update_from_dyn(self.free_dyn[-1], self.global_dyn[-1])

    def reshape_free_to_global_dyn(self, supervisor: "SupervisorClass") -> None:
        """Map ``free_dyn`` to the stored global, displacement, and force dynamics."""
        n_times = supervisor.timepoints.shape[0]
        driven_displacement_dyn = supervisor.impulse_fn(supervisor.timepoints)
        driven_velocity_dyn = vmap(supervisor.dimpulse_fn)(supervisor.timepoints)
        zeros = jnp.zeros((n_times, 1))

        # displacements
        m1_displacement_dyn = jnp.concatenate((driven_displacement_dyn[:, None], self.free_dyn[:, 0, 0::2], zeros), axis=1)
        m1_velocity_dyn = jnp.concatenate((driven_velocity_dyn[:, None], self.free_dyn[:, 1, 0::2], zeros), axis=1)
        m2_displacement_dyn = jnp.concatenate((driven_displacement_dyn[:, None], self.free_dyn[:, 0, 1::2], zeros), axis=1)
        m2_velocity_dyn = jnp.concatenate((driven_velocity_dyn[:, None], self.free_dyn[:, 1, 1::2], zeros), axis=1)
        self.global_dyn = jnp.stack((m1_displacement_dyn, m1_velocity_dyn, m2_displacement_dyn, m2_velocity_dyn), axis=1)
        self.u_dyn = self.global_dyn[:, 0, 1:-1]
        self.delta_dyn = self.global_dyn[:, 2, 1:-1] - self.u_dyn

        # forces
        self.F_dyn = self.Variabs.k1 * (self.global_dyn[:, 0, :-1] - self.global_dyn[:, 0, 1:])
