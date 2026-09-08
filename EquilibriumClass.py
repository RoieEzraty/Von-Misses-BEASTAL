"""Time integration of the constrained bistable chain."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import grad
from jax.experimental.ode import odeint

if TYPE_CHECKING:
    from StateClass import StateClass
    from SupervisorClass import SupervisorClass
    from VariablesClass import VariablesClass


class EquilibriumClass:
    """Calculate the dynamic response for a configured chain."""

    def __init__(self, variables: "VariablesClass") -> None:
        self.variables = variables
        self.force_fn = grad(self.potential_energy_constrained_ends, argnums=0)
        self.sol_free: jnp.ndarray | None = None

    def potential_energy_constrained_ends(
        self,
        free_dof_displacement: jnp.ndarray,
        time: float,
        stiffness_vals: jnp.ndarray,
        timepoints: jnp.ndarray,
        impulse_data: jnp.ndarray,
    ) -> jnp.ndarray:
        """Return total potential energy with driven-left and fixed-right ends."""
        variables = self.variables
        imposed_displacement = jnp.interp(time, timepoints, impulse_data)
        displacement1 = jnp.concatenate(
            (jnp.atleast_1d(imposed_displacement), free_dof_displacement[0::2], jnp.zeros(1))
        )
        displacement2 = jnp.concatenate(
            (jnp.atleast_1d(imposed_displacement), free_dof_displacement[1::2], jnp.zeros(1))
        )
        coupling_energy = 0.5 * jnp.sum(
            stiffness_vals[:-1, 0] * jnp.diff(displacement1) ** 2
        )
        local_energy = jnp.sum(
            variables.potential_fn(
                stiffness_vals[:, 1:].T, displacement2 - displacement1
            )
        )
        return coupling_energy + local_energy

    def rhs(
        self,
        state: jnp.ndarray,
        time: float,
        m1s: jnp.ndarray,
        m2s: jnp.ndarray,
        c1s: jnp.ndarray,
        c2s: jnp.ndarray,
        mu_ks: jnp.ndarray,
        beta: float,
        stiffness_vals: jnp.ndarray,
        timepoints: jnp.ndarray,
        impulse_data: jnp.ndarray,
    ) -> jnp.ndarray:
        """Return displacement and velocity derivatives for free DOFs."""
        displacement, velocity = state
        mass_vals = jnp.stack((m1s, m2s), axis=1).reshape(-1)
        relative_velocity = velocity[1::2] - velocity[0::2]
        damping_un = (
            c1s * velocity[0::2]
            + mu_ks * 9.81 * m1s * jnp.tanh(velocity[0::2] * beta)
            - c2s * relative_velocity
        )
        damping_vn = c2s * relative_velocity
        damping = jnp.stack((damping_un, damping_vn), axis=1).reshape(-1)
        # force_fn differentiates potential energy, so mechanical force is -grad(U).
        force = -self.force_fn(
            displacement, time, stiffness_vals, timepoints, impulse_data
        )
        acceleration = (force - damping) / mass_vals
        return jnp.stack((velocity, acceleration))

    def solve(
        self, state: "StateClass", supervisor: "SupervisorClass"
    ) -> jnp.ndarray:
        """Integrate the configured dynamics and store the free-DOF solution."""
        if state.state0_local is None:
            raise RuntimeError("Initialize the local state before calling solve().")
        if supervisor.impulse_data is None:
            raise RuntimeError("Program the impulse before calling solve().")
        self.sol_free = odeint(
            self.rhs,
            state.state0_local,
            supervisor.timepoints,
            state.m1s_free,
            state.m2s_free,
            state.c1s_free,
            state.c2s_free,
            state.mu_ks_free,
            self.variables.beta,
            self.variables.stiffness_vals,
            supervisor.timepoints,
            supervisor.impulse_data,
            rtol=supervisor.rtol,
            atol=supervisor.atol,
            mxstep=supervisor.mxstep,
        )
        return self.sol_free
