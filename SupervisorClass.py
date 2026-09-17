"""External input programming for the simulation."""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np
import jax.numpy as jnp
from jax import grad
from numpy import zeros
from numpy.typing import NDArray

import helpers_builders
from config import ExperimentConfig


class SupervisorClass:
    """Hold integration times, initial states, and the imposed displacement."""

    def __init__(self, cfg: ExperimentConfig) -> None:
        # read from CFG
        if cfg.Sprvsr.n_timepoints < 2:
            raise ValueError("n_timepoints must be at least two.")
        if cfg.Sprvsr.duration <= 0:
            raise ValueError("duration must be positive.")
        self.initial_states = cfg.Sprvsr.initial_states
        self.desired_state = cfg.Sprvsr.desired_state
        self.impulse_type = cfg.Sprvsr.impulse_type
        self.amplitude = cfg.Sprvsr.amplitude
        self.frequency = cfg.Sprvsr.frequency
        self.gaussian_width = cfg.Sprvsr.gaussian_width
        self.start_time = cfg.Sprvsr.start_time
        self.second_pulse_factor = cfg.Sprvsr.second_pulse_factor
        self.timepoints = jnp.linspace(0, cfg.Sprvsr.duration, cfg.Sprvsr.n_timepoints)
        self.rtol = cfg.Sprvsr.rtol
        self.atol = cfg.Sprvsr.atol
        self.mxstep = cfg.Sprvsr.mxstep

        self.alpha: float = cfg.Sprvsr.alpha
        self.A_norm: float = copy.copy(self.amplitude)
        self.algorithm: str = cfg.Sprvsr.algorithm
        if self.algorithm not in {"short", "long"}:
            raise ValueError("algorithm must be either 'short' or 'long'.")
        self.loss_type: str = cfg.Sprvsr.loss_type
        if self.loss_type == "state":
            self.loss_fn = self.loss_state
        self.T: int = cfg.Sprvsr.T  # training steps
        self.dLoss_do_in_t: NDArray[np.float32] = zeros((self.T, cfg.Variabs.n_units), dtype=np.float32)  # (T,B)
        self.loss_in_t: NDArray[np.float32] = zeros((self.T,), dtype=np.float32)  # (T,)
        self.update_A_in_t: NDArray[np.float32] = zeros((self.T,), dtype=np.float32)  # (T,)
        self.reset_A_in_t: NDArray[np.float32] = np.full((self.T,), np.nan, dtype=np.float32)  # (T,)
        self.positivity_in_t: NDArray[np.float32] = zeros((self.T,), dtype=np.float32)  # (T,)
        if len(self.desired_state) != cfg.Variabs.n_units or any(bit not in "01" for bit in self.desired_state):
            raise ValueError("desired_state must contain one binary entry per physical unit.")
        self.desired_state: NDArray[np.float32] = np.asarray([int(bit) for bit in self.desired_state], dtype=np.float32)
        self.measured_state: NDArray[np.float32] | None = None
        self.dLoss_do: NDArray[np.float32] = zeros((cfg.Variabs.n_units,), dtype=np.float32)
        self.loss: float = 0.0
        self.update_A_nxt: float = 0.0
        self.reset_A_nxt: float | None = None

        self.impulse_dyn: jnp.ndarray | None = None

    def program_impulse(self, impulse_type: Optional[str] = None, timepoints: Optional[jnp.ndarray] = None, amplitude: Optional[float] = None,
                        start_time: Optional[float] = None, frequency: Optional[float] = None, width: Optional[float] = None) -> jnp.ndarray:
        """Create and store the configured displacement history."""
        # inherit from self
        impulse_type = self.impulse_type if impulse_type is None else impulse_type
        timepoints = self.timepoints if timepoints is None else timepoints
        amplitude = self.amplitude if amplitude is None else amplitude
        start_time = self.start_time if start_time is None else start_time
        frequency = self.frequency if frequency is None else frequency

        # user specified
        if impulse_type == "single_sine_cycle":
            impulse_dyn = helpers_builders.single_sine_cycle(timepoints, amplitude, start_time, frequency)
        elif impulse_type == "gaussian":
            width = self.gaussian_width if width is None else width
            impulse_dyn = helpers_builders.gaussian_impulse(timepoints, amplitude, start_time, width=width, frequency=frequency)
        elif impulse_type == "two_gaussian":
            width = self.gaussian_width if width is None else width
            if width is None:
                width = 1 / (4 * jnp.sqrt(2 * jnp.log(2)) * frequency)
            impulse_dyn = helpers_builders.twogaussian_impulse(timepoints, amplitude, width, start_time, self.second_pulse_factor)
        else:
            raise ValueError(f"Unknown impulse_type: {impulse_type}")

        # save in self
        self.impulse_dyn = impulse_dyn
        return impulse_dyn

    def measure(self, State: "StateClass") -> None:
        """Store the current measured state for the configured loss."""
        if self.loss_type == "state":
            self.measured_state = np.asarray(list(State.state), dtype=np.float32)

    def calc_loss(self, t: int, measured_state: NDArray[np.number] | None = None) -> float:
        """Store the squared state loss and ``dLoss_do = partial L / partial B`` at step ``t``."""
        if measured_state is not None:
            self.measured_state = np.asarray(measured_state, dtype=np.float32)
        self.dLoss_do = self.loss_fn()
        self.loss = float(np.sum(np.square(self.dLoss_do / 2)))
        self.dLoss_do_in_t[t] = self.dLoss_do
        self.loss_in_t[t] = self.loss
        return self.loss

    def calc_update_vals(self, t: int) -> tuple[float, ...]:
        """Calculate ``A(t)`` and return the pulse amplitudes to apply in order.

        The long algorithm returns ``(-A(t-1), A(t))`` when its reset condition is met; otherwise both algorithms return only ``(A(t),)``.
        """
        dLoss_prev = zeros(self.dLoss_do.shape, dtype=np.float32) if t == 0 else self.dLoss_do_in_t[t-1]
        A_prev = 0.0 if t == 0 else float(self.update_A_in_t[t-1])
        dLoss_last_prev = self._last_nonzero(dLoss_prev)
        dLoss_last_curr = self._last_nonzero(self.dLoss_do)
        self.update_A_nxt = dLoss_last_curr * (dLoss_last_prev * A_prev + self.alpha * self.A_norm)
        self.update_A_in_t[t] = self.update_A_nxt

        self.reset_A_nxt = None
        self.reset_A_in_t[t] = np.nan
        if self.algorithm == "long" and t > 0:
            last_curr_index = self._last_nonzero_index(self.dLoss_do)
            if last_curr_index is not None and np.any(np.asarray(self.dLoss_do[:last_curr_index]) * np.asarray(dLoss_prev[:last_curr_index]) != 0):
                self.reset_A_nxt = -A_prev
                self.reset_A_in_t[t] = self.reset_A_nxt
        return (self.update_A_nxt,) if self.reset_A_nxt is None else (self.reset_A_nxt, self.update_A_nxt)

    @staticmethod
    def _last_nonzero(dLoss: NDArray[np.number]) -> float:
        """Return the final nonzero entry, or zero when all entries are zero."""
        nonzero_indices = np.flatnonzero(dLoss)
        return float(dLoss[nonzero_indices[-1]]) if nonzero_indices.size else 0.0

    @staticmethod
    def _last_nonzero_index(dLoss: NDArray[np.number]) -> int | None:
        """Return the index of the final nonzero entry, or ``None`` when all entries are zero."""
        nonzero_indices = np.flatnonzero(dLoss)
        return int(nonzero_indices[-1]) if nonzero_indices.size else None

    # ------------
    # Impulse
    # -----------
    def impulse_fn(self, time: jnp.ndarray) -> jnp.ndarray:
        """Interpolate the programmed displacement at ``time``."""
        if self.impulse_dyn is None:
            raise RuntimeError("Call program_impulse() before evaluating the impulse.")
        return jnp.interp(time, self.timepoints, self.impulse_dyn)

    def dimpulse_fn(self, time: float) -> jnp.ndarray:
        """Return the derivative of the interpolated displacement."""
        return grad(self.impulse_fn)(time)

    # ------------
    # Loss functions
    # -----------
    def loss_state(self) -> NDArray[np.float32]:
        """Return ``partial L / partial B = -2 * (desired_state - measured_state)``."""
        if self.measured_state is None:
            raise RuntimeError("Set measured_state or pass it to calc_loss() before calculating the loss.")
        if self.measured_state.shape != self.desired_state.shape:
            raise ValueError("measured_state must contain one binary entry per physical unit.")
        return np.asarray(-2 * (self.desired_state - self.measured_state), dtype=np.float32)
