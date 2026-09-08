"""Small numerical helpers shared by the simulation classes."""

from __future__ import annotations

import jax.numpy as jnp


def pixel_per_meter(raw_data: jnp.ndarray, reference_length: float = 86e-3) -> jnp.ndarray:
    """Return the image calibration in pixels per metre."""
    data = jnp.asarray(raw_data)
    if data.ndim == 3:
        n_pixels = jnp.abs(data[0, 0, 0] - data[0, 1, 0])
    elif data.ndim == 2:
        n_pixels = jnp.abs(data[0, 0] - data[1, 0])
    else:
        raise ValueError("raw_data must be a two- or three-dimensional array.")
    return n_pixels / reference_length


def potential_order4(ks: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """Evaluate the symmetric fourth-order bistable potential."""
    k2, k3, k4 = ks
    return k2 / 2 * x**2 - k3 / 3 * x**3 + k4 / 4 * x**4


def potential_order4_summed(ks: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """Evaluate and sum the fourth-order potential over its last axis."""
    return jnp.sum(potential_order4(ks, x), axis=-1)


def bistable_potential(params: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """Evaluate a Von Mises truss potential for one or more parameter sets."""
    effective_k, length, theta0, offset = params
    rest_length = length / jnp.cos(theta0) - 2 * offset
    a0 = (rest_length + 2 * offset) * jnp.sin(theta0)
    extension = length / jnp.cos(theta0) - jnp.sqrt((a0 - x) ** 2 + length**2)
    return effective_k * extension**2 / 2


def bistable_potential_summed(params: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """Evaluate and sum the Von Mises truss potential over its last axis."""
    return jnp.sum(bistable_potential(params, x), axis=-1)


def state_to_number(state: str) -> int:
    """Convert a binary system-state string to its integer value."""
    if not state or any(bit not in "01" for bit in state):
        raise ValueError("state must be a non-empty binary string, e.g. '001'.")
    return int(state, 2)


def twogaussian_impulse(
    t: jnp.ndarray,
    amplitude: float,
    width: float,
    start_time: float,
    second_pulse_factor: float = 1.5,
) -> jnp.ndarray:
    """Return two Gaussian displacement pulses."""
    return amplitude * jnp.exp(-(t - start_time) ** 2 / width**2) + amplitude * jnp.exp(
        -(t - second_pulse_factor * start_time) ** 2 / width**2
    )


def gaussian_impulse(
    t: jnp.ndarray,
    amplitude: float,
    start_time: float,
    width: float | None = None,
    frequency: float | None = None,
) -> jnp.ndarray:
    """Return a Gaussian displacement pulse."""
    if width is None:
        if frequency is None or frequency <= 0:
            raise ValueError("A positive frequency or a width must be provided.")
        width = 1 / (4 * jnp.sqrt(2 * jnp.log(2)) * frequency)
    if width <= 0:
        raise ValueError("width must be positive.")
    return amplitude * jnp.exp(-(t - start_time) ** 2 / width**2)


def single_sine_cycle(
    t: jnp.ndarray,
    amplitude: float,
    start_time: float,
    frequency: float,
) -> jnp.ndarray:
    """Return one sinusoidal displacement cycle followed by zero displacement."""
    if frequency <= 0:
        raise ValueError("frequency must be positive.")
    phase = frequency * (t - start_time)
    return jnp.where(
        (phase >= 0) & (phase < 1),
        amplitude * jnp.sin(2 * jnp.pi * phase),
        0.0,
    )
