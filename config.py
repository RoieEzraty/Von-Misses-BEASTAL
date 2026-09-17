"""User-editable configuration for the bistable mass-in-mass simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import jax.numpy as jnp


@dataclass(frozen=True)
class VariablesConfig:
    """Physical and geometric parameters."""

    n_units: int = 3  # number of Von Mises trusses
    driven_node: str = "1st"  # impulse given to right "1st" or left "last" sides
    truss_model: str = "Trusses"  # "Trusses" (account for tangent angle) or "4th Order" (polynomial approximation with 2 wells)
    # truss_model: str = "4th Order"  # "Trusses" or "4th Order"
    setup = 'experiment_full'  # Audrey's parameters
    # setup = 'increasing_theta0' # Options: 'experiment_full', 'experiment_single', 'increasing_b', or 'increasing_m'
    # setup = 'increasing_b' # Options: 'experiment_full', 'experiment_single', 'increasing_b', or 'increasing_m'

    # parameters for 4th order polynomial, currently not in use
    k2: float = 125.0
    k4: float = 309828.0

    # actual truss parameters
    experimental_data_dir: Path = Path("Audrey_exp_data")
    reference_length: float = 86e-3
    increasing_b_mm: tuple[float, ...] = (8.3, 8.7, 9.3)  # b parameters for each truss increasing
    increasing_theta0_rise_mm: tuple[float, ...] = (8.3, 10.7, 12.3)  # rest angle for each truss increasing 
    increasing_theta0_span_mm: float = 40.0

    # relevant for both
    k_truss: float = 1428.0  # stiffness of one Von Mises truss arm
    m1 = 70*10**(-3) # m_out [kg], 1st paper
    # m1: float = 33.6e-3  # outer, 2nd paper beginning
    m2 = 30e-3*jnp.ones(n_units-2) # uniform physical inner masses [kg], 1st paper
    # m2: float = 3.8e-3  # inner, 2nd paper beginning
    increasing_m_range: tuple[float, float] = (3.8, 8.5)
    k1 = 125 # [N/m]  # outer, 1st paper
    # k1: float = 2 * 1020.0  # outer, 2nd paper 
    c1: float = 0.22  # damping of outer mass [kg/s], 1st paper
    c2: float = 0.1  # damping of inner mass [kg/s], 1st paper
    mu_k: float = 0.0675  # coefficient of kinetic friction
    beta: float = 100.0   # smoothing factor for friction force [s/m]


@dataclass(frozen=True)
class SupervisorConfig:
    """Initial-state, input-pulse, and integration parameters."""

    # initial_states: tuple[str, ...] = ("000", "001")
    initial_states: tuple[str, ...] = ("000", "001", "010", "011", "100", "101", "110", "111")
    desired_state: str = "001"
    impulse_type: str = "single_sine_cycle"  # Options: 'single_sine_cycle' or 'gaussian'
    # amplitude = 22*10**(-3) # Peak amplitude [m], 1st paper to buckle
    # amplitude: float = 1.8e-3  # probe 2nd paper
    amplitude: float = 3.0e-3  # my try Sep14
    # frequency: float = 23.3  # probe 2nd paper
    frequency: float = 23.3  # my try Sep14
    gaussian_width: float | None = None
    start_time: float = 0.025  # Start-time offset [s] for both input types
    second_pulse_factor: float = 1.5
    n_timepoints: int = 1000  # for equilibrium integration
    # f_cutoff: int = 100  # [Hz]
    # duration: float = n_timepoints / (4*f_cutoff)
    duration = 0.8

    # for ODE solution, not in use, should I omit those?
    rtol: float = 1e-8
    atol: float = 1e-8
    mxstep: int = 10_000

    # training params, not relevant for Nathan
    T: int = 100
    alpha: float = 0.1
    algorithm: str = "short"  # Options: "short" or "long" (resetting)
    loss_type: str = "state"


@dataclass(frozen=True)
class OutputConfig:
    """State classification and plotting choices."""

    state_identification_threshold: float = 15e-3
    force_arrival_threshold_fraction: float = 0.05
    plot_potential: bool = True
    plot_responses: bool = True
    compare_endpoint_forces: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level simulation configuration."""

    Variabs: VariablesConfig = field(default_factory=VariablesConfig)
    Sprvsr: SupervisorConfig = field(default_factory=SupervisorConfig)
    Output: OutputConfig = field(default_factory=OutputConfig)


CFG = ExperimentConfig()
