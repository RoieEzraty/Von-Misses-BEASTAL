"""User-editable configuration for the bistable mass-in-mass simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class VariablesConfig:
    """Physical and geometric parameters."""

    n_units: int = 3
    driven_node: str = "1st"
    truss_model: str = "Trusses"  # "Trusses" or "4th Order"
    # setup = 'experiment_full'
    setup = 'increasing_theta0' # Options: 'experiment_full', 'experiment_single', 'increasing_b', or 'increasing_m'
    # setup = 'increasing_b' # Options: 'experiment_full', 'experiment_single', 'increasing_b', or 'increasing_m'

    # 4th order, currently not in use
    k2: float = 125.0
    k4: float = 309828.0

    # actual truss
    experimental_data_dir: Path = Path("Audrey_exp_data")
    reference_length: float = 86e-3
    increasing_b_mm: tuple[float, ...] = (8.3, 8.7, 9.3)
    increasing_theta0_rise_mm: tuple[float, ...] = (8.3, 10.7, 12.3)
    increasing_theta0_span_mm: float = 40.0

    # relevant for both
    k_truss: float = 1428.0  # stiffness of one Von Mises truss arm
    # m1 = 70*10**(-3) # m_out [kg], 1st paper
    m1: float = 33.6e-3  # outer, 2nd paper beginning
    # m2 = 30e-3*jnp.ones(n_units-2) # uniform physical inner masses [kg], 1st paper
    m2: float = 3.8e-3  # inner, 2nd paper beginning
    increasing_m_range: tuple[float, float] = (3.8, 8.5)
    # k1 = 125 # [N/m]  # outer, 1st paper
    k1: float = 2 * 1020.0  # outer, 2nd paper 
    c1: float = 0.22  # damping of outer mass [kg/s], 1st paper
    c2: float = 0.1  # damping of inner mass [kg/s], 1st paper
    mu_k: float = 0.0675  # coefficient of kinetic friction
    beta: float = 100.0   # smoothing factor for friction force [s/m]


@dataclass(frozen=True)
class SupervisorConfig:
    """Initial-state, input-pulse, and integration parameters."""

    initial_phases: tuple[str, ...] = ("001", "000")
    impulse_type: str = "single_sine_cycle"  # Options: 'single_sine_cycle' or 'gaussian'
    # amplitude = 22*10**(-3) # Peak amplitude [m], 1st paper to buckle
    amplitude: float = 1.8e-3  # probe 2nd paper
    frequency: float = 23.3
    gaussian_width: float | None = None
    start_time: float = 0.025  # Start-time offset [s] for both input types
    second_pulse_factor: float = 1.5
    duration: float = 0.4
    n_timepoints: int = 300

    # should I omit those?
    rtol: float = 1e-8
    atol: float = 1e-8
    mxstep: int = 10_000


@dataclass(frozen=True)
class OutputConfig:
    """State classification and plotting choices."""

    state_threshold: float = 15e-3
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
