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
    setup: str = "increasing_theta0"

    k2: float = 125.0
    k4: float = 309_828.0
    k_truss: float = 1_428.0  # stiffness of one Von Mises truss arm
    experimental_data_dir: Path = Path("Audrey_exp_data")
    reference_length: float = 86e-3
    increasing_b_mm: tuple[float, ...] = (8.3, 8.7, 9.3)
    increasing_theta0_rise_mm: tuple[float, ...] = (8.3, 10.7, 12.3)
    increasing_theta0_span_mm: float = 40.0

    m1: float = 33.6e-3
    m2: float = 3.8e-3
    increasing_m_range_g: tuple[float, float] = (3.8, 8.5)
    k1: float = 2 * 1_020.0
    c1: float = 0.22
    c2: float = 0.1
    mu_k: float = 0.0675
    beta: float = 100.0


@dataclass(frozen=True)
class SupervisorConfig:
    """Initial-state, input-pulse, and integration parameters."""

    initial_phases: tuple[str, ...] = ("001", "000")
    impulse_type: str = "single_sine_cycle"
    amplitude: float = 1.8e-3
    frequency: float = 23.3
    gaussian_width: float | None = None
    start_time: float = 0.025
    second_pulse_factor: float = 1.5
    duration: float = 0.4
    n_timepoints: int = 300
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
