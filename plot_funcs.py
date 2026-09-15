from __future__ import annotations

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["pdf.fonttype"] = 42

import colors, helpers_builders
from config import CFG

colors_lst, red, custom_cmap, shim = colors.color_scheme(scheme="mine", add_shim=True)


def plot_impulse(timepoints, impulse_data):
    """Plot an imposed displacement history in millimetres."""
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.plot(timepoints, impulse_data * 1e3, color="k")
    ax.set(xlabel="Time (s)", ylabel=r"$\bar{u}(t)$ (mm)", title="Applied Displacement", xlim=(0, float(jnp.max(timepoints))))
    ax.grid(False)
    fig.tight_layout()
    return fig, ax

# -------------------------------------------------
# Importants
# -------------------------------------------------
# Define plotting function
def plot_response(
    num_solution: np.ndarray,
    F: np.ndarray,
    timepoints: np.ndarray,
    alpha: float = 1,
    cmap_temporal=custom_cmap,
    sup_title: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot displacement responses and endpoint forces in 2x2 panels.

    F contains spring forces in N, with time along rows and springs along
    columns. Return the figure and 2x2 main axes, excluding colorbar axes.
    Use plot_laplace_fourier separately for the endpoint-force transforms.
    """
    if figsize is None:
        figsize = (8, 5)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    (ax1, ax2), (ax3, ax4) = axes[:2]

    data_set = num_solution*10**(3)

    applied_disp = data_set[:, 0, 0]
    un_disps = data_set[:, 0, 1:-1]
    delta_disps = data_set[:, 2, 1:-1] - un_disps

    t_end = float(jnp.max(timepoints))
    time_edges = jnp.linspace(0, t_end, un_disps.shape[0] + 1)
    unit_edges = jnp.arange(un_disps.shape[1] + 1)

    ax1.plot(timepoints, applied_disp, color='k', alpha=alpha)
    p2 = ax2.pcolor(time_edges, unit_edges, un_disps.T, cmap=cmap_temporal)
    p3 = ax3.pcolor(time_edges, unit_edges, delta_disps.T, cmap=cmap_temporal)

    plt.colorbar(p2, ax=ax2, pad=0.01, label='$u_n$ (mm)')
    plt.colorbar(p3, ax=ax3, pad=0.01, label=r'$\delta_n$ (mm)')

    ax1.set_title('Applied Displacement', fontsize=12)
    ax2.set_title('Displacement, $u_n$', fontsize=12)
    ax3.set_title(r'Displacement, $\delta_n$', fontsize=12)

    ax2.set_yticks(jnp.arange(0.5, 0.5+un_disps.shape[1]))
    ax2.set_yticklabels(jnp.arange(un_disps.shape[1]), fontsize=12)
    ax3.set_yticks(jnp.arange(0.5, 0.5+un_disps.shape[1]))
    ax3.set_yticklabels(jnp.arange(un_disps.shape[1]), fontsize=12)

    ax1.set_xlim([0, t_end])
    ax2.set_xlim([0, t_end])
    ax3.set_xlim([0, t_end])

    ax1.set_xlabel('Time (s)', fontsize=12)
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax3.set_xlabel('Time (s)', fontsize=12)

    ax1.set_ylabel('Displacement (mm)', fontsize=12)
    ax2.set_ylabel('$n$', fontsize=12)
    ax3.set_ylabel('$n$', fontsize=12)

    # Previous per-unit displacement-in-time plots:
    # for i in range(n_units):
    #     ax3.plot(timepoints, un_disps[:, i], color=trace_colors[i], alpha=alpha, label=f'$u_{i}$')
    #     ax4.plot(timepoints, delta_disps[:, i], color=trace_colors[i], alpha=alpha, label=fr'$\delta_{i}$')

    first_truss_force = F[:, 0]
    final_truss_force = F[:, -1]
    ax4.plot(timepoints, first_truss_force, color=colors_lst[0], alpha=alpha, label='First truss')
    ax4.plot(timepoints, final_truss_force, color=colors_lst[1], alpha=alpha, label='Final truss')
    ax4.set_xlim([0, t_end])
    ax4.set_xlabel('Time (s)', fontsize=12)
    ax4.set_ylabel('Force (N)', fontsize=12)
    ax4.set_title('Endpoint Truss Forces', fontsize=12)
    ax4.legend(fontsize=8, loc='upper right')

    if sup_title is not None:
        fig.suptitle(sup_title)
        
    fig.tight_layout()
    return fig, axes


def plot_laplace_fourier(F: np.ndarray, timepoints: np.ndarray, *, laplace_sigma: float = 0.0, alpha: float = 1, 
                         sup_title: str | None = None, figsize: tuple[float, float] = (8, 3)) -> tuple[plt.Figure, np.ndarray]:
    """Plot endpoint-force Fourier and Laplace transforms in a 1x2 figure.

    F contains spring forces in N, with time along rows and springs along
    columns; only the first and last columns are used. Times must be uniform.
    FFT real parts are solid and imaginary parts dotted, one color per spring.
    Laplace magnitudes use s=laplace_sigma+2j*pi*f, with sigma in 1/s.
    Show the lower half of the nonnegative frequency range in Hz without
    doubling FFT values. Return the figure and a length-two array of axes.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fft_ax, laplace_ax = axes
    for force, color, label in zip(
        (F[:, 0], F[:, -1]),
        colors_lst[:2],
        ('First truss', 'Final truss'),
    ):
        frequencies, spectrum = helpers_builders.force_fft(timepoints, force)
        nonnegative = frequencies >= 0
        frequencies = frequencies[nonnegative]
        s_values = laplace_sigma + 2j * np.pi * frequencies
        _, laplace = helpers_builders.force_laplace(timepoints, force, s_values)
        fft_ax.plot(frequencies, spectrum[nonnegative].real, color=color, alpha=alpha,
                    linestyle='-', label=f'{label} (real)')
        fft_ax.plot(frequencies, spectrum[nonnegative].imag, color=color, alpha=alpha,
                    linestyle=':', label=f'{label} (imaginary)')
        laplace_ax.plot(frequencies, np.abs(laplace), color=color, alpha=alpha, label=label)
    fft_ax.set_title('Endpoint Force FFT', fontsize=12)
    fft_ax.set_ylabel('Transform (N s)', fontsize=12)
    laplace_ax.set_ylabel('Transform magnitude (N s)', fontsize=12)
    laplace_ax.set_title(fr'Endpoint Force Laplace ($\sigma={laplace_sigma:g}$ s$^{{-1}}$)', fontsize=12)
    for ax in (fft_ax, laplace_ax):
        ax.set_xlabel('Frequency (Hz)', fontsize=12)
        ax.set_xlim(0, float(frequencies[-1]) / 2 if frequencies[-1] > 0 else 1.0)
        ax.legend(fontsize=8)
        ax.grid(False)

    if sup_title is not None:
        fig.suptitle(sup_title)
    fig.tight_layout()
    return fig, axes


# -------------------------------------------------
# Potential
# -------------------------------------------------
def plot_potential_O4(k_fit4, x_min, x_max):
    """
    Plot the bistable potential energy landscape.

    Parameters
    ----------
    k_fit4 : array-like
        Coefficients for the 4th order potential.
    x_min : float
        Minimum displacement value for the inner mass.
    x_max : float
        Maximum displacement value for the inner mass.
    """
    d1 = 0  # displacement of outer mass
    d2 = jnp.linspace(x_min, x_max, 1000) # displacement of inner mass
    fitted_energy4 = helpers_builders.potential_order4(k_fit4, d2)

    # Visualize the energy landscape
    plt.figure(figsize=[4, 3])
    plt.plot(d2 * 10**3, fitted_energy4 * 10**3, 'k')
    plt.xlabel('Displacement (mm)')
    plt.ylabel('Energy (mJ)')
    plt.title('Bistable Energy Landscape')
    plt.tight_layout()


def plot_potential(variables, x_min=-0.005, x_max=0.031):
    """Plot the configured physical-unit potential energy landscapes."""
    displacement = jnp.linspace(x_min, x_max, 1000)
    fig, ax = plt.subplots(figsize=(4, 3))
    if variables.truss_model == "4th Order":
        energy = helpers_builders.potential_order4(variables.k_fit4, displacement)
        ax.plot(displacement * 1e3, energy * 1e3, color="k")
    else:
        params = variables.bistable_potential_params[1:-1].T
        energy = helpers_builders.bistable_potential(params, displacement[:, None])
        ax.plot(displacement * 1e3, energy * 1e3)
    ax.set(xlabel="Displacement (mm)", ylabel="Energy (mJ)", title="Bistable Energy Landscape")
    fig.tight_layout()
    return fig, ax


def detect_force_arrival(timepoints: jnp.ndarray, force: jnp.ndarray, start_time: float,
                         threshold_fraction: float = 0.05,) -> float:
    """Return the first post-input time at a fraction of peak force."""
    if not 0 < threshold_fraction <= 1:
        raise ValueError("threshold_fraction must lie in (0, 1].")
    post_input = timepoints >= start_time
    peak_force = jnp.max(jnp.where(post_input, jnp.abs(force), 0.0))
    if float(peak_force) == 0.0:
        raise ValueError("Cannot detect arrival in a zero force signal.")
    has_arrived = post_input & (jnp.abs(force) >= threshold_fraction * peak_force)
    return float(timepoints[int(jnp.argmax(has_arrived))])


def plot_force_comparison(solutions_by_phase, timepoints, spring_stiffness, start_time, threshold_fraction=0.05, *, 
                          show_transform: bool = True, laplace_sigma: float = 0.0):
    """Compare two phases in 2x2 panels, optionally adding two transform rows.

    With show_transform=True, row 3 shows complex FFTs of the endpoint force
    differences (solid real, dotted imaginary); row 4 shows Laplace magnitudes
    at s=laplace_sigma+2j*pi*f. Columns show unaligned and delay-aligned data.
    All four transforms use the common valid time window ending at t[-1]-delay,
    avoiding extrapolation of the advanced final force. Frequencies are in Hz,
    sigma in 1/s, and transform values in N s. Return figure, axes, and delay.
    """
    if len(solutions_by_phase) != 2:
        raise ValueError("Force comparison requires exactly two initial phases.")
    phase_a, phase_b = tuple(solutions_by_phase)
    forces = {}
    for phase, solution in solutions_by_phase.items():
        forces[phase] = {"first": spring_stiffness * (solution[:, 0, 0] - solution[:, 0, 1]),
                         "final": spring_stiffness * (solution[:, 0, -2] - solution[:, 0, -1])}

    first_arrival = detect_force_arrival(timepoints, forces[phase_a]["first"], start_time, threshold_fraction)
    final_arrival = detect_force_arrival(timepoints, forces[phase_a]["final"], start_time, threshold_fraction)
    delay = final_arrival - first_arrival
    if delay < 0:
        raise ValueError("Detected final-spring arrival before first-spring arrival.")

    delta_first = forces[phase_a]["first"] - forces[phase_b]["first"]
    delta_final = forces[phase_a]["final"] - forces[phase_b]["final"]
    delta_final_aligned = jnp.interp(timepoints + delay, timepoints, delta_final, left=jnp.nan, right=jnp.nan)
    final_spring_id = next(iter(solutions_by_phase.values())).shape[2] - 1
    fig, axes = plt.subplots(4 if show_transform else 2, 2, figsize=(10, 12) if show_transform else (8, 6), sharex=False)

    for phase_id, phase in enumerate((phase_a, phase_b)):
        first_force = forces[phase]["first"]
        final_force = forces[phase]["final"]
        final_aligned = jnp.interp(timepoints + delay, timepoints, final_force, left=jnp.nan, right=jnp.nan)
        for ax, final_signal, final_label in zip(axes[0], (final_force, final_aligned), ("t", r"t+\tau")):
            ax.plot(timepoints, first_force, color=colors_lst[phase_id], label=fr"$F_1$, initial {phase}")
            ax.plot(timepoints, final_signal, color=colors_lst[phase_id], linestyle="--",
                    label=fr"$F_{{{final_spring_id}}}({final_label})$, initial {phase}")

    for ax, final_difference, final_label in zip(axes[1], (delta_final, delta_final_aligned), ("t", r"t+\tau")):
        ax.plot(timepoints, delta_first, color=colors_lst[0], label=r"$\Delta F_1(t)$")
        ax.plot(timepoints, final_difference, color=red, linestyle="--",
                label=fr"$\Delta F_{{{final_spring_id}}}({final_label})$")
        ax.axhline(0, color="k", linewidth=0.8, alpha=0.35)

    axes[0, 0].axvline(first_arrival, color="k", linestyle=":", alpha=0.4)
    axes[0, 0].axvline(final_arrival, color="k", linestyle=":", alpha=0.4)
    axes[0, 1].axvline(first_arrival, color="k", linestyle=":", alpha=0.4)
    axes[0, 0].set_title("Endpoint Forces — Without Delay Alignment")
    axes[0, 1].set_title(fr"Endpoint Forces — With $\tau={delay * 1e3:.1f}$ ms")
    axes[1, 0].set_title("Force Differences — Without Delay Alignment")
    axes[1, 1].set_title(fr"Force Differences — With $\tau={delay * 1e3:.1f}$ ms")
    axes[0, 0].set_ylabel("Force (N)")
    axes[1, 0].set_ylabel(fr"$F^{{{phase_a}}}-F^{{{phase_b}}}$ (N)")
    for ax in axes[1]:
        ax.set_xlabel("Time (s)")
    for ax in axes[:2].flat:
        ax.grid(False)
        ax.set_xlim(0, float(jnp.max(timepoints)))
        ax.legend(fontsize=7)
    if show_transform:
        times = np.asarray(timepoints)
        valid = times + delay <= times[-1]
        transform_times = times[valid]
        if transform_times.size < 2:
            raise ValueError("Delay alignment leaves fewer than two samples for transforms.")
        first_difference = np.asarray(delta_first)[valid]
        for column, final_difference in enumerate((delta_final, delta_final_aligned)):
            fft_ax, laplace_ax = axes[2, column], axes[3, column]
            for signal, color, label in zip((first_difference, np.asarray(final_difference)[valid]), (colors_lst[0], red), 
                                            ('First spring difference', 'Final spring difference')):
                frequencies, spectrum = helpers_builders.force_fft(transform_times, signal)
                nonnegative = frequencies >= 0
                frequencies, spectrum = frequencies[nonnegative], spectrum[nonnegative]
                _, laplace = helpers_builders.force_laplace(transform_times, signal, 
                                                            laplace_sigma + 2j * np.pi * frequencies)
                fft_ax.plot(frequencies, spectrum.real, color=color, linestyle='-', label=f'{label} (real)')
                fft_ax.plot(frequencies, spectrum.imag, color=color, linestyle=':', label=f'{label} (imaginary)')
                laplace_ax.plot(frequencies, np.abs(laplace), color=color, label=label)
            alignment = 'Delay=0' if column == 0 else fr'Delay$={delay * 1e3:.1f}$ ms'
            fft_ax.set_title(f'Force Difference FFT {phase_a} - {phase_b}, {alignment}')
            laplace_ax.set_title(fr'Laplace ($\sigma={laplace_sigma:g}$ s$^{{-1}}$) — {alignment}')
            fft_ax.set_ylabel('Transform (N s)')
            laplace_ax.set_ylabel('Transform magnitude (N s)')
            for ax in (fft_ax, laplace_ax):
                ax.set_xlabel('Frequency (Hz)')
                ax.set_xlim(0, float(frequencies[-1]) / 2 if frequencies[-1] > 0 else 1.0)
                if ax == laplace_ax:
                    ax.set_ylim([-0.005, 0.03])
                else:
                    ax.set_ylim([-0.028, 0.028])
                ax.grid(False)
                ax.legend(fontsize=7)
    fig.tight_layout()
    return fig, axes, delay
