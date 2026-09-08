from __future__ import annotations

import jax.numpy as jnp
import matplotlib.pyplot as plt

plt.rcParams["pdf.fonttype"] = 42

import colors, helpers_builders
from config import CFG

colors_lst, red, custom_cmap, shim = colors.color_scheme(scheme="mine", add_shim=True)


def plot_impulse(timepoints, impulse_data):
    """Plot an imposed displacement history in millimetres."""
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.plot(timepoints, impulse_data * 1e3, color="k")
    ax.set(
        xlabel="Time (s)",
        ylabel=r"$\bar{u}(t)$ (mm)",
        title="Applied Displacement",
        xlim=(0, float(jnp.max(timepoints))),
    )
    ax.grid(False)
    fig.tight_layout()
    return fig, ax

# -------------------------------------------------
# Importants
# -------------------------------------------------
# Define plotting function
def plot_response(
    num_solution,
    timepoints,
    alpha=1,
    cmap_temporal=custom_cmap,
    sup_title=None,
    figsize=[8, 5],
    spring_stiffness=CFG.Variabs.k1,
):
    """Plot the applied displacement, unit response, and endpoint forces."""
    n_units = num_solution.shape[2] - 2

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

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

    first_truss_force = spring_stiffness * (num_solution[:, 0, 0] - num_solution[:, 0, 1])
    final_truss_force = spring_stiffness * (num_solution[:, 0, -2] - num_solution[:, 0, -1])
    ax4.plot(timepoints, first_truss_force, color=colors_lst[0], alpha=alpha, label='First truss')
    ax4.plot(timepoints, final_truss_force, color=colors_lst[1], alpha=alpha, label='Final truss')
    ax4.set_xlim([0, t_end])
    ax4.set_xlabel('Time (s)', fontsize=12)
    ax4.set_ylabel('Force (N)', fontsize=12)
    ax4.set_title('Endpoint Truss Forces', fontsize=12)
    ax4.legend(fontsize=8, loc='upper right')

    if sup_title is not None:
        plt.suptitle(f'{sup_title}')
        
    plt.tight_layout()


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
    ax.set(
        xlabel="Displacement (mm)",
        ylabel="Energy (mJ)",
        title="Bistable Energy Landscape",
    )
    fig.tight_layout()
    return fig, ax


def detect_force_arrival(
    timepoints: jnp.ndarray,
    force: jnp.ndarray,
    start_time: float,
    threshold_fraction: float = 0.05,
) -> float:
    """Return the first post-input time at a fraction of peak force."""
    if not 0 < threshold_fraction <= 1:
        raise ValueError("threshold_fraction must lie in (0, 1].")
    post_input = timepoints >= start_time
    peak_force = jnp.max(jnp.where(post_input, jnp.abs(force), 0.0))
    if float(peak_force) == 0.0:
        raise ValueError("Cannot detect arrival in a zero force signal.")
    has_arrived = post_input & (jnp.abs(force) >= threshold_fraction * peak_force)
    return float(timepoints[int(jnp.argmax(has_arrived))])


def plot_force_comparison(
    solutions_by_phase,
    timepoints,
    spring_stiffness,
    start_time,
    threshold_fraction=0.05,
):
    """Compare endpoint forces for exactly two initial phase descriptions."""
    if len(solutions_by_phase) != 2:
        raise ValueError("Force comparison requires exactly two initial phases.")
    phase_a, phase_b = tuple(solutions_by_phase)
    forces = {}
    for phase, solution in solutions_by_phase.items():
        forces[phase] = {
            "first": spring_stiffness * (solution[:, 0, 0] - solution[:, 0, 1]),
            "final": spring_stiffness * (solution[:, 0, -2] - solution[:, 0, -1]),
        }

    first_arrival = detect_force_arrival(
        timepoints, forces[phase_a]["first"], start_time, threshold_fraction
    )
    final_arrival = detect_force_arrival(
        timepoints, forces[phase_a]["final"], start_time, threshold_fraction
    )
    delay = final_arrival - first_arrival
    if delay < 0:
        raise ValueError("Detected final-spring arrival before first-spring arrival.")

    delta_first = forces[phase_a]["first"] - forces[phase_b]["first"]
    delta_final = forces[phase_a]["final"] - forces[phase_b]["final"]
    delta_final_aligned = jnp.interp(
        timepoints + delay, timepoints, delta_final, left=jnp.nan, right=jnp.nan
    )
    final_spring_id = next(iter(solutions_by_phase.values())).shape[2] - 1
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True)

    for phase_id, phase in enumerate((phase_a, phase_b)):
        first_force = forces[phase]["first"]
        final_force = forces[phase]["final"]
        final_aligned = jnp.interp(
            timepoints + delay,
            timepoints,
            final_force,
            left=jnp.nan,
            right=jnp.nan,
        )
        for ax, final_signal, final_label in zip(
            axes[0], (final_force, final_aligned), ("t", r"t+\tau")
        ):
            ax.plot(timepoints, first_force, color=colors_lst[phase_id],
                    label=fr"$F_1$, initial {phase}")
            ax.plot(timepoints, final_signal, color=colors_lst[phase_id],
                    linestyle="--",
                    label=fr"$F_{{{final_spring_id}}}({final_label})$, initial {phase}")

    for ax, final_difference, final_label in zip(
        axes[1], (delta_final, delta_final_aligned), ("t", r"t+\tau")
    ):
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
    for ax in axes.flat:
        ax.grid(False)
        ax.set_xlim(0, float(jnp.max(timepoints)))
        ax.legend(fontsize=7)
    fig.tight_layout()
    return fig, axes, delay
