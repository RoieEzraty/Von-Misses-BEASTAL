from __future__ import annotations

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams["pdf.fonttype"] = 42

import colors, helpers_builders
from config import CFG

colors_lst, red, custom_cmap, shim = colors.color_scheme(scheme="mine", add_shim=True)


# -------------------------------------------------
# Importants
# -------------------------------------------------
def plot_training(update_A_in_t: np.ndarray, state_in_t: np.ndarray, loss_in_t: np.ndarray, *,
                  sup_title: str | None = None, figsize: tuple[float, float] = (8, 7)) -> tuple[plt.Figure, np.ndarray]:
    """Plot pulse amplitude, binary system state, and absolute loss over training steps.

    ``state_in_t`` must have shape ``(T, n_physical_units)``. The pulse
    amplitudes are supplied in metres and displayed in millimetres. Return the
    figure and the three axes, ordered from top to bottom.
    """
    amplitudes = np.asarray(update_A_in_t).reshape(-1)
    states = np.asarray(state_in_t)
    losses = np.asarray(loss_in_t).reshape(-1)
    if states.ndim != 2:
        raise ValueError("state_in_t must be a 2D array with shape (T, n_physical_units).")
    if not (amplitudes.size == states.shape[0] == losses.size):
        raise ValueError("update_A_in_t, state_in_t, and loss_in_t must contain the same number of training steps.")
    if amplitudes.size == 0 or states.shape[1] == 0:
        raise ValueError("Training histories must contain at least one step and one physical unit.")

    training_steps = np.arange(amplitudes.size)
    step_edges = np.arange(amplitudes.size + 1) - 0.5
    unit_edges = np.arange(states.shape[1] + 1) - 0.5
    state_cmap = ListedColormap([colors_lst[1], colors_lst[2]], name="binary_state")
    state_norm = BoundaryNorm([-0.5, 0.5, 1.5], state_cmap.N)
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True, height_ratios=(1, 1.4, 1))
    amplitude_ax, state_ax, loss_ax = axes

    amplitude_ax.plot(training_steps, amplitudes * 1e3, color=colors_lst[0], marker="o", markersize=3)
    state_map = state_ax.pcolormesh(step_edges, unit_edges, states.T, cmap=state_cmap, norm=state_norm, shading="flat")
    loss_ax.plot(training_steps, np.abs(losses), color=red, marker="o", markersize=3)
    fig.colorbar(state_map, ax=state_ax, ticks=(0, 1), pad=0.01, label="State")

    amplitude_ax.set(ylabel="Pulse amplitude (mm)", title="Training Pulse Amplitude")
    state_ax.set(ylabel="Physical unit, $n$", title="System State", yticks=np.arange(states.shape[1]))
    loss_ax.set(xlabel="Training step, $t$", ylabel=r"$|L|$", title="Absolute Loss")
    for ax in axes:
        ax.grid(False)
        ax.set_xlim(-0.5, amplitudes.size - 0.5)
    loss_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if sup_title is not None:
        fig.suptitle(sup_title)
    fig.tight_layout()
    return fig, axes


# -------------------------------------------------
# IMPULSE
# -------------------------------------------------
def plot_impulse(timepoints, impulse_dyn):
    """Plot an imposed displacement history in millimetres."""
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.plot(timepoints, impulse_dyn * 1e3, color="k")
    ax.set(xlabel="Time (s)", ylabel=r"$\bar{u}(t)$ (mm)", title="Applied Displacement", xlim=(0, float(jnp.max(timepoints))))
    ax.grid(False)
    fig.tight_layout()
    return fig, ax


# -------------------------------------------------
# TIP FORCES
# -------------------------------------------------
def plot_force_comparison(F_dyn_by_state, timepoints, start_time, threshold_fraction=0.05, *,
                          show_transform: bool = True, show_delay: bool = False, laplace_sigma: float = 0.0):
    """Compare the force histories of two initial states, optionally adding two transform rows.

    With show_transform=True, row 3 shows complex FFTs of the endpoint force
    differences (solid real, dotted imaginary); row 4 shows Laplace magnitudes
    at s=laplace_sigma+2j*pi*f. With show_delay=True, a second column shows
    delay-aligned data and all transforms use the common valid time window ending
    at t[-1]-delay, avoiding extrapolation of the advanced final force. Frequencies
    are in Hz, sigma in 1/s, and transform values in N s. Return figure, axes, and delay.
    """
    if len(F_dyn_by_state) != 2:
        raise ValueError("Force comparison requires exactly two initial states.")
    state_a, state_b = tuple(F_dyn_by_state)
    forces_dyn = {state: {"first": F_dyn[:, 0], "final": F_dyn[:, -1]} for state, F_dyn in F_dyn_by_state.items()}

    first_arrival = detect_force_arrival(timepoints, forces_dyn[state_a]["first"], start_time, threshold_fraction)
    final_arrival = detect_force_arrival(timepoints, forces_dyn[state_a]["final"], start_time, threshold_fraction)
    delay = final_arrival - first_arrival
    if delay < 0:
        raise ValueError("Detected final-spring arrival before first-spring arrival.")

    delta_first_F_dyn = forces_dyn[state_a]["first"] - forces_dyn[state_b]["first"]
    delta_final_F_dyn = forces_dyn[state_a]["final"] - forces_dyn[state_b]["final"]
    final_differences_dyn = [delta_final_F_dyn]
    final_labels = ["t"]
    if show_delay:
        final_differences_dyn.append(jnp.interp(timepoints + delay, timepoints, delta_final_F_dyn, left=jnp.nan, right=jnp.nan))
        final_labels.append(r"t+\tau")
    final_spring_id = next(iter(F_dyn_by_state.values())).shape[1]
    nrows, ncols = (4 if show_transform else 2), (2 if show_delay else 1)
    fig, axes = plt.subplots(nrows, ncols, figsize=((10 if show_delay else 5), (12 if show_transform else 6)), sharex=False, squeeze=False)

    for state_id, state in enumerate((state_a, state_b)):
        first_force_dyn = forces_dyn[state]["first"]
        final_force_dyn = forces_dyn[state]["final"]
        final_signals_dyn = [final_force_dyn]
        if show_delay:
            final_signals_dyn.append(jnp.interp(timepoints + delay, timepoints, final_force_dyn, left=jnp.nan, right=jnp.nan))
        for ax, final_signal_dyn, final_label in zip(axes[0], final_signals_dyn, final_labels):
            ax.plot(timepoints, first_force_dyn, color=colors_lst[state_id], label=fr"$F_1$, initial {state}")
            ax.plot(timepoints, final_signal_dyn, color=colors_lst[state_id], linestyle="--",
                    label=fr"$F_{{{final_spring_id}}}({final_label})$, initial {state}")

    for ax, final_difference_dyn, final_label in zip(axes[1], final_differences_dyn, final_labels):
        ax.plot(timepoints, delta_first_F_dyn, color=colors_lst[0], label=r"$\Delta F_1(t)$")
        ax.plot(timepoints, final_difference_dyn, color=red, linestyle="--",
                label=fr"$\Delta F_{{{final_spring_id}}}({final_label})$")
        ax.axhline(0, color="k", linewidth=0.8, alpha=0.35)

    axes[0, 0].axvline(first_arrival, color="k", linestyle=":", alpha=0.4)
    axes[0, 0].axvline(final_arrival, color="k", linestyle=":", alpha=0.4)
    axes[0, 0].set_title("Endpoint Forces — Without Delay Alignment")
    axes[1, 0].set_title("Force Differences — Without Delay Alignment")
    if show_delay:
        axes[0, 1].axvline(first_arrival, color="k", linestyle=":", alpha=0.4)
        axes[0, 1].set_title(fr"Endpoint Forces — With $\tau={delay * 1e3:.1f}$ ms")
        axes[1, 1].set_title(fr"Force Differences — With $\tau={delay * 1e3:.1f}$ ms")
    axes[0, 0].set_ylabel("Force (N)")
    axes[1, 0].set_ylabel(fr"$F^{{{state_a}}}-F^{{{state_b}}}$ (N)")
    for ax in axes[1]:
        ax.set_xlabel("Time (s)")
    for ax in axes[:2].flat:
        ax.grid(False)
        ax.set_xlim(0, float(jnp.max(timepoints)))
        ax.legend(fontsize=7)
    if show_transform:
        times = np.asarray(timepoints)
        valid = times + delay <= times[-1] if show_delay else np.ones(times.shape, dtype=bool)
        transform_times = times[valid]
        if transform_times.size < 2:
            raise ValueError("Delay alignment leaves fewer than two samples for transforms.")
        first_difference_dyn = np.asarray(delta_first_F_dyn)[valid]
        for column, final_difference_dyn in enumerate(final_differences_dyn):
            fft_ax, laplace_ax = axes[2, column], axes[3, column]
            for signal_dyn, color, label in zip((first_difference_dyn, np.asarray(final_difference_dyn)[valid]), (colors_lst[0], red),
                                            ('First spring difference', 'Final spring difference')):
                frequencies, spectrum = helpers_builders.force_fft(transform_times, signal_dyn)
                nonnegative = frequencies >= 0
                frequencies, spectrum = frequencies[nonnegative], spectrum[nonnegative]
                _, laplace = helpers_builders.force_laplace(transform_times, signal_dyn,
                                                            laplace_sigma + 2j * np.pi * frequencies)
                fft_ax.plot(frequencies, spectrum.real, color=color, linestyle='-', label=f'{label} (real)')
                fft_ax.plot(frequencies, spectrum.imag, color=color, linestyle=':', label=f'{label} (imaginary)')
                laplace_ax.plot(frequencies, np.abs(laplace), color=color, label=label)
            alignment = 'Delay=0' if column == 0 else fr'Delay$={delay * 1e3:.1f}$ ms'
            fft_ax.set_title(f'Force Difference FFT {state_a} - {state_b}, {alignment}')
            laplace_ax.set_title(fr'Laplace ($\sigma={laplace_sigma:g}$ s$^{{-1}}$) — {alignment}')
            fft_ax.set_ylabel('Transform (N s)')
            laplace_ax.set_ylabel('Transform magnitude (N s)')
            for ax in (fft_ax, laplace_ax):
                ax.set_xlabel('Frequency (Hz)')
                ax.set_xlim(0, float(frequencies[-1]) / 4 if frequencies[-1] > 0 else 1.0)
                if ax == laplace_ax:
                    ax.set_ylim([-0.005, 0.06])
                else:
                    ax.set_ylim([-0.028, 0.028])
                ax.grid(False)
                ax.legend(fontsize=7)
    fig.tight_layout()
    return fig, axes, delay


def plot_response(u_dyn: np.ndarray, delta_dyn: np.ndarray, F_dyn: np.ndarray, timepoints: np.ndarray, impulse_dyn: np.ndarray,
                  alpha: float = 1, cmap_temporal=custom_cmap, sup_title: str | None = None, 
                  figsize: tuple[float, float] | None = None) -> tuple[plt.Figure, np.ndarray]:
    """Plot ``u``, ``delta``, the imposed displacement, and endpoint forces.

    All inputs contain simulation time along rows. ``F_dyn`` contains spring
    forces in N. Return the figure and 2x2 main axes, excluding colorbar axes.
    """
    if figsize is None:
        figsize = (8, 5)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    (ax1, ax2), (ax3, ax4) = axes[:2]

    applied_displacement_dyn = impulse_dyn * 1e3
    u_mm_dyn = u_dyn * 1e3
    delta_mm_dyn = delta_dyn * 1e3

    t_end = float(jnp.max(timepoints))
    time_edges = jnp.linspace(0, t_end, u_mm_dyn.shape[0] + 1)
    unit_edges = jnp.arange(u_mm_dyn.shape[1] + 1)

    ax1.plot(timepoints, applied_displacement_dyn, color='k', alpha=alpha)
    p2 = ax2.pcolor(time_edges, unit_edges, u_mm_dyn.T, cmap=cmap_temporal)
    p3 = ax3.pcolor(time_edges, unit_edges, delta_mm_dyn.T, cmap=cmap_temporal)

    plt.colorbar(p2, ax=ax2, pad=0.01, label='$u_n$ (mm)')
    plt.colorbar(p3, ax=ax3, pad=0.01, label=r'$\delta_n$ (mm)')

    ax1.set_title('Applied Displacement', fontsize=12)
    ax2.set_title('Displacement, $u_n$', fontsize=12)
    ax3.set_title(r'Displacement, $\delta_n$', fontsize=12)

    ax2.set_yticks(jnp.arange(0.5, 0.5+u_mm_dyn.shape[1]))
    ax2.set_yticklabels(jnp.arange(u_mm_dyn.shape[1]), fontsize=12)
    ax3.set_yticks(jnp.arange(0.5, 0.5+u_mm_dyn.shape[1]))
    ax3.set_yticklabels(jnp.arange(u_mm_dyn.shape[1]), fontsize=12)

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
    #     ax3.plot(timepoints, u_mm_dyn[:, i], color=trace_colors[i], alpha=alpha, label=f'$u_{i}$')
    #     ax4.plot(timepoints, delta_mm_dyn[:, i], color=trace_colors[i], alpha=alpha, label=fr'$\delta_{i}$')

    first_truss_force_dyn = F_dyn[:, 0]
    final_truss_force_dyn = F_dyn[:, -1]
    ax4.plot(timepoints, first_truss_force_dyn, color=colors_lst[0], alpha=alpha, label='First truss')
    ax4.plot(timepoints, final_truss_force_dyn, color=colors_lst[1], alpha=alpha, label='Final truss')
    ax4.set_xlim([0, t_end])
    ax4.set_xlabel('Time (s)', fontsize=12)
    ax4.set_ylabel('Force (N)', fontsize=12)
    ax4.set_title('Endpoint Truss Forces', fontsize=12)
    ax4.legend(fontsize=8, loc='upper right')

    if sup_title is not None:
        fig.suptitle(sup_title)
        
    fig.tight_layout()
    return fig, axes


def plot_laplace_fourier(F_dyn: np.ndarray, timepoints: np.ndarray, *, laplace_sigma: float = 0.0, alpha: float = 1,
                         sup_title: str | None = None, figsize: tuple[float, float] = (8, 3)) -> tuple[plt.Figure, np.ndarray]:
    """Plot endpoint-force Fourier and Laplace transforms in a 1x2 figure.

    F_dyn contains spring forces in N, with time along rows and springs along
    columns; only the first and last columns are used. Times must be uniform.
    FFT real parts are solid and imaginary parts dotted, one color per spring.
    Laplace magnitudes use s=laplace_sigma+2j*pi*f, with sigma in 1/s.
    Show the lower half of the nonnegative frequency range in Hz without
    doubling FFT values. Return the figure and a length-two array of axes.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fft_ax, laplace_ax = axes
    for force_dyn, color, label in zip((F_dyn[:, 0], F_dyn[:, -1]), colors_lst[:2], ('First truss', 'Final truss')):
        frequencies, spectrum = helpers_builders.force_fft(timepoints, force_dyn)
        nonnegative = frequencies >= 0
        frequencies = frequencies[nonnegative]
        s_values = laplace_sigma + 2j * np.pi * frequencies
        _, laplace = helpers_builders.force_laplace(timepoints, force_dyn, s_values)
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
        ax.set_xlim(0, float(frequencies[-1]) / 4 if frequencies[-1] > 0 else 1.0)
        if ax == laplace_ax:
            ax.set_ylim([-0.005, 0.2])
        ax.legend(fontsize=8)
        ax.grid(False)

    if sup_title is not None:
        fig.suptitle(sup_title)
    fig.tight_layout()
    return fig, axes

# -------------------------
# TRAINING POST PROCESSING
# -------------------------

def plot_success_and_training_t(M_mat: np.ndarray, t_mat: np.ndarray, *, figsize: tuple[float, float] = (5, 5)) -> tuple[plt.Figure, plt.Axes]:
    """Plot training success, coloring successful transitions by training time.

    ``M_mat`` and ``t_mat`` must be equally sized square matrices. Successful
    entries (``M_mat == 1``) run from light to dark as training time increases;
    failed entries are red and diagonal entries are white.
    """
    M_mat, t_mat = np.asarray(M_mat), np.asarray(t_mat)
    if M_mat.ndim != 2 or M_mat.shape[0] != M_mat.shape[1] or M_mat.shape != t_mat.shape:
        raise ValueError("M_mat and t_mat must be equally sized square matrices.")
    n_states = M_mat.shape[0]
    n_bits = max(1, int(np.ceil(np.log2(n_states))))
    labels = [format(i, f"0{n_bits}b") for i in range(n_states)]
    diagonal = np.eye(n_states, dtype=bool)
    successful = (M_mat == 1) & ~diagonal & np.isfinite(t_mat)
    failed = (M_mat == 0) & ~diagonal
    if not np.any(successful):
        raise ValueError("M_mat contains no successful transitions with finite training times.")

    style_colors, _, _ = colors.color_scheme()
    time_cmap = LinearSegmentedColormap.from_list("training_time", [style_colors[1], style_colors[0]])
    failure_cmap = ListedColormap([style_colors[4]])
    success_times = np.ma.masked_where(~successful, t_mat)
    failures = np.ma.masked_where(~failed, np.ones_like(M_mat, dtype=float))

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(failures, cmap=failure_cmap, vmin=0, vmax=1, origin="lower")
    image = ax.imshow(success_times, cmap=time_cmap, origin="lower")
    colorbar_ax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.1)
    fig.colorbar(image, cax=colorbar_ax, label=r"Training time, $t$")
    ax.set(xlabel="Desired buckle", ylabel="Initial buckle", xticks=np.arange(n_states), yticks=np.arange(n_states), 
           xticklabels=labels, yticklabels=labels)
    ax.tick_params(axis="x", labelrotation=90)
    ax.grid(False)
    fig.tight_layout()
    return fig, ax


# -------------------------------------------------
# Potential
# -------------------------------------------------
def plot_potential(potential_params: jnp.ndarray, truss_model: str, x_min: float = -0.005, x_max: float = 0.031):
    """Plot fourth-order or Von Mises potentials from packed parameter rows."""
    potential_params = jnp.asarray(potential_params)
    if potential_params.ndim == 1:
        potential_params = potential_params[None, :]
    if potential_params.ndim != 2:
        raise ValueError("potential_params must contain one row per potential.")
    displacement = jnp.linspace(x_min, x_max, 1000)
    fig, ax = plt.subplots(figsize=(4, 3))
    if truss_model == "4th Order":
        energy = helpers_builders.potential_order4(potential_params.T, displacement[:, None])
    elif truss_model == "Trusses":
        energy = helpers_builders.bistable_potential(potential_params.T, displacement[:, None])
    else:
        raise ValueError(f"Unknown potential model: {truss_model}")
    ax.plot(displacement * 1e3, energy * 1e3)
    ax.set(xlabel="Displacement (mm)", ylabel="Energy (mJ)", title="Bistable Energy Landscape")
    fig.tight_layout()
    ax.set_ylim([-0.5, 2.5])
    return fig, ax


def detect_force_arrival(timepoints: jnp.ndarray, force_dyn: jnp.ndarray, start_time: float,
                         threshold_fraction: float = 0.05,) -> float:
    """Return the first post-input time at a fraction of peak force."""
    if not 0 < threshold_fraction <= 1:
        raise ValueError("threshold_fraction must lie in (0, 1].")
    post_input = timepoints >= start_time
    peak_force = jnp.max(jnp.where(post_input, jnp.abs(force_dyn), 0.0))
    if float(peak_force) == 0.0:
        raise ValueError("Cannot detect arrival in a zero force signal.")
    has_arrived = post_input & (jnp.abs(force_dyn) >= threshold_fraction * peak_force)
    return float(timepoints[int(jnp.argmax(has_arrived))])
