from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import MaxNLocator

plt.rcParams["pdf.fonttype"] = 42


# -----------------
# IMPORT
# -----------------
def read_M_and_t(M_file, t_file):
    M_mat = np.loadtxt(M_file, delimiter=",")
    t_mat = np.loadtxt(t_file, delimiter=",")
    if M_mat.shape != t_mat.shape:
        raise ValueError(f"Success and training-time matrices must have the same shape; got {M_mat.shape} and {t_mat.shape}.")
    return M_mat, t_mat


# -----------------
# EXPORT
# ----------------
def export_importants_single_run(u_dyn: ArrayLike, delta_dyn: ArrayLike, F_dyn: ArrayLike, timepoints: ArrayLike, impulse_dyn: ArrayLike,
                                 initial_state: str, final_state: str, output_dir: str | Path = "single_run_exports") -> Path:
    """Export the main time-dependent quantities from one impulse simulation to CSV."""
    timepoints_arr = np.asarray(timepoints, dtype=float)
    impulse_arr = np.asarray(impulse_dyn, dtype=float)
    if timepoints_arr.ndim != 1 or impulse_arr.shape != timepoints_arr.shape:
        raise ValueError("timepoints and impulse_dyn must be one-dimensional arrays with the same shape.")

    named_dynamics = (("u", "m", u_dyn), ("delta", "m", delta_dyn), ("F", "N", F_dyn))
    matrices, column_names = [timepoints_arr[:, None], impulse_arr[:, None]], ["time_s", "impulse_m"]
    for quantity, unit, values in named_dynamics:
        matrix = np.asarray(values, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix[:, None]
        if matrix.ndim != 2 or matrix.shape[0] != timepoints_arr.size:
            raise ValueError(f"{quantity}_dyn must have one row per timepoint; got shape {matrix.shape}.")
        matrices.append(matrix)
        column_names.extend(f"{quantity}_{index}_{unit}" for index in range(1, matrix.shape[1] + 1))

    if not initial_state or not final_state or any(bit not in "01" for bit in initial_state + final_state):
        raise ValueError("initial_state and final_state must be nonempty binary strings.")
    export_dir = Path(output_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    export_path = export_dir / f"single_impulse_initial_{initial_state}_final_{final_state}.csv"
    np.savetxt(export_path, np.column_stack(matrices), delimiter=",", header=",".join(column_names), comments="")
    return export_path
