from __future__ import annotations

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import MaxNLocator

plt.rcParams["pdf.fonttype"] = 42

def read_M_and_t(M_file, t_file):
    M_mat = np.loadtxt(M_file, delimiter=",")
    t_mat = np.loadtxt(t_file, delimiter=",")
    if M_mat.shape != t_mat.shape:
        raise ValueError(f"Success and training-time matrices must have the same shape; got {M_mat.shape} and {t_mat.shape}.")
    return M_mat, t_mat