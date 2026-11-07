import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ------------------------------------------------------------------ parameters
TAU_MIN, TAU_MAX, G_MIN, G_MAX = 0.01, 5.0, 1e-3, 10.0
GRID_FILE = "notitle_RESULTS_mean_hitting_time_PVM_gamma2_0.000000_phi1_0.000000_phi2_0.000000_N_16_target_8_resolution_100x100_1000_runs.txt"

# ------------------------------------------------------------------ data loading
print("loading grid from:", GRID_FILE)
Tgrid = np.loadtxt(GRID_FILE)
NG, NTAU = Tgrid.shape

# Linearly distributed tau values, log-spaced gamma_dec values matched to dimensions
tau_g = np.linspace(TAU_MIN, TAU_MAX, NTAU)
g_g = G_MIN * (G_MAX / G_MIN) ** (np.arange(NG) / (NG - 1))

# Normalise with the bottom row of pixels (Tgrid[0, :])
logR = np.log10(Tgrid[0, :] / Tgrid)

# ------------------------------------------------------------------ figure (Panel D only)
fig, d = plt.subplots(figsize=(8, 6))
im = d.pcolormesh(tau_g, g_g, logR, shading="auto", cmap="RdBu", norm=TwoSlopeNorm(0, -0.7, 0.7))
d.contour(tau_g, g_g, logR, levels=[0], colors="k", linewidths=1.6)
d.set(yscale="log", xlabel=r"$\tau$", ylabel=r"$\gamma_{dec}$")
d.set_title(r"$\log_{10}(\langle n\tau\rangle_{\text{bottom}} / \langle n\tau\rangle)$: blue = faster than bottom row", fontsize=11)
fig.colorbar(im, ax=d)

fig.tight_layout()
fig.savefig("noise_advantage_panel_d.pdf", dpi=150)
print("saved noise_advantage_panel_d.pdf")