import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Simulation Parameters
num_sites = 20
M = 1500

# 1. Load Metadata
with open("RESULTS_3D_metadata.txt", "r") as f:
    lines = f.readlines()
    num_phi1, phi1_min, phi1_max = map(float, lines[0].split())
    num_gamma2, gamma2_min, gamma2_max = map(float, lines[1].split())
    num_tau, tau_min, tau_max = map(float, lines[2].split())

num_phi1, num_gamma2, num_tau = int(num_phi1), int(num_gamma2), int(num_tau)

# Create coordinate vectors
phi1_vals = np.linspace(phi1_min, phi1_max, num_phi1)
gamma2_vals = np.linspace(gamma2_min, gamma2_max, num_gamma2)
tau_vals = np.linspace(tau_min, tau_max, num_tau)

# 2. Load Binary Data & Reshape to 3D Tensor (phi_1, gamma_2, tau)
filename_results = f"RESULTS_mean_hitting_time_PVM_N_{num_sites}_resolution_{num_phi1}x{num_gamma2}x{num_tau}_{M}_runs.bin"
data = np.fromfile(filename_results, dtype=np.float64)
data = data.reshape((num_phi1, num_gamma2, num_tau))

# 3. Setup Interactive Plotting Window
fig, ax = plt.subplots(figsize=(9, 7))
plt.subplots_adjust(bottom=0.20)

initial_g_idx = num_gamma2 // 2

# Slice along gamma_2: data[:, g_idx, :] yields (phi_1 vs tau) heatmap
slice_data = data[:, initial_g_idx, :].T  # Transpose so tau is Y-axis and phi_1 is X-axis

im = ax.imshow(
    slice_data,
    origin='lower',
    aspect='auto',
    extent=[phi1_vals[0], phi1_vals[-1], tau_vals[0], tau_vals[-1]],
    cmap='viridis_r'
)

cbar = fig.colorbar(im, ax=ax)
cbar.set_label('mean detection time', rotation=270, labelpad=15)

ax.set_title(f"$\gamma_2 = {gamma2_vals[initial_g_idx]:.3f}$")
ax.set_xlabel(r"$\phi_1$")
ax.set_ylabel(r"$\tau$")

# 4. Add Interactive Slider for Gamma_2
slider_ax = plt.axes([0.20, 0.05, 0.65, 0.03])
gamma2_slider = Slider(
    ax=slider_ax,
    label=r'$\gamma_2$ Index',
    valmin=0,
    valmax=num_gamma2 - 1,
    valinit=initial_g_idx,
    valfmt='%d'
)

# Slider update callback function
def update(val):
    g_idx = int(gamma2_slider.val)
    new_slice = data[:, g_idx, :].T
    im.set_data(new_slice)
    im.set_clim(vmin=new_slice.min(), vmax=new_slice.max())
    ax.set_title(f"$\gamma_2 = {gamma2_vals[g_idx]:.3f}$")
    fig.canvas.draw_idle()

gamma2_slider.on_changed(update)

plt.show()