import os

import matplotlib.pyplot as plt  # for plotting
import numpy as np  # for doing stuff with numbers

# BEFORE RUNNING THIS, ENSURE THE PARAMETERS MATCH THE C++ RUN!!!

##########################################
# 1. parameters (must match main.cpp)
##########################################
num_sites = 12
target_site = num_sites // 2  # opposite end
num_phi_points = 150  # horizontal resolution
num_tau_points = int(50 * num_sites)  # vertical resolution
phi_min = -np.pi / num_sites
phi_max = +np.pi / num_sites
tau_min = 0.50
tau_max = int(1.1 * num_sites)
T_max = 230

gamma_2 = 0.0
M = 500

# one fixed colour range for ALL panels (so they are directly comparable)
vmin_val = 0
vmax_val = T_max

# folder containing the RESULTS_*.txt files ('.' = same folder as this script)
data_dir = '.'

# the panels, left to right: the exact filenames of the RESULTS files
# (the 01_, 02_, ... prefixes set the order; edit/reorder/remove freely)
files = [
    '01_RESULTS_SMEARING_NONE_5percentPHI0.026180_mean_hitting_time_PVM_gamma2_0.000000_N_12_resolution_150x600_500_runs.txt',
    '02_RESULTS_SMEARING_GAUSSIAN_10percentPHI0.052360_mean_hitting_time_PVM_gamma2_0.000000_N_12_resolution_150x600_500_runs.txt',
    '03_RESULTS_SMEARING_GAUSSIAN_10percent_mean_hitting_time_PVM_gamma2_0.000000_N_12_resolution_150x600_500_runs.txt',
    '04_both_RESULTS_SMEARING_GAUSSIAN_10percentPHI0.052360_mean_hitting_time_PVM_gamma2_0.000000_N_12_resolution_150x600_500_runs.txt',
]


##########################################
# 2. side-by-side heatmaps (shared y-axis, shared colour scale, no titles)
##########################################
n_panels = len(files)
fig, axes = plt.subplots(1, n_panels, figsize=(2.4 * n_panels + 1.2, 6.0),
                         sharey=True, gridspec_kw={'wspace': 0.06})
extent = [phi_min, phi_max, tau_min, tau_max]

ticks = [-np.pi / num_sites, 0, np.pi / num_sites]
tick_labels = [r'$-\pi/N$', r'$0$', r'$+\pi/N$']
minor_ticks = [-0.5 * np.pi / num_sites, 0.5 * np.pi / num_sites]  # unlabeled, at +-pi/(2N)

fonts = 12
for i, filename in enumerate(files):
    mean_hitting_times = np.loadtxt(os.path.join(data_dir, filename))

    im = axes[i].imshow(
        mean_hitting_times,
        extent=extent,
        origin='lower',
        aspect='auto',
        cmap='viridis_r',
        vmin=vmin_val,
        vmax=vmax_val,
    )

    axes[i].set_xlabel(r'$\phi_1$', fontsize=fonts)
    if i == 0:
        axes[i].set_ylabel(r'$\tau$', fontsize=fonts)

    axes[i].set_xticks(ticks)
    axes[i].set_xticklabels(tick_labels, fontsize=10)
    axes[i].set_xticks(minor_ticks, minor=True)
    # keep the outermost labels inside their own panel so they never collide
    # with the neighbouring panel's labels
    axes[i].get_xticklabels()[0].set_ha('left')
    axes[i].get_xticklabels()[-1].set_ha('right')

# single colourbar on the right
fig.subplots_adjust(right=0.89)
cbar_ax = fig.add_axes([0.915, 0.15, 0.013, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label(r'$\langle n \tau \rangle$', fontsize=fonts)

# save the combined figure as a single PDF
output_filename = (f'CPLUSPLUS_smearing_comparison_phi1_vs_tau_mean_hitting_time_PVM'
                   f'_gamma2_{gamma_2}_N_{num_sites}.pdf')
plt.savefig(output_filename, bbox_inches='tight')
print(f'Saved combined PDF as: {output_filename}')
plt.show()