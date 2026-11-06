import os

import matplotlib.pyplot as plt  # for plotting
import numpy as np  # for doing stuff with numbers
from matplotlib.colors import LogNorm  # ratios are multiplicative -> log scale

# BEFORE RUNNING THIS, ENSURE THE PARAMETERS MATCH THE C++ RUN!!!

##########################################
# 1. parameters (must match main.cpp)
##########################################
num_sites = 12
target_site = num_sites // 2  # opposite end
phi_min = -np.pi / num_sites
phi_max = +np.pi / num_sites
tau_min = 0.50
tau_max = int(1.1 * num_sites)

gamma_2 = 0.0

# colour scale: ratio <t_noisy>/<t_noiseless>, symmetric in log space around 1
# (1/ratio_max ... ratio_max). White = same performance as without noise.
# With 'RdBu': RED  = ratio < 1 -> the noisy scenario is FASTER (noise helps)
#              BLUE = ratio > 1 -> the noisy scenario is SLOWER (noise hurts)
# (use 'RdBu_r' to swap the two colours)
cmap = 'RdBu'
ratio_max = 4

# folder containing the RESULTS_*.txt files ('.' = same folder as this script)
data_dir = '.'

# reference (denominator) = the noiseless run
reference_file = '01_RESULTS_SMEARING_NONE_5percentPHI0.026180_mean_hitting_time_PVM_gamma2_0.000000_N_12_resolution_150x600_500_runs.txt'

# panels, left to right: the noisy runs, each divided by the reference
noisy_files = [
    '02_RESULTS_SMEARING_GAUSSIAN_10percentPHI0.052360_mean_hitting_time_PVM_gamma2_0.000000_N_12_resolution_150x600_500_runs.txt',
    '03_RESULTS_SMEARING_GAUSSIAN_10percent_mean_hitting_time_PVM_gamma2_0.000000_N_12_resolution_150x600_500_runs.txt',
    '04_both_RESULTS_SMEARING_GAUSSIAN_10percentPHI0.052360_mean_hitting_time_PVM_gamma2_0.000000_N_12_resolution_150x600_500_runs.txt',
]

reference = np.loadtxt(os.path.join(data_dir, reference_file))

##########################################
# 2. side-by-side ratio maps (shared y-axis, shared colour scale, no titles)
##########################################
n_panels = len(noisy_files)
fig, axes = plt.subplots(1, n_panels, figsize=(2.4 * n_panels + 1.2, 5.0),
                         sharey=True, gridspec_kw={'wspace': 0.08})
extent = [phi_min, phi_max, tau_min, tau_max]
norm = LogNorm(vmin=1 / ratio_max, vmax=ratio_max)

ticks = [-np.pi / num_sites, 0, np.pi / num_sites]
tick_labels = [r'$-\pi/N$', r'$0$', r'$+\pi/N$']
minor_ticks = [-0.5 * np.pi / num_sites, 0.5 * np.pi / num_sites]  # unlabeled, at +-pi/(2N)

fonts = 12
for i, filename in enumerate(noisy_files):
    noisy = np.loadtxt(os.path.join(data_dir, filename))
    ratio = noisy / reference  # < 1 where the noisy scenario hits faster

    im = axes[i].imshow(
        ratio,
        extent=extent,
        origin='lower',
        aspect='auto',
        cmap=cmap,
        norm=norm,
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
cbar_ticks = [1 / ratio_max, 0.5, 1, 2, ratio_max]
cbar.set_ticks(cbar_ticks)
cbar.set_ticklabels([f'{t:g}' for t in cbar_ticks])
cbar.ax.minorticks_off()
cbar.set_label(r'$\langle n \tau \rangle \, / \, \langle n \tau \rangle_{\mathrm{no\ noise}}$',
               fontsize=fonts)

# save the combined figure as a single PDF
output_filename = (f'CPLUSPLUS_smearing_ratio_phi1_vs_tau_mean_hitting_time_PVM'
                   f'_gamma2_{gamma_2}_N_{num_sites}.pdf')
plt.savefig(output_filename, bbox_inches='tight')
print(f'Saved combined PDF as: {output_filename}')
plt.show()