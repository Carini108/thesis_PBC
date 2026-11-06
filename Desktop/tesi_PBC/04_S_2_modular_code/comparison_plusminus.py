import numpy as np  # for doing stuff with numbers
import matplotlib.pyplot as plt  # for plotting
import os

#####################
# PARAMETERS
#####################

n_Meas = 150  # measurements available (limited resource)

# graph's features
num_sites = 21  # size N of the graph
gamma_2 = 0.0
phi_2 = 0.0

# resolution (as in the C++ output files)
num_phi_points = 140  # horizontal resolution
num_tau_points = 140  # vertical resolution

# boundaries
phi_min = -np.pi / num_sites
phi_max = +np.pi / num_sites
tau_min = 0.02
tau_max = 4.00

# the two initial states to compare (left, right)
initial_states = ['+', '-']

#####################
# LOAD DATA FROM THE C++ SIMULATION
#####################

def build_filename(sign):
    return (
        f'RESULTS_SurvProb_gamma2_{gamma_2:.6f}_phi2_{phi_2:.6f}_N_{num_sites}'
        f'_target_{sign}_resolution_{num_phi_points}x{num_tau_points}_{n_Meas}_measurements.txt'
    )

def load_matrix(sign):
    filename = build_filename(sign)
    # fallback: some systems store '+' as '_' in the filename
    if not os.path.exists(filename):
        alt = filename.replace(f'_target_{sign}_', '_target__' + '_' if sign == '+' else f'_target_{sign}_')
        if os.path.exists(alt):
            filename = alt
    return np.loadtxt(filename)

data = [load_matrix(s) for s in initial_states]

# unique color scale shared by both panels
vmin = min(d.min() for d in data)
vmax = max(d.max() for d in data)

#####################
# MAKE THE FIGURE
#####################

fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True, gridspec_kw={'wspace': 0.14})
extent = [phi_min, phi_max, tau_min, tau_max]
fonts = 12

ticks = [-np.pi / num_sites, -0.5 * np.pi / num_sites, 0, 0.5 * np.pi / num_sites, np.pi / num_sites]
tick_labels = [r'$-\pi/N$', r'$-\pi/(2N)$', r'$0$', r'$+\pi/(2N)$', r'$+\pi/N$']

for ax, survival_probabilities in zip(axes, data):
    im = ax.imshow(
        survival_probabilities,
        extent=extent,
        origin='lower',
        aspect='auto',
        cmap='magma_r',
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel(r'$\phi_1$', fontsize=fonts)
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, fontsize=9)
    # no titles, as requested

axes[0].set_ylabel(r'$\tau$', fontsize=fonts)

# single colorbar on the right
fig.subplots_adjust(right=0.88)
cbar_ax = fig.add_axes([0.91, 0.15, 0.025, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label(r'$S(n)$', fontsize=fonts)

output_filename = (
    f'CPP_phi1_VS_tau_SurvProb_gamma2_{gamma_2:.6f}_phi2_{phi_2:.6f}_N_{num_sites}'
    f'_minus_plus_comparison_resolution_{num_phi_points}x{num_tau_points}_{n_Meas}_measurements.pdf'
)
plt.savefig(output_filename, bbox_inches='tight')
print(f'Saved combined PDF as: {output_filename}')
plt.show()