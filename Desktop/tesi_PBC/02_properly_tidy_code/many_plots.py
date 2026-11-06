import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend: no windows are ever opened
import matplotlib.pyplot as plt

#####################
# PARAMETERS
#####################

sizes = [13, 19, 14, 18, 35, 36, 23, 24]

M = 1000          # number of runs
T_max = 200.0

gamma_2 = 0.0
phi_2 = 0.0

# resolution
num_phi_points = 200
num_tau_points = 100

# tau boundaries (do not depend on N)
tau_min = 0.02
tau_max = 1.45

fonts = 12

#####################
# LOOP OVER SIZES
#####################

for num_sites in sizes:

    # target
    target_site = f'{num_sites // 2}'  # SINGLE @ opposite end
    #target_site = f'{num_sites // 2}_{num_sites // 2 + 1}'  # DOUBLE
    #target_site = f'{num_sites // 2 - 1}_{num_sites // 2}_{num_sites // 2 + 1}'  # TRIPLE
    print(f'N = {num_sites}, targets: {target_site}')

    # phi boundaries (depend on N)
    phi_min = -np.pi / num_sites
    phi_max = +np.pi / num_sites

    # load data
    data_file = (f'RESULTS_mean_hitting_time_PVM_gamma2_{gamma_2:.6f}_phi2_{phi_2:.6f}'
                 f'_N_{num_sites}_target_{target_site}'
                 f'_resolution_{num_phi_points}x{num_tau_points}_{M}_runs.txt')
    try:
        mean_hitting_times = np.loadtxt(data_file)
    except OSError:
        print(f'  -> file not found, skipping: {data_file}')
        continue

    # figure
    fig, ax = plt.subplots(figsize=(8, 6))
    extent = [phi_min, phi_max, tau_min, tau_max]

    im = ax.imshow(
        mean_hitting_times,
        extent=extent,
        origin='lower',
        aspect='auto',
        cmap='viridis_r',
        vmin=tau_min,
        vmax=T_max
    )

    ax.set_xlabel(r'$\phi_1$', fontsize=fonts)
    ax.set_ylabel(r'$\tau$', fontsize=fonts)
    ax.set_title(rf'$N = {num_sites}$, $i = {target_site}$, $\gamma_2 = {gamma_2}$, $\phi_2 = {phi_2}$',
                 fontsize=fonts)

    ticks = [-np.pi/num_sites, -0.5*np.pi/num_sites, 0, 0.5*np.pi/num_sites, np.pi/num_sites]
    tick_labels = [r'$-\pi/N$', r'$-\pi/(2N)$', r'$0$', r'$+\pi/(2N)$', r'$+\pi/N$']
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r'$\langle n\tau \rangle$', fontsize=fonts)

    filename = (f'CPLUSPLUS_phi1_vs_tau_mean_hitting_time_PVM_gamma2_{gamma_2:.6f}'
                f'_phi2_{phi_2:.6f}_N_{num_sites}_target_{target_site}'
                f'_resolution_{num_phi_points}x{num_tau_points}')
    fig.savefig(filename + '.pdf')
    plt.close(fig)  # free memory, nothing is displayed