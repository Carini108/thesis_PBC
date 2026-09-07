import numpy as np # for doing stuff with numbers
import matplotlib.pyplot as plt # for plotting
import pandas as pd # to easily print matrices
import math

# IDEA: test different phis and taus (2D GRID) for fixed size N

#####################
# PARAMETERS
#####################

M = 500 # number of runs

# graph's features
num_sites = 20 # size N of the graph 
gamma_2 = 0.0

# targets 
target_site = f'{num_sites // 2+1}' # SINGLE @ opposite end
#target_site = f'{num_sites // 2}_{num_sites // 2 + 1}' # DOUBLE @ opposite end
#target_site = f'{num_sites // 2 - 1}_{num_sites // 2}_{num_sites // 2 + 1}' # TRIPLE @ opposite end
print(f'Targets: {target_site}')
#relative_phase = 0.0 #for the superposition

# resolution
num_phi_points = 150 # horizontal resolution
num_tau_points = 150 # vertical resolution

#boundaries 
phi_min = -np.pi/num_sites
phi_max = +np.pi/num_sites
tau_min = 0.02
tau_max = 4.00

#####################
# LOAD DATA FROM THE C++ SIMULATION
#####################

mean_hitting_times = np.loadtxt(f'RESULTS_mean_hitting_time_PVM_gamma2_{gamma_2:.6f}_N_{num_sites}_target_{target_site}_resolution_{num_phi_points}x{num_tau_points}_{M}_runs.txt')

#####################
# MAKE THE FIGURE
#####################

plt.figure(figsize=(8,6))
extent = [phi_min, phi_max, tau_min, tau_max]
fonts = 12

im = plt.imshow(
    mean_hitting_times, 
    extent=extent, 
    origin='lower', 
    aspect='auto', 
    cmap='viridis_r'
)

plt.xlabel(rf'$\phi_1$', fontsize = fonts)
plt.ylabel(rf'$\tau$', fontsize = fonts)
plt.title(rf'$N = {num_sites}$, $i={target_site}$, $\gamma_2 = {gamma_2}$', fontsize = fonts)

ticks = [-np.pi/num_sites, -0.5*np.pi/num_sites, 0, 0.5*np.pi/num_sites , np.pi/num_sites]
tick_labels = [rf'$-\pi/N$', rf'$-\pi/(2N)$', rf'$0$', rf'$+\pi/(2N)$', rf'$+\pi/N$']
plt.xticks(ticks, tick_labels)

cbar = plt.colorbar(im)
cbar.set_label(rf'$\langle n\tau \rangle$', fontsize=fonts)
filename = rf'CPLUSPLUS_phi1_vs_tau_mean_hitting_time_PVM_gamma2_{gamma_2}_N_{num_sites}_target_{target_site}_resolution_{num_phi_points}x{num_tau_points}'
plt.savefig(filename+'.pdf')
plt.show()