import numpy as np # for doing stuff with numbers
import matplotlib.pyplot as plt # for plotting
import pandas as pd # to easily print matrices
import math

# IDEA: test different phis and taus (2D GRID) for fixed size N

#####################
# PARAMETERS
#####################

M = 500 # number of runs
T_max = 200.0

# graph's features
num_sites = 11 # size N of the graph 
gamma_2 = 0.0 
phi_2 = 0.0
tau = 1.3 # fixed tau parameter

# targets 
target_site = f'{num_sites // 2}' # SINGLE @ opposite end
#target_site = f'{num_sites // 2}_{num_sites // 2 + 1}' # DOUBLE @ opposite end
#target_site = f'{num_sites // 2 - 1}_{num_sites // 2}_{num_sites // 2 + 1}' # TRIPLE @ opposite end
print(f'Targets: {target_site}')
#relative_phase = 0.0 #for the superposition

# resolution
num_phi_points = 150 # horizontal resolution
num_gammadec_points = 150 # vertical resolution

# boundaries 
phi_min = -np.pi/num_sites
phi_max = +np.pi/num_sites
gammadec_min = 1e-15
gammadec_max = 1.0

#####################
# LOAD DATA FROM THE C++ SIMULATION
#####################

mean_hitting_times = np.loadtxt(f'RESULTS_mean_hitting_time_PVM_gamma2_{gamma_2:.6f}_tau_{tau:.6f}_phi2_{phi_2:.6f}_N_{num_sites}_target_{target_site}_resolution_{num_phi_points}x{num_gammadec_points}_{M}_runs.txt')

#####################
# MAKE THE FIGURE
#####################

plt.figure(figsize=(8,6))
extent = [phi_min, phi_max, gammadec_min, gammadec_max]
fonts = 12
vmin_val = 0.0
vmax_val = T_max

phi_values = np.linspace(phi_min, phi_max, num_phi_points)
gammadec_values = np.logspace(np.log10(gammadec_min), np.log10(gammadec_max), num_gammadec_points)

im = plt.pcolormesh(
    phi_values, 
    gammadec_values, 
    mean_hitting_times, 
    shading='auto', 
    cmap='plasma_r',
    vmin=vmin_val,
    vmax=vmax_val
)

plt.xlabel(rf'$\phi_1$', fontsize = fonts)
plt.ylabel(rf'$\gamma_{{dec}}$', fontsize = fonts)
plt.title(rf'$N = {num_sites}$, $i= {target_site} $, $\tau = {tau}$, $\gamma_2 = {gamma_2}$, $\phi_2$ = {phi_2}', fontsize = fonts)
plt.yscale('log')

ticks = [-np.pi/num_sites, -0.5*np.pi/num_sites, 0, 0.5*np.pi/num_sites , np.pi/num_sites]
tick_labels = [rf'$-\pi/N$', rf'$-\pi/(2N)$', rf'$0$', rf'$+\pi/(2N)$', rf'$+\pi/N$']
plt.xticks(ticks, tick_labels)

cbar = plt.colorbar(im)
cbar.set_label(rf'$\langle n\tau \rangle$', fontsize=fonts)
filename = rf'CPLUSPLUS_phi1_vs_gammadec_mean_hitting_time_PVM_gamma2_{gamma_2:.6f}_tau_{tau:.6f}_phi2_{phi_2:.6f}_N_{num_sites}_target_{target_site}_resolution_{num_phi_points}x{num_gammadec_points}'
plt.savefig(filename+'.pdf')