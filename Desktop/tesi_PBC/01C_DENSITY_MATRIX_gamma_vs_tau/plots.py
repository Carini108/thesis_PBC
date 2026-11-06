import numpy as np # for doing stuff with numbers
import matplotlib.pyplot as plt # for plotting
import pandas as pd # to easily print matrices
import math

# IDEA: test different phis and taus (2D GRID) for fixed size N

#####################
# PARAMETERS
#####################

M = 1000 # number of runs
T_max = 200.0

# graph's features
num_sites = 16 # size N of the graph 
gamma_2 = 0.0 
phi_2 = 0.0
phi_1 = 0.0 # fixed phi_1 parameter

# targets 
target_site = f'{num_sites // 2}' # SINGLE @ opposite end
#target_site = f'{num_sites // 2}_{num_sites // 2 + 1}' # DOUBLE @ opposite end
#target_site = f'{num_sites // 2 - 1}_{num_sites // 2}_{num_sites // 2 + 1}' # TRIPLE @ opposite end
print(f'Targets: {target_site}')
#relative_phase = 0.0 #for the superposition

# resolution
num_tau_points = 100 # horizontal resolution
num_gammadec_points = 100 # vertical resolution

# boundaries 
tau_min = 0.001
tau_max = 5.0
gammadec_min = 1e-4
gammadec_max = 100.0

#####################
# LOAD DATA FROM THE C++ SIMULATION
#####################

mean_hitting_times = np.loadtxt(f'notitle_RESULTS_mean_hitting_time_PVM_gamma2_{gamma_2:.6f}_phi1_{phi_1:.6f}_phi2_{phi_2:.6f}_N_{num_sites}_target_{target_site}_resolution_{num_tau_points}x{num_gammadec_points}_{M}_runs.txt')

#####################
# MAKE THE FIGURE
#####################

plt.figure(figsize=(8,6))
extent = [tau_min, tau_max, gammadec_min, gammadec_max]
fonts = 12
vmin_val = 0.0
vmax_val = 100

tau_values = np.linspace(tau_min, tau_max, num_tau_points)
gammadec_values = np.logspace(np.log10(gammadec_min), np.log10(gammadec_max), num_gammadec_points)

im = plt.pcolormesh(
    tau_values, 
    gammadec_values, 
    mean_hitting_times, 
    shading='auto', 
    cmap='viridis_r',
    vmin=vmin_val,
    vmax=vmax_val
)

plt.xlabel(rf'$\tau$', fontsize = fonts)
plt.ylabel(rf'$\gamma_{{dec}}$', fontsize = fonts)
#plt.title(rf'$N = {num_sites}$, $i= {target_site} $, $\phi_1 = {phi_1}$, $\gamma_2 = {gamma_2}$, $\phi_2 = {phi_2}$', fontsize = fonts)
plt.yscale('log')

ticks = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
tick_labels = [rf'${t:.1f}$' for t in ticks]
plt.xticks(ticks, tick_labels)

cbar = plt.colorbar(im)
cbar.set_label(rf'$\langle n\tau \rangle$', fontsize=fonts)
filename = rf'notitle_nolog_CPLUSPLUS_tau_vs_gammadec_mean_hitting_time_PVM_gamma2_{gamma_2:.6f}_phi1_{phi_1:.6f}_phi2_{phi_2:.6f}_N_{num_sites}_target_{target_site}_resolution_{num_tau_points}x{num_gammadec_points}'
plt.savefig(filename+'.pdf')