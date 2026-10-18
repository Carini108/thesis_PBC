import numpy as np # for doing stuff with numbers
import matplotlib.pyplot as plt # for plotting
import pandas as pd # to easily print matrices
import math

# IDEA: test different phis and taus (2D GRID) for fixed size N

#####################
# PARAMETERS
#####################

# LIMITED RESOURCE
T_max = 540.0; # cutoff time (limited resource)
# n_Meas = 150; # measurements available (limited resource)

# graph's features
num_sites = 21 # size N of the graph 
gamma_2 = 0.0 
phi_2 = 0.0

# targets 
target_site = f'{num_sites // 2}' # SINGLE @ opposite end
#target_site = f'{num_sites // 2}_{num_sites // 2 + 1}' # DOUBLE @ opposite end
#target_site = f'{num_sites // 2 - 1}_{num_sites // 2}_{num_sites // 2 + 1}' # TRIPLE @ opposite end
print(f'Targets: {target_site}')

# resolution
num_phi_points = 800 # horizontal resolution
num_tau_points = 800 # vertical resolution

# boundaries 
phi_min = -np.pi/num_sites
phi_max = +np.pi/num_sites
tau_min = 0.02
tau_max = 4.00

#####################
# LOAD DATA FROM THE C++ SIMULATION
#####################

filename_base = f'SurvProb_gamma2_{gamma_2:.6f}_phi2_{phi_2:.6f}_N_{num_sites}_target_+_resolution_{num_phi_points}x{num_tau_points}_Tmax_{T_max:.6f}'
# filename_base = f'SurvProb_gamma2_{gamma_2:.6f}_phi2_{phi_2:.6f}_N_{num_sites}_target_+_resolution_{num_phi_points}x{num_tau_points}_{n_Meas}_measurements'
survival_probabilities = np.loadtxt('RESULTS_'+filename_base+'.txt')

#####################
# MAKE THE FIGURE
#####################

plt.figure(figsize=(8,6))
extent = [phi_min, phi_max, tau_min, tau_max]
fonts = 12

im = plt.imshow(
    survival_probabilities, 
    extent=extent, 
    origin='lower', 
    aspect='auto', 
    cmap='magma_r'
)

plt.xlabel(rf'$\phi_1$', fontsize = fonts)
plt.ylabel(rf'$\tau$', fontsize = fonts)
plt.title(rf'$N = {num_sites}$, $i= | + \rangle $, $\gamma_2 = {gamma_2}$, $\phi_2$ = {phi_2}', fontsize = fonts)

ticks = [-np.pi/num_sites, -0.5*np.pi/num_sites, 0, 0.5*np.pi/num_sites , np.pi/num_sites]
tick_labels = [rf'$-\pi/N$', rf'$-\pi/(2N)$', rf'$0$', rf'$+\pi/(2N)$', rf'$+\pi/N$']
plt.xticks(ticks, tick_labels)

cbar = plt.colorbar(im)
cbar.set_label(rf'$S(n)$', fontsize=fonts)
PDFfilename = 'CPP_phi1_VS_tau_'+filename_base
plt.savefig(PDFfilename+'.pdf')
plt.show()