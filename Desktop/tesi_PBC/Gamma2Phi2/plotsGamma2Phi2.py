import numpy as np # for doing stuff with numbers
import matplotlib.pyplot as plt # for plotting
import pandas as pd # to easily print matrices
import math

# IDEA: test different values for phi2 and gamma2 (2D GRID) for fixed size N

#####################
# PARAMETERS
#####################

# LIMITED RESOURCE
T_max = 250.0; # cutoff time (limited resource)
# n_Meas = 150; # measurements available (limited resource)

# graph's features 
# (gamma_1 = 1 because gamma_2 is in terms of gamma_1)
num_sites = 21 # size N of the graph 
tau = 1.0
phi_1 = 0.0

# targets 
target_site = f'{num_sites // 2}' # SINGLE @ opposite end
#target_site = f'{num_sites // 2}_{num_sites // 2 + 1}' # DOUBLE @ opposite end
#target_site = f'{num_sites // 2 - 1}_{num_sites // 2}_{num_sites // 2 + 1}' # TRIPLE @ opposite end
print(f'Targets: {target_site}')

# resolution
num_phi2_points = 400 # horizontal resolution
num_gamma2_points = 400 # vertical resolution

# boundaries 
phi_2_min = -np.pi
phi_2_max = +np.pi*2.
gamma_2_min = 0.0
gamma_2_max = 1.0

#####################
# LOAD DATA FROM THE C++ SIMULATION
#####################

filename_base = f'SurvProb_tau_{tau:.6f}_phi1_{phi_1:.6f}_N_{num_sites}_target_+_resolution_{num_phi2_points}x{num_gamma2_points}_Tmax_{T_max:.6f}'
# filename_base = f'SurvProb_tau_{tau:.6f}_phi1_{phi_1:.6f}_N_{num_sites}_target_+_resolution_{num_phi2_points}x{num_gamma2_points}_{n_Meas}_measurements'
survival_probabilities = np.loadtxt('RESULTS_'+filename_base+'.txt')

#####################
# MAKE THE FIGURE
#####################

plt.figure(figsize=(8,6))
extent = [phi_2_min, phi_2_max, gamma_2_min, gamma_2_max]
fonts = 12

im = plt.imshow(
    survival_probabilities, 
    extent=extent, 
    origin='lower', 
    aspect='auto', 
    cmap='managua'
)

plt.xlabel(rf'$\phi_2$', fontsize = fonts)
plt.ylabel(rf'$\gamma_2$', fontsize = fonts)
plt.title(rf'$N = {num_sites}$, $i= | + \rangle $, $\tau = {tau}$, $\phi_1$ = {phi_1}', fontsize = fonts)

ticks = [-np.pi, -0.5*np.pi, 0, 0.5*np.pi , np.pi, 2* np.pi]
tick_labels = [rf'$-\pi$', rf'$-\pi/2$', rf'$0$', rf'$+\pi/2$', rf'$+\pi$', rf'$+2\pi$']
plt.xticks(ticks, tick_labels)

cbar = plt.colorbar(im)
#cbar.set_label(rf'$S(n)$', fontsize=fonts)
cbar.set_label(rf'$S(T)$', fontsize=fonts)
PDFfilename = 'CPP_phi2_VS_gamma2_'+filename_base
plt.savefig(PDFfilename+'.pdf')
plt.show()