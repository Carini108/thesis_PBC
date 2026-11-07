import numpy as np # for doing stuff with numbers
import matplotlib.pyplot as plt # for plotting
from matplotlib.colors import TwoSlopeNorm, LogNorm

# IDEA: look at the grid written by main.cpp with model = QSW (quantum stochastic walk).
#       p = 0 row -> coherent quantum walk;  p = 1 row -> classical random walk (rate gamma_inc per bond).
#       So the classical baseline is just the last row of the grid: nothing else has to be computed.

#####################
# PARAMETERS (same as in main.cpp)
#####################

n_pixels = 60

num_sites = 16
gamma_2 = 0.0
phi_1 = 0.0
phi_2 = 0.0
gamma_inc = 1.0
target_site = f'{num_sites // 2}'      # SINGLE @ opposite end

num_tau_points = n_pixels     # horizontal resolution
num_noise_points = n_pixels   # vertical resolution (the extra p = 0 row is NOT counted here, as in the file name)
log_tau_axis = False          # as in main.cpp (False = linearly spaced tau)
tau_min, tau_max = 0.001, 5.0 # only used in the file name when log_tau_axis = False
exact = True                  # True: exact solver; False: Monte Carlo -> set M
M = 2500

# p values drawn in panel (a), besides the two references p = 0 (black dashed) and p = 1 (black dotted)
p_show = [0.004, 0.025, 0.25, 0.5]
# colour-blind-safe (Okabe-Ito); this 4-subset was chosen to stay distinguishable under
# deuteranopia, protanopia and tritanopia (simulated, minimum pairwise Delta E ~ 21 in CIELAB)
colors = ['#0072B2', '#56B4E9', '#D55E00', '#CC79A7']   # blue, sky blue, vermillion, reddish purple

stat_tag = 'exact' if exact else f'{M}_runs'
lin_tag = '' if log_tau_axis else f'_linTau_{tau_min:.6f}_{tau_max:.6f}'
base = (f'mean_hitting_time_QSW_PVM_gamma2_{gamma_2:.6f}_phi1_{phi_1:.6f}_phi2_{phi_2:.6f}_gammainc_{gamma_inc:.6f}'
        f'_N_{num_sites}_target_{target_site}_resolution_{num_tau_points}x{num_noise_points}{lin_tag}_{stat_tag}')

#####################
# LOAD DATA FROM THE C++ SIMULATION
#####################

T = np.loadtxt(f'RESULTS_{base}.txt')            # <n tau>: rows = p, columns = tau
S = np.loadtxt(f'UNDETECTED_{base}.txt')         # fraction of runs not detected within T_max
with open(f'AXES_{base}.txt') as f:
    tau = np.array(f.readline().split(), dtype=float)
    p = np.array(f.readline().split(), dtype=float)

has_p0 = p[0] == 0.0                              # exactly coherent row
has_p1 = np.isclose(p[-1], 1.0)                   # classical row
pos = np.where(p > 0)[0]                          # rows that can live on a log axis
if not has_p1:
    raise SystemExit('the last row must be p = 1 (classical walk): use p_max = 1 in main.cpp')
classical = T[-1]                                 # <n tau> of the classical random walk, as a function of tau
tau_is_log = bool(np.allclose(np.diff(np.log(tau)), np.diff(np.log(tau))[0], rtol=1e-3))
tau_scale = 'log' if tau_is_log else 'linear'

fonts = 11
fig = plt.figure(figsize=(13, 9.5))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05])

#####################
# (a) cuts at fixed p:  <n tau> vs tau
#####################

a = fig.add_subplot(gs[0, :])
shown_rows = []
for pv, c in zip(p_show, colors):
    i = pos[:-1][np.argmin(np.abs(np.log(p[pos[:-1]]) - np.log(pv)))]      # nearest row with 0 < p < 1
    shown_rows.append(i)
    a.plot(tau, T[i], color=c, lw=1.6, zorder=2, label=rf'$p={p[i]:.3g}$')
if has_p0:
    a.plot(tau, T[0], color='k', ls='--', lw=1.2, zorder=3, label=r'$p=0$ (coherent)')
a.plot(tau, classical, color='k', ls=':', lw=1.8, zorder=3, label=r'$p=1$ (classical)')
a.set(xlabel=r'$\tau$', ylabel=r'$\langle n\tau\rangle$', xlim=(tau[0], tau[-1]), ylim=(0, 150), xscale=tau_scale)
a.set_title(r'(a) cuts at fixed $p$', fontsize=fonts)
a.legend(fontsize=9, ncol=3, loc='upper left')

#####################
# (b) dark-state indicator: fraction of runs never detected within T_max
#####################

b = fig.add_subplot(gs[1, 0])
im_b = b.pcolormesh(tau, p[pos], np.clip(S[pos], 1e-4, 1.0), shading='auto', cmap='magma_r',
                    norm=LogNorm(1e-4, 1.0), rasterized=True)
b.set(xscale=tau_scale, yscale='log', xlabel=r'$\tau$', ylabel=r'$p$')
b.set_title(r'(b) $S(T_{max})$: fraction never detected', fontsize=fonts)
fig.colorbar(im_b, ax=b)

#####################
# (c) advantage over the classical walk (the p = 1 row)
#####################

c_ = fig.add_subplot(gs[1, 1])
logR = np.log10(classical[None, :] / T[pos])
im_c = c_.pcolormesh(tau, p[pos], logR, shading='auto', cmap='RdBu', norm=TwoSlopeNorm(0, -0.7, 0.7), rasterized=True)
if len(pos) > 2:
    c_.contour(tau, p[pos], logR, levels=[0], colors='k', linewidths=1.6)
c_.set(xscale=tau_scale, yscale='log', xlabel=r'$\tau$', ylabel=r'$p$')
c_.set_title(r'(c) $\log_{10}(\langle n\tau\rangle_{p=1}/\langle n\tau\rangle)$: blue = faster than classical', fontsize=fonts)
fig.colorbar(im_c, ax=c_)

fig.tight_layout()
filename = f'CPLUSPLUS_QSW_{base}'
fig.savefig(filename + '.pdf')
fig.savefig(filename + '.png', dpi=150)

#####################
# QUICK NUMBERS (only the rows drawn in panel (a))
#####################

rows = ([0] if has_p0 else []) + shown_rows + [len(p) - 1]
print(f'{"p":>8} {"min_tau <nt>":>13} {"tau_opt":>8}')
for i in rows:
    j = T[i].argmin()
    print(f'{p[i]:8.4f} {T[i, j]:13.2f} {tau[j]:8.3f}')