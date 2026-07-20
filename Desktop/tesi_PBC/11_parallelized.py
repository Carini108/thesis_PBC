import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import concurrent.futures
import time
import os

# ==============================================================================
# PARAMETRI DI CONFIGURAZIONE
# ==============================================================================
NUM_SITES = 20
TARGET_SITE = NUM_SITES // 2
NUM_PHI_POINTS = 250
NUM_TAU_POINTS = 250
PHI_MIN = -np.pi / NUM_SITES
PHI_MAX = +np.pi / NUM_SITES
TAU_MIN = 0.02
TAU_MAX = 4.00
T_MAX = 200
M = 1000
SEED_BASE = 13

# Costanti fisiche
GAMMA = 1.0
GAMMA_1 = 1.0
GAMMA_2 = 0.5
PHI_2 = 0.0
PHASE_2 = np.exp(1j * PHI_2)

# ==============================================================================
# FUNZIONE WORKER (Per il parallelismo)
# ==============================================================================
def simulate_phi_column(args):
    """
    Esegue la simulazione per una singola colonna di phi.
    Questa funzione viene eseguita in parallelo.
    """
    p_idx, phi_1, tau_values, seed = args
    
    # RNG locale per thread-safety
    rng = np.random.default_rng(seed)
    
    # Costruzione Hamiltonian
    phase_1 = np.exp(1j * phi_1)
    
    # Creazione L
    diag = np.zeros((NUM_SITES, NUM_SITES), dtype=complex)
    # Hop primo vicino
    L = -GAMMA_1 * (np.eye(NUM_SITES, k=1) * phase_1 + np.eye(NUM_SITES, k=-1) * np.conjugate(phase_1))
    # Hop secondo vicino
    L += -GAMMA_2 * (np.eye(NUM_SITES, k=2) * PHASE_2 + np.eye(NUM_SITES, k=-2) * np.conjugate(PHASE_2))
    # PBC
    L += -GAMMA_1 * (np.eye(NUM_SITES, k=(NUM_SITES-1)) + np.eye(NUM_SITES, k=-(NUM_SITES-1)))
    L += -GAMMA_2 * (np.eye(NUM_SITES, k=(NUM_SITES-2)) + np.eye(NUM_SITES, k=-(NUM_SITES-2)))
    
    H = GAMMA * L
    psi_0 = np.zeros(NUM_SITES, dtype=complex)
    psi_0[0] = 1.0
    
    column_results = np.zeros(len(tau_values))
    
    for t_idx, tau in enumerate(tau_values):
        U_tau = expm(-1j * H * tau)
        total_hitting_time = 0.0
        
        for _ in range(M):
            psi_step = psi_0.copy()
            time_elapsed = 0.0
            detected = False
            
            while (not detected) and time_elapsed < T_MAX:
                psi_step = U_tau @ psi_step # Operatore @ è più veloce di dot
                time_elapsed += tau
                
                # Inlined PVM (massima velocità)
                prob_success = np.abs(psi_step[TARGET_SITE]) ** 2
                if rng.random() >= prob_success:
                    psi_step[TARGET_SITE] = 0.0
                    norm = np.linalg.norm(psi_step)
                    if norm > 0:
                        psi_step /= norm
                else:
                    detected = True
            
            total_hitting_time += time_elapsed
            
        column_results[t_idx] = total_hitting_time / M
        
    return p_idx, column_results

# ==============================================================================
# MAIN (Punto di ingresso)
# ==============================================================================
if __name__ == '__main__':
    print(f"--- Avvio Simulazione (N={NUM_SITES}, M={M}) ---")
    
    phi_values = np.linspace(PHI_MIN, PHI_MAX, NUM_PHI_POINTS)
    tau_values = np.linspace(TAU_MIN, TAU_MAX, NUM_TAU_POINTS)
    mean_hitting_times = np.zeros((NUM_TAU_POINTS, NUM_PHI_POINTS))
    
    # Preparazione task per i worker
    tasks = [(i, phi_values[i], tau_values, SEED_BASE + i) for i in range(NUM_PHI_POINTS)]
    
    start_time = time.time()
    
    # Esecuzione parallela
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(simulate_phi_column, tasks))
    
    # Recomposizione risultati
    for p_idx, col_data in results:
        mean_hitting_times[:, p_idx] = col_data
        
    end_time = time.time()
    print(f"Simulazione completata in {end_time - start_time:.2f} secondi.")
    
    # Salvataggio dati
    filename = f'data_N{NUM_SITES}_M{M}'
    np.save(filename + '.npy', mean_hitting_times)
    
    # Plotting
    plt.figure(figsize=(8,6))
    im = plt.imshow(mean_hitting_times, extent=[PHI_MIN, PHI_MAX, TAU_MIN, TAU_MAX],
                    origin='lower', aspect='auto', cmap='viridis_r')
    plt.colorbar(im, label='Mean Hitting Time')
    plt.xlabel(r'$\phi_1$')
    plt.ylabel(r'$\tau$')
    plt.title(f'N = {NUM_SITES}, gamma_2 = {GAMMA_2}')
    plt.savefig(filename + '.pdf', dpi=300)
    print(f"Dati e grafico salvati come {filename}.*")
    plt.show()