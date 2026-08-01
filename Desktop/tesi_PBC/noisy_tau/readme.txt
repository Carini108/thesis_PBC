================================ GAUSSIAN SMEARING 0.25 ================================

    // SMEARING parameters
    SmearingType smearing_mode = SmearingType::GAUSSIAN; // Choose: NONE, UNIFORM, or GAUSSIAN
    double smearing_error = 0.25; // Absolute error (half-width for uniform, standard deviation for gaussian)

    // test different phis and taus (2D GRID) for fixed size N
    int num_sites = 20;
    int target_site = num_sites / 2; // opposite end
    int num_phi_points = 170; // horizontal resolution
    int num_tau_points = 170; // vertical resolution
    double phi_min = -M_PI / num_sites;
    double phi_max = +M_PI / num_sites;
    double tau_min = 0.60;
    double tau_max = 3.20;
    // time evolution parameters and number of MC runs
    double T_max = 200.0; // cutoff time (limited resource)
    int M = 2000; // number of samples of the hitting time

    // couplings and constants relevant to H
    double on_site_energy = 0.0;
    double gamma = 1.0; // hopping rate
    double gamma_1 = 1.0;
    double gamma_2 = 0.0;
    //phi_1 = 0.0
    double phi_2 = 0.0;


(cwq) stefanodavidecarini@MacBook-Air-di-Stefano noisy_tau % g++ -O3 -std=c++17 -Xpreprocessor -fopenmp -I$(brew --prefix eigen)/include/eigen3 -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp main.cpp -o qw_sim_noisy_tau && ./qw_sim_noisy_tau
-----------------------------------
We assume tau is NOISY! (experimental fluctuations...)
-----------------------------------
Number of sites N = 20
Maximum evolution time T = 200
Monte Carlo runs per grid point M = 2000
-----------------------------------
In this code we perform a PVM once every tau!
Data exported successfully! Run the Python script to plot.
Total execution time: 10384.8 seconds.


================================  ================================