#include "first_detection.hpp"
#include <string>
#include <fstream>
#include <iostream>
#include <chrono>
#include <limits>
#include <unsupported/Eigen/MatrixFunctions>

// IDEA: mean hitting times shown over a 2D grid 
// of (noise strength) vs tau for fixed size N
//
// Two noise models can be run (switch NoiseModel below):
//   INTRINSIC_DECOHERENCE : vertical axis = gamma_dec  (Milburn, closed-form step in the energy basis)
//   QSW                   : vertical axis = p, the mixing parameter of the quantum stochastic walk
//                           d rho/dt = -(1-p) i[H,rho] + p sum_ij ( L_ij rho L_ij^+ - 1/2 {L_ij^+ L_ij, rho} )
//                           p = 0 -> coherent quantum walk,  p = 1 -> classical random walk

enum class NoiseModel { INTRINSIC_DECOHERENCE, QSW };

int main() {

    auto start_time = std::chrono::high_resolution_clock::now();

    // #####################
    // (0) CHOOSE THE MODEL AND THE SOLVER
    // #####################

    const NoiseModel model = NoiseModel::QSW;
    const bool is_qsw = (model == NoiseModel::QSW);
    // QSW only:
    //   true  -> EXACT mean hitting time from the deterministic 'no click' recursion: no sampling noise and
    //            roughly M times cheaper than Monte Carlo (M is irrelevant in this case)
    //   false -> Monte Carlo with M samples (slow: use it to cross-check a few grid points)
    const bool use_exact_solver = true;

    std::vector<int> site_sizes = { 16 };
    for (int varying_size : site_sizes) {

        // #####################
        // (1) PARAMETERS
        // #####################
        
        // number of sites
        int num_sites = varying_size;
        // grid resolution
        int n_pixels = 60;
        int num_tau_points = n_pixels; // horizontal resolution
        int num_noise_points = n_pixels; // vertical resolution (log-spaced)
        // grid boundaries
        double tau_min = 0.001;
        double tau_max = 5.0;
        // tau axis spacing: true = logarithmic (as before); false = linear, e.g. tau_min = 1.3, tau_max = 5 with
        // ~400 points to resolve the narrow dark-state resonances, which a coarse log grid undersamples
        const bool log_tau_axis = false;
        double noise_min = 1e-3;                  // gamma_dec_min  or  p_min
        double noise_max = is_qsw ? 1.0 : 10.0;   // gamma_dec_max  or  p_max (p = 1 is the classical random walk)
        // QSW: one extra row p = 0 (exactly coherent walk) placed BELOW the log-spaced rows
        const bool include_zero_noise_row = is_qsw;
        // fixed phi_1 parameter
        double phi_1 = 0.0;
        Complex phase_1 = std::polar(1.0, phi_1);
        // time evolution parameters and number of MC runs
        double T_max = 200.0; // cutoff time (limited resource)
        int M = 2500; // number of samples for the average hitting time (not used by the exact QSW solver)
        // couplings and constants relevant to H
        double on_site_energy = 0.0;
        double gamma = 1.; // hopping rate
        double gamma_1 = 1.;
        double gamma_2 = 0.;
        double phi_2 = 0.;
        // double phi_2 = +M_PI / num_sites;
        Complex phase_2 = std::polar(1.0, phi_2);
        // QSW only: rate of the incoherent hops (per bond, per direction) at p = 1.
        // With gamma_inc = gamma the p = 1 row is exactly the classical random walk with the same hopping rate.
        double gamma_inc = gamma;

        // #####################
        // (2) CONSTRUCT TARGETS
        // #####################

        // local measurement on m-dimensional subspace 
        const std::vector<int> target_sites = {num_sites/2}; 
        // const std::vector<int> target_sites = {num_sites/2 -1 , num_sites/2, num_sites/2 + 1}; 
        MatrixXc proj_P = create_subspace_projector(num_sites, target_sites);
        
        /*
        // equal-amplitude superposition 
        VectorXc target_state = VectorXc::Zero(num_sites); // becomes chi in the function
        // double relative_phase = 0.0; // to get the + superpositions
        double relative_phase = M_PI; // to get the - superpositions
        Complex exp_rel_phase = std::polar(1.0, relative_phase);
        target_state(num_sites/2) = 1.0;
        target_state(num_sites/2 + 1) = 1.0 * exp_rel_phase;
        double norm = target_state.norm();
        if (norm > 0) {
                target_state = target_state / norm;
            } else {
                std::cout<<"Unphysical zero-length state!"<<std::endl;
                std::exit(1);
            }
        MatrixXc proj_P = create_state_projector(target_state);
        */

        // #####################
        // (2.1) DYNAMICS: everything that does NOT depend on the grid indices is built once
        // #####################

        // LAPLACIAN MATRIX for a ring (L = D - A) - we kill the (regular) diagonal
        MatrixXc L = build_Laplacian(num_sites, on_site_energy, gamma_1, gamma_2, phase_1, phase_2);
        MatrixXc H = gamma * L;

        // INITIALISATION of the state
        VectorXc psi_0 = VectorXc::Zero(num_sites);
        psi_0(0) = 1.0; // start at node 0
        MatrixXc rho_0 = psi_0 * psi_0.adjoint(); // | 0 > < 0 |

        // intrinsic decoherence: diagonalise H to work in the energy basis
        MatrixXc V;
        Eigen::VectorXd energies;
        if (!is_qsw) {
            Eigen::SelfAdjointEigenSolver<MatrixXc> eigensolver(H);
            V = eigensolver.eigenvectors();
            energies = eigensolver.eigenvalues();
        }

        // QSW: incoherent hops j -> i along every bond of the graph of H, L_ij = sqrt(gamma_inc) |i><j|
        std::vector<MatrixXc> jump_ops;
        if (is_qsw) jump_ops = build_jump_operators_from_graph(H, gamma_inc);

        // #####################
        // (3) PRINT PARAMETERS
        // #####################

        std::cout << "-----------------------------------\n";
        std::cout << "Model: " << (is_qsw ? "quantum stochastic walk (QSW), noise axis = p" 
                                          : "intrinsic decoherence (Milburn), noise axis = gamma_dec") << "\n";
        if (is_qsw) {
            std::cout << "Solver: " << (use_exact_solver ? "EXACT (deterministic no-click recursion)" : "Monte Carlo") << "\n";
            std::cout << "gamma_inc = " << gamma_inc << ", number of jump operators = " << jump_ops.size() << "\n";
        }
        std::cout << "Number of sites N = " << num_sites << "\n";
        std::cout << "Fixed phi_1 = " << phi_1 << "\n";
        std::cout << "Maximum evolution time T = " << T_max << "\n";
        if (!(is_qsw && use_exact_solver)) std::cout << "Monte Carlo runs per grid point M = " << M << "\n";
        std::cout << "Resolution = " << num_tau_points << "x" << num_noise_points 
                  << (include_zero_noise_row ? " (+1 row at p = 0)" : "") << "\n";
        std::cout << "On-site energies (diagonal) = " << on_site_energy << "\n";
        std::cout << "gamma_1 = " << gamma_1 << ", gamma_2 = " << gamma_2 << "\n";
        std::cout << "tau varies, phi_2 = " << phi_2 << "\n";
        std::cout << "In this code we perform a PVM once every tau!\n";
        std::cout << "Projector:\n" << proj_P << "\n"; // print to check...
        std::cout << "-----------------------------------\n";

        // #####################
        // (4) BUILD EMPTY GRID
        // #####################

        // start constructing the grid by defining the 'checkerboard'...
        std::vector<double> tau_values(num_tau_points);
        for (int i = 0; i < num_tau_points; ++i) {
            const double frac = static_cast<double>(i) / static_cast<double>(num_tau_points - 1);
            tau_values[i] = log_tau_axis ? tau_min * std::pow(tau_max / tau_min, frac) 
                                         : tau_min + (tau_max - tau_min) * frac;
        }
        std::vector<double> noise_values; // gamma_dec or p
        if (include_zero_noise_row) noise_values.push_back(0.0);
        for (int i = 0; i < num_noise_points; ++i)
            noise_values.push_back(noise_min * std::pow(noise_max / noise_min, static_cast<double>(i) / static_cast<double>(num_noise_points - 1)));
        const int num_rows = static_cast<int>(noise_values.size());

        // ...and setting all values to zero
        Eigen::MatrixXd mean_hitting_times = Eigen::MatrixXd::Zero(num_rows, num_tau_points);
        // fraction of runs NOT detected within T_max (the clean dark-state indicator); QSW only
        Eigen::MatrixXd undetected_fraction = Eigen::MatrixXd::Zero(num_rows, num_tau_points);

        std::cout << "All parameters initialised! Simulation starting...\n";
        std::cout << "===================================\n";

        // #####################
        // (5) OpenMP PARALLELISATION
        // #####################

        #pragma omp parallel for schedule(dynamic)
        for (int g_idx = 0; g_idx < num_rows; ++g_idx) { 
            
            const double noise = noise_values[g_idx]; // gamma_dec or p

            // QSW: the Liouvillian depends on p only, so it is built once per row
            MatrixXc Lsup;
            if (is_qsw) Lsup = build_QSW_superoperator(H, jump_ops, noise);

            for (int t_idx = 0; t_idx < num_tau_points; ++t_idx) {

                // for reproducibility (thread-safe seeding)
                std::mt19937 gen(13 + g_idx * num_tau_points + t_idx); 
                std::uniform_real_distribution<double> dis(0.0, 1.0);

                // run over all values 
                double tau = tau_values[t_idx];

                // #####################
                // (5.1) DYNAMICS + MEASUREMENT
                // #####################
                HittingStats stats;

                if (is_qsw) {
                    // one step of duration tau = fixed N^2 x N^2 matrix exp(tau * Liouvillian)
                    MatrixXc prop = build_propagator(Lsup, tau);
                    if (use_exact_solver) {
                        stats = exact_hitting_time_propagator(T_max, tau, prop, rho_0, proj_P);
                    } else {
                        stats = run_MC_hitting_times_dens_mat_propagator(M, T_max, tau, prop, rho_0, proj_P, gen, dis);
                    }
                } else {
                    // average over M Monte Carlo runs (closed-form step in the energy basis)
                    stats.mean_time = run_MC_hitting_times_dens_mat_intrinsic_decoherence(M, T_max, tau, V, energies, noise, rho_0, proj_P, gen, dis); 
                    stats.undetected_fraction = std::numeric_limits<double>::quiet_NaN(); // not tracked in this branch
                }
                mean_hitting_times(g_idx, t_idx) = stats.mean_time; 
                undetected_fraction(g_idx, t_idx) = stats.undetected_fraction;
            }

            #pragma omp critical
            {
                std::cout << "noise step number " << g_idx + 1 << "/" << num_rows << "\n";
            }
        }
        
        // #####################
        // (6) FINISHING UP
        // #####################
        
        std::cout << "===================================\n";
        std::cout << "All simulations completed!\n";
        std::cout << "===================================\n";

        // #####################
        // (6.1) filenaming
        // #####################

        // get your ducks in a row: collect targets in a string 
        
        std::string target_sites_str = "";
        for (size_t i = 0; i < target_sites.size(); ++i) {
            target_sites_str += std::to_string(target_sites[i]);
            if (i < target_sites.size() - 1) target_sites_str += "_"; // underscore for spacing
        }
        std::cout << "Targets: " << target_sites_str << std::endl;

        // (Milburn file names are exactly the ones of the previous version)
        const std::string stat_tag = (is_qsw && use_exact_solver) ? std::string("exact") : std::to_string(M) + "_runs";
        std::string base_filename = std::string("mean_hitting_time_") + (is_qsw ? "QSW_" : "") + "PVM_gamma2_" + std::to_string(gamma_2) + 
                                    "_phi1_" + std::to_string(phi_1) + 
                                    "_phi2_" + std::to_string(phi_2) + 
                                    (is_qsw ? "_gammainc_" + std::to_string(gamma_inc) : std::string("")) + 
                                    "_N_" + std::to_string(num_sites) + 
                                    "_target_" + target_sites_str + 
                                    "_resolution_" + std::to_string(num_tau_points) + 
                                    "x" + std::to_string(num_noise_points) + 
                                    (log_tau_axis ? std::string("") : "_linTau_" + std::to_string(tau_min) + "_" + std::to_string(tau_max)) + 
                                    "_" + stat_tag;
        
        // branch filenaming 
        std::string filename_results = "RESULTS_" + base_filename + ".txt";
        std::string filename_axes = "AXES_" + base_filename + ".txt";
        std::string filename_undetected = "UNDETECTED_" + base_filename + ".txt";
        std::string filename_optimal_values = std::string("tau_vs_") + (is_qsw ? "p_" : "gammadec_") + base_filename + ".txt";

        // #####################
        // (6.2) saving 2D grid
        // #####################

        // save grid data to file for Python plotting (mean_hit... is a MatrixXd!)
        std::ofstream grid_file(filename_results);
        grid_file.precision(10);
        for (int i = 0; i < num_rows; ++i) {
            for (int j = 0; j < num_tau_points; ++j) {
                grid_file << mean_hitting_times(i, j) << (j == num_tau_points - 1 ? "" : " "); // separate data at the end of the line
            }
            grid_file << "\n";
        }
        grid_file.close();

        // axes: first line = tau values, second line = noise values (p or gamma_dec; the first one is 0 if the p = 0 row is included)
        std::ofstream axes_file(filename_axes);
        axes_file.precision(12);
        for (int j = 0; j < num_tau_points; ++j) axes_file << tau_values[j] << (j == num_tau_points - 1 ? "" : " ");
        axes_file << "\n";
        for (int i = 0; i < num_rows; ++i) axes_file << noise_values[i] << (i == num_rows - 1 ? "" : " ");
        axes_file << "\n";
        axes_file.close();

        // fraction of undetected runs at T_max (QSW only)
        if (is_qsw) {
            std::ofstream undetected_file(filename_undetected);
            undetected_file.precision(10);
            for (int i = 0; i < num_rows; ++i) {
                for (int j = 0; j < num_tau_points; ++j) {
                    undetected_file << undetected_fraction(i, j) << (j == num_tau_points - 1 ? "" : " ");
                }
                undetected_file << "\n";
            }
            undetected_file.close();
        }

        // #####################
        // (6.3) optimization: looking for the min
        // #####################

        int min_noise_idx, min_tau_idx;
        double min_time = mean_hitting_times.minCoeff(&min_noise_idx, &min_tau_idx);

        double optimal_noise_val = noise_values[min_noise_idx];
        double optimal_tau_val = tau_values[min_tau_idx];

        // the optimum is only meaningful if it is NOT on the edge of the scanned region
        // (QSW: p = 0 [extra row] and p = 1 are physical limits, not artificial edges of the grid)
        const bool lower_noise_edge_is_artificial = !include_zero_noise_row;
        const bool upper_noise_edge_is_artificial = !(is_qsw && noise_max >= 1.0);
        const bool noise_on_edge = (min_noise_idx == 0 && lower_noise_edge_is_artificial) 
                                || (min_noise_idx == num_rows - 1 && upper_noise_edge_is_artificial);
        const bool tau_on_edge = (min_tau_idx == 0) || (min_tau_idx == num_tau_points - 1);

        // #####################
        // (6.4) saving optimal values
        // #####################

        std::ofstream f(filename_optimal_values);
        f << "Minimum mean hitting time: " << min_time << "\n";
        if (is_qsw && use_exact_solver) f << "Exact solver (no Monte Carlo)\n";
        else f << "Number of Monte Carlo runs per point M: " << M << "\n";
        f << "Optimal parameters: \\tau = " << optimal_tau_val << ", " << (is_qsw ? "p" : "\\gamma_dec") << " = " << optimal_noise_val << "\n";
        if (noise_on_edge || tau_on_edge) 
            f << "WARNING: the minimum lies on the edge of the scanned region -> extend the grid before quoting it as 'optimal'.\n";
        f.close();
    }

    std::cout << "===================================\n";
    std::cout << "Data exported successfully! Run the Python script to plot.\n";
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;
    std::cout << "Elapsed time: " << elapsed_seconds.count() << " seconds.\n";
    std::cout << "===================================\n";

    return 0;
}