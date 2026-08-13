#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <random>
#include <fstream>
#include <iomanip>
#include <chrono> // Added for time measurement
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

using Complex = std::complex<double>;
using MatrixXc = Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic>;
using VectorXc = Eigen::Matrix<Complex, Eigen::Dynamic, 1>;

// ########################################## PROJECTIVE MEASUREMENT FUNCTION (PVM)

void on_site_PVM(VectorXc& state, int target_site, bool& detection_successful, 
                std::mt19937& gen, std::uniform_real_distribution<double>& dis) {
    
    // prepare probability to extract outcome and 'blank' state to write the result
    double prob_success = std::norm(state(target_site)); // this is |<i|psi>|^2
    VectorXc collapsed_state = VectorXc::Zero(state.size());
    detection_successful = false;

    if (dis(gen) >= prob_success) {
        // print(rf"Walker is NOT at site {target_site}.")
        // projecting on the complementary subspace
        collapsed_state = state;
        collapsed_state(target_site) = 0.0;
        // renormalize
        double norm = collapsed_state.norm();
        if (norm > 0) {
            collapsed_state = collapsed_state / norm;
        }
    } else {
        //print(rf"Success! Walker at site {target_site}.")
        collapsed_state(target_site) = state(target_site) / std::sqrt(prob_success);
        detection_successful = true;
    }

    state = collapsed_state;
}

// ####################################################################################
// ####################################################################################
// #################################################################################### MAIN CODE
// ####################################################################################
// ####################################################################################

int main() {
    // Start timing execution
    auto start_time = std::chrono::high_resolution_clock::now();

    // Disable Eigen's internal multi-threading so OpenMP manages parallelization efficiently
    Eigen::setNbThreads(1);

    // ####################################################################################
    // IDEA: mean hitting times over a 3D grid of phi_1 vs gamma_2 vs tau for a fixed size N
    // ####################################################################################

    // ##########################################
    // 1. define parameters for a qw on a line
    // ##########################################

    int num_sites = 20;
    int target_site = num_sites / 2; // opposite end
    int num_phi1_points = 56; // horizontal resolution
    int num_gamma2_points = 100; // gamma_2 steps (to be scrolled...)
    int num_tau_points = 100; // vertical resolution
    double phi1_min = 0.0;
    double phi1_max = M_PI / num_sites;
    double gamma2_min = 0.0;
    double gamma2_max = 1.0;
    double tau_min = 0.02;
    double tau_max = 4.00;
    // time evolution parameters and number of MC runs
    double T_max = 200.0; // cutoff time (limited resource)
    int M = 1500; // number of samples of the hitting time

    std::cout << "-----------------------------------\n";
    std::cout << "Number of sites N = " << num_sites << "\n";
    std::cout << "Maximum evolution time T = " << T_max << "\n";
    std::cout << "Monte Carlo runs per grid point M = " << M << "\n";
    std::cout << "-----------------------------------\n";
    std::cout << "In this code we perform a PVM once every tau!\n";
    std::cout << "-----------------------------------\n";

    // couplings and constants relevant to H
    double on_site_energy = 0.0;
    double gamma = 1.0; // hopping rate
    double gamma_1 = 1.0;
    double phi_2 = 0.0;
    Complex phase_2 = std::polar(1.0, phi_2);

    // start constructing the grid by defining the 'checkerboard'...
    std::vector<double> phi1_values(num_phi1_points);
    for (int i = 0; i < num_phi1_points; ++i) {
        phi1_values[i] = phi1_min + i * (phi1_max - phi1_min) / (num_phi1_points - 1);
    }
    std::vector<double> gamma2_values(num_gamma2_points);
    for (int i = 0; i < num_gamma2_points; ++i) {
        gamma2_values[i] = gamma2_min + i * (gamma2_max - gamma2_min) / (num_gamma2_points - 1);
    }
    std::vector<double> tau_values(num_tau_points);
    for (int i = 0; i < num_tau_points; ++i) {
        tau_values[i] = tau_min + i * (tau_max - tau_min) / (num_tau_points - 1);
    }

    // ...and setting all values to zero (1D array layout for 3D grid: [p_idx][g_idx][t_idx])
    std::vector<double> mean_hitting_times_3d((size_t)num_phi1_points * num_gamma2_points * num_tau_points, 0.0);

    std::cout << "Parameters initialized!\n";
    std::cout << "===================================\n";
    // goto finish;

    // OpenMP parallelization over the outer grid loop for maximum speed
    #pragma omp parallel for schedule(dynamic)
    for (int p_idx = 0; p_idx < num_phi1_points; ++p_idx) {
        
        double phi_1 = phi1_values[p_idx];
        Complex phase_1 = std::polar(1.0, phi_1);

        for (int g_idx = 0; g_idx < num_gamma2_points; ++g_idx) {

            // for reproducibility (thread-safe seeding)
            std::mt19937 gen(13 + p_idx * num_gamma2_points + g_idx); 
            std::uniform_real_distribution<double> dis(0.0, 1.0);

            double gamma_2 = gamma2_values[g_idx];

            // ##########################################
            // 2. Laplacian matrix for a ring (L = D - A)
            // ##########################################

            MatrixXc L = MatrixXc::Zero(num_sites, num_sites);

            // on-site energies & first/second neighbor hoppings
            for (int i = 0; i < num_sites; ++i) {
                L(i, i) = 2.0 * on_site_energy;
                if (i < num_sites - 1) {
                    L(i, i + 1) += -gamma_1 * phase_1;
                    L(i + 1, i) += -gamma_1 * std::conj(phase_1);
                }
                if (i < num_sites - 2) {
                    L(i, i + 2) += -gamma_2 * phase_2;
                    L(i + 2, i) += -gamma_2 * std::conj(phase_2);
                }
            }

            // add periodic boundary conditions (it becomes a ring...)
            if (num_sites > 1) {
                L(0, num_sites - 1) += -gamma_1;
                L(num_sites - 1, 0) += -gamma_1;
            }
            if (num_sites > 2) {
                L(0, num_sites - 2) += -gamma_2;
                L(num_sites - 2, 0) += -gamma_2;
                L(1, num_sites - 1) += -gamma_2;
                L(num_sites - 1, 1) += -gamma_2;
            }

            // ##########################################
            // 3. initialization of the state
            // ##########################################

            VectorXc psi_0 = VectorXc::Zero(num_sites);
            psi_0(0) = 1.0; // start at node 0

            // ##########################################
            // 4. time evolution (time-indep. H) with PVM
            // ##########################################

            MatrixXc H = gamma * L;

            for (int t_idx = 0; t_idx < num_tau_points; ++t_idx) {
                double tau = tau_values[t_idx];

                // coherent unitary evolution step 
                MatrixXc arg = -Complex(0.0, 1.0) * H * tau;
                MatrixXc U_tau = arg.exp(); 
                
                double total_hitting_time = 0.0;

                // MC loop to get an average estimate of the time it takes to first measure
                for (int run = 0; run < M; ++run) {
                    VectorXc psi_step = psi_0;
                    double time = 0.0;
                    bool detected = false;

                    while (!detected && time < T_max) {
                        psi_step = U_tau * psi_step; // whole tau long unitary step
                        time += tau;
                        on_site_PVM(psi_step, target_site, detected, gen, dis); // attempt to measure
                    }

                    total_hitting_time += time;
                }

                size_t flat_idx = (size_t)p_idx * num_gamma2_points * num_tau_points 
                                + (size_t)g_idx * num_tau_points 
                                + (size_t)t_idx; // take the correct index in the cubic grating

                mean_hitting_times_3d[flat_idx] = total_hitting_time / M; // mean
            }
        }

        #pragma omp critical
        {
            std::cout << "phi_1 step number " << p_idx + 1 << "/" << num_phi1_points << " completed.\n";
        }
    }

    // ##########################################
    // 5. finish: 
    // ##########################################

    std::cout << "===================================\n";
    std::cout << "All simulations completed!\n";
    std::cout << "===================================\n";

    std::string filename_results = "RESULTS_mean_hitting_time_PVM_N_" + 
                           std::to_string(num_sites) + "_resolution_" + 
                           std::to_string(num_phi1_points) + "x" + 
                           std::to_string(num_gamma2_points) + "x" + 
                           std::to_string(num_tau_points) + "_" + 
                           std::to_string(M)+ "_runs.bin";

    // save grid data to file for Python plotting 
    std::ofstream grid_file(filename_results, std::ios::binary); 
    grid_file.write(reinterpret_cast<const char*>(mean_hitting_times_3d.data()), mean_hitting_times_3d.size() * sizeof(double));
    grid_file.close();

    // export grid parameter metadata to a text file for Python to read
    std::ofstream meta("RESULTS_3D_metadata.txt");
    meta << num_phi1_points << " " << phi1_min << " " << phi1_max << "\n";
    meta << num_gamma2_points << " " << gamma2_min << " " << gamma2_max << "\n";
    meta << num_tau_points << " " << tau_min << " " << tau_max << "\n";
    meta.close();

    // ##########################################
    // 6. find shortest time and associated parameters
    // ##########################################

    size_t min_idx = 0;
    double min_time = mean_hitting_times_3d[0];

    for (size_t i = 1; i < mean_hitting_times_3d.size(); ++i) {
        if (mean_hitting_times_3d[i] < min_time) {
            min_time = mean_hitting_times_3d[i];
            min_idx = i;
        }
    }

    // calculate the index in the 3d space
    size_t min_p_idx = min_idx / (num_gamma2_points * num_tau_points);
    size_t rem = min_idx % (num_gamma2_points * num_tau_points);
    size_t min_g_idx = rem / num_tau_points;
    size_t min_t_idx = rem % num_tau_points;

    double optimal_phi1_val = phi1_values[min_p_idx];
    double optimal_gamma2_val = gamma2_values[min_g_idx];
    double optimal_tau_val = tau_values[min_t_idx];

    std::string filename = "RESULTS_phi1_vs_gamma2_vs_tau_mean_hitting_time_PVM_N_" + 
                           std::to_string(num_sites) + "_resolution_" + 
                           std::to_string(num_phi1_points) + "x" + 
                           std::to_string(num_gamma2_points) + "x" + 
                           std::to_string(num_tau_points) + "_" + 
                           std::to_string(M)+ "_runs.txt";

    std::ofstream f(filename);
    f << "Minimum mean hitting time to site " << target_site << ": " << min_time << "\n";
    f << "Number of Monte Carlo runs per point M: " << M << "\n";
    f << "Optimal parameters: \\phi_1 = " << optimal_phi1_val << ", \\gamma_2 = " << optimal_gamma2_val << ", \\tau = " << optimal_tau_val << "\n";
    f.close();

    // Stop timing execution and calculate total elapsed time
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    std::cout << "Data exported successfully! Run the Python script to plot.\n";
    std::cout << "Total execution time: " << std::fixed << std::setprecision(2) << elapsed.count() << " seconds.\n";
    
    return 0;
}