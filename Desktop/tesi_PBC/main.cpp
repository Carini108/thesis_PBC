#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <random>
#include <fstream>
#include <iomanip>
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
    // ####################################################################################
    // IDEA: mean hitting times over a 2D grid of tau vs phi_1 for a fixed size N
    // ####################################################################################

    // ##########################################
    // 1. define parameters for a qw on a line
    // ##########################################

    // test different phis and taus (2D GRID) for fixed size N
    int num_sites = 20;
    int target_site = num_sites / 2; // opposite end
    int num_phi_points = 300; // horizontal resolution
    int num_tau_points = 300; // vertical resolution
    double phi_min = -M_PI / num_sites;
    double phi_max = +M_PI / num_sites;
    double tau_min = 0.02;
    double tau_max = 4.00;
    // time evolution parameters and number of MC runs
    double T_max = 200.0; // cutoff time (limited resource)
    int M = 1000; // number of samples of the hitting time

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
    double gamma_2 = 0.3;
    //phi_1 = 0.0
    double phi_2 = 0.0;
    //phase_1 = np.exp(1j * phi_1)
    Complex phase_2 = std::polar(1.0, phi_2);

    // start constructing the grid by defining the 'checkerboard'...
    std::vector<double> phi_values(num_phi_points);
    for (int i = 0; i < num_phi_points; ++i) {
        phi_values[i] = phi_min + i * (phi_max - phi_min) / (num_phi_points - 1);
    }
    std::vector<double> tau_values(num_tau_points);
    for (int i = 0; i < num_tau_points; ++i) {
        tau_values[i] = tau_min + i * (tau_max - tau_min) / (num_tau_points - 1);
    }

    // ...and setting all values to zero
    Eigen::MatrixXd mean_hitting_times = Eigen::MatrixXd::Zero(num_tau_points, num_phi_points);

    std::cout << "Parameters initialized!\n";
    std::cout << "===================================\n";

    // OpenMP parallelization over the outer grid loop for maximum speed
    #pragma omp parallel for schedule(dynamic)
    for (int p_idx = 0; p_idx < num_phi_points; ++p_idx) { // the first variable is the index, the second the value
        
        // for reproducibility (thread-safe seeding)
        std::mt19937 gen(13 + p_idx); 
        std::uniform_real_distribution<double> dis(0.0, 1.0);

        double phi_1 = phi_values[p_idx];
        Complex phase_1 = std::polar(1.0, phi_1);

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

        #pragma omp critical
        {
            std::cout << "State at time = 0 initialized! phi_1 = " << phi_1 << "\n";
        }

        // ##########################################
        // 4. time evolution (time-indep. H) without PVM
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

            mean_hitting_times(t_idx, p_idx) = total_hitting_time / M; // mean
        }

        #pragma omp critical
        {
            std::cout << "phi_1 step number " << p_idx + 1 << "/" << num_phi_points << "\n";
        }
    }

    std::cout << "===================================\n";
    std::cout << "All simulations completed!\n";
    std::cout << "===================================\n";

    // Save grid data to file for Python plotting
    std::ofstream grid_file("mean_hitting_times_grid.txt");
    for (int i = 0; i < num_tau_points; ++i) {
        for (int j = 0; j < num_phi_points; ++j) {
            grid_file << mean_hitting_times(i, j) << (j == num_phi_points - 1 ? "" : " ");
        }
        grid_file << "\n";
    }
    grid_file.close();

    // ##########################################
    // 6. find shortest time and associated parameters
    // ##########################################

    int min_tau_idx, min_phi_idx;
    double min_time = mean_hitting_times.minCoeff(&min_tau_idx, &min_phi_idx);

    double optimal_tau_val = tau_values[min_tau_idx];
    double optimal_phi_val = phi_values[min_phi_idx];

    std::string filename = "phi1_vs_tau_mean_hitting_time_PVM_gamma2_" + 
                           std::to_string(static_cast<int>(gamma_2)) + ".0_N_" + 
                           std::to_string(num_sites) + "_resolution_" + 
                           std::to_string(num_phi_points) + "x" + std::to_string(num_tau_points) + 
                           std::to_string(M)+ "_runs";

    std::ofstream f(filename + ".txt");
    f << "Minimum mean hitting time to site " << target_site << ": " << min_time << "\n";
    f << "Number of Monte Carlo runs per point M: " << M << "\n";
    f << "Optimal parameters: \\phi_1 = " << optimal_phi_val << ", \\tau = " << optimal_tau_val << "\n";
    f.close();

    std::cout << "Data exported successfully! Run the Python script to plot.\n";
    return 0;
}