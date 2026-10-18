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

// ########################################## GENERALIZED PVM (accepts a generic projector P)

void generalized_PVM(VectorXc& state, const MatrixXc& P, bool& detection_successful,
                    std::mt19937& gen, std::uniform_real_distribution<double>& dis){
    
    VectorXc projected_state = P * state; // unnormalized projected state
    double prob_success = projected_state.squaredNorm(); // get probability of success
    detection_successful = false; 

    if (dis(gen) >= prob_success) { // unsuccessful
        VectorXc collapsed_state = state - projected_state; // (I-P) |psi>
        double norm = collapsed_state.norm();

        if (norm > 0) {
            state = collapsed_state / norm;
        } else {
            state = collapsed_state;
            std::cout<<"Unphysical zero-length state!"<<std::endl;
            std::exit(1);
        }
    } else { // successful
        double norm = std::sqrt(prob_success);
        if (norm > 0) {
            state = projected_state / norm;
        } else {
            state = projected_state;
            std::cout<<"Unphysical zero-length state!"<<std::endl;
            std::exit(1);
        }
        detection_successful = true;
    }

}

// ########################################## PROJECT ON A SUBSET OF SITES (accepts the index of the non-zero entries)

MatrixXc create_subspace_projector(int num_sites, const std::vector<int>& target_sites) { 
    MatrixXc P = MatrixXc::Zero(num_sites, num_sites);
    for (int site : target_sites) {
        if (site >= 0 && site < num_sites) {
            P(site, site) = 1.0;
        } else {
            std::cout<<"Wrong indexes for the states! Cannot build projector."<<std::endl;
            std::exit(1);
        }
    }
    return P;
}

// ########################################## PROJECT ON ARBITRARY STATES (accepts a ket)

MatrixXc create_state_projector(const VectorXc& chi) {
    double numerical_tolerance = 1e-8;
    if ( abs(chi.norm()-1.0) > numerical_tolerance) {
        std::cout<<"State chi is not properly normalized! Cannot build projector."<<std::endl;
        std::exit(1);
    }
    return chi * chi.adjoint(); 
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
    int num_sites = 21;

    // ##########################################
    // construct relevant TARGETS
    // ##########################################
    // site at the opposite end
    int target_site = num_sites / 2; 
    // equal-amplitude superposition 
    VectorXc target_state = VectorXc::Zero(num_sites); // becomes chi in the function
    double relative_phase = 0.0; // to get the + superpositions
    //double relative_phase = M_PI; // to get the - superpositions
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

    // more parameters for the grid
    int num_phi_points = 150; // horizontal resolution
    int num_tau_points = 150; // vertical resolution
    double phi_min = -M_PI / num_sites;
    double phi_max = +M_PI / num_sites;
    double tau_min = 0.02;
    double tau_max = 4.00;
    // time evolution parameters and number of MC runs
    double T_max = 180.0; // cutoff time (limited resource)
    int M = 1000; // number of samples of the hitting time

    // step parameter for S_2(n)
    int num_steps = 100; 
    // grid to store S_2(n) survival probabilities
    Eigen::MatrixXd S2_grid = Eigen::MatrixXd::Zero(num_tau_points, num_phi_points);
    // non-detection projector: P_fail = I - proj_P
    MatrixXc P_fail = MatrixXc::Identity(num_sites, num_sites) - proj_P;

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
    double gamma_2 = 0.0;
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
    // goto finish;

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
            L(0, num_sites - 1) += -gamma_1 * std::conj(phase_1);
            L(num_sites - 1, 0) += -gamma_1 * phase_1;
        }
        if (num_sites > 2) {
            L(0, num_sites - 2) += -gamma_2 * std::conj(phase_2);
            L(num_sites - 2, 0) += -gamma_2 * phase_2;
            L(1, num_sites - 1) += -gamma_2 * std::conj(phase_2);
            L(num_sites - 1, 1) += -gamma_2 * phase_2;
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

            // step operator for the surviving (non-detected) subspace
            MatrixXc O_matrix = P_fail * U_tau;
            // evolve initial state n times: |psi_n> = O^n |psi_0>
            VectorXc psi_n = psi_0;
            for (int step = 0; step < num_steps; ++step) {
                psi_n = O_matrix * psi_n;
            }
            // S_2(n) is the squared norm of the surviving unnormalized state
            S2_grid(t_idx, p_idx) = psi_n.squaredNorm();
            
            goto skip;

            double total_hitting_time = 0.0;

            // MC loop to get an average estimate of the time it takes to first measure
            for (int run = 0; run < M; ++run) {
                VectorXc psi_step = psi_0;
                double time = 0.0;
                bool detected = false;

                while (!detected && time < T_max) {
                    psi_step = U_tau * psi_step; // whole tau long unitary step
                    time += tau;

// ##########################################
// HERE IS WHERE THE CODE BECOMES GENERALIZED!
                    generalized_PVM(psi_step, proj_P, detected, gen, dis);
                    // in lieu of on_site_PVM(psi_step, target_site, detected, gen, dis); 
// ##########################################

                }

                total_hitting_time += time;
            }

            mean_hitting_times(t_idx, p_idx) = total_hitting_time / M; // mean

            skip:
        }

        #pragma omp critical
        {
            std::cout << "phi_1 step number " << p_idx + 1 << "/" << num_phi_points << "\n";
        }
    }

    // finish: 

    std::cout << "===================================\n";
    std::cout << "All simulations completed!\n";
    std::cout << "===================================\n";

    std::string filename_S2 = "RESULTS_S2_n" + std::to_string(num_steps) 
                            + "_gamma2_" + std::to_string(gamma_2) 
                            + "_N_" + std::to_string(num_sites) 
                            + "_RELphase_" + std::to_string(relative_phase) 
                            + "_resolution_" + std::to_string(num_phi_points) 
                            + "x" + std::to_string(num_tau_points) + ".txt";

    std::ofstream grid_file_S2(filename_S2);
    for (int i = 0; i < num_tau_points; ++i) {
        for (int j = 0; j < num_phi_points; ++j) {
            grid_file_S2 << S2_grid(i, j) << (j == num_phi_points - 1 ? "" : " ");
        }
        grid_file_S2 << "\n";
    }
    grid_file_S2.close();

    std::string filename_results = "RESULTS_mean_hitting_time_PVM_gamma2_" + 
                           std::to_string(gamma_2) 
                           + "_N_" + std::to_string(num_sites) 
                           + "_RELphase_" + std::to_string(relative_phase) 
                           + "_resolution_" + std::to_string(num_phi_points) 
                           + "x" + std::to_string(num_tau_points) + "_" +
                           std::to_string(M)+ "_runs.txt";

    // Save grid data to file for Python plotting
    std::ofstream grid_file(filename_results);
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

    std::string filename_optimal_values = "phi1_vs_tau_mean_hitting_time_PVM_gamma2_" + 
                           std::to_string(gamma_2) 
                           + "_N_" + std::to_string(num_sites) 
                           + "_RELphase_" + std::to_string(relative_phase) 
                           + "_resolution_" + std::to_string(num_phi_points) 
                           + "x" + std::to_string(num_tau_points) 
                           + "_" + std::to_string(M)+ "_runs";

    std::ofstream f(filename_optimal_values + ".txt");
    f << "Minimum mean hitting time to site " << target_site << ": " << min_time << "\n";
    f << "Number of Monte Carlo runs per point M: " << M << "\n";
    f << "Optimal parameters: \\phi_1 = " << optimal_phi_val << ", \\tau = " << optimal_tau_val << "\n";
    f.close();

    std::cout << "Data exported successfully! Run the Python script to plot.\n";
    return 0;
}