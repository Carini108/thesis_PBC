#include "first_detection.hpp"
#include <string>
#include <fstream>
#include <iostream>
#include <chrono>
#include <unsupported/Eigen/MatrixFunctions>

// IDEA: mean hitting times shown over a 2D grid 
// of tau vs phi_1 for fixed size N

int main() {

    auto start_time = std::chrono::high_resolution_clock::now();

    // #####################
    // (1) PARAMETERS
    // #####################
    
    // number of sites
    int num_sites = 21;
    // grid resolution
    int num_phi_points = 140; // horizontal resolution
    int num_tau_points = 140; // vertical resolution
    // grid boundaries
    double phi_min = -M_PI / num_sites;
    double phi_max = +M_PI / num_sites;
    double tau_min = 0.02;
    double tau_max = 4.00;
    // time evolution parameters and number of MC runs
    //double T_max = 540.0; // cutoff time (limited resource)
    int n_Meas = 150; // measurements available (limited resource)
    // couplings and constants relevant to H
    double on_site_energy = 0.0;
    double gamma = 1.; // hopping rate
    double gamma_1 = 1.;
    double gamma_2 = 0.;
    double phi_2 = 0.;
    // double phi_2 = +M_PI / num_sites;
    Complex phase_2 = std::polar(1.0, phi_2);

    // #####################
    // (2) CONSTRUCT TARGETS
    // #####################

    // #####################
    // (2.1) DEFINITION of the subspaces for the projection
    // #####################

    /*
    // local measurement on m-dimensional subspace 
    const std::vector<int> target_sites = {num_sites/2}; 
    // const std::vector<int> target_sites = {num_sites/2 -1 , num_sites/2, num_sites/2 + 1}; 
    MatrixXc proj_P = create_subspace_projector(num_sites, target_sites);
    */

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
    
    // #####################
    // (2.2) INITIALISATION of the state and of the projectors
    // #####################

    VectorXc psi_0 = VectorXc::Zero(num_sites);
    psi_0(0) = 1.0; // start at node 0
    MatrixXc proj_P = create_state_projector(target_state);
    MatrixXc complementary_proj = MatrixXc::Identity(psi_0.size(), psi_0.size()) - proj_P; // I - P

    // #####################
    // (3) PRINT PARAMETERS
    // #####################

    std::cout << "-----------------------------------\n";
    std::cout << "Number of sites N = " << num_sites << "\n";
    // std::cout << "Maximum evolution time T = " << T_max << "\n";
    std::cout << "Maximum number of measurements n = " << n_Meas << "\n";
    std::cout << "Resolution = " << num_phi_points << "x" << num_tau_points << "\n";
    std::cout << "On-site energies (diagonal) = " << on_site_energy << "\n";
    std::cout << "gamma_1 = " << gamma_1 << ", gamma_2 = " << gamma_2 << "\n";
    std::cout << "phi_1 varies, phi_2 = " << phi_2 << "\n";
    std::cout << "In this code we DO NOT simulate a PVM once every tau! This is a deterministic calculation of survival probabilites. There is no averaging going on.\n";
    std::cout << "Projector:\n" << proj_P << "\n"; // print to check...
    std::cout << "-----------------------------------\n";

    // #####################
    // (4) BUILD EMPTY GRID
    // #####################

    // start constructing the grid by defining the 'checkerboard'...
    std::vector<double> phi_values(num_phi_points);
    for (int i = 0; i < num_phi_points; ++i)
        phi_values[i] = phi_min + i * (phi_max - phi_min) / (num_phi_points - 1);
    std::vector<double> tau_values(num_tau_points);
    for (int i = 0; i < num_tau_points; ++i)
        tau_values[i] = tau_min + i * (tau_max - tau_min) / (num_tau_points - 1);

    // ...and setting all values to zero
    Eigen::MatrixXd Survival_Probs = Eigen::MatrixXd::Zero(num_tau_points, num_phi_points);

    std::cout << "All parameters initialized! Calculation starting...\n";
    std::cout << "===================================\n";

    // #####################
    // (5) OpenMP PARALLELISATION
    // #####################

    #pragma omp parallel for schedule(dynamic)
    for (int p_idx = 0; p_idx < num_phi_points; ++p_idx) { 
        
        // for reproducibility (thread-safe seeding)
        std::mt19937 gen(13 + p_idx); 
        std::uniform_real_distribution<double> dis(0.0, 1.0);

        // run over all values 
        double phi_1 = phi_values[p_idx];
        Complex phase_1 = std::polar(1.0, phi_1);

        // #####################
        // (5.1) LAPLACIAN MATRIX for a ring (L = D - A) - we kill the (regular) diagonal
        // #####################

        MatrixXc L = build_Laplacian(num_sites, on_site_energy, gamma_1, gamma_2, phase_1, phase_2);

        #pragma omp critical
        {
            std::cout << "State at time = 0 initialized! phi_1 = " << phi_1 << "\n";
        }

        // #####################
        // (5.2) DYNAMICS
        // #####################
        MatrixXc H = gamma * L;

        for (int t_idx = 0; t_idx < num_tau_points; ++t_idx) {

            // "central" value of tau for this pixel
            double tau = tau_values[t_idx];

            // coherent unitary evolution step, defined once and for all
            MatrixXc arg = -Complex(0.0, 1.0) * H * tau;
            MatrixXc U_tau = arg.exp(); 
            
            // NO average! This is just || O^n |\psi_0> ||^2 
            // double survival_probability = get_Surv_Prob_fixed_Tmax(T_max, tau, U_tau, psi_0, complementary_proj);
            double survival_probability = get_Surv_Prob_fixed_nMeasurements(n_Meas, tau, U_tau, psi_0, complementary_proj);
            Survival_Probs(t_idx, p_idx) = survival_probability; 

        }

        #pragma omp critical
        {
            std::cout << "phi_1 step number " << p_idx + 1 << "/" << num_phi_points << "\n";
        }
    }
    
    // #####################
    // (6) FINISHING UP
    // #####################
    
    std::cout << "===================================\n";
    std::cout << "All calculations completed!\n";
    std::cout << "===================================\n";

    // #####################
    // (6.1) filenaming
    // #####################

    // get your ducks in a row: collect targets in a string 
    /*
    std::string target_sites_str = "";
    for (size_t i = 0; i < target_sites.size(); ++i) {
        target_sites_str += std::to_string(target_sites[i]);
        if (i < target_sites.size() - 1) target_sites_str += "_"; // underscore for spacing
    }
    std::cout << "Targets: " << target_sites_str << std::endl;
    */

    std::string base_filename = "SurvProb_gamma2_" + std::to_string(gamma_2) + 
                                "_phi2_" + std::to_string(phi_2) + 
                                "_N_" + std::to_string(num_sites) + 
                                "_target_-" + // target_sites_str + 
                                "_resolution_" + std::to_string(num_phi_points) + 
                                "x" + std::to_string(num_tau_points) + 
                                //"_Tmax_" + std::to_string(T_max);
                                "_" + std::to_string(n_Meas) + "_measurements";

    // branch filenaming 
    std::string filename_results = "RESULTS_" + base_filename + ".txt";
    std::string filename_optimal_values = "phi1_VS_tau_" + base_filename + ".txt";

    // #####################
    // (6.2) saving 2D grid
    // #####################

    // save grid data to file for Python plotting (mean_hit... is a MatrixXd!)
    std::ofstream grid_file(filename_results);
    for (int i = 0; i < num_tau_points; ++i) {
        for (int j = 0; j < num_phi_points; ++j) {
            grid_file << Survival_Probs(i, j) << (j == num_phi_points - 1 ? "" : " "); // separate data at the end of the line
        }
        grid_file << "\n";
    }
    grid_file.close();

    // #####################
    // (6.3) optimization: looking for the min
    // #####################

    int min_tau_idx, min_phi_idx;
    double max_prob = Survival_Probs.maxCoeff(&min_tau_idx, &min_phi_idx);

    double optimal_tau_val = tau_values[min_tau_idx];
    double optimal_phi_val = phi_values[min_phi_idx];

    // #####################
    // (6.4) saving optimal values
    // #####################

    std::ofstream f(filename_optimal_values);
    f << "Maximum survival probability: " << max_prob << "\n";
    f << "Optimal parameters: \\phi_1 = " << optimal_phi_val << ", \\tau = " << optimal_tau_val << "\n";
    f.close();

    std::cout << "===================================\n";
    std::cout << "Data exported successfully! Run the Python script to plot.\n";
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;
    std::cout << "Elapsed time: " << elapsed_seconds.count() << " seconds.\n";
    std::cout << "===================================\n";

    return 0;
}
