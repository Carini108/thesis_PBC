#include "first_detection.hpp"
#include <string>
#include <fstream>
#include <iostream>
#include <unsupported/Eigen/MatrixFunctions>

// IDEA: mean hitting times shown over a 2D grid 
// of tau vs phi_1 for fixed size N

int main() {

    // #####################
    // 1) PARAMETERS
    // #####################
    
    // number of sites
    int num_sites = 21;
    // grid resolution
    int num_phi_points = 150; // horizontal resolution
    int num_tau_points = 150; // vertical resolution
    // grid boundaries
    double phi_min = -M_PI / num_sites;
    double phi_max = +M_PI / num_sites;
    double tau_min = 0.02;
    double tau_max = 4.00;
    // time evolution parameters and number of MC runs
    double T_max = 180.0; // cutoff time (limited resource)
    int M = 200; // number of samples of the hitting time
    // couplings and constants relevant to H
    double on_site_energy = 0.0;
    double gamma = 1.0; // hopping rate
    double gamma_1 = 1.0;
    double gamma_2 = 0.0;
    //phi_1 = 0.0
    double phi_2 = 0.0;
    //phase_1 = np.exp(1j * phi_1)
    Complex phase_2 = std::polar(1.0, phi_2);

    // #####################
    // 2) CONSTRUCT TARGETS
    // #####################

    // local measurement on m-D subspace (cannot tell on which one the particle is)
    const std::vector<int> target_sites = {num_sites/2+1}; 
    //const std::vector<int> target_sites = {num_sites/2 -1 , num_sites/2, num_sites/2 + 1}; 
    MatrixXc multisite_projector = create_subspace_projector(num_sites, target_sites);

    // #####################
    // 3) PRINT PARAMETERS
    // #####################

    std::cout << "-----------------------------------\n";
    std::cout << "Number of sites N = " << num_sites << "\n";
    std::cout << "Maximum evolution time T = " << T_max << "\n";
    std::cout << "Monte Carlo runs per grid point M = " << M << "\n";
    std::cout << "In this code we perform a PVM once every tau!\n";
    std::cout << "-----------------------------------\n";
    std::cout << "Multisite Projector:\n" << multisite_projector << "\n"; // print to check...

    // #####################
    // 4) BUILD EMPTY GRID
    // #####################

    // start constructing the grid by defining the 'checkerboard'...
    std::vector<double> phi_values(num_phi_points);
    for (int i = 0; i < num_phi_points; ++i)
        phi_values[i] = phi_min + i * (phi_max - phi_min) / (num_phi_points - 1);
    std::vector<double> tau_values(num_tau_points);
    for (int i = 0; i < num_tau_points; ++i)
        tau_values[i] = tau_min + i * (tau_max - tau_min) / (num_tau_points - 1);

    // ...and setting all values to zero
    Eigen::MatrixXd mean_hitting_times = Eigen::MatrixXd::Zero(num_tau_points, num_phi_points);

    std::cout << "All parameters initialized! Simulation starting...\n";
    std::cout << "===================================\n";

    // #####################
    // 5) OpenMP parallelization over the outer grid loop for maximum speed
    // #####################

    #pragma omp parallel for schedule(dynamic)
    for (int p_idx = 0; p_idx < num_phi_points; ++p_idx) { // the first variable is the index, the second the value
        
        // for reproducibility (thread-safe seeding)
        std::mt19937 gen(13 + p_idx); 
        std::uniform_real_distribution<double> dis(0.0, 1.0);

        // run over all values
        double phi_1 = phi_values[p_idx];
        Complex phase_1 = std::polar(1.0, phi_1);

        // #####################
        // 5.1) Laplacian matrix for a ring (L = D - A)
        // #####################

        MatrixXc L = build_Laplacian(num_sites, on_site_energy, gamma_1, gamma_2, phase_1, phase_2);

        // #####################
        // 5.2) initialisation of the state
        // #####################

        VectorXc psi_0 = VectorXc::Zero(num_sites);
        psi_0(0) = 1.0; // start at node 0

        #pragma omp critical
        {
            std::cout << "State at time = 0 initialized! phi_1 = " << phi_1 << "\n";
        }

        // #####################
        // 5.3) DYNAMICS
        // #####################
        MatrixXc H = gamma * L;

        for (int t_idx = 0; t_idx < num_tau_points; ++t_idx) {
            double tau = tau_values[t_idx];

            // coherent unitary evolution step 
            MatrixXc arg = -Complex(0.0, 1.0) * H * tau;
            MatrixXc U_tau = arg.exp(); 
            
            double total_hitting_time = run_Monte_Carlo_hitting_times(M, T_max, tau, U_tau, psi_0, multisite_projector, gen, dis);

            mean_hitting_times(t_idx, p_idx) = total_hitting_time; // mean
        }

        #pragma omp critical
        {
            std::cout << "phi_1 step number " << p_idx + 1 << "/" << num_phi_points << "\n";
        }
    }

    
    // #####################
    // 6) finish: 
    // #####################
    
    std::cout << "===================================\n";
    std::cout << "All simulations completed!\n";
    std::cout << "===================================\n";

    // #####################
    // 6.1) filenaming
    // #####################

    // get your ducks in a row: collect targets in a string 
    std::string target_sites_str = "";
    for (size_t i = 0; i < target_sites.size(); ++i) {
        target_sites_str += std::to_string(target_sites[i]);
        if (i < target_sites.size() - 1) target_sites_str += "_";
    }
    std::cout << "Targets: " << target_sites_str << std::endl;

    std::string base_filename = "mean_hitting_time_PVM_gamma2_" + std::to_string(gamma_2) + 
                                "_N_" + std::to_string(num_sites) + 
                                "_target_" + target_sites_str + 
                                "_resolution_" + std::to_string(num_phi_points) + 
                                "x" + std::to_string(num_tau_points) + 
                                "_" + std::to_string(M) + "_runs";
    
    // split filenaming 
    std::string filename_results = "RESULTS_" + base_filename + ".txt";
    std::string filename_optimal_values = "phi1_vs_tau_" + base_filename + ".txt";

    // #####################
    // 6.2) saving 2D grid
    // #####################

    // save grid data to file for Python plotting
    std::ofstream grid_file(filename_results);
    for (int i = 0; i < num_tau_points; ++i) {
        for (int j = 0; j < num_phi_points; ++j) {
            grid_file << mean_hitting_times(i, j) << (j == num_phi_points - 1 ? "" : " ");
        }
        grid_file << "\n";
    }
    grid_file.close();

    // #####################
    // 6.3) optimization: looking for the min
    // #####################

    int min_tau_idx, min_phi_idx;
    double min_time = mean_hitting_times.minCoeff(&min_tau_idx, &min_phi_idx);

    double optimal_tau_val = tau_values[min_tau_idx];
    double optimal_phi_val = phi_values[min_phi_idx];

    // #####################
    // saving optimal values
    // #####################

    std::ofstream f(filename_optimal_values);
    f << "Minimum mean hitting time: " << min_time << "\n";
    f << "Number of Monte Carlo runs per point M: " << M << "\n";
    f << "Optimal parameters: \\phi_1 = " << optimal_phi_val << ", \\tau = " << optimal_tau_val << "\n";
    f.close();

    std::cout << "===================================\n";
    std::cout << "Data exported successfully! Run the Python script to plot.\n";

    return 0;
}
