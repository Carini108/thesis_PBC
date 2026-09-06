#include "first_detection.hpp"
#include <iostream>
#include <cmath>

// ####### Projective Measurement Function (PVM)
void on_site_PVM(VectorXc& state, const int target_site, bool& detection_successful, 
    std::mt19937& gen, std::uniform_real_distribution<double>& dis) {
    
    // norm() to get probability for the successful measurement
    // |<i|psi>|^2, because in the Complex lib norm(z) is |z|^2
    double prob_success = std::norm(state(target_site)); 
    // 'blank' state to write the result
    VectorXc collapsed_state = VectorXc::Zero(state.size());
    detection_successful = false;

    if (dis(gen) >= prob_success) {
        // UNSUCCESSFUL: project on the complementary subspace
        collapsed_state = state;
        collapsed_state(target_site) = 0.0; 
        // renormalise
        double norm = collapsed_state.norm(); // in this case .norm() comes with the square root
        if (norm > 0) {
            collapsed_state = collapsed_state / norm;
        } else {
            std::cout<<"Unphysical zero-length state!"<<std::endl;
            std::exit(1);
        }
    } else {
        // SUCCESSFUL: project on the target subspace
        collapsed_state(target_site) = state(target_site)/std::sqrt(prob_success);
        detection_successful = true;
    }

    state = collapsed_state;
}

// ####### Generalized PVM (accepts a generic projector P)
void generalized_PVM(VectorXc& state, const MatrixXc& P, bool& detection_successful, 
    std::mt19937& gen, std::uniform_real_distribution<double>& dis) {
    
    // probability for the successful measurement
    VectorXc projected_state = P * state; // unnormalised projected state
    double prob_success = projected_state.squaredNorm();
    // 'blank' state to write the result
    VectorXc collapsed_state = VectorXc::Zero(state.size());
    detection_successful = false;

    if (dis(gen) >= prob_success) {
        // UNSUCCESSFUL: project on the complementary subspace
        collapsed_state = state - projected_state; // (I-P) |psi>
        //renormalise
        double norm = collapsed_state.norm();
        if (norm > 0) {
            state = collapsed_state / norm;
        } else {
            std::cout<<"Unphysical zero-length state!"<<std::endl;
            std::exit(1);
        } 
    } else {
        // SUCCESSFUL: project on the target subspace
        double norm = std::sqrt(prob_success);
        if (norm > 0) {
            state = projected_state / norm;
        } else {
            std::cout<<"Unphysical zero-length state!"<<std::endl;
            std::exit(1);
        } 
        detection_successful = true;
    }
}

// ####### Project on a subset of sites (accepts the indices)
MatrixXc create_subspace_projector(int num_sites, const std::vector<int>& target_sites) {
   
    MatrixXc P = MatrixXc::Zero(num_sites, num_sites);
    for (int site : target_sites) {
        if (site >=0 && site < num_sites) {
            P(site, site) = 1.0;
        } else {
            std::cout<<"Wrong indices for the states! Cannot build the projector."<<std::endl;
            std::exit(1);
        }
    }

    return P;
}

// ####### Project on arbitrary states (accepts a ket)
MatrixXc create_state_projector(const VectorXc& chi) {
    
    double numerical_tolerance = 1e-8;
    if ( std::abs(chi.norm()-1.0) > numerical_tolerance ) {
        std::cout<<"State chi is not properly normalised! Cannot build the projector."<<std::endl;
        std::exit(1);
    }

    return chi * chi.adjoint();
}

// ####### Build the Laplacian for the dynamics
MatrixXc build_Laplacian(int num_sites, double on_site_energy, 
    double gamma_1, double gamma_2, Complex phase_1, Complex phase_2) {

    // initialise an empty matrix
    MatrixXc L = MatrixXc::Zero(num_sites, num_sites);
    // on-site energies and first/second neighbour couplings
    for (int i=0; i<num_sites; ++i) {
        L(i,i) = 2.0 * on_site_energy;
        if (i<num_sites-1) {
            L(i, i+1) += -gamma_1 * phase_1;
            L(i+1, i) += -gamma_1 * std::conj(phase_1);
        }
        if (i<num_sites-2) {
            L(i, i+2) += -gamma_2 * phase_2;
            L(i+2, i) += -gamma_2 * std::conj(phase_2); 
        }
    } 
    // periodic boundary condition (continuous phase growth!)
    if (num_sites>1) {
        L(0, num_sites-1) += -gamma_1 * std::conj(phase_1);
        L(num_sites-1, 0) += -gamma_1 * phase_1;
    }
    if (num_sites>2) {
        L(0, num_sites-2) += -gamma_2 * std::conj(phase_2);
        L(num_sites-2, 0) += -gamma_2 * phase_2;
        L(1, num_sites-1) += -gamma_2 * std::conj(phase_2);
        L(num_sites-1, 1) += -gamma_2 * phase_2;
    }

    return L;
}

// ####### Run one Monte Carlo iteration (modify this to implement random times...)
double run_Monte_Carlo_hitting_times(int M, double T_max, double tau, 
    const MatrixXc& U_tau, const VectorXc& psi_0, const MatrixXc& projector, 
    std::mt19937& gen, std::uniform_real_distribution<double>& dis) {
    
    double total_hitting_time = 0.0; // this is gonna be < n tau >
    // MC loop to estimate the time of first detection
    for (int run=0; run<M; ++run) {
        VectorXc psi_step = psi_0; 
        double time = 0.0;
        bool detected = false;
        // continue unless detected and until total time runs out
        while (!detected && time <T_max) {
            psi_step = U_tau * psi_step;
            time += tau;
            generalized_PVM(psi_step, projector, detected, gen, dis);
        }
        total_hitting_time += time;
    }

    return total_hitting_time / static_cast<double>(M);
}