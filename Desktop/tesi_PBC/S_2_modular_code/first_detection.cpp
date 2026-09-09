#include "first_detection.hpp"
#include <iostream>
#include <cmath>

// CAREFUL WITH HOW NORMS ARE CALCULATED!
// in the <complex> lib norm(z) is |z|^2
// in the <Eigen> lib v.squaredNorm() is ||v||^2
//                and v.norm() is just ||v|| (square root embedded...)

// ####### Projective Measurement Function (PVM)
void on_site_PVM(VectorXc& state, const int target_site, bool& detection_successful, 
    std::mt19937& gen, std::uniform_real_distribution<double>& dis) {
    
    // |<i|psi>|^2 to get probability for the successful measurement
    double prob_success = std::norm(state(target_site)); 
    // create a 'blank' state to write the result
    VectorXc collapsed_state = VectorXc::Zero(state.size());

    if (dis(gen) >= prob_success) {
        // UNSUCCESSFUL: project on the complementary subspace
        detection_successful = false;
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
        detection_successful = true;
        collapsed_state(target_site) = state(target_site)/std::sqrt(prob_success);
    }

    state = collapsed_state;
}

// ####### Generalised PVM (accepts a generic projector P)
void generalised_PVM(VectorXc& state, const MatrixXc& P, bool& detection_successful, 
    std::mt19937& gen, std::uniform_real_distribution<double>& dis) {
    
    // probability for the successful measurement
    VectorXc projected_state = P * state; // unnormalised projected state
    double prob_success = projected_state.squaredNorm(); // <psi | P | psi >
    // 'blank' state to write the result
    VectorXc collapsed_state = VectorXc::Zero(state.size());

    if (dis(gen) >= prob_success) {
        // UNSUCCESSFUL: project on the complementary subspace
        detection_successful = false;
        collapsed_state = state - projected_state; // (I-P) |psi>
        //renormalise
        double norm = collapsed_state.norm();
        if (norm > 0) {
            state = collapsed_state / norm; // normalised (I-P)|psi>
        } else {
            std::cout<<"Unphysical zero-length state!"<<std::endl;
            std::exit(1);
        } 
    } else {
        // SUCCESSFUL: project on the target subspace
        detection_successful = true;
        double norm = std::sqrt(prob_success);
        if (norm > 0) {
            state = projected_state / norm; // normalised P|psi>
        } else {
            std::cout<<"Unphysical zero-length state!"<<std::endl;
            std::exit(1);
        } 
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

// ####### Project on an arbitrary state (accepts a ket)
MatrixXc create_state_projector(const VectorXc& chi) {
    
    double numerical_tolerance = 1e-9;
    if ( std::abs(chi.norm()-1.0) > numerical_tolerance ) {
        std::cout<<"State chi is not properly normalised! Cannot build the projector."<<std::endl;
        std::exit(1);
    }

    return chi * chi.adjoint();
}

// ####### Test Hermiticity
bool is_hermitian(const MatrixXc& L, double tol) {
    // trivial check of cardinality: matrix must be square!
    if (L.rows() != L.cols()) return false;
    // Eigen's built-in check tests VS the conjugate transpose (L^\dagger)
    return L.isApprox(L.adjoint(), tol);
}

// ####### Build the Laplacian for the dynamics
MatrixXc build_Laplacian(int num_sites, double on_site_energy, 
    double gamma_1, double gamma_2, Complex phase_1, Complex phase_2) {

    // initialise an empty matrix
    MatrixXc L = MatrixXc::Zero(num_sites, num_sites);

    // on-site energies and first/second neighbour couplings
    for (int i = 0; i < num_sites; ++i) {
        
        // on-site diagonal energy
        L(i, i) = 2.0 * on_site_energy;
        
        // nearest-neighbour coupling (requires N >= 3 to avoid double-counting on N=2)
        if (num_sites > 2) {
            int n1 = (i + 1) % num_sites; // modulo operation for PBC
            L(i, n1) += -gamma_1 * phase_1;
            L(n1, i) += -gamma_1 * std::conj(phase_1);
        } else if (num_sites == 2 && i == 0) {
            // Special 2-site edge case (single bond, i.e. the off-diag entries)
            L(0, 1) = -gamma_1 * phase_1;
            L(1, 0) = -gamma_1 * std::conj(phase_1);
        }
        
        // next-nearest-neighbour coupling (requires N >= 5 to avoid overlapping bonds) 
        if (num_sites >= 5) {
            int n2 = (i + 2) % num_sites;
            L(i, n2) += -gamma_2 * phase_2;
            L(n2, i) += -gamma_2 * std::conj(phase_2);
        } else if (num_sites == 4 && i < 2) {
            // Special 4-site edge case: bonds cross exactly halfway.
            // Only define them for i=0 and i=1 to prevent double-counting.
            int n2 = i + 2;
            L(i, n2) = -gamma_2 * phase_2;
            L(n2, i) = -gamma_2 * std::conj(phase_2);
        }
    } 
    // comment this out for greater speed... 
    /*
    if (!is_hermitian(L)) {
        std::cout<<"L is not Hermitian!"<<std::endl;
        std::exit(1);
    } 
    */
    return L;
}

// ####### Run M Monte Carlo iterations (modify this to implement random times...)
double run_Monte_Carlo_hitting_times(int M, double T_max, double tau, 
    const MatrixXc& U_tau, const VectorXc& psi_0, const MatrixXc& projector, 
    std::mt19937& gen, std::uniform_real_distribution<double>& dis) {
    
    double total_hitting_time = 0.0; // this is gonna be < n \tau >
    // MC loop to estimate the time of first detection
    for (int run=0; run<M; ++run) {
        VectorXc psi_step = psi_0; 
        double time = 0.0; // reset time to zero at the beginning of every run
        bool detected = false;
        // continue unless detected and until total time resource runs out
        while (!detected && time <T_max) {
            psi_step = U_tau * psi_step;
            time += tau;
            generalised_PVM(psi_step, projector, detected, gen, dis);
        }
        total_hitting_time += time;
    }

    return total_hitting_time / static_cast<double>(M); // average < n > * \tau
}