#include "first_detection.hpp"
#include <iostream>
#include <cmath>
#include <unsupported/Eigen/MatrixFunctions> // matrix exponential (QSW propagator)

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

// ####### Generalised PVM (accepts a density matrix and a generic projector P)
void generalised_PVM_density_matrix(MatrixXc& rho, const MatrixXc& P, bool& detection_successful, 
    std::mt19937& gen, std::uniform_real_distribution<double>& dis) {
    
    // probability for the successful measurement: p(m)=Tr[P_m \rho]
    Complex trace_against_P = (P*rho).trace();
    double prob_success = std::real(trace_against_P);

    // checks that trace is in fact a probability
    /*
    double tol = 1e-12;
    if ( std::abs(std::sqrt(std::norm(trace_against_P))-prob_success) > tol ) {
        std::cout<<"Probability is not a real number!"<<std::endl;
        exit(1); 
    }
    if ( prob_success > (1.0+tol) ) {
        std::cout<<"Probability is > 1!"<<std::endl;
        exit(2); 
    }
    if ( prob_success < (-tol) ) {
        std::cout<<"Probability is < 0!>"<<std::endl;
        exit(3); 
    }
    */

    if (dis(gen) >= prob_success) {
        // UNSUCCESSFUL: project on the complementary subspace
        // \rho|_(not m) = (1-P_m) \rho (1-P_m) / (1-p(m))
        detection_successful = false;
        MatrixXc I = MatrixXc::Identity(rho.rows(), rho.cols());
        MatrixXc P_comp = I - P;
        MatrixXc collapsed_rho = P_comp * rho * P_comp / (1-prob_success); //renormalised
        rho = collapsed_rho;
        
    } else {
        // SUCCESSFUL: project on the target subspace
        // \rho|_m = P_m \rho P_m / p(m)
        detection_successful = true;
        MatrixXc collapsed_rho = P * rho * P / prob_success; //renormalised
        rho = collapsed_rho;
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

// ####### Run M Monte Carlo iterations for density matrices
double run_Monte_Carlo_hitting_times_density_matrix(int M, double T_max, double tau, 
    const MatrixXc& U_tau, const MatrixXc& U_tau_dagger, const MatrixXc& rho_0, const MatrixXc& projector, 
    std::mt19937& gen, std::uniform_real_distribution<double>& dis) {
    
    double total_hitting_time = 0.0; 
    // MC loop 
    for (int run=0; run<M; ++run) {
        MatrixXc rho_step = rho_0; 
        double time = 0.0; // reset time to zero at the beginning of every run
        bool detected = false;
        // continue unless detected and until total time resource runs out
        while (!detected && time <T_max) {
            rho_step = U_tau * rho_step * U_tau_dagger;
            time += tau;
            generalised_PVM_density_matrix(rho_step, projector, detected, gen, dis);
        }
        total_hitting_time += time;
    }

    return total_hitting_time / static_cast<double>(M); // average time

}

// ####### Intrinsic evolution time step for a density matrix
void evolve_intrinsic_decoherence(MatrixXc& rho, const MatrixXc& V, const MatrixXc& M){

    MatrixXc rho_eigen = V.adjoint() * rho * V; // diagonalise
    MatrixXc rho_evolved_eigen = M.cwiseProduct(rho_eigen);
    rho = V * rho_evolved_eigen * V.adjoint(); // go back

}

// ####### Run M Monte Carlo iterations for density matrices WITH INTRINSIC DECOHERENCE
double run_MC_hitting_times_dens_mat_intrinsic_decoherence(int M, double T_max, double tau, 
    const MatrixXc& V, const Eigen::VectorXd& energies, double gamma_decoherence, const MatrixXc& rho_0, const MatrixXc& projector, 
    std::mt19937& gen, std::uniform_real_distribution<double>& dis) {
    
    int size = rho_0.rows();
    MatrixXc Matr(size, size);
    for (int i=0; i<size; i++){
        for (int j=0; j<size; j++) {
            double deltaE = energies(i)-energies(j);
            Complex exponent = ( - Complex(0.0,1.0) * deltaE - 0.5 * gamma_decoherence * (deltaE * deltaE) ) * tau;
            Matr(i, j) = std::exp(exponent);
        }
    }

    double total_hitting_time = 0.0; 
    // MC loop 
    for (int run=0; run<M; ++run) {
        MatrixXc rho_step = rho_0; 
        double time = 0.0; // reset time to zero at the beginning of every run
        bool detected = false;
        // continue unless detected and until total time resource runs out
        while (!detected && time <T_max) {
            evolve_intrinsic_decoherence(rho_step, V, Matr) ;
            time += tau;
            generalised_PVM_density_matrix(rho_step, projector, detected, gen, dis);
        }
        total_hitting_time += time;
    }

    return total_hitting_time / static_cast<double>(M); // average time

}


// ###########################################################################
// ####### QUANTUM STOCHASTIC WALK (QSW) = generic Lindblad dynamics
// ###########################################################################
//
//   d rho/dt = -(1-p) i [H, rho] + p sum_k ( L_k rho L_k^dag - 1/2 { L_k^dag L_k, rho } )
//
// There is no closed form for exp(tau * Liouvillian) (the jump terms are not diagonal in the energy
// basis, so the Hadamard-product trick of the intrinsic decoherence does not work), but none is needed:
// the master equation is LINEAR in rho, hence ONE STEP of duration tau IS A FIXED N^2 x N^2 MATRIX
// (the "propagator"). We build it once per (p, tau) with a matrix exponential and then simply apply it.
//
// Vectorisation: stack the columns of rho, vec(rho)[a + N*b] = rho(a,b). A column-major MatrixXc is
// already stored in this order, so rho.data() IS vec(rho) (no copies needed). With this convention
//      vec(A rho B) = S vec(rho)   with   S[a2 + N*b2, a + N*b] = A(a2,a) * B(b,b2)
// (that is S = B^T (x) A, Kronecker product).

// ####### Superoperator (N^2 x N^2) of the linear map  rho -> A rho B
MatrixXc superop_AxB(const MatrixXc& A, const MatrixXc& B) {

    const int N = A.rows();
    MatrixXc S = MatrixXc::Zero(N*N, N*N);
    for (int b = 0; b < N; ++b) {
        for (int b2 = 0; b2 < N; ++b2) {
            const Complex B_b_b2 = B(b, b2);
            if (B_b_b2 == Complex(0.0, 0.0)) continue; // nothing to write
            for (int a = 0; a < N; ++a) {
                for (int a2 = 0; a2 < N; ++a2) {
                    S(a2 + N*b2, a + N*b) = A(a2, a) * B_b_b2;
                }
            }
        }
    }
    return S;
}

// ####### Incoherent hops of the classical random walk living on the graph of H:
// one jump operator L = sqrt(rate) |i><j| (hop j -> i) for every ordered pair (i,j) with H(i,j) != 0
std::vector<MatrixXc> build_jump_operators_from_graph(const MatrixXc& H, double rate, double tol) {

    const int N = H.rows();
    std::vector<MatrixXc> jump_ops;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i != j && std::abs(H(i, j)) > tol) {
                MatrixXc L = MatrixXc::Zero(N, N);
                L(i, j) = std::sqrt(rate);
                jump_ops.push_back(L);
            }
        }
    }
    return jump_ops;
}

// ####### Liouvillian of the QSW as an N^2 x N^2 matrix
// p = 0: purely coherent walk,  p = 1: purely incoherent (classical) walk, 0 < p < 1: mixture
// (any other noise can be added by appending its jump operators to jump_ops)
MatrixXc build_QSW_superoperator(const MatrixXc& H, const std::vector<MatrixXc>& jump_ops, double p) {

    const int N = H.rows();
    const MatrixXc I = MatrixXc::Identity(N, N);

    // coherent part: -(1-p) i [H, rho] = -(1-p) i ( H rho - rho H )
    const Complex coherent_prefactor = Complex(0.0, -1.0) * (1.0 - p);
    MatrixXc Lsup = coherent_prefactor * (superop_AxB(H, I) - superop_AxB(I, H));

    // incoherent part: p sum_k ( L rho L^dag - 1/2 L^dag L rho - 1/2 rho L^dag L )
    if (p > 0.0) {
        for (const MatrixXc& L : jump_ops) {
            const MatrixXc Ldag = L.adjoint();
            const MatrixXc LdagL = Ldag * L;
            MatrixXc D = superop_AxB(L, Ldag);
            D -= Complex(0.5, 0.0) * superop_AxB(LdagL, I);
            D -= Complex(0.5, 0.0) * superop_AxB(I, LdagL);
            Lsup += Complex(p, 0.0) * D;
        }
    }
    return Lsup;
}

// ####### One-step propagator exp(tau * Lsup): rho(t+tau) = unvec( prop * vec(rho(t)) )
MatrixXc build_propagator(const MatrixXc& Lsup, double tau) {

    const MatrixXc arg = Complex(tau, 0.0) * Lsup;
    MatrixXc prop = arg.exp(); // needs <unsupported/Eigen/MatrixFunctions>
    return prop;
}

// ####### Evolution step: rho <- unvec( prop * vec(rho) ). 'work' is a scratch vector of size N^2
void evolve_with_propagator(MatrixXc& rho, const MatrixXc& prop, VectorXc& work) {

    const int N2 = static_cast<int>(rho.size());
    Eigen::Map<VectorXc> vec_rho(rho.data(), N2); // column-major storage: this IS vec(rho)
    work.noalias() = prop * vec_rho;
    vec_rho = work;
}

// ####### Run M Monte Carlo iterations with a given one-step propagator (same logic as before)
// returns the mean hitting time and the fraction of runs NOT detected within T_max
HittingStats run_MC_hitting_times_dens_mat_propagator(int M, double T_max, double tau,
    const MatrixXc& prop, const MatrixXc& rho_0, const MatrixXc& projector,
    std::mt19937& gen, std::uniform_real_distribution<double>& dis) {

    VectorXc work(rho_0.size());
    double total_hitting_time = 0.0;
    int num_undetected = 0;
    // MC loop
    for (int run = 0; run < M; ++run) {
        MatrixXc rho_step = rho_0;
        double time = 0.0; // reset time to zero at the beginning of every run
        bool detected = false;
        // continue unless detected and until total time resource runs out
        while (!detected && time < T_max) {
            evolve_with_propagator(rho_step, prop, work);
            time += tau;
            generalised_PVM_density_matrix(rho_step, projector, detected, gen, dis);
        }
        if (!detected) ++num_undetected;
        total_hitting_time += time;
    }

    HittingStats stats;
    stats.mean_time = total_hitting_time / static_cast<double>(M);
    stats.undetected_fraction = static_cast<double>(num_undetected) / static_cast<double>(M);
    return stats;
}

// ####### EXACT mean hitting time (no Monte Carlo at all)
// As long as no click occurs the (unnormalised) state evolves DETERMINISTICALLY:
//      rho~_{n+1} = Q E_tau(rho~_n) Q ,   Q = 1 - P ,   S_n = Tr rho~_n = P(no click in the first n measurements)
// so, with the same cutoff convention as the MC loop (time = n*tau, n <= n_steps),
//      <n tau> = tau * sum_{n=0}^{n_steps-1} S_n ,    S_{n_steps} = fraction of runs undetected at T_max.
// In vectorised form the 'no click' map is the N^2 x N^2 matrix K = (Q (x) Q) * prop.
HittingStats exact_hitting_time_propagator(double T_max, double tau,
    const MatrixXc& prop, const MatrixXc& rho_0, const MatrixXc& projector) {

    const int N = rho_0.rows();
    const int N2 = N * N;

    const MatrixXc Q = MatrixXc::Identity(N, N) - projector;
    const MatrixXc K = superop_AxB(Q, Q) * prop;

    VectorXc v(N2), w(N2);
    for (int b = 0; b < N; ++b)
        for (int a = 0; a < N; ++a)
            v(a + N*b) = rho_0(a, b); // v = vec(rho_0)

    // number of steps: exactly what the MC loop performs when no detection happens
    int n_steps = 0;
    double time = 0.0;
    while (time < T_max) { time += tau; ++n_steps; }

    double survival = 1.0;     // S_0
    double sum_survival = 0.0;
    for (int n = 0; n < n_steps; ++n) {
        sum_survival += survival;
        w.noalias() = K * v;
        v = w;
        survival = 0.0;
        for (int a = 0; a < N; ++a) survival += std::real(v(a + N*a)); // trace
    }

    HittingStats stats;
    stats.mean_time = tau * sum_survival;
    stats.undetected_fraction = survival;
    return stats;
}
