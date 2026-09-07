#ifndef FIRST_DETECTION_HPP
#define FIRST_DETECTION_HPP

// #####################
// libraries after the guard clauses
// #####################

#include <vector>
#include <complex>
#include <random>
#include <Eigen/Dense>

// #####################
// we define some "shortcuts" to invoke types
// #####################

using Complex = std::complex<double>;
// X refers to being dynamic, c stands for complex: it works as follows
// Eigen::Matrix<ScalarType, RowsAtCompileTime, ColsAtCompileTime>
using MatrixXc = Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic>;
using VectorXc = Eigen::Matrix<Complex, Eigen::Dynamic, 1>;

// #####################
// helper functions for the main
// #####################

// ####### Projective Measurement Function (PVM)
void on_site_PVM(VectorXc& state, const int target_site, bool& detection_successful, 
    std::mt19937& gen, std::uniform_real_distribution<double>& dis);

// ####### Generalised PVM (accepts a generic projector P)
void generalised_PVM(VectorXc& state, const MatrixXc& P, bool& detection_successful, 
    std::mt19937& gen, std::uniform_real_distribution<double>& dis);

// ####### Project on a subset of sites (accepts the indices)
MatrixXc create_subspace_projector(int num_sites, const std::vector<int>& target_sites);

// ####### Project on an arbitrary state (accepts a ket)
MatrixXc create_state_projector(const VectorXc& chi);

// ####### Test Hermiticity
bool is_hermitian(const MatrixXc& L, double tol = 1e-12);

// ####### Build the Laplacian for the dynamics
MatrixXc build_Laplacian(int num_sites, double on_site_energy, 
    double gamma_1, double gamma_2, Complex phase_1, Complex phase_2);

// ####### Run M Monte Carlo iterations
double run_Monte_Carlo_hitting_times(int M, double T_max, double tau, 
    const MatrixXc& U_tau, const VectorXc& psi_0, const MatrixXc& projector, 
    std::mt19937& gen, std::uniform_real_distribution<double>& dis);

#endif // FIRST_DETECTION_HPP