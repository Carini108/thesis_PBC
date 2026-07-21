To run the simulation in C++:

g++ -O3 -std=c++17 -Xpreprocessor -fopenmp -I$(brew --prefix eigen)/include/eigen3 -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp main.cpp -o qw_sim && ./qw_sim