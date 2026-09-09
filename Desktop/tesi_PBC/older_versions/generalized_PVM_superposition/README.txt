To run the simulation in C++:

g++ -O3 -std=c++17 -Xpreprocessor -fopenmp -I$(brew --prefix eigen)/include/eigen3 -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp main.cpp -o gen_PVM && ./gen_PVM

conda activate cwq

python plot_results.py