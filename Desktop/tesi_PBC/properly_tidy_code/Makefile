CXX = g++
CXXFLAGS = -std=c++17 -03 -Wall -fopenmp

EIGEN_INC ?= -I/usr/include/eigen3 -I/usr/include/eigen3

TARGET = simulate_qw
OBJS = main.o first_detection.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

main.o: main.cpp first_detection.hpp
	$(CXX) $(CXXFLAGS) $(EIGEN_INC) -c $< -o $@

first_detection.o: first_detection.cpp first_detection.hpp
	$(CXX) $(CXXFLAGS) $(EIGEN_INC) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run