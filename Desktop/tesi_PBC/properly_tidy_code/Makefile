CXX = /opt/homebrew/opt/llvm/bin/clang++

CXXFLAGS = -std=c++17 -O3 -Wall \
           -I/opt/homebrew/include/eigen3 \
           -I/opt/homebrew/opt/libomp/include \
           -isysroot $(shell xcrun --show-sdk-path) \
           -Xpreprocessor -fopenmp

LDFLAGS = -L/opt/homebrew/opt/libomp/lib -lomp

TARGET = simulate_qw
OBJS = main.o first_detection.o

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(TARGET) $(LDFLAGS)

main.o: main.cpp
	$(CXX) $(CXXFLAGS) -c main.cpp -o main.o

first_detection.o: first_detection.cpp
	$(CXX) $(CXXFLAGS) -c first_detection.cpp -o first_detection.o

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(OBJS)

cleanall:
	rm -f $(OBJS) $(TARGET)