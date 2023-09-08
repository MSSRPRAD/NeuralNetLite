CXX = g++
CXXFLAGS = -std=c++20 -Wall -Og -g
LIB_DIR = ./NeuralNetLite
TESTS_DIR = ./tests
TEST_TARGET = test

SRC_FILES = $(LIB_DIR)/MatUtils.cpp $(TESTS_DIR)/main.cpp

all: $(TEST_TARGET)

$(TEST_TARGET): $(SRC_FILES)
	$(CXX) $(CXXFLAGS) $^ -o $@

clean:
	rm -f $(TEST_TARGET)