CXX = g++
CXXFLAGS = -std=c++20 -Wall -Og -g
LIB_DIR = ./NeuralNetLite
TESTS_DIR = ./tests
TEST_TARGET = test

SRC_FILES = $(LIB_DIR)/MatUtils.cpp $(LIB_DIR)/Metrics.cpp $(LIB_DIR)/Linear.cpp $(LIB_DIR)/FeedForwardNet.cpp $(TESTS_DIR)/main.cpp

all: $(TEST_TARGET)

$(TEST_TARGET): $(SRC_FILES)
	$(CXX) $(CXXFLAGS) $^ -o $@

clean:
	rm -f $(TEST_TARGET)