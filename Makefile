CXX = g++
CXXFLAGS = -std=c++20 -Wall -Og -g
LIB_DIR = ./NeuralNetLite
STB_DIR = ./stb
TESTS_DIR = ./tests
XOR_TARGET = xor
CONV_TARGET = conv
IMG_TARGET = image

all: $(XOR_TARGET) $(CONV_TARGET) $(IMG_TARGET)

SRC_XOR = $(LIB_DIR)/MatUtils.cpp $(LIB_DIR)/Metrics.cpp $(LIB_DIR)/Linear.cpp $(LIB_DIR)/FeedForwardNet.cpp $(TESTS_DIR)/xor.cpp
$(XOR_TARGET): $(SRC_XOR)
	$(CXX) $(CXXFLAGS) $^ -o $@

SRC_CONV = $(LIB_DIR)/MatUtils.cpp $(LIB_DIR)/Metrics.cpp $(LIB_DIR)/Linear.cpp $(LIB_DIR)/FeedForwardNet.cpp $(TESTS_DIR)/conv.cpp
$(CONV_TARGET): $(SRC_CONV)
	$(CXX) $(CXXFLAGS) $^ -o $@

SRC_IMAGE = $(LIB_DIR)/MatUtils.cpp $(LIB_DIR)/Metrics.cpp $(LIB_DIR)/Linear.cpp $(LIB_DIR)/FeedForwardNet.cpp $(TESTS_DIR)/image.cpp $(STB_DIR)/stb_image.h $(STB_DIR)/stb_image_write.h
$(IMG_TARGET): $(SRC_IMAGE)
	$(CXX) $(CXXFLAGS) $^ -o $@

clean:
	rm -f $(XOR_TARGET) $(CONV_TARGET)