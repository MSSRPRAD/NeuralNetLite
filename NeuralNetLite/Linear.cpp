#include "MatUtils.hpp"
#include "Linear.hpp"
#include <cassert>
#include <iostream>
#include <vector>


// Implement Constructor of DenseLayer
DenseLayer::DenseLayer(size_t input_size, size_t output_size){
    // Makes 0.0 in weights
    weights.reshape({input_size, output_size});
    weights.fill();
    bias.reshape({1, output_size});
    bias.fill();
    input.reshape({1, input_size});
}

// Forward Propogation of DenseLayer
TensorLite DenseLayer::forward(const TensorLite &input) const{
    return input.multiply(weights).add(bias);
}

// Implement Constructor of Activation
Activation::Activation(std::function<double(double)> ac, std::function<double(double)> acDer, size_t input_size){
    activation = ac;
    activationDer = acDer;
    inputs.reshape({1, input_size});
}

// Implement Forward Propogation of Activation
TensorLite Activation::forward(const TensorLite &input) const {
    TensorLite result = input.apply(activation);
    return result;
}

// Implement Backward Propogation of Activation
TensorLite Activation::backward(const TensorLite &output_errors) {
    // OS , 1 x 1, OS 
    TensorLite result = inputs.apply(activationDer).multiply(output_errors);
    return result;
}