#include "Linear.hpp"
#include "MatUtils.hpp"
#include <cassert>
#include <iostream>
#include <vector>

// Implement Constructor of DenseLayer
DenseLayer::DenseLayer(size_t input_size, size_t output_size, double_t learning_rate)
{
    // Makes 0.0 in weights
    weights.reshape({ input_size, output_size });
    weights.fill();
    bias.reshape({ 1, output_size });
    bias.fill();
    inputs.reshape({ 1, input_size });
    learning_rate = learning_rate;
}

// Forward Propogation of DenseLayer
TensorLite DenseLayer::forward(const TensorLite& input)
{
    
    inputs.setData(input.data);
    return inputs.multiply(weights).add(bias);
}

// Backward Propogation of DenseLayer
TensorLite DenseLayer::backward(const TensorLite& output_errors)
{
    // (1, OS)*(IS, OS)^T = (1, IS)
    TensorLite input_errors = output_errors.multiply(weights.transpose());
    // Update weights
    // (1, IS)^T x (1, OS) = (IS, OS)
    weights.subInPlace(inputs.transpose().multiply(output_errors).multiplyByConstant(learning_rate));
    // Update bias
    // (1, OS) - (1, OS)
    bias.subInPlace(output_errors.multiplyByConstant(learning_rate));
    return input_errors;
}

Activation::Activation(std::function<double_t(double_t)> ac, std::function<double_t(double_t)> acDer, size_t input_size){
    activation = ac;
    activationDer = acDer;
    inputs.reshape({1, input_size});
}
// Implement Forward Propogation of Activation
TensorLite Activation::forward(const TensorLite& input)
{
    inputs = input;
    TensorLite result = inputs.apply(activation);
    return result;
}

// Implement Backward Propogation of Activation
TensorLite Activation::backward(const TensorLite& output_errors)
{
    /*
        1. Return after multiplying pairwise the derivative of the inputs by the derivative of the activation
    */
    TensorLite result = inputs.apply(activationDer).multiplyPairWise(output_errors);
    return result;
}