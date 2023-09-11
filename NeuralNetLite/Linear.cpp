#include "Linear.hpp"
#include "MatUtils.hpp"
#include <cassert>
#include <iostream>
#include <vector>

// Implement Constructor of DenseLayer
DenseLayer::DenseLayer(size_t input_size, size_t output_size, double_t _learning_rate)
{
    // Makes 0.0 in weights
    weights = TensorLite({ input_size, output_size });
    weights.fill();
    bias = TensorLite({ 1, output_size });
    bias.fill();
    inputs = TensorLite({ 1, input_size });
    learning_rate = _learning_rate;
}

void DenseLayer::print() const {
    weights.print();
    bias.print();
}

void Activation::print() const {
    std::cout<<"Activation Layer\n";
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
    // std::cout<<"weights diff: \n";
    // weights.print();
    // (1, OS)*(IS, OS)^T = (1, IS)
    TensorLite input_errors = output_errors.multiply(weights.transpose());
    // Update weights
    // (1, IS)^T x (1, OS) = (IS, OS)
    // inputs.transpose().multiply(output_errors).print();
    // weights.subInPlace(inputs.transpose().multiply(output_errors).multiplyByConstant(learning_rate));
    // weights.multiplyByConstantInPlace(0.9);
    // std::cout<<"printing difference in weights: \n";
    // inputs.transpose().multiply(output_errors).print();
    // std::cout << "\n----\n";
    // inputs.transpose().multiply(output_errors).multiplyByConstant(learning_rate).print();
    // std::cout << "finished printing diff\n";
    weights.subInPlace(inputs.transpose().multiply(output_errors).multiplyByConstant(learning_rate));
    // std::cout<<"weights after: \n";
    // weights.print();
    // Update bias
    // (1, OS) - (1, OS)
    // bias.subInPlace(output_errors.multiplyByConstant(learning_rate));
    bias.subInPlace(output_errors.multiplyByConstant(learning_rate));
    return input_errors;
}

Activation::Activation(std::function<double_t(double_t)> ac, std::function<double_t(double_t)> acDer, size_t input_size){
    activation = ac;
    activationDer = acDer;
    inputs = TensorLite({1, input_size});
}
// Implement Forward Propogation of Activation
TensorLite Activation::forward(const TensorLite& input)
{
    inputs.setData(input.data);
    return inputs.apply(activation);
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


SoftMax::SoftMax(size_t input_size) {
    inputs = TensorLite({1, input_size});
}

TensorLite SoftMax::forward(const TensorLite& input) {

    inputs = input;
    // Calculate the SoftMax scores
    TensorLite exp_input = input.apply([](double_t x) { return std::exp(x); });
    TensorLite softmax_scores = exp_input.multiplyByConstant(1.0 / exp_input.sum());

    return softmax_scores;
}

TensorLite SoftMax::backward(const TensorLite& output_errors) {
    // Calculate the derivative of SoftMax
    TensorLite softmax_derivative = inputs.apply([](double_t x) {
        double_t ex = std::exp(x);
        return ex / (ex * ex);
    });

    // Compute the gradient using the chain rule
    TensorLite result = softmax_derivative.multiplyPairWise(output_errors);

    return result;
}