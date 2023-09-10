#include "MatUtils.hpp"
#include "Metrics.hpp"
#include <cassert>
#include <iostream>
#include <vector>

double_t MSE(const TensorLite& predictions, const TensorLite& labels) {
    assert (predictions.dim != labels.dim);

    double_t mse = 0.0;
    size_t totalElements = predictions.data.size();

    for (size_t i = 0; i < totalElements; ++i) {
        double_t diff = predictions.data[i] - labels.data[i];
        mse += diff * diff;
    }

    return mse / totalElements;
}

double_t sigmoid(double_t input) {
    return 1.0 / (1.0 + std::exp(-input));
}

double_t sigmoidDer(double_t input) {
    double_t sigmoid_output = sigmoid(input);
    return sigmoid_output * (1.0 - sigmoid_output);
}

double_t tanH(double_t input) {
    return std::tanh(input);
}

double_t tanHDer(double_t input) {
    double_t tanh_output = tanh(input);
    return 1.0 - (tanh_output * tanh_output);
}

double_t relu(double_t input) {
    return std::max(0.0, input);
}

double_t reluDer(double_t input) {
    return (input > 0.0) ? 1.0 : 0.0;
}