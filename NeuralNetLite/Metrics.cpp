#include "MatUtils.hpp"
#include "Metrics.hpp"
#include <cassert>
#include <iostream>
#include <vector>

double MSE(const TensorLite& predictions, const TensorLite& labels) {
    assert (predictions.dim != labels.dim);

    double mse = 0.0;
    size_t totalElements = predictions.data.size();

    for (size_t i = 0; i < totalElements; ++i) {
        double diff = predictions.data[i] - labels.data[i];
        mse += diff * diff;
    }

    return mse / totalElements;
}

double sigmoid(double input) {
    return 1.0 / (1.0 + std::exp(-input));
}

double sigmoidDer(double input) {
    double sigmoid_output = sigmoid(input);
    return sigmoid_output * (1.0 - sigmoid_output);
}

double tanH(double input) {
    return std::tanh(input);
}

double tanHDer(double input) {
    double tanh_output = tanh(input);
    return 1.0 - (tanh_output * tanh_output);
}

double relu(double input) {
    return std::max(0.0, input);
}

double reluDer(double input) {
    return (input > 0.0) ? 1.0 : 0.0;
}