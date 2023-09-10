#pragma once
#include "FeedForwardNet.hpp"
#include "Linear.hpp"
#include "MatUtils.hpp"
#include "Metrics.hpp"
#include <iostream>
#include <vector>

// Constructor
FeedForwardNet::FeedForwardNet()
{
    // Do nothing
}

// Add Layer
void FeedForwardNet::addLinearLayer(Linear *layer)
{
    layers.push_back(layer);
}

// Feed Forward
TensorLite FeedForwardNet::forward(const TensorLite& input)
{
    TensorLite output = input;
    for (auto layer : layers) {
        output = layer->forward(output);
    }
    return output;
}

// Backward Propogation
TensorLite FeedForwardNet::backward(const TensorLite& output_errors)
{
    TensorLite out = output_errors;
    for (auto layer : layers) {
        out = layer->backward(output_errors);
    }
    return out;
}

void FeedForwardNet::fit(const TensorLite& X_train, const TensorLite& Y_train, size_t epochs)
{
    size_t max = epochs;
    while (epochs--) {
        for (size_t i = 0; i < X_train.dim[0]; i++) {

            // Get target and input
            TensorLite target = TensorLite({ 1, Y_train.dim[1] });
            std::vector<double_t> target_data;
            for (size_t j = 0; j < Y_train.dim[1]; j++) {
                target_data.push_back(Y_train.data[Y_train.dim[1] * i + j]);
            }
            target.setData(target_data);
            std::vector<double_t> input_data;
            TensorLite input = TensorLite({ 1, X_train.dim[1] });
            for (size_t j = 0; j < X_train.dim[1]; j++) {
                input_data.push_back(X_train.data[X_train.dim[1] * i + j]);
            }
            input.setData(input_data);

            // Forward pass
            TensorLite output = forward(input);

            // Backward pass
            TensorLite output_errors = output.sub(target);
            backward(output_errors);

            // Print stats
            double_t loss_mse = MSE(output, target);
            std::cout << "Epoch: " << epochs << "/" << max << std::endl;
            std::cout << "Loss (MSE): " << loss_mse << std::endl;
        };
    }
}