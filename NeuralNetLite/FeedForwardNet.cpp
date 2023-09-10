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
    for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
        out = (*it)->backward(out);
    }
    return out;
}

void FeedForwardNet::fit(const TensorLite& X_train, const TensorLite& Y_train, size_t epochs) {
    size_t max = epochs;
    for(size_t epoch = 0; epoch < max; epoch++){ 
        for (size_t i = 0; i < X_train.dim[0]; i++) {

            // Get target and input
            TensorLite target = Y_train.iloc(i);
            TensorLite input = X_train.iloc(i);

            // Forward pass
            TensorLite output = forward(input);

            // Backward pass
            TensorLite output_errors = output.sub(target);
            backward(output_errors);
            // Print stats
            if(epoch % 10 == 0){
                std::cout<<"Forward Pass Results: \n";
                input.print();
                target.print();
                output.print();
                double_t loss_mse = MSE(output, target);
                std::cout << "Epoch: " << epoch << "/" << max << std::endl;
                std::cout << "Loss (MSE): " << loss_mse << std::endl;
            }
        };
    }
}