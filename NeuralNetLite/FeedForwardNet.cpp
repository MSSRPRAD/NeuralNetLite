#pragma once
#include "FeedForwardNet.hpp"
#include"MatUtils.hpp"
#include"Metrics.hpp"
#include"Linear.hpp"
#include<iostream>
#include<vector>

// Constructor
FeedForwardNet::FeedForwardNet(){
    // Do nothing
}

// Add Layer
void FeedForwardNet::addLinearLayer(Linear *layer){
    layers.push_back(layer);
}

// Feed Forward
TensorLite FeedForwardNet::forward(const TensorLite &input) {
    TensorLite output = input;
    for (auto layer : layers) {
        output = layer->forward(output);
    }
    return output;
}

// Backward Propogation
TensorLite FeedForwardNet::backward(const TensorLite &output_errors) {
    TensorLite output_errors = output_errors;
    for (auto it : layers) {
        output_errors = it->backward(output_errors);
    }
    return output_errors;
}

void FeedForwardNet::fit(const TensorLite &X_train, const TensorLite &Y_train, size_t epochs) {
    size_t max = epochs;
    while(epochs--){
        for(size_t i = 0; i < X_train.dim[0]; i++){
            
            // Get X_train and Y_train
            TensorLite target = TensorLite({1, Y_train.dim[1]});
            std::vector<double> target_data;
            for(size_t j = 0; j < Y_train.dim[1]; j++){
                target_data.push_back(Y_train.data[Y_train.dim[1]*i+j]);
            }
            TensorLite input = TensorLite({1, X_train.dim[1]});
            for(size_t j = 0; j < X_train.dim[1]; j++){
                target_data.push_back(X_train.data[Y_train.dim[1]*i+j]);
            }

            // Forward pass
            TensorLite output = forward(input);

            // Backward pass
            TensorLite output_errors = output.sub(target);
            backward(output_errors);
            
            // Print stats
            double_t loss_mse = MSE(output, target);
            std::cout<<"Epoch: "<<epochs<<"/"<<max<<std::endl;
            std::cout<<"Loss (MSE): "<<loss_mse<<std::endl;

        };
    }
}