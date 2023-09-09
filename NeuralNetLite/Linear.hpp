#pragma once
#include"MatUtils.hpp"
#include<iostream>
#include<vector>

// Linear Transforms a TensorLite
class Linear {
    public:
    // Forward Propogation
        virtual TensorLite forward(const TensorLite &input) const;
    // Backward Propogation
        virtual TensorLite backward(const TensorLite &input);
};

// Dense Layer
class DenseLayer : public Linear {
    public:
        DenseLayer(size_t input_size, size_t output_size);

        TensorLite forward(const TensorLite &input) const override;

        TensorLite backward(const TensorLite &input) override;

        // Attributes
        TensorLite input;
        TensorLite weights;
        TensorLite bias;
};

// Activation Layer
class Activation : public Linear {
    public:
        Activation(std::function<double(double)> ac, std::function<double(double)> acDer, size_t output_size);
        TensorLite forward(const TensorLite &input) const override;
        TensorLite backward(const TensorLite &input) override;

        // Attributes
        TensorLite inputs;
        std::function<double(double)> activation;
        std::function<double(double)> activationDer;
};