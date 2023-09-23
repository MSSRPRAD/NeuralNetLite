#pragma once
#include "MatUtils.hpp"
#include <iostream>
#include <vector>

// Linear Transforms a TensorLite
class Linear {
    public:
        virtual void print() const {};
        // Forward Propogation
        virtual TensorLite forward(const TensorLite& input){};
        // Backward Propogation
        virtual TensorLite backward(const TensorLite& input){};
};

// Dense Layer
class DenseLayer : public Linear {
public:
    DenseLayer(size_t input_size, size_t output_size, double_t learning_rate);

    void print() const override;

    TensorLite forward(const TensorLite& input) override;

    TensorLite backward(const TensorLite& input) override;

    // Attributes
    double_t learning_rate;
    TensorLite inputs;
    TensorLite weights;
    TensorLite bias;
};

// Activation Layer
class Activation : public Linear {
public:
    Activation(std::function<double_t(double_t)> ac, std::function<double_t(double_t)> acDer, size_t input_size);

    void print() const override;
    TensorLite forward(const TensorLite& input) override;
    TensorLite backward(const TensorLite& input) override;

    // Attributes
    TensorLite inputs;
    std::function<double_t(double_t)> activation;
    std::function<double_t(double_t)> activationDer;
};

// Activation Layer
class SoftMax : public Linear {
public:
    SoftMax(size_t input_size);

    TensorLite forward(const TensorLite& input) override;
    TensorLite backward(const TensorLite& input) override;

    // Attributes
    TensorLite inputs;
};


// Convolution Layer
class Conv : public Linear {
public:
    Conv(std::vector<size_t> input_dim, std::vector<size_t> kernel_size);
    
    void print() const override;

    TensorLite forward(const TensorLite& input) override;
    TensorLite backward(const TensorLite& input) override;

    // Attributes
    TensorLite inputs;
    TensorLite kernel;

    // Methods
    void setKernel(const TensorLite &new_kernel);
};