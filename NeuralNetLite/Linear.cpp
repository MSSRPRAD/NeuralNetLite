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

// Convolution Layer
Conv::Conv(std::vector<size_t> input_dim, std::vector<size_t> kernel_size){
    inputs = TensorLite(input_dim);
    kernel = TensorLite(kernel_size);
}

void Conv::print() const {
    std::cout<<"Inputs:\n\n";
    inputs.print();
    std::cout<<"\n\nKernel:\n\n";
    kernel.print();
}

TensorLite Conv::forward(const TensorLite &input){
    inputs = input;

    // Some Checks to prevent errors
    assert(input.dim.size() == kernel.dim.size());
    if(inputs.dim.size() == 3 && kernel.dim.size() == 3){
        assert(inputs.dim.size() == 3 && kernel.dim.size() == 3 && inputs.dim[0] == kernel.dim[0]);
    }
    for(size_t i = 0; i < inputs.dim.size(); i++){
        assert(kernel.dim[i] <= inputs.dim[i]);
    }

    // Convolution Operation (assuming there is some depth)
    const size_t input_height = input.dim[1];
    const size_t input_width = input.dim[2];
    const size_t input_depth = input.dim[0];
    const size_t kernel_height = kernel.dim[1];
    const size_t kernel_width = kernel.dim[2];
    const size_t kernel_depth = kernel.dim[0];

    TensorLite output = TensorLite({kernel_depth, input_height-kernel_height+1, input_width-kernel_width+1});
    const size_t output_height = output.dim[1];
    const size_t output_width = output.dim[2];
    const size_t output_depth = output.dim[0];

    for(size_t row = 0; row <= input_height-kernel_height+1; row++){
        for(size_t col = 0; col <= input_width-kernel_width+1; col++){
            // std::cout<<"row: "<<row<<"| col: "<<col<<"\n";
            // Fill for all depths
            for(int d = 0; d < kernel_depth; d++){
                // std::cout<<"d: "<<d<<"\n";
                double_t sum = 0.0;
                for(size_t k = row; k < row+kernel_height; k++){
                    for(size_t l = col; l < col+kernel_width; l++){
                        size_t inputIdx = col + row*input_width + d*input_width*input_height;
                        size_t kernelIdx = (l-col) + (k-row)*kernel_width + d*(kernel_width*kernel_height);
                        // std::cout<<"inputs.data[inputIdx]: "<<inputs.data[inputIdx]<<"| kernel.data[kernelIdx]: "<<kernel.data[kernelIdx]<<"\n";
                        sum += inputs.data[inputIdx]*kernel.data[kernelIdx];
                        // std::cout<<"sum: "<<sum<<"\n";
                    }
                }
                // if(d==1) col-=output_width; row-=output_height;
                size_t outputIdx = col + row*output_width +  d*output_height*output_width;
                output.data[outputIdx] = sum;
                std::cout<<"keeping sum="<<sum<<"| in outputIdx="<<outputIdx<<"\n";
            }
        }
    }

    return output;
}

void Conv::setKernel(const TensorLite &new_kernel){
    kernel = new_kernel;
}


// To Do
TensorLite Conv::backward(const TensorLite &output_errors){
    return output_errors;
}