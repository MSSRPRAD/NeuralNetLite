#include "../NeuralNetLite/Linear.hpp"
#include "../NeuralNetLite/MatUtils.hpp"
#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>

int main()
{
    // Create input and kernel
    std::vector<size_t> input_dim = {2, 4, 4};
    std::vector<size_t> kernel_dim = {2, 2, 2};
    Conv convolution_layer(input_dim, kernel_dim);
    TensorLite kernel = TensorLite(kernel_dim);
    kernel.setData({1, -2, -1, 1, 1, 2, -1, -1});
    convolution_layer.setKernel(kernel);
    TensorLite inputs(input_dim);
    std::vector<double_t> input_data(32, 4.0);
    inputs.setData(input_data);

    std::cout<<"---------------DEBUG INFO---------------\n\n";

    // Convolute
    TensorLite output = convolution_layer.forward(inputs);

    std::cout<<"\n\n---------------DEBUG INFO---------------\n\n";

    // Print Stuff:
    std::cout<<"\nInput[0]:\n";
    inputs.print_dim(0);
    std::cout<<"\nInput[1]:\n";
    inputs.print_dim(1);
    std::cout<<"\nKernel[0]:\n";
    convolution_layer.kernel.print_dim(0);
    std::cout<<"\nKernel[1]:\n";
    convolution_layer.kernel.print_dim(1);
    std::cout<<"\nOutput[0]:\n";
    output.print_dim(0);
    std::cout<<"\nOutput[1]:\n";
    output.print_dim(0);

    std::cout<<"Exiting.....";

    return 0;
}