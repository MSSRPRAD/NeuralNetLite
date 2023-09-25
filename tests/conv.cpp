#include "../NeuralNetLite/Linear.hpp"
#include "../NeuralNetLite/MatUtils.hpp"
#include <iostream>
#include <vector>
#include <ctime>
#include <random>
#include <cstdlib>

size_t getRandom(){
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<int> distbn(0, 2);
    return distbn(gen);
}

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
    std::vector<double_t> input_data(32);
    for(size_t i = 0; i < 32; i++){
        input_data[i] = static_cast<double_t>(getRandom());
    }
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
    output.print_dim(1);

    std::cout<<"Exiting.....";

    return 0;
}