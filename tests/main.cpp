#include <iostream>
#include <vector>
#include "../NeuralNetLite/MatUtils.hpp"
#include "../NeuralNetLite/Metrics.hpp"
#include "../NeuralNetLite/Linear.hpp"

int main() {
    
    std::cout << "\n\nCompiling Successfully!\n\n";
    TensorLite tensor = TensorLite({2,2});
    tensor.setData({-1,2,-3,4});
    std::cout << "Tensor:\n";
    tensor.print();
    std::cout << "\n\n";
    tensor = tensor.apply(relu);
    tensor.print();
    std::cout<<"\n\nEND\n\n";
    return 0;
}
