#include <iostream>
#include <vector>
#include "../NeuralNetLite/MatUtils.hpp"

int main() {
    // Create two tensors
    // TensorLite tensor1({3, 3});
    // TensorLite tensor2({3, 3});
    TensorLite tensor3({3, 2});

    // Populate the tensors with some data (for demonstration purposes)
    // std::vector<double> data1 = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    // std::vector<double> data2 = {9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0};
    std::vector<double> data3 = {9.0, 8.0, 7.0, 6.0, 5.0, 4.0};
    // tensor1.setData(data1);
    // tensor2.setData(data2);
    tensor3.setData(data3);
    return 1;
    // Print the original tensors
    // std::cout << "Tensor 1:" << std::endl;
    // tensor1.print();
    // std::cout << "Tensor 2:" << std::endl;
    // tensor2.print();
    // std::cout << "Tensor 3:" << std::endl;
    // tensor3.print();

    // // Perform some operations
    // TensorLite sum = tensor1.add(tensor2);
    // TensorLite product = tensor1.multiply(tensor2);
    // TensorLite product1 = tensor1.multiply(tensor3);
    // TensorLite scaled = tensor1.multiplyByConstant(2.0);
    // TensorLite transposed = tensor3.transpose();
    // TensorLite pairwise = tensor1.multiplyPairWise(tensor2);

    // // Print the results
    // std::cout << "Sum:" << std::endl;
    // sum.print();
    // std::cout << "Product (Tensor1xTensor2):" << std::endl;
    // product.print();
    // std::cout << "Product (Tensor1xTensor3)" << std::endl;
    // product1.print();
    // std::cout << "Multiply Pair Wise (Tensor1xTensor1)" << std::endl;
    // pairwise.print();
    // std::cout << "Tensor1x2" << std::endl;
    // scaled.print();
    // std::cout << "Transposed:" << std::endl;
    // transposed.print();

    return 0;
}
