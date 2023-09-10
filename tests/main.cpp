#include "../NeuralNetLite/FeedForwardNet.hpp"
#include "../NeuralNetLite/Linear.hpp"
#include "../NeuralNetLite/MatUtils.hpp"
#include "../NeuralNetLite/Metrics.hpp"
#include <iostream>
#include <vector>

int main()
{

    std::cout << "\n\nCompiling Successfully!\n\n";

    std::cout << "Testing XOR Problem\n\n"
              << std::endl;

    // Dataset

    TensorLite X_train({ 4, 2 });
    X_train.setData({ 0, 0, 0, 1, 1, 0, 1, 1 });
    X_train.print();
    std::cout << "\n\n";
    TensorLite Y_train({ 4, 1 });
    Y_train.setData({ 0, 1, 1, 0 });
    Y_train.print();
    std::cout << "\n\n";

    // Model
    FeedForwardNet neuralNetwork = FeedForwardNet();
    neuralNetwork.addLinearLayer(DenseLayer(2, 2, 0.1));
    neuralNetwork.addLinearLayer(Activation(sigmoid, sigmoidDer, 2));
    neuralNetwork.addLinearLayer(DenseLayer(2, 1, 0.1));
    neuralNetwork.addLinearLayer(Activation(sigmoid, sigmoidDer, 1));
    neuralNetwork.fit(X_train, Y_train, 100);
    return 0;
}
