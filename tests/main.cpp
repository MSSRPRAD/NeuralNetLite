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
    TensorLite Y_train({ 4, 2 });
    Y_train.setData({ 1, 0, 0, 1, 0, 1, 1, 0 });
    Y_train.print();
    std::cout << "\n\n";

    // Model
    FeedForwardNet neuralNetwork = FeedForwardNet();
    neuralNetwork.addLinearLayer(new DenseLayer(2, 6, 1));
    neuralNetwork.addLinearLayer(new Activation(relu, reluDer, 6));
    neuralNetwork.addLinearLayer(new DenseLayer(6, 3, 1));
    neuralNetwork.addLinearLayer(new Activation(relu, reluDer, 3));
    neuralNetwork.addLinearLayer(new DenseLayer(3, 2, 1));
    neuralNetwork.addLinearLayer(new Activation(sigmoid, sigmoidDer, 2));
    neuralNetwork.fit(X_train, Y_train, 100);
    return 0;
}
