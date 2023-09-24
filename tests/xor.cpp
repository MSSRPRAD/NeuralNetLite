#include "../NeuralNetLite/FeedForwardNet.hpp"
#include "../NeuralNetLite/Linear.hpp"
#include "../NeuralNetLite/MatUtils.hpp"
#include "../NeuralNetLite/Metrics.hpp"
#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>


void generateXORLikeSample(TensorLite &inputTensor, TensorLite &targetTensor, size_t index) {
    double x = static_cast<double_t>(rand()) / RAND_MAX;
    double y = static_cast<double_t>(rand()) / RAND_MAX;

    inputTensor.data[index * 2] = x;
    inputTensor.data[index * 2 + 1] = y;

    if (x < 0.4 && y < 0.4) {
        targetTensor.data[index * 2] = 1;
        targetTensor.data[index * 2 + 1] = 0; // First quadrant
    } else if (x >= 0.6 && y < 0.4) {
        targetTensor.data[index * 2] = 0;
        targetTensor.data[index * 2 + 1] = 1; // Second quadrant
    } else if (x < 0.4 && y >= 0.6) {
        targetTensor.data[index * 2] = 0;
        targetTensor.data[index * 2 + 1] = 1; // Third quadrant
    } else if(x >= 0.6 && y >= 0.6) {
        targetTensor.data[index * 2] = 1;
        targetTensor.data[index * 2 + 1] = 0; // Fourth quadrant
    }
}

int main()
{

    std::cout << "\n\nCompiling Successfully!\n\n";

    std::cout << "Testing XOR Problem\n\n"
              << std::endl;

    // Dataset

    // srand(static_cast<unsigned int>(time(nullptr)));

    // // Create a dataset with 500 data points
    // TensorLite X_train({50, 2});
    // TensorLite Y_train({50, 2});

    // for (size_t i = 0; i < 50; ++i) {
    //     generateXORLikeSample(X_train, Y_train, i);
    // }

    TensorLite X_train = TensorLite({4, 2});
    TensorLite Y_train = TensorLite({4, 2});
    X_train.setData({0, 0, 0, 1, 1, 0, 1, 1});
    Y_train.setData({1, 0, 0, 1, 0, 1, 1, 0});

    // Print the generated dataset
    X_train.print();
    std::cout << "\n\n";
    Y_train.print();
    std::cout << "\n\n";

    // Model
    double_t LR = 0.01;
    FeedForwardNet neuralNetwork = FeedForwardNet();
    neuralNetwork.addLinearLayer(new DenseLayer(2, 5, LR));
    neuralNetwork.addLinearLayer(new Activation(relu, reluDer, 5));
    neuralNetwork.addLinearLayer(new DenseLayer(5, 2, LR));
    neuralNetwork.addLinearLayer(new Activation(sigmoid, sigmoidDer, 2));
    neuralNetwork.fit(X_train, Y_train, 10000);
    return 0;
}
