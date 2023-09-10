#pragma once
#pragma once
#include"MatUtils.hpp"
#include"Metrics.hpp"
#include"Linear.hpp"
#include<iostream>
#include<vector>

#include<iostream>
#include<vector>

class FeedForwardNet {
    public:
        FeedForwardNet();
        TensorLite forward(const TensorLite &input);
        TensorLite backward(const TensorLite &input);
        void addLinearLayer(Linear layer);
        void fit(const TensorLite &X_train, const TensorLite &Y_train, size_t epochs);
        // Attributes
        std::vector<Linear> layers;
};