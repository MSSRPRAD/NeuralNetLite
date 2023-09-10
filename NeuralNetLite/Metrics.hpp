#pragma once

#include "MatUtils.hpp"

double_t MSE(const TensorLite& predictions, const TensorLite& labels);
double_t sigmoid(double_t input);
double_t sigmoidDer(double_t input);
double_t relu(double_t input);
double_t reluDer(double_t input);
double_t tanH(double_t input);
double_t tanHDer(double_t input);