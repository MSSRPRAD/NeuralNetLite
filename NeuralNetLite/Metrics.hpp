#pragma once

#include "MatUtils.hpp"

static double MSE(const TensorLite &predictions, const TensorLite &labels);
double sigmoid(double input);
double sigmoidDer(double input);
double relu(double input);
double reluDer(double input);
double tanH(double input);
double tanHDer(double input);