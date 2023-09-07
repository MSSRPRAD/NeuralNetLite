#pragma once

#include <iostream>
#include <vector>
class TensorLite {

public:
  // Constructor
  TensorLite(std::vector<size_t> dimensions);

  // Accessory Functions
  void setData(const std::vector<double> &Data);

  // Common Matrix Operations
  TensorLite add(const TensorLite &other) const;
  void addInPlace(const TensorLite &other);

  TensorLite multiply(const TensorLite &other) const;

  TensorLite multiplyByConstant(double constant) const;
  void multiplyByConstantInPlace(double constant);

  TensorLite multiplyPairWise(const TensorLite &other) const;
  void multiplyPairWiseInPlace(const TensorLite &other);

  TensorLite transpose() const;

  // Print
  void print() const;

  // Activation Functions

  TensorLite sigmoid() const;
  TensorLite sigmoidInPlace();
  TensorLite sigmoidDer() const;
  TensorLite sigmoidDerInPlace() const;

  TensorLite relu() const;
  TensorLite reluInPlace();
  TensorLite reluDer() const;
  TensorLite reluDerInPlace() const;

  TensorLite tanh() const;
  TensorLite tanhInPlace();
  TensorLite tanhDer() const;
  TensorLite tanhDerInPlace() const;

  // Error Functions
  TensorLite meanSquaredError(const TensorLite &target) const;
  TensorLite meanSquaredErrorDer(const TensorLite &target) const;

  // Attributes
  std::vector<size_t> dim;
  std::vector<double> data;

};
