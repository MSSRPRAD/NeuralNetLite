#pragma once

#include <iostream>
#include <vector>
#include <functional>
#include <random>
class TensorLite {

public:
  // Constructor
  TensorLite(std::vector<size_t> dimensions);
  
  // Accessory Functions
  void setData(const std::vector<double> &Data);

  // Common Matrix Operations
  TensorLite add(const TensorLite &other) const;
  void addInPlace(const TensorLite &other);

  TensorLite sub(const TensorLite &other) const;
  void subInPlace(const TensorLite &other);

  TensorLite multiply(const TensorLite &other) const;

  TensorLite multiplyByConstant(double constant) const;
  void multiplyByConstantInPlace(double constant);

  TensorLite multiplyPairWise(const TensorLite &other) const;
  void multiplyPairWiseInPlace(const TensorLite &other);

  void fill(size_t val);
  void fill();

  void reshape(std::vector<size_t> dimensions);

  TensorLite transpose() const;

  double sum() const;

  void applyInPlace(std::function<double(double)> func) {
        for (double &value : data) {
            value = func(value);
        }
    }

  TensorLite apply(const std::function<double(double)> func) const {
    TensorLite result = TensorLite(dim);
      size_t SIZE = data.size();
      for (size_t i = 0; i < SIZE; ++i) {
        result.data[i] = func(data[i]);
      }
    return result;
  }

  // Print
  void print() const;

  // Attributes
  std::vector<size_t> dim;
  std::vector<double> data;

};
