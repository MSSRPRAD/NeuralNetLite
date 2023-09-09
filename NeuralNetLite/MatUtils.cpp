#include "MatUtils.hpp"
#include <cassert>
#include <iostream>
#include <vector>

// Implement Constructor of TensorLite
TensorLite::TensorLite(std::vector<size_t> dimensions) : dim(dimensions) {
  size_t SIZE = 1;
  for (size_t di : dimensions) {
    SIZE *= di;
  }
  data.assign(SIZE, 0.0);
}

// Set Data
void TensorLite::setData(const std::vector<double> &Data) {
  assert(data.size() == Data.size());
  data = Data;
}

// Common Matrix Operations
TensorLite TensorLite::add(const TensorLite &other) const {
  assert(dim == other.dim);
  TensorLite result(dim);
  for (size_t i = 0; i < data.size(); ++i) {
    result.data[i] = data[i] + other.data[i];
  }

  return result;
}

void TensorLite::addInPlace(const TensorLite &other) {
  assert(dim == other.dim);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] += other.data[i];
  }
  return;
}

TensorLite TensorLite::sub(const TensorLite &other) const {
  assert(dim == other.dim);
  TensorLite result(dim);
  for (size_t i = 0; i < data.size(); ++i) {
    result.data[i] = data[i] - other.data[i];
  }

  return result;
}

void TensorLite::subInPlace(const TensorLite &other) {
  assert(dim == other.dim);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] -= other.data[i];
  }
  return;
}

TensorLite TensorLite::multiplyByConstant(double constant) const {
  TensorLite result(dim);
  for (size_t i = 0; i < data.size(); ++i) {
    result.data[i] = data[i] * constant;
  }
  return result;
}

void TensorLite::multiplyByConstantInPlace(double constant) {
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] *= constant;
  }
  return;
}

TensorLite TensorLite::multiplyPairWise(const TensorLite &other) const {
  assert(other.dim == dim);
  TensorLite result(dim);
  for (size_t i = 0; i < data.size(); ++i) {
    result.data[i] = data[i] * other.data[i];
  }
  return result;
}

void TensorLite::multiplyPairWiseInPlace(const TensorLite &other) {
  assert(other.dim == dim);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] *= other.data[i];
  }
  return;
}

TensorLite TensorLite::transpose() const {

  const std::vector<size_t> origDim = dim;

  // Original Dimensions
  const size_t origRows = origDim[0];
  const size_t origCols = origDim[1];

  // Transposed Dimensions
  const size_t newRows = origDim[1];
  const size_t newCols = origDim[0];

  // Create new Tensor
  TensorLite result({newRows, newCols});

  for (size_t row = 0; row < newRows; ++row) {
    for (size_t col = 0; col < newCols; ++col) {

      const size_t origIndex = col * origCols + row;

      const size_t transposedIndex = row * newCols + col;

      result.data[transposedIndex] = data[origIndex];
    }
  }

  return result;
}

TensorLite TensorLite::multiply(const TensorLite &other) const {
  assert(dim[1] == other.dim[0]);

  size_t resRows = dim[0];
  size_t resCols = other.dim[1];
  size_t inner = dim[1];

  TensorLite result({resRows, resCols});

  for(size_t i = 0; i < resRows; ++i){
    for (size_t j = 0; j < resCols; j++){
      result.data[i*resCols+j] = 0;
      for(size_t k = 0; k < inner; k++){
        result.data[i*resCols+j] += data[i*inner+k]*other.data[k*resCols+j];
      }
    }
  }

  return result;
}



// Hardcoded for 2d rn
void TensorLite::print() const {
  for (size_t i = 0; i < dim[0]; i++) {
    for (size_t j = 0; j < dim[1]; j++) {
      std::cout << data[i * dim[1] + j] << " ";
    }
    std::cout << "\n";
  }
}

// Sum
double TensorLite::sum() const {
  double sum = 0;
  double SIZE = data.size();
  for(size_t i = 0; i < SIZE; i++){
    sum+=data[i];
  }
  return sum;
}


void TensorLite::fill(size_t val){
    size_t SIZE = data.size();
    {
      for(size_t i = 0; i < SIZE; i++){
        data[i] = val;
      }
    }
}


void TensorLite::fill(){
    size_t SIZE = data.size();
    {
      std::random_device rd;
      std::mt19937 gen(rd());  // Mersenne Twister engine
      std::uniform_real_distribution<double> distribution(0.0, 1.0);  // Specify mean and standard deviation
      for(size_t i = 0; i < SIZE; i++){
        data[i] = distribution(gen);
      }
    }
}

void TensorLite::reshape(std::vector<size_t> dimensions){
    dim = dimensions;
    size_t SIZE = 1;
    for(auto it: dimensions){
        SIZE*=it;
    }
    data.resize(0.0, SIZE);
}