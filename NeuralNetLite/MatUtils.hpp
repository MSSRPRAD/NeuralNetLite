#pragma once

#include <functional>
#include <iostream>
#include <random>
#include <vector>
class TensorLite {

public:
    // Constructor
    TensorLite();
    TensorLite(std::vector<size_t> dimensions);

    // Accessory Functions
    void setData(const std::vector<double_t>& Data);

    // Common Matrix Operations
    TensorLite add(const TensorLite& other) const;
    void addInPlace(const TensorLite& other);

    TensorLite sub(const TensorLite& other) const;
    void subInPlace(const TensorLite& other);

    TensorLite multiply(const TensorLite& other) const;

    TensorLite multiplyByConstant(double_t constant) const;
    void multiplyByConstantInPlace(double_t constant);

    TensorLite multiplyPairWise(const TensorLite& other) const;
    void multiplyPairWiseInPlace(const TensorLite& other);

    void fill(size_t val);
    void fill();

    void reshape(std::vector<size_t> dimensions);

    TensorLite transpose() const;

    double_t sum() const;

    TensorLite iloc(size_t pos) const;

    void applyInPlace(std::function<double_t(double_t)> func)
    {
        for (double_t& value : data) {
            value = func(value);
        }
    }

    TensorLite apply(const std::function<double_t(double_t)> func) const
    {
        TensorLite result = TensorLite(dim);
        size_t SIZE = data.size();
        for (size_t i = 0; i < SIZE; ++i) {
            result.data[i] = func(data[i]);
        }
        return result;
    }

    // Print
    void print() const;
    void print_dim(size_t d) const;

    // Attributes
    std::vector<size_t> dim;
    std::vector<double_t> data;
};
