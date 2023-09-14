//
//  Matrix.cpp
//  XORNeural network
//
//  Created by Aldric Labarthe on 08/09/2023.
//

#include "Matrix.hpp"


template <typename element_type, std::size_t ROWS, std::size_t COLUMNS>
std::function<element_type(element_type, std::size_t, std::size_t)> Matrix<element_type, ROWS, COLUMNS>::NO_ACTION_TRANSFORM = [](element_type x, std::size_t row, std::size_t column) {return x;};

template <typename element_type, std::size_t ROWS, std::size_t COLUMNS>
std::function<element_type(element_type, std::size_t, std::size_t)> Matrix<element_type, ROWS, COLUMNS>::IDENTITY_TRANSFORM = [](element_type x, std::size_t row, std::size_t column) { return ((column%2 != 0 && row%2 != 0) || (column%2 == 0 && row%2 == 0)) ? 1 : 0;};

template <typename element_type, std::size_t ROWS, std::size_t COLUMNS>
std::function<element_type(element_type, std::size_t, std::size_t)> Matrix<element_type, ROWS, COLUMNS>::RANDOM_TRANSFORM = [](element_type x, std::size_t row, std::size_t column) { std::random_device rd;  std::mt19937 gen(rd()); std::normal_distribution<double> distribution(0.0,3); return distribution(gen); };



template <typename element_type, std::size_t ROWS, std::size_t COLUMNS>
Matrix<element_type, ROWS, COLUMNS> Matrix<element_type, ROWS, COLUMNS>::EMPTY (true);

template <typename element_type, std::size_t ROWS, std::size_t COLUMNS>
Matrix<element_type, ROWS, COLUMNS> Matrix<element_type, ROWS, COLUMNS>::IDENTITY = Matrix<element_type, ROWS, COLUMNS>::transform(Matrix<element_type, ROWS, COLUMNS>::EMPTY, Matrix<element_type, ROWS, COLUMNS>::IDENTITY_TRANSFORM);
