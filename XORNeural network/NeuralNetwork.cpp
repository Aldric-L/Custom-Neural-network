//
//  NeuralNetwork.cpp
//  XORNeural network
//
//  Created by Aldric Labarthe on 07/09/2023.
//

#include "NeuralNetwork.hpp"

namespace akml {

template <const std::size_t NBLAYERS>
std::function<float(float)> NeuralNetwork<NBLAYERS>::SIGMOID = [](float x) {return 1/(1+exp(-x));};

template <const std::size_t NBLAYERS>
std::function<float(float)> NeuralNetwork<NBLAYERS>::NO_ACTION = [](float x) {return x;};

}
