//
//  NeuralFunctions.cpp
//  XORNeural network
//
//  Created by Aldric Labarthe on 27/01/2024.
//

#include "NeuralFunctions.hpp"

namespace akml {
    namespace ActivationFunctions {
        const struct ActivationFunction<float> SIGMOID = {
            .function = [](float x) {return 1/(1+exp(-x));},
            .derivative = [](float x) { return exp(-x)/std::pow(1+exp(-x),2); }
        };

        const struct ActivationFunction<float> RELU = {
            .function = [](float x) {return std::max(0.f, x); },
            .derivative = [](float x) { return (x > 0) ? 1.f : 0.f; }
        };

        const struct ActivationFunction<float> NO_ACTION = {
            .function = [](float x) {return x;},
            .derivative = [](float x) { return 1; }
        };
    }
}
