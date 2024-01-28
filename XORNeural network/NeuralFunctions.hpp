//
//  NeuralFunctions.hpp
//  XORNeural network
//
//  Created by Aldric Labarthe on 27/01/2024.
//

#ifndef NeuralFunctions_hpp
#define NeuralFunctions_hpp

#include "AKML_consts.hpp"
#include "Matrices.hpp"
#include <stdio.h>
#include <cmath>
#include <functional>

namespace akml {
    template<akml::Arithmetic T>
    struct ActivationFunction {
        const std::function<T(T)> function;
        const std::function<T(T)> derivative;
    };

    /*
    To be continued...
    template<akml::Arithmetic T, akml::MatrixInterfaceConcept<T> MATRIX_TYPE>
    struct ErrorFunction {
        const std::function<T(MATRIX_TYPE&, MATRIX_TYPE&)> function;
        const std::function<T(MATRIX_TYPE&, MATRIX_TYPE&)> derivative;
    };*/

}

#endif /* NeuralFunctions_hpp */
