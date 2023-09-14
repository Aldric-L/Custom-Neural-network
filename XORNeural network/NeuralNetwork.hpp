//
//  NeuralNetwork.hpp
//  XORNeural network
//
//  Created by Aldric Labarthe on 07/09/2023.
//

#ifndef NeuralNetwork_hpp
#define NeuralNetwork_hpp

#include <stdio.h>
#include <vector>
#include <array>
#include <string>
#include <stdexcept>
#include <functional>

#include "NeuralLayer.hpp"
#include "Matrix.hpp"

template <const std::size_t NBLAYERS>
class NeuralNetwork {

    
public:
    static std::function<float(float)> SIGMOID;
    static std::function<float(float)> NO_ACTION;
    
    std::array<AbstractNeuralLayer*, NBLAYERS> layers;
    
    inline NeuralNetwork() {
        for (std::size_t i(0); i < NBLAYERS; i++){
            layers[i] = nullptr;
        }
    };
    
    inline ~NeuralNetwork() {
        for (std::size_t i(0); i < NBLAYERS; i++){
            if (layers[i] != nullptr)
                delete layers[i];
        }
    };
    
    template <std::size_t INPUTNUMBER>
    inline NeuralLayer<INPUTNUMBER, 1>* setFirstLayer(){
        layers[0] = new NeuralLayer<INPUTNUMBER, 1>(0);
        layers[0]->setFirstRow(true);
        layers[0]->setActivationFunction(NeuralNetwork<NBLAYERS>::NO_ACTION);
        return (NeuralLayer<INPUTNUMBER, 1>*)layers[0];
    };
    
    template <std::size_t NEURON_NUMBER, std::size_t PREVIOUS_NEURON_NUMBER>
    inline NeuralLayer<NEURON_NUMBER, PREVIOUS_NEURON_NUMBER>* setLayer(std::size_t layer=0, const std::function<float(float)> activation_function=NeuralNetwork<NBLAYERS>::SIGMOID){
        if (layer == 0){
            for (std::size_t i(0); i < NBLAYERS; i++){
                if (layers[i] == nullptr)
                    layer = i + 1;
            }
        }

        layers[layer-1] = new NeuralLayer<NEURON_NUMBER, PREVIOUS_NEURON_NUMBER>(layer-1);
        layers[layer-1]->setActivationFunction(activation_function);
        
        Matrix<float, NEURON_NUMBER, 1> new_biases;
        new_biases.transform(Matrix<float, NEURON_NUMBER, 1>::RANDOM_TRANSFORM);
        layers[layer-1]->setBiases(&new_biases);
        Matrix<float, NEURON_NUMBER, PREVIOUS_NEURON_NUMBER> new_weights;
        new_weights.transform(Matrix<float, NEURON_NUMBER, 1>::RANDOM_TRANSFORM);
        layers[layer-1]->setWeights(&new_weights);
        return (NeuralLayer<NEURON_NUMBER, PREVIOUS_NEURON_NUMBER>*)layers[layer-1];
    };
    
    inline AbstractNeuralLayer* getLayer(std::size_t layer){
        return layers[layer-1];
    };
    
    template <std::size_t INPUTNUMBER, std::size_t OUTPUTNUMBER>
    inline Matrix<float, OUTPUTNUMBER, 1>* process(Matrix<float, INPUTNUMBER, 1> &input){
        layers[0]->setInput(&input);
        for (std::size_t i(1); i < NBLAYERS; i++){
            if (layers[i] == nullptr)
                throw std::exception();
            layers[i]->setPreviousActivationLayer(layers[i-1]->getActivationLayer());
        }
        
        Matrix<float, OUTPUTNUMBER, 1>* activLayer(0);
        activLayer = (Matrix<float, OUTPUTNUMBER, 1>*)layers[NBLAYERS-1]->getActivationLayer();
        //std::cout << *activLayer;
        return activLayer;
    }
};

#endif /* NeuralNetwork_hpp */
