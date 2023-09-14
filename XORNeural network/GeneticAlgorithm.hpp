//
//  GeneticAlgorithm.hpp
//  XORNeural network
//
//  Created by Aldric Labarthe on 13/09/2023.
//
#ifndef GeneticAlgorithm_hpp
#define GeneticAlgorithm_hpp

#include <stdio.h>
#include <cmath>
#include <functional>
#include <array>

#include "NeuralNetwork.hpp"
#include "NeuralLayer.hpp"
#include "Matrix.hpp"

class BaseGeneticAlgorithm {
    
    
public:
    template <typename NEURAL_NET_TYPE, typename NEURAL_LAYER_TYPE>
    static void mergeLayers(const std::size_t layer, NEURAL_NET_TYPE* parent1, NEURAL_NET_TYPE* parent2, NEURAL_NET_TYPE& child){
        NEURAL_LAYER_TYPE* parent1_layer = (NEURAL_LAYER_TYPE*)parent1->getLayer(layer);
        NEURAL_LAYER_TYPE* parent2_layer = (NEURAL_LAYER_TYPE*)parent2->getLayer(layer);
        
    }
};


template <size_t NBLAYERS, size_t INPUTNUMBER, size_t OUTPUTNUMBER, size_t TRAINING_LENGTH, size_t POP_SIZE=20>
class GeneticAlgorithm : public BaseGeneticAlgorithm{
    std::array<Matrix <float, INPUTNUMBER, 1>, TRAINING_LENGTH> inputs;
    std::array<Matrix <float, OUTPUTNUMBER, 1>, TRAINING_LENGTH> outputs;
    std::function<void(NeuralNetwork<NBLAYERS>&)> NNInstructions;
    std::array<NeuralNetwork<NBLAYERS>*, POP_SIZE> networksPopulation;
    
    
public:
    inline GeneticAlgorithm(const std::array<Matrix <float, INPUTNUMBER, 1>, TRAINING_LENGTH> in, const std::array<Matrix <float, OUTPUTNUMBER, 1>, TRAINING_LENGTH> out, const std::function<void(NeuralNetwork<NBLAYERS>&)> instructions) : inputs(in), outputs(out), NNInstructions(instructions) {
        for (std::size_t i(0); i < POP_SIZE; i++){
            networksPopulation[i] = new NeuralNetwork<NBLAYERS>;
            NNInstructions(*networksPopulation[i]);
        }
        
    };
    
    inline ~GeneticAlgorithm() {
        for (std::size_t i(0); i < POP_SIZE; i++){
            if (networksPopulation[i] != nullptr)
                delete networksPopulation[i];
        }
    };
    
    
    /*
     Choices about generation of new population :
     Top 1/3 is selected. They multiply and their offspring represents 2/3 of new generation
     The remainder is generated randomly (introduction of immigration)
     */
    
    inline /*NeuralNetwork<NBLAYERS>**/ void trainNetworks(const int iterations,
        const std::function<void(NeuralNetwork<NBLAYERS>&, NeuralNetwork<NBLAYERS>*, NeuralNetwork<NBLAYERS>*)> merging_instructions){
        std::array< std::array< Matrix <float, OUTPUTNUMBER, 1>* , TRAINING_LENGTH>, POP_SIZE> localoutputs;
        std::array< float, POP_SIZE> MSE;
        for (std::size_t netid(0); netid < POP_SIZE; netid++){
            MSE[netid] = 0;
            for (std::size_t inputid(0); inputid < TRAINING_LENGTH; inputid++){
                localoutputs[netid][inputid] = networksPopulation[netid]->template process<INPUTNUMBER, OUTPUTNUMBER>(inputs[inputid]);
                float localMSE(0);
                std::cout << "Processing MSE : Local output :";
                std::cout << *localoutputs[netid][inputid];
                std::cout << "Expected output :";
                std::cout << outputs[inputid];
                for (std::size_t outputIncr(0); outputIncr < OUTPUTNUMBER; outputIncr++){
                    localMSE += std::pow(localoutputs[netid][inputid]->read(outputIncr+1, 1) - outputs[inputid].read(outputIncr+1, 1), 2);
                }
                MSE[netid] += localMSE;
            }
            std::cout << "MSE of " << MSE[netid] << std::endl;
        }
    }
    
};


#endif /* GeneticAlgorithm_hpp */
