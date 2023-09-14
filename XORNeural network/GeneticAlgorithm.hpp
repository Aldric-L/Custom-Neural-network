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
#include <algorithm>
#include <utility>

#include "NeuralNetwork.hpp"
#include "NeuralLayer.hpp"
#include "Matrix.hpp"

class BaseGeneticAlgorithm {
    
public:
    template <typename NEURAL_NET_TYPE, typename NEURAL_LAYER_TYPE>
    static void mergeLayers(const std::size_t layer, NEURAL_NET_TYPE* parent1, NEURAL_NET_TYPE* parent2, NEURAL_NET_TYPE& child){
        NEURAL_LAYER_TYPE* parent1_layer = (NEURAL_LAYER_TYPE*)parent1->getLayer(layer);
        NEURAL_LAYER_TYPE* parent2_layer = (NEURAL_LAYER_TYPE*)parent2->getLayer(layer);
        NEURAL_LAYER_TYPE* child_layer = (NEURAL_LAYER_TYPE*)child.getLayer(layer);
        
        // Biases :
        for (std::size_t row(0); row < child_layer->getNeuronNumber(); row++){
            if (row %2 == 0)
                child_layer->getBiasesAccess()->operator()(row, 1) = parent1_layer->getBiasesAccess()->operator()(row, 1);
            else
                child_layer->getBiasesAccess()->operator()(row, 1) = parent2_layer->getBiasesAccess()->operator()(row, 1);
            /*child_layer->getBiasesAccess()->operator()(row+1, 1) = (parent1_layer->getBiasesAccess()->operator()(row+1, 1)+parent2_layer->getBiasesAccess()->operator()(row+1, 1))/2;*/
        }
        /*if (parent2_layer != parent1_layer){
            std::cout << "For instance, we had randomly in weights : " << std::endl;
            std::cout << *(child_layer->getWeightsAccess());
        }*/
        
        // weights :
        for (std::size_t row(0); row < child_layer->getNeuronNumber(); row++){
            for (std::size_t col(0); col < child_layer->getPreviousNeuronNumber(); col++){
                if (((child_layer->getNeuronNumber() > 1) && (row+col) %2 == 0) || col%2==0)
                    child_layer->getWeightsAccess()->operator()(row, col) = parent1_layer->getWeightsAccess()->operator()(row, col);
                else
                    child_layer->getWeightsAccess()->operator()(row, col) = parent2_layer->getWeightsAccess()->operator()(row, col);
                
                    /*child_layer->getWeightsAccess()->operator()(row+1, col+1) = (parent1_layer->getWeightsAccess()->operator()(row+1, col+1) + parent2_layer->getWeightsAccess()->operator()(row+1, col+1))/2;*/
            }
        }
        
        /*if (parent2_layer != parent1_layer){
            std::cout << "Parent 1: " << std::endl;
            std::cout << *(parent1_layer->getWeightsAccess());
            std::cout << "Parent 2: " << std::endl;
            std::cout << *(parent2_layer->getWeightsAccess());
            std::cout << "Final : " << std::endl;
            std::cout << *(child_layer->getWeightsAccess());
            
        }*/
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
     Top 2/3 is selected. They multiply and their offspring represents less than 2/3 of new generation
     The remainder is generated randomly (introduction of immigration)
     */
    
    inline NeuralNetwork<NBLAYERS>* trainNetworks(const int iterations,
        const std::function<void(NeuralNetwork<NBLAYERS>&, NeuralNetwork<NBLAYERS>*, NeuralNetwork<NBLAYERS>*)> merging_instructions){
        for (std::size_t iteration(1); iteration <= iterations; iteration++){
            std::array< std::array< Matrix <float, OUTPUTNUMBER, 1>* , TRAINING_LENGTH>, POP_SIZE> localoutputs;
            std::array< std::pair<float, NeuralNetwork<NBLAYERS>*>, POP_SIZE> MSE;
            for (std::size_t netid(0); netid < POP_SIZE; netid++){
                float MSE_i = 0;
                for (std::size_t inputid(0); inputid < TRAINING_LENGTH; inputid++){
                    localoutputs[netid][inputid] = networksPopulation[netid]->template process<INPUTNUMBER, OUTPUTNUMBER>(inputs[inputid]);
                    float localMSE(0);
                    //std::cout << "Processing MSE : Local output :";
                    //std::cout << *localoutputs[netid][inputid];
                    //std::cout << "Expected output :";
                    //std::cout << outputs[inputid];
                    for (std::size_t outputIncr(0); outputIncr < OUTPUTNUMBER; outputIncr++){
                        localMSE += std::pow(localoutputs[netid][inputid]->read(outputIncr+1, 1) - outputs[inputid].read(outputIncr+1, 1), 2);
                    }
                    MSE_i += localMSE;
                }
                MSE_i = MSE_i/POP_SIZE;
                //std::cout << "MSE of " << MSE_i << std::endl;
                MSE[netid] = {MSE_i, networksPopulation[netid]};
            }
            std::sort(MSE.begin(), MSE.end());
            std::reverse(MSE.begin(), MSE.end());
            std::array<NeuralNetwork<NBLAYERS>*, POP_SIZE> newNetworksPopulation;
            
            //std::cout << "\n\n\n\nTriage et reproduction" << std::endl;
            
            float mean_MSE(0);
            for (std::size_t i(0); i < POP_SIZE; i++){
                //std::cout << "Processing MSE=" << MSE[i].first << std::endl;
                if (i < std::round(POP_SIZE/6)){
                    newNetworksPopulation[i] = new NeuralNetwork<NBLAYERS>;
                    NNInstructions(*newNetworksPopulation[i]);
                }else if (std::round(POP_SIZE/6) <= i && i < std::round(5*POP_SIZE/6)) {
                    //std::cout << "\nWe merge it" << std::endl;
                    newNetworksPopulation[i] = new NeuralNetwork<NBLAYERS>;
                    NNInstructions(*newNetworksPopulation[i]);
                    merging_instructions(*newNetworksPopulation[i], networksPopulation[i-1], networksPopulation[i]);
                }else if (std::round(5*POP_SIZE/6) <= i){
                    //std::cout << "\nWe keep it" << std::endl;
                    newNetworksPopulation[i] = new NeuralNetwork<NBLAYERS>;
                    NNInstructions(*newNetworksPopulation[i]);
                    merging_instructions(*newNetworksPopulation[i], networksPopulation[i], networksPopulation[i]);
                }
                mean_MSE += MSE[i].first;
            }
            std::cout << "Best MSE = " << MSE[POP_SIZE-1].first << " - Mean=" << mean_MSE/POP_SIZE << std::endl;
            for (std::size_t i(0); i < POP_SIZE; i++){
                delete networksPopulation[i];
                networksPopulation[i] = newNetworksPopulation[i];
            }
        }
        
        return networksPopulation[POP_SIZE-1];
        
    }
    
};


#endif /* GeneticAlgorithm_hpp */
