//
//  GeneticAlgorithm.hpp
//  XORNeural network
//
//  Created by Aldric Labarthe on 13/09/2023.
//
#ifndef GeneticAlgorithm_hpp
#define GeneticAlgorithm_hpp

#include <iostream>
#include <stdio.h>
#include <cmath>
#include <functional>
#include <array>
#include <algorithm>
#include <random>
#include <string>
#include <utility>

#include "NeuralNetwork.hpp"
#include "NeuralLayer.hpp"
#include "Matrix.hpp"

class GeneticAlgorithmMethods {
public:
    template <typename NEURAL_NET_TYPE, typename NEURAL_LAYER_TYPE>
    static void mergeLayers(const std::size_t layer, NEURAL_NET_TYPE* parent1, NEURAL_NET_TYPE* parent2, NEURAL_NET_TYPE& child){
        NEURAL_LAYER_TYPE* parent1_layer = (NEURAL_LAYER_TYPE*)parent1->getLayer(layer);
        NEURAL_LAYER_TYPE* parent2_layer = (NEURAL_LAYER_TYPE*)parent2->getLayer(layer);
        NEURAL_LAYER_TYPE* child_layer = (NEURAL_LAYER_TYPE*)child.getLayer(layer);
        
        if (parent2_layer == parent1_layer){
            *(child_layer->getWeightsAccess()) = *(parent2_layer->getWeightsAccess());
            *(child_layer->getBiasesAccess()) = *(parent2_layer->getBiasesAccess());
            return;
        }
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> distribution(0.0,0.15);
        
        // Biases :
        for (std::size_t row(0); row < child_layer->getNeuronNumber(); row++){
            if (row %2 == 0)
                child_layer->getBiasesAccess()->operator()(row+1, 1) = parent1_layer->getBiasesAccess()->operator()(row+1, 1);
            else
                child_layer->getBiasesAccess()->operator()(row+1, 1) = parent2_layer->getBiasesAccess()->operator()(row+1, 1);
            /*child_layer->getBiasesAccess()->operator()(row+1, 1) = (parent1_layer->getBiasesAccess()->operator()(row+1, 1)+parent2_layer->getBiasesAccess()->operator()(row+1, 1))/2;*/
            child_layer->getBiasesAccess()->operator()(row+1, 1) = child_layer->getBiasesAccess()->operator()(row+1, 1) * (1+distribution(gen));

        }
        /*if (parent2_layer != parent1_layer){
            std::cout << "For instance, we had randomly in biases : " << std::endl;
            std::cout << *(child_layer->getBiasesAccess());
        }*/
        
        // weights :
        for (std::size_t row(0); row < child_layer->getNeuronNumber(); row++){
            for (std::size_t col(0); col < child_layer->getPreviousNeuronNumber(); col++){
                if ((row+col) %2 == 0)
                    child_layer->getWeightsAccess()->operator()(row+1, col+1) = parent1_layer->getWeightsAccess()->operator()(row+1, col+1);
                else
                    child_layer->getWeightsAccess()->operator()(row+1, col+1) = parent2_layer->getWeightsAccess()->operator()(row+1, col+1);
                
                    /*child_layer->getWeightsAccess()->operator()(row+1, col+1) = (parent1_layer->getWeightsAccess()->operator()(row+1, col+1) + parent2_layer->getWeightsAccess()->operator()(row+1, col+1))/2;*/
                
                    child_layer->getWeightsAccess()->operator()(row+1, col+1) = child_layer->getWeightsAccess()->operator()(row+1, col+1) * (1+distribution(gen));
            }
        }
        /*if (parent2_layer != parent1_layer){
            std::cout << "Parent 1: " << std::endl;
            std::cout << *(parent1_layer->getBiasesAccess());
            std::cout << "Parent 2: " << std::endl;
            std::cout << *(parent2_layer->getBiasesAccess());
            std::cout << "Final : " << std::endl;
            std::cout << *(child_layer->getBiasesAccess());
            
        }*/
    }
};

template <size_t NBLAYERS, size_t INPUTNUMBER, size_t OUTPUTNUMBER, size_t POP_SIZE=20>
class BaseGeneticAlgorithm : public GeneticAlgorithmMethods {
protected:
    std::function<void(NeuralNetwork<NBLAYERS>&)> NNInstructions;
    std::array<NeuralNetwork<NBLAYERS>*, POP_SIZE> networksPopulation;
    std::function<Matrix <float, OUTPUTNUMBER, 1>(Matrix <float, OUTPUTNUMBER, 1>)>* postactivation_process = nullptr;
    std::function<void(NeuralNetwork<NBLAYERS>*, NeuralNetwork<NBLAYERS>*)>* debug_printer = nullptr;
    
public:
    static std::function<Matrix <float, OUTPUTNUMBER, 1>(Matrix <float, OUTPUTNUMBER, 1>)> ACTIVATE_ROUND;

    inline BaseGeneticAlgorithm(const std::function<void(NeuralNetwork<NBLAYERS>&)> instructions) : NNInstructions(instructions){
        for (std::size_t i(0); i < POP_SIZE; i++){
            networksPopulation[i] = new NeuralNetwork<NBLAYERS> (std::string("RANDOM_INIT"));
            NNInstructions(*networksPopulation[i]);
        }
    }
    
    inline ~BaseGeneticAlgorithm() {
        for (std::size_t i(0); i < POP_SIZE; i++){
            if (networksPopulation[i] != nullptr)
                delete networksPopulation[i];
        }
    };
    
    inline void setPostActivationProcess(std::function<Matrix <float, OUTPUTNUMBER, 1>(Matrix <float, OUTPUTNUMBER, 1>)> func){
        postactivation_process = &func;
    }
    
    inline void setDebugFunction(std::function<void(NeuralNetwork<NBLAYERS>*, NeuralNetwork<NBLAYERS>*)> func){
        debug_printer = &func;
    }
    
    inline std::array<NeuralNetwork<NBLAYERS>*, POP_SIZE> getNetworksPopulation() {
        return networksPopulation;
    }
    
    inline void generateNewGeneration(const std::array< std::pair<float, NeuralNetwork<NBLAYERS>*>, POP_SIZE>& MSE, std::size_t iteration,
                                      const std::function<void(NeuralNetwork<NBLAYERS>&, NeuralNetwork<NBLAYERS>*, NeuralNetwork<NBLAYERS>*)> merging_instructions) {
        std::array<NeuralNetwork<NBLAYERS>*, POP_SIZE> newNetworksPopulation;
        
        float mean_MSE(0);
        for (std::size_t i(0); i < std::round(0.5*POP_SIZE/6); i++){
            newNetworksPopulation[i] = new NeuralNetwork<NBLAYERS> (std::string("RANDOM_") + std::to_string(iteration));
            NNInstructions(*newNetworksPopulation[i]);
            mean_MSE += MSE[i].first;
        }
        for (std::size_t i(std::round(0.5*POP_SIZE/6)); i < std::round(7*POP_SIZE/8); i++){
            newNetworksPopulation[i] = new NeuralNetwork<NBLAYERS> (std::string("CHILD_") + std::to_string(iteration));
            NNInstructions(*newNetworksPopulation[i]);
            merging_instructions(*newNetworksPopulation[i],
                                 MSE[POP_SIZE-i-1].second,
                                 MSE[POP_SIZE-i].second);
            mean_MSE += MSE[i].first;
        }
        for (std::size_t i(std::round(7*POP_SIZE/8)); i < POP_SIZE; i++){
            newNetworksPopulation[i] = new NeuralNetwork<NBLAYERS> (MSE[i].second->getCustomOriginField());
            NNInstructions(*newNetworksPopulation[i]);
            merging_instructions(*newNetworksPopulation[i], MSE[i].second, MSE[i].second);
            mean_MSE += MSE[i].first;
                
            if (debug_printer != nullptr)
                debug_printer->operator()(MSE[i].second, newNetworksPopulation[i]);
        }
        mean_MSE = mean_MSE/POP_SIZE;
        std::cout << "Best MSE = " << MSE[POP_SIZE-1].first << " - Mean=" << mean_MSE << " - Origin: " << MSE[POP_SIZE-1].second->getCustomOriginField() << std::endl;
        for (std::size_t i(0); i < POP_SIZE; i++){
            delete networksPopulation[i];
            networksPopulation[i] = newNetworksPopulation[i];
        }
    }
};

template <size_t NBLAYERS, size_t INPUTNUMBER, size_t OUTPUTNUMBER, size_t POP_SIZE=20>
class ManualGeneticAlgorithm : public BaseGeneticAlgorithm<NBLAYERS, INPUTNUMBER, POP_SIZE>{
    
};

template <size_t NBLAYERS, size_t INPUTNUMBER, size_t OUTPUTNUMBER, size_t TRAINING_LENGTH, size_t POP_SIZE=20>
class GeneticAlgorithm : public BaseGeneticAlgorithm<NBLAYERS, INPUTNUMBER, OUTPUTNUMBER, POP_SIZE> {
protected:
    std::array<Matrix <float, INPUTNUMBER, 1>, TRAINING_LENGTH> inputs;
    std::array<Matrix <float, OUTPUTNUMBER, 1>, TRAINING_LENGTH> outputs;
    
    
public:
    static std::function<float(std::array<Matrix <float, OUTPUTNUMBER, 1>, TRAINING_LENGTH>, std::array< Matrix <float, OUTPUTNUMBER, 1> , TRAINING_LENGTH>)> MSE;
    static std::function<float(std::array<Matrix <float, OUTPUTNUMBER, 1>, TRAINING_LENGTH>, std::array< Matrix <float, OUTPUTNUMBER, 1> , TRAINING_LENGTH>)> ERRORS_COUNT;
    
    
    inline GeneticAlgorithm(const std::array<Matrix <float, INPUTNUMBER, 1>, TRAINING_LENGTH> in, const std::array<Matrix <float, OUTPUTNUMBER, 1>, TRAINING_LENGTH> out, const std::function<void(NeuralNetwork<NBLAYERS>&)> instructions) : BaseGeneticAlgorithm<NBLAYERS, INPUTNUMBER, OUTPUTNUMBER, POP_SIZE>(instructions), inputs(in), outputs(out) {
    };
    
    
    /*
     Choices about generation of new population :
     Flop 1/12 is replaced by new networks generated randomly (introduction of immigration)
     Top (7/8-1/12) is selected and multiplied so that their offspring represents less than 2/3 of new generation
     The remainder is generated using former top players
     */
    
    inline NeuralNetwork<NBLAYERS>* trainNetworks(const int iterations,
        const std::function<void(NeuralNetwork<NBLAYERS>&, NeuralNetwork<NBLAYERS>*, NeuralNetwork<NBLAYERS>*)> merging_instructions,
        const std::function<float(std::array<Matrix <float, OUTPUTNUMBER, 1>, TRAINING_LENGTH>, std::array< Matrix <float, OUTPUTNUMBER, 1> , TRAINING_LENGTH>)> evalfunc = GeneticAlgorithm<NBLAYERS, INPUTNUMBER, OUTPUTNUMBER, TRAINING_LENGTH, POP_SIZE>::MSE) {
        for (std::size_t iteration(1); iteration <= iterations; iteration++){
            std::array< std::array< Matrix <float, OUTPUTNUMBER, 1> , TRAINING_LENGTH>, POP_SIZE> localoutputs;
            std::array< std::pair<float, NeuralNetwork<NBLAYERS>*>, POP_SIZE> MSE;
            for (std::size_t netid(0); netid < POP_SIZE; netid++){
                float MSE_i = 0;
                for (std::size_t inputid(0); inputid < TRAINING_LENGTH; inputid++){
                    localoutputs[netid][inputid] = *(this->networksPopulation[netid]->template process<INPUTNUMBER, OUTPUTNUMBER>(inputs[inputid]));
                    if (this->postactivation_process != nullptr)
                        localoutputs[netid][inputid] = this->postactivation_process->operator()(localoutputs[netid][inputid]);
                }
                MSE_i = evalfunc(outputs, localoutputs[netid]);
                MSE[netid] = {MSE_i, this->networksPopulation[netid]};
            }
            std::sort(MSE.begin(), MSE.end(), [](const std::pair<float, NeuralNetwork<NBLAYERS>*> &x, const std::pair<float, NeuralNetwork<NBLAYERS>*> &y){
                 return x.first > y.first;
             });
            
            this->generateNewGeneration(MSE, iteration, merging_instructions);
            
            if (MSE[POP_SIZE-1].first == 0)
                break;
        }
        
        return this->networksPopulation[POP_SIZE-1];
        
    }
    
};


#endif /* GeneticAlgorithm_hpp */
