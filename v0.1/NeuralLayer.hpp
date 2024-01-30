//
//  NeuralLayer.hpp
//  AKML Project
//
//  Created by Aldric Labarthe on 09/09/2023.
//

#ifndef NeuralLayer_hpp
#define NeuralLayer_hpp

#include "Matrices.hpp"
#include "NeuralFunctions.cpp"

namespace akml {

class AbstractNeuralLayer {
protected:
    const akml::ActivationFunction<float>* activationFunction;
    bool isFirstRow = false;
    
public:
    const std::size_t neuronNumber, previousNeuronNumber, layerId;
    
    AbstractNeuralLayer(const std::size_t neuronNumber, const std::size_t previousNeuronNumber, const std::size_t layerId) : neuronNumber(neuronNumber), previousNeuronNumber(previousNeuronNumber), layerId(layerId), activationFunction(nullptr) {}
    
    inline virtual ~AbstractNeuralLayer() {};
    
    inline std::size_t getNeuronNumber(){ return neuronNumber; }
    inline std::size_t getPreviousNeuronNumber(){ return previousNeuronNumber; }
    
    inline void setActivationFunction(const akml::ActivationFunction<float>& actfunc){ activationFunction = &actfunc; }
    
    inline void setFirstRow(const bool first_row){ isFirstRow = first_row; }
    
    inline virtual void setInput(MatrixInterface<float>* arg) = 0;
    inline virtual void setBiases(const MatrixInterface<float>* new_biases) = 0;
    inline virtual void setWeights(const MatrixInterface<float>* new_weights) = 0;
    inline virtual void setPreviousActivationLayer(MatrixInterface<float>* argument) = 0;
    inline virtual MatrixInterface<float>* getActivationLayer() = 0;
};


template <std::size_t NEURON_NUMBER, std::size_t PREVIOUS_NEURON_NUMBER>
class NeuralLayer : public AbstractNeuralLayer {
private:
    StaticMatrix<float, NEURON_NUMBER, 1> ownActivationLayer;
    StaticMatrix<float, PREVIOUS_NEURON_NUMBER, 1>* previousActivationLayer;
    StaticMatrix<float, NEURON_NUMBER, PREVIOUS_NEURON_NUMBER> weights;
    StaticMatrix<float, NEURON_NUMBER, 1> biases;
    
public:
    
    NeuralLayer(const std::size_t layerId) : AbstractNeuralLayer(NEURON_NUMBER, PREVIOUS_NEURON_NUMBER, layerId), previousActivationLayer(nullptr) {
        weights.transform([](float x) {return 1;});
    }
    
    inline ~NeuralLayer() {
        // No need to delete anything as every previousActivationLayer points toward the first activationLayer that will be destroyed automatically by the first layer
        /*if (previousActivationLayer != nullptr)
            delete previousActivationLayer;*/
    };
    
    inline void setInput(MatrixInterface<float>* arg) {
        if (!isFirstRow)
            throw std::exception();
        
        ownActivationLayer = *((StaticMatrix<float, NEURON_NUMBER, 1>*)arg);
        /*if (previousActivationLayer != nullptr)
            delete previousActivationLayer;*/
        //previousActivationLayer = nullptr;
    }
    
    inline void setBiases(const MatrixInterface<float>* arg) {
        biases = *((StaticMatrix<float, NEURON_NUMBER, 1>*)arg);
    }
    
    inline StaticMatrix<float, NEURON_NUMBER, 1>* getBiasesAccess(){ return &biases; }
    
    inline void setWeights(const MatrixInterface<float>* arg) {
        weights = *((StaticMatrix<float, NEURON_NUMBER, PREVIOUS_NEURON_NUMBER>*)arg);
    }
    
    inline StaticMatrix<float, NEURON_NUMBER, PREVIOUS_NEURON_NUMBER>* getWeightsAccess (){ return &weights; }
    
    inline StaticMatrix<float, PREVIOUS_NEURON_NUMBER, 1>* getPreviousActivationLayer (){ return previousActivationLayer; }
        
    inline void setPreviousActivationLayer(MatrixInterface<float>* arg) {
        StaticMatrix<float, PREVIOUS_NEURON_NUMBER, 1>* prev(0);
        prev = (StaticMatrix<float, PREVIOUS_NEURON_NUMBER, 1>*)arg;
        previousActivationLayer = prev;
    }
    
    inline MatrixInterface<float>* getActivationLayer(){
        if (!isFirstRow){
            ownActivationLayer = StaticMatrix<float, NEURON_NUMBER, 1>::product(weights, *previousActivationLayer);
            ownActivationLayer += biases;
            if (activationFunction == nullptr)
                throw std::invalid_argument("No activation function provided");
            ownActivationLayer.transform(activationFunction->function);
        }
        return &ownActivationLayer;
    }
    
    inline void saveLayer(std::string* buffer, int id){
        *buffer += ("AKML_LAYER_BIASES " + std::to_string(id) + "\n");
        // Biases :
        for (std::size_t row(0); row < this->getNeuronNumber(); row++){
            *buffer += std::to_string(this->getBiasesAccess()->operator()(row+1, 1)) + "\n";
        }
        
        *buffer += "AKML_LAYER_END_BIASES \n";
        *buffer += "AKML_LAYER_WEIGHTS " + std::to_string(id) + "\n";
        // weights :
        for (std::size_t row(0); row < this->getNeuronNumber(); row++){
            for (std::size_t col(0); col < this->getPreviousNeuronNumber(); col++){
                *buffer += std::to_string(this->getWeightsAccess()->operator()(row+1, col+1)) + " ";
            }
            *buffer += "\n";
        }
        *buffer += "AKML_LAYER_END_WEIGHTS \n";
    }
    
};

}

#endif /* NeuralLayer_hpp */
