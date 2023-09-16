//
//  NeuralLayer.hpp
//  XORNeural network
//
//  Created by Aldric Labarthe on 09/09/2023.
//

#ifndef NeuralLayer_hpp
#define NeuralLayer_hpp

#include <stdio.h>
#include <string>
#include <exception>
#include "Matrix.hpp"

class AbstractNeuralLayer {
protected:
    std::function<float(float)> activationFunction;
    bool isFirstRow = false;
    
public:
    const std::size_t neuronNumber, previousNeuronNumber, layerId;
    
    AbstractNeuralLayer(const std::size_t neuronNumber, const std::size_t previousNeuronNumber, const std::size_t layerId) : neuronNumber(neuronNumber), previousNeuronNumber(previousNeuronNumber), layerId(layerId) {}
    
    inline std::size_t getNeuronNumber(){ return neuronNumber; }
    inline std::size_t getPreviousNeuronNumber(){ return previousNeuronNumber; }
    
    inline void setActivationFunction(std::function<float(float)> actfunc){ activationFunction = actfunc; }
    
    inline void setFirstRow(const bool first_row){ isFirstRow = first_row; }
    
    inline virtual void setInput(MatrixPrototype<float>* arg) { return; };
    inline virtual void setBiases(MatrixPrototype<float>* new_biases) { return; };
    inline virtual void setWeights(MatrixPrototype<float>* new_weights) { return; };
    inline virtual void setPreviousActivationLayer(MatrixPrototype<float>* argument) { return; };
    inline virtual MatrixPrototype<float>* getActivationLayer() { MatrixPrototype<float>* rtrn = new MatrixPrototype<float>(1, 1); return rtrn; };
};


template <std::size_t NEURON_NUMBER, std::size_t PREVIOUS_NEURON_NUMBER>
class NeuralLayer : public AbstractNeuralLayer {
private:
    Matrix<float, NEURON_NUMBER, 1> ownActivationLayer;
    Matrix<float, PREVIOUS_NEURON_NUMBER, 1>* previousActivationLayer;
    Matrix<float, NEURON_NUMBER, PREVIOUS_NEURON_NUMBER> weights;
    Matrix<float, NEURON_NUMBER, 1> biases;
    
public:
    
    NeuralLayer(const std::size_t layerId) : AbstractNeuralLayer(NEURON_NUMBER, PREVIOUS_NEURON_NUMBER, layerId) {
        weights.transform([](float x, std::size_t row, std::size_t column) {return 1;});
        previousActivationLayer = nullptr;
    }
    
    inline ~NeuralLayer() {
        if (previousActivationLayer != nullptr)
            delete previousActivationLayer;
    };
    
    inline void setInput(MatrixPrototype<float>* arg) {
        if (!isFirstRow)
            throw std::exception();
        
        
        Matrix<float, NEURON_NUMBER, 1>* layer(0);
        layer = (Matrix<float, NEURON_NUMBER, 1>*)arg;
        ownActivationLayer = *layer;
        previousActivationLayer = new Matrix<float, PREVIOUS_NEURON_NUMBER, 1>;
        previousActivationLayer->operator()(1,1) = 1;
    }
    
    inline void setBiases(MatrixPrototype<float>* arg) {
        Matrix<float, NEURON_NUMBER, 1>* new_biases(0);
        new_biases = (Matrix<float, NEURON_NUMBER, 1>*)arg;
        biases = *new_biases;
    }
    
    inline Matrix<float, NEURON_NUMBER, 1>* getBiasesAccess(){
        Matrix<float, NEURON_NUMBER, 1>* b_point (&biases);
        return b_point;    }
    
    inline void setWeights(MatrixPrototype<float>* arg) {
        Matrix<float, NEURON_NUMBER, PREVIOUS_NEURON_NUMBER>* new_weights(0);
        new_weights = (Matrix<float, NEURON_NUMBER, PREVIOUS_NEURON_NUMBER>*)arg;
        weights = *new_weights;
    }
    
    inline Matrix<float, NEURON_NUMBER, PREVIOUS_NEURON_NUMBER>* getWeightsAccess (){
        Matrix<float, NEURON_NUMBER, PREVIOUS_NEURON_NUMBER>* w_point (&weights);
        return w_point;
    }
    
    inline void setPreviousActivationLayer(MatrixPrototype<float>* arg) {
        //std::cout << "Prev activ layer received : " << std::endl;
        Matrix<float, PREVIOUS_NEURON_NUMBER, 1>* prev(0);
        prev = (Matrix<float, PREVIOUS_NEURON_NUMBER, 1>*)arg;
        previousActivationLayer = prev;
        //std::cout << *previousActivationLayer;
    }
    
    inline MatrixPrototype<float>* getActivationLayer(){
        //std::cout << "-------- Process layer " << layerId << " - NNumber=" << NEURON_NUMBER <<  std::endl;
        if (!isFirstRow){
            /*std::cout << "Use Previous layer : " << std::endl;
            std::cout << *previousActivationLayer;
            std::cout << "Use weights : " << std::endl;
            std::cout << weights;
            std::cout << "Use biases : " << std::endl;
            std::cout << biases;*/
            ownActivationLayer = Matrix<float, NEURON_NUMBER, 1>::product(weights, *previousActivationLayer);
            /*std::cout << "Product : " << std::endl;
            std::cout << ownActivationLayer;*/
            ownActivationLayer += biases;
            /*std::cout << "Plus biases : " << std::endl;
            std::cout << ownActivationLayer;*/
            ownActivationLayer.transform(activationFunction);
            /*std::cout << "Final : " << std::endl;
            std::cout << ownActivationLayer;*/
        }else {
            //std::cout << "First row, use :" << std::endl;
            //std::cout << ownActivationLayer;
        }
        return &ownActivationLayer;
        //if (ownActivationLayer == Matrix<float, NEURON_NUMBER, 1>::EMPTY){
            
        //}
        //std::cout << "Preexisting matrix ?" << std::endl;
        //std::cout << ownActivationLayer;
        return &ownActivationLayer;
    }
    
};

#endif /* NeuralLayer_hpp */
