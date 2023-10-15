//
//  NeuralNetwork.hpp
//  XORNeural network
//
//  Created by Aldric Labarthe on 07/09/2023.
//

#ifndef NeuralNetwork_hpp
#define NeuralNetwork_hpp

#include "AKML_consts.hpp"
#include "UtilityLoops.hpp"
#include "Matrix.hpp"
#include "NeuralLayer.hpp"

namespace akml {

template <const std::size_t NBLAYERS=AKML_NEURAL_LAYER_NB>
class NeuralNetwork {
protected:
    const std::string customOriginField;
    std::array<AbstractNeuralLayer*, NBLAYERS> layers;
    
    template<int i, int j>
    struct setLayer_functor {
        static void run(akml::NeuralNetwork<NBLAYERS>& neuralnet, std::function<float(float)>& activation_func) {
            //std::cout << "i=" << i << " j=" << j << std::endl;
            neuralnet.setLayer<akml::NN_STRUCTURE[j], akml::NN_STRUCTURE[i]>(i+2, activation_func);
        }
    };
    
public:
    
    static inline std::function<float(float)> SIGMOID = [](float x) {return 1/(1+exp(-x));};
    static inline std::function<float(float)> RELU = [](float x) {return std::max(0.f, x); };

    static inline std::function<float(float)> NO_ACTION = [](float x) {return x;};
    
    inline NeuralNetwork(std::string customOriginField="") : customOriginField(customOriginField) {
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
    
    inline std::string getCustomOriginField(){ return customOriginField; }
    
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
        
        return  (Matrix<float, OUTPUTNUMBER, 1>*)layers[NBLAYERS-1]->getActivationLayer();
    }
    
    static inline std::function<void(akml::NeuralNetwork<NBLAYERS>&)> DEFAULT_INIT_INSTRUCTIONS = [](NeuralNetwork<NBLAYERS>& net) {
        if (NBLAYERS != akml::NEURAL_LAYER_NB)
            throw std::invalid_argument("You should not use a default initializer for a neural network which is not standard in layers number.");
        
        net.setFirstLayer<akml::NN_STRUCTURE[0]>();
        akml::for_<0, NBLAYERS-1>::template run<setLayer_functor,akml::NeuralNetwork<NBLAYERS>&, std::function<float(float)>&>(net, akml::NeuralNetwork<NBLAYERS>::SIGMOID);
    };
    
    static inline std::function<void(akml::NeuralNetwork<NBLAYERS>&)> DEFAULT_INITRELU_INSTRUCTIONS = [](NeuralNetwork<NBLAYERS>& net) {
        if (NBLAYERS != akml::NEURAL_LAYER_NB)
            throw std::invalid_argument("You should not use a default initializer for a neural network which is not standard in layers number.");
        
        net.setFirstLayer<akml::NN_STRUCTURE[0]>();
        if (NBLAYERS < 3)
            return akml::NeuralNetwork<NBLAYERS>::DEFAULT_INIT_INSTRUCTIONS;

        akml::for_<0, NBLAYERS-2>::template run<setLayer_functor,akml::NeuralNetwork<NBLAYERS>&, std::function<float(float)>&>(net, akml::NeuralNetwork<NBLAYERS>::RELU);
        akml::for_<NBLAYERS-2, NBLAYERS-1>::template run<setLayer_functor,akml::NeuralNetwork<NBLAYERS>&, std::function<float(float)>&>(net, akml::NeuralNetwork<NBLAYERS>::SIGMOID);
    };
};

}

#endif /* NeuralNetwork_hpp */
