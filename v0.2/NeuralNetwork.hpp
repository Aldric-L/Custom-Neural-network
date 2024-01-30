//
//  NeuralNetwork.hpp
//  AKML Project
//
//  Created by Aldric Labarthe on 07/09/2023.
//

#ifndef NeuralNetwork_hpp
#define NeuralNetwork_hpp

#include <fstream>

#include "AKML_consts.hpp"
#include "UtilityLoops.hpp"
#include "Matrices.hpp"
#include "NeuralLayer.hpp"

namespace akml {

class NeuralNetwork {
public:
    typedef std::vector<std::pair<std::size_t, const akml::ActivationFunction<float>*>> initialize_list_type;
    
protected:
    const std::string customOriginField;
    std::vector<NeuralLayer*> layers;
    std::size_t layers_nb, input_dim, output_dim;
    
public:
    
    inline NeuralNetwork(const std::size_t nb_layers, const std::string customOriginField="") : customOriginField(customOriginField), layers_nb(nb_layers), input_dim(0), output_dim(0) {
        layers.reserve(nb_layers);
    };
    
    inline NeuralNetwork(const initialize_list_type& layers_list, const std::string customOriginField="") : customOriginField(customOriginField), layers_nb(layers_list.size()), input_dim(layers_list[0].first), output_dim(layers_list[layers_list.size()-1].first) {
        layers.reserve(layers_nb);
        for (std::size_t i(0); i < layers_list.size(); i++){
            layers.push_back(new NeuralLayer(i, layers_list[i].first, (i!=0) ? layers.at(i-1) : nullptr));
            layers[i]->setActivationFunction(layers_list[i].second);
            if (i != 0){
                layers[i]->setBiases(akml::transform(layers[i]->getBiases(), layers[i]->getBiases().RANDOM_TRANSFORM));
                layers[i]->setWeights(akml::transform(layers[i]->getWeights(), layers[i]->getWeights().RANDOM_TRANSFORM));
            }
        }
    };
    
    inline NeuralNetwork(const NeuralNetwork& othernet) :
        customOriginField(othernet.getCustomOriginField()),
        layers(std::move(othernet.getLayers())),
        layers_nb(othernet.getLayerNb()),
        input_dim(othernet.getLayer(0)->getNeuronNumber()),
        output_dim(othernet.getLayer(othernet.getLayerNb()-1)->getNeuronNumber()) {
    };
    
    inline void construct(const initialize_list_type layers_list) {
        input_dim = layers_list[0].first;
        output_dim = layers_list[layers_list.size()-1].first;
        layers_nb = layers_list.size();
        layers.reserve(layers_nb);
        for (std::size_t i(0); i < layers_list.size(); i++){
            layers.push_back(new NeuralLayer(i, layers_list[i].first, (i!=0) ? layers.at(i-1) : nullptr));
            layers[i]->setActivationFunction(layers_list[i].second);
            if (i != 0){
                layers[i]->setBiases(akml::transform(layers[i]->getBiases(), layers[i]->getBiases().RANDOM_TRANSFORM));
                layers[i]->setWeights(akml::transform(layers[i]->getWeights(), layers[i]->getWeights().RANDOM_TRANSFORM));
            }
        }
        
    }
    
    inline ~NeuralNetwork() {
        for (std::size_t i(0); i < layers_nb; i++){
            if (layers[i] != nullptr)
                delete layers[i];
        }
    };
    
    inline std::string getCustomOriginField() const { return customOriginField; }
    inline NeuralLayer* getLayer(const std::size_t layer) const { return layers[layer-1]; };
    inline std::vector<NeuralLayer*> getLayers() const { return layers; };
    inline std::size_t getLayerNb() const { return layers_nb; };
    
    inline akml::DynamicMatrix<float> process(const akml::DynamicMatrix<float> &input){
        if (layers_nb == 0)
            throw std::invalid_argument("Fatal error: processing a non-initialized network");
        
        layers[0]->setInput(input);
        /*for (std::size_t i(1); i < layers_nb; i++){
            if (layers[i] == nullptr)
                throw std::exception();
            layers[i]->setPreviousActivationLayer(layers[i-1]->getActivationLayer());
        }*/
        
        return layers.back()->getActivationLayer();
    }
    
};

}

#endif /* NeuralNetwork_hpp */
