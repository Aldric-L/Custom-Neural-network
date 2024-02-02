#include <iostream>
#include "AKML.hpp"
#include "NeuralFunctions.hpp"
#include "GeneticAlgorithm.hpp"

int main (){
    using SINGLETON = std::array <std::array <float, 1>, 1>;
    
    std::cout << "Hello, terrible world!" << std::endl;
    
    akml::NeuralNetwork *bestnet;
    
    // XOR DIM 2
    //akml::NeuralNetwork::initialize_list_type xor2_initList = {{ std::make_pair(2, nullptr), std::make_pair(2, &akml::ActivationFunctions::RELU), std::make_pair(1, &akml::ActivationFunctions::SIGMOID) }};
    /*akml::NeuralNetwork::initialize_list_type xor2_initList = {{ std::make_pair(2, nullptr), std::make_pair(2, &akml::ActivationFunctions::SIGMOID), std::make_pair(1, &akml::ActivationFunctions::SIGMOID) }};
    std::vector<akml::DynamicMatrix <float>> inputs = {{ akml::make_dynamic_vector<float>(1.f,0.f), akml::make_dynamic_vector<float>(0.f,0.f), akml::make_dynamic_vector<float>(0.f,1.f), akml::make_dynamic_vector<float>(1.f,1.f) }};
    
    std::vector<akml::DynamicMatrix <float>> outputs = {{ (SINGLETON){{ {1} }}, (SINGLETON){{ {0} }}, (SINGLETON){{ {1} }}, (SINGLETON){{ {0} }} }};*/
    
    /*akml::GeneticAlgorithm ga (100, xor2_initList);
    bestnet = ga.trainNetworks(5000, inputs, outputs);*/
    
    // XOR DIM 3
    akml::NeuralNetwork::initialize_list_type xor3_initList = {{ std::make_pair(3, nullptr), std::make_pair(3, &akml::ActivationFunctions::RELU), std::make_pair(3, &akml::ActivationFunctions::RELU), std::make_pair(1, &akml::ActivationFunctions::SIGMOID) }};
    
    std::vector<akml::DynamicMatrix <float>> inputs = {{ 
        akml::make_dynamic_vector<float>(1.f,0.f,0.f), akml::make_dynamic_vector<float>(0.f,0.f, 1.f),
        akml::make_dynamic_vector<float>(0.f,1.f,0.f), akml::make_dynamic_vector<float>(1.f,1.f,0.f),
        akml::make_dynamic_vector<float>(1.f,0.f,1.f), akml::make_dynamic_vector<float>(0.f,1.f,1.f),
        akml::make_dynamic_vector<float>(1.f,1.f,1.f) }};
        
    std::vector<akml::DynamicMatrix <float>> outputs = {{ (SINGLETON){{ {1} }}, (SINGLETON){{ {1} }}, 
        (SINGLETON){{ {1} }}, (SINGLETON){{ {0} }},
        (SINGLETON){{ {0} }}, (SINGLETON){{ {0} }}, (SINGLETON){{ {0} }} }};
    
    
    // Genetic algorithm
    /*akml::GeneticAlgorithm ga (100, xor3_initList);
    bestnet = ga.trainNetworks(5000, inputs, outputs);*/

    // Gradient descent
    /*bestnet = new akml::NeuralNetwork (xor3_initList);
    bestnet->stochGradientTraining(inputs, outputs, 3, 0.02, 50000);*/
    
    for (std::size_t inputid(0); inputid<inputs.size(); inputid++){
        std::cout << "\nTesting with " << std::endl;
        std::cout << inputs[inputid];
        std::cout << "Output :" << std::endl;
        std::cout << bestnet->process(inputs[inputid]);
        std::cout << "Output expected :" << std::endl;
        std::cout << outputs[inputid];
    }
    
    return 0;
}
