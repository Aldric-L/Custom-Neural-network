include_directories(${PROJECT_SOURCE_DIR}/AKML-lib/)
add_library(AKML AKML.hpp AKML_consts.hpp UtilityLoops.hpp GeneticAlgorithm.cpp GeneticAlgorithm.hpp Matrix.hpp NeuralLayer.hpp NeuralNetwork.hpp )
