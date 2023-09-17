//
//  main.cpp
//  XORNeural network
//
//  Created by Aldric Labarthe on 07/09/2023.
//

//                                                                 AA     LL
//                 pzr--}fC/][dn1wYnfJ                            AAAA    LL
//              c{?_x[v){Ur}][ft(vX(|wzrmOcQ                     AA |AA   LL
//            #][_j(L}u-[[{x[{)q|)fztfXxUQuJkUm                 AAAAAAA   LL
//        .])-1-U)t}-v})}}]L/(}hZjjmtXdYCZuCkJCnLC|Xx          |AA   AAA  LLLLLLLL
//      I{(}11_-+/-q{p}f[1Oj{){QnfL|J00kC*#aqh0zQnjvwz/fCj^
//    "tf1f(fxtujXzt(jcf{{vX/t|bOxJ8hmJ0zwLwJCqzCxYLrYXXYcxx?
//   >()vxcJpUaXMUfdtcuj{]XrtrkkQC0#WhwkhokwCCzChbdcQB@BuO|vxn[
//  "f(|xcOhW&OB@B*BMvJ//(cCzXmqzhB%Mo*qrbUZLbma0&paB@B&X*Zw)xruLx/"
//  rv/    a%Mw88(  ./jujn0nvu   mh%%B%B%ad*Qk0Y0#B@@@@@@*puQqdt/ntCujx/)
// .xu  O  oh88c  .`.1ptL|OXY   @@@@c   #mJhd_        BBqhoLYkd#ju/j|rtnfnrjxjx~'
// `YuO  doBa`    nOCOXmQhmw    @@@B"    #ozbJk,         {?{jb*QbO#W*8*#mha0LxfIl`
//  cCZwdk&,      jzzQcak*u.    %%%%B    WoddJmO_            ddddbM*p/U0dbM*p/Ueudk
//   hMW          J[fY:          wBB8    M*p/U0{                 wLwJCqzCxYwLwJCqzfdfd
//                !qv              !qv   `}czjf:                    wLwJCqzCxYwLwJCddddfd
//

#include <iostream>
#include <functional>
#include <cmath>

#include "Matrix.cpp"
#include "NeuralNetwork.cpp"
#include "GeneticAlgorithm.cpp"

 
int main(int argc, const char * argv[]) {
    std::cout << "Hello, terrible World!\n";
    
    /*Matrix<int, 1, 2> A;
    A(1,1) = 3;
    A(1,2) = 5;
    A(2,1) = 4;
    A(2,2) = 6;
    Matrix<int, 1, 2>::cout(A);
    
    Matrix<int, 2, 2> B;
    B(1,1) = 1;
    B(1,2) = 2;
    B(2,1) = 3;
    B(2,2) = 4;
    Matrix<int, 2, 2>::cout(B);
    
    //Matrix<int, 1, 2> C = Matrix<int, 1, 2>::product<1,2,2,2> (A, B);
    //Matrix<int, 1, 2>::cout(C);
    
    B = A;
    std::cout << B;
    
    //Matrix<float, 2, 2> D;
    //std::array <std::array <int, 2>, 2> test;
    //test = {{ {1,2},{3, 4} }};
    //D = {{ {1,2},{3, 4} }};
    //std::cout << D;
    
    //D.transform([](float x) {return 1/(1+exp(x));});
    D.transform(Matrix<float, 2, 2>::NO_ACTION_TRANSFORM);
    std::cout << D;
    //bool test = D==Matrix<float, 2, 2>::EMPTY;
    //std::cout << test;
    
    //std::cout << Matrix<float, 2, 2>::EMPTY;
    
    //Matrix<int, 2, 2>::cout(D);
    //std::cout << E;
    //const std::size_t layoutNeuron[3] = {0, 1,1};
    */
     
    NeuralNetwork<4> neuralnet;
    neuralnet.setFirstLayer<3>();
    neuralnet.setLayer<6, 3>(2);
    neuralnet.setLayer<3, 6>(3);
    neuralnet.setLayer<1, 3>(4);
    /*Matrix<float, 3, 1> Input = {{1, 0, 1}};
    
    std::cout << *neuralnet.process<3, 1>(Input);
    
    std::cout << "\n\n\n New gen"<< std::endl;
    
    Matrix<float, 3, 1> Input2 = {{12, 13, 1}};
    std::cout << *neuralnet.process<3, 1>(Input2);*/
    
    
    //XOR with dim3
    /*std::array<Matrix <float, 3, 1>, 8> inputs =
    {{ {{ {1,0,0} }}, {{ {0,0,1} }}, {{ {0,1,0} }},
       {{ {1,1,0} }}, {{ {1,0,1} }}, {{ {1,1,0} }},
       {{ {0,0,0} }}, {{ {1,1,1} }} }};
    
    std::array<Matrix <float, 1, 1>, 8> outputs =
    {{ {{ {1} }}, {{ {1} }}, {{ {1} }},
       {{ {0} }}, {{ {0} }}, {{ {0} }},
       {{ {0} }}, {{ {0} }} }};
    
    std::function<void(NeuralNetwork<4>&)> init_instructions = [](NeuralNetwork<4>& net) {
        net.setFirstLayer<3>();
        net.setLayer<6, 3>(2);
        net.setLayer<3, 6>(3);
        net.setLayer<1, 3>(4);
    };
    Matrix<float, 3, 1> Input = {{1, 0, 1}};
    std::cout << *neuralnet.process<3, 1>(Input);
    
    std::function<void(NeuralNetwork<4>&, NeuralNetwork<4>*, NeuralNetwork<4>*)> merging_instructions =
    [](NeuralNetwork<4>& child, NeuralNetwork<4>* parent1, NeuralNetwork<4>* parent2) {
        BaseGeneticAlgorithm::mergeLayers<NeuralNetwork<4>, NeuralLayer<6, 3>>(2, parent1, parent2, child);
        BaseGeneticAlgorithm::mergeLayers<NeuralNetwork<4>, NeuralLayer<3, 6>>(3, parent1, parent2, child);
        BaseGeneticAlgorithm::mergeLayers<NeuralNetwork<4>, NeuralLayer<1, 3>>(4, parent1, parent2, child);
    };
    
    GeneticAlgorithm<4, 3, 1, 8, 100> ga (inputs, outputs, init_instructions);
    ga.setPostActivationProcess(GeneticAlgorithm<4, 3, 1, 8, 100>::ACTIVATE_ROUND);
    NeuralNetwork<4>* bestnet = ga.trainNetworks(500, merging_instructions,
                                                 GeneticAlgorithm<4, 3, 1, 8, 100>::ERRORS_COUNT);
    std::cout << GeneticAlgorithm<4, 3, 1, 8, 100>::ACTIVATE_ROUND(*bestnet->process<3, 1>(Input));
    */
    
    std::array<Matrix <float, 2, 1>, 4> inputs = {{ {{ {1,0} }}, {{ {0,0} }}, {{ {0,1} }}, {{ {1,1} }} }};
    
    std::array<Matrix <float, 1, 1>, 4> outputs = {{ {{ {1} }}, {{ {0} }}, {{ {0} }}, {{ {1} }} }};
    
    std::function<void(NeuralNetwork<3>&)> init_instructions = [](NeuralNetwork<3>& net) {
        net.setFirstLayer<2>();
        net.setLayer<2, 2>(2);
        net.setLayer<1, 2>(3);
    };
    Matrix<float, 2, 1> Input = {{1, 0}};
    //std::cout << *neuralnet.process<3, 1>(Input);
    
    std::function<void(NeuralNetwork<3>&, NeuralNetwork<3>*, NeuralNetwork<3>*)> merging_instructions =
    [](NeuralNetwork<3>& child, NeuralNetwork<3>* parent1, NeuralNetwork<3>* parent2) {
        if (parent1 == parent2)
            BaseGeneticAlgorithm::mergeLayers<NeuralNetwork<3>, NeuralLayer<2, 1>>(1, parent1, parent2, child);

        BaseGeneticAlgorithm::mergeLayers<NeuralNetwork<3>, NeuralLayer<2, 2>>(2, parent1, parent2, child);
        BaseGeneticAlgorithm::mergeLayers<NeuralNetwork<3>, NeuralLayer<1, 2>>(3, parent1, parent2, child);
    };
    
    std::function<void(NeuralNetwork<3>*, NeuralNetwork<3>*)> debugfunc = [](NeuralNetwork<3>* net1, NeuralNetwork<3>* net2) {
        std::cout << "Layer 1 : " << std::endl;
        NeuralLayer<2, 1>* net1_layer1 = (NeuralLayer<2, 1>*)net1->getLayer(1);
        NeuralLayer<2, 1>* net2_layer1 = nullptr;
        if (net2 != nullptr)
            NeuralLayer<2, 1>* net2_layer1 = (NeuralLayer<2, 1>*)net2->getLayer(1);

        std::cout << "Biases : " << std::endl;
        std::cout << *(net1_layer1->getBiasesAccess());
        if (net2 != nullptr)
            std::cout << *(net2_layer1->getBiasesAccess());
        
        std::cout << "Weights : " << std::endl;
        std::cout << *(net1_layer1->getWeightsAccess());
        if (net2 != nullptr)
            std::cout << *(net2_layer1->getWeightsAccess());
        
        std::cout << "Previous Activationfunc : " << std::endl;
        std::cout << *(net1_layer1->getPreviousActivationLayer());
        if (net2 != nullptr)
            std::cout << *(net2_layer1->getPreviousActivationLayer());
        
        std::cout << "Activationfunc : " << std::endl;
        std::cout << *((Matrix<float, 2, 1>*)net1_layer1->getActivationLayer());
        if (net2 != nullptr)
            std::cout << *((Matrix<float, 2, 1>*)net2_layer1->getActivationLayer());
        
        std::cout << "\nLayer 2 : " << std::endl;
        NeuralLayer<2, 2>* net1_layer2 = (NeuralLayer<2, 2>*)net1->getLayer(2);
        NeuralLayer<2, 2>* net2_layer2 = nullptr;
        if (net2 != nullptr)
            NeuralLayer<2, 2>* net2_layer2 = (NeuralLayer<2, 2>*)net2->getLayer(2);
        
        std::cout << "Biases : " << std::endl;
        std::cout << *(net1_layer2->getBiasesAccess());
        if (net2 != nullptr)
            std::cout << *(net2_layer2->getBiasesAccess());
        
        std::cout << "Weights : " << std::endl;
        std::cout << *(net1_layer2->getWeightsAccess());
        if (net2 != nullptr)
            std::cout << *(net2_layer2->getWeightsAccess());
        
        std::cout << "Previous Activationfunc : " << std::endl;
        std::cout << *(net1_layer2->getPreviousActivationLayer());
        if (net2 != nullptr)
            std::cout << *(net2_layer2->getPreviousActivationLayer());
        
        std::cout << "Activationfunc : " << std::endl;
        std::cout << *((Matrix<float, 2, 2>*)net1_layer2->getActivationLayer());
        if (net2 != nullptr)
            std::cout << *((Matrix<float, 2, 2>*)net2_layer2->getActivationLayer());
        
        std::cout << "\nLayer 3 : " << std::endl;
        NeuralLayer<1, 2>* net1_layer3 = (NeuralLayer<1, 2>*)net1->getLayer(3);
        NeuralLayer<2, 1>* net2_layer3 = nullptr;
        if (net2 != nullptr)
            NeuralLayer<1, 2>* net2_layer3 = (NeuralLayer<1, 2>*)net2->getLayer(3);
        
        std::cout << "Biases : " << std::endl;
        std::cout << *(net1_layer3->getBiasesAccess());
        if (net2 != nullptr)
            std::cout << *(net2_layer3->getBiasesAccess());
        
        std::cout << "Weights : " << std::endl;
        std::cout << *(net1_layer3->getWeightsAccess());
        if (net2 != nullptr)
            std::cout << *(net2_layer3->getWeightsAccess());
        
        std::cout << "Previous Activation : " << std::endl;
        std::cout << *(net1_layer3->getPreviousActivationLayer());
        if (net2 != nullptr)
            std::cout << *(net2_layer3->getPreviousActivationLayer());
        
        std::cout << "Activation : " << std::endl;
        std::cout << *((Matrix<float, 1, 2>*)net1_layer3->getActivationLayer());
        if (net2 != nullptr)
            std::cout << *((Matrix<float, 1, 2>*)net2_layer3->getActivationLayer());
    };
    
    GeneticAlgorithm<3, 2, 1, 4, 100> ga (inputs, outputs, init_instructions);
    //ga.setDebugFunction(debugfunc);
    
    //ga.setPostActivationProcess(GeneticAlgorithm<3, 2, 1, 4, 100>::ACTIVATE_ROUND);
    /*NeuralNetwork<3>* bestnet = ga.trainNetworks(500, merging_instructions,
                                                 GeneticAlgorithm<3, 2, 1, 4, 100>::ERRORS_COUNT);*/
    NeuralNetwork<3>* bestnet = ga.trainNetworks(500, merging_instructions,
                                                 GeneticAlgorithm<3, 2, 1, 4, 100>::MSE);
    
    
    for (std::size_t inputid(0); inputid<4; inputid++){
        std::cout << "Testing with " << std::endl;
        std::cout << inputs[inputid];
        std::cout << "Output :" << std::endl;
        std::cout << *bestnet->process<2, 1>(inputs[inputid]);
        std::cout << "Output expected :" << std::endl;
        std::cout << outputs[inputid];
    }
    
    //std::cout << "Debugging NN" << std::endl;
    //debugfunc(bestnet, nullptr);
    
    //std::cout << GeneticAlgorithm<3, 2, 1, 4, 100>::ACTIVATE_ROUND(*bestnet->process<2, 1>(Input));
    
    //char hello;
    //std::cin >> hello;
    return 0;
}
