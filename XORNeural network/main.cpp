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
#include "GeneticAlgorithm.hpp"

 
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
    
    std::array<Matrix <float, 3, 1>, 8> inputs =
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
    
    // Prototype 1
    // We create a new network from instructions and we change layers manually
    // Then we return the network so that GeneticAlgorithm can update its pointers
    /*std::function<NeuralNetwork<4>(const std::function<void(NeuralNetwork<4>&)>, NeuralNetwork<4>*, NeuralNetwork<4>*)> merging_instructions =
    [](const std::function<void(NeuralNetwork<4>&)> init_instructions, NeuralNetwork<4>* net1, NeuralNetwork<4>* net2) {
        NeuralNetwork<4> childNetwork;
        init_instructions(childNetwork);
        childNetwork.replaceLayer(2, GeneticAlgorithm::mergeLayers<6,3>(net1.getLayer<6,3>(2), net2.getLayer<6,3>(2)));
        return childNetwork;
    };*/
    
    // Prototype 2
    // Main difference : we provide an already built new neuralnet and mergeLayers replace directly in childNetwork
    std::function<void(NeuralNetwork<4>&, NeuralNetwork<4>*, NeuralNetwork<4>*)> merging_instructions =
    [](NeuralNetwork<4>& child, NeuralNetwork<4>* parent1, NeuralNetwork<4>* parent2) {
        BaseGeneticAlgorithm::mergeLayers<NeuralNetwork<4>, NeuralLayer<6, 3>>(2, parent1, parent2, child);
        BaseGeneticAlgorithm::mergeLayers<NeuralNetwork<4>, NeuralLayer<3, 6>>(3, parent1, parent2, child);
        BaseGeneticAlgorithm::mergeLayers<NeuralNetwork<4>, NeuralLayer<1, 3>>(4, parent1, parent2, child);
    };
    
    GeneticAlgorithm<4, 3, 1, 8, 100> ga (inputs, outputs, init_instructions);
    NeuralNetwork<4>* bestnet = ga.trainNetworks(50, merging_instructions);
    std::cout << *bestnet->process<3, 1>(Input);
    
    
    //char hello;
    //std::cin >> hello;
    return 0;
}
