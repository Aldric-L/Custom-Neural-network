
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
     
    akml::NeuralNetwork<4> neuralnet;
    neuralnet.setFirstLayer<3>();
    neuralnet.setLayer<6, 3>(2);
    neuralnet.setLayer<3, 6>(3);
    neuralnet.setLayer<1, 3>(4);
    
    //XOR with dim3
    std::array<akml::Matrix <float, 3, 1>, 8> inputs =
    {{ {{ {1,0,0} }}, {{ {0,0,1} }}, {{ {0,1,0} }},
       {{ {1,1,0} }}, {{ {1,0,1} }}, {{ {1,1,0} }},
       {{ {0,0,0} }}, {{ {1,1,1} }} }};
    
    std::array<akml::Matrix <float, 1, 1>, 8> outputs =
    {{ {{ {1} }}, {{ {1} }}, {{ {1} }},
       {{ {0} }}, {{ {0} }}, {{ {0} }},
       {{ {0} }}, {{ {0} }} }};
    
    std::function<void(akml::NeuralNetwork<4>&)> init_instructions = [](akml::NeuralNetwork<4>& net) {
        net.setFirstLayer<3>();
        net.setLayer<6, 3>(2);
        net.setLayer<3, 6>(3);
        net.setLayer<1, 3>(4);
    };
    
    std::function<void(akml::NeuralNetwork<4>&, akml::NeuralNetwork<4>*, akml::NeuralNetwork<4>*)> merging_instructions =
    [](akml::NeuralNetwork<4>& child, akml::NeuralNetwork<4>* parent1, akml::NeuralNetwork<4>* parent2) {
        if (parent1 == parent2)
            akml::GeneticAlgorithmMethods::mergeLayers<akml::NeuralNetwork<4>, akml::NeuralLayer<3, 1>>(1, parent1, parent2, child);
        akml::GeneticAlgorithmMethods::mergeLayers<akml::NeuralNetwork<4>, akml::NeuralLayer<6, 3>>(2, parent1, parent2, child);
        akml::GeneticAlgorithmMethods::mergeLayers<akml::NeuralNetwork<4>, akml::NeuralLayer<3, 6>>(3, parent1, parent2, child);
        akml::GeneticAlgorithmMethods::mergeLayers<akml::NeuralNetwork<4>, akml::NeuralLayer<1, 3>>(4, parent1, parent2, child);
    };
    
    akml::GeneticAlgorithm<4, 3, 1, 8> ga (inputs, outputs, init_instructions);
    akml::NeuralNetwork<4>* bestnet = ga.trainNetworks(5000, merging_instructions,
                                                 akml::GeneticAlgorithm<4, 3, 1, 8>::MSE);
        
    for (std::size_t inputid(0); inputid<8; inputid++){
        std::cout << "Testing with " << std::endl;
        std::cout << inputs[inputid];
        std::cout << "Output :" << std::endl;
        std::cout << *bestnet->process<3, 1>(inputs[inputid]);
        std::cout << "Output expected :" << std::endl;
        std::cout << outputs[inputid];
    }
    
    // XOR DIM2
    /*
    std::array<akml::Matrix <float, 2, 1>, 4> inputs = {{ {{ {1,0} }}, {{ {0,0} }}, {{ {0,1} }}, {{ {1,1} }} }};
    
    std::array<akml::Matrix <float, 1, 1>, 4> outputs = {{ {{ {1} }}, {{ {0} }}, {{ {0} }}, {{ {1} }} }};
    
    std::function<void(akml::NeuralNetwork<3>&)> init_instructions = [](akml::NeuralNetwork<3>& net) {
        net.setFirstLayer<2>();
        net.setLayer<2, 2>(2);
        net.setLayer<1, 2>(3);
    };
    
    std::function<void(akml::NeuralNetwork<3>&, akml::NeuralNetwork<3>*, akml::NeuralNetwork<3>*)> merging_instructions =
    [](akml::NeuralNetwork<3>& child, akml::NeuralNetwork<3>* parent1, akml::NeuralNetwork<3>* parent2) {
        if (parent1 == parent2)
            akml::GeneticAlgorithmMethods::mergeLayers<akml::NeuralNetwork<3>, akml::NeuralLayer<2, 1>>(1, parent1, parent2, child);

        akml::GeneticAlgorithmMethods::mergeLayers<akml::NeuralNetwork<3>, akml::NeuralLayer<2, 2>>(2, parent1, parent2, child);
        akml::GeneticAlgorithmMethods::mergeLayers<akml::NeuralNetwork<3>, akml::NeuralLayer<1, 2>>(3, parent1, parent2, child);
    };
    
    
    akml::GeneticAlgorithm<3, 2, 1, 4, 100> ga (inputs, outputs, init_instructions);
    akml::NeuralNetwork<3>* bestnet = ga.trainNetworks(500, merging_instructions,
                                                       akml::GeneticAlgorithm<3, 2, 1, 4, 100>::MSE);
    
    for (std::size_t inputid(0); inputid<4; inputid++){
        std::cout << "Testing with " << std::endl;
        std::cout << inputs[inputid];
        std::cout << "Output :" << std::endl;
        std::cout << *bestnet->process<2, 1>(inputs[inputid]);
        std::cout << "Output expected :" << std::endl;
        std::cout << outputs[inputid];
    }*/
    
    return 0;
}
