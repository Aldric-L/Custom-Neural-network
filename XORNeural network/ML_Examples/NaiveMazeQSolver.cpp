//
//  NaiveMazeQSolver.cpp
//  This algorithm is an example of a naïve AI Maze Solver implementing only QLearning techniques
//  The aim of this work is to illustrate how much DeepQLearning is an improvement for this type of problems
//
//  Created by Aldric Labarthe on 29/10/2023.
//

#include <stdio.h>
#include <exception>
#include <stdexcept>
#include <iostream>
#include <array>
#include <functional>
#include <algorithm>
#include <cmath>
#include <random>

const float wallpenalty = -0.8;
const float moovepenalty = -0.04;
const float winpenalty = 1;
const float alreadyseenpenalty = -0.1;
//const float alreadyseenpenalty = 0;
const float explorationpenalty = 0.1;
const float explorationrate = 0.4;


void NaiveMazeQSolver(){
    /*const std::array<std::array<bool, 5>, 5> map = {{
        {{ 0, 1, 0, 1, 0}},
        {{ 0, 1, 0, 1, 0}},
        {{ 0, 1, 0, 1, 0}},
        {{ 0, 0, 0, 0, 0}},
        {{ 0, 1, 0, 0, 0}}
    }};*/
    
    const std::array<std::array<bool, 5>, 5> map = {{
        {{ 0, 1, 0, 1, 0}},
        {{ 0, 0, 0, 1, 0}},
        {{ 0, 1, 1, 1, 0}},
        {{ 0, 0, 0, 0, 0}},
        {{ 0, 1, 0, 0, 0}}
    }};
    
    struct PossibleActions {
        float left = 0;
        float right = 0;
        float top = 0;
        float down = 0;
    };
    
    const std::array<int, 2> start = { 0, 0 };
    const std::array<int, 2> exit = { 0, 4 };
    
    class Agent {
        std::array<std::array<bool, 5>, 5> visited_map;
        std::array<std::array<PossibleActions*, 5>, 5> *qmap;
        
    public:
        std::array<int, 2> pos = { 0 , 0};
        
        Agent(int posx, int posy, std::array<std::array<PossibleActions*, 5>, 5>* qmap_pointer) {
            pos = {posx, posy};
            qmap = qmap_pointer;
            for (int row(0); row < 5; row++){
                for (int col(0); col < 5; col++){
                    visited_map[row][col] = 0;
                }
            }
            visited_map[posx][posy] = 1;
        }
        
        std::array<int, 2> chooseWhereToGo(){
            std::array<int, 2> wheretogo;
            char decidedMoove = 'L';
            float reward = 0;
            if ((*qmap)[pos[0]][pos[1]] == nullptr)
                throw std::invalid_argument("We just hit a nullptr case");
            
            if (pos[1]-1 >= 0 && visited_map[pos[0]][pos[1]-1] == 1){
                    if ((*qmap)[pos[0]][pos[1]]->left-alreadyseenpenalty > reward){
                        decidedMoove = 'L';
                        reward = (*qmap)[pos[0]][pos[1]]->left-alreadyseenpenalty;
                    }
            }else {
                    decidedMoove = 'L';
                    reward = (*qmap)[pos[0]][pos[1]]->left;
            }
            wheretogo = {pos[0], pos[1]-1};
            if ((*qmap)[pos[0]][pos[1]]->right > reward){
                if (pos[1]+1 < 5 && visited_map[pos[0]][pos[1]+1] == 1){
                    if ((*qmap)[pos[0]][pos[1]]->right-alreadyseenpenalty > reward){
                        decidedMoove = 'R';
                        reward = (*qmap)[pos[0]][pos[1]]->right-alreadyseenpenalty;
                        wheretogo = {pos[0], pos[1]+1};
                    }
                }else {
                    decidedMoove = 'R';
                    reward = (*qmap)[pos[0]][pos[1]]->right;
                    wheretogo = {pos[0], pos[1]+1};
                }
            }
            if ((*qmap)[pos[0]][pos[1]]->top > reward){
                if (pos[0]-1 >= 0 && visited_map[pos[0]-1][pos[1]] == 1){
                    if ((*qmap)[pos[0]][pos[1]]->top-alreadyseenpenalty > reward){
                        decidedMoove = 'T';
                        reward = (*qmap)[pos[0]][pos[1]]->top-alreadyseenpenalty;
                        wheretogo = {pos[0]-1, pos[1]};
                    }
                }else {
                    decidedMoove = 'T';
                    reward = (*qmap)[pos[0]][pos[1]]->top;
                    wheretogo = {pos[0]-1, pos[1]};
                }
            }
            if ((*qmap)[pos[0]][pos[1]]->down > reward){
                if (pos[0]+1 < 5 && visited_map[pos[0]+1][pos[1]] == 1){
                    if ((*qmap)[pos[0]][pos[1]]->down-alreadyseenpenalty > reward){
                        decidedMoove = 'D';
                        reward = (*qmap)[pos[0]][pos[1]]->down-alreadyseenpenalty;
                        wheretogo = {pos[0]+1, pos[1]};
                    }
                }else {
                    decidedMoove = 'D';
                    reward = (*qmap)[pos[0]][pos[1]]->down;
                    wheretogo = {pos[0]+1, pos[1]};
                }
            }
            std::random_device rd; std::mt19937 gen(rd()); std::uniform_int_distribution<> distrib(1, 10);
            if (distrib(gen) <= explorationrate*10){
                std::cout << "RANDOM Mode enabled" << std::endl;
                std::uniform_int_distribution<> unitdistrib(0, 1);
                if (distrib(gen)%2==0){
                    if (unitdistrib(gen)==0)
                        wheretogo = {pos[0], pos[1]-1};
                    else
                        wheretogo = {pos[0], pos[1]+1};
                }else{
                    if (unitdistrib(gen)==0)
                        wheretogo = {pos[0]-1, pos[1]};
                    else
                        wheretogo = {pos[0]+1, pos[1]};
                }
            }else {
                std::cout << "I will head to " << decidedMoove << std::endl;
            }
            return wheretogo;
        }
        
        void makememory(int posx, int posy){
            visited_map[posx][posy] = 1;
        }
        
        bool haveYouBeenThere(int posx, int posy){
            return visited_map[posx][posy];
        }
    };
    
    std::array<std::array<PossibleActions*, 5>, 5> qmap;
    for (int row(0); row < 5; row++){
        for (int col(0); col < 5; col++){
            if (map[row][col])
                qmap[row][col] = nullptr;
            else
                qmap[row][col] = new PossibleActions;
        }
    }
    
    Agent* player = nullptr;
    for (int iteration(0); iteration < 10000; iteration++){
        std::cout << "Beginning iteration " << iteration << std::endl;
        if (player == nullptr){
            player = new Agent(start[0], start[1], &qmap);
            std::cout << "Initializing Agent" << std::endl;
        }
        std::array<int, 2> wheretogo = player->chooseWhereToGo();
        std::array<int, 2> curpos = player->pos;
        char direction;
        if (wheretogo[0] < player->pos[0])
            direction = 'T';
        else if (wheretogo[0] > player->pos[0])
            direction = 'D';
        else if (wheretogo[1] < player->pos[1])
            direction = 'L';
        else
            direction = 'R';
        
        float reward_to_give = explorationpenalty;
        
        std::cout << "Agent attempts to go to x=" << wheretogo[0] << " y=" << wheretogo[1]  << "(interpreted as " << direction << ")"<< std::endl;
        
        if (wheretogo[0] < 0 || wheretogo[0] >= 5 || wheretogo[1] < 0 || wheretogo[1] >= 5 || map[wheretogo[0]][wheretogo[1]]){
            std::cout << "Forbidden choice, killing agent and updating qmap" << std::endl;
            //delete player;
            //player = nullptr;
            reward_to_give = wallpenalty;
        }else {
            float prevpenalty = 0;
            if (direction == 'D')
                prevpenalty = qmap[curpos[0]][curpos[1]]->down;
            else if (direction == 'T')
                prevpenalty = qmap[curpos[0]][curpos[1]]->top;
            else if (direction == 'L')
                prevpenalty = qmap[curpos[0]][curpos[1]]->left;
            else if (direction == 'R')
                prevpenalty = qmap[curpos[0]][curpos[1]]->right;
            
            if (player->haveYouBeenThere(wheretogo[0], wheretogo[1]))
                reward_to_give = std::max((float)-0.5, prevpenalty+alreadyseenpenalty);
            
            std::cout << "Agent gets " << reward_to_give << " for heading to " << direction << std::endl;
            
            player->pos = wheretogo;
            player->makememory(wheretogo[0], wheretogo[1]);
            
            if (wheretogo == exit){
                reward_to_give += winpenalty;
                std::cout << "Won ! Break;" << std::endl;
                for (int row(0); row < 5; row++){
                    for (int col(0); col < 5; col++){
                        std::cout << player->haveYouBeenThere(row, col) << " ";
                    }
                    std::cout << std::endl;
                }
                break;
            }
        }
        if (direction == 'D')
            qmap[curpos[0]][curpos[1]]->down = reward_to_give;
        else if (direction == 'T')
            qmap[curpos[0]][curpos[1]]->top = reward_to_give;
        else if (direction == 'L')
            qmap[curpos[0]][curpos[1]]->left = reward_to_give;
        else if (direction == 'R')
            qmap[curpos[0]][curpos[1]]->right = reward_to_give;
        
    }
    
    
}
