//
// Created by flore on 10/01/2024.
//

#ifndef MACHINELEARNING_LINEARMODEL_H
#define MACHINELEARNING_LINEARMODEL_H
#include <vector>
#include "library.h"
#include "UUID.hpp"
#include <limits>
#include <cmath>
#include <iostream>

class LinearModel {
private:
    Real step;
    std::vector<Real> weight;
    std::vector<std::vector<Real>> X;
    std::vector<Real> Y;
public:
    LinearModel(Real step,const Real* entries, const Real* output,Integer entrySize, Integer entryCount) : step(step){
        for (int i = 0; i < entryCount; ++i) {
            Y.push_back(output[i]);
            std::vector<Real> entry(entrySize);
            for (int j = 0; j < entrySize; ++j) {
                entry.push_back(entries[i*entrySize+j]);
            }
            X.push_back(entry);
        }
        for (int i = 0; i < entrySize+1; ++i) {
            auto max=static_cast<Real>(std::numeric_limits<uint64_t>().max());
            weight.push_back(ML_RAND);
        }
    }
    //0 PLA, 1 Rosenblatt, 2 pseudoInvert
    void Train(int times,int mode){
        for (int i = 0; i < times; ++i) {
            Integer k= static_cast<Integer>(std::floor(ML_RAND * static_cast<Real>(X.size())));
            auto x=X[k];
            Real y=Y[k];
            switch(mode){
                case 1 : trainOncePLA(x,y);break;
                case 0 : trainOnceRosenblatt(x,y);break;
                case 2 : throw "not implemented yet"; break;
            }
            std::cout<<weight[0]<<" weights "<< weight[1] << " 2 =>" << weight[2]<<std::endl;
        }
    }
    Real predict(const Real* x) {
        Real result=0;
        result+=weight[0];
        for (int i = 1; i < weight.size(); ++i) {
            result+=x[i-1]*weight[i];
        }
        return result >= 0 ? 1.0 : -1.0;
    }
    Real predict(const std::vector<Real> x) {
        Real result=0;
        result+=weight[0];
        for (int i = 1; i < weight.size(); ++i) {
            result+=x[i-1]*weight[i];
        }
        return result >= 0 ? 1.0 : -1.0;
    }
private:
    void trainOnceRosenblatt(std::vector<Real> input,Real expected){
        Real dist=step*(expected - predict(input));
        weight[0]=weight[0]+dist*1;
        for (int i = 1; i < weight.size(); ++i) {
            weight[i] = weight[i] + dist * input[i - 1];
        }
    }
    void trainOncePLA(std::vector<Real> input,Real expected){
        weight[0]=weight[0]+step * expected*1;
        for (int i = 0; i < weight.size(); ++i) {
            weight[i] = weight[i] + step * expected *  input[i - 1];
        }
    }

};


#endif //MACHINELEARNING_LINEARMODEL_H
