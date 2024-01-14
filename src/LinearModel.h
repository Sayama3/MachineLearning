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
            weight.push_back((ML_RAND*max)-max/2.0f);
        }
    }
    //0 PLA, 1 Rosenblatt, 2 pseudoInvert
    void Train(int times,int mode){
        for (int i = 0; i < times; ++i) {
            Integer k= static_cast<Integer>(std::floor(ML_RAND * static_cast<Real>(X.size())));
            auto x=X[k];
            Real y=Y[k];
            switch(mode){
                case 0 : trainOncePLA(x,y);break;
                case 1 : trainOnceRosenblatt(x,y);break;
                case 2 : throw "not implemented yet"; break;
            }
        }
    }
    Real predict(const Real* x) {
        Real result=0;
        result+=weight[0];
        for (int i = 1; i < weight.size(); ++i) {
            result+=(x[i-1]*weight[i]);
        }
        return result;
    }
    Real predict(const std::vector<Real> x) {
        Real result=0;
        result+=weight[0];
        for (int i = 1; i < weight.size(); ++i) {
            result+=(x[i-1]*weight[i]);
        }

        return result;
    }
private:
    void trainOnceRosenblatt(std::vector<Real> input,Real expected){
        Real dist=step*(expected - predict(input));
        for (int i = 0; i < weight.size(); ++i) {
            weight[i] = weight[i] + dist * (i == 0 ? 1 : input[i - 1]);
        }
    }
    void trainOncePLA(std::vector<Real> input,Real expected){
        for (int i = 0; i < weight.size(); ++i) {
            weight[i] = weight[i] + step * expected * (i == 0 ? 1 : input[i - 1]);
        }
    }

};


#endif //MACHINELEARNING_LINEARMODEL_H
