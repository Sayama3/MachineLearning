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
    Integer size;
    std::vector<Real> weight;
    bool isClassification;
public:
    LinearModel(bool isClassification,Real step,Integer entrySize) : size(entrySize),step(step),isClassification(isClassification){
        for (int i = 0; i < entrySize+1; ++i) {
            auto max=static_cast<Real>(std::numeric_limits<uint64_t>().max());
            weight.push_back(ML_RAND);
        }
    }
    void Train(int times,const Real* entries, const Real* output, Integer entryCount){
        for (int i = 0; i < times; ++i) {
            Integer k= static_cast<Integer>(std::floor(ML_RAND * static_cast<Real>(entryCount)));
            std::vector<Real> x(entries+k*size,entries+(k+1)*size);
            Real y=output[k];
            std::cout<<i<<"/"<<times<<"Params"<<weight[0]<<"<= [0] weights =>[1] "<< weight[1] << " [2] =>" << weight[2]<<std::endl;
            if(isClassification){
                trainOnceRosenblatt(x,y);
            }
            else{
                std::cout<<"not implemented yet"<<std::endl;
                throw "not implemented yet";
            }
        }
    }
    Real predict(const Real* x) {
        std::vector<Real> v(x,x+size);
        return predict(v);
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
        auto p=predict(input);
        Real dist=step*(expected - p);
        std::cout<<input[0]<<","<<input[1]<<"Training"<<expected<<" vs "<<p<<" inp "<<input[2]<<std::endl;
        weight[0]=weight[0]+dist*1;
        for (int i = 1; i < weight.size(); ++i) {
            weight[i] = weight[i] + dist * input[i - 1];
        }
    }

};


#endif //MACHINELEARNING_LINEARMODEL_H
