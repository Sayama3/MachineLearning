//
// Created by flore on 10/01/2024.
//

#include "LinearModel.h"
void LinearModel::train(int times,const Real* entries, const Real* output, Integer entryCount){
    for (int i = 0; i < times; ++i) {
        Integer k= static_cast<Integer>(std::floor(ML_RAND * static_cast<Real>(entryCount)));
        std::vector<Real> x(entries+k*size,entries+(k+1)*size);
        Real y=output[k];
//std::cout<<i<<"/"<<times<<"Params"<<weight[0]<<"<= [0] weights =>[1] "<< weight[1] << " [2] =>" << weight[2]<<std::endl;
        if(isClassification){
            trainOnceRosenblatt(x,y);
        }
        else{
            std::cout<<"not implemented yet"<<std::endl;
            throw "not implemented yet";
        }
    }
}
Real LinearModel::predict(const std::vector<Real> x) {
    Real result=0;
    result+=weight[0];
    for (int i = 1; i < weight.size(); ++i) {
        result+=x[i-1]*weight[i];
    }
    return result >= 0 ? 1.0 : -1.0;
}
void LinearModel::trainOnceRosenblatt(std::vector<Real> input,Real expected){
    auto p=predict(input);
    Real dist=step*(expected - p);
//std::cout<<input[0]<<","<<input[1]<<"Training"<<expected<<" vs "<<p<<" inp "<<input[2]<<std::endl;
    weight[0]=weight[0]+dist*1;
    for (int i = 1; i < weight.size(); ++i) {
        weight[i] = weight[i] + dist * input[i - 1];
    }
}


