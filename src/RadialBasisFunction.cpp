//
// Created by flore on 20/01/2024.
//

#include "RadialBasisFunction.h"

void RadialBasisFunction::updateRepresentants(std::vector<std::vector<Real>> inputs,Integer nb) {
    representants.clear();
    std::vector<std::vector<std::vector<Real>>> classes;
    int k=inputs.size();
    for (int i = 0; i < nb; ++i) {
        int random=static_cast<int>(ML_RAND*k);
        std::vector<Real> rep=inputs[random];
        representants.push_back(rep);
        std::vector<std::vector<Real>> cl;
        classes.push_back(cl);
    }
    bool stop=true;
    //TO-DO exitCondition check lastRepresentants
    while(!stop){
            //Clear existing classes
            for(auto cl : classes)
                cl.clear();
            //Place all elements in their respective classes based on closest representants
            for(const auto x : inputs) {
                classes[closest(x)].push_back(x);
            }
            //Calculate new representants based on class average
            for (int i = 0; i < classes.size(); ++i) {
                std::vector<Real> sum;
                std::vector<std::vector<Real>> &currClass = classes[i];
                int classSize= currClass.size();
                for (int j = 0; j < currClass[0].size(); ++j)
                    sum.push_back(0.0);
                for(const auto& member : currClass)
                    for (int j = 0; j < member.size(); ++j)
                        sum[j]+=member[j]/classSize;
                representants[i]=sum;
            }
        }
}

void RadialBasisFunction::train(std::vector<std::vector<Real>> inputs,  Eigen::Matrix<Real, Eigen::Dynamic,1> outputs) {
    int nbRep=static_cast<int>(inputs.size()*0.1);
    int r=inputs[0].size();
    updateRepresentants(inputs,nbRep);
    Eigen::Matrix<Real ,Eigen::Dynamic,Eigen::Dynamic> mat(r,nbRep);
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < nbRep; ++j) {
            mat(i,j)=std::exp(-1.0*gamma*std::pow(dist(inputs[i],representants[j]),2));
        }
    }
     m_W=(mat.transpose()*mat.inverse())*mat.transpose()*outputs;
}
int RadialBasisFunction::closest(const std::vector<Real> element){
    auto min=representants[0];
    int iMin=0;
    for (int i = 1; i < representants.size(); ++i) {
        if(dist(min,representants[i])){
            iMin=i;
            min=representants[i];
        }
    }
    return iMin;
}
Real RadialBasisFunction::dist(const std::vector<Real> v1,const std::vector<Real> v2){
    Real sum=0;
    for (int i = 0; i <std::min(v1.size(),v2.size()); ++i) {
        sum+=std::pow(v1[0]-v2[0],2);
    }
    return std::sqrt(sum);

}

Real RadialBasisFunction::predict(Eigen::Matrix<Real, Eigen::Dynamic, 1> inputs) {
    return (m_W*inputs)(0,0);
}
