//
// Created by flore on 20/01/2024.
//

#include "RadialBasisFunction.h"

void RadialBasisFunction::updateRepresentants(const std::vector<std::vector<Real>>& inputs,Integer nb) {
    representants.clear();
    std::vector<std::vector<std::vector<Real>>> classes;
    int k=inputs.size();
    int sizeOfInput=k>0 ? inputs[0].size() : 0;
    std::vector<int> availableIndices(k);
    std::iota(availableIndices.begin(), availableIndices.end(), 0);
    while(representants.size()<nb && !availableIndices.empty()){
        //We ensure we have unique representants
        int pickedI=static_cast<int>(ML_RAND*availableIndices.size());
        int random=availableIndices[pickedI];
        availableIndices.erase(availableIndices.begin()+pickedI);
        std::cout<<"Random selected : "<<random<<std::endl;
        std::vector<Real> rep=inputs[random];
        representants.push_back(rep);
        std::vector<std::vector<Real>> cl;
        classes.push_back(cl);
    }
    bool stop=false;
    //TO-DO exitCondition check lastRepresentants
    while(!stop){
            //We'll stop except if one of the representant changed
            stop=true;
            //Clear existing classes
            std::cout<<"Clearing current classes"<<std::endl;
            for(auto cl : classes)
                cl.clear();
            //Place all elements in their respective classes based on closest representants
            std::cout<<"Assigning classes"<<std::endl;
            for(const auto x : inputs) {
                classes[closest(x)].push_back(x);
            }
            std::cout<<"Determining representants position"<<std::endl;
            //Calculate new representants based on class average
            for (int i = 0; i < classes.size(); ++i) {
                std::vector<Real> sum;
                std::vector<std::vector<Real>> &currClass = classes[i];
                int classSize= currClass.size();
                std::cout<<"Initializing sum array of size "<<currClass.size()<<std::endl;
                for (int j = 0; j < sizeOfInput; ++j)
                    sum.push_back(0.0);
                std::cout<<"Calculating average of class "<<i<<std::endl;
                for(const auto& member : currClass)
                    for (int j = 0; j < member.size(); ++j)
                        sum[j]+=member[j]/classSize;
                //We compare coord by coord the new value for i representant with the previous one
                //If all are equals we don't ask to continue
                std::cout<<"Checking if representant changed since last iteration, did it already change : "<<!stop<<std::endl;
                for(int coord=0;coord<representants[i].size();++coord) {
                    if (sum[coord] != representants[i][coord]) {
                        std::cout<<" checking coord "<<coord<<" : "<<sum[coord]<<std::endl;
                        stop = false;
                    }
                }
                std::cout<<"Assigning new pos to rep : "<<i<<std::endl;
                representants[i]=sum;
            }
        }
}

void RadialBasisFunction::train(Integer nbOfRepresentants,const std::vector<std::vector<Real>>& inputs,  const Eigen::Matrix<Real, Eigen::Dynamic,1>& outputs) {
    std::cout<<nbOfRepresentants<<" nbOfRepresentants,inputsCount "<<inputs.size()<<" Input size ; "<<inputs[0].size()<<std::endl;
    int r=inputs[0].size();
    updateRepresentants(inputs, nbOfRepresentants);
    std::cout << r << "r,nbRpe" << nbOfRepresentants << std::endl;
    Eigen::Matrix<Real ,Eigen::Dynamic,Eigen::Dynamic> mat(r, nbOfRepresentants);
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < nbOfRepresentants; ++j) {
            Real value=Exp(dist(inputs[i],representants[j]));
            std::cout<<i<<"i,j"<<j<<" = "<<value<<std::endl;
            mat(i,j)=value;
        }
    }
    std::cout<<"Computed Bell curve Xi x REPj"<<std::endl;
    std::cout<<mat<<std::endl<<std::endl;
    auto Mt=mat.transpose();
    std::cout<<"Mt"<<std::endl<<Mt<<std::endl<<std::endl;
    std::cout<<"Mt*M"<<std::endl<<Mt*mat<<std::endl;
    std::cout<<"(prod)-1"<<std::endl<<(Mt*mat).inverse()<<std::endl<<std::endl;
    std::cout<<"(prod)-1*prod"<<std::endl<<(Mt*mat).inverse()*(Mt*mat)<<std::endl<<std::endl;
    std::cout<<"prod-1*tM"<<std::endl<<(Mt*mat).inverse()*Mt<<std::endl;
    std::cout<<"outputs"<<std::endl<<outputs<<std::endl;
    std::cout<<"prod-1*tM*outputs"<<std::endl<<(Mt*mat).inverse()*Mt*outputs<<std::endl;
    std::cout<<" W : "<<m_W<<std::endl;
    m_W=((mat.transpose()*mat).inverse()*mat.transpose()*outputs).transpose();
    std::cout<<"Updated w"<<std::endl<<m_W<<std::endl;
}
int RadialBasisFunction::closest(const std::vector<Real>& element){
    Real min=dist(element,representants[0]);
    int iMin=0;
    for (int i = 1; i < representants.size(); ++i) {
        Real d=dist(element,representants[i]);
        if(d<min){
            iMin=i;
            min=d;
        }
    }
    return iMin;
}
Real RadialBasisFunction::dist(const std::vector<Real>& v1,const std::vector<Real>& v2){
    Real sum=0;
    for (int i = 0; i <std::min(v1.size(),v2.size()); ++i) {
        sum+=std::pow(v1[i]-v2[i],2);
    }
    return std::sqrt(sum);
}
Real RadialBasisFunction::dist(const std::vector<Real>& v1,const Eigen::Matrix<Real,Eigen::Dynamic,1>& v2){
    Real sum=0;
    for (int i = 0; i <std::min(v1.size(),(unsigned long long)v2.rows()); ++i) {
        sum+=std::pow(v1[i]-v2[i],2);
    }
    return std::sqrt(sum);
}

Real RadialBasisFunction::predict(bool isClassification, const Eigen::Matrix<Real, Eigen::Dynamic, 1> &inputs) {
    Real sum=0.0;
    bool log=false;//ML_RAND<0.01;
    if(log)
        std::cout<<inputs<<std::endl;
    for(int r=0;r<representants.size();++r){
        sum+=m_W[r]*Exp(dist(representants[r],inputs));
        if(log)
            std::cout<<"Evaluated : "<<r<<" to "<<std::exp(-gamma*dist(representants[r],inputs))<< " * "<<m_W[r]<<std::endl;
    }
    return isClassification ? (sum>=0.0 ? 1.0 : -1.0) : sum;
}
