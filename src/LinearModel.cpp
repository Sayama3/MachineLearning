//
// Created by flore on 10/01/2024.
//

#include "LinearModel.h"
#include <iostream>
#include <fstream>
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

LinearModel::LinearModel(const std::filesystem::path &fullPath) {
	if(!is_regular_file(fullPath))
	{
		std::cerr << "The file '" << fullPath <<"' is not a regular file." << std::endl;
		return;
	}
	std::ifstream saveFile(fullPath, std::ios::binary);
	Integer classif = isClassification;
	saveFile.read(reinterpret_cast<char*>(&step), sizeof(step));
	saveFile.read(reinterpret_cast<char*>(&size), sizeof(size));
	saveFile.read(reinterpret_cast<char*>(&classif), sizeof(classif));
	isClassification = classif;

	Integer weightCount;
	saveFile.read(reinterpret_cast<char*>(&weightCount), sizeof(weightCount));
	weight.resize(weightCount);
	for (int i = 0; i < weightCount; ++i) {
		saveFile.read(reinterpret_cast<char*>(&weight[i]), sizeof(weight[i]));
	}
}

bool LinearModel::save(const std::filesystem::path &fullPath) {
	std::ofstream saveFile(fullPath, std::ios::binary | std::ios::app | std::ios::trunc);
	Integer classif = isClassification;
	saveFile.write(reinterpret_cast<const char*>(&step), sizeof(step));
	saveFile.write(reinterpret_cast<const char*>(&size), sizeof(size));
	saveFile.write(reinterpret_cast<const char*>(&classif), sizeof(classif));

	Integer weightCount = weight.size();
	saveFile.write(reinterpret_cast<const char*>(&weightCount), sizeof(weightCount));
	for (int i = 0; i < weightCount; ++i) {
		saveFile.write(reinterpret_cast<const char*>(&weight[i]), sizeof(weight[i]));
	}
	saveFile.flush();
	saveFile.close();

	return true;
}

Real LinearModel::getStep() const {
	return step;
}

Integer LinearModel::getSize() const {
	return size;
}

Integer LinearModel::IsClassification() const {
	return isClassification;
}

Integer LinearModel::weightCount() const {
	return weight.size();
}

Real LinearModel::getWeight(Integer i) const {
	return weight[i];
}


