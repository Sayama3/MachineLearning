#include "library.h"
#include "MultiLayerPerceptron.hpp"
#include "LinearModel.h"

#include <iostream>
#include <vector>

using namespace GG::ML;

//static TypeId s_MLPId = 0;
//static TypeId s_LinearId = 0;
static std::vector<MultiLayerPerceptron*>* s_MLPs = nullptr;
static std::vector<LinearModel*>* s_Linears = nullptr;

#define ML_LOG(str) std::cout << str << std::endl

int infos()
{
    std::cout << "Library 'MachineLearning' is working." << std::endl;
    return 0;
}

void initialize()
{
	ML_LOG("Initialize");
	s_MLPs = new std::vector<MultiLayerPerceptron *>();
    s_Linears = new std::vector<LinearModel*>();
}

void update(Real timestep)
{

}

void shutdown()
{
	ML_LOG("Shutdown");
	if(s_MLPs)
	{
		for (auto ptr : (*s_MLPs)) {
			delete ptr;
		}
		delete s_MLPs;
	}

    if(s_Linears)
	{
		for (auto ptr : (*s_Linears)) {
			delete ptr;
		}
		delete s_Linears;
	}
}

TypeId mlpCreate(const Integer* entries, Integer count)
{
	if(!s_MLPs) return -1;
	s_MLPs->push_back(new MultiLayerPerceptron(entries, count));
	auto index = s_MLPs->size() - 1;
	ML_LOG("Create mlp at index '" << std::to_string(index) << "'");
	return index;
}
TypeId linearCreate(bool isClassification,Real step,Integer entrySize){
	if(!s_Linears) return -1;
	s_Linears->push_back(new LinearModel(isClassification,step,entrySize));
	auto index = s_Linears->size() - 1;
	ML_LOG("Create linear at index '" << std::to_string(index) << "'");
    return index;
}


bool linearIsValid(TypeId id)
{
	return s_Linears && id >= 0 && id < (*s_Linears).size() && (*s_Linears)[id];
}

bool mlpIsValid(TypeId id)
{
    return s_MLPs && id >= 0 && id < (*s_MLPs).size() && (*s_MLPs)[id];
}

void mlpDelete(TypeId id)
{
	if(!mlpIsValid(id)){ML_LOG("'mlpDelete' - id '" << std::to_string(id) << "' doesn't exist."); return;}

	delete (*s_MLPs)[id];
	(*s_MLPs)[id] = nullptr;
}


void mlpPropagate(TypeId id, const Real* rawInputs, Integer rawInputsCount, bool isClassification)
{
	if(!mlpIsValid(id)){ML_LOG("'mlpPropagate' - id '" << std::to_string(id) << "' doesn't exist."); return;}

	(*s_MLPs)[id]->Propagate(rawInputs ,rawInputsCount ,isClassification);
}

Integer mlpPredict(TypeId id, const Real* rawInputs, Integer rawInputsCount, bool isClassification)
{
	if(!mlpIsValid(id)){ML_LOG("'mlpPredict' - id '" << std::to_string(id) << "' doesn't exist."); return 0;}

	auto val = (*s_MLPs)[id]->Predict(rawInputs, rawInputsCount , isClassification);
	//ML_LOG("mlp predict count : " << std::to_string(val));
	return val;
}

Real mlpGetPredictData(TypeId id, Integer index)
{
	if(!mlpIsValid(id)){ML_LOG("'mlpGetPredictData' - id '" << std::to_string(id) << "' doesn't exist."); return 0;}
	auto val = (*s_MLPs)[id]->GetPredictData(index);
	//ML_LOG("mlp predict data at index '" << std::to_string(index) << "' : " << std::to_string(val));
	return val;
}

void mlpTrain(TypeId id, const Real* rawAllInputs, Integer rawAllInputsWidth, Integer rawAllInputsHeight, const Real* rawExcpectedOutputs, Integer rawExcpectedOutputsWidth, Integer rawExcpectedOutputsHeight, bool isClassification, Real alpha, Integer maxIter)
{
	if(!mlpIsValid(id)){ML_LOG("'mlpTrain' - id '" << std::to_string(id) << "' doesn't exist."); return;}

	(*s_MLPs)[id]->Train(rawAllInputs , rawAllInputsWidth , rawAllInputsHeight , rawExcpectedOutputs , rawExcpectedOutputsWidth , rawExcpectedOutputsHeight , isClassification , alpha , maxIter);
}
void linearTrain(TypeId id,Integer count,const Real* entries, const Real* output, Integer entryCount)
{
    if(!linearIsValid(id)){ML_LOG("'linearTrain' - id '" << std::to_string(id) << "' doesn't exist."); return;}

    (*s_Linears)[id]->train(count,entries,output,entryCount);
}
Real linearEvaluate(TypeId id,const Real* entries){
    if(!linearIsValid(id)){ML_LOG("'linearEvaluate' - id '" << std::to_string(id) << "' doesn't exist."); return 0.0;}
    return (*s_Linears)[id]->predict(entries);
}
void linearDelete(TypeId id)
{
    if(!linearIsValid(id)){ML_LOG("'linearDelete' - id '" << std::to_string(id) << "' doesn't exist."); return;}

    delete (*s_Linears)[id];
    (*s_Linears)[id] = nullptr;
}
