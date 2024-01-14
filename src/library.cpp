#include "library.h"
#include "MultiLayerPerceptron.hpp"
#include "LinearModel.h"

#include <iostream>
#include <vector>

using namespace GG::ML;

static TypeId s_MLPId = 0;
static TypeId s_LinearId = 0;
static std::vector<MultiLayerPerceptron*>* s_MLPs = nullptr;
static std::vector<LinearModel*>* s_Linears = nullptr;

int infos()
{
    std::cout << "Library 'MachineLearning' is working." << std::endl;
    return 0;
}

void initialize()
{
	s_MLPs = new std::vector<MultiLayerPerceptron *>();
    s_Linears = new std::vector<LinearModel*>();
	for (int i = 0; i < s_MLPId; ++i)
	{
		s_MLPs->push_back(nullptr);
	}
    for (int i = 0; i < s_LinearId; ++i)
    {
        s_Linears->push_back(nullptr);
    }
}

void update(Real timestep)
{

}

void shutdown()
{
	delete s_MLPs;
	s_MLPId = 0;
    delete s_Linears;
    s_LinearId = 0;
}

TypeId mlpCreate(const Integer* entries, Integer count)
{
	if(!s_MLPs) return s_MLPId++;
	TypeId id = s_MLPId++;
	s_MLPs->push_back(new MultiLayerPerceptron(entries, count));
	return id;
}
TypeId linearCreate(Real step,const Real* entries, const Real*output,Integer entrySize, Integer entryCount){
    if(!s_Linears) return s_LinearId++;
    TypeId  id = s_LinearId++;
    s_Linears->push_back(new LinearModel(step,entries,output,entrySize,entryCount));
    return id;
}


bool mlpIsValid(TypeId id)
{
	return s_Linears && id < s_LinearId && (*s_Linears)[id];
}
bool linearIsValid(TypeId id)
{
    return s_MLPs && id < s_MLPId && (*s_MLPs)[id];
}

void mlpDelete(TypeId id)
{
	if(!mlpIsValid(id)) return;

	delete (*s_MLPs)[id];
	(*s_MLPs)[id] = nullptr;
}


void mlpPropagate(TypeId id, const Real* rawInputs, Integer rawInputsCount, bool isClassification)
{
	if(!mlpIsValid(id)) return;

	(*s_MLPs)[id]->Propagate(rawInputs ,rawInputsCount ,isClassification);
}

Real mlpPredict(TypeId id, const Real* rawInputs, Integer rawInputsCount, bool isClassification)
{
	if(!mlpIsValid(id)) return -1.0;

	return (*s_MLPs)[id]->Predict(rawInputs, rawInputsCount , isClassification);
}

void mlpTrain(TypeId id, const Real* rawAllInputs, Integer rawAllInputsWidth, Integer rawAllInputsHeight, const Real* rawExcpectedOutputs, Integer rawExcpectedOutputsWidth, Integer rawExcpectedOutputsHeight, bool isClassification, float alpha, Integer maxIter)
{
	if(!mlpIsValid(id)) return;

	(*s_MLPs)[id]->Train(rawAllInputs , rawAllInputsWidth , rawAllInputsHeight , rawExcpectedOutputs , rawExcpectedOutputsWidth , rawExcpectedOutputsHeight , isClassification , alpha , maxIter);
}
void linearTrain(TypeId id,Integer count,Integer mode){
    if(!linearIsValid(id)) return;

    (*s_Linears)[id]->Train(count,mode);
}
Real linearEvaluate(TypeId id,const Real* entries){
    if(!linearIsValid(id)) return 0.0;
    return (*s_Linears)[id]->predict(entries);
}
void linearDelete(TypeId id)
{
    if(!linearIsValid(id)) return;

    delete (*s_Linears)[id];
    (*s_Linears)[id] = nullptr;
}
