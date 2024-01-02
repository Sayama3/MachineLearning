#include "library.h"
#include "MultiLayerPerceptron.hpp"

#include <iostream>
#include <vector>

using namespace GG::ML;

static TypeId s_MLPId = 0;
static std::vector<MultiLayerPerceptron*>* s_MLPs = nullptr;

void infos()
{
    std::cout << "Library 'MachineLearning' is working." << std::endl;
}

void initialize()
{
	s_MLPs = new std::vector<MultiLayerPerceptron *>();
	for (int i = 0; i < s_MLPId; ++i)
	{
		s_MLPs->push_back(nullptr);
	}
}

void update(Real timestep)
{

}

void shutdown()
{
	delete s_MLPs;
	s_MLPId = 0;
}

TypeId mlpCreate(const Integer* entries, Integer count)
{
	if(!s_MLPs) return s_MLPId++;
	TypeId id = s_MLPId++;
	s_MLPs->push_back(new MultiLayerPerceptron(entries, count));
	return id;
}

bool mlpIsValid(TypeId id)
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
