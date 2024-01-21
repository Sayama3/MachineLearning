#include "library.h"
#include "MultiLayerPerceptron.hpp"
#include "LinearModel.h"
#include "RadialBasisFunction.h"

#include <Eigen/Dense>

#include <iostream>
#include <vector>

using namespace GG::ML;

//static TypeId s_MLPId = 0;
//static TypeId s_LinearId = 0;
static std::vector<MultiLayerPerceptron*>* s_MLPs = nullptr;
static std::vector<LinearModel*>* s_Linears = nullptr;
static std::vector<RadialBasisFunction*>* s_RBFs = nullptr;

#define ML_LOG(str) std::cout << str << std::endl

// =============== GLOBAL =============== //

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
	s_RBFs = new std::vector<RadialBasisFunction*>();
}

void update(Real timestep)
{

}

void shutdown()
{
	ML_LOG("Shutdown");
	if(s_MLPs)
	{
		for (MultiLayerPerceptron* ptr : (*s_MLPs)) {
			delete ptr;
		}
		delete s_MLPs;
	}

	if(s_Linears)
	{
		for (LinearModel* ptr : (*s_Linears)) {
			delete ptr;
		}
		delete s_Linears;
	}

	if(s_RBFs)
	{
		for (RadialBasisFunction* ptr : (*s_RBFs)) {
			delete ptr;
		}
		delete s_RBFs;
	}
}

// =============== MULTI LAYER PERCEPTRON =============== //


TypeId mlpCreate(const Integer* entries, Integer count)
{
	if(!s_MLPs) return -1;
	s_MLPs->push_back(new MultiLayerPerceptron(entries, count));
	auto index = s_MLPs->size() - 1;
	ML_LOG("Create mlp at index '" << std::to_string(index) << "'");
	return index;
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

void mlpTrain(TypeId id, const Real* rawAllInputs, Integer inputSize, Integer inputsCount, const Real* rawExcpectedOutputs, Integer outputSize, Integer outputsCount, bool isClassification, Real alpha, Integer maxIter)
{
	if(!mlpIsValid(id)){ML_LOG("'mlpTrain' - id '" << std::to_string(id) << "' doesn't exist."); return;}

	(*s_MLPs)[id]->Train(rawAllInputs, inputSize, inputsCount , rawExcpectedOutputs, outputSize, outputsCount , isClassification , alpha , maxIter);
}

// =============== LINEAR MODEL =============== //

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


// =============== RADIAL BASIS FUNCTIONS =============== //


TypeId rbfCreate(Real gamma)
{
	if(!s_RBFs) return -1;
	s_RBFs->push_back(new RadialBasisFunction(gamma));
	auto index = s_RBFs->size() - 1;
	ML_LOG("Create RadialBasisFunction at index '" << std::to_string(index) << "'");
	return index;
}

bool rbfIsValid(TypeId id)
{
	return s_RBFs && id >= 0 && id < (*s_RBFs).size() && (*s_RBFs)[id];
}

void rbfDelete(TypeId id)
{
	if(!rbfIsValid(id)){ML_LOG("'rbfDelete' - id '" << std::to_string(id) << "' doesn't exist."); return;}

	delete (*s_RBFs)[id];
	(*s_RBFs)[id] = nullptr;
}

Real rbfPredict(TypeId id, const Real* rawInputs, Integer rawInputsCount)
{
	if(!rbfIsValid(id)){ML_LOG("'rbfDelete' - id '" << std::to_string(id) << "' doesn't exist."); return 0.0;}

	Eigen::Matrix<Real, Eigen::Dynamic,1> mat(rawInputsCount, 1);
	for (int row = 0; row < rawInputsCount; ++row) {
		mat(row, 0) = rawInputs[row];
	}

	return (*s_RBFs)[id]->predict(mat);
}

void rbfTrain(TypeId id, const Real* rawAllInputs, Integer numberOfInputSubArray, Integer sizeOfInputSubArray, const Real* rawMatrixOutputRowAligned, Integer sizeOfRow, Integer numberOfRow)
{
	if(!rbfIsValid(id)){ML_LOG("'rbfTrain' - id '" << std::to_string(id) << "' doesn't exist."); return;}

	// Create Matrix
	Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> outputMatrix(sizeOfRow, numberOfRow);
	for (int column = 0; column < numberOfRow; ++column) {
		for (int row = 0; row < sizeOfRow; ++row) {
			outputMatrix(row, column) = rawMatrixOutputRowAligned[column * sizeOfRow + row];
		}
	}

	// Create vector
	std::vector<std::vector<Real>> inputs(numberOfInputSubArray);
	for (int subArray = 0; subArray < numberOfInputSubArray; ++subArray) {
		inputs.emplace_back(sizeOfInputSubArray);
		for (int indexInSubArray = 0; indexInSubArray < sizeOfInputSubArray; ++indexInSubArray) {
			inputs[subArray][indexInSubArray] = rawAllInputs[(subArray * sizeOfInputSubArray) + indexInSubArray];
		}
	}

	(*s_RBFs)[id]->train(inputs, outputMatrix);
}

