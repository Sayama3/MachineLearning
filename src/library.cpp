#include "library.h"
#include "MultiLayerPerceptron.hpp"
#include "LinearModel.h"
#include "RadialBasisFunction.h"

#include <Eigen/Dense>

#include <iostream>
#include <vector>
#include <filesystem>

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

void mlpSave(TypeId id, const char* fullPath)
{
	if(!mlpIsValid(id)){ML_LOG("'mlpSave' - id '" << std::to_string(id) << "' doesn't exist."); return;}
	std::filesystem::path path(fullPath);
	std::filesystem::create_directories(path.parent_path());
	auto& mlp = (*s_MLPs)[id];
	bool success = mlp->Save(path);
	if(!success)
	{
		std::cerr << "Save of MLP '" << std::to_string(id) << "' has failed." << std::endl;
	}
}

TypeId mlpLoad(const char* fullPath)
{
	if(!s_MLPs) return -1;
	std::filesystem::path path(fullPath);
	s_MLPs->push_back(new MultiLayerPerceptron(path));
	TypeId index = s_MLPs->size() - 1;
	ML_LOG("Create mlp at index '" << std::to_string(index) << "'");
	return index;
}

Integer mlpGetLayersCount(TypeId id)
{
	if(!mlpIsValid(id)) {ML_LOG("'mlpGetLayersCount' - id '" << id << "' doesn't exist."); return -1;}
	return (*s_MLPs)[id]->GetLayersCount();
}

Integer mlpGetLayer(TypeId id, Integer index)
{
	if(!mlpIsValid(id)) {ML_LOG("'mlpGetLayer' - id '" << id << "' doesn't exist."); return -1;}
	return (*s_MLPs)[id]->GetLayer(index);
}

Integer mlpGetWeightCount(TypeId id)
{
	if(!mlpIsValid(id)) {ML_LOG("'mlpGetWeightCount' - id '" << id << "' doesn't exist."); return -1;}
	return (*s_MLPs)[id]->GetWeightCount();
}

Integer mlpGetSubWeightCount(TypeId id, Integer subIndex)
{
	if(!mlpIsValid(id)) {ML_LOG("'mlpGetSubWeightCount' - id '" << id << "' doesn't exist."); return -1;}
	return (*s_MLPs)[id]->GetSubWeightCount(subIndex);
}

Integer mlpGetSubSubWeightCount(TypeId id, Integer subIndex, Integer subSubIndex)
{
	if(!mlpIsValid(id)) {ML_LOG("'mlpGetSubSubWeightCount' - id '" << id << "' doesn't exist."); return -1;}
	return (*s_MLPs)[id]->GetSubSubWeightCount(subIndex, subSubIndex);
}

Real mlpGetWeight(TypeId id, Integer subIndex, Integer subSubIndex, Integer subSubSubIndex)
{
	if(!mlpIsValid(id)) {ML_LOG("'mlpGetWeight' - id '" << id << "' doesn't exist."); return -1;}
	return (*s_MLPs)[id]->GetWeight(subIndex, subSubIndex, subSubSubIndex);
}

void mlpSetWeight(TypeId id, Integer subIndex, Integer subSubIndex, Integer subSubSubIndex, Real weight)
{
	if(!mlpIsValid(id)) {ML_LOG("'mlpGetWeight' - id '" << id << "' doesn't exist."); return;}
	return (*s_MLPs)[id]->SetWeight(subIndex, subSubIndex, subSubSubIndex, weight);
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

void linearSave(TypeId id, const char* fullPath)
{
	if(!linearIsValid(id)){ML_LOG("'linearTrain' - id '" << std::to_string(id) << "' doesn't exist."); return;}
	std::filesystem::path path(fullPath);
	std::filesystem::create_directories(path.parent_path());
	bool success = (*s_Linears)[id]->save(path);
	if(!success)
	{
		std::cerr << "Save of Linear '" << std::to_string(id) << "' has failed." << std::endl;
	}
}

TypeId linearLoad(const char* fullPath)
{
	if(!s_Linears) return -1;
	std::filesystem::path path(fullPath);
	s_Linears->push_back(new LinearModel(path));
	auto index = s_Linears->size() - 1;
	ML_LOG("Create linear at index '" << std::to_string(index) << "'");
	return index;
}

Real linearGetStep(TypeId id)
{
	if(!linearIsValid(id)){ML_LOG("'linearGetStep' - id '" << std::to_string(id) << "' doesn't exist."); return -1;}
	return (*s_Linears)[id]->getStep();
}

Integer linearGetSize(TypeId id)
{
	if(!linearIsValid(id)){ML_LOG("'linearGetSize' - id '" << std::to_string(id) << "' doesn't exist."); return -1;}
	return (*s_Linears)[id]->getSize();
}

Integer linearIsClassification(TypeId id)
{
	if(!linearIsValid(id)){ML_LOG("'linearIsClassification' - id '" << std::to_string(id) << "' doesn't exist."); return -1;}
	return (*s_Linears)[id]->IsClassification();
}

Integer linearWeightCount(TypeId id)
{
	if(!linearIsValid(id)){ML_LOG("'linearWeightCount' - id '" << std::to_string(id) << "' doesn't exist."); return -1;}
	return (*s_Linears)[id]->weightCount();
}

Real linearGetWeight(TypeId id, Integer index)
{
	if(!linearIsValid(id)){ML_LOG("'linearGetWeight' - id '" << std::to_string(id) << "' doesn't exist."); return -1;}
	return (*s_Linears)[id]->getWeight(index);
}

void linearSetWeight(TypeId id, Integer index, Real weight)
{
	if(!linearIsValid(id)){ML_LOG("'linearGetWeight' - id '" << std::to_string(id) << "' doesn't exist."); return;}
	return (*s_Linears)[id]->setWeight(index, weight);
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

Real rbfPredict(TypeId id, bool isClassification, const Real* rawInputs, Integer rawInputsCount)
{
	if(!rbfIsValid(id)){ML_LOG("'rbfPredict' - id '" << std::to_string(id) << "' doesn't exist."); return 0.0;}

	Eigen::Matrix<Real, Eigen::Dynamic,1> mat(rawInputsCount, 1);
	for (int row = 0; row < rawInputsCount; ++row) {
		mat(row, 0) = rawInputs[row];
	}

	return (*s_RBFs)[id]->predict(isClassification,mat);
}
void rbfTrain(TypeId id, Integer sizeOfModel, const Real* rawAllInputs, Integer numberOfInputSubArray, Integer sizeOfInputSubArray, const Real* rawMatrixOutputRowAligned, Integer sizeOfRow, Integer numberOfRow,Integer maxKMeanIter)
{
        if (!rbfIsValid(id)) {
            ML_LOG("'rbfTrain' - id '" << std::to_string(id) << "' doesn't exist.");
            return;
        }

        std::cout << sizeOfInputSubArray << "iMax,jMax" << numberOfInputSubArray << std::endl;
        // Create Matrix
        std::vector<std::vector<Real>> inputs;
        for (int subArray = 0; subArray < numberOfInputSubArray; ++subArray) {
            inputs.emplace_back();
            for (int indexInSubArray = 0; indexInSubArray < sizeOfInputSubArray; ++indexInSubArray) {
                inputs[subArray].push_back(rawAllInputs[(subArray * sizeOfInputSubArray) + indexInSubArray]);
            }
        }
        //Create Vector
        //std::cout<<sizeOfRow<<"iMax,jMax"<<numberOfRow<<std::endl;
        Eigen::Matrix<Real, Eigen::Dynamic, 1> outputMatrix(numberOfRow);
        {
            for (int j = 0; j < numberOfRow; ++j) {
                //std::cout<<"j"<<j<< " gets outputed : "<<rawMatrixOutputRowAligned[j];
                outputMatrix(j) = rawMatrixOutputRowAligned[j];
            }
        }
	(*s_RBFs)[id]->train(sizeOfModel,inputs, outputMatrix,maxKMeanIter);
}
void rbfTrainDefault(TypeId id, Integer sizeOfModel, const Real* rawAllInputs, Integer numberOfInputSubArray, Integer sizeOfInputSubArray, const Real* rawMatrixOutputRowAligned, Integer sizeOfRow, Integer numberOfRow) {
    rbfTrain(id, sizeOfModel, rawAllInputs, numberOfInputSubArray, sizeOfInputSubArray,
             rawMatrixOutputRowAligned, sizeOfRow, numberOfRow, 1000);
}

void rbfSave(TypeId id, const char* fullPath)
{
	if (!rbfIsValid(id)) {
		ML_LOG("'rbfSave' - id '" << std::to_string(id) << "' doesn't exist.");
		return;
	}
	auto* rbf = (*s_RBFs)[id];
	std::filesystem::path path(fullPath);
	std::filesystem::create_directories(path.parent_path());
	bool success = rbf->save(path);
	if(!success)
	{
		std::cerr << "Save of Linear '" << std::to_string(id) << "' has failed." << std::endl;
	}
}

TypeId rbfLoad(const char* fullPath)
{
	if(!s_RBFs) return -1;
	std::filesystem::path path(fullPath);
	s_RBFs->push_back(new RadialBasisFunction(path));
	auto index = s_RBFs->size() - 1;
	ML_LOG("Create RadialBasisFunction at index '" << std::to_string(index) << "'");
	return index;
}

Real rbfGetGamma(TypeId id)
{
	if(!rbfIsValid(id)) {ML_LOG("'rbfGetGamma' - id '" << std::to_string(id) << "' doesn't exist."); return -1;}
	return (*s_RBFs)[id]->getGamma();
}

Integer rbfGetRows(TypeId id)
{
	if(!rbfIsValid(id)) {ML_LOG("'rbfGetRows' - id '" << std::to_string(id) << "' doesn't exist."); return -1;}
	return (*s_RBFs)[id]->getRows();
}

Integer rbfGetCols(TypeId id)
{
	if(!rbfIsValid(id)) {ML_LOG("'rbfGetCols' - id '" << std::to_string(id) << "' doesn't exist."); return -1;}
	return (*s_RBFs)[id]->getCols();
}

Real rbfGetWeight(TypeId id, Integer row, Integer col)
{
	if(!rbfIsValid(id)) {ML_LOG("'rbfGetWeight' - id '" << std::to_string(id) << "' doesn't exist."); return -1;}
	return (*s_RBFs)[id]->getWeight(row, col);
}
Integer rbfGetSize(TypeId id)
{
	if(!rbfIsValid(id)) {ML_LOG("'rbfGetSize' - id '" << std::to_string(id) << "' doesn't exist."); return -1;}
	return (*s_RBFs)[id]->getSize();
}
Real rbfGetWeightByIndex(TypeId id, Integer index)
{
	if(!rbfIsValid(id)) {ML_LOG("'rbfGetWeightByIndex' - id '" << std::to_string(id) << "' doesn't exist."); return -1;}
	return (*s_RBFs)[id]->getWeight(index);
}
void rbfSetWeight(TypeId id, Integer row, Integer col, Real weight)
{
	if(!rbfIsValid(id)) {ML_LOG("'rbfGetWeight' - id '" << std::to_string(id) << "' doesn't exist."); return;}
	(*s_RBFs)[id]->setWeight(row, col, weight);
}

void rbfSetWeightByIndex(TypeId id, Integer index, Real weight)
{
	if(!rbfIsValid(id)) {ML_LOG("'rbfGetWeightByIndex' - id '" << std::to_string(id) << "' doesn't exist."); return;}
	(*s_RBFs)[id]->setWeight(index, weight);
}

