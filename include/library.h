#pragma once
#include <machinelearning_export.hpp>

extern "C"
{
	typedef int Integer;
	typedef int TypeId;
	typedef double Real;

    ML_API int infos();

	// Library functions
    ML_API void initialize();
    ML_API void update(Real timestep);
    ML_API void shutdown();

	// MultiLayerPerceptron functions
	ML_API TypeId mlpCreate(const Integer* entries, Integer count);
	ML_API void mlpDelete(TypeId id);
	ML_API bool mlpIsValid(TypeId id);
	ML_API Integer mlpPredict(TypeId id, const Real* rawInputs, Integer rawInputsCount, bool isClassification = true);
	ML_API Real mlpGetPredictData(TypeId id, Integer index);
	ML_API void mlpTrain(TypeId id, const Real* rawAllInputs, Integer inputSize, Integer inputsCount, const Real* rawExcpectedOutputs, Integer outputSize, Integer outputsCount, bool isClassification = true, Real alpha = 0.01f, Integer maxIter = 1000);
	ML_API void mlpSave(TypeId id, const char* fullPath);
	ML_API TypeId mlpLoad(const char* fullPath);

	ML_API Integer mlpGetLayersCount(TypeId id);
	ML_API Integer mlpGetLayer(TypeId id, Integer index);

	ML_API Integer mlpGetWeightCount(TypeId id);
	ML_API Integer mlpGetSubWeightCount(TypeId id, Integer subIndex);
	ML_API Integer mlpGetSubSubWeightCount(TypeId id, Integer subIndex, Integer subSubIndex);
	ML_API Real mlpGetWeight(TypeId id, Integer subIndex, Integer subSubIndex, Integer subSubSubIndex);
	ML_API void mlpSetWeight(TypeId id, Integer subIndex, Integer subSubIndex, Integer subSubSubIndex, Real weight);

	// Linear Functions
	ML_API bool linearIsValid(TypeId id);
    ML_API TypeId linearCreate(bool isClassification,Real step,Integer entrySize);
    ML_API void linearTrain(TypeId id,Integer count,const Real* entries, const Real* output, Integer entryCount);

    ML_API Real linearEvaluate(TypeId id,const Real* entries);
    ML_API void linearDelete(TypeId id);
	ML_API void linearSave(TypeId id, const char* fullPath);
	ML_API TypeId linearLoad(const char* fullPath);

	ML_API Real linearGetStep(TypeId id);
	ML_API Integer linearGetSize(TypeId id);
	ML_API Integer linearIsClassification(TypeId id);
	ML_API Integer linearWeightCount(TypeId id);
	ML_API Real linearGetWeight(TypeId id, Integer index);
	ML_API void linearSetWeight(TypeId id, Integer index, Real weight);

	// RBF Function
	ML_API TypeId rbfCreate(Real gamma);
	ML_API bool rbfIsValid(TypeId id);
	ML_API void rbfDelete(TypeId id);
	ML_API Real rbfPredict(TypeId id, bool isClassification,const Real* rawInputs, Integer rawInputsCount);
    ML_API void rbfTrain(TypeId id, Integer sizeOfModel,const Real* rawAllInputs, Integer numberOfInputSubArray, Integer sizeOfInputSubArray, const Real* matrixOutputRowAligned, Integer sizeOfRow, Integer numberOfRow,Integer maxKMeanIter);
    ML_API void rbfTrainDefault(TypeId id, Integer sizeOfModel,const Real* rawAllInputs, Integer numberOfInputSubArray, Integer sizeOfInputSubArray, const Real* matrixOutputRowAligned, Integer sizeOfRow, Integer numberOfRow);
	ML_API void rbfSave(TypeId id, const char* fullPath);
	ML_API TypeId rbfLoad(const char* fullPath);

	ML_API Real rbfGetGamma(TypeId id);
	ML_API Integer rbfGetRows(TypeId id);
	ML_API Integer rbfGetCols(TypeId id);
	ML_API Real rbfGetWeight(TypeId id, Integer row, Integer col);
	ML_API void rbfSetWeight(TypeId id, Integer row, Integer col, Real weight);
	ML_API Integer rbfGetSize(TypeId id);
	ML_API Real rbfGetWeightByIndex(TypeId id, Integer index);
	ML_API void rbfSetWeightByIndex(TypeId id, Integer index, Real weight);
}