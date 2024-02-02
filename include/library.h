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

	ML_API bool linearIsValid(TypeId id);
    ML_API TypeId linearCreate(bool isClassification,Real step,Integer entrySize);
    ML_API void linearTrain(TypeId id,Integer count,const Real* entries, const Real* output, Integer entryCount);

    ML_API Real linearEvaluate(TypeId id,const Real* entries);
    ML_API void linearDelete(TypeId id);
	ML_API void linearSave(TypeId id, const char* fullPath);
	ML_API TypeId linearLoad(const char* fullPath);

	ML_API TypeId rbfCreate(Real gamma);
	ML_API bool rbfIsValid(TypeId id);
	ML_API void rbfDelete(TypeId id);
	ML_API Real rbfPredict(TypeId id, bool isClassification,const Real* rawInputs, Integer rawInputsCount);
    ML_API void rbfTrain(TypeId id, Integer sizeOfModel,const Real* rawAllInputs, Integer numberOfInputSubArray, Integer sizeOfInputSubArray, const Real* matrixOutputRowAligned, Integer sizeOfRow, Integer numberOfRow,Integer maxKMeanIter);
    ML_API void rbfTrainDefault(TypeId id, Integer sizeOfModel,const Real* rawAllInputs, Integer numberOfInputSubArray, Integer sizeOfInputSubArray, const Real* matrixOutputRowAligned, Integer sizeOfRow, Integer numberOfRow);
	ML_API void rbfSave(TypeId id, const char* fullPath);
	ML_API TypeId rbfLoad(const char* fullPath);

}