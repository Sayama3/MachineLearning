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
	ML_API void mlpPropagate(TypeId id, const Real* rawInputs, Integer rawInputsCount, bool isClassification);
	ML_API Integer mlpPredict(TypeId id, const Real* rawInputs, Integer rawInputsCount, bool isClassification = true);
	ML_API Real mlpGetPredictData(TypeId id, Integer index);
	ML_API void mlpTrain(TypeId id, const Real* rawAllInputs, Integer rawAllInputsWidth, Integer rawAllInputsHeight, const Real* rawExcpectedOutputs, Integer rawExcpectedOutputsWidth, Integer rawExcpectedOutputsHeight, bool isClassification = true, Real alpha = 0.01f, Integer maxIter = 1000);

	ML_API bool linearIsValid(TypeId id);
    ML_API TypeId linearCreate(bool isClassification,Real step,Integer entrySize);
    ML_API void linearTrain(TypeId id,Integer count,const Real* entries, const Real* output, Integer entryCount);

    ML_API Real linearEvaluate(TypeId id,const Real* entries);
    ML_API void linearDelete(TypeId id);

}