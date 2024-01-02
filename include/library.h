#pragma once
#include <machinelearning_export.hpp>

extern "C"
{
	typedef long long int Integer;
	typedef double Real;

    ML_API void infos();

	// Library functions
    ML_API void initialize();
    ML_API void update(Real timestep);
    ML_API void shutdown();

	// MultiLayerPerceptron functions
	ML_API void mlpCreate(const Integer* entries, Integer count);
	ML_API void mlpRun(const Real* data, Integer count);
}