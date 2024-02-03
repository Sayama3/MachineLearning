//
// Created by Sayama on 18/12/2023.
//

#pragma once

#include "library.h"
#include "Vector2D.hpp"
#include "Vector.hpp"
#include <vector>
#include <filesystem>

namespace GG::ML {

	class MultiLayerPerceptron
	{
	private:
		std::vector<Integer> m_D; // d
		std::vector<std::vector<std::vector<Real>>> m_W;
		std::vector<std::vector<Real>> m_X;
		std::vector<std::vector<Real>> m_Deltas;
		Integer m_L;

	public:
		MultiLayerPerceptron(const Integer* layerCounts, Integer layerCount);
		MultiLayerPerceptron(const std::filesystem::path& path);
		~MultiLayerPerceptron();

		Integer Predict(const Real* rawInputs, Integer rawInputsCount, bool isClassification = true);
		Real GetPredictData(Integer index);
		void Train(const Real* rawAllInputs, Integer inputSize, Integer inputsCount, const Real* rawExcpectedOutputs, Integer outputSize, Integer outputsCount, bool isClassification = true, Real alpha = 0.01f, Integer maxIter = 1000);

		bool Save(const std::filesystem::path& path);

		Integer GetLayersCount();

		Integer GetLayer(Integer i);

		Integer GetWeightCount();

		Integer GetSubWeightCount(Integer i);

		Integer GetSubSubWeightCount(Integer i, Integer i1);

		Real GetWeight(Integer i, Integer i1, Integer i2);

	private:
		void Propagate(const std::vector<Real>& inputs, bool isClassification);
		Integer Predict(const std::vector<Real>& inputs, bool isClassification);
		void initialize(bool initWeight = true);
	};

} // GG
// ML
