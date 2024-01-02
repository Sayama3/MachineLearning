//
// Created by Sayama on 18/12/2023.
//

#pragma once

#include "library.h"
#include "Vector2D.hpp"
#include "Vector.hpp"
#include <vector>

namespace GG::ML {

	class MultiLayerPerceptron
	{
	private:
		std::vector<Integer> m_Ds; // d
		std::vector<std::vector<std::vector<Real>>> m_Weights;
		std::vector<std::vector<Real>> m_Xs;
		std::vector<std::vector<Real>> m_Deltas;
		Integer m_Last;

	public:
		MultiLayerPerceptron(const Integer* layerCounts, Integer layerCount);
		~MultiLayerPerceptron();

		void Propagate(const Real* rawInputs, Integer rawInputsCount, bool isClassification);
		Real Predict(const Real* rawInputs, Integer rawInputsCount, bool isClassification = true);
		void Train(const Real* rawAllInputs, Integer rawAllInputsWidth, Integer rawAllInputsHeight, const Real* rawExcpectedOutputs, Integer rawExcpectedOutputsWidth, Integer rawExcpectedOutputsHeight, bool isClassification = true, float alpha = 0.01f, Integer maxIter = 1000);

	private:
		void Propagate(const std::vector<Real>& inputs, bool isClassification);
		Real Predict(const std::vector<Real>& inputs, bool isClassification);
		void initialize();
	};

} // GG
// ML
