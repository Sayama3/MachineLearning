//
// Created by Sayama on 18/12/2023.
//

#pragma once

#include "library.h"
#include "Vector2D.hpp"
#include <vector>

namespace GG::ML {

	class MultiLayerPerceptron
	{
	private:
		std::vector<Real> m_Data;
		std::vector<Integer> m_LayerCounts;
	public:
		MultiLayerPerceptron(const Integer* layerCounts, Integer layerCount);
		~MultiLayerPerceptron();

		void setData(const Real* data, Integer dataCount);

		void Propagate(const std::vector<float>& inputs, bool isClassification);
		void Predict(const std::vector<float>& inputs, bool isClassification = true);
		void Train(const Vector2D<float>& allInputs, const Vector2D<float>& allExpectedOutput, bool isClassification = true, float alpha = 0.01f, int maxIter = 1000);

	private:
		void initialize();
	};

} // GG
// ML
