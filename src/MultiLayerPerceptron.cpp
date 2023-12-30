//
// Created by Sayama on 18/12/2023.
//

#include "MultiLayerPerceptron.hpp"
#include <algorithm>

namespace GG::ML {

	MultiLayerPerceptron::MultiLayerPerceptron(const Integer *layerCounts,Integer layerCount)
	: m_Data(), m_LayerCounts(layerCounts, layerCounts + layerCount)
	{
		initialize();
	}

	MultiLayerPerceptron::~MultiLayerPerceptron()
	{
	}

	void MultiLayerPerceptron::setData(const Real *data, Integer dataCount)
	{
		m_Data.assign(data, data + dataCount);
	}

	void MultiLayerPerceptron::initialize()
	{

	}

	void MultiLayerPerceptron::Propagate(const std::vector<float>& inputs, bool isClassification)
	{

	}

	void MultiLayerPerceptron::Predict(const std::vector<float>& inputs, bool isClassification)
	{

	}

	void MultiLayerPerceptron::Train(const Vector2D<float>& allInputs, const Vector2D<float>& allExpectedOutput, bool isClassification, float alpha, int maxIter)
	{
	}

} // GG
// ML