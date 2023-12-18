//
// Created by Sayama on 18/12/2023.
//

#include "MultiLayerPerceptron.hpp"
#include <algorithm>

namespace GG::ML {
	MultiLayerPerceptron::MultiLayerPerceptron(const Real *data, Integer dataCount, const Integer *layerCounts,Integer layerCount)
	: m_Data(data, data + dataCount), m_LayerCounts(layerCounts, layerCounts + layerCount)
	{
	}

	MultiLayerPerceptron::MultiLayerPerceptron(const Integer *layerCounts,Integer layerCount)
	: m_Data(), m_LayerCounts(layerCounts, layerCounts + layerCount)
	{
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
} // GG
// ML