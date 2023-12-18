//
// Created by Sayama on 18/12/2023.
//

#pragma once

#include "library.h"
#include <vector>

namespace GG::ML {

	class MultiLayerPerceptron
	{
	private:
		std::vector<Real> m_Data;
		std::vector<Integer> m_LayerCounts;


	public:
		MultiLayerPerceptron(const Real* data, Integer dataCount, const Integer* layerCounts, Integer layerCount);
		MultiLayerPerceptron(const Integer* layerCounts, Integer layerCount);
		~MultiLayerPerceptron();

		void setData(const Real* data, Integer dataCount);

	private:
		void initialize();
	};

} // GG
// ML
