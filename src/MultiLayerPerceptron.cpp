//
// Created by Sayama on 18/12/2023.
//

#include "MultiLayerPerceptron.hpp"
#include "UUID.hpp"
#include <limits>
#include <cmath>


namespace GG::ML {

	MultiLayerPerceptron::MultiLayerPerceptron(const Integer *layerCounts,Integer layerCount)
	: m_Ds(layerCounts, layerCounts + layerCount), m_Last(layerCount - 1)
	{
		initialize();
	}

	MultiLayerPerceptron::~MultiLayerPerceptron() = default;

	void MultiLayerPerceptron::initialize()
	{
		m_Weights.reserve(m_Ds.size());
		for (Integer l = 0; l < m_Ds.size(); ++l)
		{
			m_Weights.emplace_back();
			if(l == 0) continue;
			m_Weights[l].reserve(m_Ds[l - 1] + 1);
			for (Integer i = 0; i < m_Ds[l - 1] + 1; ++i)
			{
				m_Weights[l].emplace_back();
				m_Weights[l][i].reserve(m_Ds[l] + 1);
				for (Integer j = 0; j < m_Ds[l] + 1; ++j)
				{
					if(j == 0)
					{
						m_Weights[l][i].push_back(0.0);
					}
					else
					{
						m_Weights[l][i].push_back((ML_RAND * 2.0) - 1.0);
					}
				}
			}
		}

		m_Xs.reserve(m_Ds.size());
		m_Deltas.reserve(m_Ds.size());
		for (Integer l = 0; l < m_Ds.size(); ++l)
		{
			m_Xs.emplace_back();
			m_Deltas.emplace_back();

			m_Xs[l].reserve(m_Ds[l] + 1);
			m_Deltas[l].reserve(m_Ds[l] + 1);

			for (Integer j = 0; j < m_Ds[l] + 1; ++j)
			{
				m_Deltas[l].push_back(0.0);
				if(j == 0)
				{
					m_Xs[l].push_back(1.0);
				}
				else
				{
					m_Xs[l].push_back(0.0);
				}
			}
		}
	}

	void MultiLayerPerceptron::Propagate(const Real* rawInputs, Integer rawInputsCount, bool isClassification)
	{
		std::vector<Real> inputs(rawInputs, rawInputs + rawInputsCount);
		Propagate(inputs, isClassification);
	}

	Real MultiLayerPerceptron::Predict(const Real* rawInputs, Integer rawInputsCount, bool isClassification)
	{
		std::vector<Real> inputs(rawInputs, rawInputs + rawInputsCount);
		return Predict(inputs, isClassification);
	}

	void MultiLayerPerceptron::Propagate(const std::vector<Real>& inputs, bool isClassification)
	{
		for (Integer j = 0; j < m_Ds[0] + 1; ++j)
		{
			m_Xs[0][j] = inputs[j - 1];
		}

		for (Integer l = 1; l < m_Ds.size(); ++l)
		{
			for (Integer j = 1; j < m_Ds[l] + 1; ++j)
			{
				Real sum = 0;
				for (Integer i = 0; i < m_Ds[l - 1] + 1; ++i)
				{
					sum += m_Weights[l][i][j] * m_Xs[l - 1][i];
				}

				if( l < m_Last || isClassification)
				{
					sum = std::tanh(sum);
				}
				m_Xs[l][j] = sum;
			}
		}
	}
	Real MultiLayerPerceptron::Predict(const std::vector<Real>& inputs, bool isClassification)
	{
		Propagate(inputs, isClassification);
		auto& arr = m_Xs[m_Last];
		return arr[arr.size() - 1];
	}

	void MultiLayerPerceptron::Train(const Real *rawAllInputs, Integer rawAllInputsWidth, Integer rawAllInputsHeight,
									 const Real *rawExpectedOutputs, Integer rawExpectedOutputsWidth, Integer rawExpectedOutputsHeight,
									 bool isClassification, float alpha, Integer maxIter)
	{
		Vector2D<Real> allInputs(rawAllInputs, rawAllInputsWidth, rawAllInputsHeight);
		Vector2D<Real> expectedOutputs(rawExpectedOutputs, rawExpectedOutputsWidth, rawExpectedOutputsHeight);

		for (Integer iter = 0; iter < maxIter; ++iter)
		{
			Integer k = static_cast<Integer>(std::floor(ML_RAND * static_cast<Real>(rawAllInputsWidth)));
			Real* inputsK = allInputs[k];
			Real* outputsK = expectedOutputs[k];
			Propagate(inputsK, rawAllInputsHeight, isClassification);
			for (Integer j = 1; j < m_Ds[m_Last] + 1; ++j)
			{
				m_Deltas[m_Last][j] = m_Xs[m_Last][j] - outputsK[j - 1];
				if(isClassification)
				{
					m_Deltas[m_Last][j] *= (1 - (m_Xs[m_Last][j] * m_Xs[m_Last][j]));
				}
			}

			for (Integer l = m_Last + 1 - 1; l >= 2; --l)
			{
				for (int i = 0; i < m_Ds[l - 1] + 1; ++i)
				{
					Real sum = 0.0;
					for (int j = 1; j < m_Ds[l] + 1; ++j)
					{
						sum += m_Weights[l][i][j] * m_Deltas[l][j];
					}
					m_Deltas[l - 1][i] = (1 - (m_Xs[l - 1][i] * m_Xs[l - 1][i])) * sum;
				}
			}

			for (int l = 1; l < m_Last + 1; ++l)
			{
				for (int i = 0; i < m_Ds[l - 1] + 1; ++i)
				{
					for (int j = 0; j < m_Ds[l] + 1; ++j)
					{
						m_Weights[l][i][j] -= alpha * (m_Xs[l - 1][i] * m_Deltas[l][j]);
					}
				}
			}
		}

	}

} // GG
// ML