//
// Created by Sayama on 18/12/2023.
//

#include "MultiLayerPerceptron.hpp"
#include "UUID.hpp"
#include <limits>
#include <cmath>
#include <iostream>


namespace GG::ML {

	MultiLayerPerceptron::MultiLayerPerceptron(const Integer *layerCounts,Integer layerCount)
	: m_D(layerCounts, layerCounts + layerCount), m_L(layerCount - 1)
	{
		initialize();
	}

	MultiLayerPerceptron::~MultiLayerPerceptron() = default;

	void MultiLayerPerceptron::initialize()
	{
		m_W.reserve(m_D.size());
		for (Integer l = 0; l < m_D.size(); ++l)
		{
			m_W.emplace_back();
			if(l == 0) {
				continue;
			}
			m_W[l].reserve(m_D[l - 1] + 1);
			for (Integer i = 0; i < m_D[l - 1] + 1; ++i)
			{
				m_W[l].emplace_back();
				m_W[l][i].reserve(m_D[l] + 1);
				for (Integer j = 0; j < m_D[l] + 1; ++j)
				{
					if(j == 0)
					{
						m_W[l][i].push_back(0.0);
					}
					else
					{
						m_W[l][i].push_back((ML_RAND * 2.0) - 1.0);
					}
				}
			}
		}

		m_X.reserve(m_D.size());
		m_Deltas.reserve(m_D.size());
		for (Integer l = 0; l < m_D.size(); ++l)
		{
			m_X.emplace_back();
			m_Deltas.emplace_back();

			m_X[l].reserve(m_D[l] + 1);
			m_Deltas[l].reserve(m_D[l] + 1);

			for (Integer j = 0; j < m_D[l] + 1; ++j)
			{
				m_Deltas[l].push_back(0.0);
				if(j == 0)
				{
					m_X[l].push_back(1.0);
				}
				else
				{
					m_X[l].push_back(0.0);
				}
			}
		}
	}

	void MultiLayerPerceptron::Propagate(const Real* rawInputs, Integer rawInputsCount, bool isClassification)
	{
		std::vector<Real> inputs(rawInputs, rawInputs + rawInputsCount);
		Propagate(inputs, isClassification);
	}

	Integer MultiLayerPerceptron::Predict(const Real* rawInputs, Integer rawInputsCount, bool isClassification)
	{
		std::vector<Real> inputs(rawInputs, rawInputs + rawInputsCount);
		return Predict(inputs, isClassification);
	}

	Real MultiLayerPerceptron::GetPredictData(Integer index)
	{
		auto& arr = m_X[m_L];
		return arr[index + 1];
	}

	void MultiLayerPerceptron::Propagate(const std::vector<Real>& inputs, bool isClassification)
	{
		for (Integer j = 1; j < m_D[0] + 1; ++j)
		{
			m_X[0][j] = inputs[j - 1];
		}

		for (Integer l = 1; l < m_D.size(); ++l)
		{
			for (Integer j = 1; j < m_D[l] + 1; ++j)
			{
				Real sum = 0;
                for (Integer i = 0; i < m_D[l - 1] + 1; ++i)
				{
                    if(i==0 && std::abs( m_X[l - 1][i]) > std::pow(10, -200)){
                        //std::cout << "Bias value > 0 " << m_X[l - 1][i];
                    }
                    sum += m_W[l][i][j] * m_X[l - 1][i];
				}
				if(l < m_L || isClassification)
				{
					sum = std::tanh(sum);
				}
				m_X[l][j] = sum;
			}
		}
	}
	Integer MultiLayerPerceptron::Predict(const std::vector<Real>& inputs, bool isClassification)
	{
		Propagate(inputs, isClassification);

		auto& arr = m_X[m_L];
		return static_cast<Integer>(arr.size()) - 1;
	}

	void MultiLayerPerceptron::Train(const Real *rawAllInputs, Integer rawAllInputsWidth, Integer rawAllInputsHeight,
									 const Real *rawExpectedOutputs, Integer rawExpectedOutputsWidth, Integer rawExpectedOutputsHeight,
									 bool isClassification, Real alpha, Integer maxIter)
	{
		Vector2D<Real> allInputs(rawAllInputs, rawAllInputsWidth, rawAllInputsHeight);
		Vector2D<Real> expectedOutputs(rawExpectedOutputs, rawExpectedOutputsWidth, rawExpectedOutputsHeight);

		for (Integer iter = 0; iter < maxIter; ++iter)
		{
            std::cout<<std::endl<<m_L<<std::endl;
			Integer k = static_cast<Integer>(std::floor(ML_RAND * static_cast<Real>(rawAllInputsWidth)));
			Real* inputsK = allInputs[k];
			Real* outputsK = expectedOutputs[k];
			Propagate(inputsK, rawAllInputsHeight, isClassification);
			for (Integer j = 1; j < m_D[m_L] + 1; ++j)
			{
				m_Deltas[m_L][j] = m_X[m_L][j] - outputsK[j - 1];
				if(isClassification)
				{
					m_Deltas[m_L][j] *= (1 - (m_X[m_L][j] * m_X[m_L][j]));
                    std::cout<<m_L<<"/"<<m_D[m_L] + 1<< " l,j "<<j<<"/"<<m_D[m_L] + 1;
                    std::cout<<";;"<<"delta : "<< m_Deltas[m_L][j]<<std::endl;

                }
			}

			for (Integer l = m_L; l >= 2; --l)
			{
				for (int i = 1; i < m_D[l - 1] + 1; ++i)
				{
					Real sum = 0.0;
                    for (int j = 1; j < m_D[l] + 1; ++j)
					{
                        std::cout<<j<<" weight ; "<<m_W[l][i][j];
						sum += m_W[l][i][j] * m_Deltas[l][j];
					}
                    std::cout<<";;"<<" signal "<< m_X[l - 1][i];
                    m_Deltas[l - 1][i] = (1 - (m_X[l - 1][i] * m_X[l - 1][i])) * sum;
                    std::cout<<";;"<<sum<<" sum, delta"<< m_Deltas[l - 1][i]<<std::endl;
                }
			}
            for (int l = 0; l < m_L+1; ++l)
                for (int i = 0; i < m_D[l] + 1; ++i)
                    std::cout<<"Value of x : " << m_X[l][i] << "with : "<<l<<" / "<<m_L+1-1<<","<< i << " / "<<m_D[l] + 1-1<<";;;";
            std::cout<<std::endl;
            for (int l = 1; l < m_L + 1; ++l)
			{
				for (int i = 0; i < m_D[l - 1] + 1; ++i)
				{
					for (int j = 1; j < m_D[l] + 1; ++j)
					{
                        const Real x = m_X[l - 1][i];
                        m_W[l][i][j] -= alpha * (x * m_Deltas[l][j]);
                        std::cout << l << "/" <<m_L+1-1 << " l,i " << i << "/" <<m_D[l - 1] + 1-1 << " j :" << j << "/" <<m_D[l] + 1-1<< std::endl;
                        if(i==0 && std::abs(x) > std::pow(10, -200)){
                            std::cout << "Bias value > 0 " << x;
                        }
                        std::cout<< alpha << " : alpha " << m_Deltas[l][j] << ": Delta " << x
                        << " : value " <<
                        (alpha * (x * m_Deltas[l][j])) << ": weightChange"
                        << "Final weight : " << m_W[l][i][j]
                        << std::endl;
                    }
				}
			}
		}

	}

} // GG
// ML