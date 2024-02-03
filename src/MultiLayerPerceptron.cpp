//
// Created by Sayama on 18/12/2023.
//

#include "MultiLayerPerceptron.hpp"
#include "UUID.hpp"
#include <limits>
#include <cmath>
#include <iostream>
#include <fstream>

namespace GG::ML {

	MultiLayerPerceptron::MultiLayerPerceptron(const Integer *layerCounts,Integer layerCount)
	: m_D(layerCounts, layerCounts + layerCount), m_L(layerCount - 1)
	{
		initialize();
	}

	MultiLayerPerceptron::~MultiLayerPerceptron() = default;

	void MultiLayerPerceptron::initialize(bool initWeight)
	{
		if(initWeight) {
			m_W.reserve(m_D.size());
			for (Integer l = 0; l < m_D.size(); ++l) {
				m_W.emplace_back();
				if (l == 0) {
					continue;
				}
				m_W[l].reserve(m_D[l - 1] + 1);
				for (Integer i = 0; i < m_D[l - 1] + 1; ++i) {
					m_W[l].emplace_back();
					m_W[l][i].reserve(m_D[l] + 1);
					for (Integer j = 0; j < m_D[l] + 1; ++j) {
						if (j == 0) {
							m_W[l][i].push_back(0.0);
						} else {
							m_W[l][i].push_back((ML_RAND * 2.0) - 1.0);
						}
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
                //std::cout<<l<<" Inited "<<j<<std::endl;
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

	Integer MultiLayerPerceptron::Predict(const Real* rawInputs, Integer rawInputsCount, bool isClassification)
	{
		std::vector<Real> inputs(rawInputs, rawInputs + rawInputsCount);
		return Predict(inputs, isClassification);
	}

	Real MultiLayerPerceptron::GetPredictData(Integer index)
	{
		auto& arr = m_X[m_L];
        //std::cout<<index<<"index,m_L"<<m_L<<" layerSize : "<<arr.size()<<" value : "<<arr[index]<<std::endl;
		return arr[index];
	}

	void MultiLayerPerceptron::Propagate(const std::vector<Real>& inputs, bool isClassification)
	{
		for (Integer j = 1; j < m_D[0] + 1; ++j)
		{
			m_X[0][j] = inputs[j - 1];
		}

		for (Integer l = 1; l < m_L+1; ++l)
		{
			for (Integer j = 1; j < m_D[l] + 1; ++j)
			{
				Real sum = 0;
                for (Integer i = 0; i < m_D[l - 1] + 1; ++i)
				{
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
        //std::cout<<static_cast<Integer>(arr.size()) - 1<<" is size of "<<m_L<< "layer" <<std::endl;
		return static_cast<Integer>(arr.size()) - 1;
	}

	void MultiLayerPerceptron::Train(const Real *rawAllInputs, Integer inputSize, Integer inputsCount,
                                     const Real *rawExpectedOutputs, Integer outputSize, Integer outputsCount,
                                     bool isClassification, Real alpha, Integer maxIter)
	{
		//Vector2D<Real> allInputs(rawAllInputs, inputSize, inputsCount);
		//Vector2D<Real> expectedOutputs(rawExpectedOutputs, outputSize, outputsCount);

		for (Integer iter = 0; iter < maxIter; ++iter)
		{
			Integer k = static_cast<Integer>(std::floor(ML_RAND * static_cast<Real>(inputsCount)));
			std::vector<Real> inputsK(rawAllInputs+ k * inputSize, rawAllInputs + (k + 1) * inputSize);
            std::vector<Real> outputsK(rawExpectedOutputs+ k * outputSize, rawExpectedOutputs + (k + 1) * outputSize);
            //std::cout<<" Input "<<k<<" selected, values : ";
            for (int i = 0; i < inputSize; ++i) {
                //std::cout<<inputsK[i]<<" ";
            }
            //std::cout<<std::endl<<"Exepected output value(s) : ";
            for (int i = 0; i < outputSize; ++i) {
                //std::cout<<outputsK[i]<<" ";
            }
            //std::cout<<std::endl;
			Propagate(inputsK, isClassification);
			for (Integer j = 1; j < m_D[m_L] + 1; ++j)
			{
				m_Deltas[m_L][j] = m_X[m_L][j] - outputsK[j - 1];
				if(isClassification)
				{
					m_Deltas[m_L][j] *= (1 - (m_X[m_L][j] * m_X[m_L][j]));
                    //std::cout<<m_L<<"/"<<m_D[m_L] + 1<< " l,j "<<j<<"/"<<m_D[m_L] + 1;
                    //std::cout<<";;"<<"delta : "<< m_Deltas[m_L][j]<<std::endl;

                }
			}

			for (Integer l = m_L; l >= 2; --l)
			{
				for (int i = 1; i < m_D[l - 1] + 1; ++i)
				{
					Real sum = 0.0;
                    for (int j = 1; j < m_D[l] + 1; ++j)
					{
                        //std::cout<<j<<" weight ; "<<m_W[l][i][j];
						sum += m_W[l][i][j] * m_Deltas[l][j];
					}
                    //std::cout<<";;"<<" signal "<< m_X[l - 1][i];
                    m_Deltas[l - 1][i] = (1 - (m_X[l - 1][i] * m_X[l - 1][i])) * sum;
                    //std::cout<<";;"<<sum<<" sum, delta"<< m_Deltas[l - 1][i]<<std::endl;
                }
			}
            //Helper cout loop
            /*for (int l = 0; l < m_L+1; ++l){
                for(int j=0;j<m_D[l]+1;++j){
                    if(l<m_L || isClassification)
                        std::cout<<std::tanh(m_X[l][j]);
                    else
                        std::cout<<m_X[l][j];
                    std::cout<<std::endl;
                    std::cout<<" Value of x : " << m_X[l][j] << " ("<<l<<","<< j<<") / ("<<m_L+1-1 << " / "<<m_D[l] + 1-1<<")"<<std::endl;
                    if(l>0 && j>0){
                        for (int i = 0; i < m_D[l-1] + 1; ++i) {
                            std::cout << "Weight : "<<m_W[l][i][j] << "(" << l << "," << i << "," << j << "),";
                        }
                    }
                }
                std::cout<<std::endl;
            }*/


            for (int l = 1; l < m_L + 1; ++l)
			{
				for (int i = 0; i < m_D[l - 1] + 1; ++i)
				{
					for (int j = 1; j < m_D[l] + 1; ++j)
					{
                        const Real x = m_X[l - 1][i];
                        m_W[l][i][j] -= alpha * (x * m_Deltas[l][j]);
                        /*std::cout << l << "/" <<m_L+1-1 << " l,i " << i << "/" <<m_D[l - 1] + 1-1 << " j :" << j << "/" <<m_D[l] + 1-1<< std::endl;
                        std::cout<< alpha << " : alpha " << m_Deltas[l][j] << ": Delta " << x
                        << " : value " <<
                        (alpha * (x * m_Deltas[l][j])) << ": weightChange"
                        << "Final weight : " << m_W[l][i][j]
                        << std::endl;*/
                    }
				}
			}
		}

	}

	MultiLayerPerceptron::MultiLayerPerceptron(const std::filesystem::path &path)
	{
		if(!is_regular_file(path))
		{
			std::cerr << "The file '" << path <<"' is not a regular file." << std::endl;
			return;
		}
		std::ifstream saveFile(path, std::ios::binary);

		//Read Layers
		Integer numberOfLayer;
		saveFile.read(reinterpret_cast<char *>(&numberOfLayer), sizeof(Integer));
		m_D.resize(numberOfLayer);
		if(numberOfLayer > 0) {
			for (int i = 0; i < numberOfLayer; ++i) {
				saveFile.read(reinterpret_cast<char *>(&m_D[i]), sizeof(Integer));
			}
		}
		m_L = m_D.size() - 1;

		// Read Weights
		Integer numberOfWeight;
		saveFile.read(reinterpret_cast<char *>(&numberOfWeight), sizeof(Integer));
		m_W.resize(numberOfWeight);
		for (int i = 0; i < numberOfWeight; ++i) {
			Integer numberOfSubArray;
			saveFile.read(reinterpret_cast<char *>(&numberOfSubArray), sizeof(Integer));
			m_W[i].resize(numberOfSubArray);
			for (int j = 0; j < numberOfSubArray; ++j) {
				Integer numberOfWeightInSubArray;
				saveFile.read(reinterpret_cast<char *>(&numberOfWeightInSubArray), sizeof(Integer));
				m_W[i][j].resize(numberOfWeightInSubArray);
				for (int k = 0; k < numberOfWeightInSubArray; ++k) {
					saveFile.read(reinterpret_cast<char*>(&m_W[i][j][k]), sizeof(Real));
				}
			}
		}

		initialize(false);
	}

	bool MultiLayerPerceptron::Save(const std::filesystem::path &path)
	{
		std::ofstream saveFile(path, std::ios::trunc | std::ios::binary | std::ios::app);

		//Write Layers
		Integer numberOfLayer = m_D.size();
		saveFile.write(reinterpret_cast<const char *>(&numberOfLayer), sizeof(Integer));
		if(numberOfLayer > 0) {
			saveFile.write(reinterpret_cast<const char *>(m_D.data()), sizeof(Integer) * numberOfLayer);
		}

		// Write Weight
		Integer numberOfWeight = m_W.size();
		saveFile.write(reinterpret_cast<const char *>(&numberOfWeight), sizeof(Integer));
		for (int i = 0; i < numberOfWeight; ++i) {
			Integer numberOfSubArray = m_W[i].size();
			saveFile.write(reinterpret_cast<const char *>(&numberOfSubArray), sizeof(Integer));
			for (int j = 0; j < numberOfSubArray; ++j) {
				Integer numberOfWeightInSubArray = m_W[i][j].size();
				saveFile.write(reinterpret_cast<const char *>(&numberOfWeightInSubArray), sizeof(Integer));
				if(numberOfWeightInSubArray > 0)
				{
					saveFile.write(reinterpret_cast<const char *>(m_W[i][j].data()), sizeof(Real) * numberOfWeightInSubArray);
				}
			}
		}
		saveFile.flush();
		saveFile.close();
		return true;
	}

	Integer MultiLayerPerceptron::GetLayersCount() {
		return m_D.size();
	}

	Integer MultiLayerPerceptron::GetLayer(Integer i) {
		return m_D[i];
	}

	Integer MultiLayerPerceptron::GetWeightCount() {
		return m_W.size();
	}

	Integer MultiLayerPerceptron::GetSubWeightCount(Integer i) {
		return m_W[i].size();
	}

	Integer MultiLayerPerceptron::GetSubSubWeightCount(Integer i, Integer i1) {
		return m_W[i][i1].size();
	}

	Real MultiLayerPerceptron::GetWeight(Integer i, Integer i1, Integer i2) {
		return m_W[i][i1][i2];
	}

} // GG
// ML