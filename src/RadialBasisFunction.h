//
// Created by flore on 20/01/2024.
//

#ifndef MACHINELEARNING_RADIALBASISFUNCTION_H
#define MACHINELEARNING_RADIALBASISFUNCTION_H
#include <vector>
#include "library.h"
#include <cmath>
#include <numeric>
#include <iostream>
#include "UUID.hpp"
#include <limits>
#include <filesystem>
#include <Eigen/Dense>

class RadialBasisFunction {
public:
    inline RadialBasisFunction(Real gamma) : gamma(gamma) {}
	RadialBasisFunction(const std::filesystem::path& fullPath);
    Real predict(bool isClassification,const Eigen::Matrix<Real, Eigen::Dynamic,1>& inputs);
    void train(Integer nbOfRepresentants,const std::vector<std::vector<Real>>& inputs,const Eigen::Matrix<Real, Eigen::Dynamic,1>& outputs,Integer maxKMeanIter);
	bool save(const std::filesystem::path& fullPath);

	Real getGamma() const;

	Integer getRows() const;

	Integer getCols() const;

	Real getWeight(Integer row, Integer col) const;

	Integer getSize() const;

	Real getWeight(Integer index) const;

	void setWeight(Integer row, Integer col, Real d);
	void setWeight(Integer index, Real d);

private:
    Real gamma;
    std::vector<std::vector<Real>> representants;
    Eigen::Matrix<Real,1,Eigen::Dynamic> m_W;
    void updateRepresentants(const std::vector<std::vector<Real>>& inputs,Integer nb,Integer maxIter);

    int closest(const std::vector<Real>& element);

    Real dist(const std::vector<Real>& v1, const std::vector<Real>& v2);

    Real dist(const std::vector<Real> &v1, const Eigen::Matrix<Real, Eigen::Dynamic,1>& v2);
    inline Real Exp(Real dist){
        return std::exp(-1.0*gamma*std::pow(dist,2));
    }
};


#endif //MACHINELEARNING_RADIALBASISFUNCTION_H
