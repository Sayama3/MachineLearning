//
// Created by flore on 10/01/2024.
//

#ifndef MACHINELEARNING_LINEARMODEL_H
#define MACHINELEARNING_LINEARMODEL_H
#include <vector>
#include "library.h"
#include "UUID.hpp"
#include <limits>
#include <cmath>
#include <iostream>
#include <filesystem>

class LinearModel {
private:
    Real step;
    Integer size;
    std::vector<Real> weight;
    bool isClassification;
public:
    inline LinearModel(bool isClassification,Real step,Integer entrySize) : size(entrySize),step(step),isClassification(isClassification){
        for (int i = 0; i < entrySize+1; ++i) {
            auto max=static_cast<Real>(std::numeric_limits<uint64_t>().max());
            weight.push_back(ML_RAND);
        }
    }
	LinearModel(const std::filesystem::path& fullPath);
    void train(int times,const Real* entries, const Real* output, Integer entryCount);
    inline Real predict(const Real* x) {
        std::vector<Real> v(x,x+size);
        return predict(v);
    }
    Real predict(const std::vector<Real> x);

	bool save(const std::filesystem::path& fullPath);

	Real getStep() const;

	Integer getSize() const;

	Integer IsClassification() const;

	Integer weightCount() const;

	Real getWeight(Integer i) const;

private:
    void trainOnceRosenblatt(std::vector<Real> input,Real expected);

};


#endif //MACHINELEARNING_LINEARMODEL_H
