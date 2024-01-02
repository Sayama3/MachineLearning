//
// Created by ianpo on 02/01/2024.
//

#pragma once

#include "library.h"
#include <array>
#include <vector>
#include <stdexcept>
#include <string>

namespace GG::ML {

	template<typename T, Integer DimensionCount>
	class Vector
	{
	public:
		Vector(const std::array<Integer, DimensionCount>& dimensionsSizes);
		Vector(const std::array<Integer, DimensionCount>& dimensionsSizes, const T& value);
		Vector(const std::vector<Integer>& dimensionsSizes);
		Vector(const std::vector<Integer>& dimensionsSizes, const T& value);
		~Vector();

		T* operator[](Integer index);
		const T* operator[](Integer index) const;

		T& operator()(Integer index);
		const T& operator()(Integer index) const;

		T& operator()(std::array<Integer, DimensionCount> indices);
		const T& operator()(std::array<Integer, DimensionCount> indices) const;

		Integer size() const;
		Integer size(Integer dimension) const;

		void resize(const std::array<Integer, DimensionCount>& dimensionsSizes);
		void resize(const std::array<Integer, DimensionCount>& dimensionsSizes, const T& value);
	private:
		std::array<Integer, DimensionCount> m_Dimensions;
		std::vector<T> m_Data;
	};

	template<typename T, Integer DimensionCount>
	Vector<T, DimensionCount>::Vector(const std::vector<Integer> &dimensionsSizes, const T &value)
	{
	}

	template<typename T, Integer DimensionCount>
	Vector<T, DimensionCount>::Vector(const std::vector<Integer> &dimensionsSizes)
	{
		if(dimensionsSizes.size() != DimensionCount)
		{
			throw std::range_error(std::string("The dimension of the vector is different from the expected one."));
		}
	}

	template<typename T, Integer DimensionCount>
	Integer Vector<T, DimensionCount>::size(Integer dimension) const {
		return m_Dimensions[dimension];
	}

	template<typename T, Integer DimensionCount>
	Integer Vector<T, DimensionCount>::size() const {
		return m_Data.size();
	}

	template<typename T, Integer DimensionCount>
	T *Vector<T, DimensionCount>::operator[](Integer index) {
		Integer size = 1;
		for (Integer i = DimensionCount - 1; i >= 1; --i)
		{
			size *= m_Dimensions[i];
		}
		return &m_Data[index * size];
	}

	template<typename T, Integer DimensionCount>
	const T *Vector<T, DimensionCount>::operator[](Integer index) const {
		Integer size = 1;
		for (Integer i = DimensionCount - 1; i >= 1; --i)
		{
			size *= m_Dimensions[i];
		}
		return &m_Data[index * size];
	}

	template<typename T, Integer DimensionCount>
	T &Vector<T, DimensionCount>::operator()(Integer index) {
		return m_Data[index];
	}

	template<typename T, Integer DimensionCount>
	const T &Vector<T, DimensionCount>::operator()(Integer index) const {
		return m_Data[index];
	}

	template<typename T, Integer DimensionCount>
	const T &Vector<T, DimensionCount>::operator()(std::array<Integer, DimensionCount> indices) const
	{
		Integer index = 0;
		Integer size = 1;
		for (Integer i = DimensionCount - 1; i >= 0; --i)
		{
			index += indices[i] * size;
			size *= m_Dimensions[i];
		}
		//TODO: Assert index <= m_Data.size();
		return m_Data[index];
	}

	template<typename T, Integer DimensionCount>
	T &Vector<T, DimensionCount>::operator()(std::array<Integer, DimensionCount> indices)
	{
		Integer index = 0;
		Integer size = 1;
		for (Integer i = DimensionCount - 1; i >= 0; --i)
		{
			index += indices[i] * size;
			size *= m_Dimensions[i];
		}
		//TODO: Assert index <= m_Data.size();
		return m_Data[index];
	}

	template<typename T, Integer DimensionCount>
	Vector<T, DimensionCount>::~Vector() {}

	template<typename T, Integer DimensionCount>
	Vector<T, DimensionCount>::Vector(const std::array<Integer, DimensionCount>& dimensionsSizes) : m_Dimensions(dimensionsSizes)
	{
		resize(m_Dimensions);
	}

	template<typename T, Integer DimensionCount>
	Vector<T, DimensionCount>::Vector(const std::array<Integer, DimensionCount>& dimensionsSizes, const T& value) : m_Dimensions(dimensionsSizes)
	{
		resize(m_Dimensions, value);
	}

	template<typename T, Integer DimensionCount>
	void Vector<T, DimensionCount>::resize(const std::array<Integer, DimensionCount>& dimensionsSizes)
	{
		Integer size = 1;
		for(auto& dimension : m_Dimensions)
		{
			size *= dimension;
		}
		m_Data.resize(size);
	}

	template<typename T, Integer DimensionCount>
	void Vector<T, DimensionCount>::resize(const std::array<Integer, DimensionCount>& dimensionsSizes, const T &value)
	{
		Integer size = 1;
		for(auto& dimension : m_Dimensions)
		{
			size *= dimension;
		}
		m_Data.resize(size, value);
	}
} // ML
// GG
