//
// Created by ianpo on 30/12/2023.
//

#pragma once

#include <vector>

namespace GG::ML {

	template<typename T>
	class Vector2D {
	public:
		Vector2D(Integer width, Integer height);
		Vector2D(Integer width, Integer height, const T& value);
		Vector2D(const T* array, Integer width, Integer height);

		Vector2D() = default;
		~Vector2D() = default;

		T* operator[](Integer index);
		const T* operator[](Integer index) const;

		T& operator()(Integer x);
		const T& operator()(Integer x) const;

		T& operator()(Integer x, Integer y);
		const T& operator()(Integer x, Integer y) const;

		[[nodiscard]] Integer size() const;
		[[nodiscard]] Integer width() const;
		[[nodiscard]] Integer height() const;

		void resize(Integer width, Integer height);
		void resize(Integer width, Integer height, const T& value);
	private:
		Integer m_Width, m_Height;
		std::vector<T> m_Data;
	};


	template<typename T>
	Vector2D<T>::Vector2D(Integer width, Integer height, const T &value) : m_Width(width), m_Height(height), m_Data(width * height, value)
	{
	}

	template<typename T>
	Vector2D<T>::Vector2D(Integer width, Integer height) : m_Width(width), m_Height(height), m_Data(width * height)
	{
	}

	template<typename T>
	Vector2D<T>::Vector2D(const T* array, Integer width, Integer height) :  m_Width(width), m_Height(height), m_Data(array, array + (width * height))
	{
	}

	template<typename T>
	void Vector2D<T>::resize(Integer width, Integer height, const T &value)
	{
		m_Data.resize(width * height, value);
		m_Width = width;
		m_Height = height;
	}

	template<typename T>
	void Vector2D<T>::resize(Integer width, Integer height)
	{
		m_Data.resize(width * height);
		m_Width = width;
		m_Height = height;
	}

	template<typename T>
	Integer Vector2D<T>::size() const {
		return m_Data.size();
	}

	template<typename T>
	Integer Vector2D<T>::width() const
	{
		return m_Width;
	}

	template<typename T>
	Integer Vector2D<T>::height() const
	{
		return m_Height;
	}

	template<typename T>
	const T &Vector2D<T>::operator()(Integer x, Integer y) const {
		return m_Data[x * m_Height + y];
	}

	template<typename T>
	T &Vector2D<T>::operator()(Integer x, Integer y) {
		return m_Data[x * m_Height + y];
	}


	template<typename T>
	const T *Vector2D<T>::operator[](Integer x) const {
		return &m_Data[x * m_Height];
	}

	template<typename T>
	T *Vector2D<T>::operator[](Integer x) {
		return &m_Data[x * m_Height];
	}

	template<typename T>
	const T &Vector2D<T>::operator()(Integer index) const {
		return m_Data[index];
	}

	template<typename T>
	T &Vector2D<T>::operator()(Integer index) {
		return m_Data[index];
	}

} // ML
// GG
