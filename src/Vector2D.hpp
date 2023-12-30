//
// Created by ianpo on 30/12/2023.
//

#pragma once

#include <vector>

namespace GG::ML {

	template<typename T>
	class Vector2D {
	public:
		Vector2D(size_t width, size_t height);
		Vector2D(size_t width, size_t height, const T& value);

		Vector2D() = default;
		~Vector2D() = default;

		T& operator[](size_t index);
		const T& operator[](size_t index) const;

		T* operator()(size_t x);
		const T* operator()(size_t x) const;

		T& operator()(size_t x, size_t y);
		const T& operator()(size_t x, size_t y) const;

		size_t size() const;
		size_t width() const;
		size_t height() const;

		void resize(size_t width, size_t height);
		void resize(size_t width, size_t height, const T& value);
	private:
		std::vector<T> m_Data;
		size_t m_Width, m_Height;
	};

	template<typename T>
	void Vector2D<T>::resize(size_t width, size_t height, const T &value)
	{
		m_Data.resize(width * height, value);
		m_Width = width;
		m_Height = height;
	}

	template<typename T>
	void Vector2D<T>::resize(size_t width, size_t height)
	{
		m_Data.resize(width * height);
		m_Width = width;
		m_Height = height;
	}

	template<typename T>
	size_t Vector2D<T>::size() const {
		return m_Data.size();
	}

	template<typename T>
	size_t Vector2D<T>::width() const
	{
		return m_Width;
	}

	template<typename T>
	size_t Vector2D<T>::height() const
	{
		return m_Height;
	}

	template<typename T>
	const T *Vector2D<T>::operator()(size_t x) const {
		return &m_Data[x * m_Height];
	}

	template<typename T>
	T *Vector2D<T>::operator()(size_t x) {
		return &m_Data[x * m_Height];
	}

	template<typename T>
	const T &Vector2D<T>::operator()(size_t x, size_t y) const {
		return m_Data[x * m_Height + y];
	}

	template<typename T>
	T &Vector2D<T>::operator()(size_t x, size_t y) {
		return m_Data[x * m_Height + y];
	}

	template<typename T>
	const T &Vector2D<T>::operator[](size_t index) const {
		return m_Data[index];
	}

	template<typename T>
	T &Vector2D<T>::operator[](size_t index) {
		return m_Data[index];
	}

	template<typename T>
	Vector2D<T>::Vector2D(size_t width, size_t height, const T &value) : m_Width(width), m_Height(height), m_Data(width * height, value)
	{
	}

	template<typename T>
	Vector2D<T>::Vector2D(size_t width, size_t height) : m_Width(width), m_Height(height), m_Data(width * height)
	{
	}

} // ML
// GG
