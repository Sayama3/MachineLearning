//
// Created by ianpo on 11/11/2023.
//


#include "UUID.hpp"
#include <random>

// Necessary element ro create a random 64bits number;
static std::random_device s_RandomDevice;
static std::mt19937_64 s_RandomEngine(s_RandomDevice());
static std::uniform_int_distribution<uint64_t> s_UniformDistribution;

UUID::UUID() : m_UUID(s_UniformDistribution(s_RandomEngine))
{
}

UUID::UUID(uint64_t id) : m_UUID(id) {
}

