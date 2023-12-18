//
// Created by Sayama on 18/12/2023.
//

#pragma once
#include <machinelearning_export.hpp>

#include <memory>
#include <filesystem>

namespace ML::Core {
}

#ifdef ML_DEBUG

//#define ML_CORE_TRACE(...)       ::ML::Core::Log::GetCoreLogger()->log(spdlog::source_loc{__FILE__, __LINE__, ML_FUNC}, spdlog::level::trace, __VA_ARGS__)
//#define ML_CORE_INFO(...)        ::ML::Core::Log::GetCoreLogger()->log(spdlog::source_loc{__FILE__, __LINE__, ML_FUNC}, spdlog::level::info, __VA_ARGS__)
//#define ML_CORE_WARNING(...)     ::ML::Core::Log::GetCoreLogger()->log(spdlog::source_loc{__FILE__, __LINE__, ML_FUNC}, spdlog::level::warn, __VA_ARGS__)
//#define ML_CORE_ERROR(...)       ::ML::Core::Log::GetCoreLogger()->log(spdlog::source_loc{__FILE__, __LINE__, ML_FUNC}, spdlog::level::err, __VA_ARGS__)
//#define ML_CORE_CRITICAL(...)    ::ML::Core::Log::GetCoreLogger()->log(spdlog::source_loc{__FILE__, __LINE__, ML_FUNC}, spdlog::level::critical, __VA_ARGS__)
//
//
//#define ML_TRACE(...)       ::ML::Core::Log::GetClientLogger()->log(spdlog::source_loc{__FILE__, __LINE__, ML_FUNC}, spdlog::level::trace, __VA_ARGS__)
//#define ML_INFO(...)        ::ML::Core::Log::GetClientLogger()->log(spdlog::source_loc{__FILE__, __LINE__, ML_FUNC}, spdlog::level::info, __VA_ARGS__)
//#define ML_WARNING(...)     ::ML::Core::Log::GetClientLogger()->log(spdlog::source_loc{__FILE__, __LINE__, ML_FUNC}, spdlog::level::warn, __VA_ARGS__)
//#define ML_ERROR(...)       ::ML::Core::Log::GetClientLogger()->log(spdlog::source_loc{__FILE__, __LINE__, ML_FUNC}, spdlog::level::err, __VA_ARGS__)
//#define ML_CRITICAL(...)    ::ML::Core::Log::GetClientLogger()->log(spdlog::source_loc{__FILE__, __LINE__, ML_FUNC}, spdlog::level::critical, __VA_ARGS__)


#define ML_CORE_TRACE(...)
#define ML_CORE_INFO(...)
#define ML_CORE_WARNING(...)
#define ML_CORE_ERROR(...)
#define ML_CORE_CRITICAL(...)


#define ML_TRACE(...)
#define ML_INFO(...)
#define ML_WARNING(...)
#define ML_ERROR(...)
#define ML_CRITICAL(...)

#else

#define ML_CORE_TRACE(...)
#define ML_CORE_INFO(...)
#define ML_CORE_WARNING(...)
#define ML_CORE_ERROR(...)
#define ML_CORE_CRITICAL(...)


#define ML_TRACE(...)
#define ML_INFO(...)
#define ML_WARNING(...)
#define ML_ERROR(...)
#define ML_CRITICAL(...)

#endif