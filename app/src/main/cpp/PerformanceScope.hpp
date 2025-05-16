#pragma once

#include "Log.hpp"
#include <chrono>
#include <string_view>

struct PerformanceScope {
	using clock = std::chrono::high_resolution_clock;

	explicit PerformanceScope(std::string_view name)
		: name(name), start(clock::now()) {}

	~PerformanceScope() {
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
			clock::now() - start
		);
		float duration_millis = (float)duration.count() / 1000.0f;
		LOG_INFO("{} took {} ms", name, duration_millis);
	}

  private:
	std::string_view name;
	clock::time_point start;
};

#define COMBINE(x, y) x##y
#define COMBINE2(x, y) COMBINE(x, y)
#define PROFILE_SCOPE(name)                                                    \
	PerformanceScope COMBINE2(__performance_scope_, __LINE__)(name);

#ifndef __FUNCTION_NAME__
#ifdef WIN32 // WINDOWS
#define FUNCTION_NAME() __FUNCTION__
#else //*NIX
#define FUNCTION_NAME() __func__
#endif
#endif

#define PROFILE_FUNCTION()                                                     \
	PerformanceScope COMBINE2(__performance_scope_, __LINE__)(FUNCTION_NAME());