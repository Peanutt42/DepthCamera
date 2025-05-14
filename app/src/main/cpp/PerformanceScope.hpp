#pragma once

#include <chrono>
#include "Log.hpp"

struct PerformanceScope {
    explicit PerformanceScope(const char *name) : name(name),
                                                  start(std::chrono::high_resolution_clock::now()) {}

    ~PerformanceScope() {
        auto duration = std::chrono::high_resolution_clock::now() - start;
        LOG_INFO("{} took {} ms", name,
                 std::chrono::duration_cast<std::chrono::microseconds>(duration).count() / 1000.0f);
    }


private:
    const char *name;
    std::chrono::high_resolution_clock::time_point start;
};

#define COMBINE(x, y) x ## y
#define COMBINE2(x, y) COMBINE(x, y)
#define PROFILE_SCOPE(name) PerformanceScope COMBINE2(__performance_scope_, __LINE__)(name);

#ifndef __FUNCTION_NAME__
#ifdef WIN32   //WINDOWS
#define FUNCTION_NAME()   __FUNCTION__
#else          //*NIX
#define FUNCTION_NAME()   __func__
#endif
#endif

#define PROFILE_FUNCTION() PerformanceScope COMBINE2(__performance_scope_, __LINE__)(FUNCTION_NAME());