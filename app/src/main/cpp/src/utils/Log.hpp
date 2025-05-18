#pragma once

#include <android/log.h>
#include <format>

template<typename... Args>
void formatted_log(int priority, const char* format, Args... args) {
	const std::string formatted =
		std::vformat(format, std::make_format_args(args...));

	__android_log_write(priority, "Native Lib", formatted.c_str());
}

#define LOG_INFO(...) formatted_log(ANDROID_LOG_INFO, __VA_ARGS__)
#define LOG_ERROR(...) formatted_log(ANDROID_LOG_ERROR, __VA_ARGS__)

#define LOG_OPT_ERROR(...)                                                     \
	{                                                                          \
		if (const auto opt_error = __VA_ARGS__) {                              \
			LOG_ERROR("{}", opt_error.value().message());                      \
		}                                                                      \
	}