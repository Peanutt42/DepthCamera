#include "Profiling.hpp"
#include <chrono>
#include <format>

std::string padding_tabs(size_t amount) {
	std::string result;
	result.reserve(amount * 4);
	for (size_t i = 0; i < amount; i++) {
		result.append("    ");
	}
	return result;
}

std::string format_duration_millis(profile_clock::duration duration) {
	auto duration_micros =
		std::chrono::duration_cast<std::chrono::microseconds>(duration);
	return std::format("{:.2f} ms", (float)duration_micros.count() / 1000.0f);
}

ProfileScope::ProfileScope(std::string_view name, ProfilingFrame& frame)
	: name(name), frame(frame), start(profile_clock::now()) {
	scope_depth = frame.start_scope();
}

ProfileScope::~ProfileScope() {
	auto duration = profile_clock::now() - start;
	frame.end_scope(ProfileScopeRecord{
		name,
		scope_depth,
		start,
		duration
	});
}

std::string ProfileScopeRecord::formatted() const {
	auto padding = padding_tabs(std::max(scope_depth, 0));
	return std::format(
		"{}{}: {}", padding, name, format_duration_millis(duration)
	);
}

int ProfilingFrame::start_scope() { return current_frame_scope_depth++; }

void ProfilingFrame::end_scope(ProfileScopeRecord scope) {
	profile_scopes.push_back(scope);
	current_frame_scope_depth--;
}

std::string ProfilingFrame::finish() {
	auto end = profile_clock::now();

	std::sort(
		profile_scopes.begin(), profile_scopes.end(),
		[](const auto& a, const auto& b) -> bool { return a.start < b.start; }
	);
	std::string profile_scopes_formatted;
	for (const auto& profile_scope : profile_scopes) {
		profile_scopes_formatted +=
			std::format("    {}\n", profile_scope.formatted());
	}
	auto frame_duration = end - start;
	auto frame_duration_ms =
		(float
		)std::chrono::duration_cast<std::chrono::microseconds>(frame_duration)
			.count() /
		1000.0f;
	auto frame_fps = 1.0 / (frame_duration_ms / 1000.0f);
	auto formatted = std::format(
		"{} Frame: {:.2f} fps ({:.2f} ms)\n{}", name, frame_fps,
		frame_duration_ms, profile_scopes_formatted
	);

	profile_scopes.clear();
	current_frame_scope_depth = 0;
	start = profile_clock::now();

	return formatted;
}

ProfilingFrame& get_depth_profiling_frame() {
	static ProfilingFrame DEPTH_PROFILING_FRAME = ProfilingFrame("Depth");
	return DEPTH_PROFILING_FRAME;
}
std::string& get_last_depth_profiling_frame_formatted() {
	static std::string LAST_DEPTH_PROFILING_FRAME_FORMATTED;
	return LAST_DEPTH_PROFILING_FRAME_FORMATTED;
}
ProfilingFrame& get_camera_profiling_frame() {
	static ProfilingFrame CAMERA_PROFILING_FRAME = ProfilingFrame("Depth");
	return CAMERA_PROFILING_FRAME;
}
std::string& get_last_camera_profiling_frame_formatted() {
	static std::string LAST_CAMERA_PROFILING_FRAME_FORMATTED;
	return LAST_CAMERA_PROFILING_FRAME_FORMATTED;
}