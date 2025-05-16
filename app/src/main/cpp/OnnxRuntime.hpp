#pragma once

#include "onnxruntime_c_api.h"
#include <cassert>
#include <onnxruntime_cxx_api.h>
#include <span>
#include <string_view>

class OnnxRuntime {
  public:
	explicit OnnxRuntime(std::span<const std::byte> model_data);

	template <typename I, typename O>
	void run_inference(std::span<I> input_data, std::span<O> output_data) {
		assert(input_type == Ort::TypeToTensorType<I>::type);
		assert(output_type == Ort::TypeToTensorType<O>::type);

		_run_inference(
			std::as_writable_bytes(input_data),
			std::as_writable_bytes(output_data)
		);
	}

	static void check_ort_status(OrtStatus* status);
	static std::string_view format_ort_error_code(OrtErrorCode error_code);
	static void log_error_ort_exception(const Ort::Exception& exception);

  private:
	void _run_inference(
		std::span<std::byte> input_data,
		std::span<std::byte> output_data
	);

  private:
	Ort::Env env;
	Ort::Session session;
	Ort::MemoryInfo memory_info;

	std::string input_name;
	std::vector<int64_t> input_shape;
	ONNXTensorElementDataType input_type;

	std::string output_name;
	std::vector<int64_t> output_shape;
	ONNXTensorElementDataType output_type;
};