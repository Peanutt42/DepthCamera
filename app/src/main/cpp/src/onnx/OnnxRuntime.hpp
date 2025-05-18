#pragma once

#include "onnxruntime_c_api.h"
#include "utils/Error.hpp"
#include <cassert>
#include <onnxruntime_cxx_api.h>
#include <span>
#include <string_view>

class OnnxRuntime {
  public:
	explicit OnnxRuntime(std::span<const std::byte> model_data);

	template<typename I, typename O>
	Option<OnnxRuntimeError>
	run_inference(std::span<I> input_data, std::span<O> output_data) {
		if (input_type != Ort::TypeToTensorType<I>::type)
			return OnnxRuntimeError::InvalidInputType;
		if (output_type != Ort::TypeToTensorType<O>::type)
			return OnnxRuntimeError::InvalidOutputType;

		return run_inference_raw(
			std::as_writable_bytes(input_data),
			std::as_writable_bytes(output_data)
		);
	}

	static void check_ort_status(OrtStatus* status);
	static std::string_view format_ort_error_code(OrtErrorCode error_code);
	static void log_error_ort_exception(const Ort::Exception& exception);

  private:
	Option<OnnxRuntimeError> run_inference_raw(
		std::span<std::byte> input_data,
		std::span<std::byte> output_data
	);

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