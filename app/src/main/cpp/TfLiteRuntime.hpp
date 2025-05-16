#pragma once

#include "PerformanceScope.hpp"
#include "TfLiteUtils.hpp"
#include "tflite/c/c_api.h" // IWYU pragma: export
#include "tflite/c/common.h"
#include <cassert>
#include <span>
#include <string_view>

/** Helper class that wraps the tflite c api */
class TfLiteRuntime {
  private:
	TfLiteModel* model = nullptr;
	TfLiteInterpreter* interpreter = nullptr;
	TfLiteInterpreterOptions* interpreter_options = nullptr;
	/// can be null if GPU delegates are not supported on this device
	TfLiteDelegate* gpu_delegate = nullptr;

  public:
	TfLiteRuntime(
		std::span<const int8_t> model_data,
		std::string_view gpu_delegate_serialization_dir,
		std::string_view model_token
	);

	~TfLiteRuntime();

	template <typename I, typename O>
	void run_inference(std::span<const I> input, std::span<O> output) {
		PROFILE_FUNCTION()

		_load_input<I>(input);
		{
			PROFILE_SCOPE("Invoking of model")
			CHECK_TFLITE_STATUS(TfLiteInterpreterInvoke, interpreter);
		}
		_read_output<O>(output);
	}

  private:
	template <typename I> void _load_input(std::span<const I> input) {
		PROFILE_SCOPE("Loading input")

		auto input_tensor = TfLiteInterpreterGetInputTensor(interpreter, 0);

		if (is_tensor_quantized(input_tensor)) {
			_load_quantized_input<I>(input, input_tensor);
		} else {
			_load_nonquantized_input(std::as_bytes(input), input_tensor, TFLITE_TYPE_FROM_TYPE<I>);
		}
	}

	template <typename O> void _read_output(std::span<O> output) {
		PROFILE_SCOPE("Reading output")

		auto output_tensor = TfLiteInterpreterGetOutputTensor(interpreter, 0);

		if (is_tensor_quantized(output_tensor)) {
			_read_quantized_output<O>(output, output_tensor);
		} else {
			_read_nonquantized_output(std::as_writable_bytes(output), output_tensor, TFLITE_TYPE_FROM_TYPE<O>);
		}
	}

	static void _load_nonquantized_input(
		std::span<const std::byte> input_bytes,
		TfLiteTensor* input_tensor,
		TfLiteType input_type
	);
	static void _read_nonquantized_output(
		std::span<std::byte> output_bytes,
		const TfLiteTensor* output_tensor,
		TfLiteType output_type
	);

	template <typename I>
	void _load_quantized_input(
		std::span<const I> input,
		TfLiteTensor* input_tensor
	) {
		const auto quantized_type_size =
			get_tflite_type_size(input_tensor->type);

		auto quantized_input_data_ptr =
			(std::byte*)TfLiteTensorData(input_tensor);
		assert(quantized_input_data_ptr != nullptr);
		const auto quantized_input_data_bytes =
			TfLiteTensorByteSize(input_tensor);
		if (quantized_input_data_bytes / quantized_type_size != input.size()) {
			LOG_ERROR(
				"input size ({}) does not match the expected size of the "
				"quantized input of the model ({})!",
				input.size(), quantized_input_data_bytes / quantized_type_size
			);
			return;
		}
		auto quantized_span = std::span<std::byte>(
			quantized_input_data_ptr, quantized_input_data_bytes
		);
		quantize<I>(
			input, quantized_span, input_tensor->type,
			*(const TfLiteAffineQuantization*)input_tensor->quantization.params
		);
	}

	template <typename O>
	void _read_quantized_output(
		std::span<O> output,
		const TfLiteTensor* output_tensor
	) {
		const auto quantized_type_size =
			get_tflite_type_size(output_tensor->type);
		auto quantized_output_data_ptr =
			(const std::byte*)TfLiteTensorData(output_tensor);
		assert(quantized_output_data_ptr != nullptr);
		const auto quantized_output_data_bytes =
			TfLiteTensorByteSize(output_tensor);
		if (quantized_output_data_bytes / quantized_type_size !=
			output.size()) {
			LOG_ERROR(
				"output size ({}) does not match the expected size of the "
				"quantized output of the model ({})!",
				output.size(), quantized_output_data_bytes / quantized_type_size
			);
			return;
		}
		auto quantized_output_span = std::span<const std::byte>(
			quantized_output_data_ptr, quantized_output_data_bytes
		);

		dequantize<O>(
			quantized_output_span, output, output_tensor->type,
			*(const TfLiteAffineQuantization*)output_tensor->quantization.params
		);
	}
};