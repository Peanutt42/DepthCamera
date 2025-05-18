#pragma once

#include "TfLiteUtils.hpp"
#include "tflite/c/c_api.h" // IWYU pragma: export
#include "tflite/c/c_api_types.h"
#include "tflite/c/common.h"
#include "utils/Error.hpp"
#include "utils/Profiling.hpp"
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
	explicit TfLiteRuntime(
		std::span<const int8_t> model_data,
		std::string_view gpu_delegate_serialization_dir,
		std::string_view model_token
	);
	~TfLiteRuntime();

	TfLiteRuntime(TfLiteRuntime&&) = delete;
	TfLiteRuntime(const TfLiteRuntime&) = delete;
	void operator=(TfLiteRuntime&&) = delete;
	void operator=(const TfLiteRuntime&) = delete;

	template<typename I, typename O>
	Option<TfLiteRuntimeError>
	run_inference(std::span<const I> input, std::span<O> output) {
		PROFILE_DEPTH_FUNCTION()

		if (const auto error = load_input<I>(input))
			return error;

		{
			PROFILE_DEPTH_SCOPE("Invoking of model")
			const auto invoke_status = TfLiteInterpreterInvoke(interpreter);
			check_tflite_status(invoke_status, "TfLiteInterpreterInvoke");
			if (invoke_status != kTfLiteOk)
				return TfLiteRuntimeError::FailedToInvokeInterpreter;
		}
		return read_output<O>(output);
	}

  private:
	template<typename I>
	Option<TfLiteRuntimeError> load_input(std::span<const I> input) {
		PROFILE_DEPTH_SCOPE("Loading input")

		auto* input_tensor = TfLiteInterpreterGetInputTensor(interpreter, 0);

		if (is_tensor_quantized(input_tensor)) {
			return load_quantized_input<I>(input, input_tensor);
		}
		return load_nonquantized_input(std::as_bytes(input), input_tensor, TFLITE_TYPE_FROM_TYPE<I>);
	}

	template<typename O>
	[[nodiscard]] Option<TfLiteRuntimeError> read_output(std::span<O> output) {
		PROFILE_DEPTH_SCOPE("Reading output")

		const auto* output_tensor =
			TfLiteInterpreterGetOutputTensor(interpreter, 0);

		if (is_tensor_quantized(output_tensor)) {
			return read_quantized_output<O>(output, output_tensor);
		}
		return read_nonquantized_output(std::as_writable_bytes(output), output_tensor, TFLITE_TYPE_FROM_TYPE<O>);
	}

	static Option<TfLiteRuntimeError> load_nonquantized_input(
		std::span<const std::byte> input_bytes,
		TfLiteTensor* input_tensor,
		TfLiteType input_type
	);
	static Option<TfLiteRuntimeError> read_nonquantized_output(
		std::span<std::byte> output_bytes,
		const TfLiteTensor* output_tensor,
		TfLiteType output_type
	);

	template<typename I>
	Option<TfLiteRuntimeError>
	load_quantized_input(std::span<const I> input, TfLiteTensor* input_tensor) {
		const auto quantized_type_size =
			get_tflite_type_size(input_tensor->type);

		if (!quantized_type_size.has_value())
			return TfLiteRuntimeError::InvalidQuantizedInputType;

		void* quantized_input_data_ptr = TfLiteTensorData(input_tensor);
		if (quantized_input_data_ptr == nullptr)
			return TfLiteRuntimeError::TensorNotYetCreated;
		const auto quantized_input_data_bytes =
			TfLiteTensorByteSize(input_tensor);
		if (quantized_input_data_bytes / *quantized_type_size != input.size())
			return TfLiteRuntimeError::InvalidInputSize;
		// NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast)
		auto quantized_span = std::span<std::byte>(
			reinterpret_cast<std::byte*>(quantized_input_data_ptr),
			quantized_input_data_bytes
		);
		return quantize<I>(
			input, quantized_span, input_tensor->type,
			*reinterpret_cast<const TfLiteAffineQuantization*>(
				input_tensor->quantization.params
			)
		);
		// NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast)
	}

	template<typename O>
	Option<TfLiteRuntimeError> read_quantized_output(
		std::span<O> output,
		const TfLiteTensor* output_tensor
	) {
		const auto quantized_type_size =
			get_tflite_type_size(output_tensor->type);
		if (!quantized_type_size.has_value())
			return TfLiteRuntimeError::InvalidQuantizedOutputType;

		const void* quantized_output_data_ptr = TfLiteTensorData(output_tensor);
		if (quantized_output_data_ptr == nullptr)
			return TfLiteRuntimeError::TensorNotYetCreated;
		const auto quantized_output_data_bytes =
			TfLiteTensorByteSize(output_tensor);
		if (quantized_output_data_bytes / *quantized_type_size != output.size())
			return TfLiteRuntimeError::InvalidOutputSize;
		// NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast)
		auto quantized_output_span = std::span<const std::byte>(
			reinterpret_cast<const std::byte*>(quantized_output_data_ptr),
			quantized_output_data_bytes
		);

		return dequantize<O>(
			quantized_output_span, output, output_tensor->type,
			*reinterpret_cast<const TfLiteAffineQuantization*>(
				output_tensor->quantization.params
			)
		);
		// NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast)
	}
};