#include "TfLiteRuntime.hpp"
#include "PerformanceScope.hpp"

#include "tflite/c/common.h"
#include <cassert>

void tflite_error_callback(
	void* /*user_data*/,
	const char* format,
	va_list args
);

TfLiteRuntime::TfLiteRuntime(
	std::span<const int8_t> model_data,
	std::string_view gpu_delegate_serialization_dir,
	std::string_view model_token
) {
	PROFILE_SCOPE("Initialize TfLiteRuntime")

	model = TfLiteModelCreate(model_data.data(), model_data.size());

	interpreter_options = TfLiteInterpreterOptionsCreate();
	TfLiteInterpreterOptionsSetErrorReporter(
		interpreter_options, tflite_error_callback, nullptr
	);
	TfLiteInterpreterOptionsSetNumThreads(interpreter_options, 4);

	gpu_delegate =
		create_gpu_delegate(gpu_delegate_serialization_dir, model_token);
	TfLiteInterpreterOptionsAddDelegate(interpreter_options, gpu_delegate);

	interpreter = TfLiteInterpreterCreate(model, interpreter_options);

	CHECK_TFLITE_STATUS(TfLiteInterpreterAllocateTensors, interpreter);
}

TfLiteRuntime::~TfLiteRuntime() {
	PROFILE_SCOPE("Shutdown TfLiteRuntime")

	TfLiteInterpreterDelete(interpreter);
	if (gpu_delegate)
		TfLiteGpuDelegateV2Delete(gpu_delegate);
	TfLiteInterpreterOptionsDelete(interpreter_options);
	TfLiteModelDelete(model);
}

void TfLiteRuntime::_load_nonquantized_input(
	std::span<const std::byte> input_bytes,
	TfLiteTensor* input_tensor,
	TfLiteType input_type
) {
	PROFILE_FUNCTION()

	if (input_tensor->type != input_type) {
		LOG_ERROR(
			"input type is {} but model requires the input to be {}",
			format_tflite_type(input_type),
			format_tflite_type(input_tensor->type)
		);
		return;
	}

	CHECK_TFLITE_STATUS(
		TfLiteTensorCopyFromBuffer, input_tensor, input_bytes.data(),
		input_bytes.size_bytes()
	);
}

void TfLiteRuntime::_read_nonquantized_output(
	std::span<std::byte> output_bytes,
	const TfLiteTensor* output_tensor,
	TfLiteType output_type
) {
	if (output_tensor->type != output_type) {
		LOG_ERROR(
			"output type is {} but model requires the output to be {}",
			format_tflite_type(output_type),
			format_tflite_type(output_tensor->type)
		);
		return;
	}

	CHECK_TFLITE_STATUS(
		TfLiteTensorCopyToBuffer, output_tensor, output_bytes.data(),
		output_bytes.size_bytes()
	);
}

void tflite_error_callback(
	void* /*user_data*/,
	const char* format,
	va_list args
) {
	va_list args_copy;
	va_copy(args_copy, args);

	int formatted_error_msg_length =
		std::vsnprintf(nullptr, 0, format, args_copy);
	std::vector<char> formatted_error_msg_buffer;
	formatted_error_msg_buffer.resize(formatted_error_msg_length + 1);
	std::vsnprintf(
		formatted_error_msg_buffer.data(), formatted_error_msg_buffer.size(),
		format, args
	);
	std::string formatted_error_msg(formatted_error_msg_buffer.data());

	LOG_ERROR("[TFLITE ERROR CALLBACK] {}", formatted_error_msg.c_str());
}