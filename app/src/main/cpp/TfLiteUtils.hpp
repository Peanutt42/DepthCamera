#pragma once

#include "Log.hpp"
#include "PerformanceScope.hpp"
#include <cassert>
#include <span>
#include <string_view>
#include <tflite/c/c_api.h>
#include <tflite/c/common.h>
#include <tflite/delegates/gpu/delegate.h>

inline static std::string_view format_tflite_type(TfLiteType type);

inline static bool is_tensor_quantized(const TfLiteTensor* tensor) {
	return tensor->quantization.type == kTfLiteAffineQuantization;
}

template <typename T>
inline static void quantize(
	std::span<const T> values,
	std::span<std::byte> quantized_values,
	TfLiteType quantized_type,
	const TfLiteAffineQuantization& quantization
);

template <>
void quantize<float>(
	std::span<const float> values,
	std::span<std::byte> quantized_values,
	TfLiteType quantized_type,
	const TfLiteAffineQuantization& quantization
) {
	PROFILE_FUNCTION()

	if (quantized_type != kTfLiteUInt8) {
		LOG_ERROR(
			"Unsupported quantization from float32 to {}",
			format_tflite_type(quantized_type)
		);
		return;
	}

	static_assert(sizeof(std::byte) == sizeof(uint8_t));
	auto quantized_uint8_values = std::span<uint8_t>(
		(uint8_t*)quantized_values.data(), quantized_values.size()
	);

	if (values.size() != quantized_uint8_values.size()) {
		LOG_ERROR(
			"values and quantized_uint8_values do not have the same length"
		);
		return;
	}

	// for now, only 1 input, 1 output
	assert(quantization.scale->size == 1);
	const float quantization_scale = quantization.scale->data[0];
	assert(quantization.zero_point->size == 1);
	const int quantization_zero_point = quantization.zero_point->data[0];

	for (size_t i = 0; i < values.size(); i++) {
		quantized_uint8_values[i] = (uint8_t)(values[i] / quantization_scale) +
									(uint8_t)quantization_zero_point;
	}
}

template <typename T>
inline static void dequantize(
	std::span<const std::byte> quantized_values,
	std::span<T> real_values,
	TfLiteType quantized_type,
	const TfLiteAffineQuantization& quantization
);
template <>
void dequantize<float>(
	std::span<const std::byte> quantized_values,
	std::span<float> real_values,
	TfLiteType quantized_type,
	const TfLiteAffineQuantization& quantization
) {
	PROFILE_FUNCTION()

	if (quantized_type != kTfLiteUInt8) {
		LOG_ERROR(
			"Unsupported dequantization from {} to float32",
			format_tflite_type(quantized_type)
		);
		return;
	}

	static_assert(sizeof(std::byte) == sizeof(uint8_t));
	auto quantized_uint8_values = std::span<const uint8_t>(
		(const uint8_t*)quantized_values.data(), quantized_values.size()
	);

	if (quantized_uint8_values.size() != real_values.size()) {
		LOG_ERROR(
			"quantized_uint8_values and real_values do not have the same length"
		);
		return;
	}

	// for now, only 1 input, 1 output
	assert(quantization.scale->size == 1);
	const float quantization_scale = quantization.scale->data[0];
	assert(quantization.zero_point->size == 1);
	const int quantization_zero_point = quantization.zero_point->data[0];

	for (size_t i = 0; i < real_values.size(); i++) {
		real_values[i] =
			quantization_scale *
			(float)(quantized_uint8_values[i] - quantization_zero_point);
	}
}

inline static TfLiteDelegate* create_gpu_delegate(
	std::string_view gpu_delegate_serialization_dir,
	std::string_view model_token
) {
	PROFILE_FUNCTION()

	TfLiteGpuDelegateOptionsV2 gpu_delegate_options =
		TfLiteGpuDelegateOptionsV2Default();
	gpu_delegate_options.is_precision_loss_allowed = true;
	gpu_delegate_options.inference_preference =
		TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER;
	gpu_delegate_options.experimental_flags |= TfLiteGpuExperimentalFlags::
		TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_SERIALIZATION;
	gpu_delegate_options.serialization_dir =
		gpu_delegate_serialization_dir.data();
	gpu_delegate_options.model_token = model_token.data();

	return TfLiteGpuDelegateV2Create(&gpu_delegate_options);
}

template <typename T>
inline constexpr TfLiteType TFLITE_TYPE_FROM_TYPE = kTfLiteNoType;

// clang-format off
template<> inline constexpr TfLiteType TFLITE_TYPE_FROM_TYPE<float> = kTfLiteFloat32;
template<> inline constexpr TfLiteType TFLITE_TYPE_FROM_TYPE<int32_t> = kTfLiteInt32;
template<> inline constexpr TfLiteType TFLITE_TYPE_FROM_TYPE<uint8_t> = kTfLiteUInt8;
template<> inline constexpr TfLiteType TFLITE_TYPE_FROM_TYPE<int64_t> = kTfLiteInt64;
template<> inline constexpr TfLiteType TFLITE_TYPE_FROM_TYPE<bool> = kTfLiteBool;
template<> inline constexpr TfLiteType TFLITE_TYPE_FROM_TYPE<int16_t> = kTfLiteInt16;
template<> inline constexpr TfLiteType TFLITE_TYPE_FROM_TYPE<int8_t> = kTfLiteInt8;
template<> inline constexpr TfLiteType TFLITE_TYPE_FROM_TYPE<TfLiteFloat16> = kTfLiteFloat16;
template<> inline constexpr TfLiteType TFLITE_TYPE_FROM_TYPE<double> = kTfLiteFloat64;
template<> inline constexpr TfLiteType TFLITE_TYPE_FROM_TYPE<uint64_t> = kTfLiteUInt64;
template<> inline constexpr TfLiteType TFLITE_TYPE_FROM_TYPE<uint32_t> = kTfLiteUInt32;
template<> inline constexpr TfLiteType TFLITE_TYPE_FROM_TYPE<uint16_t> = kTfLiteUInt16;
template<> inline constexpr TfLiteType TFLITE_TYPE_FROM_TYPE<TfLiteBFloat16> = kTfLiteBFloat16;
// clang-format on

/// returns 0 if unknown
inline static size_t get_tflite_type_size(TfLiteType type) {
	switch (type) {
	default:
		return 0; // unknown
	case kTfLiteFloat32:
		return sizeof(float);
	case kTfLiteInt32:
		return sizeof(int32_t);
	case kTfLiteUInt8:
		return sizeof(uint8_t);
	case kTfLiteInt64:
		return sizeof(int64_t);
	case kTfLiteBool:
		return sizeof(bool);
	case kTfLiteInt16:
		return sizeof(int16_t);
	case kTfLiteInt8:
		return sizeof(int8_t);
	case kTfLiteFloat16:
		return 2;
	case kTfLiteFloat64:
		return sizeof(double);
	case kTfLiteUInt64:
		return sizeof(uint64_t);
	case kTfLiteUInt32:
		return sizeof(uint32_t);
	case kTfLiteUInt16:
		return sizeof(uint16_t);
	case kTfLiteBFloat16:
		return 2;
	}
}

std::string_view format_tflite_type(TfLiteType type) {
	switch (type) {
	default:
		return "unknown";
	case kTfLiteNoType:
		return "no type";
	case kTfLiteFloat32:
		return "float32";
	case kTfLiteInt32:
		return "int32";
	case kTfLiteUInt8:
		return "uint8";
	case kTfLiteInt64:
		return "int64";
	case kTfLiteString:
		return "string";
	case kTfLiteBool:
		return "bool";
	case kTfLiteInt16:
		return "int16";
	case kTfLiteComplex64:
		return "complex64";
	case kTfLiteInt8:
		return "int8";
	case kTfLiteFloat16:
		return "float16";
	case kTfLiteFloat64:
		return "float64";
	case kTfLiteComplex128:
		return "complex128";
	case kTfLiteUInt64:
		return "uint64";
	case kTfLiteResource:
		return "resource";
	case kTfLiteVariant:
		return "variant";
	case kTfLiteUInt32:
		return "uint32";
	case kTfLiteUInt16:
		return "uint16";
	case kTfLiteInt4:
		return "int4";
	case kTfLiteBFloat16:
		return "bfloat16";
	}
}

inline static std::string_view format_tflite_status(TfLiteStatus status) {
	switch (status) {
	case kTfLiteOk:
		return "ok";
	case kTfLiteError:
		return "general error";
	case kTfLiteDelegateError:
		return "delegate error";
	case kTfLiteApplicationError:
		return "application error";
	case kTfLiteDelegateDataNotFound:
		return "delegate data not found";
	case kTfLiteDelegateDataWriteError:
		return "delegate data write error";
	case kTfLiteDelegateDataReadError:
		return "delegate data read error";
	case kTfLiteUnresolvedOps:
		return "unresolved Ops";
	case kTfLiteCancelled:
		return "canceled";
	case kTfLiteOutputShapeNotKnown:
		return "output shape not known";
	default:
		return "unknown";
	}
}

/// prints an error if status is != kTfLiteOk
inline static void check_tflite_status(
	TfLiteStatus status,
	std::string_view tflite_function_name
) {
	if (status == kTfLiteOk)
		return;

	LOG_ERROR(
		"{} returned {} during", tflite_function_name,
		format_tflite_status(status)
	);
}

#define CHECK_TFLITE_STATUS(function, ...)                                     \
	check_tflite_status(function(__VA_ARGS__), #function)