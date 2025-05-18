#pragma once

#include <optional>
#include <string_view>

template<typename T>
class [[nodiscard]] Option : public std::optional<T> {
  public:
	using std::optional<T>::optional;
	using std::optional<T>::operator=;
	using std::optional<T>::operator bool;
};

// NOLINTBEGIN(readability-identifier-naming)
template<typename T>
Option<T> Some(T value) {
	return Option<T>(value);
}
inline constexpr std::nullopt_t None{
	std::nullopt_t::__secret_tag{}, std::nullopt_t::__secret_tag{}
};
// NOLINTEND(readability-identifier-naming)

#define DEFINE_ERROR(type_name, ...)                                           \
	class [[nodiscard]] type_name {                                            \
	  public:                                                                  \
		enum Error : std::uint8_t { __VA_ARGS__ };                             \
		[[nodiscard]] explicit(false) constexpr type_name(Error error)         \
			: error(error) {}                                                  \
		[[nodiscard]] constexpr operator Error() const { return error; }       \
		[[nodiscard]] inline std::string_view message() const;                 \
                                                                               \
	  private:                                                                 \
		Error error;                                                           \
	}

DEFINE_ERROR(
	BitmapError,
	FormatNotRGBA8888,
	WrongOutputArraySize,
	FailedToLockPixels
);
std::string_view BitmapError::message() const {
	switch (error) {
	case Error::FormatNotRGBA8888:
		return "format is not RGBA8888";
	case Error::WrongOutputArraySize:
		return "output array has the wrong size";
	case Error::FailedToLockPixels:
		return "failed to lock pixels of bitmap";
	}
}

DEFINE_ERROR(ImageError, WrongOutPixelSize);
std::string_view ImageError::message() const {
	switch (error) {
	case Error::WrongOutPixelSize:
		return "out pixels have wrong size";
	}
}

DEFINE_ERROR(ColormapError, WrongOutputArraySize);
std::string_view ColormapError::message() const {
	switch (error) {
	case Error::WrongOutputArraySize:
		return "wrong output array size";
	}
}

DEFINE_ERROR(
	OnnxRuntimeError,
	InvalidInputType,
	InvalidOutputType,
	InvalidInputCount,
	InvalidOutputCount,
	RunInferenceException,
);
std::string_view OnnxRuntimeError::message() const {
	switch (error) {
	case Error::InvalidInputType:
		return "invalid input type";
	case Error::InvalidOutputType:
		return "invalid output type";
	case Error::InvalidInputCount:
		return "invalid input count";
	case Error::InvalidOutputCount:
		return "invalid output count";
	case Error::RunInferenceException:
		return "run_inference onnx exception";
	}
}

DEFINE_ERROR(
	TfLiteRuntimeError,
	FailedToInvokeInterpreter,
	InvalidInputType,
	InvalidOutputType,
	InvalidQuantizedInputType,
	InvalidQuantizedOutputType,
	TensorNotYetCreated,
	InvalidInputSize,
	InvalidOutputSize,
	FailedToCopyToBuffer,
	FailedToCopyFromBuffer,
	UnsupportedTypeQuantization,
	UnsupportedAsymmetricQuantization
);
std::string_view TfLiteRuntimeError::message() const {
	switch (error) {
	case Error::FailedToInvokeInterpreter:
		return "failed to invoke tflite interpreter";
	case Error::InvalidInputType:
		return "invalid input type";
	case Error::InvalidOutputType:
		return "invalid output type";
	case Error::InvalidQuantizedInputType:
		return "invalid quantized input type";
	case Error::InvalidQuantizedOutputType:
		return "invalid quantized output type";
	case Error::TensorNotYetCreated:
		return "tensor not yet created";
	case Error::InvalidInputSize:
		return "invalid input size";
	case Error::InvalidOutputSize:
		return "invalid output size";
	case Error::FailedToCopyToBuffer:
		return "failed to copy to buffer";
	case Error::FailedToCopyFromBuffer:
		return "failed to copy from buffer";
	case Error::UnsupportedTypeQuantization:
		return "quantization of this type is unsupported";
	case Error::UnsupportedAsymmetricQuantization:
		return "asymmetric quantization is unsupported";
	}
}