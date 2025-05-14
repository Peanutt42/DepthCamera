#pragma once

#include <string_view>
#include <span>
#include "tflite/c/c_api.h"
#include "tflite/delegates/gpu/delegate.h"
#include "Log.hpp"

/** Helper class that wraps the tflite c api */
class TfLiteRuntime {
public:
    TfLiteRuntime(std::span<const int8_t> model_data,
                  std::string_view gpu_delegate_serialization_dir, std::string_view model_token);

    ~TfLiteRuntime();

    template<typename I, typename O>
    void run_inference(std::span<const I> input, std::span<O> output) {
        auto input_bytes = std::as_bytes(input);
        auto output_bytes = std::as_writable_bytes(output);
        _run_inference(input_bytes.data(), input_bytes.size(), output_bytes.data(),
                       output_bytes.size());
    }

    /// prints an error if status is != kTfLiteOk
    static void check_tflite_status(TfLiteStatus status);

    static std::string_view format_tflite_status(TfLiteStatus status);

private:
    void _run_inference(const void *input, size_t input_data_size,
                        void *output, size_t output_data_size);

    static TfLiteDelegate *create_gpu_delegate(std::string_view gpu_delegate_serialization_dir,
                                               std::string_view model_token);

private:
    TfLiteModel *model = nullptr;
    TfLiteInterpreter *interpreter = nullptr;
    TfLiteInterpreterOptions *interpreter_options = nullptr;
    /// can be null if GPU delegates are not supported on this device
    TfLiteDelegate *gpu_delegate = nullptr;
};