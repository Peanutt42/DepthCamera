#include "TfLiteRuntime.hpp"
#include "PerformanceScope.hpp"

#include "tflite/core/c/c_api_experimental.h"

void tflite_error_callback(void *user_data, const char *format, va_list args) {
    va_list args_copy;
    va_copy(args_copy, args);

    int formatted_error_msg_length = std::vsnprintf(nullptr, 0, format, args_copy);
    std::vector<char> formatted_error_msg_buffer;
    formatted_error_msg_buffer.resize(formatted_error_msg_length + 1);
    std::vsnprintf(formatted_error_msg_buffer.data(), formatted_error_msg_buffer.size(), format,
                   args);
    std::string formatted_error_msg(formatted_error_msg_buffer.data());

    LOG_ERROR("[TFLITE ERROR CALLBACK] {} (user_data: {})", formatted_error_msg.c_str(), user_data);
}

TfLiteRuntime::TfLiteRuntime(std::span<const int8_t> model_data,
                             std::string_view gpu_delegate_serialization_dir,
                             std::string_view model_token) {
    PROFILE_SCOPE("Initialize TfLiteRuntime")

    model = TfLiteModelCreate(model_data.data(), model_data.size());

    interpreter_options = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsSetErrorReporter(interpreter_options, tflite_error_callback, nullptr);
    TfLiteInterpreterOptionsSetNumThreads(interpreter_options, 4);

    gpu_delegate = create_gpu_delegate(gpu_delegate_serialization_dir, model_token);
    TfLiteInterpreterOptionsAddDelegate(interpreter_options, gpu_delegate);

    interpreter = TfLiteInterpreterCreate(model, interpreter_options);

    TfLiteInterpreterAllocateTensors(interpreter);
}

TfLiteRuntime::~TfLiteRuntime() {
    PROFILE_SCOPE("Shutdown TfLiteRuntime")

    TfLiteInterpreterDelete(interpreter);
    if (gpu_delegate)
        TfLiteGpuDelegateV2Delete(gpu_delegate);
    TfLiteInterpreterOptionsDelete(interpreter_options);
    TfLiteModelDelete(model);
}

void TfLiteRuntime::_run_inference(const void *input, size_t input_data_size,
                                   void *output, size_t output_data_size) {
    PROFILE_SCOPE("Total Inference")

    {
        PROFILE_SCOPE("Loading Input")

        TfLiteTensor *input_tensor = TfLiteInterpreterGetInputTensor(interpreter, 0);
        check_tflite_status(
                TfLiteTensorCopyFromBuffer(input_tensor, input, input_data_size));
    }

    {
        PROFILE_SCOPE("Invoking of model")

        check_tflite_status(TfLiteInterpreterInvoke(interpreter));
    }

    {
        PROFILE_SCOPE("Reading output")

        const TfLiteTensor *output_tensor = TfLiteInterpreterGetOutputTensor(interpreter, 0);
        check_tflite_status(
                TfLiteTensorCopyToBuffer(output_tensor, output, output_data_size));
    }
}

TfLiteDelegate *TfLiteRuntime::create_gpu_delegate(std::string_view gpu_delegate_serialization_dir,
                                                   std::string_view model_token) {
    TfLiteGpuDelegateOptionsV2 gpu_delegate_options = TfLiteGpuDelegateOptionsV2Default();
    gpu_delegate_options.is_precision_loss_allowed = true;
    gpu_delegate_options.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER;
    gpu_delegate_options.experimental_flags |= TfLiteGpuExperimentalFlags::TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_SERIALIZATION;
    gpu_delegate_options.serialization_dir = gpu_delegate_serialization_dir.data();
    gpu_delegate_options.model_token = model_token.data();

    return TfLiteGpuDelegateV2Create(&gpu_delegate_options);
}

void TfLiteRuntime::check_tflite_status(TfLiteStatus
                                        status) {
    if (status == kTfLiteOk)
        return;

    LOG_ERROR("TfLite function returned {}", format_tflite_status(status));
}

std::string_view TfLiteRuntime::format_tflite_status(TfLiteStatus
                                                     status) {
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