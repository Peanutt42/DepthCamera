#include <vector>
#include <memory>
#include <jni.h>
#include <android/log.h>

#include "NativeJavaScopes.hpp"
#include "TfLiteRuntime.hpp"
#include "DepthEstimation.hpp"
#include "ImageUtils.hpp"


static std::unique_ptr<TfLiteRuntime> depth_estimation_tflite_runtime = nullptr;

extern "C"
JNIEXPORT void JNICALL
Java_com_example_depthcamera_NativeLib_initDepthTfLiteRuntime(JNIEnv *env, jobject /*thiz*/,
                                                              jbyteArray model,
                                                              jstring gpu_delegate_serialization_dir,
                                                              jstring model_token) {

    NativeByteArrayScope model_data_array(env, model);
    NativeStringScope gpu_delegate_serialization_dir_string(env, gpu_delegate_serialization_dir);
    NativeStringScope model_token_string(env, model_token);

    depth_estimation_tflite_runtime = std::make_unique<TfLiteRuntime>(
            model_data_array.as_span(),
            gpu_delegate_serialization_dir_string.c_str(), model_token_string.c_str());
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_depthcamera_NativeLib_shutdownDepthTfLiteRuntime(JNIEnv * /*env*/,
                                                                  jobject /*thiz*/) {
    depth_estimation_tflite_runtime.reset(nullptr);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_depthcamera_NativeLib_runDepthInference(JNIEnv *env, jobject /*thiz*/,
                                                         jfloatArray input,
                                                         jfloatArray output, jfloat mean_r,
                                                         jfloat mean_g, jfloat mean_b,
                                                         jfloat stddev_r, jfloat stddev_g,
                                                         jfloat stddev_b) {
    if (depth_estimation_tflite_runtime == nullptr) {
        LOG_ERROR("TfLiteRuntime not initialized!");
        return;
    }

    NativeFloatArrayScope input_array(env, input);
    NativeFloatArrayScope output_array(env, output);

    std::array<float, 3> mean = {mean_r, mean_g, mean_b};
    std::array<float, 3> stddev = {stddev_r, stddev_g, stddev_b};

    run_depth_estimation(*depth_estimation_tflite_runtime, input_array.as_span(),
                         output_array.as_span(), mean, stddev);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_depthcamera_NativeLib_depthColormap(JNIEnv *env, jobject /*thiz*/,
                                                     jfloatArray depth_values,
                                                     jintArray colormapped_pixels) {
    NativeFloatArrayScope depth_value_array(env, depth_values);
    NativeIntArrayScope colormapped_pixel_array(env, colormapped_pixels);

    if (depth_value_array.size() == colormapped_pixel_array.size()) {
        depth_colormap(depth_value_array.as_span(), colormapped_pixel_array.as_span());
    } else {
        LOG_ERROR("depth and colormapped pixel array should have the same length! ({} and {})",
                  depth_value_array.size(), colormapped_pixel_array.size());
    }
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_depthcamera_NativeLib_bitmapToRgbFloatArray(JNIEnv *env, jobject /*thiz*/,
                                                             jobject bitmap,
                                                             jfloatArray out_float_array) {

    NativeFloatArrayScope out_float_array_scope(env, out_float_array);
    bitmap_to_rgb_float_array(env, bitmap, out_float_array_scope.as_span());
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_depthcamera_NativeLib_imageBytesToArgbIntArray(JNIEnv *env, jobject /*thiz*/,
                                                                jbyteArray image_bytes,
                                                                jintArray out_int_array) {

    NativeByteArrayScope image_byte_array(env, image_bytes);
    NativeIntArrayScope out_int_array_scope(env, out_int_array);

    image_bytes_to_argb_int_array(image_byte_array.as_span(), out_int_array_scope.as_span());
}