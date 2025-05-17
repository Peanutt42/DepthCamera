#include <android/log.h>
#include <jni.h>
#include <memory>

#include "DepthEstimation.hpp"
#include "onnx/OnnxRuntime.hpp"
#include "tflite/TfLiteRuntime.hpp"
#include "utils/ImageUtils.hpp"
#include "utils/Log.hpp"
#include "utils/NativeJavaScopes.hpp"
#include "utils/Profiling.hpp"

static std::unique_ptr<TfLiteRuntime> depth_estimation_tflite_runtime = nullptr;

static std::unique_ptr<OnnxRuntime> depth_estimation_onnx_runtime = nullptr;

extern "C" JNIEXPORT void JNICALL
Java_com_example_depthcamera_NativeLib_initDepthTfLiteRuntime(
	JNIEnv* env,
	jobject /*thiz*/,
	jbyteArray model,
	jstring gpu_delegate_serialization_dir,
	jstring model_token
) {

	NativeByteArrayScope model_data(env, model);
	NativeStringScope gpu_delegate_serialization_dir_string(
		env, gpu_delegate_serialization_dir
	);
	NativeStringScope model_token_string(env, model_token);

	depth_estimation_tflite_runtime = std::make_unique<TfLiteRuntime>(
		model_data.as_span(), gpu_delegate_serialization_dir_string.get(),
		model_token_string.get()
	);
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_depthcamera_NativeLib_shutdownDepthTfLiteRuntime(
	JNIEnv* /*env*/,
	jobject /*thiz*/
) {
	depth_estimation_tflite_runtime.reset(nullptr);
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_depthcamera_NativeLib_runDepthTfLiteInference(
	JNIEnv* env,
	jobject /*thiz*/,
	jfloatArray input,
	jfloatArray output,
	jfloat mean_r,
	jfloat mean_g,
	jfloat mean_b,
	jfloat stddev_r,
	jfloat stddev_g,
	jfloat stddev_b
) {
	if (depth_estimation_tflite_runtime == nullptr) {
		LOG_ERROR("TfLiteRuntime not initialized!");
		return;
	}

	NativeFloatArrayScope input_array(env, input);
	NativeFloatArrayScope output_array(env, output);

	std::array<float, 3> mean = {mean_r, mean_g, mean_b};
	std::array<float, 3> stddev = {stddev_r, stddev_g, stddev_b};

	run_depth_estimation(
		*depth_estimation_tflite_runtime, input_array.as_span(),
		output_array.as_span(), mean, stddev
	);
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_depthcamera_NativeLib_initDepthOnnxRuntime(
	JNIEnv* env,
	jobject /*thiz*/,
	jbyteArray model
) {
	NativeByteArrayScope model_data(env, model);

	depth_estimation_onnx_runtime =
		std::make_unique<OnnxRuntime>(std::as_bytes(model_data.as_span()));
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_depthcamera_NativeLib_shutdownDepthOnnxRuntime(
	JNIEnv* /*env*/,
	jobject /*thiz*/
) {
	depth_estimation_onnx_runtime.reset(nullptr);
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_depthcamera_NativeLib_runDepthOnnxInference(
	JNIEnv* env,
	jobject /*thiz*/,
	jfloatArray input_data,
	jfloatArray output_data,
	jfloat mean_r,
	jfloat mean_g,
	jfloat mean_b,
	jfloat stddev_r,
	jfloat stddev_g,
	jfloat stddev_b
) {
	if (depth_estimation_onnx_runtime == nullptr) {
		LOG_ERROR("OnnxRuntime not initialized!");
		return;
	}

	NativeFloatArrayScope input_array(env, input_data);
	NativeFloatArrayScope output_array(env, output_data);

	std::array<float, 3> mean = {mean_r, mean_g, mean_b};
	std::array<float, 3> stddev = {stddev_r, stddev_g, stddev_b};

	run_depth_estimation(
		*depth_estimation_onnx_runtime, input_array.as_span(),
		output_array.as_span(), mean, stddev
	);
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_depthcamera_NativeLib_depthColormap(
	JNIEnv* env,
	jobject /*thiz*/,
	jfloatArray depth_values,
	jintArray colormapped_pixels
) {
	NativeFloatArrayScope depth_value_array(env, depth_values);
	NativeIntArrayScope colormapped_pixel_array(env, colormapped_pixels);

	if (depth_value_array.size() == colormapped_pixel_array.size()) {
		depth_colormap(
			depth_value_array.as_span(), colormapped_pixel_array.as_span()
		);
	} else {
		LOG_ERROR(
			"depth and colormapped pixel array should have the same length! "
			"({} and {})",
			depth_value_array.size(), colormapped_pixel_array.size()
		);
	}
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_depthcamera_NativeLib_bitmapToRgbChwFloatArray(
	JNIEnv* env,
	jobject /*thiz*/,
	jobject bitmap,
	jfloatArray out_float_array
) {

	NativeFloatArrayScope out_float_array_scope(env, out_float_array);
	bitmap_to_rgb_chw_float_array(env, bitmap, out_float_array_scope.as_span());
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_depthcamera_NativeLib_bitmapToRgbHwc255FloatArray(
	JNIEnv* env,
	jobject /*thiz*/,
	jobject bitmap,
	jfloatArray out_float_array
) {

	NativeFloatArrayScope out_float_array_scope(env, out_float_array);
	bitmap_to_rgb_hwc_255_float_array(
		env, bitmap, out_float_array_scope.as_span()
	);
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_depthcamera_NativeLib_imageBytesToArgbIntArray(
	JNIEnv* env,
	jobject /*thiz*/,
	jbyteArray image_bytes,
	jintArray out_int_array
) {

	NativeByteArrayScope image_byte_array(env, image_bytes);
	NativeIntArrayScope out_int_array_scope(env, out_int_array);

	image_bytes_to_argb_int_array(
		image_byte_array.as_span(), out_int_array_scope.as_span()
	);
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_depthcamera_NativeLib_newDepthFrame(
	JNIEnv* /*env*/,
	jobject /*this*/
) {
	get_last_depth_profiling_frame_formatted() =
		get_depth_profiling_frame().finish();
}
extern "C" JNIEXPORT jstring JNICALL
Java_com_example_depthcamera_NativeLib_formatDepthFrame(
	JNIEnv* env,
	jobject /*this*/
) {
	return env->NewStringUTF(get_last_depth_profiling_frame_formatted().c_str()
	);
}
extern "C" JNIEXPORT void JNICALL
Java_com_example_depthcamera_NativeLib_newCameraFrame(
	JNIEnv* /*env*/,
	jobject /*this*/
) {
	get_last_camera_profiling_frame_formatted() =
		get_camera_profiling_frame().finish();
}
extern "C" JNIEXPORT jstring JNICALL
Java_com_example_depthcamera_NativeLib_formatCameraFrame(
	JNIEnv* env,
	jobject /*this*/
) {
	return env->NewStringUTF(get_last_camera_profiling_frame_formatted().c_str()
	);
}