#include "ImageUtils.hpp"
#include "Log.hpp"
#include "PerformanceScope.hpp"

void check_android_bitmap_result(int result) {
	if (result == ANDROID_BITMAP_RESULT_SUCCESS)
		return;

	switch (result) {
	case ANDROID_BITMAP_RESULT_BAD_PARAMETER:
		LOG_ERROR("Android Bitmap error: Bad Parameter");
		break;
	case ANDROID_BITMAP_RESULT_JNI_EXCEPTION:
		LOG_ERROR("Android Bitmap error: JNI Exception");
		break;
	case ANDROID_BITMAP_RESULT_ALLOCATION_FAILED:
		LOG_ERROR("Android Bitmap error: Allocation failed");
		break;
	default:
		LOG_ERROR("Android Bitmap error: Unknown code: {}", result);
		break;
	}
}

void bitmap_to_rgb_float_array(
	JNIEnv* env,
	jobject bitmap,
	std::span<float> out_float_array
) {
	PROFILE_FUNCTION()

	AndroidBitmapInfo info;
	check_android_bitmap_result(AndroidBitmap_getInfo(env, bitmap, &info));

	if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
		LOG_ERROR("Bitmap format is not RGBA 8888!");
		return;
	}

	if (out_float_array.size() < info.width * info.height * 3) {
		LOG_ERROR("out_float_array is too small for this bitmap!");
		return;
	}

	void* addressPtr = nullptr;
	check_android_bitmap_result(
		AndroidBitmap_lockPixels(env, bitmap, &addressPtr)
	);
	// RGBA 8888 -> one int for each pixel
	int* pixelPtr = (int*)addressPtr;

	size_t i = 0;
	size_t j = 0;
	for (; i < info.width * info.height; i++) {
		int pixel_color = pixelPtr[i];
		out_float_array[j++] = (float)(pixel_color >> 16 & 255);
		out_float_array[j++] = (float)(pixel_color >> 8 & 255);
		out_float_array[j++] = (float)(pixel_color & 255);
	}

	check_android_bitmap_result(AndroidBitmap_unlockPixels(env, bitmap));
}

void image_bytes_to_argb_int_array(
	std::span<const jbyte> image_bytes,
	std::span<jint> out_pixels
) {
	PROFILE_FUNCTION()

	if (image_bytes.size_bytes() != out_pixels.size_bytes()) {
		LOG_ERROR("image_bytes and out_pixels do not have the same size!");
		return;
	}

	size_t i = 0;
	size_t j = 0;
	for (; i < out_pixels.size(); i++) {
		auto r = image_bytes[j++];
		auto g = image_bytes[j++];
		auto b = image_bytes[j++];
		auto a = image_bytes[j++];
		out_pixels[i] = color_argb(a, r, g, b);
	}
}