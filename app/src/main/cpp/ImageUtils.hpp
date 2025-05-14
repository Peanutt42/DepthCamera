#pragma once

#include <android/bitmap.h>
#include <span>
#include "Log.hpp"
#include "PerformanceScope.hpp"

/// argb 8888 formatted
constexpr inline int color_argb(int a, int r, int g, int b) {
    return (a << 24) | (r << 16) | (g << 8) | b;
}

/// argb 8888 formatted
constexpr inline int color_rgb(int r, int g, int b) {
    return color_argb(255, r, g, b);
}

void check_android_bitmap_result(int result);

/// writes the 3 rgb channel components into the float array, each rgb float channel is between 0.0f and 255.0f
void bitmap_to_rgb_float_array(JNIEnv *env, jobject bitmap, std::span<float> out_float_array);

/// image_bytes should have 4 bytes (4 argb channels) for each pixel
void
image_bytes_to_argb_int_array(std::span<const jbyte> image_bytes, std::span<jint> out_pixels);