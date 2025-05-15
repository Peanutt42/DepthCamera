package com.example.depthcamera

import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.PixelFormat
import android.media.Image
import android.util.Size
import com.example.depthcamera.performance.PerformanceInfo

/** Kotlin interface with NativeLib c++ code */
object NativeLib {
	init {
		System.loadLibrary("NativeLib")
	}

	external fun initDepthTfLiteRuntime(
		model: ByteArray,
		gpuDelegateSerializationDir: String,
		modelToken: String
	)

	external fun shutdownDepthTfLiteRuntime()

	external fun runDepthInference(
		input: FloatArray,
		output: FloatArray,
		meanR: Float,
		meanG: Float,
		meanB: Float,
		stddevR: Float,
		stddevG: Float,
		stddevB: Float
	)


	external fun depthColormap(depthValues: FloatArray, colormappedPixels: IntArray)

	external fun bitmapToRgbFloatArray(bitmap: Bitmap, outFloatArray: FloatArray)

	external fun imageBytesToArgbIntArray(imageBytes: ByteArray, outIntArray: IntArray)


	/** @param input values should be between 0.0f and 1.0f */
	fun depthColorMap(input: FloatArray, inputImageSize: Size): Bitmap =
		PerformanceInfo.measureDepthScope("depthColorMap") {
			val colormappedPixels = IntArray(input.size)

			depthColormap(input, colormappedPixels)

			return@measureDepthScope Bitmap.createBitmap(
				colormappedPixels,
				inputImageSize.width,
				inputImageSize.height,
				Bitmap.Config.ARGB_8888
			)
		}

	fun bitmapToFloatArray(bitmap: Bitmap): FloatArray =
		PerformanceInfo.measureDepthScope("bitmapToFloatArray") {
			val floatArray = FloatArray(bitmap.width * bitmap.height * 3)

			bitmapToRgbFloatArray(bitmap, floatArray)

			return@measureDepthScope floatArray
		}

	fun imageToBitmap(image: Image, rotationDegrees: Float): Bitmap =
		PerformanceInfo.measureCameraScope("imageToBitmap") {
			require(image.format == PixelFormat.RGBA_8888)

			val pixelBuffer = image.planes[0].buffer
			val pixelBytes = ByteArray(pixelBuffer.remaining())
			pixelBuffer.get(pixelBytes)
			require(pixelBytes.size == image.width * image.height * 4)

			val pixels = IntArray(image.width * image.height)

			imageBytesToArgbIntArray(pixelBytes, pixels)

			return@measureCameraScope rotateBitmap(
				Bitmap.createBitmap(
					pixels,
					image.width,
					image.height,
					Bitmap.Config.ARGB_8888
				),
				rotationDegrees
			)
		}

	fun rotateBitmap(bitmap: Bitmap, rotationDegrees: Float): Bitmap =
		PerformanceInfo.measureCameraScope("rotateBitmap") {
			Bitmap.createBitmap(
				bitmap,
				0,
				0,
				bitmap.width,
				bitmap.height,
				Matrix().apply { postRotate(rotationDegrees) },
				false
			)
		}
}