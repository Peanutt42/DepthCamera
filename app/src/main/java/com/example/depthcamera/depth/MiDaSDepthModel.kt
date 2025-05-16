package com.example.depthcamera.depth

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import android.util.Size
import androidx.core.graphics.scale
import com.example.depthcamera.DepthCameraApp
import com.example.depthcamera.NativeLib

class MiDaSDepthModel(context: Context) : DepthModel {
	private companion object {
		private const val MODEL_NAME = "midas_v2_1_256x256.tflite"
		private const val INPUT_IMAGE_DIM = 256
		private val INPUT_IMAGE_SIZE = Size(INPUT_IMAGE_DIM, INPUT_IMAGE_DIM)
		private val NORM_MEAN = floatArrayOf(123.675f, 116.28f, 103.53f)
		private val NORM_STD = floatArrayOf(58.395f, 57.12f, 57.375f)
	}

	init {
		val modelData = context.assets.open(MODEL_NAME).readBytes()

		val gpuDelegateCacheDirectory =
			createSerializedGpuDelegateCacheDirectory(context)
		val modelToken = getModelToken(context, MODEL_NAME)

		// cleanup old cached gpu delegate files
		if (gpuDelegateCacheDirectory.exists()) {
			for (file in gpuDelegateCacheDirectory.listFiles()!!) {
				if (!file.name.contains(modelToken)) {
					try {
						Log.i(
							DepthCameraApp.APP_LOG_TAG,
							"Deleting old gpu delegate cache file: ${file.name}"
						)
						file.delete()
					} catch (_: SecurityException) {
					}
				}
			}
		}

		NativeLib.initDepthTfLiteRuntime(
			modelData,
			gpuDelegateCacheDirectory.path,
			modelToken
		)
	}

	override fun close() {
		NativeLib.shutdownDepthTfLiteRuntime()
	}

	override fun getInputSize(): Size = INPUT_IMAGE_SIZE

	override fun predictDepth(input: Bitmap): FloatArray {
		Log.i(DepthCameraApp.APP_LOG_TAG, "Input resolution: ${input.width} X ${input.height}")

		val scaled = input.scale(INPUT_IMAGE_DIM, INPUT_IMAGE_DIM)
		val input = NativeLib.bitmapToRgbHwc255FloatArray(scaled)
		var output = FloatArray(INPUT_IMAGE_DIM * INPUT_IMAGE_DIM)

		NativeLib.runDepthTfLiteInference(
			input,
			output,
			NORM_MEAN[0],
			NORM_MEAN[1],
			NORM_MEAN[2],
			NORM_STD[0],
			NORM_STD[1],
			NORM_STD[2]
		)

		return output
	}
}