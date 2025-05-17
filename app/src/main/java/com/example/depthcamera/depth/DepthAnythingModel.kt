package com.example.depthcamera.depth

import android.content.Context
import android.graphics.Bitmap
import android.util.Size
import androidx.core.graphics.scale
import com.example.depthcamera.NativeLib

class DepthAnythingModel(context: Context) : DepthModel {
	private companion object {
		private const val MODEL_NAME = "depth_anything_v2_vits_210x210.onnx"
		private const val INPUT_IMAGE_DIM = 210
		private val INPUT_IMAGE_SIZE = Size(INPUT_IMAGE_DIM, INPUT_IMAGE_DIM)
		private val NORM_MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
		private val NORM_STD = floatArrayOf(0.229f, 0.224f, 0.225f)
	}

	init {
		val modelData = context.assets.open(MODEL_NAME).readBytes()

		NativeLib.initDepthOnnxRuntime(modelData)
	}

	override fun close() {
		NativeLib.shutdownDepthOnnxRuntime()
	}

	override fun getName(): String = MODEL_NAME

	override fun getInputSize(): Size = INPUT_IMAGE_SIZE

	override fun predictDepth(input: Bitmap): FloatArray {
		val scaled = input.scale(INPUT_IMAGE_DIM, INPUT_IMAGE_DIM)
		val input = NativeLib.bitmapToRgbChwFloatArray(scaled)
		var output = FloatArray(INPUT_IMAGE_DIM * INPUT_IMAGE_DIM)

		NativeLib.runDepthOnnxInference(
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