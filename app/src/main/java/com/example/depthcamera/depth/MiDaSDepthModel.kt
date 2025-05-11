package com.example.depthcamera.depth

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import android.util.Size
import com.example.depthcamera.DepthCameraApp
import com.example.depthcamera.performance.PerformanceInfo
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.TensorProcessor
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import org.tensorflow.lite.support.tensorbuffer.TensorBufferFloat

class MiDaSDepthModel(context: Context) : DepthModel {
	companion object {
		private const val MODEL_NAME = "lite-model_midas_v2_1_small_1_lite_1.tflite"
		const val INPUT_IMAGE_DIM = 256
		val INPUT_IMAGE_SIZE = Size(INPUT_IMAGE_DIM, INPUT_IMAGE_DIM)
		private val NORM_MEAN = floatArrayOf(123.675f, 116.28f, 103.53f)
		private val NORM_STD = floatArrayOf(58.395f, 57.12f, 57.375f)
	}

	private var interpreter: Interpreter

	private val inputTensorProcessor =
		ImageProcessor.Builder()
			.add(ResizeOpWrapper(INPUT_IMAGE_DIM, INPUT_IMAGE_DIM, ResizeOp.ResizeMethod.BILINEAR))
			.add(NormalizeOpWrapper(NORM_MEAN, NORM_STD))
			.build()

	private val outputTensorProcessor = TensorProcessor.Builder()
		.add(MinMaxScalingOp())
		.build()

	init {
		val modelBytes = FileUtil.loadMappedFile(context, MODEL_NAME)

		interpreter = PerformanceInfo.measureDepthScope("Load TFLite interpreter") {
			createTFLiteInterpreter(context, modelBytes, MODEL_NAME)
		}
	}

	override fun getInputSize(): Size = INPUT_IMAGE_SIZE

	override fun predictDepth(input: Bitmap): FloatArray {
		Log.i(DepthCameraApp.APP_LOG_TAG, "Input resolution: ${input.width} X ${input.height}")

		lateinit var inputTensor: TensorImage
		lateinit var outputTensor: TensorBuffer
		PerformanceInfo.measureDepthScope("Loading input") {
			inputTensor = TensorImage.fromBitmap(input)
			outputTensor = TensorBufferFloat.createFixedSize(
				intArrayOf(INPUT_IMAGE_DIM, INPUT_IMAGE_DIM, 1),
				DataType.FLOAT32
			)
		}

		PerformanceInfo.measureDepthScope("Inference") {
			inputTensor = inputTensorProcessor.process(inputTensor)

			interpreter.run(inputTensor.buffer, outputTensor.buffer)
		}

		val output = PerformanceInfo.measureDepthScope("Postprocessing") {
			outputTensor = outputTensorProcessor.process(outputTensor)
			outputTensor.floatArray
		}

		return output
	}
}