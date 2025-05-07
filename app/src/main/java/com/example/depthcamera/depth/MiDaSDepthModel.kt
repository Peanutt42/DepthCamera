package com.example.depthcamera.depth

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import android.util.Size
import com.example.depthcamera.DepthCameraApp
import com.example.depthcamera.performance.PerformanceInfo
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.TensorOperator
import org.tensorflow.lite.support.common.TensorProcessor
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import org.tensorflow.lite.support.tensorbuffer.TensorBufferFloat

class MiDaSDepthModel(context: Context) : DepthModel() {
	companion object {
		private const val MODEL_NAME = "lite-model_midas_v2_1_small_1_lite_1.tflite"
		const val INPUT_IMAGE_DIM = 256
		val INPUT_IMAGE_SIZE = Size(INPUT_IMAGE_DIM, INPUT_IMAGE_DIM)
		private const val NUM_THREADS = 8
		private val NORM_MEAN = floatArrayOf(123.675f, 116.28f, 103.53f)
		private val NORM_STD = floatArrayOf(58.395f, 57.12f, 57.375f)
	}

	private var interpreter: Interpreter

	private val inputTensorProcessor =
		ImageProcessor.Builder()
			.add(ResizeOp(INPUT_IMAGE_DIM, INPUT_IMAGE_DIM, ResizeOp.ResizeMethod.BILINEAR))
			.add(NormalizeOp(NORM_MEAN, NORM_STD))
			.build()

	private val outputTensorProcessor = TensorProcessor.Builder().add(DepthScalingOp()).build()

	init {
		val modelBytes = FileUtil.loadMappedFile(context, MODEL_NAME)

		interpreter = PerformanceInfo.measureScope("Load TFLite interpreter") {
			val interpreterOptions =
				Interpreter.Options().apply {
					val compatibilityList = CompatibilityList()
					if (compatibilityList.isDelegateSupportedOnThisDevice) {
						val gpuDelegateCacheDirectory =
							createSerializedGpuDelegateCacheDirectory(context)
						val modelToken = getModelToken(context, MODEL_NAME)
						val gpuDelegateOptions = compatibilityList.bestOptionsForThisDevice
							.setSerializationParams(gpuDelegateCacheDirectory.path, modelToken)
						this.addDelegate(GpuDelegate(gpuDelegateOptions))
					} else {
						this.numThreads = NUM_THREADS
					}
				}
			return@measureScope Interpreter(modelBytes, interpreterOptions)
		}
	}

	override fun predictDepth(input: Bitmap): FloatArray {
		Log.i(DepthCameraApp.APP_LOG_TAG, "Input resolution: ${input.width} X ${input.height}")

		var inputTensor = TensorImage.fromBitmap(input)

		var outputTensor =
			TensorBufferFloat.createFixedSize(
				intArrayOf(INPUT_IMAGE_DIM, INPUT_IMAGE_DIM, 1),
				DataType.FLOAT32
			)

		PerformanceInfo.measureScope("Inference") {
			inputTensor = inputTensorProcessor.process(inputTensor)

			interpreter.run(inputTensor.buffer, outputTensor.buffer)

			outputTensor = outputTensorProcessor.process(outputTensor)
		}

		return outputTensor.floatArray
	}

	override fun getInputSize(): Size {
		return INPUT_IMAGE_SIZE
	}

	/** scales raw model output to relative 0.0f for min and 1.0f for max output value */
	class DepthScalingOp : TensorOperator {
		override fun apply(input: TensorBuffer): TensorBuffer {
			val values = input.floatArray
			val max = values.maxOrNull()!!
			val min = values.minOrNull()!!
			if (max - min > Float.MIN_VALUE) {
				for (i in values.indices) {
					var p = ((values[i] - min) / (max - min))
					if (p < 0) {
						p += 1.0f
					}
					values[i] = p
				}
			} else {
				for (i in values.indices) {
					values[i] = 0.0f
				}
			}

			val output = TensorBufferFloat.createFrom(input, DataType.FLOAT32)
			output.loadArray(values)
			return output
		}
	}
}