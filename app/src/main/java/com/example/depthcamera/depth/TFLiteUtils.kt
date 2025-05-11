package com.example.depthcamera.depth

import android.content.Context
import android.graphics.PointF
import android.util.Log
import com.example.depthcamera.DepthCameraApp
import com.example.depthcamera.performance.PerformanceInfo
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.TensorOperator
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageOperator
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import org.tensorflow.lite.support.tensorbuffer.TensorBufferFloat
import java.nio.MappedByteBuffer

/** Helper function to create a TFLite Interpreter with GPU Delegate support (even caching) */
fun createTFLiteInterpreter(context: Context, modelBytes: MappedByteBuffer, modelName: String): Interpreter {
	val interpreterOptions =
		Interpreter.Options().apply {
			val compatibilityList = CompatibilityList()
			if (compatibilityList.isDelegateSupportedOnThisDevice) {
				val gpuDelegateCacheDirectory =
					createSerializedGpuDelegateCacheDirectory(context)
				val modelToken = getModelToken(context, modelName)

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

				val gpuDelegateOptions = compatibilityList.bestOptionsForThisDevice
					.setSerializationParams(gpuDelegateCacheDirectory.path, modelToken)
				this.addDelegate(GpuDelegate(gpuDelegateOptions))
			} else {
				this.numThreads = 8
			}
		}
	return Interpreter(modelBytes, interpreterOptions)
}

/** added performance scope tracking when applying operator */
class ResizeOpWrapper(
	targetHeight: Int,
	targetWidth: Int,
	resizeMethod: ResizeOp.ResizeMethod
) : ImageOperator {
	private var implementation = ResizeOp(targetHeight, targetWidth, resizeMethod)

	override fun apply(image: TensorImage): TensorImage =
		PerformanceInfo.measureDepthScope("ResizeOp") {
			implementation.apply(image)
		}

	override fun inverseTransform(
		point: PointF?,
		inputImageHeight: Int,
		inputImageWidth: Int
	): PointF? = implementation.inverseTransform(point, inputImageHeight, inputImageWidth)

	override fun getOutputImageHeight(inputImageHeight: Int, inputImageWidth: Int): Int =
		implementation.getOutputImageHeight(inputImageHeight, inputImageWidth)

	override fun getOutputImageWidth(inputImageHeight: Int, inputImageWidth: Int): Int =
		implementation.getOutputImageWidth(inputImageHeight, inputImageWidth)
}

/** added performance scope tracking when applying operator */
class NormalizeOpWrapper(mean: FloatArray, stddev: FloatArray) : TensorOperator {
	private var implementation = NormalizeOp(mean, stddev)

	override fun apply(input: TensorBuffer): TensorBuffer =
		PerformanceInfo.measureDepthScope("NormalizeOp") {
			implementation.apply(input)
		}
}

/** scales raw model output from [min;max] output range to [0.0;1.0] */
class MinMaxScalingOp : TensorOperator {
	override fun apply(input: TensorBuffer): TensorBuffer =
		PerformanceInfo.measureDepthScope("MinMaxScalingOp") {
			val values = input.floatArray
			val max = values.maxOrNull()!!
			val min = values.minOrNull()!!
			val maxMinDiff = max - min
			if (maxMinDiff > Float.MIN_VALUE) {
				for (i in values.indices) {
					var p = ((values[i] - min) / maxMinDiff)
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
			return@measureDepthScope output
		}
}