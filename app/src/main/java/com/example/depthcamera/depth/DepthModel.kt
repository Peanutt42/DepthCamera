package com.example.depthcamera.depth

import android.graphics.Bitmap
import android.util.Size

data class DepthPredictionResult(var output: FloatArray, var inferenceTimeMillis: Long)

abstract class DepthModel {
	abstract fun predictDepth(input: Bitmap): DepthPredictionResult

	abstract fun getInputSize(): Size
}
