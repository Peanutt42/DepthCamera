package com.example.depthcamera.performance

import android.util.Log
import com.example.depthcamera.performance.PerformanceInfo.formatted
import com.example.depthcamera.utils.getCurrentTime
import kotlin.time.DurationUnit


/**
 * Global helper class that collects all performance information
 * [formatted] is used to generate the performance info text overlay
 */
object PerformanceInfo {
	const val DEPTH_PERFORMANCE_SCOPE_TAG = "Depth Performance"
	const val CAMERA_PERFORMANCE_SCOPE_TAG = "Camera Performance"

	var depthPerformanceFrame = PerformanceFrame("Depth")
	/** last finished depth performance frame */
	var lastDepthPerformanceFrame: PerformanceFrame? = null

	var cameraPerformanceFrame = PerformanceFrame("Camera")
	/** last finished camera performance frame */
	var lastCameraPerformanceFrame: PerformanceFrame? = null


	fun newDepthFrame() {
		lastDepthPerformanceFrame = depthPerformanceFrame.finish()
	}

	fun newCameraFrame() {
		lastCameraPerformanceFrame = cameraPerformanceFrame.finish()
	}

	/** @sample measureDepthScopeSample */
	inline fun <T> measureDepthScope(name: String, crossinline fn: () -> T): T {
		val scopeDepth = depthPerformanceFrame.startScope()
		val start = getCurrentTime()
		val result = fn()
		val duration = depthPerformanceFrame.endScope(name, scopeDepth, start)
		Log.i(
			DEPTH_PERFORMANCE_SCOPE_TAG,
			"$name took ${duration.toString(DurationUnit.MILLISECONDS, 2)}"
		)
		return result
	}

	/** @sample measureCameraScopeSample */
	inline fun <T> measureCameraScope(name: String, crossinline fn: () -> T): T {
		val scopeDepth = cameraPerformanceFrame.startScope()
		val start = getCurrentTime()
		val result = fn()
		val duration = cameraPerformanceFrame.endScope(name, scopeDepth, start)
		Log.i(
			CAMERA_PERFORMANCE_SCOPE_TAG,
			"$name took ${duration.toString(DurationUnit.MILLISECONDS, 2)}"
		)
		return result
	}

	fun formatted(): String {
		val depthPerformanceFrameFormatted = lastDepthPerformanceFrame?.formatted()
		val cameraPerformanceFrameFormatted = lastCameraPerformanceFrame?.formatted()

		return "$depthPerformanceFrameFormatted\n\n$cameraPerformanceFrameFormatted"
	}
}


@Suppress("UNUSED_FUNCTION")
private fun measureDepthScopeSample() {
	PerformanceInfo.measureDepthScope("scope name") {
		// call any function in this scope to measure its performance
	}
}

@Suppress("UNUSED_FUNCTION")
private fun measureCameraScopeSample() {
	PerformanceInfo.measureCameraScope("scope name") {
		// call any function in this scope to measure its performance
	}
}