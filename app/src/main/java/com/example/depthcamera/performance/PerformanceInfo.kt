package com.example.depthcamera.performance

import android.util.Log

/**
 * Global helper class that collects all performance information
 * [formatted] is used to generate the performance info text overlay
 */
object PerformanceInfo {
	const val PERFORMANCE_SCOPE_TAG = "Performance Scope"

	/** key: scope name, value: time in millis */
	var latestPerformanceScopes = HashMap<String, Long>()

	var depthFrameMillis = 0L
	private var lastDepthFrameMillis = System.currentTimeMillis()

	var lastCameraFrameDurationMillis = 0L
	private var lastCameraFrameMillis = System.currentTimeMillis()

	fun newDepthFrame() {
		val now = System.currentTimeMillis()
		depthFrameMillis = now - lastDepthFrameMillis
		lastDepthFrameMillis = now
	}

	fun newCameraFrame() {
		val now = System.currentTimeMillis()
		lastCameraFrameDurationMillis = now - lastCameraFrameMillis
		lastCameraFrameMillis = now
	}

	/** @sample measureScopeSample */
	inline fun <T> measureScope(name: String, crossinline fn: () -> T): T {
		val startMillis = System.currentTimeMillis()
		val result = fn()
		val durationMillis = System.currentTimeMillis() - startMillis

		Log.i(PERFORMANCE_SCOPE_TAG, "$name took $durationMillis ms")
		latestPerformanceScopes[name] = durationMillis

		return result
	}

	fun formatted(): String {
		val depthFps = 1.0 / (depthFrameMillis * 0.001)
		val formattedDepthFps = formatDecimal(depthFps)
		val cameraFps = 1.0 / (lastCameraFrameDurationMillis * 0.001)
		val formattedCameraFps = formatDecimal(cameraFps)
		// sorted, so that the slowest scope is at the top
		val latestPerformanceScopesFormatted =
			latestPerformanceScopes
				.map { it.key to it.value }
				.sortedByDescending { it.second }
				.joinToString("\n") { "${it.first}: ${it.second} ms" }
		return "Depth estimations per second: $formattedDepthFps ($depthFrameMillis ms)\n" +
			"Camera FPS: $formattedCameraFps ($lastCameraFrameDurationMillis ms)\n\n" +
			latestPerformanceScopesFormatted
	}
}

private fun formatDecimal(value: Double): String {
	return "%.2f".format(value)
}

@Suppress("UNUSED_FUNCTION")
private fun measureScopeSample() {
	PerformanceInfo.measureScope("scope name") {
		// call any function in this scope to measure its performance
	}
}
