package com.example.depthcamera.performance

class PerformanceInfo {
	private var lastFrameMillis = System.currentTimeMillis()
	private var lastCameraFrameMillis = System.currentTimeMillis()

	var frameMillis = 0L
	var inferenceMillis = 0L
	var cameraFrameAcquiringMillis = 0L
	var lastCameraFrameDurationMillis = 0L

	fun newFrame() {
		val now = System.currentTimeMillis()
		frameMillis = now - lastFrameMillis
		lastFrameMillis = now
	}

	fun newCameraFrame() {
		val now = System.currentTimeMillis()
		lastCameraFrameDurationMillis = now - lastCameraFrameMillis
		lastCameraFrameMillis = now
	}

	fun formatted(): String {
		val fps = 1.0 / (frameMillis * 0.001)
		val formattedFps = formatDecimal(fps)
		val cameraFps = 1.0 / (lastCameraFrameDurationMillis * 0.001)
		val formattedCameraFps = formatDecimal(cameraFps)
		return "FPS (Depth estimations per second): $formattedFps ($frameMillis ms)\n" +
			"Inference: $inferenceMillis ms\n" +
			"Camera FPS: $formattedCameraFps ($lastCameraFrameDurationMillis ms)\n" +
			"Camera frame acquire: $cameraFrameAcquiringMillis ms"
	}
}

private fun formatDecimal(value: Double): String {
	return "%.2f".format(value)
}
