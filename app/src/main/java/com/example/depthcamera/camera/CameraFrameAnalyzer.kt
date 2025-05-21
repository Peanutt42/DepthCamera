package com.example.depthcamera.camera

import android.annotation.SuppressLint
import android.graphics.Bitmap
import android.widget.ImageView
import android.widget.TextView
import androidx.annotation.OptIn
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.example.depthcamera.DepthCameraApp
import com.example.depthcamera.NativeLib
import com.example.depthcamera.depth.DepthModel
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicReference

/**
 * Helper class that analyses the camera feed images in realtime
 */
@SuppressLint("SetTextI18n")
class CameraFrameAnalyzer(
	private var depthCameraApp: DepthCameraApp,
	private var depthView: ImageView,
	private var performanceText: TextView,
) : ImageAnalysis.Analyzer {

	private var processingExecutor = Executors.newSingleThreadExecutor()
	private var latestCameraFrame = AtomicReference<Bitmap?>(null)

	init {
		CoroutineScope(processingExecutor.asCoroutineDispatcher()).launch {
			while (isActive) {
				val frame = latestCameraFrame.getAndSet(null)

				if (frame != null) {
					NativeLib.newDepthFrame()

					val predictionOutput = depthCameraApp.depthModel.predictDepth(frame)

					val inputWidth = frame.width
					val inputHeight = frame.height

					withContext(Dispatchers.Main) {
						val colorMappedImage = NativeLib.depthColorMap(
							predictionOutput,
							depthCameraApp.depthModel.getInputSize()
						)
						depthView.setImageBitmap(colorMappedImage)

						val formattedInputResolution = "${inputWidth}x${inputHeight}"
						val modelName = depthCameraApp.depthModel.getName()
						val modelInputSize = depthCameraApp.depthModel.getInputSize()
						val formattedModelInputSize =
							"${modelInputSize.width}x${modelInputSize.height}"
						performanceText.text =
							"Model: $modelName\nCamera resolution: $formattedInputResolution --> Model input: $formattedModelInputSize\n\n${NativeLib.formatDepthFrame()}\n${NativeLib.formatCameraFrame()}"
					}
				}
			}
		}
	}

	@OptIn(ExperimentalGetImage::class)
	override fun analyze(image: ImageProxy) {
		if (image.image != null) {
			NativeLib.newCameraFrame()

			val inputBitmap =
				NativeLib.imageToBitmap(image.image!!, image.imageInfo.rotationDegrees.toFloat())

			latestCameraFrame.set(inputBitmap)
		}
		image.close()
	}
}
