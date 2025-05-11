package com.example.depthcamera.camera

import android.graphics.Bitmap
import android.widget.ImageView
import android.widget.TextView
import androidx.annotation.OptIn
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.example.depthcamera.depth.DepthModel
import com.example.depthcamera.depth.depthColorMap
import com.example.depthcamera.performance.PerformanceInfo
import com.example.depthcamera.utils.toBitmap
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
class CameraFrameAnalyzer(
	private var depthModel: DepthModel,
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
					PerformanceInfo.newDepthFrame()

					val predictionOutput = depthModel.predictDepth(frame)

					withContext(Dispatchers.Main) {
						val colorMappedImage = PerformanceInfo.measureDepthScope("Depth colormap") {
							depthColorMap(
								predictionOutput,
								depthModel.getInputSize()
							)
						}
						depthView.setImageBitmap(colorMappedImage)

						performanceText.text = PerformanceInfo.formatted()
					}
				}
			}
		}
	}

	@OptIn(ExperimentalGetImage::class)
	override fun analyze(image: ImageProxy) {
		if (image.image != null) {
			PerformanceInfo.newCameraFrame()

			val inputBitmap = PerformanceInfo.measureCameraScope("Convert input to bitmap") {
				image.image!!.toBitmap(image.imageInfo.rotationDegrees)
			}

			latestCameraFrame.set(inputBitmap)
		}
		image.close()
	}
}
