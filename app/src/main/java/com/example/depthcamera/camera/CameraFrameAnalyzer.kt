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
					val predictionOutput = depthModel.predictDepth(frame)
					PerformanceInfo.newFrame()

					withContext(Dispatchers.Main) {
						draw(
							depthColorMap(
								predictionOutput,
								depthModel.getInputSize()
							)
						)

						performanceText.text = PerformanceInfo.formatted()
					}
				}
			}
		}
	}

	@OptIn(ExperimentalGetImage::class)
	override fun analyze(image: ImageProxy) {
		if (image.image != null) {
			PerformanceInfo.measureScope("Camera frame acquire") {
				latestCameraFrame.set(image.image!!.toBitmap(image.imageInfo.rotationDegrees))
			}

			PerformanceInfo.newCameraFrame()
		}
		image.close()
	}

	private fun draw(image: Bitmap) {
		depthView.setImageBitmap(image)
	}
}
