package com.example.depthcamera

import android.graphics.Bitmap
import android.widget.ImageView
import android.widget.TextView
import androidx.annotation.OptIn
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.coroutines.Job
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.isActive
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicReference


class CameraFrameAnalyzer(
    private var depthModel: DepthModel,
    private var depthView: ImageView,
    private var performanceText: TextView,
) : ImageAnalysis.Analyzer {

    var showDepth = true

    private var performanceInfo = PerformanceInfo()

    private var processingExecutor = Executors.newSingleThreadExecutor()
    private var processingJob: Job
    private var latestCameraFrame = AtomicReference<Bitmap?>(null)

    init {
        processingJob = CoroutineScope(processingExecutor.asCoroutineDispatcher()).launch {
            while (isActive) {
                val frame = latestCameraFrame.getAndSet(null)

                if (frame != null) {
                    val predictionResult = depthModel.predictDepth(frame)
                    performanceInfo.inferenceMillis = predictionResult.inferenceTimeMillis
                    performanceInfo.newFrame()

                    withContext(Dispatchers.Main) {
                        if (showDepth) {
                            draw(
                                depthColorMap(
                                    predictionResult.output,
                                    depthModel.getInputSize()
                                )
                            )
                        }

                        performanceText.setText(performanceInfo.formatted())
                    }
                }
            }
        }
    }

    @OptIn(ExperimentalGetImage::class)
    override fun analyze(image: ImageProxy) {
        if (image.image != null) {
            var performanceScope = PerformanceScope("acquireCameraImage + Image::toBitmap")

            latestCameraFrame.set(image.image!!.toBitmap(image.imageInfo.rotationDegrees))

            performanceInfo.cameraFrameAcquiringMillis = performanceScope.finish()
            performanceInfo.newCameraFrame()
        }
        image.close()
    }

    private fun draw(image: Bitmap) {
        depthView.setImageBitmap(image)
        depthView.scaleType = ImageView.ScaleType.FIT_CENTER
    }
}