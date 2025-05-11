package com.example.depthcamera.utils

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.Image
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import androidx.core.graphics.createBitmap
import com.example.depthcamera.performance.PerformanceInfo

fun Image.toBitmap(rotationDegrees: Int = 0): Bitmap =
	PerformanceInfo.measureCameraScope("Image::toBitmap") {
		val yBuffer = this.planes[0].buffer
		val uBuffer = this.planes[1].buffer
		val vBuffer = this.planes[2].buffer
		val ySize = yBuffer.remaining()
		val uSize = uBuffer.remaining()
		val vSize = vBuffer.remaining()
		val nv21 = ByteArray(ySize + uSize + vSize)
		yBuffer.get(nv21, 0, ySize)
		vBuffer.get(nv21, ySize, vSize)
		uBuffer.get(nv21, ySize + vSize, uSize)
		val yuvImage = YuvImage(nv21, ImageFormat.NV21, this.width, this.height, null)
		val out = ByteArrayOutputStream()
		yuvImage.compressToJpeg(Rect(0, 0, yuvImage.width, yuvImage.height), 100, out)
		val yuv = out.toByteArray()

		return@measureCameraScope BitmapFactory.decodeByteArray(yuv, 0, yuv.size)
			.rotateBitmap(rotationDegrees.toFloat())
	}

fun Bitmap.rotateBitmap(rotationDegrees: Float): Bitmap = PerformanceInfo.measureCameraScope("Bitmap::rotateBitmap") {
	if (rotationDegrees == 0.0f) {
		return@measureCameraScope this
	}
	val matrix = Matrix()
	matrix.postRotate(rotationDegrees)
	return@measureCameraScope Bitmap.createBitmap(this, 0, 0, this.width, this.height, matrix, false)
}
