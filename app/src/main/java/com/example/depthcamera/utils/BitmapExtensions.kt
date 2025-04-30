package com.example.depthcamera.utils

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.Image
import com.example.depthcamera.performance.PerformanceScope
import java.io.ByteArrayOutputStream

fun Image.toBitmap(rotationDegrees: Int = 0): Bitmap {
	val performanceScope = PerformanceScope("Image::toBitmap")

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

	val result =
		BitmapFactory.decodeByteArray(yuv, 0, yuv.size).rotateBitmap(rotationDegrees.toFloat())

	performanceScope.finish()

	return result
}

fun Bitmap.rotateBitmap(rotationDegrees: Float = 0.0f): Bitmap {
	val matrix = Matrix()
	matrix.postRotate(rotationDegrees)
	return Bitmap.createBitmap(this, 0, 0, this.width, this.height, matrix, false)
}
