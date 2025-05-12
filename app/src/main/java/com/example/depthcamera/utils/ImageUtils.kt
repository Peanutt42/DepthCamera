package com.example.depthcamera.utils

import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.PixelFormat
import android.media.Image
import com.example.depthcamera.performance.PerformanceInfo

/** requires [format] to be [PixelFormat.RGBA_8888] */
fun Image.toBitmap(rotationDegrees: Float): Bitmap =
	PerformanceInfo.measureCameraScope("Image::toBitmap") {
		require(this.format == PixelFormat.RGBA_8888)

		val pixelBuffer = this.planes[0].buffer
		val pixelBytes = ByteArray(pixelBuffer.remaining())
		pixelBuffer.get(pixelBytes)
		require(pixelBytes.size == this.width * this.height * 4)
		val pixels = IntArray(this.width * this.height)
		for (i in pixels.indices) {
			val r = pixelBytes[i * 4].toInt()
			val g = pixelBytes[i * 4 + 1].toInt()
			val b = pixelBytes[i * 4 + 2].toInt()
			val a = pixelBytes[i * 4 + 3].toInt()

			pixels[i] = Color.argb(a, r, g, b)
		}

		return@measureCameraScope Bitmap.createBitmap(
			pixels,
			this.width,
			this.height,
			Bitmap.Config.ARGB_8888
		).rotateBitmap(rotationDegrees)
	}

fun Bitmap.rotateBitmap(rotationDegrees: Float): Bitmap =
	PerformanceInfo.measureCameraScope("Bitmap::rotateBitmap") {
		Bitmap.createBitmap(
			this,
			0,
			0,
			this.width,
			this.height,
			Matrix().apply { postRotate(rotationDegrees) },
			false
		)
	}