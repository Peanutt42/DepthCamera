package com.example.depthcamera

import android.app.Application
import com.example.depthcamera.camera.CameraManager
import com.example.depthcamera.depth.DepthModel
import com.example.depthcamera.depth.DepthAnythingModel
import com.example.depthcamera.depth.MiDaSDepthModel

/**
 * App class that holds everything that should persist when switching to another app,
 * for example the camera handle and the loaded depth model
 */
class DepthCameraApp : Application() {
	var cameraManager = CameraManager()
	lateinit var depthModel: DepthModel

	companion object {
		const val APP_LOG_TAG = "Depth Camera"
	}

	override fun onCreate() {
		super.onCreate()

		depthModel = MiDaSDepthModel(this)//DepthAnythingModel(this)
	}
}
