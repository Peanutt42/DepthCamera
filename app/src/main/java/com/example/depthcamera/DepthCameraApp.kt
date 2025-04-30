package com.example.depthcamera

import android.app.Application
import com.example.depthcamera.camera.CameraManager
import com.example.depthcamera.depth.DepthModel
import com.example.depthcamera.depth.MiDaSDepthModel

class DepthCameraApp : Application() {
	var cameraManager = CameraManager()
	lateinit var depthModel: DepthModel

	companion object {
		const val APP_LOG_TAG = "Depth Camera"
	}

	override fun onCreate() {
		super.onCreate()

		depthModel = MiDaSDepthModel(this)
	}
}
