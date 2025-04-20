package com.example.depthcamera

import android.app.Application
import android.content.Context
import android.util.Log
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.google.common.util.concurrent.ListenableFuture
import java.util.concurrent.ExecutionException

class DepthCameraApp : Application() {

    lateinit var cameraProviderListenableFuture: ListenableFuture<ProcessCameraProvider>

    lateinit var depthModel: DepthModel

    init {
        instance = this
    }

    companion object {
        const val APP_LOG_TAG = "Depth Camera"

        private lateinit var instance: DepthCameraApp

        fun applicationContext(): Context {
            return instance.applicationContext
        }
    }

    override fun onCreate() {
        super.onCreate()

        instance = this

        depthModel = MiDaSDepthModel()
    }

    fun initCamera(bindCameraPreviewCallback: (cameraProvider: ProcessCameraProvider) -> Unit) {
        cameraProviderListenableFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderListenableFuture.addListener(
            {
                try {
                    val cameraProvider: ProcessCameraProvider =
                        cameraProviderListenableFuture.get()
                    bindCameraPreviewCallback(cameraProvider)
                } catch (e: ExecutionException) {
                    Log.e(APP_LOG_TAG, e.message!!)
                } catch (e: InterruptedException) {
                    Log.e(APP_LOG_TAG, e.message!!)
                }
            },
            ContextCompat.getMainExecutor(this)
        )
    }
}
