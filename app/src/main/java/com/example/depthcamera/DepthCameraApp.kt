package com.example.depthcamera

import android.app.Application
import android.content.Context

class DepthCameraApp : Application() {
    init {
        instance = this
    }

    companion object {
        private lateinit var instance: DepthCameraApp

        fun applicationContext(): Context {
            return instance.applicationContext
        }
    }

    override fun onCreate() {
        super.onCreate()
        instance = this
    }
}
