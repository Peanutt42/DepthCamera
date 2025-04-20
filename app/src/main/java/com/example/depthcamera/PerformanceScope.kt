package com.example.depthcamera

import android.util.Log

class PerformanceScope(private val name: String) {
    private val start = System.currentTimeMillis()

    companion object {
        const val PERFORMANCE_SCOPE_TAG = "Performance Scope"
    }

    fun getDurationMillis(): Long {
        return System.currentTimeMillis() - start
    }

    // returns: duration in millis
    fun finish(): Long {
        val durationMillis = getDurationMillis()
        Log.i(PERFORMANCE_SCOPE_TAG, "$name took $durationMillis ms")
        return durationMillis
    }
}
