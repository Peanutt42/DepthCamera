package com.example.depthcamera.utils

import kotlin.time.Duration
import kotlin.time.Duration.Companion.nanoseconds

/**
 * like [System.nanoTime] but with "type-safe" time unit conversion
 * @return some duration since an arbitrary point in time, only meaningful when looking at the difference in values between two time points
 */
fun getCurrentTime(): Duration {
	return System.nanoTime().nanoseconds
}