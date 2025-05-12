package com.example.depthcamera.performance

import com.example.depthcamera.utils.getCurrentTime
import kotlin.time.Duration
import java.util.Collections
import kotlin.time.DurationUnit

class PerformanceScope(
	val name: String,
	val start: Duration,
	val duration: Duration,
	/** how deep are we in nested scopes */
	val depth: Int,
) {
	fun formatted(): String {
		return "${paddingTabs(depth.coerceAtLeast(0))}$name: ${duration.toString(DurationUnit.MILLISECONDS, 2)}"
	}
}

/**
 * A PerformanceFrame collects all instrumented performance scopes until [finish] is called when a new frame begins.
 * If [end] is Null, the frame is not finished yet -> [finish] should be called.
 * The PerformanceFrame can be called by multiple threads simultaneously, though it will block while reading/writing (this is needed, since it will be called from coroutines as well)
 */
class PerformanceFrame(
	val name: String,
	private var currentFrameScopes: MutableCollection<PerformanceScope> = Collections.synchronizedList(
		mutableListOf<PerformanceScope>()
	),
	/**
	 * inside of how many nested scopes are we right now?
	 * gets incremented when a scope starts and decremented when a scope ends
	 */
	private var currentFrameScopeDepth: Int = 0,
	private var start: Duration = getCurrentTime(),
	private var end: Duration? = null
) {

	/** resets any information other than the name, marking a new frame has begun, and returning the finished frame */
	fun finish(): PerformanceFrame {
		var finishedFrame = PerformanceFrame(
			name,
			currentFrameScopes.toMutableList(),
			currentFrameScopeDepth,
			start,
			getCurrentTime()
		)
		currentFrameScopeDepth = 0
		currentFrameScopes.clear()
		start = getCurrentTime()
		end = null
		return finishedFrame
	}

	/** @return nested scope depth */
	fun startScope(): Int {
		val scopeDepth = currentFrameScopeDepth
		currentFrameScopeDepth++
		return scopeDepth
	}

	/** @return duration of the scope */
	fun endScope(name: String, scopeDepth: Int, start: Duration): Duration {
		val duration = getCurrentTime() - start
		currentFrameScopeDepth--
		currentFrameScopes.add(PerformanceScope(name, start, duration, scopeDepth))
		return duration
	}

	fun formatted(): String {
		val formattedScopes = currentFrameScopes
			.sortedBy { scope -> scope.start }
			.joinToString("\n") { scope -> "    " + scope.formatted() }
		val end = (end ?: getCurrentTime())
		val duration = end - start
		val fps = 1.0 / duration.toDouble(DurationUnit.SECONDS)
		return "$name Frame: ${"%.2f".format(fps)} fps (${duration.toString(DurationUnit.MILLISECONDS, 2)})\n$formattedScopes"
	}
}

private fun paddingTabs(amount: Int): String {
	var stringBuilder = StringBuilder(amount * 4)
	var i = 0
	while (i < amount) {
		stringBuilder.append("    ")
		i++
	}
	return stringBuilder.toString()
}