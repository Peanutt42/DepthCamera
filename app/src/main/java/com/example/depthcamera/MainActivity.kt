package com.example.depthcamera

import android.Manifest
import android.content.ActivityNotFoundException
import android.content.Intent
import android.content.pm.PackageManager
import android.hardware.camera2.CameraCharacteristics
import android.net.Uri
import android.os.Bundle
import android.provider.Settings
import android.util.Log
import android.util.Range
import android.view.View
import android.widget.Button
import android.widget.CheckBox
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.OptIn
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.CameraInfo
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.google.common.util.concurrent.ListenableFuture
import java.util.concurrent.ExecutionException
import java.util.concurrent.Executors
import kotlin.math.atan

class MainActivity : ComponentActivity() {
    companion object {
        const val APP_LOG_TAG = "Depth Camera"
    }

    private lateinit var cameraProviderListenableFuture: ListenableFuture<ProcessCameraProvider>
    private lateinit var cameraFrameAnalyzer: CameraFrameAnalyzer

    private var cameraPreview: Preview? = null
    private var cameraPreviewView: PreviewView? = null
    private var cameraPermissionNotice: LinearLayout? = null
    private var allowCameraPermission: Button? = null

    private var depthPreviewImage: ImageView? = null
    private var depthAnalysisView: ImageAnalysis? = null

    private var performanceText: TextView? = null

    private var showDepthCheckbox: CheckBox? = null

    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted ->
            if (isGranted) {
                initCamera()
            } else {
                cameraPermissionNotice!!.visibility = View.VISIBLE
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        enableEdgeToEdge()
        setContentView(R.layout.activity_main)

        cameraPreviewView = findViewById(R.id.camera_view)
        depthPreviewImage = findViewById(R.id.depth_preview_image)
        performanceText = findViewById(R.id.performance_text)
        cameraPermissionNotice = findViewById(R.id.camera_permission_notice)
        allowCameraPermission = findViewById(R.id.allow_camera_permission_btn)
        allowCameraPermission!!.setOnClickListener { openCameraPermissionSettings() }

        cameraFrameAnalyzer =
            CameraFrameAnalyzer(MiDaSDepthModel(), depthPreviewImage!!, performanceText!!)

        showDepthCheckbox = findViewById(R.id.show_depth)
        showDepthCheckbox!!.isChecked = cameraFrameAnalyzer.showDepth
        showDepthCheckbox!!.setOnClickListener {
            cameraFrameAnalyzer.showDepth = !cameraFrameAnalyzer.showDepth
            showDepthCheckbox!!.isChecked = cameraFrameAnalyzer.showDepth
            depthPreviewImage!!.visibility =
                if (cameraFrameAnalyzer.showDepth) {
                    View.VISIBLE
                } else {
                    View.INVISIBLE
                }
        }
    }

    override fun onResume() {
        super.onResume()

        requestCameraPermission()
        initCamera()
    }

    private fun initCamera() {
        if (cameraPermissionGranted()) {
            cameraPermissionNotice!!.visibility = View.GONE
            cameraProviderListenableFuture = ProcessCameraProvider.getInstance(this)
            cameraProviderListenableFuture.addListener(
                {
                    try {
                        val cameraProvider: ProcessCameraProvider =
                            cameraProviderListenableFuture.get()
                        bindCameraPreview(cameraProvider)
                    } catch (e: ExecutionException) {
                        Log.e(APP_LOG_TAG, e.message!!)
                    } catch (e: InterruptedException) {
                        Log.e(APP_LOG_TAG, e.message!!)
                    }
                },
                ContextCompat.getMainExecutor(this)
            )
        } else {
            cameraPermissionNotice!!.visibility = View.VISIBLE
        }
    }

    private fun requestCameraPermission() {
        if (cameraPermissionGranted()) {
            cameraPermissionNotice!!.visibility = View.GONE
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun openCameraPermissionSettings() {
        if (cameraPermissionGranted()) {
            cameraPermissionNotice!!.visibility = View.GONE
        } else {
            val intent =
                Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS).apply {
                    data =
                        Uri.fromParts(
                            "package",
                            DepthCameraApp.applicationContext().packageName,
                            null
                        )
                    flags = Intent.FLAG_ACTIVITY_NEW_TASK
                }

            try {
                startActivity(intent)
            } catch (_: ActivityNotFoundException) {
                val fallbackIntent =
                    Intent(Settings.ACTION_MANAGE_APPLICATIONS_SETTINGS).apply {
                        flags = Intent.FLAG_ACTIVITY_NEW_TASK
                    }
                startActivity(fallbackIntent)
            }
        }
    }

    private fun cameraPermissionGranted(): Boolean {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
                PackageManager.PERMISSION_GRANTED
    }

    private fun bindCameraPreview(cameraProvider: ProcessCameraProvider) {
        if (cameraPreview != null) cameraProvider.unbind(cameraPreview)

        if (depthAnalysisView != null) cameraProvider.unbind(depthAnalysisView)

        cameraPreviewView!!.scaleType = PreviewView.ScaleType.FILL_END

        cameraPreview = Preview.Builder().setTargetFrameRate(Range<Int>(60, 120)).build()
        cameraPreview!!.setSurfaceProvider(cameraPreviewView!!.surfaceProvider)

        depthAnalysisView =
            ImageAnalysis.Builder()
                .setImageQueueDepth(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setResolutionSelector(performanceResolutionSelector())
                .build()
        depthAnalysisView!!.setAnalyzer(Executors.newCachedThreadPool(), cameraFrameAnalyzer)

        cameraProvider.bindToLifecycle(
            (this as LifecycleOwner),
            mostWideCameraSelector(cameraProvider),
            depthAnalysisView,
            cameraPreview
        )
    }
}

@OptIn(ExperimentalCamera2Interop::class)
private fun mostWideCameraSelector(cameraProvider: ProcessCameraProvider): CameraSelector {
    var widestCamera: CameraInfo? = null
    var maxFov = 0.0

    for (cameraInfo in cameraProvider.availableCameraInfos) {
        if (cameraInfo.lensFacing != CameraSelector.LENS_FACING_BACK) {
            continue
        }

        val camera2CameraInfo = Camera2CameraInfo.from(cameraInfo)
        val focalLengths =
            camera2CameraInfo.getCameraCharacteristic(
                CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS
            )
        if (focalLengths != null && focalLengths.size > 1) {
            val focalLength = focalLengths[0]

            val sensorSize =
                camera2CameraInfo.getCameraCharacteristic(
                    CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE
                )
            if (sensorSize != null) {
                val fov = calculateFov(focalLength.toDouble(), sensorSize.width.toDouble())

                if (fov > maxFov) {
                    maxFov = fov
                    widestCamera = cameraInfo
                }
            }
        }
    }

    return widestCamera?.cameraSelector ?: CameraSelector.DEFAULT_BACK_CAMERA
}

private fun performanceResolutionSelector(): ResolutionSelector {
    return ResolutionSelector.Builder()
        .setAllowedResolutionMode(ResolutionSelector.PREFER_CAPTURE_RATE_OVER_HIGHER_RESOLUTION)
        .setResolutionStrategy(
            ResolutionStrategy(
                MiDaSDepthModel.INPUT_IMAGE_SIZE,
                ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER
            )
        )
        .build()
}

private fun calculateFov(focalLength: Double, sensorWidth: Double): Double {
    return 2 * atan(sensorWidth / (2 * focalLength))
}
