package com.yolo.detector.ui

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.exifinterface.media.ExifInterface
import androidx.lifecycle.lifecycleScope
import com.yolo.detector.databinding.ActivityMainBinding
import com.yolo.detector.ml.YoloDetector
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "MainActivity"
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }

    private lateinit var binding: ActivityMainBinding
    private lateinit var detector: YoloDetector
    private lateinit var cameraExecutor: ExecutorService

    private var imageCapture: ImageCapture? = null
    private var isInferenceRunning = false

    // ─── Permission launcher ──────────────────────────────────────────────────

    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { perms ->
            if (perms[Manifest.permission.CAMERA] == true) {
                startCamera()
            } else {
                Toast.makeText(this, "Camera permission required", Toast.LENGTH_LONG).show()
            }
        }

    // ─── Gallery picker ───────────────────────────────────────────────────────

    private val galleryLauncher =
        registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
            uri?.let { runInferenceOnUri(it) }
        }

    // ─── Lifecycle ────────────────────────────────────────────────────────────

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Load model once
        detector = YoloDetector(
            context = applicationContext,
            modelPath = "best.tflite",
            labelsPath = "labels.txt",
            confidenceThreshold = 0.35f,
            iouThreshold = 0.45f
        )

        cameraExecutor = Executors.newSingleThreadExecutor()

        setupButtons()
        checkAndRequestPermissions()
    }

    override fun onDestroy() {
        super.onDestroy()
        detector.close()
        cameraExecutor.shutdown()
    }

    // ─── UI setup ─────────────────────────────────────────────────────────────

    private fun setupButtons() {
        // Capture image from camera and run inference
        binding.btnCapture.setOnClickListener {
            if (!isInferenceRunning) captureAndInfer()
        }

        // Pick image from gallery and run inference
        binding.btnGallery.setOnClickListener {
            if (!isInferenceRunning) galleryLauncher.launch("image/*")
        }

        // Clear results and return to live preview
        binding.btnClear.setOnClickListener {
            clearResults()
        }
    }

    // ─── Camera ───────────────────────────────────────────────────────────────

    private fun checkAndRequestPermissions() {
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
        }
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(binding.cameraPreview.surfaceProvider)
            }

            imageCapture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
                .build()

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    imageCapture
                )
            } catch (e: Exception) {
                Log.e(TAG, "Camera bind failed", e)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    // ─── Inference from camera capture ───────────────────────────────────────

    private fun captureAndInfer() {
        val capture = imageCapture ?: return
        setLoading(true)

        // Save to temp file then load bitmap
        val photoFile = File(
            cacheDir,
            "capture_${SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())}.jpg"
        )
        val outputOptions = ImageCapture.OutputFileOptions.Builder(photoFile).build()

        capture.takePicture(
            outputOptions,
            ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageSavedCallback {
                override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                    val bitmap = loadAndOrientBitmap(photoFile)
                    if (bitmap != null) {
                        showCapturedImage(bitmap)
                        runInferenceOnBitmap(bitmap)
                    } else {
                        setLoading(false)
                        Toast.makeText(this@MainActivity, "Failed to load image", Toast.LENGTH_SHORT).show()
                    }
                }

                override fun onError(exc: ImageCaptureException) {
                    setLoading(false)
                    Log.e(TAG, "Photo capture failed: ${exc.message}", exc)
                    Toast.makeText(this@MainActivity, "Capture failed", Toast.LENGTH_SHORT).show()
                }
            }
        )
    }

    // ─── Inference from gallery URI ───────────────────────────────────────────

    private fun runInferenceOnUri(uri: Uri) {
        lifecycleScope.launch {
            setLoading(true)
            val bitmap = withContext(Dispatchers.IO) {
                try {
                    contentResolver.openInputStream(uri)?.use {
                        BitmapFactory.decodeStream(it)
                    }
                } catch (e: Exception) {
                    null
                }
            }
            if (bitmap == null) {
                setLoading(false)
                Toast.makeText(this@MainActivity, "Could not load image", Toast.LENGTH_SHORT).show()
                return@launch
            }
            showCapturedImage(bitmap)
            runInferenceOnBitmap(bitmap)
        }
    }

    // ─── Core inference pipeline ──────────────────────────────────────────────

    private fun runInferenceOnBitmap(bitmap: Bitmap) {
        lifecycleScope.launch {
            val detections = withContext(Dispatchers.Default) {
                try {
                    val start = System.currentTimeMillis()
                    val result = detector.detect(bitmap)
                    val elapsed = System.currentTimeMillis() - start
                    Log.i(TAG, "Inference: ${elapsed}ms, ${result.size} detections")
                    Pair(result, elapsed)
                } catch (e: Exception) {
                    Log.e(TAG, "Inference error", e)
                    null
                }
            }

            setLoading(false)
            isInferenceRunning = false

            if (detections == null) {
                Toast.makeText(this@MainActivity, "Inference failed", Toast.LENGTH_SHORT).show()
                return@launch
            }

            val (results, elapsed) = detections
            displayResults(results, bitmap.width, bitmap.height, elapsed)
        }
    }

    // ─── Results display ──────────────────────────────────────────────────────

    private fun showCapturedImage(bitmap: Bitmap) {
        // Switch from camera preview to static image view
        binding.cameraPreview.visibility = View.GONE
        binding.ivCaptured.visibility = View.VISIBLE
        binding.ivCaptured.setImageBitmap(bitmap)
        binding.overlay.clearDetections()
        binding.btnClear.visibility = View.VISIBLE
        binding.btnCapture.visibility = View.GONE
    }

    private fun displayResults(
        detections: List<YoloDetector.Detection>,
        imageW: Int,
        imageH: Int,
        elapsedMs: Long
    ) {
        binding.overlay.setDetections(detections, imageW, imageH)

        val summary = if (detections.isEmpty()) {
            "No objects detected  •  ${elapsedMs}ms"
        } else {
            val counts = detections.groupBy { it.className }
                .entries.joinToString(", ") { "${it.value.size}× ${it.key}" }
            "$counts  •  ${elapsedMs}ms"
        }
        binding.tvResult.text = summary
        binding.tvResult.visibility = View.VISIBLE
    }

    private fun clearResults() {
        binding.overlay.clearDetections()
        binding.tvResult.visibility = View.GONE
        binding.btnClear.visibility = View.GONE
        binding.btnCapture.visibility = View.VISIBLE
        binding.ivCaptured.visibility = View.GONE
        binding.ivCaptured.setImageBitmap(null)
        binding.cameraPreview.visibility = View.VISIBLE
    }

    // ─── Helpers ──────────────────────────────────────────────────────────────

    private fun setLoading(loading: Boolean) {
        isInferenceRunning = loading
        binding.progressBar.visibility = if (loading) View.VISIBLE else View.GONE
        binding.btnCapture.isEnabled = !loading
        binding.btnGallery.isEnabled = !loading
    }

    /**
     * Loads a JPEG file and applies EXIF rotation so the bitmap is upright.
     */
    private fun loadAndOrientBitmap(file: File): Bitmap? {
        return try {
            val bitmap = BitmapFactory.decodeFile(file.absolutePath) ?: return null
            val exif = ExifInterface(file.absolutePath)
            val orientation = exif.getAttributeInt(
                ExifInterface.TAG_ORIENTATION,
                ExifInterface.ORIENTATION_NORMAL
            )
            val rotation = when (orientation) {
                ExifInterface.ORIENTATION_ROTATE_90  -> 90f
                ExifInterface.ORIENTATION_ROTATE_180 -> 180f
                ExifInterface.ORIENTATION_ROTATE_270 -> 270f
                else -> 0f
            }
            if (rotation == 0f) bitmap
            else {
                val matrix = Matrix().apply { postRotate(rotation) }
                Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
                    .also { if (it != bitmap) bitmap.recycle() }
            }
        } catch (e: Exception) {
            Log.e(TAG, "loadAndOrientBitmap failed", e)
            null
        }
    }
}
