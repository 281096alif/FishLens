package com.yolo.detector.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Core YOLOv11 TFLite inference engine.
 *
 * Model output shape: [1, (4 + NUM_CLASSES), 8400]
 *   - 8400 candidate boxes from the detection head (for 640×640 input)
 *   - First 4 values per box: cx, cy, w, h (normalised 0-1)
 *   - Next NUM_CLASSES values: class confidence scores (already scaled, no sigmoid needed
 *     for models exported with --int8 or standard TFLite export)
 *
 * We do NMS here in Kotlin (no need for external library).
 */
class YoloDetector(
    private val context: Context,
    private val modelPath: String = "best.tflite",
    private val labelsPath: String = "labels.txt",
    val confidenceThreshold: Float = 0.35f,
    val iouThreshold: Float = 0.45f
) {

    companion object {
        private const val TAG = "YoloDetector"
        const val INPUT_SIZE = 640
        private const val NUM_BYTES_PER_CHANNEL = 4  // float32
    }

    data class Detection(
        val boundingBox: RectF,   // pixel coords relative to INPUT_SIZE square
        val classIndex: Int,
        val className: String,
        val confidence: Float
    )

    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null
    private lateinit var labels: List<String>
    private var numClasses: Int = 0

    // Output tensor: shape [1, (4+numClasses), 8400]
    private lateinit var outputBuffer: Array<Array<FloatArray>>

    init {
        setupInterpreter()
        loadLabels()
    }

    private fun setupInterpreter() {
        try {
            val model = FileUtil.loadMappedFile(context, modelPath)
            val options = Interpreter.Options().apply {
                numThreads = 4

                // Try GPU delegate first (Samsung Exynos / Snapdragon GPU)
                val compatList = CompatibilityList()
                if (compatList.isDelegateSupportedOnThisDevice) {
                    Log.i(TAG, "GPU delegate supported — using hardware acceleration")
                    gpuDelegate = GpuDelegate(
                        compatList.bestOptionsForThisDevice
                    )
                    addDelegate(gpuDelegate!!)
                } else {
                    Log.i(TAG, "GPU delegate not supported — using CPU with 4 threads")
                }
            }
            interpreter = Interpreter(model, options)

            // Inspect output shape so we're robust to any num_classes
            val outputShape = interpreter!!.getOutputTensor(0).shape()
            // Expected: [1, 4+numClasses, 8400]
            numClasses = outputShape[1] - 4
            outputBuffer = Array(1) { Array(outputShape[1]) { FloatArray(outputShape[2]) } }
            Log.i(TAG, "Model loaded. Classes=$numClasses, anchors=${outputShape[2]}")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load model: ${e.message}", e)
            throw e
        }
    }

    private fun loadLabels() {
        labels = try {
            context.assets.open(labelsPath).bufferedReader().readLines()
                .map { it.trim() }.filter { it.isNotEmpty() }
        } catch (e: Exception) {
            Log.w(TAG, "labels.txt not found, generating default class names")
            (0 until numClasses).map { "class_$it" }
        }
    }

    /**
     * Run inference on a bitmap (any size — will be letterboxed to 640×640).
     * Returns list of detections in ORIGINAL image coordinate space.
     */
    fun detect(bitmap: Bitmap): List<Detection> {
        val (letterboxed, scale, padLeft, padTop) = letterbox(bitmap)
        val inputBuffer = bitmapToByteBuffer(letterboxed)

        // Reset output buffer
        outputBuffer[0].forEach { row -> row.fill(0f) }

        interpreter?.run(inputBuffer, outputBuffer)

        val rawDetections = parseOutput(outputBuffer[0])
        val afterNms = applyNms(rawDetections)

        // Map coordinates back to original image space
        return afterNms.map { det ->
            val box = det.boundingBox
            val origBox = RectF(
                (box.left - padLeft) / scale,
                (box.top - padTop) / scale,
                (box.right - padLeft) / scale,
                (box.bottom - padTop) / scale
            )
            // Clamp to original image bounds
            origBox.left = origBox.left.coerceIn(0f, bitmap.width.toFloat())
            origBox.top = origBox.top.coerceIn(0f, bitmap.height.toFloat())
            origBox.right = origBox.right.coerceIn(0f, bitmap.width.toFloat())
            origBox.bottom = origBox.bottom.coerceIn(0f, bitmap.height.toFloat())
            det.copy(boundingBox = origBox)
        }
    }

    // ─── Pre-processing ──────────────────────────────────────────────────────

    data class LetterboxResult(
        val bitmap: Bitmap,
        val scale: Float,
        val padLeft: Float,
        val padTop: Float
    )

    private fun letterbox(src: Bitmap): LetterboxResult {
        val scale = minOf(
            INPUT_SIZE.toFloat() / src.width,
            INPUT_SIZE.toFloat() / src.height
        )
        val newW = (src.width * scale).toInt()
        val newH = (src.height * scale).toInt()
        val padLeft = (INPUT_SIZE - newW) / 2f
        val padTop = (INPUT_SIZE - newH) / 2f

        val result = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888)
        val canvas = android.graphics.Canvas(result)
        // Fill with grey (YOLO default letterbox colour)
        canvas.drawColor(android.graphics.Color.rgb(114, 114, 114))
        val scaled = Bitmap.createScaledBitmap(src, newW, newH, true)
        canvas.drawBitmap(scaled, padLeft, padTop, null)
        if (scaled != src) scaled.recycle()

        return LetterboxResult(result, scale, padLeft, padTop)
    }

    private fun bitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val buffer = ByteBuffer.allocateDirect(
            1 * INPUT_SIZE * INPUT_SIZE * 3 * NUM_BYTES_PER_CHANNEL
        ).apply { order(ByteOrder.nativeOrder()) }

        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        for (pixel in pixels) {
            // Normalise to [0, 1] — float32 YOLO TFLite model
            buffer.putFloat(((pixel shr 16) and 0xFF) / 255.0f)  // R
            buffer.putFloat(((pixel shr 8) and 0xFF) / 255.0f)   // G
            buffer.putFloat((pixel and 0xFF) / 255.0f)            // B
        }
        buffer.rewind()
        return buffer
    }

    // ─── Post-processing ─────────────────────────────────────────────────────

    /**
     * Parse raw model output: shape [4+numClasses, 8400]
     * YOLO output is transposed for TFLite: rows = features, cols = anchors.
     * Each anchor i: cx=out[0][i], cy=out[1][i], w=out[2][i], h=out[3][i],
     * class scores = out[4..4+nc][i]
     */
    private fun parseOutput(output: Array<FloatArray>): List<Detection> {
        val detections = mutableListOf<Detection>()
        val numAnchors = output[0].size

        for (i in 0 until numAnchors) {
            // Find best class
            var maxScore = confidenceThreshold
            var maxClass = -1
            for (c in 0 until numClasses) {
                val score = output[4 + c][i]
                if (score > maxScore) {
                    maxScore = score
                    maxClass = c
                }
            }
            if (maxClass == -1) continue

            // cx, cy, w, h — in [0, INPUT_SIZE] pixels (YOLO TFLite exports in pixel coords)
            val cx = output[0][i]
            val cy = output[1][i]
            val w  = output[2][i]
            val h  = output[3][i]

            val left   = cx - w / 2f
            val top    = cy - h / 2f
            val right  = cx + w / 2f
            val bottom = cy + h / 2f

            // Filter invalid boxes
            if (w <= 0f || h <= 0f) continue

            val label = labels.getOrElse(maxClass) { "class_$maxClass" }
            detections.add(
                Detection(
                    boundingBox = RectF(left, top, right, bottom),
                    classIndex = maxClass,
                    className = label,
                    confidence = maxScore
                )
            )
        }
        return detections
    }

    /**
     * Non-Maximum Suppression.
     * Sorts by confidence, greedily keeps boxes whose IoU with any
     * already-kept box of the SAME class is below the threshold.
     */
    private fun applyNms(detections: List<Detection>): List<Detection> {
        if (detections.isEmpty()) return emptyList()

        val sorted = detections.sortedByDescending { it.confidence }
        val kept = mutableListOf<Detection>()

        val suppressed = BooleanArray(sorted.size)
        for (i in sorted.indices) {
            if (suppressed[i]) continue
            kept.add(sorted[i])
            for (j in i + 1 until sorted.size) {
                if (suppressed[j]) continue
                // Only suppress within same class
                if (sorted[i].classIndex != sorted[j].classIndex) continue
                if (iou(sorted[i].boundingBox, sorted[j].boundingBox) >= iouThreshold) {
                    suppressed[j] = true
                }
            }
        }
        return kept
    }

    private fun iou(a: RectF, b: RectF): Float {
        val interLeft   = maxOf(a.left, b.left)
        val interTop    = maxOf(a.top, b.top)
        val interRight  = minOf(a.right, b.right)
        val interBottom = minOf(a.bottom, b.bottom)

        if (interRight <= interLeft || interBottom <= interTop) return 0f

        val interArea = (interRight - interLeft) * (interBottom - interTop)
        val aArea = a.width() * a.height()
        val bArea = b.width() * b.height()
        return interArea / (aArea + bArea - interArea)
    }

    fun close() {
        interpreter?.close()
        gpuDelegate?.close()
    }
}
