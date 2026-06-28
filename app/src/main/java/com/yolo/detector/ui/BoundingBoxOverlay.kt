package com.yolo.detector.ui

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import com.yolo.detector.ml.YoloDetector

/**
 * Transparent overlay View that draws bounding boxes and labels
 * on top of a displayed image. Call setDetections() after inference.
 */
class BoundingBoxOverlay @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private var detections: List<YoloDetector.Detection> = emptyList()
    private var imageWidth: Int = 1
    private var imageHeight: Int = 1

    // 20 visually distinct colours for class colouring
    private val CLASS_COLORS = intArrayOf(
        0xFFE53935.toInt(), 0xFF8E24AA.toInt(), 0xFF1E88E5.toInt(), 0xFF00ACC1.toInt(),
        0xFF43A047.toInt(), 0xFFFFB300.toInt(), 0xFFF4511E.toInt(), 0xFF6D4C41.toInt(),
        0xFF546E7A.toInt(), 0xFF00897B.toInt(), 0xFF7CB342.toInt(), 0xFFFDD835.toInt(),
        0xFFEF9A9A.toInt(), 0xFFCE93D8.toInt(), 0xFF90CAF9.toInt(), 0xFF80DEEA.toInt(),
        0xFFA5D6A7.toInt(), 0xFFFFE082.toInt(), 0xFFFFAB91.toInt(), 0xFFBCAAA4.toInt()
    )

    private val boxPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 4f
        isAntiAlias = true
    }

    private val fillPaint = Paint().apply {
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 38f
        typeface = Typeface.DEFAULT_BOLD
        isAntiAlias = true
    }

    private val textBgPaint = Paint().apply {
        style = Paint.Style.FILL
    }

    /**
     * @param detections  list from YoloDetector.detect()
     * @param srcWidth    width of the original image that was inferred
     * @param srcHeight   height of the original image
     */
    fun setDetections(
        detections: List<YoloDetector.Detection>,
        srcWidth: Int,
        srcHeight: Int
    ) {
        this.detections = detections
        this.imageWidth = srcWidth.coerceAtLeast(1)
        this.imageHeight = srcHeight.coerceAtLeast(1)
        invalidate()
    }

    fun clearDetections() {
        detections = emptyList()
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (detections.isEmpty()) return

        // Scale detection coords (in original image pixels) to view pixels
        val scaleX = width.toFloat() / imageWidth
        val scaleY = height.toFloat() / imageHeight

        for (det in detections) {
            val color = CLASS_COLORS[det.classIndex % CLASS_COLORS.size]
            boxPaint.color = color
            textBgPaint.color = color

            val box = RectF(
                det.boundingBox.left  * scaleX,
                det.boundingBox.top   * scaleY,
                det.boundingBox.right * scaleX,
                det.boundingBox.bottom * scaleY
            )

            // Semi-transparent fill
            fillPaint.color = (color and 0x00FFFFFF) or 0x22000000
            canvas.drawRect(box, fillPaint)
            canvas.drawRect(box, boxPaint)

            // Label: "ClassName 87%"
            val label = "${det.className}  ${"%.0f".format(det.confidence * 100)}%"
            val textBounds = Rect()
            textPaint.getTextBounds(label, 0, label.length, textBounds)
            val labelBgHeight = textBounds.height() + 16f
            val labelBgWidth  = textBounds.width()  + 16f

            val labelTop = if (box.top > labelBgHeight) box.top - labelBgHeight else box.top
            val labelBg = RectF(
                box.left, labelTop,
                box.left + labelBgWidth, labelTop + labelBgHeight
            )
            canvas.drawRect(labelBg, textBgPaint)
            canvas.drawText(
                label,
                box.left + 8f,
                labelTop + labelBgHeight - 8f,
                textPaint
            )
        }
    }
}
