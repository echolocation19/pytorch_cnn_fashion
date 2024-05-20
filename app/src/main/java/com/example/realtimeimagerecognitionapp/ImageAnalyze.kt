package com.example.realtimeimagerecognitionapp

import android.content.Context
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream

class ImageAnalyze(context: Context) : ImageAnalysis.Analyzer {

    private lateinit var listener: OnAnalyzeListener
    private var lastAnalyzedTimestamp = 0L
    private val model = Module.load(getAssetFilePath(context, "weights1.pt"))

    interface OnAnalyzeListener {
        fun getAnalyzeResult(inferredCategory: String, score: Float)
    }

    override fun analyze(image: ImageProxy, rotationDegrees: Int) {
        val currentTimestamp = System.currentTimeMillis()

        if (currentTimestamp - lastAnalyzedTimestamp >= 0.5) {
            lastAnalyzedTimestamp = currentTimestamp
            val inputTensor = TensorImageUtils
                .imageYUV420CenterCropToFloat32Tensor(
                image.image,
                rotationDegrees,
                224,
                    224,
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                TensorImageUtils.TORCHVISION_NORM_STD_RGB
            )
            val outputTensor = model.forward(IValue.from(inputTensor)).toTensor()
            val scores = outputTensor.dataAsFloatArray

            var maxScore = 0F
            var maxScoreIdx = 0
            for (i in scores.indices) {
                if (scores[i] > maxScore) {
                    maxScore = scores[i]
                    maxScoreIdx = i
                }
            }
            val inferredCategory = ImageNetClasses().IMAGENET_CLASSES[maxScoreIdx.coerceAtMost(2)]
            listener.getAnalyzeResult(inferredCategory, maxScore)
        }
    }

    private fun getAssetFilePath(context: Context, assetName: String): String {
        val file = File(context.filesDir, assetName)
        if (file.exists() && file.length() > 0) {
            return file.absolutePath
        }
        context.assets.open(assetName).use { inputStream ->
            FileOutputStream(file).use { outputStream ->
                val buffer = ByteArray(4 * 1024)
                var read: Int
                while (inputStream.read(buffer).also { read = it } != -1) {
                    outputStream.write(buffer, 0, read)
                }
                outputStream.flush()
            }
            return file.absolutePath
        }
    }

    fun setOnAnalyzeListener(listener: OnAnalyzeListener){
        this.listener = listener
    }
}