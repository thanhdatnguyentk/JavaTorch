package com.user.nn.predict;

import java.util.Arrays;

/**
 * Chứa kết quả dự đoán từ một model.
 * Bao gồm class được dự đoán, xác suất (probabilities), confidence, và top-k kết quả.
 */
public class PredictionResult {
    
    /** Class index được dự đoán (argmax) */
    private final int predictedClass;
    
    /** Confidence (xác suất cao nhất) */
    private final float confidence;
    
    /** Mảng xác suất cho tất cả các class (softmax output) */
    private final float[] probabilities;
    
    /** Raw logits output từ model */
    private final float[] logits;
    
    /** Tên class nếu có label mapping */
    private final String predictedLabel;
    
    /** Top-K indices */
    private final int[] topKIndices;
    
    /** Top-K probabilities */
    private final float[] topKProbabilities;
    
    /** Top-K labels */
    private final String[] topKLabels;

    public PredictionResult(int predictedClass, float confidence, float[] probabilities,
                           float[] logits, String predictedLabel,
                           int[] topKIndices, float[] topKProbabilities, String[] topKLabels) {
        this.predictedClass = predictedClass;
        this.confidence = confidence;
        this.probabilities = probabilities;
        this.logits = logits;
        this.predictedLabel = predictedLabel;
        this.topKIndices = topKIndices;
        this.topKProbabilities = topKProbabilities;
        this.topKLabels = topKLabels;
    }

    /** Trả về class index được dự đoán */
    public int getPredictedClass() { return predictedClass; }

    /** Trả về confidence (xác suất cao nhất) */
    public float getConfidence() { return confidence; }

    /** Trả về mảng xác suất cho tất cả class */
    public float[] getProbabilities() { return probabilities; }

    /** Trả về raw logits */
    public float[] getLogits() { return logits; }

    /** Trả về tên label của class được dự đoán */
    public String getPredictedLabel() { return predictedLabel; }

    /** Trả về top-K class indices */
    public int[] getTopKIndices() { return topKIndices; }

    /** Trả về top-K probabilities */
    public float[] getTopKProbabilities() { return topKProbabilities; }

    /** Trả về top-K labels */
    public String[] getTopKLabels() { return topKLabels; }

    /** Số lượng class */
    public int getNumClasses() { return probabilities != null ? probabilities.length : 0; }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("╔══════════════════════════════════════════╗\n");
        sb.append("║         PREDICTION RESULT                ║\n");
        sb.append("╠══════════════════════════════════════════╣\n");
        
        if (predictedLabel != null && !predictedLabel.isEmpty()) {
            sb.append(String.format("║ Predicted: %-30s ║\n", predictedLabel));
        }
        sb.append(String.format("║ Class ID:  %-30d ║\n", predictedClass));
        sb.append(String.format("║ Confidence: %-29.4f ║\n", confidence));
        
        if (topKIndices != null && topKIndices.length > 0) {
            sb.append("╠══════════════════════════════════════════╣\n");
            sb.append("║ Top-K Predictions:                       ║\n");
            for (int i = 0; i < topKIndices.length; i++) {
                String label = (topKLabels != null && i < topKLabels.length) 
                    ? topKLabels[i] : "class_" + topKIndices[i];
                sb.append(String.format("║  #%d: %-20s (%.4f)      ║\n", 
                    i + 1, label, topKProbabilities[i]));
            }
        }
        
        sb.append("╚══════════════════════════════════════════╝");
        return sb.toString();
    }
}
