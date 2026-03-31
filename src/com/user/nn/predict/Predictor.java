package com.user.nn.predict;

import com.user.nn.core.*;
import com.user.nn.core.Module;

import java.io.IOException;
import java.util.*;

/**
 * Lớp cơ sở để thực hiện dự đoán (inference) từ một model đã train.
 * Hỗ trợ:
 *   - Predict đơn lẻ (single sample)
 *   - Predict theo batch
 *   - Top-K predictions
 *   - Softmax probabilities
 *   - Label mapping
 *   - Automatic no-grad mode
 *   - CPU/GPU inference
 */
public class Predictor {

    protected final Module model;
    protected String[] labels;
    protected int topK = 5;
    protected Tensor.Device device = Tensor.Device.CPU;
    protected boolean verbose = false;

    /**
     * Tạo Predictor từ một Module đã train.
     *
     * @param model Module đã được train (hoặc load weights)
     */
    public Predictor(Module model) {
        this.model = model;
        this.model.eval(); // Luôn ở eval mode khi predict
    }

    /**
     * Tạo Predictor từ model + label names.
     *
     * @param model  Module đã train
     * @param labels Mảng tên label (index tương ứng với class ID)
     */
    public Predictor(Module model, String[] labels) {
        this(model);
        this.labels = labels;
    }

    // ======================== BUILDER METHODS ========================

    /** Thiết lập top-K predictions */
    public Predictor topK(int k) {
        this.topK = k;
        return this;
    }

    /** Thiết lập device (CPU/GPU) */
    public Predictor device(Tensor.Device device) {
        this.device = device;
        if (device == Tensor.Device.GPU) {
            model.toGPU();
        } else {
            model.toCPU();
        }
        return this;
    }

    /** Thiết lập label names */
    public Predictor labels(String[] labels) {
        this.labels = labels;
        return this;
    }

    /** Bật/tắt verbose mode */
    public Predictor verbose(boolean verbose) {
        this.verbose = verbose;
        return this;
    }

    // ======================== LOAD MODEL ========================

    /**
     * Load weights từ file vào model hiện tại.
     *
     * @param path Đường dẫn file weights (.bin)
     * @return this (builder pattern)
     */
    public Predictor loadWeights(String path) throws IOException {
        model.load(path);
        if (verbose) {
            System.out.println("[Predictor] Loaded weights from: " + path);
            System.out.println("[Predictor] Model parameters: " + model.countParameters());
        }
        return this;
    }

    // ======================== PREDICTION ========================

    /**
     * Dự đoán cho một Tensor đầu vào.
     *
     * @param input Tensor đầu vào (đã preprocessing)
     * @return PredictionResult chứa kết quả dự đoán
     */
    public PredictionResult predict(Tensor input) {
        // Disable grad tracking cho inference
        boolean prevGrad = Torch.is_grad_enabled();
        Torch.set_grad_enabled(false);

        try {
            // Đảm bảo model ở eval mode
            model.eval();

            // Move input sang đúng device
            if (device == Tensor.Device.GPU) {
                input.toGPU();
            }

            // Forward pass
            long startMs = System.currentTimeMillis();
            Tensor logits = model.forward(input);
            long inferenceMs = System.currentTimeMillis() - startMs;

            if (verbose) {
                System.out.printf("[Predictor] Inference time: %dms%n", inferenceMs);
            }

            // Chuyển về CPU để xử lý kết quả
            logits.toCPU();
            System.out.println("DEBUG Predictor predict logits: " + java.util.Arrays.toString(logits.data));

            // Xử lý kết quả cho từng sample trong batch
            int batch = logits.shape[0];
            int numClasses = logits.shape[1];

            // Nếu batch = 1, trả về kết quả đơn lẻ
            if (batch == 1) {
                return processSinglePrediction(logits.data, numClasses, 0);
            }

            // Nếu batch > 1, trả về kết quả của sample đầu tiên
            // (Dùng predictBatch() để lấy tất cả)
            return processSinglePrediction(logits.data, numClasses, 0);

        } finally {
            Torch.set_grad_enabled(prevGrad);
        }
    }

    /**
     * Dự đoán cho nhiều samples (batch).
     *
     * @param input Tensor batch đầu vào [N, ...]
     * @return Mảng PredictionResult cho từng sample
     */
    public PredictionResult[] predictBatch(Tensor input) {
        boolean prevGrad = Torch.is_grad_enabled();
        Torch.set_grad_enabled(false);

        try {
            model.eval();

            if (device == Tensor.Device.GPU) {
                input.toGPU();
            }

            long startMs = System.currentTimeMillis();
            Tensor logits = model.forward(input);
            long inferenceMs = System.currentTimeMillis() - startMs;

            if (verbose) {
                System.out.printf("[Predictor] Batch inference time: %dms (batch=%d)%n", 
                    inferenceMs, logits.shape[0]);
            }

            logits.toCPU();

            int batch = logits.shape[0];
            int numClasses = logits.shape[1];

            PredictionResult[] results = new PredictionResult[batch];
            for (int i = 0; i < batch; i++) {
                results[i] = processSinglePrediction(logits.data, numClasses, i);
            }
            return results;

        } finally {
            Torch.set_grad_enabled(prevGrad);
        }
    }

    /**
     * Chỉ trả về class index (nhanh nhất, không tính softmax).
     *
     * @param input Tensor đầu vào
     * @return Class index được dự đoán
     */
    public int predictClass(Tensor input) {
        boolean prevGrad = Torch.is_grad_enabled();
        Torch.set_grad_enabled(false);

        try {
            model.eval();
            if (device == Tensor.Device.GPU) input.toGPU();

            Tensor logits = model.forward(input);
            logits.toCPU();
            
            System.out.println("DEBUG logits: " + java.util.Arrays.toString(logits.data));

            int numClasses = logits.shape[1];
            return argmax(logits.data, 0, numClasses);

        } finally {
            Torch.set_grad_enabled(prevGrad);
        }
    }

    /**
     * Chỉ trả về label (nhanh nhất).
     *
     * @param input Tensor đầu vào
     * @return Label được dự đoán
     */
    public String predictLabel(Tensor input) {
        int cls = predictClass(input);
        return (labels != null && cls < labels.length) ? labels[cls] : "class_" + cls;
    }

    // ======================== INTERNAL HELPERS ========================

    /**
     * Xử lý kết quả dự đoán cho 1 sample trong batch.
     */
    protected PredictionResult processSinglePrediction(float[] allLogits, int numClasses, int sampleIdx) {
        int offset = sampleIdx * numClasses;

        // Extract logits cho sample này
        float[] sampleLogits = new float[numClasses];
        System.arraycopy(allLogits, offset, sampleLogits, 0, numClasses);

        // Softmax
        float[] probs = softmax(sampleLogits);

        // Argmax
        int predicted = argmax(probs, 0, numClasses);
        float confidence = probs[predicted];

        // Top-K
        int k = Math.min(topK, numClasses);
        int[] topKIdx = topKIndices(probs, k);
        float[] topKProbs = new float[k];
        String[] topKLbls = new String[k];
        for (int i = 0; i < k; i++) {
            topKProbs[i] = probs[topKIdx[i]];
            topKLbls[i] = (labels != null && topKIdx[i] < labels.length) 
                ? labels[topKIdx[i]] : "class_" + topKIdx[i];
        }

        // Label
        String label = (labels != null && predicted < labels.length) 
            ? labels[predicted] : "class_" + predicted;

        return new PredictionResult(
            predicted, confidence, probs, sampleLogits,
            label, topKIdx, topKProbs, topKLbls
        );
    }

    /**
     * Tính softmax cho 1D array.
     */
    protected static float[] softmax(float[] logits) {
        float max = Float.NEGATIVE_INFINITY;
        for (float v : logits) if (v > max) max = v;

        double sum = 0.0;
        float[] probs = new float[logits.length];
        for (int i = 0; i < logits.length; i++) {
            probs[i] = (float) Math.exp(logits[i] - max);
            sum += probs[i];
        }
        for (int i = 0; i < probs.length; i++) {
            probs[i] /= (float) sum;
        }
        return probs;
    }

    /**
     * Tìm index có giá trị lớn nhất trong mảng.
     */
    protected static int argmax(float[] arr, int offset, int length) {
        int maxIdx = 0;
        float maxVal = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < length; i++) {
            if (arr[offset + i] > maxVal) {
                maxVal = arr[offset + i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }

    /**
     * Trả về top-K indices theo thứ tự giảm dần.
     */
    protected static int[] topKIndices(float[] arr, int k) {
        Integer[] indices = new Integer[arr.length];
        for (int i = 0; i < arr.length; i++) indices[i] = i;

        Arrays.sort(indices, (a, b) -> Float.compare(arr[b], arr[a]));

        int[] topK = new int[k];
        for (int i = 0; i < k; i++) topK[i] = indices[i];
        return topK;
    }

    // ======================== GETTERS ========================

    /** Trả về model */
    public Module getModel() { return model; }

    /** Trả về labels */
    public String[] getLabels() { return labels; }

    /** Trả về top-K setting */
    public int getTopK() { return topK; }

    /** Trả về device */
    public Tensor.Device getDevice() { return device; }
}
