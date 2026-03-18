package com.user.nn.predict;

import com.user.nn.core.*;
import com.user.nn.core.Module;

import java.io.IOException;

/**
 * Pipeline tiện lợi kết hợp: load model → preprocess → predict.
 * Cung cấp fluent API cho việc thiết lập và chạy inference nhanh chóng.
 * 
 * Ví dụ sử dụng:
 * <pre>
 *     PredictionResult result = PredictionPipeline
 *         .create(model)
 *         .loadWeights("model.bin")
 *         .labels(CIFAR10_LABELS)
 *         .device(Device.GPU)
 *         .topK(5)
 *         .verbose(true)
 *         .predict(inputTensor);
 * </pre>
 */
public class PredictionPipeline {

    private Module model;
    private String weightsPath;
    private String[] labels;
    private Tensor.Device device = Tensor.Device.CPU;
    private int topK = 5;
    private boolean verbose = false;

    // Image-specific
    private int channels = -1;
    private int height = -1;
    private int width = -1;
    private float[] normMean;
    private float[] normStd;

    private PredictionPipeline(Module model) {
        this.model = model;
    }

    // ======================== STATIC FACTORY ========================

    /** Tạo pipeline từ model */
    public static PredictionPipeline create(Module model) {
        return new PredictionPipeline(model);
    }

    // ======================== FLUENT CONFIGURATION ========================

    /** Load weights từ file */
    public PredictionPipeline loadWeights(String path) {
        this.weightsPath = path;
        return this;
    }

    /** Thiết lập labels */
    public PredictionPipeline labels(String[] labels) {
        this.labels = labels;
        return this;
    }

    /** Thiết lập device */
    public PredictionPipeline device(Tensor.Device device) {
        this.device = device;
        return this;
    }

    /** Thiết lập top-K */
    public PredictionPipeline topK(int k) {
        this.topK = k;
        return this;
    }

    /** Bật verbose */
    public PredictionPipeline verbose(boolean verbose) {
        this.verbose = verbose;
        return this;
    }

    /** Thiết lập image dimensions (kích hoạt ImagePredictor mode) */
    public PredictionPipeline imageSize(int channels, int height, int width) {
        this.channels = channels;
        this.height = height;
        this.width = width;
        return this;
    }

    /** Thiết lập normalization cho ảnh */
    public PredictionPipeline normalize(float[] mean, float[] std) {
        this.normMean = mean;
        this.normStd = std;
        return this;
    }

    // ======================== BUILD & PREDICT ========================

    /**
     * Build predictor từ configuration.
     */
    public Predictor build() throws IOException {
        if (weightsPath != null) {
            model.load(weightsPath);
            if (verbose) {
                System.out.println("[Pipeline] Loaded weights: " + weightsPath);
            }
        }

        Predictor predictor;

        if (channels > 0 && height > 0 && width > 0) {
            // ImagePredictor mode
            ImagePredictor imgPred = new ImagePredictor(model, channels, height, width, labels);
            if (normMean != null && normStd != null) {
                imgPred.normalize(normMean, normStd);
            }
            predictor = imgPred;
        } else {
            predictor = new Predictor(model, labels);
        }

        predictor.topK(topK).device(device).verbose(verbose);
        return predictor;
    }

    /**
     * Build và predict ngay lập tức (convenience method).
     */
    public PredictionResult predict(Tensor input) throws IOException {
        return build().predict(input);
    }

    /**
     * Build và predict batch.
     */
    public PredictionResult[] predictBatch(Tensor input) throws IOException {
        return build().predictBatch(input);
    }

    /**
     * Build ImagePredictor và predict từ pixel array.
     */
    public PredictionResult predictFromPixels(float[] pixels) throws IOException {
        Predictor pred = build();
        if (pred instanceof ImagePredictor) {
            return ((ImagePredictor) pred).predictFromPixels(pixels);
        }
        throw new IllegalStateException("Cần gọi imageSize() trước khi dùng predictFromPixels");
    }
}
