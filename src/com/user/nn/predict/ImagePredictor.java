package com.user.nn.predict;

import com.user.nn.core.*;
import com.user.nn.core.Module;

/**
 * Predictor chuyên biệt cho bài toán phân loại ảnh (Image Classification).
 * Hỗ trợ:
 *   - Normalize ảnh (mean/std)
 *   - Resize/reshape input
 *   - Dự đoán từ raw pixel data
 *   - Các label sets phổ biến (CIFAR-10, MNIST, Fashion-MNIST...)
 */
public class ImagePredictor extends Predictor {

    private float[] mean;
    private float[] std;
    private int channels;
    private int height;
    private int width;

    /** CIFAR-10 class names */
    public static final String[] CIFAR10_LABELS = {
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    };

    /** MNIST class names */
    public static final String[] MNIST_LABELS = {
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
    };

    /** Fashion-MNIST class names */
    public static final String[] FASHION_MNIST_LABELS = {
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    };

    /**
     * Tạo ImagePredictor.
     *
     * @param model    Model đã train
     * @param channels Số kênh ảnh (1=grayscale, 3=RGB)
     * @param height   Chiều cao ảnh
     * @param width    Chiều rộng ảnh
     */
    public ImagePredictor(Module model, int channels, int height, int width) {
        super(model);
        this.channels = channels;
        this.height = height;
        this.width = width;
    }

    /**
     * Tạo ImagePredictor với label mapping.
     */
    public ImagePredictor(Module model, int channels, int height, int width, String[] labels) {
        super(model, labels);
        this.channels = channels;
        this.height = height;
        this.width = width;
    }

    // ======================== BUILDER METHODS ========================

    /**
     * Thiết lập normalization parameters (mean và std cho mỗi channel).
     * Ví dụ CIFAR-10: mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
     */
    public ImagePredictor normalize(float[] mean, float[] std) {
        if (mean.length != channels || std.length != channels) {
            throw new IllegalArgumentException(
                "mean/std phải có độ dài bằng số channels (" + channels + ")");
        }
        this.mean = mean;
        this.std = std;
        return this;
    }

    // ======================== PREDICT FROM PIXEL ARRAYS ========================

    /**
     * Dự đoán từ mảng pixel 1D (đã flatten).
     * Format: [C * H * W] (channel-first layout)
     *
     * @param pixels Mảng pixel values [0, 1] hoặc [0, 255]
     * @return PredictionResult
     */
    public PredictionResult predictFromPixels(float[] pixels) {
        int expectedLen = channels * height * width;
        if (pixels.length != expectedLen) {
            throw new IllegalArgumentException(
                "Expected " + expectedLen + " pixels, got " + pixels.length);
        }

        float[] processed = preprocessPixels(pixels);
        Tensor input = Torch.tensor(processed, 1, channels, height, width);
        return predict(input);
    }

    /**
     * Dự đoán batch từ mảng pixel 2D.
     * Mỗi hàng là 1 ảnh flatten [C * H * W].
     *
     * @param pixelBatch Mảng [N][C*H*W]
     * @return Mảng PredictionResult
     */
    public PredictionResult[] predictFromPixelBatch(float[][] pixelBatch) {
        int n = pixelBatch.length;
        int imageSize = channels * height * width;

        float[] batchData = new float[n * imageSize];
        for (int i = 0; i < n; i++) {
            float[] processed = preprocessPixels(pixelBatch[i]);
            System.arraycopy(processed, 0, batchData, i * imageSize, imageSize);
        }

        Tensor input = Torch.tensor(batchData, n, channels, height, width);
        return predictBatch(input);
    }

    /**
     * Dự đoán từ pixel data dạng int [0-255].
     */
    public PredictionResult predictFromRgb(int[] rgbPixels) {
        float[] floatPixels = new float[rgbPixels.length];
        for (int i = 0; i < rgbPixels.length; i++) {
            floatPixels[i] = rgbPixels[i] / 255.0f;
        }
        return predictFromPixels(floatPixels);
    }

    // ======================== PREPROCESSING ========================

    /**
     * Tiền xử lý pixel: normalize nếu có setting.
     */
    private float[] preprocessPixels(float[] pixels) {
        // Auto-detect: nếu giá trị > 1.0 → scale về [0, 1]
        boolean needsScale = false;
        for (float v : pixels) {
            if (v > 1.0f) { needsScale = true; break; }
        }

        float[] result = pixels.clone();

        if (needsScale) {
            for (int i = 0; i < result.length; i++) {
                result[i] /= 255.0f;
            }
        }

        // Apply normalization per channel
        if (mean != null && std != null) {
            int pixelsPerChannel = height * width;
            for (int c = 0; c < channels; c++) {
                int offset = c * pixelsPerChannel;
                for (int i = 0; i < pixelsPerChannel; i++) {
                    result[offset + i] = (result[offset + i] - mean[c]) / std[c];
                }
            }
        }

        return result;
    }

    // ======================== FACTORY METHODS ========================

    /**
     * Tạo ImagePredictor cho CIFAR-10 (32x32, 3 channels, 10 classes).
     */
    public static ImagePredictor forCifar10(Module model) {
        return new ImagePredictor(model, 3, 32, 32, CIFAR10_LABELS);
    }

    /**
     * Tạo ImagePredictor cho MNIST (28x28, 1 channel, 10 classes).
     */
    public static ImagePredictor forMnist(Module model) {
        return new ImagePredictor(model, 1, 28, 28, MNIST_LABELS);
    }

    /**
     * Tạo ImagePredictor cho Fashion-MNIST (28x28, 1 channel, 10 classes).
     */
    public static ImagePredictor forFashionMnist(Module model) {
        return new ImagePredictor(model, 1, 28, 28, FASHION_MNIST_LABELS);
    }

    // ======================== GETTERS ========================

    public int getChannels() { return channels; }
    public int getHeight() { return height; }
    public int getWidth() { return width; }
    public float[] getMean() { return mean; }
    public float[] getStd() { return std; }
}
