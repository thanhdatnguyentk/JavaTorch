package com.user.nn.predict;

import com.user.nn.core.*;
import com.user.nn.core.Module;
import com.user.nn.dataloaders.Data;
import com.user.nn.metrics.Accuracy;

/**
 * BatchPredictor hỗ trợ dự đoán trên toàn bộ DataLoader.
 * Thường dùng để evaluate model trên test set hoặc tạo predictions file.
 * 
 * Hỗ trợ:
 *   - Iterate qua DataLoader
 *   - Tính accuracy tự động
 *   - Thu thập tất cả predictions
 *   - Progress tracking
 */
public class BatchPredictor {

    private final Predictor predictor;
    private boolean showProgress = true;

    public BatchPredictor(Predictor predictor) {
        this.predictor = predictor;
    }

    /**
     * Tạo BatchPredictor từ model.
     */
    public BatchPredictor(Module model) {
        this.predictor = new Predictor(model);
    }

    /**
     * Tạo BatchPredictor từ model với labels.
     */
    public BatchPredictor(Module model, String[] labels) {
        this.predictor = new Predictor(model, labels);
    }

    /** Bật/tắt progress display */
    public BatchPredictor showProgress(boolean show) {
        this.showProgress = show;
        return this;
    }

    // ======================== PREDICTION TRÊN DATALOADER ========================

    /**
     * Dự đoán và tính accuracy trên DataLoader.
     *
     * @param loader  DataLoader chứa test data
     * @return Accuracy trên test set
     */
    public float evaluateAccuracy(Data.DataLoader loader) {
        Accuracy metric = new Accuracy();
        metric.reset();

        boolean prevGrad = Torch.is_grad_enabled();
        Torch.set_grad_enabled(false);

        try {
            predictor.getModel().eval();
            int batchCount = 0;

            for (Tensor[] batch : loader) {
                try (MemoryScope scope = new MemoryScope()) {
                    Tensor xBatch = batch[0];
                    Tensor yBatch = batch[1];

                    scope.track(xBatch);
                    scope.track(yBatch);

                    if (predictor.getDevice() == Tensor.Device.GPU) {
                        xBatch.toGPU();
                    }

                    int bs = xBatch.shape[0];
                    int[] batchLabels = new int[bs];
                    for (int i = 0; i < bs; i++) {
                        batchLabels[i] = (int) yBatch.data[i];
                    }

                    Tensor logits = predictor.getModel().forward(xBatch);
                    metric.update(logits, batchLabels);

                    batchCount++;
                    if (showProgress && batchCount % 10 == 0) {
                        System.out.printf("[BatchPredictor] Processed %d batches, running accuracy: %.4f%n",
                            batchCount, metric.compute());
                    }
                }
            }

            float finalAcc = metric.compute();
            if (showProgress) {
                System.out.printf("[BatchPredictor] Final accuracy: %.4f (total batches: %d)%n",
                    finalAcc, batchCount);
            }
            return finalAcc;

        } finally {
            Torch.set_grad_enabled(prevGrad);
        }
    }

    /**
     * Thu thập tất cả predicted class indices trên DataLoader.
     *
     * @param loader DataLoader
     * @return Mảng predicted class indices
     */
    public int[] collectPredictions(Data.DataLoader loader) {
        java.util.List<Integer> allPreds = new java.util.ArrayList<>();

        boolean prevGrad = Torch.is_grad_enabled();
        Torch.set_grad_enabled(false);

        try {
            predictor.getModel().eval();

            for (Tensor[] batch : loader) {
                try (MemoryScope scope = new MemoryScope()) {
                    Tensor xBatch = batch[0];
                    scope.track(xBatch);

                    if (predictor.getDevice() == Tensor.Device.GPU) {
                        xBatch.toGPU();
                    }

                    Tensor logits = predictor.getModel().forward(xBatch);
                    logits.toCPU();

                    int bs = logits.shape[0];
                    int numClasses = logits.shape[1];

                    for (int i = 0; i < bs; i++) {
                        int pred = Predictor.argmax(logits.data, i * numClasses, numClasses);
                        allPreds.add(pred);
                    }
                }
            }

        } finally {
            Torch.set_grad_enabled(prevGrad);
        }

        int[] result = new int[allPreds.size()];
        for (int i = 0; i < allPreds.size(); i++) {
            result[i] = allPreds.get(i);
        }
        return result;
    }

    /**
     * Thu thập predictions kèm theo confidence scores.
     *
     * @param loader DataLoader
     * @return PredictionResult[] cho tất cả samples
     */
    public PredictionResult[] collectDetailedPredictions(Data.DataLoader loader) {
        java.util.List<PredictionResult> allResults = new java.util.ArrayList<>();

        boolean prevGrad = Torch.is_grad_enabled();
        Torch.set_grad_enabled(false);

        try {
            predictor.getModel().eval();

            for (Tensor[] batch : loader) {
                try (MemoryScope scope = new MemoryScope()) {
                    Tensor xBatch = batch[0];
                    scope.track(xBatch);

                    if (predictor.getDevice() == Tensor.Device.GPU) {
                        xBatch.toGPU();
                    }

                    Tensor logits = predictor.getModel().forward(xBatch);
                    logits.toCPU();

                    int bs = logits.shape[0];
                    int numClasses = logits.shape[1];

                    for (int i = 0; i < bs; i++) {
                        PredictionResult result = predictor.processSinglePrediction(
                            logits.data, numClasses, i);
                        allResults.add(result);
                    }
                }
            }

        } finally {
            Torch.set_grad_enabled(prevGrad);
        }

        return allResults.toArray(new PredictionResult[0]);
    }

    /**
     * Tạo confusion matrix đơn giản.
     * 
     * @param loader     DataLoader
     * @param numClasses Số lượng class
     * @return int[actual][predicted] confusion matrix
     */
    public int[][] confusionMatrix(Data.DataLoader loader, int numClasses) {
        int[][] matrix = new int[numClasses][numClasses];

        boolean prevGrad = Torch.is_grad_enabled();
        Torch.set_grad_enabled(false);

        try {
            predictor.getModel().eval();

            for (Tensor[] batch : loader) {
                try (MemoryScope scope = new MemoryScope()) {
                    Tensor xBatch = batch[0];
                    Tensor yBatch = batch[1];

                    scope.track(xBatch);
                    scope.track(yBatch);

                    if (predictor.getDevice() == Tensor.Device.GPU) {
                        xBatch.toGPU();
                    }

                    Tensor logits = predictor.getModel().forward(xBatch);
                    logits.toCPU();

                    int bs = logits.shape[0];
                    int nc = logits.shape[1];

                    for (int i = 0; i < bs; i++) {
                        int actual = (int) yBatch.data[i];
                        int predicted = Predictor.argmax(logits.data, i * nc, nc);
                        if (actual >= 0 && actual < numClasses && predicted >= 0 && predicted < numClasses) {
                            matrix[actual][predicted]++;
                        }
                    }
                }
            }

        } finally {
            Torch.set_grad_enabled(prevGrad);
        }

        return matrix;
    }

    /**
     * In confusion matrix đẹp ra console.
     */
    public void printConfusionMatrix(int[][] matrix, String[] labels) {
        int n = matrix.length;
        String[] lbls = labels != null ? labels : new String[n];
        if (labels == null) {
            for (int i = 0; i < n; i++) lbls[i] = "C" + i;
        }

        // Header
        int maxLabelLen = 0;
        for (String l : lbls) maxLabelLen = Math.max(maxLabelLen, l.length());
        maxLabelLen = Math.max(maxLabelLen, 6);

        System.out.println("\n═══ Confusion Matrix ═══");
        System.out.printf("%" + maxLabelLen + "s │", "");
        for (String l : lbls) System.out.printf(" %6s", l.length() > 6 ? l.substring(0, 6) : l);
        System.out.println();
        
        for (int i = 0; i < maxLabelLen; i++) System.out.print("─");
        System.out.print("─┼");
        for (int j = 0; j < n; j++) System.out.print("───────");
        System.out.println();

        for (int i = 0; i < n; i++) {
            System.out.printf("%" + maxLabelLen + "s │", lbls[i]);
            for (int j = 0; j < n; j++) {
                System.out.printf(" %6d", matrix[i][j]);
            }
            System.out.println();
        }
        System.out.println();
    }

    // ======================== GETTERS ========================

    public Predictor getPredictor() { return predictor; }
}
