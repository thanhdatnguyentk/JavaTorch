package com.user.nn.examples;

import com.user.nn.core.*;
import com.user.nn.dataloaders.*;
import com.user.nn.models.cv.ViT;
import com.user.nn.predict.*;

/**
 * Ví dụ sử dụng thư viện predict để thực hiện inference với model đã train.
 * 
 * Demo bao gồm:
 *   1. Load model đã train
 *   2. Predict đơn lẻ (single image)
 *   3. Predict batch
 *   4. Sử dụng ImagePredictor cho CIFAR-10
 *   5. Sử dụng PredictionPipeline (fluent API)
 *   6. BatchPredictor cho evaluation trên test set
 */
public class PredictDemo {

    public static void main(String[] args) throws Exception {
        System.out.println("╔══════════════════════════════════════════╗");
        System.out.println("║        PREDICTION LIBRARY DEMO           ║");
        System.out.println("╚══════════════════════════════════════════╝\n");

        // ============================================================
        // 1. Khởi tạo model (ViT cho CIFAR-10)
        // ============================================================
        System.out.println(">>> 1. Initializing ViT model...");
        ViT model = new ViT(32, 4, 3, 10, 64, 4, 4, 128, 0.1f);
        System.out.println("    Model parameters: " + model.countParameters());

        // Uncomment dòng bên dưới để load weights đã train:
        // model.load("vit_cifar10.bin");

        // ============================================================
        // 2. Predict đơn lẻ với Predictor cơ bản
        // ============================================================
        System.out.println("\n>>> 2. Basic Predictor - Single prediction");
        
        Predictor predictor = new Predictor(model, ImagePredictor.CIFAR10_LABELS)
            .topK(5)
            .verbose(true);

        // Tạo ảnh giả (random) để demo
        Tensor fakeImage = Torch.rand(new int[]{1, 3, 32, 32});
        PredictionResult result = predictor.predict(fakeImage);
        System.out.println(result);

        // ============================================================
        // 3. Predict class nhanh (chỉ lấy class index)
        // ============================================================
        System.out.println("\n>>> 3. Quick class prediction");
        int predictedClass = predictor.predictClass(fakeImage);
        String predictedLabel = predictor.predictLabel(fakeImage);
        System.out.println("    Predicted class: " + predictedClass);
        System.out.println("    Predicted label: " + predictedLabel);

        // ============================================================
        // 4. ImagePredictor cho CIFAR-10
        // ============================================================
        System.out.println("\n>>> 4. ImagePredictor for CIFAR-10");
        
        ImagePredictor imgPredictor = ImagePredictor.forCifar10(model)
            .normalize(
                new float[]{0.4914f, 0.4822f, 0.4465f},   // mean
                new float[]{0.2023f, 0.1994f, 0.2010f}     // std
            );
        imgPredictor.topK(3).verbose(true);

        // Dự đoán từ pixel array
        float[] fakePixels = new float[3 * 32 * 32];
        for (int i = 0; i < fakePixels.length; i++) {
            fakePixels[i] = (float) Math.random();
        }
        
        PredictionResult imgResult = imgPredictor.predictFromPixels(fakePixels);
        System.out.println(imgResult);

        // ============================================================
        // 5. Batch prediction
        // ============================================================
        System.out.println("\n>>> 5. Batch prediction");
        
        float[][] batchPixels = new float[4][3 * 32 * 32];
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < batchPixels[i].length; j++) {
                batchPixels[i][j] = (float) Math.random();
            }
        }
        
        PredictionResult[] batchResults = imgPredictor.predictFromPixelBatch(batchPixels);
        for (int i = 0; i < batchResults.length; i++) {
            System.out.printf("  Image %d: %s (%.4f)%n", 
                i, batchResults[i].getPredictedLabel(), batchResults[i].getConfidence());
        }

        // ============================================================
        // 6. PredictionPipeline (Fluent API)
        // ============================================================
        System.out.println("\n>>> 6. PredictionPipeline (Fluent API)");
        
        Predictor pipeline = PredictionPipeline
            .create(model)
            // .loadWeights("vit_cifar10.bin")  // Uncomment khi có file weights
            .labels(ImagePredictor.CIFAR10_LABELS)
            .imageSize(3, 32, 32)
            .normalize(
                new float[]{0.4914f, 0.4822f, 0.4465f},
                new float[]{0.2023f, 0.1994f, 0.2010f}
            )
            .topK(5)
            .verbose(true)
            .build();
        
        PredictionResult pipelineResult = pipeline.predict(fakeImage);
        System.out.println("    Pipeline result: " + pipelineResult.getPredictedLabel());

        // ============================================================
        // 7. BatchPredictor trên DataLoader (nếu có data)
        // ============================================================
        System.out.println("\n>>> 7. BatchPredictor on DataLoader");
        System.out.println("    (Skipped - cần data thật, uncomment code bên dưới)");
        
        /*
        // Uncomment khi có CIFAR-10 data:
        Cifar10Loader.prepareData();
        Object[] testBatch = Cifar10Loader.loadBatch("test_batch.bin");
        float[][] testImages = (float[][]) testBatch[0];
        int[] testLabels = (int[]) testBatch[1];

        Data.Dataset testDataset = new Data.Dataset() {
            public int len() { return 1000; }
            public Tensor[] get(int index) {
                Tensor x = Torch.tensor(testImages[index], 3, 32, 32);
                Tensor y = Torch.tensor(new float[] { testLabels[index] }, 1);
                return new Tensor[] { x, y };
            }
        };
        Data.DataLoader testLoader = new Data.DataLoader(testDataset, 64, false, 2);

        BatchPredictor batchPredictor = new BatchPredictor(predictor);
        float accuracy = batchPredictor.evaluateAccuracy(testLoader);
        System.out.println("    Test Accuracy: " + accuracy);

        // Confusion Matrix
        int[][] cm = batchPredictor.confusionMatrix(testLoader, 10);
        batchPredictor.printConfusionMatrix(cm, ImagePredictor.CIFAR10_LABELS);
        
        testLoader.shutdown();
        */

        System.out.println("\n╔══════════════════════════════════════════╗");
        System.out.println("║          DEMO COMPLETED!                 ║");
        System.out.println("╚══════════════════════════════════════════╝");
    }
}
