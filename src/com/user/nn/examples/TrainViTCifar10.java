package com.user.nn.examples;

import com.user.nn.core.*;
import com.user.nn.optim.*;
import com.user.nn.optim.*;
import com.user.nn.dataloaders.*;
import com.user.nn.metrics.*;
import com.user.nn.models.cv.ViT;
import java.util.*;

/**
 * Train a Vision Transformer (ViT) on CIFAR-10 using pure Java framework.
 */
public class TrainViTCifar10 {

    public static void main(String[] args) throws Exception {
        // MixedPrecision.enable(); // Opt-in to Tensor Cores if supported

        System.out.println("Preparing CIFAR-10 data...");
        Cifar10Loader.prepareData();

        System.out.println("Loading subsets of CIFAR-10 clusters...");
        // Use fewer batches for the example demonstration to save time
        int trainBatches = 2; 
        float[][] trainImages = new float[trainBatches * 10000][3072];
        int[] trainLabels = new int[trainBatches * 10000];

        for (int i = 1; i <= trainBatches; i++) {
            Object[] batch = Cifar10Loader.loadBatch("data_batch_" + i + ".bin");
            float[][] imgs = (float[][]) batch[0];
            int[] lbls = (int[]) batch[1];
            System.arraycopy(imgs, 0, trainImages, (i - 1) * 10000, 10000);
            System.arraycopy(lbls, 0, trainLabels, (i - 1) * 10000, 10000);
        }

        Object[] testBatch = Cifar10Loader.loadBatch("test_batch.bin");
        float[][] testImages = (float[][]) testBatch[0];
        int[] testLabels = (int[]) testBatch[1];

        NN lib = new NN();
        
        // ViT Parameters: imgSize=32, patchSize=4, inChannels=3, numClasses=10
        // Small configuration to avoid OOM and speed up training:
        // embedDim=64, depth=4, numHeads=4, mlpDim=128
        System.out.println("Initializing Vision Transformer (ViT)...");
        ViT model = new ViT(32, 4, 3, 10, 64, 4, 4, 128, 0.1f);
        
        // Move to GPU
        System.out.println("Moving ViT to GPU...");
        model.to(Tensor.Device.GPU);

        float lr = 0.0005f; // ViT often needs smaller learning rate than CNN
        int epochs = 20; 
        int batchSize = 128;
        Optim.Adam optimizer = new Optim.Adam(model.parameters(), lr);

        final int N = trainImages.length;
        Data.Dataset trainDataset = new Data.Dataset() {
            @Override
            public int len() { return N; }
            @Override
            public Tensor[] get(int index) {
                // Pre-reshape to [3, 32, 32]
                Tensor x = Torch.tensor(trainImages[index], 3, 32, 32);
                Tensor y = Torch.tensor(new float[] { trainLabels[index] }, 1);
                return new Tensor[] { x, y };
            }
        };

        Data.DataLoader loader = new Data.DataLoader(trainDataset, batchSize, true, 4);
        
        // --- 1. Move Test Loader Out of Epoch Loop ---
        Data.Dataset testDataset = new Data.Dataset() {
            @Override
            public int len() { return 2000; } // Only evaluate on subset for speed in example
            @Override
            public Tensor[] get(int index) {
                Tensor x = Torch.tensor(testImages[index], 3, 32, 32);
                Tensor y = Torch.tensor(new float[] { testLabels[index] }, 1);
                return new Tensor[] { x, y };
            }
        };
        Data.DataLoader testLoader = new Data.DataLoader(testDataset, 128, false, 2);
        
        Accuracy accMetric = new Accuracy();
        
        // --- 4. Add Optimizer Scheduler ---
        Scheduler.StepLR scheduler = new Scheduler.StepLR(optimizer, 5, 0.5f); // Half LR every 5 epochs

        System.out.println("Starting Training Loop...");
        for (int epoch = 0; epoch < epochs; epoch++) {
            float epochLoss = 0f;
            int numBatches = 0;
            accMetric.reset();
            
            // Ensure training mode
            model.train();

            long epochStart = System.currentTimeMillis();
            for (Tensor[] batch : loader) {
                try (MemoryScope scope = new MemoryScope()) {
                    Tensor xBatch = batch[0].to(Tensor.Device.GPU);
                    scope.track(xBatch);
                    
                    int bs = xBatch.shape[0];
                    int[] batchLabels = new int[bs];
                    for (int i = 0; i < bs; i++) batchLabels[i] = (int) batch[1].data[i];

                    optimizer.zero_grad();
                    Tensor logits = model.forward(xBatch);
                    Tensor loss = Functional.cross_entropy_tensor(logits, batchLabels);
                    loss.backward();
                    optimizer.step();

                    epochLoss += loss.data[0];
                    numBatches++;
                    accMetric.update(logits, batchLabels);

                    if (numBatches % 20 == 0) {
                        System.out.printf("  [Epoch %d/%d] Batch %d/%d | Loss: %.4f | Acc: %.4f | LR: %.6f%n",
                                epoch + 1, epochs, numBatches, N / batchSize, loss.data[0], accMetric.compute(), optimizer.getLearningRate());
                    }
                }
            }
            long epochEnd = System.currentTimeMillis();

            float trainAcc = accMetric.compute();
            float avgLoss = epochLoss / numBatches;
            
            // --- 2. Quản lý trạng thái Evaluator rõ ràng ---
            model.eval(); // Explicit eval mode
            float testAcc = Evaluator.evaluate(model, testLoader, accMetric);
            
        System.out.printf(">>> Epoch %d/%d | Loss: %.4f | Train Acc: %.4f | Test Acc: %.4f | Time: %dms%n",
                    epoch + 1, epochs, avgLoss, trainAcc, testAcc, (epochEnd - epochStart));
            
            // Step Scheduler
            scheduler.step();
        }
        
        loader.shutdown();
        testLoader.shutdown();
        
        // --- 3. Tích hợp Checkpointing (Model Saving) ---
        System.out.println("Saving model...");
        model.save("vit_cifar10.bin");

        // ============================================================
        //  PREDICTION DEMO - Sử dụng thư viện predict
        // ============================================================
        System.out.println("\n╔══════════════════════════════════════════╗");
        System.out.println("║       PREDICTION WITH TRAINED MODEL      ║");
        System.out.println("╚══════════════════════════════════════════╝\n");

        // 1. Tạo ImagePredictor từ model đã train
        com.user.nn.predict.ImagePredictor predictor = 
            com.user.nn.predict.ImagePredictor.forCifar10(model);
        predictor.topK(5).verbose(true);

        // 2. Predict một ảnh từ test set
        System.out.println(">>> Predicting a single test image...");
        float[] sampleImage = testImages[0];
        com.user.nn.predict.PredictionResult result = predictor.predictFromPixels(sampleImage);
        System.out.println(result);
        System.out.println("    Actual label: " + com.user.nn.predict.ImagePredictor.CIFAR10_LABELS[testLabels[0]]);

        // 3. Batch predict trên vài ảnh test
        System.out.println("\n>>> Batch predicting 10 test images...");
        float[][] sampleBatch = new float[10][];
        for (int i = 0; i < 10; i++) sampleBatch[i] = testImages[i];
        com.user.nn.predict.PredictionResult[] batchResults = predictor.predictFromPixelBatch(sampleBatch);
        
        int correct = 0;
        for (int i = 0; i < batchResults.length; i++) {
            boolean isCorrect = batchResults[i].getPredictedClass() == testLabels[i];
            if (isCorrect) correct++;
            System.out.printf("  Image %d: predicted=%s, actual=%s %s%n",
                i, batchResults[i].getPredictedLabel(),
                com.user.nn.predict.ImagePredictor.CIFAR10_LABELS[testLabels[i]],
                isCorrect ? "✓" : "✗");
        }
        System.out.printf("  Batch accuracy: %d/%d%n", correct, 10);
        
        System.out.println("\nTraining Complete!");
    }
}
