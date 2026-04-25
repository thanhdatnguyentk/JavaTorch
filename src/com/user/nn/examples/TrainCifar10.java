package com.user.nn.examples;

import com.user.nn.utils.dashboard.DashboardServer;
import com.user.nn.utils.dashboard.DashboardIntegrationHelper;
import com.user.nn.utils.visualization.TrainingHistory;
import java.util.HashMap;
import java.util.Map;


import com.user.nn.core.*;
import com.user.nn.containers.*;
import com.user.nn.layers.*;
import com.user.nn.activations.*;
import com.user.nn.pooling.*;
import com.user.nn.optim.*;
import com.user.nn.dataloaders.*;
import com.user.nn.metrics.*;
import java.io.*;
import java.util.*;
import jcuda.runtime.JCuda;

/**
 * Train a CNN on CIFAR-10 using pure Java framework with autograd and
 * optim.Adam.
 */
public class TrainCifar10 {

    public static void main(String[] args) throws Exception {
        MixedPrecision.enable(); // Opt-in to Tensor Cores (FP16)


        System.out.println("Preparing CIFAR-10 data...");
        Cifar10Loader.prepareData();

        System.out.println("Loading CIFAR-10 data into memory...");
        int trainBatches = SmokeTest.getBatches(5);
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
        Sequential model = new Sequential();
        model.add(new Conv2d(3, 16, 3, 3, 32, 32, 1, 1, true));
        model.add(new ReLU());
        model.add(new MaxPool2d(2, 2, 2, 2, 0, 0, 16, 32, 32));
        model.add(new Conv2d(16, 32, 3, 3, 16, 16, 1, 1, true));
        model.add(new ReLU());
        model.add(new MaxPool2d(2, 2, 2, 2, 0, 0, 32, 16, 16));

        // Flatten conv output to (batch, 32*8*8)
        model.add(new com.user.nn.containers.Flatten());
        int flattenSize = 32 * 8 * 8; 
        model.add(new Linear(flattenSize, 128, true));
        model.add(new ReLU());
        model.add(new Dropout(0.3f));
        model.add(new Linear(128, 10, true));

        long seed = 123L;
        for (Parameter p : model.parameters()) {
            Tensor t = p.getTensor();
            float scale = (float) Math.sqrt(2.0 / t.numel());
            Random rng = new Random(seed++);
            for (int i = 0; i < t.data.length; i++) {
                t.data[i] = (float) (rng.nextGaussian() * scale);
            }
            t.requires_grad = true;
        }

        float lr = 0.001f;
        int epochs = SmokeTest.getEpochs(100); 
        int batchSize = 128;
        Optim.Adam optimizer = new Optim.Adam(model.parameters(), lr);

        // Initialize GPU Memory Pool based on model size
        GpuMemoryPool.autoInit(model);

        // Move to GPU
        System.out.println("Moving model to GPU...");
        model.toGPU();
        System.out.println("Model moved to GPU");

        final int N = trainImages.length;
        Data.Dataset trainDataset = new Data.Dataset() {
            @Override
            public int len() { return N; }
            @Override
            public Tensor[] get(int index) {
                Tensor x = Torch.tensor(trainImages[index], 3, 32, 32);
                Tensor y = Torch.tensor(new float[] { trainLabels[index] }, 1);
                return new Tensor[] { x, y };
            }
        };

        Data.DataLoader loader = new Data.DataLoader(trainDataset, batchSize, true, 4);
        Accuracy accMetric = new Accuracy();
        TrainingHistory history = new TrainingHistory();
        DashboardServer dashboard = new DashboardServer(7070, history).start();
        dashboard.setTaskType("classification");
        dashboard.setModelInfo("CNN-CIFAR10", epochs);
        String[] classLabels = com.user.nn.predict.ImagePredictor.CIFAR10_LABELS;
        try {
            com.user.nn.predict.ImagePredictor predictor = com.user.nn.predict.ImagePredictor.forCifar10(model);
            DashboardIntegrationHelper.setupImagePredictorHandler(dashboard, "classify_image", predictor);
        } catch(Exception e) {}


        for (int epoch = 0; epoch < epochs; epoch++) {
            float epochLoss = 0f;
            int numBatches = 0;
            accMetric.reset();
            model.train();

            for (Tensor[] batch : loader) {
                try (MemoryScope scope = new MemoryScope()) {
                    Tensor xBatch = batch[0];
                    scope.track(xBatch);
                    scope.track(batch[1]);
                    
                    xBatch.toGPU();
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

                    // Batch-level broadcast for real-time dashboard
                    if (numBatches % 10 == 0) {
                        try {
                            Map<String, Object> batchMetrics = new HashMap<>();
                            batchMetrics.put("loss", loss.data[0]);
                            batchMetrics.put("batch", numBatches);
                            batchMetrics.put("train_acc", accMetric.compute());
                            dashboard.broadcastTaskMetrics(epoch + 1, batchMetrics);
                        } catch (Exception e) {}
                        System.out.printf("  Epoch %d batch %d/%d  loss=%.4f%n",
                                epoch + 1, numBatches, N / batchSize, loss.data[0]);
                    }

                    // Pause support
                    while (dashboard.isTrainingPaused()) {
                        try { Thread.sleep(200); } catch (InterruptedException ie) { break; }
                    }
                }
            }

            float trainAcc = accMetric.compute();
            float avgLoss = epochLoss / numBatches;
            dashboard.setCurrentEpoch(epoch + 1);

            // Evaluation
            Data.Dataset testDataset = new Data.Dataset() {
                @Override
                public int len() { return testImages.length; }
                @Override
                public Tensor[] get(int index) {
                    Tensor x = Torch.tensor(testImages[index], 3, 32, 32);
                    Tensor y = Torch.tensor(new float[] { testLabels[index] }, 1);
                    return new Tensor[] { x, y };
                }
            };
            Data.DataLoader testLoader = new Data.DataLoader(testDataset, 256, false, 2);
            float testAcc = Evaluator.evaluate(model, testLoader, accMetric);

            // --- Confusion Matrix ---
            int numClasses = 10;
            int[][] cm = new int[numClasses][numClasses];
            List<Map<String, Object>> livePreds = new ArrayList<>();
            model.eval();
            int sampleCount = 0;
            for (int ti = 0; ti < Math.min(200, testImages.length); ti++) {
                try (MemoryScope evalScope = new MemoryScope()) {
                    Tensor tx = Torch.tensor(testImages[ti], 3, 32, 32);
                    tx.toGPU();
                    Tensor out = model.forward(Torch.reshape(tx, 1, 3, 32, 32));
                    int pred = 0; float maxVal = out.data[0];
                    for (int c = 1; c < numClasses; c++) {
                        if (out.data[c] > maxVal) { maxVal = out.data[c]; pred = c; }
                    }
                    cm[testLabels[ti]][pred]++;

                    // Build live prediction grid (up to 9 samples)
                    if (sampleCount < 9) {
                        List<Map<String, Object>> topK = new ArrayList<>();
                        float[] scores = new float[numClasses];
                        float sum = 0;
                        for (int c = 0; c < numClasses; c++) { scores[c] = (float) Math.exp(out.data[c]); sum += scores[c]; }
                        for (int c = 0; c < numClasses; c++) scores[c] /= sum;
                        Integer[] indices = new Integer[numClasses];
                        for (int c = 0; c < numClasses; c++) indices[c] = c;
                        Arrays.sort(indices, (a, b) -> Float.compare(scores[b], scores[a]));
                        for (int k = 0; k < Math.min(3, numClasses); k++) {
                            topK.add(DashboardIntegrationHelper.buildTopKEntry(classLabels[indices[k]], scores[indices[k]]));
                        }
                        String imgB64 = DashboardIntegrationHelper.encodePixelsToBase64(testImages[ti], 3, 32, 32);
                        livePreds.add(DashboardIntegrationHelper.buildLivePrediction(
                            imgB64, classLabels[pred], classLabels[testLabels[ti]],
                            pred == testLabels[ti], topK));
                        sampleCount++;
                    }
                }
            }

            System.out.printf("Epoch %d/%d  avg_loss=%.4f  train_acc=%.4f  test_acc=%.4f%n",
                    epoch + 1, epochs, avgLoss, trainAcc, testAcc);
            testLoader.shutdown();

            // Rich dashboard broadcast
            try {
                Map<String, Float> metrics = new HashMap<>();
                metrics.put("loss", avgLoss);
                metrics.put("train_acc", trainAcc);
                metrics.put("test_acc", testAcc);
                history.record(epoch + 1, metrics);
                DashboardIntegrationHelper.broadcastClassificationDetailed(
                    dashboard, epoch + 1, metrics, cm, classLabels, livePreds);
            } catch (Exception dashEx) {}
        }
        loader.shutdown();

        // ============================================================
        //  PREDICTION - Sử dụng thư viện predict
        // ============================================================
        System.out.println("\n>>> Saving model...");
        model.save("cifar10_cnn.bin");

        System.out.println("\n╔══════════════════════════════════════════╗");
        System.out.println("║       PREDICTION WITH TRAINED MODEL      ║");
        System.out.println("╚══════════════════════════════════════════╝\n");

        com.user.nn.predict.ImagePredictor predictor = 
            com.user.nn.predict.ImagePredictor.forCifar10(model);
        predictor.topK(5).verbose(true);

        // Predict single image
        System.out.println(">>> Predicting a single test image...");
        com.user.nn.predict.PredictionResult result = predictor.predictFromPixels(testImages[0]);
        System.out.println(result);
        System.out.println("    Actual: " + com.user.nn.predict.ImagePredictor.CIFAR10_LABELS[testLabels[0]]);

        // Batch predict 10 images
        System.out.println("\n>>> Batch predicting 10 test images...");
        float[][] sampleBatch = new float[10][];
        for (int i = 0; i < 10; i++) sampleBatch[i] = testImages[i];
        com.user.nn.predict.PredictionResult[] batchResults = predictor.predictFromPixelBatch(sampleBatch);

        int correct = 0;
        for (int i = 0; i < batchResults.length; i++) {
            boolean ok = batchResults[i].getPredictedClass() == testLabels[i];
            if (ok) correct++;
            System.out.printf("  Image %d: predicted=%s, actual=%s %s%n",
                i, batchResults[i].getPredictedLabel(),
                com.user.nn.predict.ImagePredictor.CIFAR10_LABELS[testLabels[i]],
                ok ? "✓" : "✗");
        }
        System.out.printf("  Batch accuracy: %d/%d%n", correct, 10);

        System.out.println("\nTraining Complete!");
        try { dashboard.exportDashboardData("dashboard_final.json"); } catch(Exception e) {}
    }

}
