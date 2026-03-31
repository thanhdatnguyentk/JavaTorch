package com.user.nn.examples;

import com.user.nn.utils.dashboard.DashboardServer;
import com.user.nn.utils.dashboard.DashboardIntegrationHelper;
import com.user.nn.utils.visualization.TrainingHistory;
import java.util.HashMap;
import java.util.Map;


import com.user.nn.core.*;
import com.user.nn.optim.*;
import com.user.nn.dataloaders.*;
import com.user.nn.metrics.*;
import com.user.nn.models.cv.LeNet;
import java.io.File;
import java.util.*;

/**
 * Train a classic LeNet-5 on Standard MNIST using the framework's autograd, 
 * optim.Adam, and cross_entropy_tensor loss.
 *
 * MNIST: 28x28 grayscale, 10 classes, 60k train / 10k test.
 */
public class TrainLeNet {

    // Standard MNIST Dataset
    static final String BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/";
    static final String DATA_DIR = "data/mnist/";

    public static void main(String[] args) throws Exception {
        MixedPrecision.enable(); // Enable Tensor Cores automatically if supported

        // --- Download data ---
        File dir = new File(DATA_DIR);
        if (!dir.exists()) dir.mkdirs();

        String[] files = {
                "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
                "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"
        };
        for (String f : files)
            MnistLoader.downloadIfMissing(BASE_URL + f, new File(DATA_DIR + f));

        // --- Load data ---
        System.out.println("Loading MNIST data...");
        float[][] trainImages = MnistLoader.loadImages(new File(DATA_DIR + "train-images-idx3-ubyte.gz"));
        int[] trainLabels = MnistLoader.loadLabels(new File(DATA_DIR + "train-labels-idx1-ubyte.gz"));
        float[][] testImages = MnistLoader.loadImages(new File(DATA_DIR + "t10k-images-idx3-ubyte.gz"));
        int[] testLabels = MnistLoader.loadLabels(new File(DATA_DIR + "t10k-labels-idx1-ubyte.gz"));
        System.out.println("Train: " + trainImages.length + " images, Test: " + testImages.length + " images");

        // --- Build model ---
        LeNet model = new LeNet();

        // Initialize parameters
        long seed = 42L;
        for (Parameter p : model.parameters()) {
            Tensor t = p.getTensor();
            float scale = (float) Math.sqrt(2.0 / t.numel());
            Random rng = new Random(seed++);
            for (int i = 0; i < t.data.length; i++) {
                t.data[i] = (float) (rng.nextGaussian() * scale);
            }
            t.requires_grad = true;
        }

        // --- Optimizer ---
        float lr = 0.001f;
        int epochs = 15;
        Optim.Adam optimizer = new Optim.Adam(model.parameters(), lr);

        // Initialize GPU Memory Pool based on model size
        GpuMemoryPool.autoInit(model);

        // Move model to GPU
        model.toGPU();

        // --- DataLoader setup ---
        int batchSize = 128;
        final int N = trainImages.length;
        final int inputDim = 784; 

        Accuracy trainAccMetric = new Accuracy();
        Accuracy testAccMetric = new Accuracy();

        Data.Dataset trainDataset = new Data.Dataset() {
            @Override
            public int len() { return N; }

            @Override
            public Tensor[] get(int index) {
                Tensor x = Torch.tensor(trainImages[index], 1, 28, 28);
                Tensor y = Torch.tensor(new float[] { trainLabels[index] }, 1);
                return new Tensor[] { x, y };
            }
        };

        Data.DataLoader trainLoader = new Data.DataLoader(trainDataset, batchSize, true, 4);

        System.out.println("Starting training for " + epochs + " epochs...");
        int totalBatches = N / batchSize;
        TrainingHistory history = new TrainingHistory();
        DashboardServer dashboard = new DashboardServer(7075, history).start();
        dashboard.setTaskType("classification");
        dashboard.setModelInfo("LeNet-5", epochs);
        String[] classLabels = com.user.nn.predict.ImagePredictor.MNIST_LABELS;
        try {
            com.user.nn.predict.ImagePredictor predictor = com.user.nn.predict.ImagePredictor.forMnist(model);
            DashboardIntegrationHelper.setupImagePredictorHandler(dashboard, "classify_image", predictor);
        } catch(Exception e) {}


        for (int epoch = 0; epoch < epochs; epoch++) {
            float epochLoss = 0f;
            int numBatches = 0;
            trainAccMetric.reset();
            model.train();

            for (Tensor[] batch : trainLoader) {
                try (MemoryScope scope = new MemoryScope()) {
                    Tensor xBatch = batch[0];
                    scope.track(xBatch);
                    scope.track(batch[1]);

                    xBatch.toGPU();

                    int bs = xBatch.shape[0];
                    int[] batchLabels = new int[bs];
                    for (int i = 0; i < bs; i++) {
                        batchLabels[i] = (int) batch[1].data[i];
                    }

                    optimizer.zero_grad();
                    Tensor logits = model.forward(xBatch);
                    Tensor loss = Functional.cross_entropy_tensor(logits, batchLabels);

                    loss.backward();
                    optimizer.step();

                    epochLoss += loss.data[0];
                    numBatches++;
                    trainAccMetric.update(logits, batchLabels);

                    // Batch-level broadcast
                    if (numBatches % 50 == 0) {
                        try {
                            Map<String, Object> bm = new HashMap<>();
                            bm.put("loss", loss.data[0]);
                            bm.put("batch", numBatches);
                            bm.put("train_acc", trainAccMetric.compute());
                            dashboard.broadcastTaskMetrics(epoch + 1, bm);
                        } catch (Exception e) {}
                        System.out.printf("  Epoch %d batch %d/%d  loss=%.4f%n",
                                epoch + 1, numBatches, totalBatches, loss.data[0]);
                    }
                    while (dashboard.isTrainingPaused()) {
                        try { Thread.sleep(200); } catch (InterruptedException ie) { break; }
                    }
                }
            }

            float trainAcc = trainAccMetric.compute();
            float avgLoss = epochLoss / numBatches;
            dashboard.setCurrentEpoch(epoch + 1);

            Data.Dataset testDataset = new Data.Dataset() {
                @Override
                public int len() { return testImages.length; }
                @Override
                public Tensor[] get(int index) {
                    Tensor x = Torch.tensor(testImages[index], 1, 28, 28);
                    Tensor y = Torch.tensor(new float[] { testLabels[index] }, 1);
                    return new Tensor[] { x, y };
                }
            };
            Data.DataLoader testLoader = new Data.DataLoader(testDataset, 256, false, 2);
            float testAcc = Evaluator.evaluate(model, testLoader, testAccMetric);

            // Confusion Matrix + Live Predictions
            int numClasses = 10;
            int[][] cm = new int[numClasses][numClasses];
            java.util.List<Map<String, Object>> livePreds = new java.util.ArrayList<>();
            model.eval();
            for (int ti = 0; ti < Math.min(200, testImages.length); ti++) {
                try (MemoryScope evalScope = new MemoryScope()) {
                    Tensor tx = Torch.tensor(testImages[ti], 1, 28, 28);
                    tx.toGPU();
                    Tensor out = model.forward(Torch.reshape(tx, 1, 1, 28, 28));
                    int pred = 0; float maxV = out.data[0];
                    for (int c = 1; c < numClasses; c++) { if (out.data[c] > maxV) { maxV = out.data[c]; pred = c; } }
                    cm[testLabels[ti]][pred]++;
                    if (livePreds.size() < 9) {
                        java.util.List<Map<String, Object>> topK = new java.util.ArrayList<>();
                        float[] sc = new float[numClasses]; float sum = 0;
                        for (int c = 0; c < numClasses; c++) { sc[c] = (float) Math.exp(out.data[c]); sum += sc[c]; }
                        for (int c = 0; c < numClasses; c++) sc[c] /= sum;
                        Integer[] idx = new Integer[numClasses]; for (int c = 0; c < numClasses; c++) idx[c] = c;
                        java.util.Arrays.sort(idx, (a, b) -> Float.compare(sc[b], sc[a]));
                        for (int k = 0; k < 3; k++) topK.add(DashboardIntegrationHelper.buildTopKEntry(classLabels[idx[k]], sc[idx[k]]));
                        livePreds.add(DashboardIntegrationHelper.buildLivePrediction(
                            DashboardIntegrationHelper.encodePixelsToBase64(testImages[ti], 1, 28, 28),
                            classLabels[pred], classLabels[testLabels[ti]], pred == testLabels[ti], topK));
                    }
                }
            }

            System.out.printf("Epoch %d/%d  avg_loss=%.4f  train_acc=%.4f  test_acc=%.4f%n",
                    epoch + 1, epochs, avgLoss, trainAcc, testAcc);
            testLoader.shutdown();

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
        trainLoader.shutdown();

        // ============================================================
        //  PREDICTION - Sử dụng thư viện predict
        // ============================================================
        System.out.println("\n╔══════════════════════════════════════════╗");
        System.out.println("║       PREDICTION WITH TRAINED MODEL      ║");
        System.out.println("╚══════════════════════════════════════════╝\n");

        model.save("lenet_mnist.bin");

        com.user.nn.predict.ImagePredictor predictor = 
            com.user.nn.predict.ImagePredictor.forMnist(model);
        predictor.topK(5).verbose(true);

        // Predict 10 test digits
        predictor.device(Tensor.Device.GPU);
        System.out.println(">>> Predicting 10 test digits with GPU...");
        int correct = 0;
        for (int i = 0; i < 10; i++) {
            com.user.nn.predict.PredictionResult result = predictor.predictFromPixels(testImages[i]);
            boolean ok = result.getPredictedClass() == testLabels[i];
            if (ok) correct++;
            System.out.printf("  Digit %d: predicted=%s, actual=%s %s%n",
                i, result.getPredictedLabel(),
                com.user.nn.predict.ImagePredictor.MNIST_LABELS[testLabels[i]],
                ok ? "✓" : "✗");
        }
        System.out.printf("  Accuracy: %d/%d%n", correct, 10);

        System.out.println("\nTraining Complete!");
        try { dashboard.exportDashboardData("dashboard_final.json"); } catch(Exception e) {}
    }
}
