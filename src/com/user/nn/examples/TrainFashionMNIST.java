package com.user.nn.examples;

import com.user.nn.utils.dashboard.DashboardServer;
import com.user.nn.utils.dashboard.DashboardIntegrationHelper;
import com.user.nn.utils.visualization.TrainingHistory;
import java.util.HashMap;
import java.util.Map;
import java.io.File;
import java.util.*;

import com.user.nn.core.*;
import com.user.nn.containers.*;
import com.user.nn.layers.*;
import com.user.nn.pooling.*;
import com.user.nn.activations.*;
import com.user.nn.optim.*;
import com.user.nn.dataloaders.*;
import com.user.nn.metrics.*;

/**
 * Optimized Fashion-MNIST training using a Convolutional Neural Network (CNN).
 * Expected accuracy: >90% (vs ~40% for MLP).
 */
public class TrainFashionMNIST {

    static final String BASE_URL = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/";
    static final String DATA_DIR = "data/fashion-mnist/";

    public static void main(String[] args) throws Exception {
        // --- Download data ---
        String[] files = {
                "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
                "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"
        };
        for (String f : files)
            MnistLoader.downloadIfMissing(BASE_URL + f, new File(DATA_DIR + f));

        // --- Load data ---
        System.out.println("Loading data...");
        float[][] trainImages = MnistLoader.loadImages(new File(DATA_DIR + "train-images-idx3-ubyte.gz"));
        int[] trainLabels = MnistLoader.loadLabels(new File(DATA_DIR + "train-labels-idx1-ubyte.gz"));
        float[][] testImages = MnistLoader.loadImages(new File(DATA_DIR + "t10k-images-idx3-ubyte.gz"));
        int[] testLabels = MnistLoader.loadLabels(new File(DATA_DIR + "t10k-labels-idx1-ubyte.gz"));
        System.out.println("Train: " + trainImages.length + " images, Test: " + testImages.length + " images");

        // --- Build Optimized CNN model ---
        Sequential model = new Sequential();
        // Layer 1: 1 -> 16 channels, 3x3 kernel, pad 1 => 28x28
        model.add(new Conv2d(1, 16, 3, 3, 28, 28, 1, 1, true));
        model.add(new ReLU());
        // Pool: 28x28 -> 14x14
        model.add(new MaxPool2d(2, 2, 2, 2, 0, 0, 16, 28, 28));

        // Layer 2: 16 -> 32 channels, 3x3 kernel, pad 1 => 14x14
        model.add(new Conv2d(16, 32, 3, 3, 14, 14, 1, 1, true));
        model.add(new ReLU());
        // Pool: 14x14 -> 7x7
        model.add(new MaxPool2d(2, 2, 2, 2, 0, 0, 32, 14, 14));

        // Flatten features = 32 * 7 * 7 = 1568
        model.add(new com.user.nn.containers.Flatten());

        // Fully Connected
        model.add(new Linear(1568, 256, true));
        model.add(new ReLU());
        model.add(new Dropout(0.2f));
        model.add(new Linear(256, 10, true));

        // --- Optimizer & Params ---
        float lr = 0.01f;
        int epochs = SmokeTest.getEpochs(30);
        int batchSize = 1024; // Faster on GPU
        Optim.Adam optimizer = new Optim.Adam(model.parameters(), lr);

        // Movement & Init
        GpuMemoryPool.autoInit(model);
        model.toGPU();

        // --- DataLoader setup ---
        Accuracy trainAccMetric = new Accuracy();
        Accuracy testAccMetric = new Accuracy();

        Data.Dataset trainDataset = new Data.Dataset() {
            @Override
            public int len() { return trainImages.length; }

            @Override
            public Tensor[] get(int index) {
                // Preprocess: Normalize to [0,1] and Reshape to (1, 28, 28)
                float[] raw = trainImages[index];
                float[] norm = new float[raw.length];
                for (int i = 0; i < raw.length; i++) norm[i] = raw[i] / 255.0f;
                
                Tensor x = Torch.tensor(norm, 1, 28, 28);
                Tensor y = Torch.tensor(new float[] { trainLabels[index] }, 1);
                return new Tensor[] { x, y };
            }
        };

        Data.DataLoader trainLoader = new Data.DataLoader(trainDataset, batchSize, true, 4);

        System.out.println("Starting optimized training (CNN + Adam) for " + epochs + " epochs...");
        int totalBatches = trainImages.length / batchSize;
        TrainingHistory history = new TrainingHistory();
        DashboardServer dashboard = new DashboardServer(7071, history).start();
        dashboard.setTaskType("classification");
        dashboard.setModelInfo("CNN-FashionMNIST", epochs);
        String[] classLabels = {"T-shirt","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Boot"};

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

                    if (numBatches % 10 == 0) {
                        System.out.printf("  Epoch %d batch %d/%d  loss=%.4f%n",
                                epoch + 1, numBatches, totalBatches, loss.data[0]);
                        try {
                            Map<String, Object> batchMetrics = new HashMap<>();
                            batchMetrics.put("loss", loss.data[0]);
                            batchMetrics.put("batch", numBatches);
                            batchMetrics.put("train_acc", trainAccMetric.compute());
                            dashboard.broadcastTaskMetrics(epoch + 1, batchMetrics);
                        } catch (Exception e) {}
                    }
                    while (dashboard.isTrainingPaused()) {
                        try { Thread.sleep(200); } catch (InterruptedException ie) { break; }
                    }
                }
            }

            float trainAcc = trainAccMetric.compute();
            float avgLoss = epochLoss / numBatches;
            dashboard.setCurrentEpoch(epoch + 1);

            // Evaluation on Test set
            Data.Dataset testDataset = new Data.Dataset() {
                @Override
                public int len() { return testImages.length; }
                @Override
                public Tensor[] get(int index) {
                    float[] raw = testImages[index];
                    float[] norm = new float[raw.length];
                    for (int i = 0; i < raw.length; i++) norm[i] = raw[i] / 255.0f;
                    return new Tensor[] { Torch.tensor(norm, 1, 28, 28), Torch.tensor(new float[] { testLabels[index] }, 1) };
                }
            };
            Data.DataLoader testLoader = new Data.DataLoader(testDataset, 1024, false, 2);
            float testAcc = Evaluator.evaluate(model, testLoader, testAccMetric);
            testLoader.shutdown();

            // Confusion Matrix + Live Predictions
            int numClasses = 10;
            int[][] cm = new int[numClasses][numClasses];
            java.util.List<Map<String, Object>> livePreds = new java.util.ArrayList<>();
            model.eval();
            for (int ti = 0; ti < Math.min(200, testImages.length); ti++) {
                try (MemoryScope evalScope = new MemoryScope()) {
                    float[] raw = testImages[ti]; float[] norm = new float[raw.length];
                    for (int j = 0; j < raw.length; j++) norm[j] = raw[j] / 255.0f;
                    Tensor tx = Torch.tensor(norm, 1, 28, 28); tx.toGPU();
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
                            DashboardIntegrationHelper.encodePixelsToBase64(norm, 1, 28, 28),
                            classLabels[pred], classLabels[testLabels[ti]], pred == testLabels[ti], topK));
                    }
                }
            }

            System.out.printf("Epoch %d/%d  avg_loss=%.4f  train_acc=%.4f  test_acc=%.4f%n",
                    epoch + 1, epochs, avgLoss, trainAcc, testAcc);

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
        model.save("fashion_mnist_cnn.bin");
        System.out.println("\nOptimized Training Complete!");
        try { dashboard.exportDashboardData("dashboard_final.json"); } catch(Exception e) {}
    }
}
