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
        DashboardServer dashboard = new DashboardServer(7070, history).start();
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

                    if (numBatches % 100 == 0) {
                        System.out.printf("  Epoch %d batch %d/%d  loss=%.4f%n",
                                epoch + 1, numBatches, totalBatches, loss.data[0]);
                    }
                }
            }

            float trainAcc = trainAccMetric.compute();
            float avgLoss = epochLoss / numBatches;
            
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

            System.out.printf("Epoch %d/%d  avg_loss=%.4f  train_acc=%.4f  test_acc=%.4f%n",
                    epoch + 1, epochs, avgLoss, trainAcc, testAcc);
                    
            testLoader.shutdown();
        
            try {
                Map<String, Float> metrics = new HashMap<>();
                metrics.put("loss", avgLoss);
                metrics.put("train_acc", trainAcc);
                metrics.put("test_acc", testAcc);
                history.record(epoch + 1, metrics);
                dashboard.broadcastMetrics(epoch + 1, metrics);
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
        System.out.println(">>> Predicting 10 test digits...");
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
