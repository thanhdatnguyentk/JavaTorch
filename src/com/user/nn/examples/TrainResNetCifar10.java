package com.user.nn.examples;

import com.user.nn.core.*;
import com.user.nn.optim.*;
import com.user.nn.dataloaders.*;
import com.user.nn.metrics.*;
import com.user.nn.models.cv.ResNet;
import com.user.nn.utils.dashboard.DashboardServer;
import com.user.nn.utils.dashboard.DashboardIntegrationHelper;
import com.user.nn.utils.progress.ProgressDataLoader;
import com.user.nn.utils.visualization.*;
import com.user.nn.utils.visualization.exporters.*;
import java.io.*;
import java.util.*;

/**
 * Train ResNet-18 on CIFAR-10.
 * Demonstrates the power of ResNet with skip connections and BatchNorm.
 */
public class TrainResNetCifar10 {

    public static void main(String[] args) throws Exception {
        MixedPrecision.enable(); // Enable FP16 (Tensor Cores)

        System.out.println("Preparing CIFAR-10 data...");
        Cifar10Loader.prepareData();

        System.out.println("Loading CIFAR-10 data into memory...");
        int trainBatches = 5;
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
        // Create ResNet-18 for CIFAR-10
        System.out.println("Creating ResNet-18 model...");
        ResNet model = ResNet.resnet18(10, 32, 32);
        
        System.out.println("Total parameters: " + model.countParameters());

        // Kaiming initialization (simplified)
        for (Parameter p : model.parameters()) {
            Tensor t = p.getTensor();
            if (t.dim() >= 2) {
                float scale = (float) Math.sqrt(2.0 / t.numel());
                Random rng = new Random();
                for (int i = 0; i < t.data.length; i++) {
                    t.data[i] = (float) (rng.nextGaussian() * scale);
                }
            }
            t.requires_grad = true;
        }

        float lr = 0.001f;
        int epochs = 2; 
        int batchSize = 64; // Reduced batch size for ResNet-18 on 3050 (4GB-8GB VRAM)
        Optim.Adam optimizer = new Optim.Adam(model.parameters(), lr);

        // Initialize GPU Memory Pool based on model size
        GpuMemoryPool.autoInit(model);

        // Move to GPU
        System.out.println("Moving ResNet to GPU...");
        model.toGPU();
        System.out.println("Model ready on GPU.");

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
        System.out.println("\n=== Training with Progress Bar & Visualization ===\n");
        DashboardServer dashboard = new DashboardServer(7070, history).start();
        dashboard.setTaskType("classification");
        dashboard.setModelInfo("ResNet-18", epochs);
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

            long startTime = System.currentTimeMillis();
            
            // Wrap DataLoader with ProgressDataLoader
            ProgressDataLoader progLoader = new ProgressDataLoader(
                loader, String.format("Epoch %d/%d", epoch + 1, epochs)
            );

            for (Tensor[] batch : progLoader) {
                try (MemoryScope scope = new MemoryScope()) {
                    Tensor xBatch = batch[0];
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

                    // Update progress bar with live metrics
                    progLoader.setPostfix("loss", String.format("%.4f", loss.data[0]));
                    progLoader.setPostfix("acc", String.format("%.4f", accMetric.compute()));

                    // Batch-level dashboard broadcast
                    if (numBatches % 10 == 0) {
                        try {
                            Map<String, Object> bm = new HashMap<>();
                            bm.put("loss", loss.data[0]);
                            bm.put("batch", numBatches);
                            bm.put("train_acc", accMetric.compute());
                            dashboard.broadcastTaskMetrics(epoch + 1, bm);
                        } catch (Exception e) {}
                    }

                    while (dashboard.isTrainingPaused()) {
                        try { Thread.sleep(200); } catch (InterruptedException ie) { break; }
                    }
                }
            }

            long endTime = System.currentTimeMillis();
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
            Data.DataLoader testLoader = new Data.DataLoader(testDataset, 128, false, 2);
            float testAcc = Evaluator.evaluate(model, testLoader, accMetric);

            // Confusion Matrix + Live Predictions
            int numClasses = 10;
            int[][] cm = new int[numClasses][numClasses];
            List<Map<String, Object>> livePreds = new ArrayList<>();
            model.eval();
            for (int ti = 0; ti < Math.min(200, testImages.length); ti++) {
                try (MemoryScope evalScope = new MemoryScope()) {
                    Tensor tx = Torch.tensor(testImages[ti], 3, 32, 32);
                    tx.toGPU();
                    Tensor out = model.forward(Torch.reshape(tx, 1, 3, 32, 32));
                    int pred = 0; float maxV = out.data[0];
                    for (int c = 1; c < numClasses; c++) { if (out.data[c] > maxV) { maxV = out.data[c]; pred = c; } }
                    cm[testLabels[ti]][pred]++;
                    if (livePreds.size() < 9) {
                        List<Map<String, Object>> topK = new ArrayList<>();
                        float[] sc = new float[numClasses]; float sum = 0;
                        for (int c = 0; c < numClasses; c++) { sc[c] = (float) Math.exp(out.data[c]); sum += sc[c]; }
                        for (int c = 0; c < numClasses; c++) sc[c] /= sum;
                        Integer[] idx = new Integer[numClasses];
                        for (int c = 0; c < numClasses; c++) idx[c] = c;
                        Arrays.sort(idx, (a, b) -> Float.compare(sc[b], sc[a]));
                        for (int k = 0; k < 3; k++) topK.add(DashboardIntegrationHelper.buildTopKEntry(classLabels[idx[k]], sc[idx[k]]));
                        livePreds.add(DashboardIntegrationHelper.buildLivePrediction(
                            DashboardIntegrationHelper.encodePixelsToBase64(testImages[ti], 3, 32, 32),
                            classLabels[pred], classLabels[testLabels[ti]], pred == testLabels[ti], topK));
                    }
                }
            }

            System.out.printf("Epoch %d/%d  avg_loss=%.4f  train_acc=%.4f  test_acc=%.4f  time=%dms%n",
                    epoch + 1, epochs, avgLoss, trainAcc, testAcc, (endTime - startTime));

            // Rich dashboard broadcast
            Map<String, Float> metrics = new HashMap<>();
            metrics.put("train_loss", avgLoss);
            metrics.put("train_acc", trainAcc);
            metrics.put("test_acc", testAcc);
            history.record(epoch, metrics);
            try {
                DashboardIntegrationHelper.broadcastClassificationDetailed(
                    dashboard, epoch + 1, metrics, cm, classLabels, livePreds);
            } catch (Exception dashEx) {}

            testLoader.shutdown();
        }
        loader.shutdown();
        
        // Save training history and visualizations
        System.out.println("\n=== Saving Training Visualizations ===");
        try {
            // Save training curves
            Plot curves = history.plot();
            PlotContext ctx = new PlotContext()
                .title("ResNet-18 Training on CIFAR-10")
                .xlabel("Epoch")
                .ylabel("Metric Value")
                .grid(true);
            FileExporter.savePNG(curves, ctx, "resnet_training_curves.png", 800, 600);
            System.out.println("Saved training curves to: resnet_training_curves.png");
            
            // Save history to CSV
            history.saveCSV("resnet_training_history.csv");
            System.out.println("Saved training history to: resnet_training_history.csv");
            
            // Report best results
            float bestTestAcc = history.getMax("test_acc");
            int bestEpoch = history.getMaxEpoch("test_acc");
            System.out.printf("\nBest test accuracy: %.4f at epoch %d\n", bestTestAcc, bestEpoch + 1);
            
        } catch (Exception e) {
            System.err.println("Warning: Could not save visualizations: " + e.getMessage());
        }
    }
}
