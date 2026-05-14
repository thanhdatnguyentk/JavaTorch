package com.user.nn.examples;

import com.user.nn.core.*;
import com.user.nn.containers.Sequential;
import com.user.nn.models.cv.ResNet;
import com.user.nn.models.cv.ViT;
import com.user.nn.layers.*;
import com.user.nn.pooling.*;
import com.user.nn.activations.*;
import com.user.nn.norm.*;
import com.user.nn.optim.*;
import com.user.nn.dataloaders.*;
import com.user.nn.metrics.*;
import com.user.nn.losses.*;
import com.user.nn.utils.visualization.TrainingHistory;

import java.io.File;
import java.io.PrintWriter;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;

public class SotaBenchmarkRunner {

    static class BenchmarkResult {
        String task;
        String architecture;
        long params;
        float trainLoss;
        float testAcc;
        long timeMs;
    }

    public static void main(String[] args) throws Exception {
        // MixedPrecision.enable(); // Disable to prevent cudaErrorIllegalAddress during sequential runs
        List<BenchmarkResult> results = new ArrayList<>();
        
        int epochs = SmokeTest.getEpochs(1);
        
        System.out.println("========== SOTA BENCHMARK SUITE ==========");
        System.out.println("Running each task for " + epochs + " epoch(s).\n");

        /*
        try {
            results.add(runFashionMnistBenchmark(epochs));
            System.gc(); Thread.sleep(1000);
        } catch (Exception e) {
            System.err.println("FashionMNIST Benchmark Failed: " + e.getMessage());
            e.printStackTrace();
        }
        */

        try {
            results.add(runCifar10ResNetBenchmark(epochs));
            System.gc(); Thread.sleep(1000);
        } catch (Exception e) {
            System.err.println("ResNet CIFAR-10 Benchmark Failed: " + e.getMessage());
            e.printStackTrace();
        }

        try {
            results.add(runCifar10ViTBenchmark(epochs));
        } catch (Exception e) {
            System.err.println("ViT CIFAR-10 Benchmark Failed: " + e.getMessage());
            e.printStackTrace();
        }

        File dir = new File("benchmark");
        if (!dir.exists()) dir.mkdirs();
        File csvFile = new File(dir, "benchmark_results.csv");
        
        try (PrintWriter writer = new PrintWriter(new FileWriter(csvFile))) {
            writer.println("Task,Architecture,Parameters,Train_Loss,Test_Accuracy,Time_ms");
            for (BenchmarkResult r : results) {
                writer.printf("%s,%s,%d,%.4f,%.4f,%d\n", 
                    r.task, r.architecture, r.params, r.trainLoss, r.testAcc, r.timeMs);
                System.out.printf("[RESULT] %s | %s | Params: %d | Loss: %.4f | Acc: %.4f | Time: %dms\n",
                    r.task, r.architecture, r.params, r.trainLoss, r.testAcc, r.timeMs);
            }
        }
        System.out.println("\nSaved benchmark results to " + csvFile.getAbsolutePath());
    }

    private static BenchmarkResult runFashionMnistBenchmark(int epochs) throws Exception {
        System.out.println("--- Starting FashionMNIST SOTA CNN ---");
        String baseUrl = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/";
        String dataDir = "data/fashion-mnist/";
        String[] files = {"train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"};
        for (String f : files) MnistLoader.downloadIfMissing(baseUrl + f, new File(dataDir + f));

        float[][] trainImages = MnistLoader.loadImages(new File(dataDir + "train-images-idx3-ubyte.gz"));
        int[] trainLabels = MnistLoader.loadLabels(new File(dataDir + "train-labels-idx1-ubyte.gz"));
        float[][] testImages = MnistLoader.loadImages(new File(dataDir + "t10k-images-idx3-ubyte.gz"));
        int[] testLabels = MnistLoader.loadLabels(new File(dataDir + "t10k-labels-idx1-ubyte.gz"));

        Sequential model = new Sequential();
        model.add(new Conv2d(1, 16, 3, 3, 1, 1, 1, 1, true));
        model.add(new BatchNorm2d(16));
        model.add(new ReLU());
        model.add(new MaxPool2d(2, 2, 2, 2, 0, 0, 16, 28, 28));
        model.add(new Conv2d(16, 32, 3, 3, 1, 1, 1, 1, true));
        model.add(new BatchNorm2d(32));
        model.add(new ReLU());
        model.add(new MaxPool2d(2, 2, 2, 2, 0, 0, 32, 14, 14));
        model.add(new com.user.nn.containers.Flatten());
        model.add(new Linear(1568, 256, true));
        model.add(new ReLU());
        model.add(new Dropout(0.2f));
        model.add(new Linear(256, 10, true));

        GpuMemoryPool.autoInit(model);
        model.toGPU();
        Optim.Adam optimizer = new Optim.Adam(model.parameters(), 0.01f);

        int N = SmokeTest.isEnabled() ? 1024 * 5 : trainImages.length;
        Data.Dataset trainDataset = new Data.Dataset() {
            @Override public int len() { return N; }
            @Override public Tensor[] get(int index) {
                float[] raw = trainImages[index];
                float[] norm = new float[raw.length];
                for (int i = 0; i < raw.length; i++) norm[i] = raw[i] / 255.0f;
                return new Tensor[] { Torch.tensor(norm, 1, 28, 28), Torch.tensor(new float[] { trainLabels[index] }, 1) };
            }
        };
        Data.DataLoader trainLoader = new Data.DataLoader(trainDataset, 1024, true, 4);

        long start = System.currentTimeMillis();
        float lossVal = 0;
        for (int e = 0; e < epochs; e++) {
            float epochLoss = 0;
            int numBatches = 0;
            for (Tensor[] batch : trainLoader) {
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
                }
            }
            lossVal = epochLoss / numBatches;
        }
        long time = System.currentTimeMillis() - start;

        int testN = SmokeTest.isEnabled() ? 1024 : testImages.length;
        Data.Dataset testDataset = new Data.Dataset() {
            @Override public int len() { return testN; }
            @Override public Tensor[] get(int index) {
                float[] raw = testImages[index];
                float[] norm = new float[raw.length];
                for (int i = 0; i < raw.length; i++) norm[i] = raw[i] / 255.0f;
                return new Tensor[] { Torch.tensor(norm, 1, 28, 28), Torch.tensor(new float[] { testLabels[index] }, 1) };
            }
        };
        Data.DataLoader testLoader = new Data.DataLoader(testDataset, 1024, false, 2);
        float acc = Evaluator.evaluate(model, testLoader, new Accuracy());

        trainLoader.shutdown();
        testLoader.shutdown();

        BenchmarkResult res = new BenchmarkResult();
        res.task = "FashionMNIST"; res.architecture = "SOTA_CNN"; res.params = model.countParameters();
        res.trainLoss = lossVal; res.testAcc = acc; res.timeMs = time;
        return res;
    }

    private static BenchmarkResult runCifar10ResNetBenchmark(int epochs) throws Exception {
        System.out.println("--- Starting CIFAR-10 ResNet-18 ---");
        Cifar10Loader.prepareData();
        int trainBatches = SmokeTest.isEnabled() ? 1 : 5;
        float[][] trainImages = new float[trainBatches * 10000][3072];
        int[] trainLabels = new int[trainBatches * 10000];
        for (int i = 1; i <= trainBatches; i++) {
            Object[] batch = Cifar10Loader.loadBatch("data_batch_" + i + ".bin");
            System.arraycopy(batch[0], 0, trainImages, (i - 1) * 10000, 10000);
            System.arraycopy(batch[1], 0, trainLabels, (i - 1) * 10000, 10000);
        }

        Object[] testBatch = Cifar10Loader.loadBatch("test_batch.bin");
        float[][] testImages = (float[][]) testBatch[0];
        int[] testLabels = (int[]) testBatch[1];

        ResNet model = ResNet.resnet18(10, 32, 32);
        GpuMemoryPool.autoInit(model);
        model.toGPU();
        Optim.Adam optimizer = new Optim.Adam(model.parameters(), 0.001f);

        int N = trainImages.length;
        Data.Dataset trainDataset = new Data.Dataset() {
            @Override public int len() { return N; }
            @Override public Tensor[] get(int index) {
                return new Tensor[] { Torch.tensor(trainImages[index], 3, 32, 32), Torch.tensor(new float[] { trainLabels[index] }, 1) };
            }
        };
        Data.DataLoader trainLoader = new Data.DataLoader(trainDataset, 64, true, 4);

        long start = System.currentTimeMillis();
        float lossVal = 0;
        for (int e = 0; e < epochs; e++) {
            float epochLoss = 0;
            int numBatches = 0;
            for (Tensor[] batch : trainLoader) {
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
                }
            }
            lossVal = epochLoss / numBatches;
        }
        long time = System.currentTimeMillis() - start;

        int testN = SmokeTest.isEnabled() ? 512 : testImages.length;
        Data.Dataset testDataset = new Data.Dataset() {
            @Override public int len() { return testN; }
            @Override public Tensor[] get(int index) {
                return new Tensor[] { Torch.tensor(testImages[index], 3, 32, 32), Torch.tensor(new float[] { testLabels[index] }, 1) };
            }
        };
        Data.DataLoader testLoader = new Data.DataLoader(testDataset, 256, false, 2);
        float acc = Evaluator.evaluate(model, testLoader, new Accuracy());

        trainLoader.shutdown();
        testLoader.shutdown();

        BenchmarkResult res = new BenchmarkResult();
        res.task = "CIFAR-10"; res.architecture = "ResNet-18"; res.params = model.countParameters();
        res.trainLoss = lossVal; res.testAcc = acc; res.timeMs = time;
        return res;
    }

    private static BenchmarkResult runCifar10ViTBenchmark(int epochs) throws Exception {
        System.out.println("--- Starting CIFAR-10 Vision Transformer (ViT) ---");
        Cifar10Loader.prepareData();
        int trainBatches = SmokeTest.isEnabled() ? 1 : 5;
        float[][] trainImages = new float[trainBatches * 10000][3072];
        int[] trainLabels = new int[trainBatches * 10000];
        for (int i = 1; i <= trainBatches; i++) {
            Object[] batch = Cifar10Loader.loadBatch("data_batch_" + i + ".bin");
            System.arraycopy(batch[0], 0, trainImages, (i - 1) * 10000, 10000);
            System.arraycopy(batch[1], 0, trainLabels, (i - 1) * 10000, 10000);
        }

        Object[] testBatch = Cifar10Loader.loadBatch("test_batch.bin");
        float[][] testImages = (float[][]) testBatch[0];
        int[] testLabels = (int[]) testBatch[1];

        ViT model = new ViT(32, 4, 3, 10, 128, 4, 4, 256, 0.1f);
        GpuMemoryPool.autoInit(model);
        model.toGPU();
        Optim.Adam optimizer = new Optim.Adam(model.parameters(), 0.0005f);

        int N = trainImages.length;
        Data.Dataset trainDataset = new Data.Dataset() {
            @Override public int len() { return N; }
            @Override public Tensor[] get(int index) {
                return new Tensor[] { Torch.tensor(trainImages[index], 3, 32, 32), Torch.tensor(new float[] { trainLabels[index] }, 1) };
            }
        };
        Data.DataLoader trainLoader = new Data.DataLoader(trainDataset, 64, true, 4);

        long start = System.currentTimeMillis();
        float lossVal = 0;
        for (int e = 0; e < epochs; e++) {
            float epochLoss = 0;
            int numBatches = 0;
            for (Tensor[] batch : trainLoader) {
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
                }
            }
            lossVal = epochLoss / numBatches;
        }
        long time = System.currentTimeMillis() - start;

        int testN = SmokeTest.isEnabled() ? 512 : testImages.length;
        Data.Dataset testDataset = new Data.Dataset() {
            @Override public int len() { return testN; }
            @Override public Tensor[] get(int index) {
                return new Tensor[] { Torch.tensor(testImages[index], 3, 32, 32), Torch.tensor(new float[] { testLabels[index] }, 1) };
            }
        };
        Data.DataLoader testLoader = new Data.DataLoader(testDataset, 128, false, 2);
        float acc = Evaluator.evaluate(model, testLoader, new Accuracy());

        trainLoader.shutdown();
        testLoader.shutdown();

        BenchmarkResult res = new BenchmarkResult();
        res.task = "CIFAR-10"; res.architecture = "ViT-4L"; res.params = model.countParameters();
        res.trainLoss = lossVal; res.testAcc = acc; res.timeMs = time;
        return res;
    }
}
