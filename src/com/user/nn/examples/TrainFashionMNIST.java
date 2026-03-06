package com.user.nn.examples;

import com.user.nn.core.*;
import com.user.nn.optim.*;
import com.user.nn.dataloaders.*;
import com.user.nn.metrics.*;
import java.io.File;
import java.util.*;

/**
 * Train an MLP on Fashion-MNIST using the framework's autograd, optim.Adam,
 * and cross_entropy_tensor loss.
 *
 * Fashion-MNIST: 28x28 grayscale, 10 classes, 60k train / 10k test.
 * Model: 784 -> 256 (ReLU) -> 128 (ReLU) -> 10
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

        // --- Build model ---
        NN lib = new NN();
        NN.Sequential model = new NN.Sequential();
        // 28x28 = 784
        model.add(new NN.Linear(lib, 784, 128, true));
        model.add(new NN.ReLU());
        model.add(new NN.Dropout(0.2f));
        model.add(new NN.Linear(lib, 128, 64, true));
        model.add(new NN.ReLU());
        model.add(new NN.Linear(lib, 64, 10, true));

        // Initialize parameters
        long seed = 42L;
        for (NN.Parameter p : model.parameters()) {
            Tensor t = p.getTensor();
            float scale = (float) Math.sqrt(2.0 / t.numel());
            Random rng = new Random(seed++);
            for (int i = 0; i < t.data.length; i++) {
                t.data[i] = (float) (rng.nextGaussian() * scale);
            }
            t.requires_grad = true;
        }

        // --- Optimizer ---
        float lr = 0.01f;
        int epochs = 50;
        Optim.SGD optimizer = new Optim.SGD(model.parameters(), lr, 0.9f); // momentum 0.9

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
                Tensor x = Torch.tensor(trainImages[index], 784);
                Tensor y = Torch.tensor(new float[] { trainLabels[index] }, 1);
                return new Tensor[] { x, y };
            }
        };

        Data.DataLoader trainLoader = new Data.DataLoader(trainDataset, batchSize, true, 4);

        // Initialize GPU Memory Pool (Arena Allocator) - auto-detect free VRAM
        GpuMemoryPool.autoInit();

        System.out.println("Starting training for " + epochs + " epochs...");
        int totalBatches = N / batchSize;

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
                    Tensor loss = NN.F.cross_entropy_tensor(logits, batchLabels);

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
            float testAcc = evaluate(model, testImages, testLabels, testAccMetric);

            System.out.printf("Epoch %d/%d  avg_loss=%.4f  train_acc=%.4f  test_acc=%.4f%n",
                    epoch + 1, epochs, avgLoss, trainAcc, testAcc);
        }

        trainLoader.shutdown();
        float finalAcc = evaluate(model, testImages, testLabels, testAccMetric);
        System.out.printf("%nFinal test accuracy: %.2f%%%n", finalAcc * 100);
    }

    static float evaluate(NN.Module model, float[][] images, int[] labels, Accuracy metric) {
        metric.reset();
        int N = images.length;
        int dim = 784;
        int evalBatch = 256;

        for (int start = 0; start < N; start += evalBatch) {
            try (MemoryScope scope = new MemoryScope()) {
                int bs = Math.min(evalBatch, N - start);
                float[] data = new float[bs * dim];
                for (int i = 0; i < bs; i++)
                    System.arraycopy(images[start + i], 0, data, i * dim, dim);
                Tensor x = Torch.tensor(data, bs, dim);
                x.toGPU();

                int[] batchLabels = new int[bs];
                for (int i = 0; i < bs; i++) batchLabels[i] = labels[start + i];

                model.eval();
                Torch.set_grad_enabled(false);
                Tensor out = model.forward(x);
                Torch.set_grad_enabled(true);
                model.train();

                metric.update(out, batchLabels);
            }
        }
        return metric.compute();
    }
}
