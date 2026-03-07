package com.user.nn.examples;

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
        NN lib = new NN();
        LeNet model = new LeNet(lib);

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
                Tensor x = Torch.tensor(trainImages[index], 784);
                Tensor y = Torch.tensor(new float[] { trainLabels[index] }, 1);
                return new Tensor[] { x, y };
            }
        };

        Data.DataLoader trainLoader = new Data.DataLoader(trainDataset, batchSize, true, 4);

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

                    // Reshape flat [batchSize, 784] to [batchSize, 1, 28, 28] for Conv2d
                    int bs = xBatch.shape[0];
                    xBatch.shape = new int[]{bs, 1, 28, 28};
                    
                    xBatch.toGPU();

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
    }

    private static float evaluate(LeNet model, float[][] images, int[] labels, Accuracy accMetric) {
        model.eval();
        accMetric.reset();
        int batchSize = 256;
        int n = images.length;
        int numBatches = (int) Math.ceil((double) n / batchSize);

        for (int b = 0; b < numBatches; b++) {
            try (MemoryScope scope = new MemoryScope()) {
                int start = b * batchSize;
                int end = Math.min(start + batchSize, n);
                int bs = end - start;
                
                float[] flatBatch = new float[bs * 784];
                int[] batchLabels = new int[bs];
                for (int i = 0; i < bs; i++) {
                    System.arraycopy(images[start + i], 0, flatBatch, i * 784, 784);
                    batchLabels[i] = labels[start + i];
                }
                
                Tensor xBatch = Torch.tensor(flatBatch, bs, 1, 28, 28); // Reshaped for Conv2d
                scope.track(xBatch);
                xBatch.toGPU();

                Tensor logits = model.forward(xBatch);
                accMetric.update(logits, batchLabels);
            }
        }
        return accMetric.compute();
    }
}
