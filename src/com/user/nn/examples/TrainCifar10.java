package com.user.nn.examples;

import com.user.nn.core.*;
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
        NN.Sequential model = new NN.Sequential();
        model.add(new NN.Conv2d(lib, 3, 16, 3, 3, 32, 32, 1, 1, true));
        model.add(new NN.ReLU());
        model.add(new NN.MaxPool2d(2, 2, 2, 2, 0, 0, 16, 32, 32));
        model.add(new NN.Conv2d(lib, 16, 32, 3, 3, 16, 16, 1, 1, true));
        model.add(new NN.ReLU());
        model.add(new NN.MaxPool2d(2, 2, 2, 2, 0, 0, 32, 16, 16));

        int flattenSize = 32 * 8 * 8; 
        model.add(new NN.Linear(lib, flattenSize, 128, true));
        model.add(new NN.ReLU());
        model.add(new NN.Dropout(0.3f));
        model.add(new NN.Linear(lib, 128, 10, true));

        long seed = 123L;
        for (NN.Parameter p : model.parameters()) {
            Tensor t = p.getTensor();
            float scale = (float) Math.sqrt(2.0 / t.numel());
            Random rng = new Random(seed++);
            for (int i = 0; i < t.data.length; i++) {
                t.data[i] = (float) (rng.nextGaussian() * scale);
            }
            t.requires_grad = true;
        }

        float lr = 0.001f;
        int epochs = 100; 
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
                Tensor x = Torch.tensor(trainImages[index], 3072);
                Tensor y = Torch.tensor(new float[] { trainLabels[index] }, 1);
                return new Tensor[] { x, y };
            }
        };

        Data.DataLoader loader = new Data.DataLoader(trainDataset, batchSize, true, 4);
        Accuracy accMetric = new Accuracy();

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
                    Tensor loss = NN.F.cross_entropy_tensor(logits, batchLabels);
                    loss.backward();
                    optimizer.step();

                    epochLoss += loss.data[0];
                    numBatches++;
                    accMetric.update(logits, batchLabels);

                    if (numBatches % 20 == 0) {
                        System.out.printf("  Epoch %d batch %d/%d  loss=%.4f%n",
                                epoch + 1, numBatches, N / batchSize, loss.data[0]);
                    }
                }
            }

            float trainAcc = accMetric.compute();
            float avgLoss = epochLoss / numBatches;
            float testAcc = evaluate(model, testImages, testLabels, accMetric);
            System.out.printf("Epoch %d/%d  avg_loss=%.4f  train_acc=%.4f  test_acc=%.4f%n",
                    epoch + 1, epochs, avgLoss, trainAcc, testAcc);
        }
        loader.shutdown();
    }

    static float evaluate(NN.Module model, float[][] images, int[] labels, Accuracy metric) {
        metric.reset();
        int N = images.length;
        int dim = 3072;
        int evalBatch = 256;

        for (int start = 0; start < N; start += evalBatch) {
            try (MemoryScope scope = new MemoryScope()) {
                int bs = Math.min(evalBatch, N - start);
                float[] data = new float[bs * dim];
                for (int i = 0; i < bs; i++)
                    System.arraycopy(images[start + i], 0, data, i * dim, dim);

                Tensor x = Torch.tensor(data, bs, dim);
                x.toGPU();

                model.eval();
                Torch.set_grad_enabled(false);
                Tensor out = model.forward(x);
                Torch.set_grad_enabled(true);
                model.train();

                int[] batchLabels = new int[bs];
                for (int i = 0; i < bs; i++) batchLabels[i] = labels[start + i];
                metric.update(out, batchLabels);
            }
        }
        return metric.compute();
    }
}
