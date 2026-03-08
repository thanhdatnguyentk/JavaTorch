package com.user.nn.examples;

import com.user.nn.core.*;
import com.user.nn.optim.*;
import com.user.nn.dataloaders.*;
import com.user.nn.metrics.*;
import com.user.nn.models.cv.ResNet;
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
        ResNet model = ResNet.resnet18(lib, 10, 32, 32);
        
        System.out.println("Total parameters: " + model.countParameters());

        // Kaiming initialization (simplified)
        for (NN.Parameter p : model.parameters()) {
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

        for (int epoch = 0; epoch < epochs; epoch++) {
            float epochLoss = 0f;
            int numBatches = 0;
            accMetric.reset();
            model.train();

            long startTime = System.currentTimeMillis();

            for (Tensor[] batch : loader) {
                try (MemoryScope scope = new MemoryScope()) {
                    Tensor xBatch = batch[0];
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

                    if (numBatches % 10 == 0) {
                        System.out.printf("  Epoch %d batch %d/%d  loss=%.4f%n",
                                epoch + 1, numBatches, N / batchSize, loss.data[0]);
                        System.out.flush();
                    }
                }
            }

            long endTime = System.currentTimeMillis();
            float trainAcc = accMetric.compute();
            float avgLoss = epochLoss / numBatches;
            
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
            
            System.out.printf("Epoch %d/%d  avg_loss=%.4f  train_acc=%.4f  test_acc=%.4f  time=%dms%n",
                    epoch + 1, epochs, avgLoss, trainAcc, testAcc, (endTime - startTime));
                    
            testLoader.shutdown();
        }
        loader.shutdown();
    }
}
