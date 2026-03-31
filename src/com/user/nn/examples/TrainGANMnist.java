package com.user.nn.examples;

import com.user.nn.utils.dashboard.DashboardServer;
import com.user.nn.utils.dashboard.DashboardIntegrationHelper;
import com.user.nn.utils.visualization.TrainingHistory;
import java.util.HashMap;
import java.util.Map;


import com.user.nn.core.*;
import com.user.nn.core.Module;
import com.user.nn.containers.*;
import com.user.nn.layers.*;
import com.user.nn.activations.*;
import com.user.nn.losses.*;
import com.user.nn.dataloaders.MnistLoader;
import com.user.nn.dataloaders.Data;
import com.user.nn.optim.*;

import java.io.File;
import java.util.*;

/**
 * Phase 4: Generative Models.
 * Training a Simple Generative Adversarial Network (GAN) on MNIST.
 */
public class TrainGANMnist {

    public static class Generator extends Module {
        public Sequential model;

        public Generator(int latentDim) {
            model = new Sequential(
                new Linear(latentDim, 256, true),
                new LeakyReLU(0.2f),
                new Linear(256, 512, true),
                new LeakyReLU(0.2f),
                new Linear(512, 1024, true),
                new LeakyReLU(0.2f),
                new Linear(1024, 784, true),
                new Tanh()
            );
            addModule("model", model);
        }

        @Override
        public Tensor forward(Tensor z) {
            return model.forward(z);
        }
    }

    public static class Discriminator extends Module {
        public Sequential model;

        public Discriminator() {
            model = new Sequential(
                new Linear(784, 512, true),
                new LeakyReLU(0.2f),
                new Dropout(0.3f),
                new Linear(512, 256, true),
                new LeakyReLU(0.2f),
                new Dropout(0.3f),
                new Linear(256, 1, true),
                new Sigmoid()
            );
            addModule("model", model);
        }

        @Override
        public Tensor forward(Tensor x) {
            return model.forward(x);
        }
    }

    public static void main(String[] args) throws Exception {
        System.out.println("=== Training GAN on MNIST ===");
        
        // Settings
        int latentDim = 100;
        int batchSize = 64;
        int epochs = 50;
        float lr = 0.0002f;
        
        // Data
        File dataDir = new File("data/mnist");
        File trainImagesGz = new File(dataDir, "train-images-idx3-ubyte.gz");
        MnistLoader.downloadIfMissing("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", trainImagesGz);
        
        float[][] allImages = MnistLoader.loadImages(trainImagesGz);
        // Normalize to [-1, 1] for Tanh output
        for (int i = 0; i < allImages.length; i++) {
            for (int j = 0; j < allImages[i].length; j++) {
                allImages[i][j] = allImages[i][j] * 2.0f - 1.0f;
            }
        }
        
        Data.Dataset dataset = new Data.BaseDataset(allImages);
        Data.DataLoader loader = new Data.DataLoader(dataset, batchSize, true, 2);
        
        NN nn = new NN();
        Generator G = new Generator(latentDim);
        Discriminator D = new Discriminator();
        
        // Move to GPU
        G.toGPU();
        D.toGPU();
        
        Optim.Adam gOpt = new Optim.Adam(G.parameters(), lr, 0.5f, 0.999f);
        Optim.Adam dOpt = new Optim.Adam(D.parameters(), lr, 0.5f, 0.999f);
        
        // Add Schedulers
        com.user.nn.optim.Scheduler.StepLR gSched = new com.user.nn.optim.Scheduler.StepLR(gOpt, 10, 0.5f);
        com.user.nn.optim.Scheduler.StepLR dSched = new com.user.nn.optim.Scheduler.StepLR(dOpt, 10, 0.5f);

        BCELoss criterion = new BCELoss();
        
        Random rand = new Random();
        
        TrainingHistory history = new TrainingHistory();
        DashboardServer dashboard = new DashboardServer(7070, history).start();
        dashboard.setTaskType("gan");
        dashboard.setModelInfo("GAN-MNIST", epochs);

        System.out.println("Starting Training Loop...");
        for (int epoch = 1; epoch <= epochs; epoch++) {
            long startTime = System.currentTimeMillis();
            double epochDLoss = 0, epochGLoss = 0;
            int batches = 0;
            
            G.train();
            D.train();

            for (Tensor[] batch : loader) {
                try (MemoryScope scope = new MemoryScope()) {
                    Tensor realImages = batch[0].to(Tensor.Device.GPU);
                    int bSize = realImages.shape[0];
                    
                    // --- 1. Train Discriminator ---
                    D.zero_grad();
                    
                    // Real loss
                    Tensor realLabels = Torch.full(new int[]{bSize, 1}, 1.0f).toGPU();
                    Tensor outputReal = D.forward(realImages);
                    Tensor dLossReal = criterion.forward(outputReal, realLabels);
                    
                    // Fake loss
                    Tensor noise = Torch.randn(new int[]{bSize, latentDim}).toGPU();
                    Tensor fakeImages = G.forward(noise);
                    Tensor fakeLabels = Torch.full(new int[]{bSize, 1}, 0.0f).toGPU();
                    Tensor outputFake = D.forward(fakeImages.detach()); // Detach G to not update G here
                    Tensor dLossFake = criterion.forward(outputFake, fakeLabels);
                    
                    Tensor dLoss = Torch.add(dLossReal, dLossFake);
                    dLoss.backward();
                    dOpt.step();
                    
                    // --- 2. Train Generator ---
                    G.zero_grad();
                    
                    // We want G(z) to be classified as real
                    Tensor outputFakeForG = D.forward(fakeImages); 
                    Tensor gLoss = criterion.forward(outputFakeForG, realLabels);
                    
                    gLoss.backward();
                    gOpt.step();
                    
                    epochDLoss += dLoss.item();
                    epochGLoss += gLoss.item();
                    batches++;
                }
            }
            
            long endTime = System.currentTimeMillis();
            double avgDLoss = epochDLoss / batches;
            double avgGLoss = epochGLoss / batches;
            System.out.printf("Epoch [%d/%d] | D Loss: %.4f | G Loss: %.4f | LR: %.6f | Time: %d ms%n",
                epoch, epochs, avgDLoss, avgGLoss, gOpt.getLearningRate(), (endTime - startTime));
            
            // Step Schedulers
            gSched.step();
            dSched.step();
            dashboard.setCurrentEpoch(epoch);

            // Broadcast GAN metrics + samples to dashboard
            try {
                Map<String, Float> metrics = new HashMap<>();
                metrics.put("g_loss", (float) avgGLoss);
                metrics.put("d_loss", (float) avgDLoss);
                history.record(epoch, metrics);

                // Generate 16 samples for dashboard
                G.eval();
                java.util.List<float[]> sampleList = new java.util.ArrayList<>();
                try (MemoryScope sScope = new MemoryScope()) {
                    Tensor sNoise = Torch.randn(new int[]{16, latentDim}).toGPU();
                    Tensor sFake = G.forward(sNoise);
                    sFake.toCPU();
                    for (int s = 0; s < 16; s++) {
                        float[] pixels = new float[784];
                        System.arraycopy(sFake.data, s * 784, pixels, 0, 784);
                        sampleList.add(pixels);
                    }
                }
                DashboardIntegrationHelper.broadcastGANDetailed(
                    dashboard, epoch, (float) avgGLoss, (float) avgDLoss, sampleList, 1, 28, 28);
                G.train();
            } catch (Exception dashEx) {}

            // Periodically log samples
            if (epoch % 5 == 0) {
                generateSamples(G, latentDim, epoch);
            }
        }
        
        // Save models
        System.out.println("Saving models...");
        G.save("gan_generator.bin");
        D.save("gan_discriminator.bin");
        
        loader.shutdown();
        System.out.println("Training Complete!");
    }
    
    private static void generateSamples(Generator G, int latentDim, int epoch) {
        G.eval();
        try (MemoryScope scope = new MemoryScope()) {
            Tensor noise = Torch.randn(new int[]{16, latentDim}).toGPU();
            Tensor fake = G.forward(noise);
            fake.toCPU();
            // In a real framework we'd save these as images. 
            // For now, just print the range of values to verify Tanh.
            float min = Float.MAX_VALUE, max = Float.MIN_VALUE;
            for (float v : fake.data) {
                if (v < min) min = v;
                if (v > max) max = v;
            }
            System.out.printf("   [Sample Stats] Range: [%.3f, %.3f]%n", min, max);
        }
        G.train();
    }
}
