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
import com.user.nn.dataloaders.MnistLoader;
import com.user.nn.dataloaders.Data;
import com.user.nn.optim.*;

import java.io.File;
import java.util.*;

/**
 * Phase 4: Generative Models.
 * Training a Variational Autoencoder (VAE) on MNIST.
 */
public class TrainVAEMnist {

    public static class VAE extends Module {
        public Sequential encoder;
        public Linear fc_mu;
        public Linear fc_logvar;
        public Sequential decoder;

        public VAE(int latentDim) {
            encoder = new Sequential(
                new Linear(784, 400, true),
                new ReLU()
            );
            fc_mu = new Linear(400, latentDim, true);
            fc_logvar = new Linear(400, latentDim, true);
            
            decoder = new Sequential(
                new Linear(latentDim, 400, true),
                new ReLU(),
                new Linear(400, 784, true),
                new Sigmoid()
            );
            
            addModule("encoder", encoder);
            addModule("fc_mu", fc_mu);
            addModule("fc_logvar", fc_logvar);
            addModule("decoder", decoder);
        }

        public Tensor[] encode(Tensor x) {
            Tensor h = encoder.forward(x);
            return new Tensor[]{ fc_mu.forward(h), fc_logvar.forward(h) };
        }

        public Tensor reparameterize(Tensor mu, Tensor logvar) {
            // std = exp(0.5 * logvar)
            Tensor std = Torch.exp(Torch.mul(logvar, 0.5f));
            Tensor eps = Torch.randn(mu.shape).toGPU();
            return Torch.add(mu, Torch.mul(std, eps));
        }

        public Tensor decode(Tensor z) {
            return decoder.forward(z);
        }

        @Override
        public Tensor forward(Tensor x) {
            Tensor[] res = encode(x);
            Tensor mu = res[0];
            Tensor logvar = res[1];
            Tensor z = reparameterize(mu, logvar);
            return decode(z); // We'd need to return mu/logvar too for loss calculation
        }
    }

    public static void main(String[] args) throws Exception {
        System.out.println("=== Training VAE on MNIST ===");
        
        int latentDim = 20;
        int batchSize = 128;
        int epochs = SmokeTest.getEpochs(20);
        float lr = 1e-3f;
        
        // Data
        File dataDir = new File("data/mnist");
        File trainImagesGz = new File(dataDir, "train-images-idx3-ubyte.gz");
        MnistLoader.downloadIfMissing("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", trainImagesGz);
        float[][] allImages = MnistLoader.loadImages(trainImagesGz);
        
        Data.Dataset dataset = new Data.BaseDataset(allImages);
        Data.DataLoader loader = new Data.DataLoader(dataset, batchSize, true, 2);
        
        NN nn = new NN();
        VAE model = new VAE(latentDim);
        model.toGPU();
        
        Optim.Adam optimizer = new Optim.Adam(model.parameters(), lr);
        com.user.nn.optim.Scheduler.StepLR scheduler = new com.user.nn.optim.Scheduler.StepLR(optimizer, 10, 0.5f);

        TrainingHistory history = new TrainingHistory();
        DashboardServer dashboard = new DashboardServer(7070, history).start();
        dashboard.setTaskType("gan");
        dashboard.setModelInfo("VAE-MNIST", epochs);

        System.out.println("Starting Training Loop...");
        for (int epoch = 1; epoch <= epochs; epoch++) {
            double totalLoss = 0;
            int count = 0;
            model.train();
            
            for (Tensor[] batch : loader) {
                try (MemoryScope scope = new MemoryScope()) {
                    Tensor x = batch[0].to(Tensor.Device.GPU);
                    optimizer.zero_grad();
                    
                    // Manual forward to get mu/logvar
                    Tensor h = model.encoder.forward(x);
                    Tensor mu = model.fc_mu.forward(h);
                    Tensor logvar = model.fc_logvar.forward(h);
                    Tensor z = model.reparameterize(mu, logvar);
                    Tensor recon_x = model.decode(z);
                    
                    // --- Loss calculation ---
                    // 1. Reconstruction loss: BCE
                    Tensor bce = Functional.binary_cross_entropy(recon_x, x);
                    float reconLoss = bce.item() * 784; 
                    
                    // 2. KL Divergence
                    // Simplified KLD: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
                    Tensor loss = Torch.add(Torch.mul(bce, 784f), 
                                           Torch.mul(
                                               Torch.mul(
                                                   Torch.sum_tensor(
                                                       Torch.add(
                                                           Torch.add(Torch.add(Torch.ones(logvar.shape).toGPU(), logvar), 
                                                                     Torch.mul(Torch.mul(mu, mu), -1f)),
                                                           Torch.mul(Torch.exp(logvar), -1f)
                                                       )
                                                   ), -0.5f), 
                                               1.0f / x.shape[0])
                                           );
                    
                    loss.backward();
                    optimizer.step();
                    
                    totalLoss += loss.item();
                    count++;
                }
                while (dashboard.isTrainingPaused()) {
                    try { Thread.sleep(200); } catch (InterruptedException ie) { break; }
                }
            }
            float avgLoss = (float)(totalLoss / count);
            dashboard.setCurrentEpoch(epoch);
            System.out.printf("Epoch %d | Loss: %.4f | LR: %.6f%n", epoch, avgLoss, optimizer.getLearningRate());
            scheduler.step();

            try {
                Map<String, Float> metricsMap = new HashMap<>();
                metricsMap.put("loss", avgLoss);
                metricsMap.put("g_loss", avgLoss); // map to gan loss for ui
                metricsMap.put("d_loss", 0f);      // mock
                history.record(epoch, metricsMap);

                // Generate 16 samples for dashboard
                model.eval();
                java.util.List<float[]> sampleList = new java.util.ArrayList<>();
                try (MemoryScope sScope = new MemoryScope()) {
                    Tensor sNoise = Torch.randn(new int[]{16, latentDim}).toGPU();
                    Tensor sFake = model.decode(sNoise);
                    sFake.toCPU();
                    for (int s = 0; s < 16; s++) {
                        float[] pixels = new float[784];
                        System.arraycopy(sFake.data, s * 784, pixels, 0, 784);
                        sampleList.add(pixels);
                    }
                }
                DashboardIntegrationHelper.broadcastGANDetailed(
                    dashboard, epoch, avgLoss, 0f, sampleList, 1, 28, 28);
                model.train();
            } catch (Exception dashEx) {}
        }
        
        System.out.println("Saving model...");
        model.save("vae_mnist.bin");
        
        loader.shutdown();
        System.out.println("Training Complete!");
    }
}
