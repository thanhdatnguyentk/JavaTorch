package com.user.nn.examples;

import com.user.nn.utils.dashboard.DashboardServer;
import com.user.nn.utils.dashboard.DashboardIntegrationHelper;
import com.user.nn.utils.visualization.TrainingHistory;
import java.util.HashMap;
import java.util.Map;


import com.user.nn.activations.LeakyReLU;
import com.user.nn.activations.Sigmoid;
import com.user.nn.activations.Tanh;
import com.user.nn.containers.Sequential;
import com.user.nn.core.MemoryScope;
import com.user.nn.core.Module;
import com.user.nn.core.Parameter;
import com.user.nn.core.Tensor;
import com.user.nn.core.Torch;
import com.user.nn.dataloaders.AnimeFaceLoader;
import com.user.nn.dataloaders.Data;
import com.user.nn.layers.Dropout;
import com.user.nn.layers.Linear;
import com.user.nn.losses.BCELoss;
import com.user.nn.optim.Optim;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Locale;

import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;

public class TrainGANAnime {
    private static final int IMAGE_SIZE = 64;
    private static final int CHANNELS = 3;
    private static final int IMAGE_DIM = IMAGE_SIZE * IMAGE_SIZE * CHANNELS;
    private static final float GRAD_CLIP_NORM = 1.0f;

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
                new Linear(1024, IMAGE_DIM, true),
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
                new Linear(IMAGE_DIM, 512, true),
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
        System.out.println("=== Training GAN on Anime Face Dataset ===");

        String datasetDir = args.length > 0 ? args[0] : "data/anime_faces";
        int epochs = args.length > 1 ? Integer.parseInt(args[1]) : 30;
        int batchSize = args.length > 2 ? Integer.parseInt(args[2]) : 64;
        int maxImages = args.length > 3 ? Integer.parseInt(args[3]) : -1;

        float lr = 0.001f;
        int latentDim = 100;
        String outDir = "generated/anime_gan";

        File dsRoot = resolveDatasetDir(datasetDir);
        if (dsRoot == null) {
            throw new IllegalArgumentException(
                    "Dataset folder not found. Tried: "
                            + new File(datasetDir).getAbsolutePath()
                            + " and "
                            + new File("../" + datasetDir).getAbsolutePath());
        }

        float[][] allImages = AnimeFaceLoader.loadImages(dsRoot, IMAGE_SIZE, maxImages, true);
        if (allImages.length == 0) {
            throw new IllegalStateException("No .jpg/.jpeg/.png images found under: " + dsRoot.getAbsolutePath());
        }

        System.out.printf(Locale.US, "Loaded %d images. Shape per sample: [%d]%n", allImages.length, IMAGE_DIM);

        Data.Dataset dataset = new Data.BaseDataset(allImages);
        Data.DataLoader loader = new Data.DataLoader(dataset, batchSize, true, 2);

        Generator G = new Generator(latentDim);
        Discriminator D = new Discriminator();

        boolean useGpu = tryMoveToGpu(G, D);
        System.out.println("Device: " + (useGpu ? "GPU" : "CPU"));

        Optim.Adam gOpt = new Optim.Adam(G.parameters(), lr, 0.5f, 0.999f);
        Optim.Adam dOpt = new Optim.Adam(D.parameters(), lr, 0.5f, 0.999f);
        BCELoss criterion = new BCELoss();

        File outputFolder = new File(outDir);
        if (!outputFolder.exists()) {
            outputFolder.mkdirs();
        }

        for (int epoch = 1; epoch <= epochs; epoch++) {
            long start = System.currentTimeMillis();
            double dLossSum = 0.0;
            double gLossSum = 0.0;
            int batches = 0;
            int seenBatches = 0;
            int skippedDLoss = 0;
            int skippedGLoss = 0;
            boolean printedDNonFiniteDetails = false;
            boolean printedGNonFiniteDetails = false;

            G.train();
            D.train();

            for (Tensor[] batch : loader) {
                try (MemoryScope scope = new MemoryScope()) {
                    seenBatches++;
                    Tensor realImages = batch[0];
                    if (useGpu) {
                        realImages.toGPU();
                    }
                    int bSize = realImages.shape[0];

                    D.zero_grad();

                    // Label smoothing makes BCE training less saturated and more stable.
                    Tensor realLabels = Torch.full(new int[] {bSize, 1}, 0.9f);
                    if (useGpu) {
                        realLabels.toGPU();
                    }
                    Tensor dOutReal = D.forward(realImages);
                    Tensor dLossReal = criterion.forward(dOutReal, realLabels);

                    Tensor noise = Torch.randn(new int[] {bSize, latentDim});
                    if (useGpu) {
                        noise.toGPU();
                    }
                    Tensor fakeImages = G.forward(noise);

                    Tensor fakeLabels = Torch.full(new int[] {bSize, 1}, 0.1f);
                    if (useGpu) {
                        fakeLabels.toGPU();
                    }
                    Tensor dOutFake = D.forward(fakeImages.detach());
                    Tensor dLossFake = criterion.forward(dOutFake, fakeLabels);

                    float dLossRealVal = dLossReal.item();
                    float dLossFakeVal = dLossFake.item();
                    if (!isFinite(dLossRealVal) || !isFinite(dLossFakeVal)) {
                        skippedDLoss++;
                        if (!printedDNonFiniteDetails) {
                            printedDNonFiniteDetails = true;
                            System.out.printf(Locale.US,
                                "Non-finite D loss at epoch %d batch %d: dReal=%s dFake=%s | dOutReal=%s | dOutFake=%s | fake=%s%n",
                                epoch,
                                seenBatches,
                                Float.toString(dLossRealVal),
                                Float.toString(dLossFakeVal),
                                tensorStats(dOutReal),
                                tensorStats(dOutFake),
                                tensorStats(fakeImages));
                        }
                        continue;
                    }
                    // Backprop each discriminator term separately to avoid unstable
                    // scalar-tensor add kernels on GPU.
                    dLossReal.backward();
                    dLossFake.backward();
                    clipGradNorm(D.parameters(), GRAD_CLIP_NORM);
                    dOpt.step();

                    G.zero_grad();
                    Tensor gOut = D.forward(fakeImages);
                    Tensor gLoss = criterion.forward(gOut, realLabels);
                    float gLossVal = gLoss.item();
                    if (!isFinite(gLossVal)) {
                        skippedGLoss++;
                        if (!printedGNonFiniteDetails) {
                            printedGNonFiniteDetails = true;
                            System.out.printf(Locale.US,
                                "Non-finite G loss at epoch %d batch %d: gLoss=%s | gOut=%s | fake=%s%n",
                                epoch,
                                seenBatches,
                                Float.toString(gLossVal),
                                tensorStats(gOut),
                                tensorStats(fakeImages));
                        }
                        continue;
                    }
                    gLoss.backward();
                    clipGradNorm(G.parameters(), GRAD_CLIP_NORM);
                    gOpt.step();

                    dLossSum += (dLossRealVal + dLossFakeVal);
                    gLossSum += gLossVal;
                    batches++;
                }
            }

            long ms = System.currentTimeMillis() - start;
            System.out.printf(Locale.US,
                "Epoch [%d/%d] | D Loss: %.4f | G Loss: %.4f | Time: %d ms | Batches: %d/%d | Skipped D/G: %d/%d%n",
                epoch,
                epochs,
                dLossSum / Math.max(1, batches),
                gLossSum / Math.max(1, batches),
                ms,
                batches,
                seenBatches,
                skippedDLoss,
                skippedGLoss);

            if (batches == 0) {
                System.out.println("Warning: no valid batches in this epoch. Loss may print as 0.0000 due to all batches skipped.");
            }

            if (epoch % 5 == 0 || epoch == epochs) {
                saveSamples(G, latentDim, useGpu, epoch, outputFolder);
            }
        }

        String gPath = new File(outputFolder, "gan_anime_generator.bin").getPath();
        String dPath = new File(outputFolder, "gan_anime_discriminator.bin").getPath();
        G.save(gPath);
        D.save(dPath);
        loader.shutdown();

        System.out.println("Saved generator: " + gPath);
        System.out.println("Saved discriminator: " + dPath);
        System.out.println("Training complete.");
    }

    private static File resolveDatasetDir(String datasetDir) {
        File direct = new File(datasetDir);
        if (direct.exists() && direct.isDirectory()) {
            return direct;
        }

        File fromExamplesTask = new File("../" + datasetDir);
        if (fromExamplesTask.exists() && fromExamplesTask.isDirectory()) {
            return fromExamplesTask;
        }

        return null;
    }

    private static boolean tryMoveToGpu(Generator g, Discriminator d) {
        try {
            g.toGPU();
            d.toGPU();
            return true;
        } catch (Throwable t) {
            g.toCPU();
            d.toCPU();
            return false;
        }
    }

    private static void saveSamples(Generator g, int latentDim, boolean useGpu, int epoch, File outputFolder)
            throws IOException {
        g.eval();
        try (MemoryScope scope = new MemoryScope()) {
            int sampleCount = 16;
            Tensor z = Torch.randn(new int[] {sampleCount, latentDim});
            if (useGpu) {
                z.toGPU();
            }
            Tensor fake = g.forward(z);
            fake.toCPU();

            for (int i = 0; i < sampleCount; i++) {
                BufferedImage img = sampleToImage(fake, i);
                File out = new File(outputFolder, String.format(Locale.US, "epoch_%03d_%02d.png", epoch, i));
                ImageIO.write(img, "png", out);
            }
        }
        g.train();
    }

    private static BufferedImage sampleToImage(Tensor batchOutput, int sampleIndex) {
        BufferedImage img = new BufferedImage(IMAGE_SIZE, IMAGE_SIZE, BufferedImage.TYPE_INT_RGB);
        int sampleOffset = sampleIndex * IMAGE_DIM;
        int hw = IMAGE_SIZE * IMAGE_SIZE;

        for (int y = 0; y < IMAGE_SIZE; y++) {
            for (int x = 0; x < IMAGE_SIZE; x++) {
                int pos = y * IMAGE_SIZE + x;
                float rf = batchOutput.data[sampleOffset + pos];
                float gf = batchOutput.data[sampleOffset + hw + pos];
                float bf = batchOutput.data[sampleOffset + 2 * hw + pos];

                int r = toU8(rf);
                int g = toU8(gf);
                int b = toU8(bf);
                img.setRGB(x, y, (r << 16) | (g << 8) | b);
            }
        }
        return img;
    }

    private static int toU8(float v) {
        int x = Math.round(((v + 1.0f) * 0.5f) * 255.0f);
        if (x < 0) {
            return 0;
        }
        if (x > 255) {
            return 255;
        }
        return x;
    }

    private static boolean isFiniteScalar(Tensor t) {
        float v = t.item();
        return !Float.isNaN(v) && !Float.isInfinite(v);
    }

    private static boolean isFinite(float v) {
        return !Float.isNaN(v) && !Float.isInfinite(v);
    }

    private static String tensorStats(Tensor t) {
        t.toCPU();
        int n = t.data.length;
        int nan = 0;
        int inf = 0;
        double sum = 0.0;
        float min = Float.POSITIVE_INFINITY;
        float max = Float.NEGATIVE_INFINITY;

        for (float v : t.data) {
            if (Float.isNaN(v)) {
                nan++;
                continue;
            }
            if (Float.isInfinite(v)) {
                inf++;
                continue;
            }
            sum += v;
            if (v < min) {
                min = v;
            }
            if (v > max) {
                max = v;
            }
        }

        int finite = n - nan - inf;
        double mean = finite > 0 ? (sum / finite) : Double.NaN;
        return String.format(Locale.US, "shape=%s finite=%d/%d nan=%d inf=%d min=%.5f max=%.5f mean=%.5f",
            java.util.Arrays.toString(t.shape), finite, n, nan, inf, min, max, mean);
    }

    private static void clipGradNorm(List<Parameter> params, float maxNorm) {
        float totalNorm = 0f;
        for (Parameter p : params) {
            Tensor g = p.getGrad();
            if (g != null) {
                g.toCPU();
                for (float v : g.data) {
                    totalNorm += v * v;
                }
            }
        }
        totalNorm = (float) Math.sqrt(totalNorm);
        if (totalNorm > maxNorm) {
            float scale = maxNorm / (totalNorm + 1e-6f);
            for (Parameter p : params) {
                Tensor g = p.getGrad();
                if (g != null) {
                    for (int i = 0; i < g.data.length; i++) {
                        g.data[i] *= scale;
                    }
                    g.markDirtyOnCPU();
                }
            }
        }
    }
}
