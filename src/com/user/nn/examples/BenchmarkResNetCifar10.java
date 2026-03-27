package com.user.nn.examples;

import com.user.nn.benchmark.BenchmarkArgs;
import com.user.nn.benchmark.BenchmarkCsv;
import com.user.nn.benchmark.BenchmarkStats;
import com.user.nn.core.Functional;
import com.user.nn.core.GpuMemoryPool;
import com.user.nn.core.MemoryScope;
import com.user.nn.core.MixedPrecision;
import com.user.nn.core.Parameter;
import com.user.nn.core.Tensor;
import com.user.nn.core.Torch;
import com.user.nn.dataloaders.Cifar10Loader;
import com.user.nn.dataloaders.Data;
import com.user.nn.metrics.Accuracy;
import com.user.nn.models.cv.ResNet;
import com.user.nn.optim.Optim;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Random;

public class BenchmarkResNetCifar10 {

    public static void main(String[] args) throws Exception {
        Map<String, String> cli = BenchmarkArgs.parse(args);

        String device = BenchmarkArgs.getString(cli, "device", "cpu").toLowerCase(Locale.ROOT);
        int epochs = BenchmarkArgs.getInt(cli, "epochs", 5);
        int batchSize = BenchmarkArgs.getInt(cli, "batchSize", 64);
        int numWorkers = BenchmarkArgs.getInt(cli, "numWorkers", 2);
        int inferWarmup = BenchmarkArgs.getInt(cli, "inferWarmup", 10);
        int inferSteps = BenchmarkArgs.getInt(cli, "inferSteps", 50);
        long seed = BenchmarkArgs.getLong(cli, "seed", 42L);
        boolean mixedPrecision = BenchmarkArgs.getBoolean(cli, "mixedPrecision", false);
        String outputDir = BenchmarkArgs.getString(cli, "outputDir", "benchmark/results");
        String runId = BenchmarkArgs.getString(cli, "runId", "resnet_cifar10_" + timestamp() + "_" + device);

        if (!"cpu".equals(device) && !"gpu".equals(device)) {
            throw new IllegalArgumentException("--device must be cpu or gpu");
        }

        Torch.manual_seed(seed);
        if (mixedPrecision) {
            MixedPrecision.enable();
        } else {
            MixedPrecision.disable();
        }

        System.out.println("Preparing CIFAR-10 data...");
        Cifar10Loader.prepareData();

        float[][] trainImages = new float[50000][3072];
        int[] trainLabels = new int[50000];
        for (int i = 1; i <= 5; i++) {
            Object[] batch = Cifar10Loader.loadBatch("data_batch_" + i + ".bin");
            float[][] imgs = (float[][]) batch[0];
            int[] lbls = (int[]) batch[1];
            System.arraycopy(imgs, 0, trainImages, (i - 1) * 10000, 10000);
            System.arraycopy(lbls, 0, trainLabels, (i - 1) * 10000, 10000);
        }

        Object[] testBatch = Cifar10Loader.loadBatch("test_batch.bin");
        float[][] testImages = (float[][]) testBatch[0];
        int[] testLabels = (int[]) testBatch[1];

        ResNet model = ResNet.resnet18(10, 32, 32);
        deterministicInit(model, seed);

        if ("gpu".equals(device)) {
            GpuMemoryPool.autoInit(model);
            System.out.println("[Benchmark][ResNet] moving model to GPU...");
            model.toGPU();
            System.out.println("[Benchmark][ResNet] model ready on GPU.");
        } else {
            model.toCPU();
        }

        Optim.Adam optimizer = new Optim.Adam(model.parameters(), 0.001f);
        Accuracy metric = new Accuracy();

        final int trainSize = trainImages.length;
        Data.Dataset trainDataset = new Data.Dataset() {
            @Override
            public int len() {
                return trainSize;
            }

            @Override
            public Tensor[] get(int index) {
                Tensor x = Torch.tensor(trainImages[index], 3, 32, 32);
                Tensor y = Torch.tensor(new float[]{trainLabels[index]}, 1);
                return new Tensor[]{x, y};
            }
        };

        Data.Dataset testDataset = new Data.Dataset() {
            @Override
            public int len() {
                return testImages.length;
            }

            @Override
            public Tensor[] get(int index) {
                Tensor x = Torch.tensor(testImages[index], 3, 32, 32);
                Tensor y = Torch.tensor(new float[]{testLabels[index]}, 1);
                return new Tensor[]{x, y};
            }
        };

        Data.DataLoader trainLoader = new Data.DataLoader(trainDataset, batchSize, true, numWorkers);

        Path runDir = Paths.get(outputDir, "JavaTorch", "resnet_cifar10", runId);
        Path epochCsv = runDir.resolve("epoch_metrics.csv");
        Path inferCsv = runDir.resolve("inference_samples.csv");
        Path summaryCsv = runDir.resolve("run_summary.csv");

        long wallStart = System.nanoTime();
        double bestAcc = Double.NEGATIVE_INFINITY;
        int bestEpoch = -1;
        long peakHeapBytes = 0L;
        long peakPoolUsedBytes = 0L;

        for (int epoch = 0; epoch < epochs; epoch++) {
            long epochStart = System.nanoTime();
            model.train();
            metric.reset();

            float epochLoss = 0f;
            int batchCount = 0;

            for (Tensor[] batch : trainLoader) {
                try (MemoryScope scope = new MemoryScope()) {
                    Tensor xBatch = batch[0];
                    int bs = xBatch.shape[0];
                    int[] labels = new int[bs];
                    for (int i = 0; i < bs; i++) {
                        labels[i] = (int) batch[1].data[i];
                    }

                    if ("gpu".equals(device)) {
                        xBatch.toGPU();
                    }

                    optimizer.zero_grad();
                    Tensor logits = model.forward(xBatch);
                    Tensor loss = Functional.cross_entropy_tensor(logits, labels);
                    loss.backward();
                    optimizer.step();

                    epochLoss += loss.data[0];
                    batchCount++;
                    metric.update(logits, labels);

                    int step = batchCount;
                    if (step == 1 || step % 100 == 0) {
                        int estimatedTotal = Math.max(1, trainSize / batchSize);
                        System.out.printf(Locale.US,
                                "[Benchmark][ResNet][Train] epoch=%d/%d batch=%d/%d loss=%.5f%n",
                                epoch + 1, epochs, step, estimatedTotal, loss.data[0]);
                    }

                    peakHeapBytes = Math.max(peakHeapBytes, usedHeapBytes());
                    peakPoolUsedBytes = Math.max(peakPoolUsedBytes, GpuMemoryPool.getUsedBytes());
                }
            }

            double trainAcc = metric.compute();
            double avgLoss = epochLoss / Math.max(1, batchCount);
            double testAcc = evaluateAccuracy(model, testDataset, 128, device);
            long epochMs = (System.nanoTime() - epochStart) / 1_000_000L;

            if (testAcc > bestAcc) {
                bestAcc = testAcc;
                bestEpoch = epoch;
            }

            LinkedHashMap<String, String> row = baseRow(runId, device, seed, batchSize, epochs, mixedPrecision);
            row.put("framework", "JavaTorch");
            row.put("task", "resnet_cifar10");
            row.put("epoch", String.valueOf(epoch + 1));
            row.put("train_loss", fmt(avgLoss));
            row.put("train_acc", fmt(trainAcc));
            row.put("val_acc", fmt(testAcc));
            row.put("epoch_time_ms", String.valueOf(epochMs));
            row.put("peak_heap_mb", fmt(bytesToMb(peakHeapBytes)));
            row.put("peak_vram_mb", fmt(bytesToMb(peakPoolUsedBytes)));
            BenchmarkCsv.appendRow(epochCsv, row);

            System.out.printf(Locale.US,
                    "[Benchmark][ResNet] epoch=%d/%d loss=%.5f train_acc=%.4f val_acc=%.4f time_ms=%d%n",
                    epoch + 1, epochs, avgLoss, trainAcc, testAcc, epochMs);
        }

        InferenceResult infer = benchmarkInference(model, testDataset, 128, device, inferWarmup, inferSteps, inferCsv,
                runId, seed, batchSize, epochs, mixedPrecision);

        long totalMs = (System.nanoTime() - wallStart) / 1_000_000L;
        LinkedHashMap<String, String> summary = baseRow(runId, device, seed, batchSize, epochs, mixedPrecision);
        summary.put("framework", "JavaTorch");
        summary.put("task", "resnet_cifar10");
        summary.put("best_val_acc", fmt(bestAcc));
        summary.put("best_epoch", String.valueOf(bestEpoch + 1));
        summary.put("total_train_time_ms", String.valueOf(totalMs));
        summary.put("inference_p50_ms", fmt(infer.p50Ms));
        summary.put("inference_p95_ms", fmt(infer.p95Ms));
        summary.put("inference_throughput_sps", fmt(infer.throughputSps));
        summary.put("peak_heap_mb", fmt(bytesToMb(peakHeapBytes)));
        summary.put("peak_vram_mb", fmt(bytesToMb(peakPoolUsedBytes)));
        BenchmarkCsv.appendRow(summaryCsv, summary);

        trainLoader.shutdown();
        if ("gpu".equals(device)) {
            GpuMemoryPool.destroy();
        }

        System.out.println("[Benchmark][ResNet] Finished. Artifacts in: " + runDir.toAbsolutePath());
    }

    private static void deterministicInit(ResNet model, long seed) {
        Random rng = new Random(seed);
        for (Parameter p : model.parameters()) {
            Tensor t = p.getTensor();
            if (t.dim() >= 2) {
                float scale = (float) Math.sqrt(2.0 / Math.max(1, t.numel()));
                for (int i = 0; i < t.data.length; i++) {
                    t.data[i] = (float) (rng.nextGaussian() * scale);
                }
            }
            t.requires_grad = true;
        }
    }

    private static double evaluateAccuracy(ResNet model, Data.Dataset dataset, int batchSize, String device) {
        Data.DataLoader loader = new Data.DataLoader(dataset, batchSize, false, 1);
        Accuracy metric = new Accuracy();

        boolean prevGrad = Torch.is_grad_enabled();
        Torch.set_grad_enabled(false);
        model.eval();

        try {
            for (Tensor[] batch : loader) {
                try (MemoryScope scope = new MemoryScope()) {
                    Tensor xBatch = batch[0];
                    int bs = xBatch.shape[0];
                    int[] labels = new int[bs];
                    for (int i = 0; i < bs; i++) {
                        labels[i] = (int) batch[1].data[i];
                    }
                    if ("gpu".equals(device)) {
                        xBatch.toGPU();
                    }
                    Tensor logits = model.forward(xBatch);
                    metric.update(logits, labels);
                }
            }
            return metric.compute();
        } finally {
            loader.shutdown();
            Torch.set_grad_enabled(prevGrad);
            model.train();
        }
    }

    private static InferenceResult benchmarkInference(
            ResNet model,
            Data.Dataset dataset,
            int batchSize,
            String device,
            int warmupSteps,
            int measureSteps,
            Path inferCsv,
            String runId,
            long seed,
            int trainBatchSize,
            int epochs,
            boolean mixedPrecision) throws IOException {

        Data.DataLoader loader = new Data.DataLoader(dataset, batchSize, false, 1);
        boolean prevGrad = Torch.is_grad_enabled();
        Torch.set_grad_enabled(false);
        model.eval();

        double[] latMs = new double[Math.max(1, measureSteps)];
        int measured = 0;
        int seen = 0;
        long totalSamples = 0;
        double totalLatencyMs = 0.0;

        try {
            for (Tensor[] batch : loader) {
                if (seen >= warmupSteps + measureSteps) {
                    break;
                }

                try (MemoryScope scope = new MemoryScope()) {
                    Tensor xBatch = batch[0];
                    if ("gpu".equals(device)) {
                        xBatch.toGPU();
                    }

                    long t0 = System.nanoTime();
                    model.forward(xBatch);
                    long t1 = System.nanoTime();

                    if (seen >= warmupSteps) {
                        double ms = (t1 - t0) / 1_000_000.0;
                        latMs[measured] = ms;
                        measured++;
                        totalLatencyMs += ms;
                        totalSamples += xBatch.shape[0];

                        LinkedHashMap<String, String> row = baseRow(runId, device, seed, trainBatchSize, epochs,
                                mixedPrecision);
                        row.put("framework", "JavaTorch");
                        row.put("task", "resnet_cifar10");
                        row.put("step", String.valueOf(measured));
                        row.put("batch_size", String.valueOf(xBatch.shape[0]));
                        row.put("latency_ms", fmt(ms));
                        BenchmarkCsv.appendRow(inferCsv, row);
                    }
                    seen++;
                }
            }
        } finally {
            loader.shutdown();
            Torch.set_grad_enabled(prevGrad);
            model.train();
        }

        if (measured == 0) {
            return new InferenceResult(Double.NaN, Double.NaN, Double.NaN);
        }

        double[] effective = new double[measured];
        System.arraycopy(latMs, 0, effective, 0, measured);

        double p50 = BenchmarkStats.percentile(effective, 50.0);
        double p95 = BenchmarkStats.percentile(effective, 95.0);
        double throughput = totalLatencyMs > 0.0 ? (totalSamples * 1000.0 / totalLatencyMs) : Double.NaN;

        return new InferenceResult(p50, p95, throughput);
    }

    private static LinkedHashMap<String, String> baseRow(
            String runId,
            String device,
            long seed,
            int batchSize,
            int epochs,
            boolean mixedPrecision) {
        LinkedHashMap<String, String> row = new LinkedHashMap<>();
        row.put("run_id", runId);
        row.put("timestamp", timestamp());
        row.put("device", device);
        row.put("seed", String.valueOf(seed));
        row.put("train_batch_size", String.valueOf(batchSize));
        row.put("epochs", String.valueOf(epochs));
        row.put("mixed_precision", String.valueOf(mixedPrecision));
        return row;
    }

    private static long usedHeapBytes() {
        Runtime rt = Runtime.getRuntime();
        return rt.totalMemory() - rt.freeMemory();
    }

    private static double bytesToMb(long bytes) {
        return bytes / (1024.0 * 1024.0);
    }

    private static String fmt(double v) {
        if (Double.isNaN(v) || Double.isInfinite(v)) {
            return "nan";
        }
        return String.format(Locale.US, "%.6f", v);
    }

    private static String timestamp() {
        return new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
    }

    private static final class InferenceResult {
        final double p50Ms;
        final double p95Ms;
        final double throughputSps;

        InferenceResult(double p50Ms, double p95Ms, double throughputSps) {
            this.p50Ms = p50Ms;
            this.p95Ms = p95Ms;
            this.throughputSps = throughputSps;
        }
    }
}
