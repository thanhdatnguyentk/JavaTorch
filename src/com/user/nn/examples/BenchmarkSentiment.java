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
import com.user.nn.dataloaders.Data;
import com.user.nn.dataloaders.MovieCommentLoader;
import com.user.nn.metrics.Accuracy;
import com.user.nn.models.SentimentModel;
import com.user.nn.optim.Optim;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.SimpleDateFormat;
import java.util.Collections;
import java.util.Date;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;

public class BenchmarkSentiment {

    public static void main(String[] args) throws Exception {
        Map<String, String> cli = BenchmarkArgs.parse(args);

        String device = BenchmarkArgs.getString(cli, "device", "cpu").toLowerCase(Locale.ROOT);
        int epochs = BenchmarkArgs.getInt(cli, "epochs", 8);
        int batchSize = BenchmarkArgs.getInt(cli, "batchSize", 16);
        int inferWarmup = BenchmarkArgs.getInt(cli, "inferWarmup", 10);
        int inferSteps = BenchmarkArgs.getInt(cli, "inferSteps", 100);
        int maxLen = BenchmarkArgs.getInt(cli, "maxLen", 20);
        long seed = BenchmarkArgs.getLong(cli, "seed", 42L);
        boolean mixedPrecision = BenchmarkArgs.getBoolean(cli, "mixedPrecision", false);
        String outputDir = BenchmarkArgs.getString(cli, "outputDir", "benchmark/results");
        String runId = BenchmarkArgs.getString(cli, "runId", "sentiment_rtpolarity_" + timestamp() + "_" + device);

        if (!"cpu".equals(device) && !"gpu".equals(device)) {
            throw new IllegalArgumentException("--device must be cpu or gpu");
        }

        Torch.manual_seed(seed);
        if (mixedPrecision) {
            MixedPrecision.enable();
        } else {
            MixedPrecision.disable();
        }

        List<MovieCommentLoader.Entry> all = MovieCommentLoader.load();
        Collections.shuffle(all, new Random(seed));
        int split = (int) (all.size() * 0.8);
        List<MovieCommentLoader.Entry> trainEntries = all.subList(0, split);
        List<MovieCommentLoader.Entry> testEntries = all.subList(split, all.size());
        System.out.printf(Locale.US, "[Benchmark][Sentiment] dataset loaded train=%d test=%d%n", trainEntries.size(), testEntries.size());

        Data.BasicTokenizer tokenizer = new Data.BasicTokenizer();
        Data.Vocabulary vocab = new Data.Vocabulary();
        for (MovieCommentLoader.Entry e : trainEntries) {
            for (String t : tokenizer.tokenize(e.text)) {
                vocab.addWord(t);
            }
        }
        System.out.printf(Locale.US, "[Benchmark][Sentiment] vocabulary size=%d%n", vocab.size());

        SentimentModel model = new SentimentModel(vocab.size(), 32, 64, 2);
        for (Parameter p : model.parameters()) {
            p.getTensor().requires_grad = true;
        }

        if ("gpu".equals(device)) {
            GpuMemoryPool.autoInit(model);
            System.out.println("[Benchmark][Sentiment] moving model to GPU...");
            model.toGPU();
            System.out.println("[Benchmark][Sentiment] model ready on GPU.");
        } else {
            model.toCPU();
        }

        Optim.Adam optimizer = new Optim.Adam(model.parameters(), 0.001f);
        Accuracy metric = new Accuracy();

        Path runDir = Paths.get(outputDir, "JavaTorch", "sentiment_rtpolarity", runId);
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

            float totalLoss = 0f;
            int numBatches = (trainEntries.size() + batchSize - 1) / batchSize;

            for (int b = 0; b < numBatches; b++) {
                try (MemoryScope scope = new MemoryScope()) {
                    int start = b * batchSize;
                    int end = Math.min(start + batchSize, trainEntries.size());
                    int currentBs = end - start;

                    float[] xData = new float[currentBs * maxLen];
                    int[] yLabels = new int[currentBs];
                    for (int i = 0; i < currentBs; i++) {
                        MovieCommentLoader.Entry entry = trainEntries.get(start + i);
                        List<String> tokens = tokenizer.tokenize(entry.text);
                        for (int j = 0; j < maxLen; j++) {
                            if (j < tokens.size()) {
                                xData[i * maxLen + j] = vocab.getId(tokens.get(j));
                            } else {
                                xData[i * maxLen + j] = 0f;
                            }
                        }
                        yLabels[i] = entry.label;
                    }

                    Tensor xBatch = Torch.tensor(xData, currentBs, maxLen);
                    if ("gpu".equals(device)) {
                        xBatch.toGPU();
                    }

                    optimizer.zero_grad();
                    Tensor logits = model.forward(xBatch);
                    Tensor loss = Functional.cross_entropy_tensor(logits, yLabels);
                    loss.backward();
                    optimizer.step();

                    totalLoss += loss.data[0];
                    metric.update(logits, yLabels);

                    int step = b + 1;
                    if (step == 1 || step % 50 == 0 || step == numBatches) {
                        System.out.printf(Locale.US,
                                "[Benchmark][Sentiment][Train] epoch=%d/%d batch=%d/%d loss=%.5f%n",
                                epoch + 1, epochs, step, numBatches, loss.data[0]);
                    }

                    peakHeapBytes = Math.max(peakHeapBytes, usedHeapBytes());
                    peakPoolUsedBytes = Math.max(peakPoolUsedBytes, GpuMemoryPool.getUsedBytes());
                }
            }

            double avgLoss = totalLoss / Math.max(1, numBatches);
            double trainAcc = metric.compute();
            double testAcc = evaluateSentiment(model, testEntries, tokenizer, vocab, maxLen, 64, device);
            long epochMs = (System.nanoTime() - epochStart) / 1_000_000L;

            if (testAcc > bestAcc) {
                bestAcc = testAcc;
                bestEpoch = epoch;
            }

            LinkedHashMap<String, String> row = baseRow(runId, device, seed, batchSize, epochs, mixedPrecision);
            row.put("framework", "JavaTorch");
            row.put("task", "sentiment_rtpolarity");
            row.put("epoch", String.valueOf(epoch + 1));
            row.put("train_loss", fmt(avgLoss));
            row.put("train_acc", fmt(trainAcc));
            row.put("val_acc", fmt(testAcc));
            row.put("epoch_time_ms", String.valueOf(epochMs));
            row.put("peak_heap_mb", fmt(bytesToMb(peakHeapBytes)));
            row.put("peak_vram_mb", fmt(bytesToMb(peakPoolUsedBytes)));
            BenchmarkCsv.appendRow(epochCsv, row);

            System.out.printf(Locale.US,
                    "[Benchmark][Sentiment] epoch=%d/%d loss=%.5f train_acc=%.4f val_acc=%.4f time_ms=%d%n",
                    epoch + 1, epochs, avgLoss, trainAcc, testAcc, epochMs);
        }

        InferenceResult infer = benchmarkInference(model, testEntries, tokenizer, vocab, maxLen, 64, device,
                inferWarmup, inferSteps, inferCsv, runId, seed, batchSize, epochs, mixedPrecision);

        long totalMs = (System.nanoTime() - wallStart) / 1_000_000L;
        LinkedHashMap<String, String> summary = baseRow(runId, device, seed, batchSize, epochs, mixedPrecision);
        summary.put("framework", "JavaTorch");
        summary.put("task", "sentiment_rtpolarity");
        summary.put("best_val_acc", fmt(bestAcc));
        summary.put("best_epoch", String.valueOf(bestEpoch + 1));
        summary.put("total_train_time_ms", String.valueOf(totalMs));
        summary.put("inference_p50_ms", fmt(infer.p50Ms));
        summary.put("inference_p95_ms", fmt(infer.p95Ms));
        summary.put("inference_throughput_sps", fmt(infer.throughputSps));
        summary.put("peak_heap_mb", fmt(bytesToMb(peakHeapBytes)));
        summary.put("peak_vram_mb", fmt(bytesToMb(peakPoolUsedBytes)));
        BenchmarkCsv.appendRow(summaryCsv, summary);

        if ("gpu".equals(device)) {
            GpuMemoryPool.destroy();
        }

        System.out.println("[Benchmark][Sentiment] Finished. Artifacts in: " + runDir.toAbsolutePath());
    }

    private static double evaluateSentiment(
            SentimentModel model,
            List<MovieCommentLoader.Entry> entries,
            Data.BasicTokenizer tokenizer,
            Data.Vocabulary vocab,
            int maxLen,
            int batchSize,
            String device) {

        Accuracy metric = new Accuracy();
        metric.reset();

        boolean prevGrad = Torch.is_grad_enabled();
        Torch.set_grad_enabled(false);
        model.eval();

        int numBatches = (entries.size() + batchSize - 1) / batchSize;
        try {
            for (int b = 0; b < numBatches; b++) {
                try (MemoryScope scope = new MemoryScope()) {
                    int start = b * batchSize;
                    int end = Math.min(start + batchSize, entries.size());
                    int currentBs = end - start;

                    float[] xData = new float[currentBs * maxLen];
                    int[] yLabels = new int[currentBs];

                    for (int i = 0; i < currentBs; i++) {
                        MovieCommentLoader.Entry entry = entries.get(start + i);
                        List<String> tokens = tokenizer.tokenize(entry.text);
                        for (int j = 0; j < maxLen; j++) {
                            if (j < tokens.size()) {
                                xData[i * maxLen + j] = vocab.getId(tokens.get(j));
                            } else {
                                xData[i * maxLen + j] = 0f;
                            }
                        }
                        yLabels[i] = entry.label;
                    }

                    Tensor xBatch = Torch.tensor(xData, currentBs, maxLen);
                    if ("gpu".equals(device)) {
                        xBatch.toGPU();
                    }
                    Tensor logits = model.forward(xBatch);
                    metric.update(logits, yLabels);
                }
            }
            return metric.compute();
        } finally {
            Torch.set_grad_enabled(prevGrad);
            model.train();
        }
    }

    private static InferenceResult benchmarkInference(
            SentimentModel model,
            List<MovieCommentLoader.Entry> entries,
            Data.BasicTokenizer tokenizer,
            Data.Vocabulary vocab,
            int maxLen,
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

        boolean prevGrad = Torch.is_grad_enabled();
        Torch.set_grad_enabled(false);
        model.eval();

        double[] latMs = new double[Math.max(1, measureSteps)];
        int measured = 0;
        int seen = 0;
        long totalSamples = 0;
        double totalLatencyMs = 0.0;

        int cursor = 0;
        try {
            while (seen < warmupSteps + measureSteps && cursor < entries.size()) {
                int end = Math.min(cursor + batchSize, entries.size());
                int currentBs = end - cursor;

                try (MemoryScope scope = new MemoryScope()) {
                    float[] xData = new float[currentBs * maxLen];
                    for (int i = 0; i < currentBs; i++) {
                        MovieCommentLoader.Entry entry = entries.get(cursor + i);
                        List<String> tokens = tokenizer.tokenize(entry.text);
                        for (int j = 0; j < maxLen; j++) {
                            if (j < tokens.size()) {
                                xData[i * maxLen + j] = vocab.getId(tokens.get(j));
                            } else {
                                xData[i * maxLen + j] = 0f;
                            }
                        }
                    }

                    Tensor xBatch = Torch.tensor(xData, currentBs, maxLen);
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
                        totalSamples += currentBs;

                        LinkedHashMap<String, String> row = baseRow(runId, device, seed, trainBatchSize, epochs,
                                mixedPrecision);
                        row.put("framework", "JavaTorch");
                        row.put("task", "sentiment_rtpolarity");
                        row.put("step", String.valueOf(measured));
                        row.put("batch_size", String.valueOf(currentBs));
                        row.put("latency_ms", fmt(ms));
                        BenchmarkCsv.appendRow(inferCsv, row);
                    }
                }

                seen++;
                cursor = end;
                if (cursor >= entries.size() && seen < warmupSteps + measureSteps) {
                    cursor = 0;
                }
            }
        } finally {
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
