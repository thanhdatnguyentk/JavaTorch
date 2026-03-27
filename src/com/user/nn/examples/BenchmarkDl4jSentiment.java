package com.user.nn.examples;

import com.user.nn.benchmark.BenchmarkArgs;
import com.user.nn.benchmark.BenchmarkCsv;
import com.user.nn.benchmark.BenchmarkStats;
import com.user.nn.dataloaders.Data;
import com.user.nn.dataloaders.MovieCommentLoader;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.EmbeddingLayer;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

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

public class BenchmarkDl4jSentiment {

    public static void main(String[] args) throws Exception {
        Map<String, String> cli = BenchmarkArgs.parse(args);

        String device = BenchmarkArgs.getString(cli, "device", "cpu").toLowerCase(Locale.ROOT);
        int epochs = BenchmarkArgs.getInt(cli, "epochs", 8);
        int batchSize = BenchmarkArgs.getInt(cli, "batchSize", 16);
        int inferWarmup = BenchmarkArgs.getInt(cli, "inferWarmup", 10);
        int inferSteps = BenchmarkArgs.getInt(cli, "inferSteps", 100);
        int maxLen = BenchmarkArgs.getInt(cli, "maxLen", 20);
        long seed = BenchmarkArgs.getLong(cli, "seed", 42L);
        String outputDir = BenchmarkArgs.getString(cli, "outputDir", "benchmark/results");
        String runId = BenchmarkArgs.getString(cli, "runId", "sentiment_rtpolarity_dl4j_" + timestamp() + "_cpu");

        System.out.println("[DL4J][Sentiment] Loading RT-Polarity data...");
        List<MovieCommentLoader.Entry> all = MovieCommentLoader.load();
        Collections.shuffle(all, new Random(seed));
        int split = (int) (all.size() * 0.8);
        List<MovieCommentLoader.Entry> trainEntries = all.subList(0, split);
        List<MovieCommentLoader.Entry> testEntries = all.subList(split, all.size());
        System.out.printf(Locale.US, "[DL4J][Sentiment] dataset loaded train=%d test=%d%n",
                trainEntries.size(), testEntries.size());

        Data.BasicTokenizer tokenizer = new Data.BasicTokenizer();
        Data.Vocabulary vocab = new Data.Vocabulary();
        for (MovieCommentLoader.Entry e : trainEntries) {
            for (String t : tokenizer.tokenize(e.text)) {
                vocab.addWord(t);
            }
        }
        System.out.printf(Locale.US, "[DL4J][Sentiment] vocabulary size=%d%n", vocab.size());

        System.out.println("[DL4J][Sentiment] Building LSTM sentiment model...");
        int vocabSize = vocab.size();
        MultiLayerConfiguration conf = buildLstmConfig(vocabSize, seed, "gpu".equals(device));
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        System.out.println("[DL4J][Sentiment] Model initialized with " + model.numParams() + " parameters");

        Path runDir = Paths.get(outputDir, "dl4j", "sentiment_rtpolarity", runId);
        Path epochCsv = runDir.resolve("epoch_metrics.csv");
        Path inferCsv = runDir.resolve("inference_samples.csv");
        Path summaryCsv = runDir.resolve("run_summary.csv");

        long wallStart = System.nanoTime();
        double bestAcc = Double.NEGATIVE_INFINITY;
        int bestEpoch = -1;
        long peakHeapBytes = 0L;

        for (int epoch = 0; epoch < epochs; epoch++) {
            long epochStart = System.nanoTime();

            float totalLoss = 0f;
            int numBatches = (trainEntries.size() + batchSize - 1) / batchSize;
            int correctTrain = 0;
            int totalTrain = 0;

            // Training loop
            for (int b = 0; b < numBatches; b++) {
                int start = b * batchSize;
                int end = Math.min(start + batchSize, trainEntries.size());
                int currentBs = end - start;

                // Create batch
                INDArray xBatch = Nd4j.zeros(currentBs, maxLen);
                INDArray yBatch = Nd4j.zeros(currentBs, 2);

                for (int i = 0; i < currentBs; i++) {
                    MovieCommentLoader.Entry entry = trainEntries.get(start + i);
                    List<String> tokens = tokenizer.tokenize(entry.text);
                    for (int j = 0; j < Math.min(maxLen, tokens.size()); j++) {
                        int wordIdx = vocab.getId(tokens.get(j));
                        xBatch.putScalar(i, j, wordIdx);
                    }
                    yBatch.putScalar(i, entry.label, 1.0f);
                }

                // Forward pass and training
                DataSet ds = new DataSet(xBatch, yBatch);
                double loss = model.score(ds);
                model.fit(ds);
                totalLoss += loss;

                // Compute accuracy
                INDArray predictions = model.output(xBatch);
                for (int i = 0; i < currentBs; i++) {
                    int predicted = Nd4j.argMax(predictions.getRow(i)).getInt(0);
                    int actual = trainEntries.get(start + i).label;
                    if (predicted == actual) {
                        correctTrain++;
                    }
                    totalTrain++;
                }

                if ((b + 1) % 50 == 0 || (b + 1) == 1) {
                    System.out.printf(Locale.US,
                            "[DL4J][Sentiment][Train] epoch=%d/%d batch=%d/%d loss=%.5f%n",
                            epoch + 1, epochs, b + 1, numBatches, loss);
                }

                peakHeapBytes = Math.max(peakHeapBytes, usedHeapBytes());
            }

            double trainAcc = (double) correctTrain / Math.max(1, totalTrain);
            double avgLoss = totalLoss / Math.max(1, numBatches);
            double testAcc = evaluateAccuracy(model, testEntries, tokenizer, vocab, maxLen, batchSize);
            long epochMs = (System.nanoTime() - epochStart) / 1_000_000L;

            if (testAcc > bestAcc) {
                bestAcc = testAcc;
                bestEpoch = epoch;
            }

            LinkedHashMap<String, String> row = baseRow(runId, device, seed, batchSize, epochs);
            row.put("framework", "dl4j");
            row.put("task", "sentiment_rtpolarity");
            row.put("epoch", String.valueOf(epoch + 1));
            row.put("train_loss", fmt(avgLoss));
            row.put("train_acc", fmt(trainAcc));
            row.put("val_acc", fmt(testAcc));
            row.put("epoch_time_ms", String.valueOf(epochMs));
            row.put("peak_heap_mb", fmt(bytesToMb(peakHeapBytes)));
            row.put("peak_vram_mb", "0.0");
            BenchmarkCsv.appendRow(epochCsv, row);

            System.out.printf(Locale.US,
                    "[DL4J][Sentiment] epoch=%d/%d loss=%.5f train_acc=%.4f val_acc=%.4f time_ms=%d%n",
                    epoch + 1, epochs, avgLoss, trainAcc, testAcc, epochMs);
        }

        // Inference benchmark
        InferenceResult infer = benchmarkInference(model, testEntries, tokenizer, vocab, maxLen, batchSize,
                inferWarmup, inferSteps, inferCsv, runId, seed, batchSize, epochs);

        long totalMs = (System.nanoTime() - wallStart) / 1_000_000L;
        LinkedHashMap<String, String> summary = baseRow(runId, device, seed, batchSize, epochs);
        summary.put("framework", "dl4j");
        summary.put("task", "sentiment_rtpolarity");
        summary.put("best_val_acc", fmt(bestAcc));
        summary.put("best_epoch", String.valueOf(bestEpoch + 1));
        summary.put("total_train_time_ms", String.valueOf(totalMs));
        summary.put("inference_p50_ms", fmt(infer.p50Ms));
        summary.put("inference_p95_ms", fmt(infer.p95Ms));
        summary.put("inference_throughput_sps", fmt(infer.throughputSps));
        summary.put("peak_heap_mb", fmt(bytesToMb(peakHeapBytes)));
        summary.put("peak_vram_mb", "0.0");
        BenchmarkCsv.appendRow(summaryCsv, summary);

        System.out.println("[DL4J][Sentiment] Finished. Artifacts in: " + runDir.toAbsolutePath());
    }

    private static MultiLayerConfiguration buildLstmConfig(int vocabSize, long seed, boolean useGpu) {
        NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.RELU)
                .updater(new org.nd4j.linalg.learning.config.Adam())
                .regularization(true)
                .l2(0.0001);

        return builder
                .list()
                .layer(0, new EmbeddingLayer.Builder()
                        .nIn(vocabSize).nOut(32)
                        .build())
                .layer(1, new GravesLSTM.Builder()
                        .nIn(32).nOut(64)
                        .activation(Activation.TANH)
                        .build())
                .layer(2, new DenseLayer.Builder()
                        .nIn(64).nOut(32)
                        .activation(Activation.RELU)
                        .build())
                .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .nIn(32).nOut(2)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.recurrent(vocabSize))
                .build();
    }

    private static double evaluateAccuracy(
            MultiLayerNetwork model,
            List<MovieCommentLoader.Entry> entries,
            Data.BasicTokenizer tokenizer,
            Data.Vocabulary vocab,
            int maxLen,
            int batchSize) {

        int correct = 0;
        int total = 0;

        int numBatches = (entries.size() + batchSize - 1) / batchSize;
        for (int b = 0; b < numBatches; b++) {
            int start = b * batchSize;
            int end = Math.min(start + batchSize, entries.size());
            int currentBs = end - start;

            INDArray xBatch = Nd4j.zeros(currentBs, maxLen);
            for (int i = 0; i < currentBs; i++) {
                MovieCommentLoader.Entry entry = entries.get(start + i);
                List<String> tokens = tokenizer.tokenize(entry.text);
                for (int j = 0; j < Math.min(maxLen, tokens.size()); j++) {
                    int wordIdx = vocab.getId(tokens.get(j));
                    xBatch.putScalar(i, j, wordIdx);
                }
            }

            INDArray predictions = model.output(xBatch);
            for (int i = 0; i < currentBs; i++) {
                int predicted = Nd4j.argMax(predictions.getRow(i)).getInt(0);
                int actual = entries.get(start + i).label;
                if (predicted == actual) {
                    correct++;
                }
                total++;
            }
        }

        return (double) correct / Math.max(1, total);
    }

    private static InferenceResult benchmarkInference(
            MultiLayerNetwork model,
            List<MovieCommentLoader.Entry> entries,
            Data.BasicTokenizer tokenizer,
            Data.Vocabulary vocab,
            int maxLen,
            int batchSize,
            int warmupSteps,
            int measureSteps,
            Path inferCsv,
            String runId,
            long seed,
            int trainBatchSize,
            int epochs) throws IOException {

        double[] latMs = new double[Math.max(1, measureSteps)];
        int measured = 0;
        int seen = 0;
        long totalSamples = 0;
        double totalLatencyMs = 0.0;
        int cursor = 0;

        while (seen < warmupSteps + measureSteps && cursor < entries.size()) {
            int end = Math.min(cursor + batchSize, entries.size());
            int currentBs = end - cursor;

            INDArray xBatch = Nd4j.zeros(currentBs, maxLen);
            for (int i = 0; i < currentBs; i++) {
                MovieCommentLoader.Entry entry = entries.get(cursor + i);
                List<String> tokens = tokenizer.tokenize(entry.text);
                for (int j = 0; j < Math.min(maxLen, tokens.size()); j++) {
                    int wordIdx = vocab.getId(tokens.get(j));
                    xBatch.putScalar(i, j, wordIdx);
                }
            }

            long t0 = System.nanoTime();
            model.output(xBatch);
            long t1 = System.nanoTime();

            if (seen >= warmupSteps) {
                double ms = (t1 - t0) / 1_000_000.0;
                latMs[measured] = ms;
                measured++;
                totalLatencyMs += ms;
                totalSamples += currentBs;

                LinkedHashMap<String, String> row = baseRow(runId, "gpu", seed, trainBatchSize, epochs);
                row.put("framework", "dl4j");
                row.put("task", "sentiment_rtpolarity");
                row.put("step", String.valueOf(measured));
                row.put("batch_size", String.valueOf(currentBs));
                row.put("latency_ms", fmt(ms));
                BenchmarkCsv.appendRow(inferCsv, row);
            }

            seen++;
            cursor = end;
            if (cursor >= entries.size() && seen < warmupSteps + measureSteps) {
                cursor = 0;
            }
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
            int epochs) {
        LinkedHashMap<String, String> row = new LinkedHashMap<>();
        row.put("run_id", runId);
        row.put("timestamp", timestamp());
        row.put("device", device);
        row.put("seed", String.valueOf(seed));
        row.put("train_batch_size", String.valueOf(batchSize));
        row.put("epochs", String.valueOf(epochs));
        row.put("mixed_precision", "false");
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
