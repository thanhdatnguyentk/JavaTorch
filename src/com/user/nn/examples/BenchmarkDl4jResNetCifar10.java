package com.user.nn.examples;

import com.user.nn.benchmark.BenchmarkArgs;
import com.user.nn.benchmark.BenchmarkCsv;
import com.user.nn.benchmark.BenchmarkStats;
import com.user.nn.dataloaders.Cifar10Loader;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
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
import java.util.Date;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;

public class BenchmarkDl4jResNetCifar10 {

    public static void main(String[] args) throws Exception {
        Map<String, String> cli = BenchmarkArgs.parse(args);

        String device = BenchmarkArgs.getString(cli, "device", "cpu").toLowerCase(Locale.ROOT);
        int epochs = BenchmarkArgs.getInt(cli, "epochs", 2);
        int batchSize = BenchmarkArgs.getInt(cli, "batchSize", 128);
        int inferWarmup = BenchmarkArgs.getInt(cli, "inferWarmup", 2);
        int inferSteps = BenchmarkArgs.getInt(cli, "inferSteps", 3);
        long seed = BenchmarkArgs.getLong(cli, "seed", 42L);
        String outputDir = BenchmarkArgs.getString(cli, "outputDir", "benchmark/results");
        String runId = BenchmarkArgs.getString(cli, "runId", "resnet_cifar10_dl4j_" + timestamp() + "_cpu");

        System.out.println("[DL4J][ResNet] Preparing CIFAR-10 data...");
        
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

        System.out.println("[DL4J][ResNet] Building ConvNet architecture...");
        MultiLayerConfiguration conf = buildConvNetConfig(seed);
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        System.out.println("[DL4J][ResNet] Model initialized with " + model.numParams() + " parameters");

        Path runDir = Paths.get(outputDir, "dl4j", "resnet_cifar10", runId);
        Path epochCsv = runDir.resolve("epoch_metrics.csv");
        Path inferCsv = runDir.resolve("inference_samples.csv");
        Path summaryCsv = runDir.resolve("run_summary.csv");

        System.out.println("[DL4J][ResNet] Starting training...");
        long wallStart = System.nanoTime();
        double bestAcc = 0.0;
        int bestEpoch = -1;
        long peakHeapBytes = 0L;

        for (int epoch = 0; epoch < epochs; epoch++) {
            long epochStart = System.nanoTime();

            float totalLoss = 0f;
            int batchCount = 0;
            int correctTrain = 0;
            int totalTrain = 0;

            // Training batches
            for (int i = 0; i < trainImages.length; i += batchSize) {
                int end = Math.min(i + batchSize, trainImages.length);
                int currentBatchSize = end - i;

                INDArray xBatch = Nd4j.zeros(currentBatchSize, 3, 32, 32);
                INDArray yBatch = Nd4j.zeros(currentBatchSize, 10);

                for (int j = 0; j < currentBatchSize; j++) {
                    float[] img = trainImages[i + j];
                    INDArray imgTensor = Nd4j.create(img).reshape(3, 1024);
                    xBatch.putRow(j, imgTensor);
                    yBatch.putScalar(j, trainLabels[i + j], 1.0f);
                }

                DataSet ds = new DataSet(xBatch, yBatch);
                double loss = model.score(ds);
                model.fit(ds);
                totalLoss += loss;

                // Compute accuracy
                INDArray predictions = model.output(xBatch);
                for (int j = 0; j < currentBatchSize; j++) {
                    int pred = Nd4j.argMax(predictions.getRow(j)).getInt(0);
                    if (pred == trainLabels[i + j]) {
                        correctTrain++;
                    }
                    totalTrain++;
                }

                batchCount++;
                if (batchCount % 10 == 0 || batchCount == 1) {
                    System.out.printf(Locale.US,
                            "[DL4J][ResNet][Train] epoch=%d/%d batch=%d/%d loss=%.5f%n",
                            epoch + 1, epochs, batchCount, (trainImages.length + batchSize - 1) / batchSize, loss);
                }

                peakHeapBytes = Math.max(peakHeapBytes, usedHeapBytes());
            }

            double trainAcc = (double) correctTrain / Math.max(1, totalTrain);
            double avgLoss = totalLoss / Math.max(1, batchCount);
            double testAcc = evaluateAccuracy(model, testImages, testLabels, 256);
            long epochMs = (System.nanoTime() - epochStart) / 1_000_000L;

            if (testAcc > bestAcc) {
                bestAcc = testAcc;
                bestEpoch = epoch;
            }

            LinkedHashMap<String, String> row = baseRow(runId, "cpu", seed, batchSize, epochs);
            row.put("framework", "dl4j");
            row.put("task", "resnet_cifar10");
            row.put("epoch", String.valueOf(epoch + 1));
            row.put("train_loss", fmt(avgLoss));
            row.put("train_acc", fmt(trainAcc));
            row.put("val_acc", fmt(testAcc));
            row.put("epoch_time_ms", String.valueOf(epochMs));
            row.put("peak_heap_mb", fmt(bytesToMb(peakHeapBytes)));
            row.put("peak_vram_mb", "0.0");
            BenchmarkCsv.appendRow(epochCsv, row);

            System.out.printf(Locale.US,
                    "[DL4J][ResNet] epoch=%d/%d loss=%.5f train_acc=%.4f val_acc=%.4f time_ms=%d%n",
                    epoch + 1, epochs, avgLoss, trainAcc, testAcc, epochMs);
        }

        InferenceResult infer = benchmarkInference(model, testImages, testLabels, 256, inferWarmup,
                inferSteps, inferCsv, runId, seed, batchSize, epochs);

        long totalMs = (System.nanoTime() - wallStart) / 1_000_000L;
        LinkedHashMap<String, String> summary = baseRow(runId, "cpu", seed, batchSize, epochs);
        summary.put("framework", "dl4j");
        summary.put("task", "resnet_cifar10");
        summary.put("best_val_acc", fmt(bestAcc));
        summary.put("best_epoch", String.valueOf(bestEpoch + 1));
        summary.put("total_train_time_ms", String.valueOf(totalMs));
        summary.put("inference_p50_ms", fmt(infer.p50Ms));
        summary.put("inference_p95_ms", fmt(infer.p95Ms));
        summary.put("inference_throughput_sps", fmt(infer.throughputSps));
        summary.put("peak_heap_mb", fmt(bytesToMb(peakHeapBytes)));
        summary.put("peak_vram_mb", "0.0");
        BenchmarkCsv.appendRow(summaryCsv, summary);

        System.out.println("[DL4J][ResNet] Finished. Artifacts in: " + runDir.toAbsolutePath());
    }

    private static MultiLayerConfiguration buildConvNetConfig(long seed) {
        return new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.RELU)
                .activation(Activation.RELU)
                .updater(new org.nd4j.linalg.learning.config.Adam())
                .regularization(true)
                .l2(0.0001)
                .convolutionMode(org.deeplearning4j.nn.conf.ConvolutionMode.Same)
                .list()
                .layer(0, new ConvolutionLayer.Builder(7, 7)
                        .nIn(3).nOut(64)
                        .stride(2, 2)
                        .activation(Activation.RELU).build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(3, 3)
                        .stride(2, 2).build())
                .layer(2, new ConvolutionLayer.Builder(3, 3)
                        .nIn(64).nOut(64)
                        .activation(Activation.RELU).build())
                .layer(3, new ConvolutionLayer.Builder(3, 3)
                        .nIn(64).nOut(64)
                        .activation(Activation.RELU).build())
                .layer(4, new ConvolutionLayer.Builder(3, 3)
                        .nIn(64).nOut(128)
                        .stride(2, 2)
                        .activation(Activation.RELU).build())
                .layer(5, new ConvolutionLayer.Builder(3, 3)
                        .nIn(128).nOut(128)
                        .activation(Activation.RELU).build())
                .layer(6, new ConvolutionLayer.Builder(3, 3)
                        .nIn(128).nOut(256)
                        .stride(2, 2)
                        .activation(Activation.RELU).build())
                .layer(7, new ConvolutionLayer.Builder(3, 3)
                        .nIn(256).nOut(256)
                        .activation(Activation.RELU).build())
                .layer(8, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG)
                        .kernelSize(7, 7).build())
                .layer(9, new DenseLayer.Builder()
                        .nIn(256).nOut(512)
                        .activation(Activation.RELU).build())
                .layer(10, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .nIn(512).nOut(10)
                        .activation(Activation.SOFTMAX).build())
                .setInputType(InputType.convolutional(32, 32, 3))
                .build();
    }

    private static double evaluateAccuracy(MultiLayerNetwork model, float[][] testImages, int[] testLabels,
            int batchSize) {
        int correct = 0;
        int total = 0;

        for (int i = 0; i < testImages.length; i += batchSize) {
            int end = Math.min(i + batchSize, testImages.length);
            int currentBatchSize = end - i;

            INDArray xBatch = Nd4j.zeros(currentBatchSize, 3, 32, 32);
            for (int j = 0; j < currentBatchSize; j++) {
                float[] img = testImages[i + j];
                xBatch.putRow(j, Nd4j.create(img).reshape(3, 1024));
            }

            INDArray predictions = model.output(xBatch);
            for (int j = 0; j < currentBatchSize; j++) {
                int pred = Nd4j.argMax(predictions.getRow(j)).getInt(0);
                if (pred == testLabels[i + j]) {
                    correct++;
                }
                total++;
            }
        }

        return (double) correct / Math.max(1, total);
    }

    private static InferenceResult benchmarkInference(
            MultiLayerNetwork model,
            float[][] testImages,
            int[] testLabels,
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

        for (int i = 0; i < testImages.length; i += batchSize) {
            if (seen >= warmupSteps + measureSteps) {
                break;
            }

            int end = Math.min(i + batchSize, testImages.length);
            int currentBatchSize = end - i;

            INDArray xBatch = Nd4j.zeros(currentBatchSize, 3, 32, 32);
            for (int j = 0; j < currentBatchSize; j++) {
                float[] img = testImages[i + j];
                xBatch.putRow(j, Nd4j.create(img).reshape(3, 1024));
            }

            long t0 = System.nanoTime();
            model.output(xBatch);
            long t1 = System.nanoTime();

            if (seen >= warmupSteps) {
                double ms = (t1 - t0) / 1_000_000.0;
                latMs[measured] = ms;
                measured++;
                totalLatencyMs += ms;
                totalSamples += currentBatchSize;

                LinkedHashMap<String, String> row = baseRow(runId, "cpu", seed, trainBatchSize, epochs);
                row.put("framework", "dl4j");
                row.put("task", "resnet_cifar10");
                row.put("step", String.valueOf(measured));
                row.put("batch_size", String.valueOf(currentBatchSize));
                row.put("latency_ms", fmt(ms));
                BenchmarkCsv.appendRow(inferCsv, row);
            }
            seen++;
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
