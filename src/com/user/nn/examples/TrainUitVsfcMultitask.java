package com.user.nn.examples;

import com.user.nn.utils.dashboard.DashboardServer;
import com.user.nn.utils.dashboard.DashboardIntegrationHelper;
import com.user.nn.utils.visualization.TrainingHistory;
import java.util.HashMap;
import java.util.Map;


import com.user.nn.benchmark.BenchmarkArgs;
import com.user.nn.benchmark.BenchmarkCsv;
import com.user.nn.benchmark.BenchmarkStats;
import com.user.nn.core.Functional;
import com.user.nn.core.GpuMemoryPool;
import com.user.nn.core.MemoryScope;
import com.user.nn.core.MixedPrecision;
import com.user.nn.core.Module;
import com.user.nn.core.Parameter;
import com.user.nn.core.Tensor;
import com.user.nn.core.Torch;
import com.user.nn.dataloaders.Data;
import com.user.nn.dataloaders.UitVsfcLoader;
import com.user.nn.metrics.Accuracy;
import com.user.nn.models.MultiTaskLSTMModel;
import com.user.nn.models.MultiTaskTransformerModel;
import com.user.nn.optim.Optim;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;

import static jcuda.runtime.JCuda.cudaDeviceSynchronize;

public class TrainUitVsfcMultitask {

    private static final String TASK_NAME = "uit_vsfc_multitask";

    private static final class TaskMetrics {
        final float accuracy;
        final float macroF1;
        final float[] precision;
        final float[] recall;
        final float[] f1;
        final int[][] confusion;

        TaskMetrics(float accuracy, float macroF1, float[] precision, float[] recall, float[] f1, int[][] confusion) {
            this.accuracy = accuracy;
            this.macroF1 = macroF1;
            this.precision = precision;
            this.recall = recall;
            this.f1 = f1;
            this.confusion = confusion;
        }
    }

    private static final class EvalResult {
        final float loss;
        final TaskMetrics sentiment;
        final TaskMetrics topic;
        final float jointExactMatch;

        EvalResult(float loss, TaskMetrics sentiment, TaskMetrics topic, float jointExactMatch) {
            this.loss = loss;
            this.sentiment = sentiment;
            this.topic = topic;
            this.jointExactMatch = jointExactMatch;
        }
    }

    private static final class BatchData {
        final float[] xData;
        final int[] sentimentLabels;
        final int[] topicLabels;
        final int batchSize;

        BatchData(float[] xData, int[] sentimentLabels, int[] topicLabels, int batchSize) {
            this.xData = xData;
            this.sentimentLabels = sentimentLabels;
            this.topicLabels = topicLabels;
            this.batchSize = batchSize;
        }
    }

    public static void main(String[] args) throws Exception {
        Map<String, String> cli = BenchmarkArgs.parse(args);

        String dataDir = getPositionalArg(args, 0, BenchmarkArgs.getString(cli, "dataDir", UitVsfcLoader.DEFAULT_DATA_DIR));
        int epochs = parsePositionalInt(args, 1, BenchmarkArgs.getInt(cli, "epochs", 10));
        int batchSize = parsePositionalInt(args, 2, BenchmarkArgs.getInt(cli, "batchSize", 32));
        int maxLen = parsePositionalInt(args, 3, BenchmarkArgs.getInt(cli, "maxLen", 64));
        int inferWarmup = BenchmarkArgs.getInt(cli, "inferWarmup", 10);
        int inferSteps = BenchmarkArgs.getInt(cli, "inferSteps", 50);
        String modelType = getPositionalArg(args, 4, BenchmarkArgs.getString(cli, "model", "lstm")).toLowerCase(Locale.ROOT);
        String device = getPositionalArg(args, 5, BenchmarkArgs.getString(cli, "device", "cpu")).toLowerCase(Locale.ROOT);
        long seed = parsePositionalLong(args, 6, BenchmarkArgs.getLong(cli, "seed", 42L));

        float alpha = parseFloatArg(cli, "alpha", 1.0f);
        float beta = parseFloatArg(cli, "beta", 1.0f);
        float learningRate = parseFloatArg(cli, "learningRate", 0.001f);
        float minLearningRate = parseFloatArg(cli, "minLearningRate", 0.0001f);
        int lrWarmupEpochs = BenchmarkArgs.getInt(cli, "lrWarmupEpochs", 0);
        String lrSchedule = BenchmarkArgs.getString(cli, "lrSchedule", "none").toLowerCase(Locale.ROOT);
        int earlyStoppingPatience = BenchmarkArgs.getInt(cli, "earlyStoppingPatience", 0);
        float earlyStoppingMinDelta = parseFloatArg(cli, "earlyStoppingMinDelta", 0.0f);
        String outputDir = BenchmarkArgs.getString(cli, "outputDir", "benchmark/results");
        String runId = BenchmarkArgs.getString(cli, "runId", "uit_vsfc_" + modelType + "_" + timestamp() + "_" + device);
        String selection = BenchmarkArgs.getString(cli, "selection", "weighted").toLowerCase(Locale.ROOT);
        String checkpointPathArg = BenchmarkArgs.getString(cli, "checkpointPath", "").trim();

        if (!"lstm".equals(modelType) && !"transformer".equals(modelType)) {
            throw new IllegalArgumentException("modelType must be lstm or transformer");
        }
        if (!"cpu".equals(device) && !"gpu".equals(device)) {
            throw new IllegalArgumentException("device must be cpu or gpu");
        }
        if (!"weighted".equals(selection) && !"sentiment".equals(selection) && !"topic".equals(selection)) {
            throw new IllegalArgumentException("selection must be one of: weighted|sentiment|topic");
        }
        if (alpha < 0f || beta < 0f || alpha + beta <= 0f) {
            throw new IllegalArgumentException("alpha/beta must be non-negative and not both zero");
        }
        if (learningRate <= 0f) {
            throw new IllegalArgumentException("learningRate must be > 0");
        }
        if (minLearningRate < 0f) {
            throw new IllegalArgumentException("minLearningRate must be >= 0");
        }
        if (minLearningRate > learningRate) {
            throw new IllegalArgumentException("minLearningRate must be <= learningRate");
        }
        if (lrWarmupEpochs < 0) {
            throw new IllegalArgumentException("lrWarmupEpochs must be >= 0");
        }
        if (!"none".equals(lrSchedule) && !"cosine".equals(lrSchedule)) {
            throw new IllegalArgumentException("lrSchedule must be one of: none|cosine");
        }
        if (earlyStoppingPatience < 0) {
            throw new IllegalArgumentException("earlyStoppingPatience must be >= 0");
        }
        if (earlyStoppingMinDelta < 0f) {
            throw new IllegalArgumentException("earlyStoppingMinDelta must be >= 0");
        }

        Torch.manual_seed(seed);
        MixedPrecision.disable();

        Path runDir = Paths.get(outputDir, "JavaTorch", TASK_NAME, runId);
        Path epochCsv = runDir.resolve("epoch_metrics.csv");
        Path inferCsv = runDir.resolve("inference_samples.csv");
        Path inferBreakdownCsv = runDir.resolve("inference_breakdown.csv");
        Path summaryCsv = runDir.resolve("run_summary.csv");
        Path perClassCsv = runDir.resolve("per_class_metrics.csv");
        Path bestCheckpointPath = checkpointPathArg.isEmpty() ? runDir.resolve("best_model.bin") : Paths.get(checkpointPathArg);
        Files.createDirectories(runDir);
        if (bestCheckpointPath.getParent() != null) {
            Files.createDirectories(bestCheckpointPath.getParent());
        }

        System.out.println("Loading UIT-VSFC dataset from: " + dataDir);
        UitVsfcLoader.DatasetSplits splits = UitVsfcLoader.load(dataDir);
        List<String> sentimentLabels = splits.sentimentEncoder.labels();
        List<String> topicLabels = splits.topicEncoder.labels();

        System.out.printf(Locale.US, "Train=%d Dev=%d Test=%d%n", splits.train.size(), splits.dev.size(), splits.test.size());
        System.out.printf(Locale.US, "Sentiment classes=%d, Topic classes=%d%n", sentimentLabels.size(), topicLabels.size());
        System.out.printf(Locale.US,
                "Config | model=%s device=%s epochs=%d batchSize=%d maxLen=%d alpha=%.3f beta=%.3f seed=%d selection=%s runId=%s%n",
                modelType, device, epochs, batchSize, maxLen, alpha, beta, seed, selection, runId);
        System.out.printf(Locale.US,
            "Training controls | lr=%.6f minLr=%.6f schedule=%s warmupEpochs=%d earlyStopPatience=%d minDelta=%.6f%n",
            learningRate, minLearningRate, lrSchedule, lrWarmupEpochs, earlyStoppingPatience, earlyStoppingMinDelta);

        UitVsfcLoader.VietnameseTokenizer tokenizer = new UitVsfcLoader.VietnameseTokenizer();
        Data.Vocabulary vocab = buildVocabulary(splits.train, tokenizer);
        Map<String, float[]> tokenIdCache = new HashMap<>(Math.max(1024, splits.train.size() + splits.dev.size() + splits.test.size()));
        System.out.println("Vocabulary size: " + vocab.size());

        Module model = createModel(modelType, vocab.size(), maxLen, sentimentLabels.size(), topicLabels.size());

        for (Parameter p : model.parameters()) {
            p.getTensor().requires_grad = true;
        }

        if ("gpu".equals(device)) {
            GpuMemoryPool.autoInit(model);
            model.toGPU();
            System.out.println("Model moved to GPU");
        } else {
            model.toCPU();
        }

        Optim.Adam optimizer = new Optim.Adam(model.parameters(), learningRate);
        List<UitVsfcLoader.Entry> shuffledTrain = new ArrayList<>(splits.train);

        long totalStartNs = System.nanoTime();
        long cumulativeEpochMs = 0L;
        float bestDevObjective = Float.NEGATIVE_INFINITY;
        int bestEpoch = -1;
        int executedEpochs = 0;
        int epochsWithoutImprove = 0;
        boolean stoppedEarly = false;

        
        TrainingHistory history = new TrainingHistory();
        DashboardServer dashboard = new DashboardServer(7070, history).start();
        try {
            com.user.nn.predict.TextPredictor predictor = com.user.nn.predict.TextPredictor.forSentiment(model, vocab, maxLen);
            DashboardIntegrationHelper.setupTextPredictorHandler(dashboard, "sentiment", predictor);
        } catch(Exception e) {}




        for (int epoch = 0; epoch < epochs; epoch++) {
            long epochStartNs = System.nanoTime();
            float currentLr = computeLearningRate(
                    learningRate,
                    minLearningRate,
                    epoch,
                    epochs,
                    lrWarmupEpochs,
                    lrSchedule);
            optimizer.setLearningRate(currentLr);
            Collections.shuffle(shuffledTrain, new Random(seed + epoch));

            model.train();
            Accuracy sentimentTrainAcc = new Accuracy();
            Accuracy topicTrainAcc = new Accuracy();
            float totalLoss = 0f;
            float totalSentLoss = 0f;
            float totalTopicLoss = 0f;

            int numBatches = (shuffledTrain.size() + batchSize - 1) / batchSize;
            for (int b = 0; b < numBatches; b++) {
                try (MemoryScope scope = new MemoryScope()) {
                    BatchData batch = createBatch(
                            shuffledTrain,
                            b * batchSize,
                            Math.min((b + 1) * batchSize, shuffledTrain.size()),
                            tokenizer,
                            vocab,
                            tokenIdCache,
                            maxLen);

                    Tensor xBatch = Torch.tensor(batch.xData, batch.batchSize, maxLen);
                    if ("gpu".equals(device)) {
                        xBatch.toGPU();
                    }

                    optimizer.zero_grad();
                    Tensor[] logits = forwardBoth(model, xBatch);

                    Tensor lossSent = Functional.cross_entropy_tensor(logits[0], batch.sentimentLabels);
                    Tensor lossTopic = Functional.cross_entropy_tensor(logits[1], batch.topicLabels);
                    Tensor weightedSent = Torch.mul(lossSent, alpha);
                    Tensor weightedTopic = Torch.mul(lossTopic, beta);
                    Tensor loss = Torch.add(weightedSent, weightedTopic);

                    loss.backward();
                    optimizer.step();

                    totalLoss += loss.data[0];
                    totalSentLoss += lossSent.data[0];
                    totalTopicLoss += lossTopic.data[0];

                    sentimentTrainAcc.update(logits[0], batch.sentimentLabels);
                    topicTrainAcc.update(logits[1], batch.topicLabels);
                }
            }

            float trainLoss = totalLoss / Math.max(1, numBatches);
            float trainSentLoss = totalSentLoss / Math.max(1, numBatches);
            float trainTopicLoss = totalTopicLoss / Math.max(1, numBatches);
            float trainSentAcc = sentimentTrainAcc.compute();
            float trainTopicAcc = topicTrainAcc.compute();

            EvalResult devEval = evaluate(
                    model,
                    splits.dev,
                    tokenizer,
                    vocab,
                    tokenIdCache,
                    maxLen,
                    batchSize,
                    device,
                    alpha,
                    beta,
                    sentimentLabels.size(),
                    topicLabels.size());

            float devObjective = selectObjective(devEval, selection, alpha, beta);
            if (isImproved(devObjective, bestDevObjective, earlyStoppingMinDelta)) {
                bestDevObjective = devObjective;
                bestEpoch = epoch + 1;
                model.save(bestCheckpointPath.toString());
                epochsWithoutImprove = 0;
            } else {
                epochsWithoutImprove++;
            }

            long epochMs = (System.nanoTime() - epochStartNs) / 1_000_000L;
            cumulativeEpochMs += epochMs;
            executedEpochs = epoch + 1;
            double epochSec = epochMs / 1000.0;
            double avgBatchTimeMs = numBatches > 0 ? (double) epochMs / numBatches : 0.0;
            double throughputSps = epochSec > 0.0 ? shuffledTrain.size() / epochSec : 0.0;

            LinkedHashMap<String, String> row = baseRow(runId, modelType, device, seed, batchSize, epochs, maxLen, alpha, beta);
            row.put("epoch", String.valueOf(epoch + 1));
            row.put("train_loss_total", fmt(trainLoss));
            row.put("train_loss_sentiment", fmt(trainSentLoss));
            row.put("train_loss_topic", fmt(trainTopicLoss));
            row.put("train_sent_acc", fmt(trainSentAcc));
            row.put("train_topic_acc", fmt(trainTopicAcc));
            row.put("dev_loss", fmt(devEval.loss));
            row.put("dev_sent_acc", fmt(devEval.sentiment.accuracy));
            row.put("dev_topic_acc", fmt(devEval.topic.accuracy));
            row.put("dev_sent_macro_f1", fmt(devEval.sentiment.macroF1));
            row.put("dev_topic_macro_f1", fmt(devEval.topic.macroF1));
            row.put("dev_joint_exact_match", fmt(devEval.jointExactMatch));
            row.put("dev_objective", fmt(devObjective));
                row.put("learning_rate", fmt(currentLr));
                row.put("epochs_without_improve", String.valueOf(epochsWithoutImprove));
            row.put("epoch_time_ms", String.valueOf(epochMs));
            row.put("cumulative_time_ms", String.valueOf(cumulativeEpochMs));
            row.put("avg_batch_time_ms", fmt(avgBatchTimeMs));
            row.put("throughput_samples_per_sec", fmt(throughputSps));
            BenchmarkCsv.appendRow(epochCsv, row);

            System.out.printf(Locale.US,
                    "Epoch %d/%d | lr=%.6f | train_loss=%.4f (sent=%.4f topic=%.4f) | train_sent_acc=%.2f%% | train_topic_acc=%.2f%% | dev_sent_macro_f1=%.4f | dev_topic_macro_f1=%.4f | dev_joint=%.4f | objective=%.4f | epoch_time=%.3fs%n",
                    epoch + 1,
                    epochs,
                    currentLr,
                    trainLoss,
                    trainSentLoss,
                    trainTopicLoss,
                    trainSentAcc * 100f,
                    trainTopicAcc * 100f,
                    devEval.sentiment.macroF1,
                    devEval.topic.macroF1,
                    devEval.jointExactMatch,
                    devObjective,
                    epochSec);

                if (earlyStoppingPatience > 0 && epochsWithoutImprove >= earlyStoppingPatience) {
                stoppedEarly = true;
                System.out.printf(Locale.US,
                    "Early stopping triggered at epoch %d/%d (patience=%d, minDelta=%.6f)%n",
                    epoch + 1,
                    epochs,
                    earlyStoppingPatience,
                    earlyStoppingMinDelta);
                break;
                }
        
            try {
                Map<String, Float> metrics = new HashMap<>();
                metrics.put("epoch", (float)epoch);
                history.record(epoch + 1, metrics);
                dashboard.broadcastMetrics(epoch + 1, metrics);
            } catch (Exception dashEx) {}
}

        if (bestEpoch <= 0) {
            throw new IllegalStateException("No best checkpoint selected during training");
        }

        model.load(bestCheckpointPath.toString());
        System.out.println("Loaded best checkpoint from: " + bestCheckpointPath.toAbsolutePath());

        EvalResult bestDevEval = evaluate(
                model,
                splits.dev,
                tokenizer,
                vocab,
                    tokenIdCache,
                maxLen,
                batchSize,
                device,
                alpha,
                beta,
                sentimentLabels.size(),
                topicLabels.size());

        EvalResult testEval = evaluate(
                model,
                splits.test,
                tokenizer,
                vocab,
                tokenIdCache,
                maxLen,
                batchSize,
                device,
                alpha,
                beta,
                sentimentLabels.size(),
                topicLabels.size());

            InferenceResult infer = benchmarkInference(
                model,
                splits.test,
                tokenizer,
                vocab,
                tokenIdCache,
                maxLen,
                batchSize,
                device,
                inferWarmup,
                inferSteps,
                inferCsv,
                inferBreakdownCsv,
                runId,
                seed,
                epochs,
                modelType,
                alpha,
                beta);

        long totalTrainMs = (System.nanoTime() - totalStartNs) / 1_000_000L;

        Path finalModelPath = runDir.resolve("uit_vsfc_" + modelType + "_multitask.bin");
        model.save(finalModelPath.toString());

        writeConfusionCsv(runDir.resolve("dev_confusion_sentiment.csv"), bestDevEval.sentiment.confusion, sentimentLabels);
        writeConfusionCsv(runDir.resolve("dev_confusion_topic.csv"), bestDevEval.topic.confusion, topicLabels);
        writeConfusionCsv(runDir.resolve("test_confusion_sentiment.csv"), testEval.sentiment.confusion, sentimentLabels);
        writeConfusionCsv(runDir.resolve("test_confusion_topic.csv"), testEval.topic.confusion, topicLabels);

        writePerClassRows(perClassCsv, runId, modelType, device, "dev", "sentiment", sentimentLabels, bestDevEval.sentiment);
        writePerClassRows(perClassCsv, runId, modelType, device, "dev", "topic", topicLabels, bestDevEval.topic);
        writePerClassRows(perClassCsv, runId, modelType, device, "test", "sentiment", sentimentLabels, testEval.sentiment);
        writePerClassRows(perClassCsv, runId, modelType, device, "test", "topic", topicLabels, testEval.topic);

        LinkedHashMap<String, String> summary = baseRow(runId, modelType, device, seed, batchSize, epochs, maxLen, alpha, beta);
        summary.put("selection", selection);
        summary.put("base_learning_rate", fmt(learningRate));
        summary.put("min_learning_rate", fmt(minLearningRate));
        summary.put("lr_schedule", lrSchedule);
        summary.put("lr_warmup_epochs", String.valueOf(lrWarmupEpochs));
        summary.put("early_stopping_patience", String.valueOf(earlyStoppingPatience));
        summary.put("early_stopping_min_delta", fmt(earlyStoppingMinDelta));
        summary.put("executed_epochs", String.valueOf(executedEpochs));
        summary.put("stopped_early", String.valueOf(stoppedEarly));
        summary.put("epochs_without_improve", String.valueOf(epochsWithoutImprove));
        summary.put("best_epoch", String.valueOf(bestEpoch));
        summary.put("best_dev_objective", fmt(bestDevObjective));
        summary.put("best_checkpoint", bestCheckpointPath.toAbsolutePath().toString());

        summary.put("dev_loss", fmt(bestDevEval.loss));
        summary.put("dev_sent_acc", fmt(bestDevEval.sentiment.accuracy));
        summary.put("dev_topic_acc", fmt(bestDevEval.topic.accuracy));
        summary.put("dev_sent_macro_f1", fmt(bestDevEval.sentiment.macroF1));
        summary.put("dev_topic_macro_f1", fmt(bestDevEval.topic.macroF1));
        summary.put("dev_joint_exact_match", fmt(bestDevEval.jointExactMatch));

        summary.put("test_loss", fmt(testEval.loss));
        summary.put("test_sent_acc", fmt(testEval.sentiment.accuracy));
        summary.put("test_topic_acc", fmt(testEval.topic.accuracy));
        summary.put("test_sent_macro_f1", fmt(testEval.sentiment.macroF1));
        summary.put("test_topic_macro_f1", fmt(testEval.topic.macroF1));
        summary.put("test_joint_exact_match", fmt(testEval.jointExactMatch));

        summary.put("dev_sent_precision", arrayToString(bestDevEval.sentiment.precision));
        summary.put("dev_sent_recall", arrayToString(bestDevEval.sentiment.recall));
        summary.put("dev_sent_f1", arrayToString(bestDevEval.sentiment.f1));
        summary.put("dev_topic_precision", arrayToString(bestDevEval.topic.precision));
        summary.put("dev_topic_recall", arrayToString(bestDevEval.topic.recall));
        summary.put("dev_topic_f1", arrayToString(bestDevEval.topic.f1));

        summary.put("test_sent_precision", arrayToString(testEval.sentiment.precision));
        summary.put("test_sent_recall", arrayToString(testEval.sentiment.recall));
        summary.put("test_sent_f1", arrayToString(testEval.sentiment.f1));
        summary.put("test_topic_precision", arrayToString(testEval.topic.precision));
        summary.put("test_topic_recall", arrayToString(testEval.topic.recall));
        summary.put("test_topic_f1", arrayToString(testEval.topic.f1));

        summary.put("inference_p50_ms", fmt(infer.p50Ms));
        summary.put("inference_p95_ms", fmt(infer.p95Ms));
        summary.put("inference_throughput_sps", fmt(infer.throughputSps));
        summary.put("inference_e2e_p50_ms", fmt(infer.e2eP50Ms));
        summary.put("inference_e2e_p95_ms", fmt(infer.e2eP95Ms));
        summary.put("inference_e2e_throughput_sps", fmt(infer.e2eThroughputSps));
        summary.put("inference_avg_batch_build_ms", fmt(infer.avgBatchBuildMs));
        summary.put("inference_avg_tensor_create_ms", fmt(infer.avgTensorCreateMs));
        summary.put("inference_avg_h2d_ms", fmt(infer.avgH2dMs));
        summary.put("inference_avg_forward_ms", fmt(infer.avgForwardMs));
        summary.put("inference_avg_end_to_end_ms", fmt(infer.avgEndToEndMs));

        summary.put("total_train_time_ms", String.valueOf(totalTrainMs));
        summary.put("final_model_path", finalModelPath.toAbsolutePath().toString());
        BenchmarkCsv.appendRow(summaryCsv, summary);

        System.out.printf(Locale.US,
                "Best epoch=%d | Dev sent_macro_f1=%.4f topic_macro_f1=%.4f joint=%.4f%n",
                bestEpoch,
                bestDevEval.sentiment.macroF1,
                bestDevEval.topic.macroF1,
                bestDevEval.jointExactMatch);

        System.out.printf(Locale.US,
                "Test | sent_acc=%.2f%% topic_acc=%.2f%% sent_macro_f1=%.4f topic_macro_f1=%.4f joint=%.4f%n",
                testEval.sentiment.accuracy * 100f,
                testEval.topic.accuracy * 100f,
                testEval.sentiment.macroF1,
                testEval.topic.macroF1,
                testEval.jointExactMatch);

        System.out.printf(Locale.US,
            "Inference | p50=%.4f ms p95=%.4f ms throughput=%.2f samples/s%n",
            infer.p50Ms,
            infer.p95Ms,
            infer.throughputSps);
        System.out.printf(Locale.US,
            "Inference breakdown | e2e_p50=%.4f ms e2e_p95=%.4f ms e2e_throughput=%.2f samples/s | batch=%.4f tensor=%.4f h2d=%.4f forward=%.4f ms%n",
            infer.e2eP50Ms,
            infer.e2eP95Ms,
            infer.e2eThroughputSps,
            infer.avgBatchBuildMs,
            infer.avgTensorCreateMs,
            infer.avgH2dMs,
            infer.avgForwardMs);

        System.out.println("Saved final model to: " + finalModelPath.toAbsolutePath());
        System.out.println("Benchmark artifacts: " + runDir.toAbsolutePath());

        if ("gpu".equals(device)) {
            GpuMemoryPool.destroy();
        }
    }

    private static Module createModel(String modelType, int vocabSize, int maxLen, int sentimentClasses, int topicClasses) {
        if ("lstm".equals(modelType)) {
            return new MultiTaskLSTMModel(vocabSize, 128, 256, sentimentClasses, topicClasses);
        }
        return new MultiTaskTransformerModel(vocabSize, 128, maxLen, 2, 4, 256, sentimentClasses, topicClasses, 0.1f);
    }

    private static Tensor[] forwardBoth(Module model, Tensor x) {
        if (model instanceof MultiTaskLSTMModel) {
            return ((MultiTaskLSTMModel) model).forwardBoth(x);
        }
        if (model instanceof MultiTaskTransformerModel) {
            return ((MultiTaskTransformerModel) model).forwardBoth(x);
        }
        throw new IllegalArgumentException("Unsupported model type: " + model.getClass().getName());
    }

    private static Data.Vocabulary buildVocabulary(
            List<UitVsfcLoader.Entry> entries,
            UitVsfcLoader.VietnameseTokenizer tokenizer) {
        Data.Vocabulary vocab = new Data.Vocabulary();
        for (UitVsfcLoader.Entry entry : entries) {
            List<String> tokens = tokenizer.tokenize(entry.text);
            for (String token : tokens) {
                vocab.addWord(token);
            }
        }
        return vocab;
    }

    private static BatchData createBatch(
            List<UitVsfcLoader.Entry> entries,
            int start,
            int end,
            UitVsfcLoader.VietnameseTokenizer tokenizer,
            Data.Vocabulary vocab,
            Map<String, float[]> tokenIdCache,
            int maxLen) {

        int bs = end - start;
        float[] xData = new float[bs * maxLen];
        int[] sentiment = new int[bs];
        int[] topic = new int[bs];

        for (int i = 0; i < bs; i++) {
            UitVsfcLoader.Entry e = entries.get(start + i);
            float[] encoded = tokenIdCache.computeIfAbsent(e.text, k -> encodeTextToIds(k, tokenizer, vocab, maxLen));
            System.arraycopy(encoded, 0, xData, i * maxLen, maxLen);
            sentiment[i] = e.sentimentId;
            topic[i] = e.topicId;
        }

        return new BatchData(xData, sentiment, topic, bs);
    }

    private static float[] encodeTextToIds(
            String text,
            UitVsfcLoader.VietnameseTokenizer tokenizer,
            Data.Vocabulary vocab,
            int maxLen) {
        float[] encoded = new float[maxLen];
        List<String> tokens = tokenizer.tokenize(text);
        int limit = Math.min(maxLen, tokens.size());
        for (int j = 0; j < limit; j++) {
            encoded[j] = vocab.getId(tokens.get(j));
        }
        return encoded;
    }

    private static EvalResult evaluate(
            Module model,
            List<UitVsfcLoader.Entry> entries,
            UitVsfcLoader.VietnameseTokenizer tokenizer,
            Data.Vocabulary vocab,
            Map<String, float[]> tokenIdCache,
            int maxLen,
            int batchSize,
            String device,
            float alpha,
            float beta,
            int sentimentClasses,
            int topicClasses) {

        boolean prevGrad = Torch.is_grad_enabled();
        Torch.set_grad_enabled(false);
        model.eval();

        float totalLoss = 0f;
        int numBatches = (entries.size() + batchSize - 1) / batchSize;

        int[][] sentConf = new int[sentimentClasses][sentimentClasses];
        int[][] topicConf = new int[topicClasses][topicClasses];

        int totalSamples = 0;
        int jointCorrect = 0;

        try {
            for (int b = 0; b < numBatches; b++) {
                try (MemoryScope scope = new MemoryScope()) {
                    int start = b * batchSize;
                    int end = Math.min((b + 1) * batchSize, entries.size());

                    BatchData batch = createBatch(entries, start, end, tokenizer, vocab, tokenIdCache, maxLen);
                    Tensor xBatch = Torch.tensor(batch.xData, batch.batchSize, maxLen);
                    if ("gpu".equals(device)) {
                        xBatch.toGPU();
                    }

                    Tensor[] logits = forwardBoth(model, xBatch);
                    Tensor lossSent = Functional.cross_entropy_tensor(logits[0], batch.sentimentLabels);
                    Tensor lossTopic = Functional.cross_entropy_tensor(logits[1], batch.topicLabels);
                    Tensor weightedSent = Torch.mul(lossSent, alpha);
                    Tensor weightedTopic = Torch.mul(lossTopic, beta);
                    Tensor loss = Torch.add(weightedSent, weightedTopic);
                    totalLoss += loss.data[0];

                    if (logits[0].isGPU()) {
                        logits[0].toCPU();
                    }
                    if (logits[1].isGPU()) {
                        logits[1].toCPU();
                    }

                    int sentClasses = logits[0].shape[1];
                    int topClasses = logits[1].shape[1];
                    for (int i = 0; i < batch.batchSize; i++) {
                        int sentPred = argmax(logits[0].data, i * sentClasses, sentClasses);
                        int topicPred = argmax(logits[1].data, i * topClasses, topClasses);
                        int sentTrue = batch.sentimentLabels[i];
                        int topicTrue = batch.topicLabels[i];

                        sentConf[sentTrue][sentPred]++;
                        topicConf[topicTrue][topicPred]++;

                        if (sentPred == sentTrue && topicPred == topicTrue) {
                            jointCorrect++;
                        }
                        totalSamples++;
                    }
                }
            }

            TaskMetrics sentMetrics = computeTaskMetrics(sentConf);
            TaskMetrics topicMetrics = computeTaskMetrics(topicConf);
            float jointExact = totalSamples > 0 ? ((float) jointCorrect / totalSamples) : 0f;
            float avgLoss = totalLoss / Math.max(1, numBatches);
            return new EvalResult(avgLoss, sentMetrics, topicMetrics, jointExact);
        } finally {
            Torch.set_grad_enabled(prevGrad);
            model.train();
        }
    }

    private static TaskMetrics computeTaskMetrics(int[][] confusion) {
        int classes = confusion.length;
        float[] precision = new float[classes];
        float[] recall = new float[classes];
        float[] f1 = new float[classes];

        int correct = 0;
        int total = 0;

        for (int i = 0; i < classes; i++) {
            int tp = confusion[i][i];
            int rowSum = 0;
            int colSum = 0;
            for (int j = 0; j < classes; j++) {
                rowSum += confusion[i][j];
                colSum += confusion[j][i];
            }

            precision[i] = safeDiv(tp, colSum);
            recall[i] = safeDiv(tp, rowSum);
            f1[i] = safeDiv(2f * precision[i] * recall[i], precision[i] + recall[i]);

            correct += tp;
            total += rowSum;
        }

        float macroF1 = 0f;
        for (float v : f1) {
            macroF1 += v;
        }
        macroF1 = classes > 0 ? (macroF1 / classes) : 0f;

        float accuracy = total > 0 ? ((float) correct / total) : 0f;
        return new TaskMetrics(accuracy, macroF1, precision, recall, f1, confusion);
    }

    private static float selectObjective(EvalResult devEval, String selection, float alpha, float beta) {
        if ("sentiment".equals(selection)) {
            return devEval.sentiment.macroF1;
        }
        if ("topic".equals(selection)) {
            return devEval.topic.macroF1;
        }
        float sum = alpha + beta;
        float wa = sum > 0f ? (alpha / sum) : 0.5f;
        float wb = sum > 0f ? (beta / sum) : 0.5f;
        return wa * devEval.sentiment.macroF1 + wb * devEval.topic.macroF1;
    }

    private static boolean isImproved(float current, float best, float minDelta) {
        if (!Float.isFinite(best)) {
            return true;
        }
        return current > (best + minDelta);
    }

    private static float computeLearningRate(
            float baseLr,
            float minLr,
            int epochIdx,
            int totalEpochs,
            int warmupEpochs,
            String schedule) {

        int epochNumber = epochIdx + 1;

        if (warmupEpochs > 0 && epochNumber <= warmupEpochs) {
            float progress = (float) epochNumber / (float) warmupEpochs;
            return Math.max(minLr, baseLr * progress);
        }

        if ("cosine".equals(schedule)) {
            int cosineTotal = Math.max(1, totalEpochs - warmupEpochs);
            int cosinePos = Math.max(0, epochNumber - warmupEpochs);
            float t = Math.min(1.0f, (float) cosinePos / (float) cosineTotal);
            float cosine = 0.5f * (1.0f + (float) Math.cos(Math.PI * t));
            return minLr + (baseLr - minLr) * cosine;
        }

        return baseLr;
    }

    private static int argmax(float[] data, int offset, int length) {
        int idx = 0;
        float best = data[offset];
        for (int i = 1; i < length; i++) {
            float v = data[offset + i];
            if (v > best) {
                best = v;
                idx = i;
            }
        }
        return idx;
    }

    private static float safeDiv(float num, float den) {
        return den == 0f ? 0f : (num / den);
    }

    private static InferenceResult benchmarkInference(
            Module model,
            List<UitVsfcLoader.Entry> entries,
            UitVsfcLoader.VietnameseTokenizer tokenizer,
            Data.Vocabulary vocab,
            Map<String, float[]> tokenIdCache,
            int maxLen,
            int batchSize,
            String device,
            int warmupSteps,
            int measureSteps,
            Path inferCsv,
            Path inferBreakdownCsv,
            String runId,
            long seed,
            int epochs,
            String modelType,
            float alpha,
            float beta) throws IOException {

        boolean prevGrad = Torch.is_grad_enabled();
        Torch.set_grad_enabled(false);
        model.eval();

        double[] latMs = new double[Math.max(1, measureSteps)];
        double[] e2eLatMs = new double[Math.max(1, measureSteps)];
        int measured = 0;
        int seen = 0;
        long totalSamples = 0;
        double totalLatencyMs = 0.0;
        double totalE2eLatencyMs = 0.0;
        double totalBatchBuildMs = 0.0;
        double totalTensorCreateMs = 0.0;
        double totalH2dMs = 0.0;
        double totalForwardMs = 0.0;
        int cursor = 0;

        try {
            while (seen < warmupSteps + measureSteps && !entries.isEmpty()) {
                long batchBuildStart = System.nanoTime();
                int end = Math.min(cursor + batchSize, entries.size());
                BatchData batch = createBatch(entries, cursor, end, tokenizer, vocab, tokenIdCache, maxLen);
                long batchBuildEnd = System.nanoTime();
                double batchBuildMs = (batchBuildEnd - batchBuildStart) / 1_000_000.0;

                try (MemoryScope scope = new MemoryScope()) {
                    long tensorCreateStart = System.nanoTime();
                    Tensor xBatch = Torch.tensor(batch.xData, batch.batchSize, maxLen);
                    long tensorCreateEnd = System.nanoTime();
                    double tensorCreateMs = (tensorCreateEnd - tensorCreateStart) / 1_000_000.0;

                    double h2dMs = 0.0;
                    if ("gpu".equals(device)) {
                        long h2dStart = System.nanoTime();
                        xBatch.toGPU();
                        cudaDeviceSynchronize();
                        long h2dEnd = System.nanoTime();
                        h2dMs = (h2dEnd - h2dStart) / 1_000_000.0;
                    }

                    long t0 = System.nanoTime();
                    forwardBoth(model, xBatch);
                    if ("gpu".equals(device)) {
                        cudaDeviceSynchronize();
                    }
                    long t1 = System.nanoTime();
                    double forwardMs = (t1 - t0) / 1_000_000.0;
                    double endToEndMs = batchBuildMs + tensorCreateMs + h2dMs + forwardMs;

                    if (seen >= warmupSteps) {
                        double ms = forwardMs;
                        latMs[measured] = ms;
                        e2eLatMs[measured] = endToEndMs;
                        measured++;
                        totalLatencyMs += ms;
                        totalE2eLatencyMs += endToEndMs;
                        totalBatchBuildMs += batchBuildMs;
                        totalTensorCreateMs += tensorCreateMs;
                        totalH2dMs += h2dMs;
                        totalForwardMs += forwardMs;
                        totalSamples += batch.batchSize;

                        LinkedHashMap<String, String> row = baseRow(
                                runId,
                                modelType,
                                device,
                                seed,
                                batchSize,
                                epochs,
                                maxLen,
                                alpha,
                                beta);
                        row.put("step", String.valueOf(measured));
                        row.put("batch_size", String.valueOf(batch.batchSize));
                        row.put("latency_ms", fmt(ms));
                        BenchmarkCsv.appendRow(inferCsv, row);

                        LinkedHashMap<String, String> breakdown = baseRow(
                                runId,
                                modelType,
                                device,
                                seed,
                                batchSize,
                                epochs,
                                maxLen,
                                alpha,
                                beta);
                        breakdown.put("step", String.valueOf(measured));
                        breakdown.put("batch_size", String.valueOf(batch.batchSize));
                        breakdown.put("batch_build_ms", fmt(batchBuildMs));
                        breakdown.put("tensor_create_ms", fmt(tensorCreateMs));
                        breakdown.put("h2d_ms", fmt(h2dMs));
                        breakdown.put("forward_ms", fmt(forwardMs));
                        breakdown.put("end_to_end_ms", fmt(endToEndMs));
                        BenchmarkCsv.appendRow(inferBreakdownCsv, breakdown);
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
            return new InferenceResult(
                    Double.NaN,
                    Double.NaN,
                    Double.NaN,
                    Double.NaN,
                    Double.NaN,
                    Double.NaN,
                    Double.NaN,
                    Double.NaN,
                    Double.NaN,
                    Double.NaN,
                    Double.NaN);
        }

        double[] effective = new double[measured];
        double[] effectiveE2e = new double[measured];
        System.arraycopy(latMs, 0, effective, 0, measured);
        System.arraycopy(e2eLatMs, 0, effectiveE2e, 0, measured);

        double p50 = BenchmarkStats.percentile(effective, 50.0);
        double p95 = BenchmarkStats.percentile(effective, 95.0);
        double e2eP50 = BenchmarkStats.percentile(effectiveE2e, 50.0);
        double e2eP95 = BenchmarkStats.percentile(effectiveE2e, 95.0);
        double throughput = totalLatencyMs > 0.0 ? (totalSamples * 1000.0 / totalLatencyMs) : Double.NaN;
        double e2eThroughput = totalE2eLatencyMs > 0.0 ? (totalSamples * 1000.0 / totalE2eLatencyMs) : Double.NaN;
        double avgBatchBuild = measured > 0 ? (totalBatchBuildMs / measured) : Double.NaN;
        double avgTensorCreate = measured > 0 ? (totalTensorCreateMs / measured) : Double.NaN;
        double avgH2d = measured > 0 ? (totalH2dMs / measured) : Double.NaN;
        double avgForward = measured > 0 ? (totalForwardMs / measured) : Double.NaN;
        double avgEndToEnd = measured > 0 ? (totalE2eLatencyMs / measured) : Double.NaN;
        return new InferenceResult(
            p50,
            p95,
            throughput,
            e2eP50,
            e2eP95,
            e2eThroughput,
            avgBatchBuild,
            avgTensorCreate,
            avgH2d,
            avgForward,
            avgEndToEnd);
    }

    private static void writeConfusionCsv(Path path, int[][] matrix, List<String> labels) throws IOException {
        Files.createDirectories(path.getParent());
        StringBuilder sb = new StringBuilder();
        sb.append("actual\\pred");
        for (String label : labels) {
            sb.append(',').append(escapeCsv(label));
        }
        sb.append('\n');

        for (int i = 0; i < matrix.length; i++) {
            String rowLabel = (i < labels.size()) ? labels.get(i) : String.valueOf(i);
            sb.append(escapeCsv(rowLabel));
            for (int j = 0; j < matrix[i].length; j++) {
                sb.append(',').append(matrix[i][j]);
            }
            sb.append('\n');
        }

        Files.writeString(path, sb.toString(), StandardCharsets.UTF_8);
    }

    private static void writePerClassRows(
            Path csvPath,
            String runId,
            String modelType,
            String device,
            String split,
            String task,
            List<String> labels,
            TaskMetrics metrics) throws IOException {

        for (int i = 0; i < labels.size(); i++) {
            LinkedHashMap<String, String> row = new LinkedHashMap<>();
            row.put("run_id", runId);
            row.put("task", TASK_NAME);
            row.put("model", modelType);
            row.put("device", device);
            row.put("split", split);
            row.put("head", task);
            row.put("class_id", String.valueOf(i));
            row.put("class_name", labels.get(i));
            row.put("precision", fmt(metrics.precision[i]));
            row.put("recall", fmt(metrics.recall[i]));
            row.put("f1", fmt(metrics.f1[i]));
            BenchmarkCsv.appendRow(csvPath, row);
        }
    }

    private static String arrayToString(float[] values) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < values.length; i++) {
            if (i > 0) {
                sb.append(';');
            }
            sb.append(fmt(values[i]));
        }
        return sb.toString();
    }

    private static String escapeCsv(String value) {
        if (value == null) {
            return "";
        }
        boolean needsQuotes = value.contains(",") || value.contains("\"") || value.contains("\n");
        if (!needsQuotes) {
            return value;
        }
        return '"' + value.replace("\"", "\"\"") + '"';
    }

    private static String getPositionalArg(String[] args, int idx, String defaultValue) {
        if (idx < args.length && !args[idx].startsWith("--")) {
            return args[idx];
        }
        return defaultValue;
    }

    private static int parsePositionalInt(String[] args, int idx, int defaultValue) {
        String value = getPositionalArg(args, idx, null);
        if (value == null) {
            return defaultValue;
        }
        try {
            return Integer.parseInt(value);
        } catch (NumberFormatException ex) {
            return defaultValue;
        }
    }

    private static long parsePositionalLong(String[] args, int idx, long defaultValue) {
        String value = getPositionalArg(args, idx, null);
        if (value == null) {
            return defaultValue;
        }
        try {
            return Long.parseLong(value);
        } catch (NumberFormatException ex) {
            return defaultValue;
        }
    }

    private static float parseFloatArg(Map<String, String> cli, String key, float defaultValue) {
        String value = cli.get(key);
        if (value == null) {
            return defaultValue;
        }
        try {
            return Float.parseFloat(value);
        } catch (NumberFormatException ex) {
            throw new IllegalArgumentException("Invalid float for --" + key + ": " + value);
        }
    }

    private static String timestamp() {
        return new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
    }

    private static String fmt(double v) {
        return String.format(Locale.US, "%.6f", v);
    }

    private static LinkedHashMap<String, String> baseRow(
            String runId,
            String modelType,
            String device,
            long seed,
            int batchSize,
            int epochs,
            int maxLen,
            float alpha,
            float beta) {
        LinkedHashMap<String, String> row = new LinkedHashMap<>();
        row.put("run_id", runId);
        row.put("timestamp", timestamp());
        row.put("framework", "JavaTorch");
        row.put("task", TASK_NAME);
        row.put("model", modelType);
        row.put("device", device);
        row.put("seed", String.valueOf(seed));
        row.put("train_batch_size", String.valueOf(batchSize));
        row.put("mixed_precision", "false");
        row.put("batch_size", String.valueOf(batchSize));
        row.put("epochs", String.valueOf(epochs));
        row.put("max_len", String.valueOf(maxLen));
        row.put("alpha", fmt(alpha));
        row.put("beta", fmt(beta));
        return row;
    }

    private static final class InferenceResult {
        final double p50Ms;
        final double p95Ms;
        final double throughputSps;
        final double e2eP50Ms;
        final double e2eP95Ms;
        final double e2eThroughputSps;
        final double avgBatchBuildMs;
        final double avgTensorCreateMs;
        final double avgH2dMs;
        final double avgForwardMs;
        final double avgEndToEndMs;

        InferenceResult(
                double p50Ms,
                double p95Ms,
                double throughputSps,
                double e2eP50Ms,
                double e2eP95Ms,
                double e2eThroughputSps,
                double avgBatchBuildMs,
                double avgTensorCreateMs,
                double avgH2dMs,
                double avgForwardMs,
                double avgEndToEndMs) {
            this.p50Ms = p50Ms;
            this.p95Ms = p95Ms;
            this.throughputSps = throughputSps;
            this.e2eP50Ms = e2eP50Ms;
            this.e2eP95Ms = e2eP95Ms;
            this.e2eThroughputSps = e2eThroughputSps;
            this.avgBatchBuildMs = avgBatchBuildMs;
            this.avgTensorCreateMs = avgTensorCreateMs;
            this.avgH2dMs = avgH2dMs;
            this.avgForwardMs = avgForwardMs;
            this.avgEndToEndMs = avgEndToEndMs;
        }
    }
}
