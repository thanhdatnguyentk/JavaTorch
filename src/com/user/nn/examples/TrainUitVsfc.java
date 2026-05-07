package com.user.nn.examples;

import com.user.nn.core.*;
import com.user.nn.core.Module;
import com.user.nn.dataloaders.UitVsfcLoader;
import com.user.nn.dataloaders.Data;
import com.user.nn.layers.Embedding;
import com.user.nn.models.MultiTaskLSTMModel;
import com.user.nn.models.MultiTaskTransformerModel;
import com.user.nn.optim.Optim;
import com.user.nn.utils.dashboard.DashboardServer;
import com.user.nn.utils.dashboard.DashboardIntegrationHelper;
import com.user.nn.utils.visualization.TrainingHistory;
import java.util.LinkedHashMap;

import java.util.*;

public class TrainUitVsfc {

    public static void main(String[] args) throws Exception {
        System.out.println("\n╔════════════════════════════════════════════════════════════╗");
        System.out.println("║    UIT-VSFC Multi-task Sentiment & Topic Classification    ║");
        System.out.println("╚════════════════════════════════════════════════════════════╝\n");

        // 1. Configurable parameters (could be parsed from args, but using defaults here for simplicity)
        String modelType = "lstm"; // "lstm" or "transformer"
        String device = "gpu";     // "cpu" or "gpu"
        int epochs = SmokeTest.getEpochs(100);
        int batchSize = 64;
        int maxLen = 64;
        float lr = 0.002f;
        int patience = 5;

        // Parse simple args if provided
        for (int i = 0; i < args.length; i++) {
            if (args[i].equals("--model") && i + 1 < args.length) modelType = args[i + 1].toLowerCase();
            if (args[i].equals("--device") && i + 1 < args.length) device = args[i + 1].toLowerCase();
            if (args[i].equals("--epochs") && i + 1 < args.length) epochs = Integer.parseInt(args[i + 1]);
            if (args[i].equals("--patience") && i + 1 < args.length) patience = Integer.parseInt(args[i + 1]);
        }

        System.out.printf("Configuration: Model=%s, Device=%s, Epochs=%d, BatchSize=%d, Patience=%d%n%n", 
                          modelType.toUpperCase(), device.toUpperCase(), epochs, batchSize, patience);

        // 2. Load Dataset
        System.out.println("Loading UIT-VSFC dataset...");
        UitVsfcLoader.DatasetSplits splits = UitVsfcLoader.load(UitVsfcLoader.DEFAULT_DATA_DIR);
        List<String> sentimentLabels = splits.sentimentEncoder.labels();
        List<String> topicLabels = splits.topicEncoder.labels();
        System.out.printf("Loaded: Train=%d, Dev=%d, Test=%d samples%n", 
                          splits.train.size(), splits.dev.size(), splits.test.size());
        System.out.printf("Classes: Sentiment=%d, Topic=%d%n%n", 
                          sentimentLabels.size(), topicLabels.size());

        // 3. Build Vocabulary
        UitVsfcLoader.VietnameseTokenizer tokenizer = new UitVsfcLoader.VietnameseTokenizer();
        Data.Vocabulary vocab = new Data.Vocabulary();
        for (UitVsfcLoader.Entry entry : splits.train) {
            for (String token : tokenizer.tokenize(entry.text)) vocab.addWord(token);
        }
        System.out.println("Vocabulary size: " + vocab.size());

        // Cache token IDs to speed up batch creation
        Map<String, float[]> tokenIdCache = new HashMap<>();

        // 4. Model Setup
        com.user.nn.core.Module model;
        if ("transformer".equals(modelType)) {
            model = new MultiTaskTransformerModel(vocab.size(), 128, maxLen, 2, 4, 256, sentimentLabels.size(), topicLabels.size(), 0.1f);
        } else {
            model = new MultiTaskLSTMModel(vocab.size(), 128, 256, sentimentLabels.size(), topicLabels.size());
        }

        for (Parameter p : model.parameters()) {
            p.getTensor().requires_grad = true;
        }

        if ("gpu".equals(device)) {
            GpuMemoryPool.autoInit(model);
            model.toGPU();
            System.out.println("Model moved to GPU.");
        }

        // 5. Optimizer
        Optim.Adam optimizer = new Optim.Adam(model.parameters(), lr);

        // 6. Dashboard Integration
        TrainingHistory history = new TrainingHistory();
        DashboardServer dashboard = new DashboardServer(7070, history).start();
        dashboard.setTaskType("nlp");
        dashboard.setModelInfo("UIT-VSFC " + modelType.toUpperCase(), epochs);
        try {
            String[] sentLabelsArr = sentimentLabels.toArray(new String[0]);
            com.user.nn.predict.TextPredictor predictor = com.user.nn.predict.TextPredictor
                    .forSentiment(model, vocab, maxLen, sentLabelsArr)
                    .setTokenizer(tokenizer::tokenize);
            DashboardIntegrationHelper.setupTextPredictorHandler(dashboard, "sentiment", predictor);
        } catch (Exception e) {}

        // 7. Training Loop
        System.out.println("\nStarting training...");
        List<UitVsfcLoader.Entry> shuffledTrain = new ArrayList<>(splits.train);
        
        float bestDevMacroF1 = -1.0f;
        int epochsWithoutImprovement = 0;

        for (int epoch = 0; epoch < epochs; epoch++) {
            Collections.shuffle(shuffledTrain, new Random(42L + epoch));
            model.train();
            
            float totalLoss = 0f;
            int numBatches = (shuffledTrain.size() + batchSize - 1) / batchSize;
            int correctSent = 0, correctTopic = 0;
            int totalSamples = 0;

            for (int b = 0; b < numBatches; b++) {
                try (MemoryScope scope = new MemoryScope()) {
                    int start = b * batchSize;
                    int end = Math.min((b + 1) * batchSize, shuffledTrain.size());
                    int currentBs = end - start;

                    // Build Batch
                    float[] xData = new float[currentBs * maxLen];
                    int[] sentLabels = new int[currentBs];
                    int[] topLabels = new int[currentBs];

                    for (int i = 0; i < currentBs; i++) {
                        UitVsfcLoader.Entry e = shuffledTrain.get(start + i);
                        float[] encoded = tokenIdCache.computeIfAbsent(e.text, k -> encodeText(k, tokenizer, vocab, maxLen));
                        System.arraycopy(encoded, 0, xData, i * maxLen, maxLen);
                        sentLabels[i] = e.sentimentId;
                        topLabels[i] = e.topicId;
                    }

                    Tensor xBatch = Torch.tensor(xData, currentBs, maxLen);
                    if ("gpu".equals(device)) xBatch.toGPU();

                    optimizer.zero_grad();
                    
                    // Forward Both Heads
                    Tensor[] logits;
                    if (model instanceof MultiTaskLSTMModel) logits = ((MultiTaskLSTMModel) model).forwardBoth(xBatch);
                    else logits = ((MultiTaskTransformerModel) model).forwardBoth(xBatch);

                    // Multi-task Loss (equal weights)
                    Tensor lossSent = Functional.cross_entropy_tensor(logits[0], sentLabels);
                    Tensor lossTopic = Functional.cross_entropy_tensor(logits[1], topLabels);
                    Tensor loss = Torch.add(lossSent, lossTopic);

                    loss.backward();
                    optimizer.step();

                    totalLoss += loss.data[0];
                    
                    if ((b + 1) % 10 == 0) {
                        System.out.printf("  Batch %d/%d - loss: %.4f%n", b + 1, numBatches, loss.data[0]);
                    }

                    // Track accuracy
                    if (logits[0].isGPU()) logits[0].toCPU();
                    if (logits[1].isGPU()) logits[1].toCPU();
                    
                    int sentClasses = logits[0].shape[1];
                    int topClasses = logits[1].shape[1];
                    
                    for (int i = 0; i < currentBs; i++) {
                        int pSent = argmax(logits[0].data, i * sentClasses, sentClasses);
                        int pTop = argmax(logits[1].data, i * topClasses, topClasses);
                        if (pSent == sentLabels[i]) correctSent++;
                        if (pTop == topLabels[i]) correctTopic++;
                    }
                    totalSamples += currentBs;

                    // Dashboard Updates
                    if ((b + 1) % 50 == 0) {
                        try {
                            float currentSentAcc = (float) correctSent / totalSamples;
                            float currentTopAcc = (float) correctTopic / totalSamples;
                            Map<String, Float> dashMetrics = new HashMap<>();
                            dashMetrics.put("loss", loss.data[0]);
                            dashMetrics.put("sent_acc", currentSentAcc);
                            dashMetrics.put("topic_acc", currentTopAcc);
                            
                            UitVsfcLoader.Entry sampleEntry = shuffledTrain.get(start);
                            
                            // 1. Calculate Softmax Confidence for Sentiment
                            int numSentClasses = sentimentLabels.size();
                            float[] sentLogits = new float[numSentClasses];
                            System.arraycopy(logits[0].data, 0, sentLogits, 0, numSentClasses);
                            float[] probs = softmax(sentLogits, 0, numSentClasses);
                            int pSent = 0;
                            float bestProb = 0;
                            for(int i=0; i<numSentClasses; i++) {
                                if(probs[i] > bestProb) {
                                    bestProb = probs[i];
                                    pSent = i;
                                }
                            }
                            
                            // 2. Real Token Importance from Model (Using the new optimized method)
                            Map<String, Float> tokenWeights = new HashMap<>();
                            Tensor xSample = Torch.narrow(xBatch, 0, 0, 1);
                            if (model instanceof MultiTaskLSTMModel) {
                                tokenWeights = ((MultiTaskLSTMModel) model).getEmbeddingNorms(xSample, tokenizer, vocab);
                            } else if (model instanceof MultiTaskTransformerModel) {
                                tokenWeights = ((MultiTaskTransformerModel) model).getEmbeddingNorms(xSample, tokenizer, vocab);
                            }
                            
                            // 3. Metrics
                            Map<String, Float> f1 = Map.of("Sentiment", currentSentAcc, "Topic", currentTopAcc);
                            
                            DashboardIntegrationHelper.broadcastNLPDetailed(
                                dashboard, epoch + 1, dashMetrics, sampleEntry.text,
                                sentimentLabels.get(pSent), bestProb,
                                f1, f1, f1, tokenWeights
                            );
                        } catch (Exception ex) {
                            ex.printStackTrace();
                        }
                    }
                    
                    while (dashboard.isTrainingPaused()) {
                        try { Thread.sleep(200); } catch (InterruptedException ie) { break; }
                    }
                }
            }

            // End of Epoch Dev Evaluation
            float trainLoss = totalLoss / numBatches;
            float trainSentAcc = (float) correctSent / totalSamples;
            float trainTopicAcc = (float) correctTopic / totalSamples;

            EvalMetrics devMetrics = evaluate(model, splits.dev, tokenizer, vocab, tokenIdCache, maxLen, batchSize, device, sentimentLabels.size(), topicLabels.size());

            System.out.printf("Epoch %d/%d - loss: %.4f | Train Acc (Sent: %.2f%%, Topic: %.2f%%) | Dev Acc (Sent: %.2f%%, Topic: %.2f%%) | Dev MacroF1 (Sent: %.4f)%n",
                epoch + 1, epochs, trainLoss, trainSentAcc * 100, trainTopicAcc * 100, 
                devMetrics.sentAcc * 100, devMetrics.topicAcc * 100, devMetrics.sentMacroF1);

            dashboard.setCurrentEpoch(epoch + 1);
            try {
                Map<String, Float> metrics = new HashMap<>();
                metrics.put("loss", trainLoss);
                metrics.put("train_sent_acc", trainSentAcc);
                metrics.put("dev_sent_acc", devMetrics.sentAcc);
                history.record(epoch + 1, metrics);
                dashboard.broadcastMetrics(epoch + 1, metrics);
            } catch (Exception ex) {}

            // Early stopping check
            if (devMetrics.sentMacroF1 > bestDevMacroF1) {
                bestDevMacroF1 = devMetrics.sentMacroF1;
                epochsWithoutImprovement = 0;
                System.out.printf("  --> Best Dev MacroF1 updated to %.4f%n", bestDevMacroF1);
                
                String bestModelFile = "uit_vsfc_" + modelType + "_best.bin";
                model.save(bestModelFile);
            } else {
                epochsWithoutImprovement++;
                System.out.printf("  --> No improvement for %d epoch(s). Best was %.4f%n", epochsWithoutImprovement, bestDevMacroF1);
                if (epochsWithoutImprovement >= patience) {
                    System.out.println("  --> Early stopping triggered after " + (epoch + 1) + " epochs.");
                    break;
                }
            }
        }

        // 8. Test Evaluation
        System.out.println("\nEvaluating on Test Set...");
        EvalMetrics testMetrics = evaluate(model, splits.test, tokenizer, vocab, tokenIdCache, maxLen, batchSize, device, sentimentLabels.size(), topicLabels.size());
        System.out.printf("Test Results: Sentiment Acc: %.2f%%, Topic Acc: %.2f%%%n", testMetrics.sentAcc * 100, testMetrics.topicAcc * 100);

        // 9. Prediction Demo
        System.out.println("\n╔══════════════════════════════════════════╗");
        System.out.println("║          PREDICTION DEMONSTRATION        ║");
        System.out.println("╚══════════════════════════════════════════╝\n");

        String[] sampleTexts = {
            "thầy dạy rất nhiệt tình và dễ hiểu",
            "phòng máy quá cũ, không đáp ứng nhu cầu",
            "chương trình học phù hợp và bổ ích",
            "giảng viên không nhiệt tình, hay đi trễ",
            "cần nâng cấp hệ thống wifi trong trường",
            "em rất hài lòng về môn học này"
        };

        boolean prevGrad = Torch.is_grad_enabled();
        Torch.set_grad_enabled(false);
        model.eval();

        for (String text : sampleTexts) {
            float[] encoded = encodeText(text, tokenizer, vocab, maxLen);
            Tensor x = Torch.tensor(encoded, 1, maxLen);
            if ("gpu".equals(device)) x.toGPU();

            Tensor[] logits;
            if (model instanceof MultiTaskLSTMModel) logits = ((MultiTaskLSTMModel) model).forwardBoth(x);
            else logits = ((MultiTaskTransformerModel) model).forwardBoth(x);

            if (logits[0].isGPU()) logits[0].toCPU();
            if (logits[1].isGPU()) logits[1].toCPU();

            int pSent = argmax(logits[0].data, 0, sentimentLabels.size());
            int pTop = argmax(logits[1].data, 0, topicLabels.size());

            System.out.printf("  \"%s\"%n", text);
            System.out.printf("    → Sentiment: %s%n", sentimentLabels.get(pSent));
            System.out.printf("    → Topic:     %s%n%n", topicLabels.get(pTop));
        }
        Torch.set_grad_enabled(prevGrad);

        // 10. Save Model
        String modelFile = "uit_vsfc_" + modelType + ".bin";
        model.save(modelFile);
        System.out.println("Saved model to " + modelFile);

        if ("gpu".equals(device)) GpuMemoryPool.destroy();
        System.out.println("Training Complete!");
    }

    // --- Helper Methods ---

    private static float[] encodeText(String text, UitVsfcLoader.VietnameseTokenizer tokenizer, Data.Vocabulary vocab, int maxLen) {
        float[] encoded = new float[maxLen];
        List<String> tokens = tokenizer.tokenize(text);
        int limit = Math.min(maxLen, tokens.size());
        for (int j = 0; j < limit; j++) encoded[j] = vocab.getId(tokens.get(j));
        return encoded;
    }

    private static int argmax(float[] data, int offset, int length) {
        int idx = 0;
        float best = data[offset];
        for (int i = 1; i < length; i++) {
            if (data[offset + i] > best) {
                best = data[offset + i];
                idx = i;
            }
        }
        return idx;
    }

    /**
     * Proper softmax over a slice of logits.
     */
    private static float[] softmax(float[] data, int offset, int numClasses) {
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < numClasses; i++) {
            if (data[offset + i] > max) max = data[offset + i];
        }
        float sum = 0;
        float[] probs = new float[numClasses];
        for (int i = 0; i < numClasses; i++) {
            probs[i] = (float) Math.exp(data[offset + i] - max);
            sum += probs[i];
        }
        for (int i = 0; i < numClasses; i++) probs[i] /= sum;
        return probs;
    }

    private static class EvalMetrics {
        float sentAcc, topicAcc, sentMacroF1;
    }

    private static EvalMetrics evaluate(com.user.nn.core.Module model, List<UitVsfcLoader.Entry> entries, 
                                       UitVsfcLoader.VietnameseTokenizer tokenizer, Data.Vocabulary vocab, 
                                       Map<String, float[]> cache, int maxLen, int batchSize, String device,
                                       int sentClasses, int topClasses) {
        boolean prevGrad = Torch.is_grad_enabled();
        Torch.set_grad_enabled(false);
        model.eval();

        int[][] sentConf = new int[sentClasses][sentClasses];
        int correctSent = 0, correctTopic = 0;
        int numBatches = (entries.size() + batchSize - 1) / batchSize;

        for (int b = 0; b < numBatches; b++) {
            try (MemoryScope scope = new MemoryScope()) {
                int start = b * batchSize;
                int end = Math.min((b + 1) * batchSize, entries.size());
                int currentBs = end - start;

                float[] xData = new float[currentBs * maxLen];
                for (int i = 0; i < currentBs; i++) {
                    UitVsfcLoader.Entry e = entries.get(start + i);
                    float[] encoded = cache.computeIfAbsent(e.text, k -> encodeText(k, tokenizer, vocab, maxLen));
                    System.arraycopy(encoded, 0, xData, i * maxLen, maxLen);
                }

                Tensor xBatch = Torch.tensor(xData, currentBs, maxLen);
                if ("gpu".equals(device)) xBatch.toGPU();

                Tensor[] logits;
                if (model instanceof MultiTaskLSTMModel) logits = ((MultiTaskLSTMModel) model).forwardBoth(xBatch);
                else logits = ((MultiTaskTransformerModel) model).forwardBoth(xBatch);

                if (logits[0].isGPU()) logits[0].toCPU();
                if (logits[1].isGPU()) logits[1].toCPU();

                for (int i = 0; i < currentBs; i++) {
                    UitVsfcLoader.Entry e = entries.get(start + i);
                    int pSent = argmax(logits[0].data, i * sentClasses, sentClasses);
                    int pTop = argmax(logits[1].data, i * topClasses, topClasses);
                    
                    if (pSent == e.sentimentId) correctSent++;
                    if (pTop == e.topicId) correctTopic++;
                    sentConf[e.sentimentId][pSent]++;
                }
            }
        }

        Torch.set_grad_enabled(prevGrad);
        model.train();

        EvalMetrics m = new EvalMetrics();
        m.sentAcc = (float) correctSent / entries.size();
        m.topicAcc = (float) correctTopic / entries.size();

        // Calculate Macro F1 for Sentiment
        float macroF1 = 0f;
        for (int i = 0; i < sentClasses; i++) {
            int tp = sentConf[i][i];
            int rowSum = 0, colSum = 0;
            for (int j = 0; j < sentClasses; j++) {
                rowSum += sentConf[i][j];
                colSum += sentConf[j][i];
            }
            float precision = colSum == 0 ? 0 : (float) tp / colSum;
            float recall = rowSum == 0 ? 0 : (float) tp / rowSum;
            float f1 = (precision + recall) == 0 ? 0 : 2 * precision * recall / (precision + recall);
            macroF1 += f1;
        }
        m.sentMacroF1 = sentClasses > 0 ? macroF1 / sentClasses : 0;

        return m;
    }
}
