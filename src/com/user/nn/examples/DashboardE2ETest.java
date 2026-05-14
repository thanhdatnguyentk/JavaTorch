package com.user.nn.examples;

import com.user.nn.utils.dashboard.DashboardServer;
import com.user.nn.utils.dashboard.DashboardIntegrationHelper;
import com.user.nn.utils.visualization.TrainingHistory;

import java.util.*;

/**
 * E2E Dashboard Test — Simulates ALL task types (Classification, GAN, NLP, Detection)
 * to verify every frontend feature works correctly without requiring GPU training.
 *
 * Usage: Run this, then open http://localhost:7070 in browser.
 * The dashboard will cycle through task types, broadcasting synthetic data.
 */
public class DashboardE2ETest {

    private static final String[] CIFAR_LABELS = {
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    };
    private static final Random rand = new Random(42);

    public static void main(String[] args) throws InterruptedException {
        int port = Integer.parseInt(System.getProperty("dashPort", "7071"));
        TrainingHistory history = new TrainingHistory();
        DashboardServer dashboard = new DashboardServer(port, history).start();
        System.out.println("╔══════════════════════════════════════════════════╗");
        System.out.println("║  Dashboard E2E Test — http://localhost:" + port + "  ║");
        System.out.println("╚══════════════════════════════════════════════════╝\n");

        // Register inference handlers
        registerHandlers(dashboard);

        // Phase 1: Classification (10 epochs)
        System.out.println(">>> Phase 1: Classification (10 epochs)");
        testClassification(dashboard, history, 10);

        Thread.sleep(2000);

        // Phase 2: GAN (10 epochs)
        System.out.println("\n>>> Phase 2: GAN (10 epochs)");
        testGAN(dashboard, history, 10);

        Thread.sleep(2000);

        // Phase 3: NLP (10 epochs)
        System.out.println("\n>>> Phase 3: NLP (10 epochs)");
        testNLP(dashboard, history, 10);

        Thread.sleep(2000);

        // Phase 4: Detection (5 epochs)
        System.out.println("\n>>> Phase 4: Detection (5 epochs)");
        testDetection(dashboard, history, 5);

        // Phase 5: System Alerts
        System.out.println("\n>>> Phase 5: System Alerts");
        DashboardIntegrationHelper.broadcastSystemAlert(dashboard, "info", "Training Complete",
            "All 4 task types simulated successfully.");
        DashboardIntegrationHelper.broadcastSystemAlert(dashboard, "warning", "VRAM Usage",
            "Peak VRAM usage was 3.2GB / 4GB (80%).");

        System.out.println("\n✓ All phases complete. Dashboard still live for inspection.");
        System.out.println("  Press Ctrl+C to stop.\n");

        // Keep alive for manual inspection
        Thread.sleep(600_000); // 10 minutes
        dashboard.stop();
    }

    private static void registerHandlers(DashboardServer dashboard) {
        // Sentiment text predictor (mock)
        dashboard.registerHandler("sentiment", (fileName, fileStream, text) -> {
            Map<String, Object> result = new HashMap<>();
            boolean positive = text != null && (text.contains("good") || text.contains("great") ||
                text.contains("amazing") || text.contains("love") || text.contains("excellent"));
            result.put("label", positive ? "POSITIVE" : "NEGATIVE");
            result.put("confidence", 0.75 + rand.nextDouble() * 0.2);
            result.put("class_index", positive ? 1 : 0);
            result.put("topK", List.of(
                Map.of("label", positive ? "POSITIVE" : "NEGATIVE", "confidence", 0.75 + rand.nextDouble() * 0.2),
                Map.of("label", positive ? "NEGATIVE" : "POSITIVE", "confidence", 0.1 + rand.nextDouble() * 0.15)
            ));
            return result;
        });

        // Image classifier (mock)
        dashboard.registerHandler("classify_image", (fileName, fileStream, text) -> {
            int classIdx = rand.nextInt(10);
            Map<String, Object> result = new HashMap<>();
            result.put("label", CIFAR_LABELS[classIdx]);
            result.put("class_index", classIdx);
            result.put("confidence", 0.6 + rand.nextDouble() * 0.35);
            List<Map<String, Object>> topK = new ArrayList<>();
            for (int k = 0; k < 3; k++) {
                int idx = (classIdx + k) % 10;
                topK.add(Map.of("label", CIFAR_LABELS[idx], "confidence", Math.max(0.05, 0.9 - k * 0.3 + rand.nextDouble() * 0.1)));
            }
            result.put("topK", topK);
            return result;
        });

        // GAN latent explorer (mock)
        dashboard.registerHandler("gan_latent", (fileName, fileStream, text) -> {
            // Generate a simple gradient image as mock
            float[] pixels = new float[784];
            for (int i = 0; i < 784; i++) {
                pixels[i] = (float) (Math.sin(i * 0.03) * 0.5 + rand.nextFloat() * 0.5);
            }
            String base64 = DashboardIntegrationHelper.encodeGeneratorOutput(pixels, 1, 28, 28);
            return Map.of("image", base64);
        });

        System.out.println("  ✓ Registered 3 inference handlers: sentiment, classify_image, gan_latent");
    }

    // ==================== CLASSIFICATION ====================

    private static void testClassification(DashboardServer dashboard, TrainingHistory history, int epochs)
            throws InterruptedException {
        dashboard.setTaskType("classification");
        dashboard.setModelInfo("CNN-CIFAR10 (E2E Test)", epochs);

        for (int epoch = 1; epoch <= epochs; epoch++) {
            float trainLoss = (float) (2.5 * Math.exp(-epoch * 0.25) + rand.nextFloat() * 0.05);
            float trainAcc = (float) Math.min(0.99, 0.3 + epoch * 0.065 + rand.nextFloat() * 0.02);
            float testAcc = (float) Math.min(0.95, trainAcc - 0.05 + rand.nextFloat() * 0.03);

            // Confusion matrix (10x10)
            int[][] cm = new int[10][10];
            for (int i = 0; i < 200; i++) {
                int actual = rand.nextInt(10);
                int predicted = (rand.nextFloat() < testAcc) ? actual : rand.nextInt(10);
                cm[actual][predicted]++;
            }

            // Live predictions (9 samples)
            List<Map<String, Object>> livePreds = new ArrayList<>();
            for (int s = 0; s < 9; s++) {
                int actual = rand.nextInt(10);
                int pred = (rand.nextFloat() < testAcc) ? actual : rand.nextInt(10);
                float[] fakePixels = new float[3072]; // 3x32x32
                for (int p = 0; p < 3072; p++) fakePixels[p] = rand.nextFloat();
                String img64 = DashboardIntegrationHelper.encodePixelsToBase64(fakePixels, 3, 32, 32);
                List<Map<String, Object>> topK = new ArrayList<>();
                topK.add(DashboardIntegrationHelper.buildTopKEntry(CIFAR_LABELS[pred], 0.6f + rand.nextFloat() * 0.35f));
                topK.add(DashboardIntegrationHelper.buildTopKEntry(CIFAR_LABELS[(pred + 1) % 10], 0.1f + rand.nextFloat() * 0.15f));
                topK.add(DashboardIntegrationHelper.buildTopKEntry(CIFAR_LABELS[(pred + 2) % 10], 0.02f + rand.nextFloat() * 0.08f));
                livePreds.add(DashboardIntegrationHelper.buildLivePrediction(
                    img64, CIFAR_LABELS[pred], CIFAR_LABELS[actual], pred == actual, topK));
            }

            Map<String, Float> metrics = new HashMap<>();
            metrics.put("loss", trainLoss);
            metrics.put("train_loss", trainLoss);
            metrics.put("train_acc", trainAcc);
            metrics.put("test_acc", testAcc);
            history.record(epoch, metrics);
            dashboard.setCurrentEpoch(epoch);

            DashboardIntegrationHelper.broadcastClassificationDetailed(
                dashboard, epoch, metrics, cm, CIFAR_LABELS, livePreds);

            System.out.printf("  Epoch %d/%d  loss=%.4f  train_acc=%.4f  test_acc=%.4f%n",
                epoch, epochs, trainLoss, trainAcc, testAcc);

            // Pause support
            while (dashboard.isTrainingPaused()) {
                Thread.sleep(200);
            }

            Thread.sleep(1200); // Simulate training time
        }
    }

    // ==================== GAN ====================

    private static void testGAN(DashboardServer dashboard, TrainingHistory history, int epochs)
            throws InterruptedException {
        dashboard.setTaskType("gan");
        dashboard.setModelInfo("GAN-MNIST (E2E Test)", epochs);

        for (int epoch = 1; epoch <= epochs; epoch++) {
            float gLoss = (float) (1.5 * Math.exp(-epoch * 0.1) + 0.5 + rand.nextFloat() * 0.2);
            float dLoss = (float) (0.7 + Math.sin(epoch * 0.5) * 0.3 + rand.nextFloat() * 0.1);

            // Generate 16 fake "images" (28x28 grayscale patterns)
            List<float[]> samples = new ArrayList<>();
            for (int s = 0; s < 16; s++) {
                float[] pixels = new float[784];
                for (int i = 0; i < 784; i++) {
                    int y = i / 28, x = i % 28;
                    pixels[i] = (float) (Math.sin((x + epoch * 2) * 0.2 + s * 0.5) *
                                         Math.cos((y + epoch) * 0.15) * 0.8 + rand.nextFloat() * 0.2 - 0.1);
                }
                samples.add(pixels);
            }

            Map<String, Float> metrics = new HashMap<>();
            metrics.put("g_loss", gLoss);
            metrics.put("d_loss", dLoss);
            metrics.put("acc", (float) Math.min(0.95, 0.1 + epoch * 0.08 + rand.nextFloat() * 0.02));
            history.record(epoch, metrics);
            dashboard.setCurrentEpoch(epoch);

            DashboardIntegrationHelper.broadcastGANDetailed(
                dashboard, epoch, gLoss, dLoss, samples, 1, 28, 28);

            System.out.printf("  Epoch %d/%d  g_loss=%.4f  d_loss=%.4f%n", epoch, epochs, gLoss, dLoss);

            while (dashboard.isTrainingPaused()) { Thread.sleep(200); }
            Thread.sleep(1200);
        }
    }

    // ==================== NLP ====================

    private static void testNLP(DashboardServer dashboard, TrainingHistory history, int epochs)
            throws InterruptedException {
        dashboard.setTaskType("nlp");
        dashboard.setModelInfo("LSTM-Sentiment (E2E Test)", epochs);

        String[] sampleTexts = {
            "This movie is absolutely amazing and wonderful!",
            "Terrible film, waste of time and money.",
            "The acting was decent but the plot was boring.",
            "One of the best movies I have ever seen!",
            "I would not recommend this to anyone.",
            "A masterpiece of modern cinema.",
            "The effects were great but story was weak.",
            "Loved every minute of it.",
            "Could not even finish watching it.",
            "Pretty average, nothing special."
        };
        String[] labels = {"POSITIVE", "NEGATIVE", "NEGATIVE", "POSITIVE", "NEGATIVE",
                          "POSITIVE", "NEGATIVE", "POSITIVE", "NEGATIVE", "NEUTRAL"};

        for (int epoch = 1; epoch <= epochs; epoch++) {
            float loss = (float) (1.2 * Math.exp(-epoch * 0.2) + rand.nextFloat() * 0.05);
            float acc = (float) Math.min(0.92, 0.5 + epoch * 0.04 + rand.nextFloat() * 0.02);

            // Pick a sample text for this epoch
            int textIdx = (epoch - 1) % sampleTexts.length;
            String text = sampleTexts[textIdx];
            String label = labels[textIdx];
            float confidence = 0.6f + rand.nextFloat() * 0.35f;

            // Token weights (mock attention)
            Map<String, Float> tokenWeights = new LinkedHashMap<>();
            for (String word : text.split("\\s+")) {
                tokenWeights.put(word, rand.nextFloat() * 0.8f);
            }

            // F1/Precision/Recall per class
            Map<String, Float> f1 = Map.of("POSITIVE", acc + 0.03f, "NEGATIVE", acc - 0.02f, "NEUTRAL", acc - 0.05f);
            Map<String, Float> precision = Map.of("POSITIVE", acc + 0.01f, "NEGATIVE", acc - 0.01f, "NEUTRAL", acc - 0.03f);
            Map<String, Float> recall = Map.of("POSITIVE", acc + 0.02f, "NEGATIVE", acc, "NEUTRAL", acc - 0.04f);

            Map<String, Float> metrics = new HashMap<>();
            metrics.put("loss", loss);
            metrics.put("acc", acc);
            metrics.put("train_acc", acc);
            metrics.put("test_acc", acc - 0.03f);
            history.record(epoch, metrics);
            dashboard.setCurrentEpoch(epoch);

            DashboardIntegrationHelper.broadcastNLPDetailed(
                dashboard, epoch, metrics, text, label, confidence,
                f1, precision, recall, tokenWeights);

            System.out.printf("  Epoch %d/%d  loss=%.4f  acc=%.4f  text=\"%s\" → %s (%.1f%%)%n",
                epoch, epochs, loss, acc, text.substring(0, Math.min(40, text.length())), label, confidence * 100);

            while (dashboard.isTrainingPaused()) { Thread.sleep(200); }
            Thread.sleep(1200);
        }
    }

    // ==================== DETECTION ====================

    private static void testDetection(DashboardServer dashboard, TrainingHistory history, int epochs)
            throws InterruptedException {
        dashboard.setTaskType("detection");
        dashboard.setModelInfo("YOLO-COCO (E2E Test)", epochs);

        String[] detClasses = {"person", "car", "dog", "cat", "bicycle", "bus", "truck"};

        for (int epoch = 1; epoch <= epochs; epoch++) {
            float totalLoss = (float) (3.0 * Math.exp(-epoch * 0.15) + rand.nextFloat() * 0.1);
            float mAP = (float) Math.min(0.85, 0.15 + epoch * 0.12 + rand.nextFloat() * 0.05);

            // Generate a fake image (3x128x128 noise)
            float[] imagePixels = new float[3 * 128 * 128];
            for (int i = 0; i < imagePixels.length; i++) imagePixels[i] = rand.nextFloat() * 2 - 1;

            // Generate bounding boxes (predicted)
            List<Map<String, Object>> predBoxes = new ArrayList<>();
            int nBoxes = 3 + rand.nextInt(5);
            for (int b = 0; b < nBoxes; b++) {
                Map<String, Object> box = new HashMap<>();
                box.put("x", 0.05 + rand.nextDouble() * 0.6);
                box.put("y", 0.05 + rand.nextDouble() * 0.6);
                box.put("w", 0.1 + rand.nextDouble() * 0.25);
                box.put("h", 0.1 + rand.nextDouble() * 0.3);
                box.put("label", detClasses[rand.nextInt(detClasses.length)]);
                box.put("score", 0.3 + rand.nextDouble() * 0.65);
                predBoxes.add(box);
            }

            // Ground truth boxes
            List<Map<String, Object>> gtBoxes = new ArrayList<>();
            for (int b = 0; b < 3; b++) {
                Map<String, Object> box = new HashMap<>();
                box.put("x", 0.1 + rand.nextDouble() * 0.5);
                box.put("y", 0.1 + rand.nextDouble() * 0.5);
                box.put("w", 0.15 + rand.nextDouble() * 0.2);
                box.put("h", 0.15 + rand.nextDouble() * 0.25);
                box.put("label", detClasses[rand.nextInt(detClasses.length)]);
                gtBoxes.add(box);
            }

            // Loss breakdown
            Map<String, Float> lossBreakdown = new LinkedHashMap<>();
            lossBreakdown.put("box_loss", totalLoss * 0.4f + rand.nextFloat() * 0.1f);
            lossBreakdown.put("obj_loss", totalLoss * 0.35f + rand.nextFloat() * 0.05f);
            lossBreakdown.put("cls_loss", totalLoss * 0.25f + rand.nextFloat() * 0.05f);

            // Per-class mAP
            Map<String, Float> perClassMAP = new LinkedHashMap<>();
            for (String cls : detClasses) {
                perClassMAP.put(cls, mAP + (rand.nextFloat() - 0.5f) * 0.2f);
            }

            Map<String, Float> metrics = new HashMap<>();
            metrics.put("loss", totalLoss);
            metrics.put("train_loss", totalLoss);
            metrics.put("acc", mAP);
            metrics.put("mAP", mAP);
            history.record(epoch, metrics);
            dashboard.setCurrentEpoch(epoch);

            DashboardIntegrationHelper.broadcastDetectionDetailed(
                dashboard, epoch, metrics, imagePixels, 128, 128,
                predBoxes, gtBoxes, lossBreakdown, perClassMAP, 15.0f + rand.nextFloat() * 10);

            System.out.printf("  Epoch %d/%d  loss=%.4f  mAP=%.4f  boxes=%d%n",
                epoch, epochs, totalLoss, mAP, nBoxes);

            while (dashboard.isTrainingPaused()) { Thread.sleep(200); }
            Thread.sleep(1500);
        }
    }
}
