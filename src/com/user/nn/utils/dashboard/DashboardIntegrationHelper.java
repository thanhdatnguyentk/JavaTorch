package com.user.nn.utils.dashboard;

import com.user.nn.predict.ImagePredictor;
import com.user.nn.predict.PredictionResult;
import com.user.nn.predict.TextPredictor;

import javax.imageio.ImageIO;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.util.ArrayList;
import java.util.Base64;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class DashboardIntegrationHelper {

    // ==================== HANDLER SETUP ====================

    public static void setupImagePredictorHandler(DashboardServer server, String taskName, ImagePredictor predictor) {
        server.registerHandler(taskName, (fileName, fileStream, text) -> {
            try {
                if (fileStream == null) {
                    return Map.of("error", "No image stream provided");
                }
                BufferedImage img = ImageIO.read(fileStream);
                if (img == null) {
                    return Map.of("error", "Could not decode the uploaded image");
                }

                int w = predictor.getWidth();
                int h = predictor.getHeight();
                int c = predictor.getChannels();

                BufferedImage resized = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
                Graphics2D g = resized.createGraphics();
                g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
                g.drawImage(img, 0, 0, w, h, null);
                g.dispose();

                float[] pixels = new float[c * h * w];
                for (int y = 0; y < h; y++) {
                    for (int x = 0; x < w; x++) {
                        int rgb = resized.getRGB(x, y);
                        int r = (rgb >> 16) & 0xFF;
                        int green = (rgb >> 8) & 0xFF;
                        int b = rgb & 0xFF;

                        if (c == 1) {
                            float gray = (r * 0.299f + green * 0.587f + b * 0.114f);
                            pixels[y * w + x] = gray;
                        } else if (c == 3) {
                            int planarSize = w * h;
                            pixels[0 * planarSize + y * w + x] = (float) r;
                            pixels[1 * planarSize + y * w + x] = (float) green;
                            pixels[2 * planarSize + y * w + x] = (float) b;
                        }
                    }
                }

                PredictionResult result = predictor.predictFromPixels(pixels);
                Map<String, Object> resp = new HashMap<>();
                resp.put("class_index", result.getPredictedClass());
                resp.put("label", result.getPredictedLabel());
                resp.put("confidence", result.getConfidence());
                resp.put("topK", result.getTopKLabels());
                return resp;

            } catch (Exception e) {
                e.printStackTrace();
                return Map.of("error", e.getMessage());
            }
        });
    }

    public static void setupTextPredictorHandler(DashboardServer server, String taskName, TextPredictor predictor) {
        server.registerHandler(taskName, (fileName, fileStream, text) -> {
            try {
                if (text == null || text.trim().isEmpty()) {
                    return Map.of("error", "No text provided");
                }
                PredictionResult result = predictor.predictText(text);
                Map<String, Object> resp = new HashMap<>();
                resp.put("class_index", result.getPredictedClass());
                resp.put("label", result.getPredictedLabel());
                resp.put("confidence", result.getConfidence());
                resp.put("topK", result.getTopKLabels());
                return resp;
            } catch (Exception e) {
                e.printStackTrace();
                return Map.of("error", e.getMessage());
            }
        });
    }

    // ==================== IMAGE ENCODING ====================

    /**
     * Chuyển đổi Generator output (CHW) thành Base64 string để hiển thị trên web.
     */
    public static String encodeGeneratorOutput(float[] pixels, int c, int h, int w) {
        BufferedImage img = new BufferedImage(w, h, c == 1 ? BufferedImage.TYPE_BYTE_GRAY : BufferedImage.TYPE_INT_RGB);
        int planarSize = w * h;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                if (c == 1) {
                    float val = (pixels[y * w + x] + 1f) / 2f * 255f; // [-1,1] -> [0,255]
                    int gray = Math.min(255, Math.max(0, (int) val));
                    img.getRaster().setSample(x, y, 0, gray);
                } else {
                    float r = (pixels[0 * planarSize + y * w + x] + 1f) / 2f * 255f;
                    float g = (pixels[1 * planarSize + y * w + x] + 1f) / 2f * 255f;
                    float b = (pixels[2 * planarSize + y * w + x] + 1f) / 2f * 255f;
                    int rgb = ((Math.min(255, Math.max(0, (int) r)) & 0xFF) << 16) |
                              ((Math.min(255, Math.max(0, (int) g)) & 0xFF) << 8) |
                               (Math.min(255, Math.max(0, (int) b)) & 0xFF);
                    img.setRGB(x, y, rgb);
                }
            }
        }
        try (ByteArrayOutputStream baos = new ByteArrayOutputStream()) {
            ImageIO.write(img, "png", baos);
            return Base64.getEncoder().encodeToString(baos.toByteArray());
        } catch (Exception e) {
            return "";
        }
    }

    /** Encode a raw pixel array [0,255] range (CHW) to base64 PNG */
    public static String encodePixelsToBase64(float[] pixels, int c, int h, int w) {
        BufferedImage img = new BufferedImage(w, h, c == 1 ? BufferedImage.TYPE_BYTE_GRAY : BufferedImage.TYPE_INT_RGB);
        int planarSize = w * h;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                if (c == 1) {
                    int gray = Math.min(255, Math.max(0, (int) pixels[y * w + x]));
                    img.getRaster().setSample(x, y, 0, gray);
                } else {
                    int r = Math.min(255, Math.max(0, (int) (pixels[0 * planarSize + y * w + x] * 255f)));
                    int g = Math.min(255, Math.max(0, (int) (pixels[1 * planarSize + y * w + x] * 255f)));
                    int b = Math.min(255, Math.max(0, (int) (pixels[2 * planarSize + y * w + x] * 255f)));
                    img.setRGB(x, y, (r << 16) | (g << 8) | b);
                }
            }
        }
        try (ByteArrayOutputStream baos = new ByteArrayOutputStream()) {
            ImageIO.write(img, "png", baos);
            return Base64.getEncoder().encodeToString(baos.toByteArray());
        } catch (Exception e) {
            return "";
        }
    }

    // ==================== CLASSIFICATION BROADCASTS ====================

    /**
     * Rich classification broadcast: metrics + confusion matrix + live prediction samples.
     */
    public static void broadcastClassificationDetailed(
            DashboardServer server, int epoch,
            Map<String, Float> metrics,
            int[][] confusionMatrix, String[] classLabels,
            List<Map<String, Object>> livePredictions) {
        
        DashboardMetricsCollector collector = server.getMetricsCollector();
        
        // Store confusion matrix
        if (confusionMatrix != null && classLabels != null) {
            collector.recordConfusionMatrix(epoch, confusionMatrix, classLabels);
        }
        
        // Store live predictions
        if (livePredictions != null) {
            collector.setLivePredictions(livePredictions);
        }
        
        Map<String, Object> payload = new HashMap<>(metrics);
        payload.put("taskType", "classification");
        if (confusionMatrix != null) {
            payload.put("confusionMatrix", confusionMatrix);
            payload.put("classLabels", classLabels);
        }
        if (livePredictions != null && !livePredictions.isEmpty()) {
            payload.put("livePredictions", livePredictions);
        }
        
        server.broadcastTaskMetrics(epoch, payload);
    }

    /** Build a single live prediction entry */
    public static Map<String, Object> buildLivePrediction(
            String imageBase64, String predictedLabel, String actualLabel,
            boolean correct, List<Map<String, Object>> topK) {
        Map<String, Object> pred = new LinkedHashMap<>();
        pred.put("image", imageBase64);
        pred.put("predicted", predictedLabel);
        pred.put("actual", actualLabel);
        pred.put("correct", correct);
        pred.put("topK", topK);
        return pred;
    }

    /** Build a top-K confidence entry */
    public static Map<String, Object> buildTopKEntry(String label, float confidence) {
        Map<String, Object> entry = new LinkedHashMap<>();
        entry.put("label", label);
        entry.put("confidence", confidence);
        return entry;
    }

    // ==================== GAN BROADCASTS ====================

    /**
     * Gửi định kỳ mẫu ảnh GAN qua WebSocket
     */
    public static void broadcastGANSamples(DashboardServer server, int epoch, float gLoss, float dLoss, List<float[]> samples, int h, int w) {
        Map<String, Object> metrics = new HashMap<>();
        metrics.put("g_loss", gLoss);
        metrics.put("d_loss", dLoss);
        
        List<String> base64Samples = new ArrayList<>();
        for (float[] s : samples) {
            base64Samples.add(encodeGeneratorOutput(s, 3, h, w));
        }
        metrics.put("samples", base64Samples);
        
        // Also store in history for time-lapse
        server.getMetricsCollector().recordGANSamples(epoch, base64Samples);
        
        server.broadcastTaskMetrics(epoch, metrics);
    }

    /**
     * Rich GAN broadcast with extended sample count.
     */
    public static void broadcastGANDetailed(
            DashboardServer server, int epoch,
            float gLoss, float dLoss,
            List<float[]> samples, int c, int h, int w) {
        
        Map<String, Object> metrics = new HashMap<>();
        metrics.put("g_loss", gLoss);
        metrics.put("d_loss", dLoss);
        
        List<String> base64Samples = new ArrayList<>();
        for (float[] s : samples) {
            base64Samples.add(encodeGeneratorOutput(s, c, h, w));
        }
        metrics.put("samples", base64Samples);
        metrics.put("sampleCount", base64Samples.size());
        
        // Store in history for time-lapse
        server.getMetricsCollector().recordGANSamples(epoch, base64Samples);
        
        server.broadcastTaskMetrics(epoch, metrics);
    }

    // ==================== DETECTION BROADCASTS ====================

    /**
     * Broadcasts Object Detection results (Image + Bounding Boxes) to the dashboard.
     */
    public static void broadcastDetection(DashboardServer server, int epoch, Map<String, Float> metrics, float[] imagePixels, int h, int w, List<Map<String, Object>> boxes) {
        Map<String, Object> payload = new HashMap<>(metrics);
        payload.put("taskType", "detection");
        payload.put("image", encodeGeneratorOutput(imagePixels, 3, h, w));
        payload.put("boxes", boxes);
        server.broadcastTaskMetrics(epoch, payload);
    }

    /**
     * Rich detection broadcast with loss breakdown, per-class mAP, and FPS.
     */
    public static void broadcastDetectionDetailed(
            DashboardServer server, int epoch,
            Map<String, Float> metrics,
            float[] imagePixels, int h, int w,
            List<Map<String, Object>> predBoxes,
            List<Map<String, Object>> gtBoxes,
            Map<String, Float> lossBreakdown,
            Map<String, Float> perClassMAP,
            float fps) {
        
        DashboardMetricsCollector collector = server.getMetricsCollector();
        if (lossBreakdown != null) collector.setLossBreakdown(lossBreakdown);
        if (perClassMAP != null) collector.setPerClassMAP(perClassMAP);
        
        Map<String, Object> payload = new HashMap<>(metrics);
        payload.put("taskType", "detection");
        if (imagePixels != null) {
            payload.put("image", encodePixelsToBase64(imagePixels, 3, h, w));
        }
        payload.put("predBoxes", predBoxes != null ? predBoxes : new ArrayList<>());
        payload.put("gtBoxes", gtBoxes != null ? gtBoxes : new ArrayList<>());
        if (lossBreakdown != null) payload.put("lossBreakdown", lossBreakdown);
        if (perClassMAP != null) payload.put("leaderboard", perClassMAP);
        payload.put("fps", fps);
        
        server.broadcastTaskMetrics(epoch, payload);
    }

    // ==================== NLP BROADCASTS ====================

    /**
     * Broadcasts NLP task metrics and samples (Text + Sentiment/Classification).
     */
    public static void broadcastNLP(DashboardServer server, int epoch, Map<String, Float> metrics, String text, String predictedLabel, float confidence) {
        Map<String, Object> payload = new HashMap<>(metrics);
        payload.put("taskType", "nlp");
        payload.put("sampleText", text);
        payload.put("predictedLabel", predictedLabel);
        payload.put("confidence", confidence);
        server.broadcastTaskMetrics(epoch, payload);
    }

    /**
     * Rich NLP broadcast with F1/Precision/Recall, text stream, attention weights.
     */
    public static void broadcastNLPDetailed(
            DashboardServer server, int epoch,
            Map<String, Float> metrics,
            String text, String predictedLabel, float confidence,
            Map<String, Float> classF1Scores,
            Map<String, Float> classPrecision,
            Map<String, Float> classRecall,
            Map<String, Float> tokenWeights) {
        
        DashboardMetricsCollector collector = server.getMetricsCollector();
        collector.addNLPTextSample(text, predictedLabel, confidence, tokenWeights);
        
        Map<String, Object> payload = new HashMap<>(metrics);
        payload.put("taskType", "nlp");
        payload.put("sampleText", text);
        payload.put("predictedLabel", predictedLabel);
        payload.put("confidence", confidence);
        
        if (classF1Scores != null) payload.put("f1Scores", classF1Scores);
        if (classPrecision != null) payload.put("precision", classPrecision);
        if (classRecall != null) payload.put("recall", classRecall);
        if (tokenWeights != null) payload.put("tokenWeights", tokenWeights);
        
        payload.put("textStream", collector.getNLPTextStream());
        payload.put("tokenDistribution", collector.getTokenDistribution());
        
        server.broadcastTaskMetrics(epoch, payload);
    }

    // ==================== SYSTEM BROADCASTS ====================

    /**
     * Broadcast a system alert to the dashboard.
     */
    public static void broadcastSystemAlert(DashboardServer server, String severity, String title, String message) {
        server.getMetricsCollector().addAlert(severity, title, message);
        
        Map<String, Object> payload = new HashMap<>();
        payload.put("taskType", "system_alert");
        payload.put("severity", severity);
        payload.put("title", title);
        payload.put("message", message);
        payload.put("alerts", server.getMetricsCollector().getAlerts());
        
        server.broadcastTaskMetrics(0, payload);
    }
}
