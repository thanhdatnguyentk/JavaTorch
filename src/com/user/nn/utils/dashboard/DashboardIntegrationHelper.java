package com.user.nn.utils.dashboard;

import com.user.nn.predict.ImagePredictor;
import com.user.nn.predict.PredictionResult;
import com.user.nn.predict.TextPredictor;

import javax.imageio.ImageIO;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.util.HashMap;
import java.util.Map;

public class DashboardIntegrationHelper {

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
}
