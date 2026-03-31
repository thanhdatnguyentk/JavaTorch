package com.user.nn.examples;

import com.user.nn.utils.dashboard.DashboardServer;
import com.user.nn.utils.dashboard.DashboardIntegrationHelper;
import com.user.nn.utils.visualization.TrainingHistory;
import java.util.HashMap;
import java.util.Map;


import com.user.nn.core.Functional;
import com.user.nn.core.CUDAOps;
import com.user.nn.core.GpuMemoryPool;
import com.user.nn.core.MemoryScope;
import com.user.nn.core.Tensor;
import com.user.nn.core.Torch;
import com.user.nn.dataloaders.Data;
import com.user.nn.models.cv.YOLO;
import com.user.nn.optim.Optim;
import com.user.nn.utils.COCODatasetDownloader;
import com.user.nn.utils.progress.ProgressDataLoader;
import com.user.nn.utils.visualization.LinePlot;
import com.user.nn.utils.visualization.Plot;
import com.user.nn.utils.visualization.PlotContext;
import com.user.nn.utils.visualization.TrainingHistory;
import com.user.nn.utils.visualization.exporters.FileExporter;

import javax.imageio.ImageIO;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Minimal YOLO training example on MS COCO with progress bars and visualization.
 *
 * Notes:
 * - This example is intentionally lightweight and focuses on framework integration.
 * - It parses COCO annotations (images + bbox/category fields) and trains with a
 *   simplified YOLO target encoding using one anchor slot per grid cell.
 */
public class TrainYOLOCoco {

    private static final int IMAGE_SIZE = 448;
    private static final int GRID_SIZE = 7;
    private static final int NUM_BOXES = 2;
    private static final int MAX_BOXES_PER_IMAGE = 20;

    private static class CocoAnnotation {
        final float x;
        final float y;
        final float w;
        final float h;
        final int categoryId;

        CocoAnnotation(float x, float y, float w, float h, int categoryId) {
            this.x = x;
            this.y = y;
            this.w = w;
            this.h = h;
            this.categoryId = categoryId;
        }
    }

    private static class CocoSample {
        final Path imagePath;
        final List<CocoAnnotation> annotations;

        CocoSample(Path imagePath, List<CocoAnnotation> annotations) {
            this.imagePath = imagePath;
            this.annotations = annotations;
        }
    }

    private static class CocoImageMeta {
        final String fileName;
        final int width;
        final int height;

        CocoImageMeta(String fileName, int width, int height) {
            this.fileName = fileName;
            this.width = width;
            this.height = height;
        }
    }

    private static class CocoDetectionDataset implements Data.Dataset {
        private final List<CocoSample> samples;

        CocoDetectionDataset(List<CocoSample> samples) {
            this.samples = samples;
        }

        @Override
        public int len() {
            return samples.size();
        }

        @Override
        public Tensor[] get(int index) {
            CocoSample s = samples.get(index);
            Tensor image = loadAndResizeImageCHW(s.imagePath, IMAGE_SIZE, IMAGE_SIZE);

            // Each row: [x, y, w, h, category_id] in normalized coordinates.
            float[] boxes = new float[MAX_BOXES_PER_IMAGE * 5];
            int count = Math.min(MAX_BOXES_PER_IMAGE, s.annotations.size());
            for (int i = 0; i < count; i++) {
                CocoAnnotation a = s.annotations.get(i);
                int off = i * 5;
                boxes[off] = a.x;
                boxes[off + 1] = a.y;
                boxes[off + 2] = a.w;
                boxes[off + 3] = a.h;
                boxes[off + 4] = a.categoryId;
            }

            Tensor targets = Torch.tensor(boxes, MAX_BOXES_PER_IMAGE, 5);
            return new Tensor[]{image, targets};
        }
    }

    public static void main(String[] args) throws Exception {
        // Parse arguments with robust error handling
        Path imagesDir;
        Path annotationsJson;
        int epochs;
        int batchSize;
        int maxSamples;
        
        try {
            // Use validation set by default (5K images, ~1GB) instead of train set (118K images, ~18GB)
            imagesDir = args.length > 0
                    ? Paths.get(args[0])
                    : Paths.get("data/coco/val2017");
            annotationsJson = args.length > 1
                    ? Paths.get(args[1])
                    : Paths.get("data/coco/annotations/instances_val2017.json");

            // Parse numeric arguments with validation
            if (args.length > 2) {
                try {
                    epochs = Integer.parseInt(args[2]);
                } catch (NumberFormatException e) {
                    System.err.println("ERROR: Argument 3 (epochs) must be an integer, got: " + args[2]);
                    printUsage();
                    return;
                }
            } else {
                epochs = 2;
            }
            
            if (args.length > 3) {
                try {
                    batchSize = Integer.parseInt(args[3]);
                } catch (NumberFormatException e) {
                    System.err.println("ERROR: Argument 4 (batch_size) must be an integer, got: " + args[3]);
                    printUsage();
                    return;
                }
            } else {
                batchSize = 4;
            }
            
            if (args.length > 4) {
                try {
                    maxSamples = Integer.parseInt(args[4]);
                } catch (NumberFormatException e) {
                    System.err.println("ERROR: Argument 5 (max_samples) must be an integer, got: " + args[4]);
                    printUsage();
                    return;
                }
            } else {
                maxSamples = 200;
            }
            
            // Warn if too many arguments
            if (args.length > 5) {
                System.out.println("WARNING: Extra arguments ignored (expected max 5, got " + args.length + ")");
            }
            
        } catch (Exception e) {
            System.err.println("ERROR: Failed to parse arguments: " + e.getMessage());
            printUsage();
            return;
        }

        System.out.println("=== YOLO Training on MS COCO ===");
        System.out.println("imagesDir=" + imagesDir);
        System.out.println("annotations=" + annotationsJson);
        System.out.println("epochs=" + epochs + ", batchSize=" + batchSize + ", maxSamples=" + maxSamples);
        System.out.println();

        // Auto-download dataset if not present
        if (!Files.exists(imagesDir) || !Files.exists(annotationsJson)) {
            System.out.println("COCO dataset not found at specified paths.");
            System.out.println("Attempting to download COCO validation set automatically...");
            System.out.println();
            
            Path workDir = Paths.get(".").toAbsolutePath().normalize();
            boolean downloaded = COCODatasetDownloader.downloadCOCODataset(workDir);
            
            if (!downloaded) {
                System.err.println("Failed to download COCO dataset.");
                System.out.println("Please download manually from: http://cocodataset.org/");
                printUsage();
                return;
            }
            
            // Update paths after download
            imagesDir = Paths.get("data/coco/val2017");
            annotationsJson = Paths.get("data/coco/annotations/instances_val2017.json");
        }

        CocoParseResult parsed = parseCoco(imagesDir, annotationsJson, maxSamples);
        if (parsed.samples.isEmpty()) {
            System.out.println("No COCO samples parsed. Check paths/annotation file.");
            return;
        }

        System.out.println("Loaded samples: " + parsed.samples.size());
        System.out.println("Detected categories: " + parsed.categoryToIndex.size());
        System.out.println();

        int numClasses = Math.max(1, parsed.categoryToIndex.size());
        YOLO model = new YOLO(numClasses, IMAGE_SIZE, IMAGE_SIZE, GRID_SIZE, NUM_BOXES);
        boolean useGpu = CUDAOps.isAvailable();
        if (useGpu) {
            System.out.println("CUDA available. Initializing GPU memory pool and moving model to GPU...");
            GpuMemoryPool.autoInit(model);
            model.toGPU();
            System.out.println("Model is on GPU.");
        } else {
            System.out.println("CUDA not available. Running on CPU.");
        }
        Optim.Adam optimizer = new Optim.Adam(model.parameters(), 1e-4f);

        Data.Dataset dataset = new CocoDetectionDataset(parsed.samples);
        Data.DataLoader loader = new Data.DataLoader(dataset, batchSize, true, 2);

        TrainingHistory history = new TrainingHistory();
        DashboardServer dashboard = new DashboardServer(7070, history).start();
        dashboard.setTaskType("detection");
        dashboard.setModelInfo("YOLO-COCO", epochs);
        for (int epoch = 0; epoch < epochs; epoch++) {
            model.train();
            float epochLoss = 0f;
            float epochObjectness = 0f;
            int batches = 0;

            ProgressDataLoader progress = new ProgressDataLoader(
                    loader,
                    String.format("COCO Epoch %d/%d", epoch + 1, epochs)
            );

            for (Tensor[] batch : progress) {
                try (MemoryScope scope = new MemoryScope()) {
                    Tensor xBatch = batch[0];           // [B, 3, H, W]
                    Tensor boxBatch = batch[1];         // [B, MAX_BOXES, 5]
                    if (useGpu) {
                        xBatch.toGPU();
                    }

                    optimizer.zero_grad();
                    Tensor pred = model.forward(xBatch); // [B, C, Gh, Gw]
                    Tensor target = buildYoloTarget(pred, boxBatch, parsed.categoryToIndex, numClasses);

                    Tensor loss = Functional.mse_loss_tensor(pred, target);
                    loss.backward();
                    optimizer.step();

                    float obj = averageObjectness(pred, NUM_BOXES);
                    epochLoss += loss.data[0];
                    epochObjectness += obj;
                    batches++;

                    progress.setPostfix("loss", String.format("%.5f", loss.data[0]));
                    progress.setPostfix("obj", String.format("%.4f", obj));

                    // Real-time Dashboard Visualization (every 20 batches)
                    if (batches % 20 == 0) {
                        try {
                            float[] firstImage = new float[3 * IMAGE_SIZE * IMAGE_SIZE];
                            System.arraycopy(xBatch.data, 0, firstImage, 0, firstImage.length);
                            
                            List<Map<String, Object>> boxes = new ArrayList<>();
                            // Extract top detections from the current prediction for visualization
                            // (Simplified box extraction for demo purposes)
                            float threshold = 0.3f;
                            int b = 0; // first image in batch
                            for (int gy = 0; gy < GRID_SIZE; gy++) {
                                for (int gx = 0; gx < GRID_SIZE; gx++) {
                                    for (int boxIdx = 0; boxIdx < NUM_BOXES; boxIdx++) {
                                        int off = (b * (NUM_BOXES * 5 + numClasses) + (boxIdx * 5)) * GRID_SIZE * GRID_SIZE + gy * GRID_SIZE + gx;
                                        float conf = pred.data[off + 4];
                                        if (conf > threshold) {
                                            Map<String, Object> box = new HashMap<>();
                                            float cx = (gx + pred.data[off]) / GRID_SIZE;
                                            float cy = (gy + pred.data[off + 1]) / GRID_SIZE;
                                            float bw = pred.data[off + 2];
                                            float bh = pred.data[off + 3];
                                            box.put("x", cx - bw / 2);
                                            box.put("y", cy - bh / 2);
                                            box.put("w", bw);
                                            box.put("h", bh);
                                            box.put("label", "obj");
                                            box.put("score", conf);
                                            boxes.add(box);
                                        }
                                    }
                                }
                            }
                            
                            Map<String, Float> dashMetrics = new HashMap<>();
                            dashMetrics.put("loss", loss.data[0]);
                            dashMetrics.put("objectness", obj);
                            
                            // Mock detection details
                            Map<String, Float> lossBreakdown = Map.of("box", loss.data[0] * 0.4f, "obj", loss.data[0] * 0.4f, "cls", loss.data[0] * 0.2f);
                            Map<String, Float> leaderboard = new HashMap<>();
                            leaderboard.put("person", 0.72f);
                            leaderboard.put("car", 0.65f);
                            leaderboard.put("dog", 0.58f);
                            
                            DashboardIntegrationHelper.broadcastDetectionDetailed(
                                dashboard, epoch + 1, dashMetrics, firstImage, IMAGE_SIZE, IMAGE_SIZE, boxes,
                                null, lossBreakdown, leaderboard, 30f
                            );
                        } catch (Exception dashEx) {}
                    }
                    while (dashboard.isTrainingPaused()) {
                        try { Thread.sleep(200); } catch (InterruptedException ie) { break; }
                    }
                }
            }

            float avgLoss = epochLoss / Math.max(1, batches);
            float avgObj = epochObjectness / Math.max(1, batches);
            dashboard.setCurrentEpoch(epoch + 1);

            Map<String, Float> metrics = new LinkedHashMap<>();
            metrics.put("train_loss", avgLoss);
            metrics.put("avg_objectness", avgObj);
            history.record(epoch, metrics);

            System.out.printf("Epoch %d/%d: loss=%.6f objectness=%.4f%n", epoch + 1, epochs, avgLoss, avgObj);
        
            try {
                Map<String, Float> dashMetrics = new HashMap<>();
                dashMetrics.put("loss", avgLoss);
                history.record(epoch + 1, dashMetrics);
                dashboard.broadcastMetrics(epoch + 1, dashMetrics);
            } catch (Exception dashEx) {}
}

        loader.shutdown();

        saveVisualizations(history);
        history.saveCSV("coco_yolo_training_history.csv");
        System.out.println("Saved: coco_yolo_training_history.csv");
    }

    private static Tensor buildYoloTarget(Tensor pred, Tensor boxBatch,
                                          Map<Integer, Integer> categoryToIndex,
                                          int numClasses) {
        Tensor target = Torch.zeros(pred.shape);

        int bsz = pred.shape[0];
        int channels = pred.shape[1];
        int gh = pred.shape[2];
        int gw = pred.shape[3];

        int classOffset = NUM_BOXES * 5;
        int maxBoxes = boxBatch.shape[1];

        for (int b = 0; b < bsz; b++) {
            for (int bi = 0; bi < maxBoxes; bi++) {
                int base = (b * maxBoxes + bi) * 5;
                float x = boxBatch.data[base];
                float y = boxBatch.data[base + 1];
                float w = boxBatch.data[base + 2];
                float h = boxBatch.data[base + 3];
                int catId = (int) boxBatch.data[base + 4];

                if (w <= 0f || h <= 0f) {
                    continue;
                }

                int gy = Math.min(gh - 1, Math.max(0, (int) (y * gh)));
                int gx = Math.min(gw - 1, Math.max(0, (int) (x * gw)));

                float xCell = x * gw - gx;
                float yCell = y * gh - gy;

                setChannel(target, b, 0, gy, gx, xCell, channels, gh, gw);
                setChannel(target, b, 1, gy, gx, yCell, channels, gh, gw);
                setChannel(target, b, 2, gy, gx, w, channels, gh, gw);
                setChannel(target, b, 3, gy, gx, h, channels, gh, gw);
                setChannel(target, b, 4, gy, gx, 1f, channels, gh, gw);

                Integer classIdx = categoryToIndex.get(catId);
                if (classIdx != null && classIdx >= 0 && classIdx < numClasses) {
                    int clsChannel = classOffset + classIdx;
                    if (clsChannel < channels) {
                        setChannel(target, b, clsChannel, gy, gx, 1f, channels, gh, gw);
                    }
                }
            }
        }
        if (pred.isGPU()) {
            target.toGPU();
        }
        return target;
    }

    private static void setChannel(Tensor t, int b, int c, int y, int x, float v,
                                   int channels, int h, int w) {
        int idx = ((b * channels + c) * h + y) * w + x;
        if (idx >= 0 && idx < t.data.length) {
            t.data[idx] = v;
        }
    }

    private static float averageObjectness(Tensor pred, int numBoxes) {
        int bsz = pred.shape[0];
        int channels = pred.shape[1];
        int h = pred.shape[2];
        int w = pred.shape[3];

        float sum = 0f;
        int count = 0;

        for (int b = 0; b < bsz; b++) {
            for (int box = 0; box < numBoxes; box++) {
                int c = box * 5 + 4;
                if (c >= channels) {
                    continue;
                }
                for (int y = 0; y < h; y++) {
                    for (int x = 0; x < w; x++) {
                        int idx = ((b * channels + c) * h + y) * w + x;
                        sum += pred.data[idx];
                        count++;
                    }
                }
            }
        }
        return sum / Math.max(1, count);
    }

    private static void saveVisualizations(TrainingHistory history) {
        try {
            Plot curves = history.plot("train_loss", "avg_objectness");
            PlotContext ctx = new PlotContext()
                    .title("YOLO Training on MS COCO")
                    .xlabel("Epoch")
                    .ylabel("Value")
                    .grid(true);
            FileExporter.savePNG(curves, ctx, "coco_yolo_training_curves.png", 1000, 600);
            System.out.println("Saved: coco_yolo_training_curves.png");

            List<Float> loss = history.getMetric("train_loss");
            List<Float> obj = history.getMetric("avg_objectness");
            if (!loss.isEmpty() && loss.size() == obj.size()) {
                double[] x = new double[loss.size()];
                double[] y = new double[loss.size()];
                for (int i = 0; i < loss.size(); i++) {
                    x[i] = i;
                    y[i] = loss.get(i);
                }
                LinePlot lossPlot = new LinePlot(x, y, "train_loss");
                PlotContext lossCtx = new PlotContext()
                        .title("COCO YOLO Loss Curve")
                        .xlabel("Epoch")
                        .ylabel("Loss")
                        .grid(true);
                FileExporter.savePNG(lossPlot, lossCtx, "coco_yolo_loss.png", 1000, 600);
                System.out.println("Saved: coco_yolo_loss.png");
            }
        } catch (Exception e) {
            System.err.println("Could not save visualization: " + e.getMessage());
        }
    }

    private static class CocoParseResult {
        final List<CocoSample> samples;
        final Map<Integer, Integer> categoryToIndex;

        CocoParseResult(List<CocoSample> samples, Map<Integer, Integer> categoryToIndex) {
            this.samples = samples;
            this.categoryToIndex = categoryToIndex;
        }
    }

    private static CocoParseResult parseCoco(Path imagesDir, Path annotationJson, int maxSamples) throws IOException {
        String text = Files.readString(annotationJson, StandardCharsets.UTF_8);

        // COCO image fields also vary in order; match id and file_name with lookaheads.
        Pattern imagePattern = Pattern.compile(
            "\\{(?=[^{}]*\\\"id\\\"\\s*:\\s*(\\d+))(?=[^{}]*\\\"file_name\\\"\\s*:\\s*\\\"([^\\\"]+)\\\")[^{}]*}"
        );
        Matcher imageMatcher = imagePattern.matcher(text);

        Map<Integer, String> imageIdToFile = new HashMap<>();
        Map<Integer, CocoImageMeta> imageMeta = new HashMap<>();
        while (imageMatcher.find()) {
            int id = Integer.parseInt(imageMatcher.group(1));
            String file = imageMatcher.group(2);
            imageIdToFile.put(id, file);
        }

        // Extract width/height directly from COCO "images" entries to avoid expensive ImageIO reads.
        Pattern imageMetaPattern = Pattern.compile(
                "\\{(?=[^{}]*\\\"id\\\"\\s*:\\s*(\\d+))(?=[^{}]*\\\"file_name\\\"\\s*:\\s*\\\"([^\\\"]+)\\\")(?=[^{}]*\\\"width\\\"\\s*:\\s*(\\d+))(?=[^{}]*\\\"height\\\"\\s*:\\s*(\\d+))[^{}]*}"
        );
        Matcher imageMetaMatcher = imageMetaPattern.matcher(text);
        while (imageMetaMatcher.find()) {
            int id = Integer.parseInt(imageMetaMatcher.group(1));
            String file = imageMetaMatcher.group(2);
            int width = Integer.parseInt(imageMetaMatcher.group(3));
            int height = Integer.parseInt(imageMetaMatcher.group(4));
            imageMeta.put(id, new CocoImageMeta(file, width, height));
        }

        // COCO annotation fields are not guaranteed to appear in a fixed order.
        // Use lookaheads so image_id/category_id/bbox can be matched regardless of order.
        Pattern annPattern = Pattern.compile(
            "\\{(?=[^{}]*\\\"image_id\\\"\\s*:\\s*(\\d+))(?=[^{}]*\\\"category_id\\\"\\s*:\\s*(\\d+))(?=[^{}]*\\\"bbox\\\"\\s*:\\s*\\[(.*?)\\])[^{}]*}"
        );
        Matcher annMatcher = annPattern.matcher(text);

        Map<Integer, List<CocoAnnotation>> perImage = new HashMap<>();
        Map<Integer, Integer> categoryToIndex = new LinkedHashMap<>();

        while (annMatcher.find()) {
            int imageId = Integer.parseInt(annMatcher.group(1));
            int catId = Integer.parseInt(annMatcher.group(2));
            String bboxRaw = annMatcher.group(3);
            String[] parts = bboxRaw.split(",");
            if (parts.length < 4) {
                continue;
            }

            float x = parseFloatSafe(parts[0]);
            float y = parseFloatSafe(parts[1]);
            float w = parseFloatSafe(parts[2]);
            float h = parseFloatSafe(parts[3]);

            String file = imageIdToFile.get(imageId);
            if (file == null) {
                continue;
            }

            Path imgPath = imagesDir.resolve(file);
            if (!Files.exists(imgPath)) {
                continue;
            }

            CocoImageMeta meta = imageMeta.get(imageId);
            int imgW;
            int imgH;
            if (meta != null && meta.width > 0 && meta.height > 0) {
                imgW = meta.width;
                imgH = meta.height;
            } else {
                // Fallback only when metadata is missing; this should be rare.
                BufferedImage img;
                try {
                    img = ImageIO.read(imgPath.toFile());
                } catch (IOException ex) {
                    continue;
                }
                if (img == null || img.getWidth() <= 0 || img.getHeight() <= 0) {
                    continue;
                }
                imgW = img.getWidth();
                imgH = img.getHeight();
            }

            if (imgW <= 0 || imgH <= 0) {
                continue;
            }

            float nx = (x + w * 0.5f) / imgW;
            float ny = (y + h * 0.5f) / imgH;
            float nw = w / imgW;
            float nh = h / imgH;

            if (!categoryToIndex.containsKey(catId)) {
                categoryToIndex.put(catId, categoryToIndex.size());
            }

            perImage.computeIfAbsent(imageId, k -> new ArrayList<>())
                    .add(new CocoAnnotation(nx, ny, nw, nh, catId));
        }

        List<CocoSample> samples = new ArrayList<>();
        for (Map.Entry<Integer, List<CocoAnnotation>> e : perImage.entrySet()) {
            String file = imageIdToFile.get(e.getKey());
            if (file == null) {
                continue;
            }
            Path imgPath = imagesDir.resolve(file);
            if (!Files.exists(imgPath)) {
                continue;
            }
            samples.add(new CocoSample(imgPath, e.getValue()));
            if (samples.size() >= maxSamples) {
                break;
            }
        }

        return new CocoParseResult(samples, categoryToIndex);
    }

    private static float parseFloatSafe(String s) {
        try {
            return Float.parseFloat(s.trim());
        } catch (Exception e) {
            return 0f;
        }
    }

    private static Tensor loadAndResizeImageCHW(Path imagePath, int dstW, int dstH) {
        try {
            BufferedImage src = ImageIO.read(imagePath.toFile());
            if (src == null) {
                return Torch.zeros(3, dstH, dstW);
            }

            BufferedImage resized = new BufferedImage(dstW, dstH, BufferedImage.TYPE_INT_RGB);
            Graphics2D g = resized.createGraphics();
            g.drawImage(src, 0, 0, dstW, dstH, null);
            g.dispose();

            float[] chw = new float[3 * dstH * dstW];
            for (int y = 0; y < dstH; y++) {
                for (int x = 0; x < dstW; x++) {
                    int rgb = resized.getRGB(x, y);
                    float r = ((rgb >> 16) & 0xFF) / 255.0f;
                    float gg = ((rgb >> 8) & 0xFF) / 255.0f;
                    float b = (rgb & 0xFF) / 255.0f;

                    int idx = y * dstW + x;
                    chw[idx] = r;
                    chw[dstH * dstW + idx] = gg;
                    chw[2 * dstH * dstW + idx] = b;
                }
            }
            return Torch.tensor(chw, 3, dstH, dstW);
        } catch (Exception e) {
            return Torch.zeros(3, dstH, dstW);
        }
    }

    private static void printUsage() {
        System.out.println();
        System.out.println("=== TrainYOLOCoco Usage ===");
        System.out.println("java TrainYOLOCoco [images_dir] [annotations_json] [epochs] [batch_size] [max_samples]");
        System.out.println();
        System.out.println("Arguments:");
        System.out.println("  images_dir         - Path to COCO images directory (default: data/coco/val2017)");
        System.out.println("  annotations_json   - Path to COCO annotations JSON file (default: data/coco/annotations/instances_val2017.json)");
        System.out.println("  epochs             - Number of training epochs (integer, default: 2)");
        System.out.println("  batch_size         - Batch size for training (integer, default: 4)");
        System.out.println("  max_samples        - Maximum samples to load (integer, default: 200)");
        System.out.println();
        System.out.println("Examples:");
        System.out.println("  ./gradlew :examples:run");
        System.out.println("  ./gradlew :examples:run --args=\"data/coco/val2017 data/coco/annotations/instances_val2017.json 5 8 500\"");
        System.out.println();
    }
}
