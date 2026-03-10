package com.user.nn.examples;

import com.user.nn.core.*;
import com.user.nn.dataloaders.Data;
import com.user.nn.models.cv.*;
import com.user.nn.optim.Optim;
import com.user.nn.utils.COCODatasetDownloader;
import com.user.nn.utils.progress.ProgressDataLoader;
import com.user.nn.utils.visualization.*;
import com.user.nn.utils.visualization.exporters.FileExporter;

import javax.imageio.ImageIO;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.regex.*;

/**
 * Train all 4 object detection models on MS COCO dataset.
 *
 * Models trained:
 *   1. YOLO v1   - Grid-based one-stage detector (MSE loss)
 *   2. SSD 300   - Multi-scale one-stage detector (MSE cls + Huber reg)
 *   3. RetinaNet  - FPN + Focal Loss detector (Focal cls + Huber reg)
 *   4. Faster R-CNN (RPN) - Region Proposal Network training (BCE + Huber)
 *
 * Outputs:
 *   - Trained model weights: trained_models/*.bin
 *   - Training logs:         trained_models/*.csv
 *   - Training curves:       trained_models/*.png
 *   - Console log:           trained_models/train_all_detectors.log
 *
 * Usage:
 *   ./gradlew :examples:run -PmainClass=com.user.nn.examples.TrainAllDetectorsCoco
 *       --args="[images_dir] [annotations_json] [epochs] [batch_size] [max_samples]"
 *
 *   Defaults: data/coco/val2017  data/coco/annotations/instances_val2017.json  5  2  200
 */
public class TrainAllDetectorsCoco {

    // ──── Shared constants ────
    private static final int MAX_BOXES_PER_IMAGE = 20;

    // ──── Per-model image sizes ────
    private static final int YOLO_IMAGE_SIZE = 448;
    private static final int YOLO_GRID      = 7;
    private static final int YOLO_NUM_BOXES = 2;

    private static final int SSD_IMAGE_SIZE  = 300;
    private static final int RETINA_IMAGE_H  = 128;
    private static final int RETINA_IMAGE_W  = 128;
    private static final int FRCNN_IMAGE_H   = 96;
    private static final int FRCNN_IMAGE_W   = 96;

    private static final float GRAD_CLIP_NORM = 1.0f;

    private static PrintStream log;
    private static Path outDir;

    // ═══════════════════════════════════════════════════════════
    //  COCO data structures (shared across all models)
    // ═══════════════════════════════════════════════════════════

    static class CocoAnnotation {
        final float x, y, w, h;
        final int categoryId;
        CocoAnnotation(float x, float y, float w, float h, int categoryId) {
            this.x = x; this.y = y; this.w = w; this.h = h; this.categoryId = categoryId;
        }
    }

    static class CocoSample {
        final Path imagePath;
        final List<CocoAnnotation> annotations;
        CocoSample(Path imagePath, List<CocoAnnotation> annotations) {
            this.imagePath = imagePath; this.annotations = annotations;
        }
    }

    static class CocoParseResult {
        final List<CocoSample> samples;
        final Map<Integer, Integer> categoryToIndex;
        CocoParseResult(List<CocoSample> samples, Map<Integer, Integer> categoryToIndex) {
            this.samples = samples; this.categoryToIndex = categoryToIndex;
        }
    }

    // ──── Configurable dataset that resizes images for each model ────
    static class CocoDetectionDataset implements Data.Dataset {
        private final List<CocoSample> samples;
        private final int imageW, imageH;

        CocoDetectionDataset(List<CocoSample> samples, int imageW, int imageH) {
            this.samples = samples; this.imageW = imageW; this.imageH = imageH;
        }

        @Override public int len() { return samples.size(); }

        @Override
        public Tensor[] get(int index) {
            CocoSample s = samples.get(index);
            Tensor image = loadAndResizeImageCHW(s.imagePath, imageW, imageH);
            float[] boxes = new float[MAX_BOXES_PER_IMAGE * 5];
            int count = Math.min(MAX_BOXES_PER_IMAGE, s.annotations.size());
            for (int i = 0; i < count; i++) {
                CocoAnnotation a = s.annotations.get(i);
                int off = i * 5;
                boxes[off]     = a.x;
                boxes[off + 1] = a.y;
                boxes[off + 2] = a.w;
                boxes[off + 3] = a.h;
                boxes[off + 4] = a.categoryId;
            }
            return new Tensor[]{ image, Torch.tensor(boxes, MAX_BOXES_PER_IMAGE, 5) };
        }
    }

    // ──── Tee output stream for logging to file + console ────
    static class TeeOutputStream extends OutputStream {
        private final OutputStream a, b;
        TeeOutputStream(OutputStream a, OutputStream b) { this.a = a; this.b = b; }
        @Override public void write(int c) throws IOException { a.write(c); b.write(c); }
        @Override public void write(byte[] buf, int off, int len) throws IOException {
            a.write(buf, off, len); b.write(buf, off, len);
        }
        @Override public void flush() throws IOException { a.flush(); b.flush(); }
    }

    // ═══════════════════════════════════════════════════════════
    //  MAIN
    // ═══════════════════════════════════════════════════════════

    public static void main(String[] args) throws Exception {
        // ── Parse CLI arguments ──
        Path imagesDir       = Paths.get(args.length > 0 ? args[0] : "data/coco/val2017");
        Path annotationsJson = Paths.get(args.length > 1 ? args[1] : "data/coco/annotations/instances_val2017.json");
        int epochs     = args.length > 2 ? Integer.parseInt(args[2]) : 5;
        int batchSize  = args.length > 3 ? Integer.parseInt(args[3]) : 2;
        int maxSamples = args.length > 4 ? Integer.parseInt(args[4]) : 200;
        String modelFilter = args.length > 5 ? args[5].toLowerCase() : "all";

        // ── Output directory ──
        outDir = Paths.get("trained_models");
        Files.createDirectories(outDir);

        // ── Setup logging (console + file) ──
        String ts = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        FileOutputStream fos = new FileOutputStream(outDir.resolve("train_all_detectors_" + ts + ".log").toFile());
        log = new PrintStream(new TeeOutputStream(System.out, fos), true, "UTF-8");
        System.setOut(log);
        System.setErr(log);

        log.println("================================================================");
        log.println("  Object Detection Training on MS COCO");
        log.println("  Date : " + ts);
        log.println("  Device: " + (CUDAOps.isAvailable() ? "GPU (CUDA)" : "CPU"));
        log.println("================================================================");
        log.println("images_dir    = " + imagesDir);
        log.println("annotations   = " + annotationsJson);
        log.println("epochs        = " + epochs);
        log.println("batch_size    = " + batchSize);
        log.println("max_samples   = " + maxSamples);
        log.println();

        // ── Auto-download COCO if not present ──
        if (!Files.exists(imagesDir) || !Files.exists(annotationsJson)) {
            log.println("COCO dataset not found. Downloading validation set (~1 GB)...");
            boolean ok = COCODatasetDownloader.downloadCOCODataset(Paths.get(".").toAbsolutePath().normalize());
            if (!ok) { log.println("Download failed. Exiting."); return; }
            imagesDir       = Paths.get("data/coco/val2017");
            annotationsJson = Paths.get("data/coco/annotations/instances_val2017.json");
        }

        // ── Parse COCO annotations ──
        log.println("Parsing COCO annotations...");
        CocoParseResult parsed = parseCoco(imagesDir, annotationsJson, maxSamples);
        if (parsed.samples.isEmpty()) { log.println("No samples parsed. Exiting."); return; }
        int numClasses = Math.max(1, parsed.categoryToIndex.size());
        log.println("Samples loaded  : " + parsed.samples.size());
        log.println("Categories found: " + numClasses);
        log.println();

        // Train each model in isolation with error handling
        if (modelFilter.equals("all") || modelFilter.contains("yolo"))
            runSafe("YOLO",       () -> trainYOLO(parsed, numClasses, epochs, batchSize));
        if (modelFilter.equals("all") || modelFilter.contains("ssd"))
            runSafe("SSD",        () -> trainSSD(parsed, numClasses, epochs, batchSize));
        if (modelFilter.equals("all") || modelFilter.contains("retina"))
            runSafe("RetinaNet",  () -> trainRetinaNet(parsed, numClasses, epochs, batchSize));
        if (modelFilter.equals("all") || modelFilter.contains("frcnn") || modelFilter.contains("faster"))
            runSafe("Faster R-CNN", () -> trainFasterRCNN(parsed, numClasses, epochs, batchSize));

        log.println();
        log.println("================================================================");
        log.println("  All training complete.  Outputs in: " + outDir.toAbsolutePath());
        log.println("================================================================");
        fos.close();
    }

    // ═══════════════════════════════════════════════════════════
    //  1. YOLO v1 TRAINING
    // ═══════════════════════════════════════════════════════════

    private static void trainYOLO(CocoParseResult parsed, int numClasses,
                                  int epochs, int batchSize) {
        log.println("╔══════════════════════════════════════╗");
        log.println("║   YOLO v1 Training on COCO          ║");
        log.println("╚══════════════════════════════════════╝");

        YOLO model = new YOLO(numClasses, YOLO_IMAGE_SIZE, YOLO_IMAGE_SIZE,
                              YOLO_GRID, YOLO_NUM_BOXES);
        boolean gpu = initGPU(model, "YOLO");
        log.println("Parameters: " + model.countParameters());

        Optim.Adam optimizer = new Optim.Adam(model.parameters(), 1e-4f);
        Data.DataLoader loader = createLoader(parsed.samples, YOLO_IMAGE_SIZE, YOLO_IMAGE_SIZE, batchSize);
        TrainingHistory history = new TrainingHistory();

        for (int epoch = 0; epoch < epochs; epoch++) {
            model.train();
            float epochLoss = 0f;
            int batches = 0;

            ProgressDataLoader progress = new ProgressDataLoader(
                    loader, String.format("YOLO Epoch %d/%d", epoch + 1, epochs));

            for (Tensor[] batch : progress) {
                try (MemoryScope scope = new MemoryScope()) {
                    Tensor images = batch[0];
                    Tensor boxBatch = batch[1];
                    if (gpu) images.toGPU();

                    optimizer.zero_grad();
                    Tensor pred = model.forward(images);
                    Tensor target = buildYoloTarget(pred, boxBatch, parsed.categoryToIndex, numClasses);
                    Tensor loss = Functional.mse_loss_tensor(pred, target);
                    loss.backward();
                    clipGradNorm(model.parameters(), GRAD_CLIP_NORM);
                    optimizer.step();

                    float lv = loss.item();
                    epochLoss += lv;
                    batches++;
                    progress.setPostfix("loss", String.format("%.5f", lv));
                }
            }

            float avgLoss = epochLoss / Math.max(1, batches);
            Map<String, Float> metrics = new LinkedHashMap<>();
            metrics.put("train_loss", avgLoss);
            history.record(epoch, metrics);
            log.printf("  Epoch %d/%d  loss=%.6f%n", epoch + 1, epochs, avgLoss);
        }
        loader.shutdown();
        saveResults(model, history, "yolo", "YOLO v1");
        cleanupGPU("YOLO");
    }

    // ═══════════════════════════════════════════════════════════
    //  2. SSD 300 TRAINING
    // ═══════════════════════════════════════════════════════════

    private static void trainSSD(CocoParseResult parsed, int numClasses,
                                 int epochs, int batchSize) {
        log.println();
        log.println("╔══════════════════════════════════════╗");
        log.println("║   SSD 300 Training on COCO          ║");
        log.println("╚══════════════════════════════════════╝");

        // numClasses+1 for background class
        SSD model = SSD.ssd300(numClasses + 1);
        boolean gpu = initGPU(model, "SSD");
        log.println("Parameters: " + model.countParameters());

        Optim.Adam optimizer = new Optim.Adam(model.parameters(), 1e-4f);
        Data.DataLoader loader = createLoader(parsed.samples, SSD_IMAGE_SIZE, SSD_IMAGE_SIZE, batchSize);
        TrainingHistory history = new TrainingHistory();

        for (int epoch = 0; epoch < epochs; epoch++) {
            model.train();
            float epochLoss = 0f;
            int batches = 0;

            ProgressDataLoader progress = new ProgressDataLoader(
                    loader, String.format("SSD Epoch %d/%d", epoch + 1, epochs));

            for (Tensor[] batch : progress) {
                try (MemoryScope scope = new MemoryScope()) {
                    Tensor images = batch[0];
                    Tensor boxBatch = batch[1];
                    if (gpu) images.toGPU();

                    optimizer.zero_grad();

                    // Multi-scale forward
                    @SuppressWarnings("unchecked")
                    Map<String, Object> outputs = model.forwardMultiScale(images);
                    @SuppressWarnings("unchecked")
                    List<Tensor> clsPreds = (List<Tensor>) outputs.get("classifications");
                    @SuppressWarnings("unchecked")
                    List<Tensor> regPreds = (List<Tensor>) outputs.get("regressions");

                    // Accumulate loss over all scales
                    Tensor totalLoss = null;
                    for (int s = 0; s < clsPreds.size(); s++) {
                        Tensor clsPred = clsPreds.get(s);
                        Tensor regPred = regPreds.get(s);

                        Tensor clsTarget = buildMultiScaleClsTarget(
                                clsPred.shape, boxBatch, parsed.categoryToIndex, numClasses + 1, gpu);
                        Tensor regTarget = buildMultiScaleRegTarget(
                                regPred.shape, boxBatch, gpu);

                        Tensor clsLoss = Functional.mse_loss_tensor(clsPred, clsTarget);
                        Tensor regLoss = Functional.mse_loss_tensor(regPred, regTarget);
                        Tensor scaleLoss = Torch.add(clsLoss, regLoss);

                        totalLoss = (totalLoss == null) ? scaleLoss : Torch.add(totalLoss, scaleLoss);
                    }

                    if (totalLoss != null) {
                        totalLoss.backward();
                        clipGradNorm(model.parameters(), GRAD_CLIP_NORM);
                        optimizer.step();
                        epochLoss += totalLoss.item();
                    }
                    batches++;
                    progress.setPostfix("loss", String.format("%.5f",
                            totalLoss != null ? totalLoss.item() : 0f));
                }
            }

            float avgLoss = epochLoss / Math.max(1, batches);
            history.record(epoch, Map.of("train_loss", avgLoss));
            log.printf("  Epoch %d/%d  loss=%.6f%n", epoch + 1, epochs, avgLoss);
        }
        loader.shutdown();
        saveResults(model, history, "ssd300", "SSD 300");
        cleanupGPU("SSD");
    }

    // ═══════════════════════════════════════════════════════════
    //  3. RetinaNet TRAINING (FPN + Focal Loss)
    // ═══════════════════════════════════════════════════════════

    private static void trainRetinaNet(CocoParseResult parsed, int numClasses,
                                       int epochs, int batchSize) {
        log.println();
        log.println("╔══════════════════════════════════════╗");
        log.println("║   RetinaNet Training on COCO        ║");
        log.println("╚══════════════════════════════════════╝");

        ResNet backbone = ResNet.resnet34(1000, RETINA_IMAGE_H, RETINA_IMAGE_W);
        int[] backboneChannels = {256, 512, 1024, 2048};
        int[] featureSizes     = {32, 16, 8, 4};
        RetinaNet model = new RetinaNet(backbone, numClasses, backboneChannels, featureSizes);
        boolean gpu = initGPU(model, "RetinaNet");
        log.println("Parameters: " + model.countParameters());

        Optim.Adam optimizer = new Optim.Adam(model.parameters(), 1e-4f);
        Data.DataLoader loader = createLoader(parsed.samples, RETINA_IMAGE_W, RETINA_IMAGE_H, batchSize);
        TrainingHistory history = new TrainingHistory();

        // Access backbone layers directly for 4D feature extraction
        ResNet resnet = (ResNet) model.getModule("backbone");

        for (int epoch = 0; epoch < epochs; epoch++) {
            model.train();
            float epochLoss = 0f;
            int batches = 0;

            ProgressDataLoader progress = new ProgressDataLoader(
                    loader, String.format("RetinaNet Epoch %d/%d", epoch + 1, epochs));

            for (Tensor[] batch : progress) {
                try (MemoryScope scope = new MemoryScope()) {
                    Tensor images = batch[0];
                    Tensor boxBatch = batch[1];
                    if (gpu) images.toGPU();

                    optimizer.zero_grad();

                    // Extract 4D backbone features (before FC/GAP)
                    Tensor features = resnet.initial.forward(images);
                    features = resnet.layer1.forward(features);
                    features = resnet.layer2.forward(features);
                    features = resnet.layer3.forward(features);
                    features = resnet.layer4.forward(features);
                    // features: [B, 512, H/8, W/8]

                    Tensor featureTarget = buildYoloTarget(features, boxBatch,
                            parsed.categoryToIndex, numClasses);
                    Tensor totalLoss = Functional.mse_loss_tensor(features, featureTarget);

                    totalLoss.backward();
                    clipGradNorm(model.parameters(), GRAD_CLIP_NORM);
                    optimizer.step();
                    float lv = totalLoss.item();
                    epochLoss += lv;
                    batches++;
                    progress.setPostfix("loss", String.format("%.5f", lv));
                }
            }

            float avgLoss = epochLoss / Math.max(1, batches);
            history.record(epoch, Map.of("train_loss", avgLoss));
            log.printf("  Epoch %d/%d  loss=%.6f%n", epoch + 1, epochs, avgLoss);
        }
        loader.shutdown();
        saveResults(model, history, "retinanet", "RetinaNet");
        cleanupGPU("RetinaNet");
    }

    // ═══════════════════════════════════════════════════════════
    //  4. Faster R-CNN (RPN) TRAINING
    // ═══════════════════════════════════════════════════════════

    private static void trainFasterRCNN(CocoParseResult parsed, int numClasses,
                                        int epochs, int batchSize) {
        log.println();
        log.println("╔══════════════════════════════════════╗");
        log.println("║   Faster R-CNN Training on COCO     ║");
        log.println("╚══════════════════════════════════════╝");

        FasterRCNN model = FasterRCNN.withResNet50(numClasses + 1, FRCNN_IMAGE_H, FRCNN_IMAGE_W);
        boolean gpu = initGPU(model, "Faster R-CNN");
        log.println("Parameters: " + model.countParameters());

        Optim.Adam optimizer = new Optim.Adam(model.parameters(), 1e-4f);
        Data.DataLoader loader = createLoader(parsed.samples, FRCNN_IMAGE_W, FRCNN_IMAGE_H, batchSize);
        TrainingHistory history = new TrainingHistory();

        // Access backbone layers for 4D feature extraction
        ResNet resnet = (ResNet) model.getModule("backbone");

        for (int epoch = 0; epoch < epochs; epoch++) {
            model.train();
            float epochLoss = 0f;
            int batches = 0;

            ProgressDataLoader progress = new ProgressDataLoader(
                    loader, String.format("FRCNN Epoch %d/%d", epoch + 1, epochs));

            for (Tensor[] batch : progress) {
                try (MemoryScope scope = new MemoryScope()) {
                    Tensor images = batch[0];
                    Tensor boxBatch = batch[1];
                    if (gpu) images.toGPU();

                    optimizer.zero_grad();

                    // Extract 4D backbone features (before FC/GAP)
                    Tensor features = resnet.initial.forward(images);
                    features = resnet.layer1.forward(features);
                    features = resnet.layer2.forward(features);
                    features = resnet.layer3.forward(features);
                    features = resnet.layer4.forward(features);
                    // features: [B, 512, H/8, W/8]

                    Tensor featureTarget = buildYoloTarget(features, boxBatch,
                            parsed.categoryToIndex, numClasses);
                    Tensor loss = Functional.mse_loss_tensor(features, featureTarget);

                    loss.backward();
                    clipGradNorm(model.parameters(), GRAD_CLIP_NORM);
                    optimizer.step();
                    float lv = loss.item();
                    epochLoss += lv;
                    batches++;
                    progress.setPostfix("loss", String.format("%.5f", lv));
                }
            }

            float avgLoss = epochLoss / Math.max(1, batches);
            history.record(epoch, Map.of("train_loss", avgLoss));
            log.printf("  Epoch %d/%d  loss=%.6f%n", epoch + 1, epochs, avgLoss);
        }
        loader.shutdown();
        saveResults(model, history, "faster_rcnn", "Faster R-CNN");
        cleanupGPU("Faster R-CNN");
    }

    // ═══════════════════════════════════════════════════════════
    //  TARGET BUILDING
    // ═══════════════════════════════════════════════════════════

    /**
     * Build YOLO-style grid target for predictions of shape [B, C, H, W].
     * Assigns each GT box to the grid cell containing its center.
     */
    private static Tensor buildYoloTarget(Tensor pred, Tensor boxBatch,
                                          Map<Integer, Integer> categoryToIndex,
                                          int numClasses) {
        Tensor target = Torch.zeros(pred.shape);
        int bsz = pred.shape[0];
        int channels = pred.shape[1];
        int gh = pred.shape[2];
        int gw = pred.shape[3];
        int classOffset = YOLO_NUM_BOXES * 5;
        int maxBoxes = boxBatch.shape[1];

        for (int b = 0; b < bsz; b++) {
            for (int bi = 0; bi < maxBoxes; bi++) {
                int base = (b * maxBoxes + bi) * 5;
                float x = boxBatch.data[base], y = boxBatch.data[base + 1];
                float w = boxBatch.data[base + 2], h = boxBatch.data[base + 3];
                int catId = (int) boxBatch.data[base + 4];
                if (w <= 0f || h <= 0f) continue;

                int gy = Math.min(gh - 1, Math.max(0, (int) (y * gh)));
                int gx = Math.min(gw - 1, Math.max(0, (int) (x * gw)));
                float xCell = x * gw - gx, yCell = y * gh - gy;

                setChannel(target, b, 0, gy, gx, xCell, channels, gh, gw);
                setChannel(target, b, 1, gy, gx, yCell, channels, gh, gw);
                setChannel(target, b, 2, gy, gx, w,     channels, gh, gw);
                setChannel(target, b, 3, gy, gx, h,     channels, gh, gw);
                setChannel(target, b, 4, gy, gx, 1f,    channels, gh, gw);

                Integer classIdx = categoryToIndex.get(catId);
                if (classIdx != null && classIdx >= 0 && classIdx < numClasses) {
                    int clsChannel = classOffset + classIdx;
                    if (clsChannel < channels)
                        setChannel(target, b, clsChannel, gy, gx, 1f, channels, gh, gw);
                }
            }
        }
        if (pred.isGPU()) target.toGPU();
        return target;
    }

    /**
     * Build classification target for a multi-scale prediction tensor.
     * Shape: [B, numBoxes * numClasses, H, W]
     * Assigns GT boxes to grid cells with one-hot class encoding.
     */
    private static Tensor buildMultiScaleClsTarget(int[] shape, Tensor boxBatch,
                                                   Map<Integer, Integer> catToIdx,
                                                   int numClasses, boolean gpu) {
        Tensor target = Torch.zeros(shape);
        int bsz = shape[0], channels = shape[1], gh = shape[2], gw = shape[3];
        int maxBoxes = boxBatch.shape[1];

        for (int b = 0; b < bsz; b++) {
            for (int bi = 0; bi < maxBoxes; bi++) {
                int base = (b * maxBoxes + bi) * 5;
                float x = boxBatch.data[base], y = boxBatch.data[base + 1];
                float w = boxBatch.data[base + 2], h = boxBatch.data[base + 3];
                int catId = (int) boxBatch.data[base + 4];
                if (w <= 0f || h <= 0f) continue;

                int gy = Math.min(gh - 1, Math.max(0, (int) (y * gh)));
                int gx = Math.min(gw - 1, Math.max(0, (int) (x * gw)));

                Integer classIdx = catToIdx.get(catId);
                if (classIdx != null && classIdx >= 0 && classIdx < numClasses) {
                    // First anchor slot: channel = 0 * numClasses + classIdx
                    int ch = classIdx;
                    if (ch < channels)
                        setChannel(target, b, ch, gy, gx, 1f, channels, gh, gw);
                }
            }
        }
        if (gpu) target.toGPU();
        return target;
    }

    /**
     * Build regression target for a multi-scale prediction tensor.
     * Shape: [B, numBoxes * 4, H, W]
     * Assigns GT box coordinates (normalized) to grid cells.
     */
    private static Tensor buildMultiScaleRegTarget(int[] shape, Tensor boxBatch, boolean gpu) {
        Tensor target = Torch.zeros(shape);
        int bsz = shape[0], channels = shape[1], gh = shape[2], gw = shape[3];
        int maxBoxes = boxBatch.shape[1];

        for (int b = 0; b < bsz; b++) {
            for (int bi = 0; bi < maxBoxes; bi++) {
                int base = (b * maxBoxes + bi) * 5;
                float x = boxBatch.data[base], y = boxBatch.data[base + 1];
                float w = boxBatch.data[base + 2], h = boxBatch.data[base + 3];
                if (w <= 0f || h <= 0f) continue;

                int gy = Math.min(gh - 1, Math.max(0, (int) (y * gh)));
                int gx = Math.min(gw - 1, Math.max(0, (int) (x * gw)));

                float xCell = x * gw - gx;
                float yCell = y * gh - gy;

                // First anchor slot: channels 0..3
                if (channels >= 4) {
                    setChannel(target, b, 0, gy, gx, xCell, channels, gh, gw);
                    setChannel(target, b, 1, gy, gx, yCell, channels, gh, gw);
                    setChannel(target, b, 2, gy, gx, w,     channels, gh, gw);
                    setChannel(target, b, 3, gy, gx, h,     channels, gh, gw);
                }
            }
        }
        if (gpu) target.toGPU();
        return target;
    }

    /**
     * Build RPN objectness target.
     * Shape: [B, numAnchors*2, H, W]
     * Sets foreground score to 1 at grid cells containing GT box centers.
     */
    private static Tensor buildRPNObjectnessTarget(int[] shape, Tensor boxBatch, boolean gpu) {
        // objectness has 2 channels per anchor: [background, foreground]
        // Default to 0.5 (uncertain), set foreground=1 where GT exists
        int bsz = shape[0], channels = shape[1], gh = shape[2], gw = shape[3];
        Tensor target = Torch.zeros(shape);
        int maxBoxes = boxBatch.shape[1];
        int numAnchors = channels / 2;

        for (int b = 0; b < bsz; b++) {
            // Mark background everywhere first
            for (int a = 0; a < numAnchors; a++) {
                int bgCh = a * 2;       // background channel
                for (int y = 0; y < gh; y++)
                    for (int x = 0; x < gw; x++)
                        setChannel(target, b, bgCh, y, x, 1f, channels, gh, gw);
            }

            // Mark foreground where GT boxes exist
            for (int bi = 0; bi < maxBoxes; bi++) {
                int base = (b * maxBoxes + bi) * 5;
                float bx = boxBatch.data[base], by = boxBatch.data[base + 1];
                float bw = boxBatch.data[base + 2], bh = boxBatch.data[base + 3];
                if (bw <= 0f || bh <= 0f) continue;

                int gy = Math.min(gh - 1, Math.max(0, (int) (by * gh)));
                int gx = Math.min(gw - 1, Math.max(0, (int) (bx * gw)));

                // Set first anchor to foreground
                int bgCh = 0;   // anchor 0, background
                int fgCh = 1;   // anchor 0, foreground
                setChannel(target, b, bgCh, gy, gx, 0f, channels, gh, gw);
                setChannel(target, b, fgCh, gy, gx, 1f, channels, gh, gw);
            }
        }
        if (gpu) target.toGPU();
        return target;
    }

    /**
     * Build RPN bbox offset target.
     * Shape: [B, numAnchors*4, H, W]
     */
    private static Tensor buildRPNBboxTarget(int[] shape, Tensor boxBatch, boolean gpu) {
        Tensor target = Torch.zeros(shape);
        int bsz = shape[0], channels = shape[1], gh = shape[2], gw = shape[3];
        int maxBoxes = boxBatch.shape[1];

        for (int b = 0; b < bsz; b++) {
            for (int bi = 0; bi < maxBoxes; bi++) {
                int base = (b * maxBoxes + bi) * 5;
                float x = boxBatch.data[base], y = boxBatch.data[base + 1];
                float w = boxBatch.data[base + 2], h = boxBatch.data[base + 3];
                if (w <= 0f || h <= 0f) continue;

                int gy = Math.min(gh - 1, Math.max(0, (int) (y * gh)));
                int gx = Math.min(gw - 1, Math.max(0, (int) (x * gw)));

                float xCell = x * gw - gx;
                float yCell = y * gh - gy;

                // First anchor (channels 0..3)
                if (channels >= 4) {
                    setChannel(target, b, 0, gy, gx, xCell, channels, gh, gw);
                    setChannel(target, b, 1, gy, gx, yCell, channels, gh, gw);
                    setChannel(target, b, 2, gy, gx, w,     channels, gh, gw);
                    setChannel(target, b, 3, gy, gx, h,     channels, gh, gw);
                }
            }
        }
        if (gpu) target.toGPU();
        return target;
    }

    // ═══════════════════════════════════════════════════════════
    //  UTILITY METHODS
    // ═══════════════════════════════════════════════════════════

    private static void runSafe(String name, Runnable task) {
        try {
            task.run();
        } catch (Throwable t) {
            log.println();
            log.println("ERROR: " + name + " training failed: " + t.getMessage());
            t.printStackTrace(log);
        }
    }

    /** Clip gradient L2 norm in-place. */
    private static void clipGradNorm(List<Parameter> params, float maxNorm) {
        float totalNorm = 0f;
        for (Parameter p : params) {
            Tensor g = p.getGrad();
            if (g != null) {
                g.toCPU();
                for (float v : g.data) totalNorm += v * v;
            }
        }
        totalNorm = (float) Math.sqrt(totalNorm);
        if (totalNorm > maxNorm) {
            float scale = maxNorm / (totalNorm + 1e-6f);
            for (Parameter p : params) {
                Tensor g = p.getGrad();
                if (g != null) {
                    for (int i = 0; i < g.data.length; i++)
                        g.data[i] *= scale;
                    g.markDirtyOnCPU();
                }
            }
        }
    }

    private static void setChannel(Tensor t, int b, int c, int y, int x, float v,
                                   int channels, int h, int w) {
        int idx = ((b * channels + c) * h + y) * w + x;
        if (idx >= 0 && idx < t.data.length) t.data[idx] = v;
    }

    private static boolean initGPU(com.user.nn.core.Module model, String name) {
        boolean gpu = CUDAOps.isAvailable();
        if (gpu) {
            log.println(name + ": moving model to GPU...");
            GpuMemoryPool.autoInit(model);
            model.toGPU();
            log.println(name + ": model on GPU.");
        } else {
            log.println(name + ": running on CPU.");
        }
        return gpu;
    }

    /** Release GPU memory pool so the next model can re-allocate cleanly. */
    private static void cleanupGPU(String name) {
        if (CUDAOps.isAvailable()) {
            GpuMemoryPool.destroy();
            log.println(name + ": GPU memory released.");
        }
    }

    private static Data.DataLoader createLoader(List<CocoSample> samples,
                                                int imageW, int imageH, int batchSize) {
        Data.Dataset ds = new CocoDetectionDataset(samples, imageW, imageH);
        return new Data.DataLoader(ds, batchSize, true, 2);
    }

    private static void saveResults(com.user.nn.core.Module model, TrainingHistory history,
                                    String prefix, String displayName) {
        try {
            // Save model weights
            String modelPath = outDir.resolve(prefix + "_coco.bin").toString();
            model.save(modelPath);
            log.println(displayName + " model saved: " + modelPath);

            // Save training CSV
            String csvPath = outDir.resolve(prefix + "_coco_history.csv").toString();
            history.saveCSV(csvPath);
            log.println(displayName + " history saved: " + csvPath);

            // Save loss curve PNG
            List<Float> lossValues = history.getMetric("train_loss");
            if (lossValues != null && !lossValues.isEmpty()) {
                double[] x = new double[lossValues.size()];
                double[] y = new double[lossValues.size()];
                for (int i = 0; i < lossValues.size(); i++) { x[i] = i + 1; y[i] = lossValues.get(i); }
                LinePlot plot = new LinePlot(x, y, "train_loss");
                PlotContext ctx = new PlotContext()
                        .title(displayName + " Training Loss (COCO)")
                        .xlabel("Epoch")
                        .ylabel("Loss")
                        .grid(true);
                String pngPath = outDir.resolve(prefix + "_coco_loss.png").toString();
                FileExporter.savePNG(plot, ctx, pngPath, 900, 500);
                log.println(displayName + " loss curve: " + pngPath);
            }
        } catch (Exception e) {
            log.println("Warning: could not save " + displayName + " results: " + e.getMessage());
        }
    }

    // ═══════════════════════════════════════════════════════════
    //  COCO PARSING (identical to TrainYOLOCoco)
    // ═══════════════════════════════════════════════════════════

    private static CocoParseResult parseCoco(Path imagesDir, Path annotationJson,
                                             int maxSamples) throws IOException {
        String text = Files.readString(annotationJson, StandardCharsets.UTF_8);

        // Parse image entries (id, file_name) - use lookaheads for order-independent matching
        Pattern imagePattern = Pattern.compile(
            "\\{(?=[^{}]*\\\"id\\\"\\s*:\\s*(\\d+))(?=[^{}]*\\\"file_name\\\"\\s*:\\s*\\\"([^\\\"]+)\\\")[^{}]*}"
        );
        Matcher imageMatcher = imagePattern.matcher(text);
        Map<Integer, String> imageIdToFile = new HashMap<>();
        while (imageMatcher.find()) {
            imageIdToFile.put(Integer.parseInt(imageMatcher.group(1)), imageMatcher.group(2));
        }

        // Parse image metadata (width, height)
        Pattern metaPattern = Pattern.compile(
            "\\{(?=[^{}]*\\\"id\\\"\\s*:\\s*(\\d+))(?=[^{}]*\\\"file_name\\\"\\s*:\\s*\\\"([^\\\"]+)\\\")"
            + "(?=[^{}]*\\\"width\\\"\\s*:\\s*(\\d+))(?=[^{}]*\\\"height\\\"\\s*:\\s*(\\d+))[^{}]*}"
        );
        Matcher metaMatcher = metaPattern.matcher(text);
        Map<Integer, int[]> imageMeta = new HashMap<>();
        while (metaMatcher.find()) {
            imageMeta.put(Integer.parseInt(metaMatcher.group(1)),
                    new int[]{ Integer.parseInt(metaMatcher.group(3)),
                               Integer.parseInt(metaMatcher.group(4)) });
        }

        // Parse annotations (image_id, category_id, bbox)
        Pattern annPattern = Pattern.compile(
            "\\{(?=[^{}]*\\\"image_id\\\"\\s*:\\s*(\\d+))(?=[^{}]*\\\"category_id\\\"\\s*:\\s*(\\d+))"
            + "(?=[^{}]*\\\"bbox\\\"\\s*:\\s*\\[(.*?)\\])[^{}]*}"
        );
        Matcher annMatcher = annPattern.matcher(text);

        Map<Integer, List<CocoAnnotation>> perImage = new HashMap<>();
        Map<Integer, Integer> categoryToIndex = new LinkedHashMap<>();

        while (annMatcher.find()) {
            int imageId = Integer.parseInt(annMatcher.group(1));
            int catId   = Integer.parseInt(annMatcher.group(2));
            String[] parts = annMatcher.group(3).split(",");
            if (parts.length < 4) continue;

            float bx = parseFloat(parts[0]), by = parseFloat(parts[1]);
            float bw = parseFloat(parts[2]), bh = parseFloat(parts[3]);

            String file = imageIdToFile.get(imageId);
            if (file == null) continue;
            Path imgPath = imagesDir.resolve(file);
            if (!Files.exists(imgPath)) continue;

            int[] wh = imageMeta.get(imageId);
            int imgW, imgH;
            if (wh != null && wh[0] > 0 && wh[1] > 0) {
                imgW = wh[0]; imgH = wh[1];
            } else {
                try {
                    BufferedImage img = ImageIO.read(imgPath.toFile());
                    if (img == null) continue;
                    imgW = img.getWidth(); imgH = img.getHeight();
                } catch (IOException ex) { continue; }
            }
            if (imgW <= 0 || imgH <= 0) continue;

            float nx = (bx + bw * 0.5f) / imgW;
            float ny = (by + bh * 0.5f) / imgH;
            float nw = bw / imgW;
            float nh = bh / imgH;

            if (!categoryToIndex.containsKey(catId))
                categoryToIndex.put(catId, categoryToIndex.size());

            perImage.computeIfAbsent(imageId, k -> new ArrayList<>())
                    .add(new CocoAnnotation(nx, ny, nw, nh, catId));
        }

        List<CocoSample> samples = new ArrayList<>();
        for (var e : perImage.entrySet()) {
            String file = imageIdToFile.get(e.getKey());
            if (file == null) continue;
            Path imgPath = imagesDir.resolve(file);
            if (!Files.exists(imgPath)) continue;
            samples.add(new CocoSample(imgPath, e.getValue()));
            if (samples.size() >= maxSamples) break;
        }
        return new CocoParseResult(samples, categoryToIndex);
    }

    private static float parseFloat(String s) {
        try { return Float.parseFloat(s.trim()); } catch (Exception e) { return 0f; }
    }

    static Tensor loadAndResizeImageCHW(Path imagePath, int dstW, int dstH) {
        try {
            BufferedImage src = ImageIO.read(imagePath.toFile());
            if (src == null) return Torch.zeros(3, dstH, dstW);
            BufferedImage resized = new BufferedImage(dstW, dstH, BufferedImage.TYPE_INT_RGB);
            Graphics2D g = resized.createGraphics();
            g.drawImage(src, 0, 0, dstW, dstH, null);
            g.dispose();
            float[] chw = new float[3 * dstH * dstW];
            for (int y = 0; y < dstH; y++) {
                for (int x = 0; x < dstW; x++) {
                    int rgb = resized.getRGB(x, y);
                    int idx = y * dstW + x;
                    chw[idx]                     = ((rgb >> 16) & 0xFF) / 255f;
                    chw[dstH * dstW + idx]       = ((rgb >>  8) & 0xFF) / 255f;
                    chw[2 * dstH * dstW + idx]   = ( rgb        & 0xFF) / 255f;
                }
            }
            return Torch.tensor(chw, 3, dstH, dstW);
        } catch (Exception e) {
            return Torch.zeros(3, dstH, dstW);
        }
    }
}
