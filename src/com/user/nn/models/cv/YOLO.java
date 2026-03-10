package com.user.nn.models.cv;

import com.user.nn.core.*;
import com.user.nn.core.Module;
import com.user.nn.layers.*;
import com.user.nn.containers.*;
import com.user.nn.activations.LeakyReLU;
import com.user.nn.pooling.MaxPool2d;
import java.util.*;

/**
 * YOLO (You Only Look Once) - Fast one-stage object detector.
 * 
 * This implementation follows YOLOv1 architecture for simplicity.
 * 
 * Key Ideas:
 *   - Divide image into S×S grid
 *   - Each grid cell predicts B bounding boxes and confidence scores
 *   - Each grid cell also predicts C class probabilities
 *   - Output tensor: [S, S, B*5 + C] where 5 = (x, y, w, h, confidence)
 * 
 * Advantages:
 *   - Very fast (45+ FPS)
 *   - Processes entire image in one pass
 *   - Good for real-time detection
 * 
 * Disadvantages:
 *   - Struggles with small objects
 *   - Lower accuracy than two-stage detectors
 *   - Each cell can only predict one class
 * 
 * Reference: "You Only Look Once: Unified, Real-Time Object Detection" (2016)
 * https://arxiv.org/abs/1506.02640
 */
public class YOLO extends Module {
    
    private Sequential backbone;
    private Sequential detectionHead;
    
    private int gridSize;      // S: image divided into S×S grid
    private int numBoxes;      // B: number of bounding boxes per grid cell
    private int numClasses;    // C: number of object classes
    private int imageH;
    private int imageW;
    
    /**
     * Create YOLO v1 model.
     * 
     * @param numClasses Number of object classes (e.g., 20 for PASCAL VOC)
     * @param imageH Input image height (typically 448)
     * @param imageW Input image width (typically 448)
     * @param gridSize Grid size (typically 7)
     * @param numBoxes Number of boxes per grid cell (typically 2)
     */
    public YOLO(int numClasses, int imageH, int imageW, int gridSize, int numBoxes) {
        this.numClasses = numClasses;
        this.imageH = imageH;
        this.imageW = imageW;
        this.gridSize = gridSize;
        this.numBoxes = numBoxes;
        
        // Build backbone (simplified from original YOLO)
        this.backbone = buildBackbone(imageH, imageW);
        addModule("backbone", backbone);
        
        // Detection head
        // Final feature map size: gridSize × gridSize
        // Output: [gridSize, gridSize, numBoxes*5 + numClasses]
        int outputChannels = numBoxes * 5 + numClasses;
        this.detectionHead = buildDetectionHead(gridSize, gridSize, outputChannels);
        addModule("detection_head", detectionHead);
    }
    
    /**
     * Simplified YOLO backbone inspired by VGG and GoogLeNet.
     * 
     * Original YOLO uses:
     * - 24 convolutional layers
     * - 2 fully connected layers
     * 
     * This is a simplified version with fewer layers for educational purposes.
     */
    private Sequential buildBackbone(int inH, int inW) {
        Sequential net = new Sequential();
        
        int h = inH, w = inW;
        
        // Block 1: Conv + Pool
        net.add(new Conv2d(3, 64, 7, 7, h, w, 2, 3, false));
        net.add(new LeakyReLU(0.1f));
        h = (h + 2*3 - 7) / 2 + 1;
        w = (w + 2*3 - 7) / 2 + 1;
        
        net.add(new MaxPool2d(2, 2, 2, 2, 0, 0, 64, h, w));
        h = (h - 2) / 2 + 1; w = (w - 2) / 2 + 1;
        
        // Block 2: Conv + Pool
        net.add(new Conv2d(64, 192, 3, 3, h, w, 1, 1, false));
        net.add(new LeakyReLU(0.1f));
        h = (h + 2*1 - 3) / 1 + 1;
        w = (w + 2*1 - 3) / 1 + 1;
        
        net.add(new MaxPool2d(2, 2, 2, 2, 0, 0, 192, h, w));
        h = (h - 2) / 2 + 1; w = (w - 2) / 2 + 1;
        
        // Block 3: Multiple convs + Pool
        net.add(new Conv2d(192, 128, 1, 1, h, w, 1, 0, false));
        net.add(new LeakyReLU(0.1f));
        
        net.add(new Conv2d(128, 256, 3, 3, h, w, 1, 1, false));
        net.add(new LeakyReLU(0.1f));
        h = (h + 2*1 - 3) / 1 + 1;
        w = (w + 2*1 - 3) / 1 + 1;
        
        net.add(new Conv2d(256, 256, 1, 1, h, w, 1, 0, false));
        net.add(new LeakyReLU(0.1f));
        
        net.add(new Conv2d(256, 512, 3, 3, h, w, 1, 1, false));
        net.add(new LeakyReLU(0.1f));
        h = (h + 2*1 - 3) / 1 + 1;
        w = (w + 2*1 - 3) / 1 + 1;
        
        net.add(new MaxPool2d(2, 2, 2, 2, 0, 0, 512, h, w));
        h = (h - 2) / 2 + 1; w = (w - 2) / 2 + 1;
        
        // Block 4: More convs + Pool
        for (int i = 0; i < 4; i++) {
            net.add(new Conv2d(512, 256, 1, 1, h, w, 1, 0, false));
            net.add(new LeakyReLU(0.1f));
            
            net.add(new Conv2d(256, 512, 3, 3, h, w, 1, 1, false));
            net.add(new LeakyReLU(0.1f));
            h = (h + 2*1 - 3) / 1 + 1;
            w = (w + 2*1 - 3) / 1 + 1;
        }
        
        net.add(new Conv2d(512, 512, 1, 1, h, w, 1, 0, false));
        net.add(new LeakyReLU(0.1f));
        
        net.add(new Conv2d(512, 1024, 3, 3, h, w, 1, 1, false));
        net.add(new LeakyReLU(0.1f));
        h = (h + 2*1 - 3) / 1 + 1;
        w = (w + 2*1 - 3) / 1 + 1;
        
        net.add(new MaxPool2d(2, 2, 2, 2, 0, 0, 1024, h, w));
        h = (h - 2) / 2 + 1; w = (w - 2) / 2 + 1;
        
        // Block 5: Final convs
        for (int i = 0; i < 2; i++) {
            net.add(new Conv2d(1024, 512, 1, 1, h, w, 1, 0, false));
            net.add(new LeakyReLU(0.1f));
            
            net.add(new Conv2d(512, 1024, 3, 3, h, w, 1, 1, false));
            net.add(new LeakyReLU(0.1f));
            h = (h + 2*1 - 3) / 1 + 1;
            w = (w + 2*1 - 3) / 1 + 1;
        }
        
        net.add(new Conv2d(1024, 1024, 3, 3, h, w, 1, 1, false));
        net.add(new LeakyReLU(0.1f));
        h = (h + 2*1 - 3) / 1 + 1;
        w = (w + 2*1 - 3) / 1 + 1;
        
        // Ensure output is gridSize × gridSize
        net.add(new Conv2d(1024, 1024, 3, 3, h, w, 2, 1, false));
        net.add(new LeakyReLU(0.1f));
        
        return net;
    }
    
    /**
     * Build detection head to produce final predictions.
     */
    private Sequential buildDetectionHead(int h, int w, int outputChannels) {
        Sequential head = new Sequential();
        
        // Additional convolutions
        head.add(new Conv2d(1024, 1024, 3, 3, h, w, 1, 1, false));
        head.add(new LeakyReLU(0.1f));
        
        head.add(new Conv2d(1024, 1024, 3, 3, h, w, 1, 1, false));
        head.add(new LeakyReLU(0.1f));
        
        // Final prediction layer
        head.add(new Conv2d(1024, outputChannels, 1, 1, h, w, 1, 0, true));
        
        return head;
    }
    
    /**
     * Forward pass.
     * 
     * @param images Input images [B, 3, H, W]
     * @return Predictions [B, gridSize, gridSize, numBoxes*5 + numClasses]
     */
    @Override
    public Tensor forward(Tensor images) {
        // Extract features
        Tensor features = backbone.forward(images);
        
        // Detection predictions
        Tensor predictions = detectionHead.forward(features);
        
        // Reshape to [B, gridSize, gridSize, numBoxes*5 + numClasses]
        int B = predictions.shape[0];
        int C = predictions.shape[1];
        int H = predictions.shape[2];
        int W = predictions.shape[3];
        
        // Permute from [B, C, H, W] to [B, H, W, C]
        // For simplicity, return as-is (user can reshape if needed)
        return predictions;
    }
    
    /**
     * Decode predictions to bounding boxes.
     * 
     * @param predictions Raw model output [B, gridSize, gridSize, numBoxes*5 + numClasses]
     * @param confidenceThresh Confidence threshold for filtering
     * @return List of detected boxes per image: [x1, y1, x2, y2, class_id, confidence]
     */
    public List<List<float[]>> decode(Tensor predictions, float confidenceThresh) {
        int B = predictions.shape[0];
        List<List<float[]>> allDetections = new ArrayList<>();
        
        for (int b = 0; b < B; b++) {
            List<float[]> imageDetections = new ArrayList<>();
            
            // Process each grid cell
            for (int i = 0; i < gridSize; i++) {
                for (int j = 0; j < gridSize; j++) {
                    // Get predictions for this cell
                    // Format: [box1_x, box1_y, box1_w, box1_h, box1_conf, 
                    //          box2_x, box2_y, box2_w, box2_h, box2_conf,
                    //          class_probs...]
                    
                    int baseIdx = ((b * gridSize + i) * gridSize + j) * (numBoxes * 5 + numClasses);
                    
                    // Process each bounding box
                    for (int boxIdx = 0; boxIdx < numBoxes; boxIdx++) {
                        int offset = boxIdx * 5;
                        
                        // Extract box parameters (relative to cell)
                        float x = predictions.data[baseIdx + offset + 0]; // [0, 1]
                        float y = predictions.data[baseIdx + offset + 1]; // [0, 1]
                        float w = predictions.data[baseIdx + offset + 2]; // [0, 1]
                        float h = predictions.data[baseIdx + offset + 3]; // [0, 1]
                        float conf = predictions.data[baseIdx + offset + 4]; // [0, 1]
                        
                        if (conf < confidenceThresh) continue;
                        
                        // Convert to absolute coordinates
                        float absX = (j + x) / gridSize * imageW;
                        float absY = (i + y) / gridSize * imageH;
                        float absW = w * imageW;
                        float absH = h * imageH;
                        
                        // Convert to corner format [x1, y1, x2, y2]
                        float x1 = absX - absW / 2;
                        float y1 = absY - absH / 2;
                        float x2 = absX + absW / 2;
                        float y2 = absY + absH / 2;
                        
                        // Find best class
                        int classOffset = numBoxes * 5;
                        float maxClassProb = 0;
                        int bestClass = 0;
                        
                        for (int c = 0; c < numClasses; c++) {
                            float classProb = predictions.data[baseIdx + classOffset + c];
                            if (classProb > maxClassProb) {
                                maxClassProb = classProb;
                                bestClass = c;
                            }
                        }
                        
                        float finalScore = conf * maxClassProb;
                        
                        if (finalScore >= confidenceThresh) {
                            imageDetections.add(new float[]{x1, y1, x2, y2, bestClass, finalScore});
                        }
                    }
                }
            }
            
            // Apply NMS
            imageDetections = applyNMS(imageDetections, 0.5f);
            allDetections.add(imageDetections);
        }
        
        return allDetections;
    }
    
    /**
     * Apply Non-Maximum Suppression to remove duplicate detections.
     */
    private List<float[]> applyNMS(List<float[]> detections, float iouThresh) {
        if (detections.isEmpty()) return detections;
        
        // Sort by confidence (descending)
        detections.sort((a, b) -> Float.compare(b[5], a[5]));
        
        List<float[]> keep = new ArrayList<>();
        boolean[] suppressed = new boolean[detections.size()];
        
        for (int i = 0; i < detections.size(); i++) {
            if (suppressed[i]) continue;
            
            float[] box1 = detections.get(i);
            keep.add(box1);
            
            // Suppress overlapping boxes of the same class
            for (int j = i + 1; j < detections.size(); j++) {
                if (suppressed[j]) continue;
                
                float[] box2 = detections.get(j);
                
                // Only compare boxes of the same class
                if (box1[4] != box2[4]) continue;
                
                float iou = computeIOU(box1, box2);
                if (iou > iouThresh) {
                    suppressed[j] = true;
                }
            }
        }
        
        return keep;
    }
    
    /**
     * Compute IOU between two boxes [x1, y1, x2, y2, ...].
     */
    private float computeIOU(float[] box1, float[] box2) {
        float x1_1 = box1[0], y1_1 = box1[1], x2_1 = box1[2], y2_1 = box1[3];
        float x1_2 = box2[0], y1_2 = box2[1], x2_2 = box2[2], y2_2 = box2[3];
        
        float interX1 = Math.max(x1_1, x1_2);
        float interY1 = Math.max(y1_1, y1_2);
        float interX2 = Math.min(x2_1, x2_2);
        float interY2 = Math.min(y2_1, y2_2);
        
        float interArea = Math.max(0, interX2 - interX1) * Math.max(0, interY2 - interY1);
        
        float area1 = (x2_1 - x1_1) * (y2_1 - y1_1);
        float area2 = (x2_2 - x1_2) * (y2_2 - y1_2);
        float unionArea = area1 + area2 - interArea;
        
        return interArea / (unionArea + 1e-6f);
    }
    
    /**
     * Compute YOLO loss (simplified version).
     * 
     * YOLO loss has 5 components:
     * 1. Localization loss (x, y)
     * 2. Size loss (w, h)
     * 3. Confidence loss (object present)
     * 4. Confidence loss (no object)
     * 5. Classification loss
     */
    public Tensor computeLoss(Tensor predictions, Tensor targets) {
        // TODO: Implement proper YOLO loss
        // This requires matching predictions to ground truth boxes
        // and applying different loss weights to different components
        
        return Torch.tensor(new float[]{0.0f}, 1);
    }
}
