# Object Detection Models

## Overview

Comprehensive implementation of state-of-the-art object detection architectures for the ML framework. Includes both **two-stage** (accuracy-focused) and **one-stage** (speed-focused) detectors.

All models are integrated with:
- ✅ Progress bars (tqdm-like) for training visualization
- ✅ Plotting utilities (matplotlib-like) for performance tracking
- ✅ Modular design following PyTorch patterns

---

## Model Architectures

### Two-Stage Detectors (Accuracy Priority)

#### 1. Faster R-CNN
**File**: `src/com/user/nn/models/cv/FasterRCNN.java`

**Architecture**:
```
Input Image → Backbone (ResNet/VGG)
           → RPN (Region Proposals)
           → ROI Pooling
           → Detection Head (Classification + BBox Regression)
```

**Key Components**:
- **RPN (Region Proposal Network)**: Generates object proposals using anchor boxes
- **ROI Pooling**: Extracts fixed-size features from variable-sized regions
- **Detection Head**: Two-branch network for classification and localization

**Strengths**:
- ✅ High accuracy (mAP ~78% on PASCAL VOC)
- ✅ Excellent localization quality
- ✅ End-to-end trainable
- ✅ Shares features between RPN and detector

**Weaknesses**:
- ❌ Slower inference (~5-10 FPS)
- ❌ Complex training procedure
- ❌ Memory intensive

**Use Cases**:
- Autonomous vehicles (high accuracy critical)
- Medical image analysis
- Satellite/aerial imagery
- Any application where accuracy > speed

**API Example**:
```java
// Create model
FasterRCNN model = FasterRCNN.withResNet50(numClasses, 600, 600);

// Forward pass
Map<String, Tensor> outputs = model.forward(images);
Tensor proposals = outputs.get("proposals");
Tensor features = outputs.get("features");

// Inference
List<Tensor> detections = model.detect(images, 0.5f, 0.5f);
```

---

### One-Stage Detectors (Speed Priority)

#### 2. YOLO (You Only Look Once)
**File**: `src/com/user/nn/models/cv/YOLO.java`

**Architecture**:
```
Input Image → Backbone (Darknet-inspired)
           → Grid Predictions [S×S×(B*5 + C)]
           → NMS → Final Detections
```

**Key Concepts**:
- **Grid-based**: Divides image into S×S grid (typically 7×7)
- **Direct prediction**: Each cell predicts B bounding boxes + confidence
- **Single forward pass**: Entire image processed at once

**Strengths**:
- ✅ Very fast (45+ FPS on GPU)
- ✅ Real-time capable
- ✅ Simple architecture
- ✅ Good for video streams

**Weaknesses**:
- ❌ Struggles with small objects
- ❌ Limited to one object per grid cell
- ❌ Lower accuracy than two-stage methods

**Use Cases**:
- Real-time surveillance
- Robotics and autonomous systems
- Mobile/edge devices
- Live video analysis

**API Example**:
```java
// Create YOLO v1
YOLO model = new YOLO(numClasses, 448, 448, 7, 2);

// Forward pass
Tensor predictions = model.forward(images);

// Decode predictions
List<List<float[]>> detections = model.decode(predictions, 0.25f);
// Each detection: [x1, y1, x2, y2, class_id, confidence]
```

---

#### 3. SSD (Single Shot MultiBox Detector)
**File**: `src/com/user/nn/models/cv/SSD.java`

**Architecture**:
```
Input → VGG Backbone → Feature Map 1 → Predictions
                    ↓
                    Extra Layers → Feature Map 2 → Predictions
                                 ↓
                                 ... → Feature Map N → Predictions
```

**Key Concepts**:
- **Multi-scale prediction**: Predicts on 6 different feature map sizes
- **Default boxes**: Similar to anchors, multiple scales and aspect ratios per location
- **Progressive downsampling**: Handles objects at different scales

**Strengths**:
- ✅ Good accuracy/speed trade-off (74% mAP at 59 FPS)
- ✅ Better at detecting objects at multiple scales than YOLO v1
- ✅ Flexible input size (SSD300, SSD512)

**Weaknesses**:
- ❌ Still struggles with very small objects
- ❌ Requires careful tuning of default boxes
- ❌ Hard negative mining needed during training

**Use Cases**:
- Embedded systems with moderate compute
- Drone/aerial detection
- Applications needing balance between speed and accuracy

**API Example**:
```java
// Create SSD300
SSD model = SSD.ssd300(numClasses);

// Or SSD512 for better accuracy
SSD model512 = SSD.ssd512(numClasses);

// Multi-scale forward pass
Map<String, Object> outputs = model.forwardMultiScale(images);
List<Tensor> classifications = (List<Tensor>) outputs.get("classifications");
List<Tensor> regressions = (List<Tensor>) outputs.get("regressions");

// Decode
List<List<float[]>> detections = model.decode(classifications, regressions, 0.5f, 0.45f);
```

---

#### 4. RetinaNet
**File**: `src/com/user/nn/models/cv/RetinaNet.java`

**Architecture**:
```
Input → Backbone (ResNet) → FPN (Feature Pyramid Network)
                          ↓
                    P3, P4, P5, P6, P7 (Multi-scale features)
                          ↓
              Classification Subnet + Box Regression Subnet
                          ↓
                    Focal Loss Training
```

**Key Innovations**:
1. **FPN (Feature Pyramid Network)**:
   - Bottom-up: Standard convolutions
   - Top-down: Upsampling + lateral connections
   - Creates rich multi-scale features

2. **Focal Loss**:
   - Addresses extreme class imbalance (1:1000 foreground:background)
   - Formula: `FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)`
   - Down-weights easy examples, focuses on hard ones

**Strengths**:
- ✅ Two-stage accuracy at one-stage speed (77% mAP at 19 FPS)
- ✅ Excellent multi-scale detection
- ✅ Elegant solution to class imbalance
- ✅ State-of-the-art when introduced

**Weaknesses**:
- ❌ More complex than YOLO/SSD
- ❌ Slower than YOLO v3+
- ❌ Requires careful anchor design

**Use Cases**:
- Cloud-based detection services
- Research and benchmarking
- Applications needing high accuracy without two-stage complexity

**API Example**:
```java
// Create RetinaNet with ResNet backbone
ResNet backbone = ResNet.resnet50(1000, 800, 800);
int[] backboneChannels = {256, 512, 1024, 2048};
int[] featureSizes = {200, 100, 50, 25};

RetinaNet model = new RetinaNet(backbone, numClasses, backboneChannels, featureSizes);

// Forward with FPN
Map<String, Object> outputs = model.forwardMultiScale(images);

// Compute focal loss
Map<String, Float> losses = model.computeLoss(
    (List<Tensor>) outputs.get("classifications"),
    (List<Tensor>) outputs.get("regressions"),
    gtBoxes, gtLabels
);
```

---

## Supporting Components

### RPN (Region Proposal Network)
**File**: `src/com/user/nn/models/cv/RPN.java`

Generates region proposals for Faster R-CNN:
- Slides 3×3 window over feature maps
- Predicts objectness (object vs background) at each location
- Predicts bounding box refinements
- Uses anchor boxes at multiple scales and aspect ratios

```java
RPN rpn = new RPN(inChannels, featureH, featureW, 
                  new float[]{128, 256, 512},     // scales
                  new float[]{0.5f, 1.0f, 2.0f}); // aspect ratios

Map<String, Tensor> outputs = rpn.forward(features);
Tensor objectness = outputs.get("objectness");
Tensor bboxDeltas = outputs.get("bbox_deltas");
Tensor anchors = outputs.get("anchors");
```

### ROI Pooling
**File**: `src/com/user/nn/layers/ROIPooling.java`

Extracts fixed-size features from variable-sized regions:
- Input: Feature maps [B, C, H, W] + ROIs [N, 5]
- Output: Pooled features [N, C, pool_h, pool_w]
- Uses max pooling within each bin

```java
ROIPooling roiPool = new ROIPooling(7, 7, 1.0f/16.0f);
Tensor pooledFeatures = roiPool.forward(featureMaps, rois);
```

### Focal Loss
**File**: `src/com/user/nn/losses/FocalLoss.java`

Addresses class imbalance in object detection:
- Standard cross-entropy: `CE = -log(p_t)`
- Focal loss: `FL = -(1 - p_t)^γ * log(p_t)`
- Easy examples (p_t → 1) contribute less
- Hard examples (p_t → 0) dominate learning

```java
FocalLoss focalLoss = new FocalLoss(0.25f, 2.0f, "mean");

// Binary classification
Tensor loss = focalLoss.forwardBinary(logits, targets);

// Multi-class classification
Tensor loss = focalLoss.forwardMultiClass(logits, targetIndices);
```

---

## Model Comparison

| Model | Type | Speed (FPS) | mAP | Strengths | Best For |
|-------|------|-------------|-----|-----------|----------|
| **Faster R-CNN** | Two-stage | 5-10 | ~78% | Highest accuracy | Critical applications |
| **YOLO v1** | One-stage | 45+ | ~63% | Fastest | Real-time video |
| **SSD300** | One-stage | 59 | ~74% | Balanced | Embedded systems |
| **RetinaNet** | One-stage | 19 | ~77% | FPN + Focal Loss | Cloud inference |

### Speed vs Accuracy Trade-off

```
Accuracy (mAP)
    ↑
78% |  Faster R-CNN ●
    |                    ● RetinaNet
74% |              ● SSD300
    |
63% |                              ● YOLO v1
    |
    └────────────────────────────────────→ Speed (FPS)
        5          19        59         45+
```

---

## Integration with Progress & Visualization

All models support the framework's progress tracking and visualization:

### Training with Progress Bars

```java
import com.user.nn.utils.progress.*;
import com.user.nn.utils.visualization.*;

// Wrap DataLoader
ProgressDataLoader progLoader = new ProgressDataLoader(
    loader, "Training YOLO"
);

// Training loop
for (Tensor[] batch : progLoader) {
    // ... forward, backward, optimize ...
    
    progLoader.setPostfix("loss", currentLoss);
    progLoader.setPostfix("mAP", currentMAP);
}
```

### Visualizing Training

```java
TrainingHistory history = new TrainingHistory();

for (int epoch = 0; epoch < epochs; epoch++) {
    // Training...
    
    Map<String, Float> metrics = new HashMap<>();
    metrics.put("train_loss", trainLoss);
    metrics.put("val_mAP", valMAP);
    history.record(epoch, metrics);
}

// Plot curves
Plot curves = history.plot();
FileExporter.savePNG(curves, "detection_training.png");

// Get best epoch
float bestMAP = history.getMax("val_mAP");
int bestEpoch = history.getMaxEpoch("val_mAP");
```

---

## Running Examples

### Demo All Models
```bash
./gradlew :core:run -PmainClass=com.user.nn.examples.ObjectDetectionDemo
```

### Train Specific Model
```bash
# Training examples not yet implemented
# Requires object detection dataset with bounding box annotations
```

---

## File Structure

```
src/com/user/nn/
├── models/cv/
│   ├── FasterRCNN.java       # Two-stage detector with RPN
│   ├── RPN.java              # Region Proposal Network
│   ├── YOLO.java             # YOLOv1 implementation
│   ├── SSD.java              # Single Shot MultiBox Detector
│   └── RetinaNet.java        # RetinaNet with FPN
├── layers/
│   └── ROIPooling.java       # ROI Pooling layer
├── losses/
│   └── FocalLoss.java        # Focal Loss for class imbalance
└── examples/
    └── ObjectDetectionDemo.java  # Comprehensive demo
```

---

## Future Enhancements

### Model Extensions
- [ ] YOLOv3/v4/v5 with better backbones and multi-scale
- [ ] Mask R-CNN for instance segmentation
- [ ] EfficientDet with compound scaling
- [ ] DETR (transformer-based detection)

### Training Features
- [ ] Object detection dataset loaders (COCO, Pascal VOC)
- [ ] Evaluation metrics (mAP, IoU, precision-recall curves)
- [ ] Data augmentation for detection (crop, flip, mixup)
- [ ] Multi-GPU training support

### Components
- [ ] ROI Align (improved version of ROI Pooling)
- [ ] Deformable convolutions
- [ ] Feature Pyramid Network improvements
- [ ] Better NMS algorithms (Soft-NMS, DIoU-NMS)

---

## References

1. **Faster R-CNN**: Ren et al., "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks" (NeurIPS 2015)

2. **YOLO**: Redmon et al., "You Only Look Once: Unified, Real-Time Object Detection" (CVPR 2016)

3. **SSD**: Liu et al., "SSD: Single Shot MultiBox Detector" (ECCV 2016)

4. **RetinaNet**: Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017)

---

## Status

✅ **Core Implementation Complete**
- All 4 major architectures implemented
- Supporting components (RPN, ROI Pooling, Focal Loss)
- Integration with progress bars and visualization
- Comprehensive demo and documentation

🚧 **Future Work**
- Full training examples with real datasets
- Evaluation metrics implementation
- Advanced model variants (YOLOv3+, Mask R-CNN, etc.)

---

**Created**: March 2026  
**Framework Version**: Compatible with ML Framework v1.0+  
**Author**: ML Framework Team
