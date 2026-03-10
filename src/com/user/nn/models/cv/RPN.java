package com.user.nn.models.cv;

import com.user.nn.core.*;
import com.user.nn.core.Module;
import com.user.nn.layers.*;
import com.user.nn.activations.*;
import java.util.*;

/**
 * Region Proposal Network (RPN) for Faster R-CNN.
 * 
 * Generates region proposals (potential object locations) from feature maps.
 * Uses anchor boxes at multiple scales and aspect ratios.
 * 
 * Architecture:
 *   1. 3x3 conv on feature maps (intermediate layer)
 *   2. Two parallel 1x1 convs:
 *      - Classification: objectness score (object vs background)
 *      - Regression: bounding box refinement (4 coordinates per anchor)
 * 
 * Reference: "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks"
 */
public class RPN extends Module {
    
    private Conv2d conv;
    private Conv2d clsLayer;  // objectness classification
    private Conv2d bboxLayer; // bounding box regression
    
    private int numAnchors;     // number of anchor boxes per location
    private int featureH;       // feature map height
    private int featureW;       // feature map width
    private int featureChannels;
    
    // Anchor generation parameters
    private float[] scales;      // e.g., [128, 256, 512] pixels
    private float[] aspectRatios; // e.g., [0.5, 1.0, 2.0]
    
    /**
     * Create RPN.
     * 
     * @param inChannels Number of input feature channels (e.g., 512 for VGG16)
     * @param featureH Feature map height
     * @param featureW Feature map width
     * @param scales Anchor scales in pixels (e.g., new float[]{128, 256, 512})
     * @param aspectRatios Anchor aspect ratios (e.g., new float[]{0.5f, 1.0f, 2.0f})
     */
    public RPN(int inChannels, int featureH, int featureW, float[] scales, float[] aspectRatios) {
        this.featureH = featureH;
        this.featureW = featureW;
        this.featureChannels = inChannels;
        this.scales = scales;
        this.aspectRatios = aspectRatios;
        this.numAnchors = scales.length * aspectRatios.length;
        
        // Intermediate 3x3 conv
        int intermediateChannels = 512;
        this.conv = new Conv2d(inChannels, intermediateChannels, 3, 3, featureH, featureW, 
                              1, 1, false);
        addModule("conv", conv);
        
        // Objectness classification: 2 * numAnchors (background, foreground)
        this.clsLayer = new Conv2d(intermediateChannels, numAnchors * 2, 1, 1, 
                                   featureH, featureW, 1, 0, true);
        addModule("cls_layer", clsLayer);
        
        // Bounding box regression: 4 * numAnchors (dx, dy, dw, dh)
        this.bboxLayer = new Conv2d(intermediateChannels, numAnchors * 4, 1, 1,
                                    featureH, featureW, 1, 0, true);
        addModule("bbox_layer", bboxLayer);
    }
    
    /**
     * Forward pass.
     * 
     * @param features Feature maps from backbone [B, C, H, W]
     * @return Map with keys:
     *         - "objectness": [B, numAnchors*2, H, W] - objectness scores
     *         - "bbox_deltas": [B, numAnchors*4, H, W] - bbox refinements
     *         - "anchors": [H*W*numAnchors, 4] - generated anchors [x1, y1, x2, y2]
     */
    public Map<String, Tensor> forwardRPN(Tensor features) {
        int B = features.shape[0];
        
        // Intermediate feature
        Tensor x = conv.forward(features);
        x = Torch.relu(x);
        
        // Objectness scores: [B, numAnchors*2, H, W]
        Tensor objectness = clsLayer.forward(x);
        
        // Bounding box deltas: [B, numAnchors*4, H, W]
        Tensor bboxDeltas = bboxLayer.forward(x);
        
        // Generate anchors (only once, independent of batch)
        Tensor anchors = generateAnchors();
        
        Map<String, Tensor> outputs = new HashMap<>();
        outputs.put("objectness", objectness);
        outputs.put("bbox_deltas", bboxDeltas);
        outputs.put("anchors", anchors);
        
        return outputs;
    }

    @Override
    public Tensor forward(Tensor features) {
        return forwardRPN(features).get("objectness");
    }
    
    /**
     * Generate anchor boxes at all spatial locations.
     * 
     * @return Tensor of shape [H*W*numAnchors, 4] with coordinates [x1, y1, x2, y2]
     */
    private Tensor generateAnchors() {
        int totalAnchors = featureH * featureW * numAnchors;
        float[] anchorData = new float[totalAnchors * 4];
        
        int stride = 16; // typical stride from input image to feature map
        int anchorIdx = 0;
        
        // For each spatial location
        for (int y = 0; y < featureH; y++) {
            for (int x = 0; x < featureW; x++) {
                // Center of this anchor location in input image coordinates
                float centerX = x * stride + stride / 2.0f;
                float centerY = y * stride + stride / 2.0f;
                
                // For each scale and aspect ratio
                for (float scale : scales) {
                    for (float ratio : aspectRatios) {
                        // Calculate anchor box dimensions
                        float w = scale * (float) Math.sqrt(ratio);
                        float h = scale / (float) Math.sqrt(ratio);
                        
                        // Convert to [x1, y1, x2, y2]
                        anchorData[anchorIdx * 4 + 0] = centerX - w / 2;  // x1
                        anchorData[anchorIdx * 4 + 1] = centerY - h / 2;  // y1
                        anchorData[anchorIdx * 4 + 2] = centerX + w / 2;  // x2
                        anchorData[anchorIdx * 4 + 3] = centerY + h / 2;  // y2
                        
                        anchorIdx++;
                    }
                }
            }
        }
        
        return Torch.tensor(anchorData, totalAnchors, 4);
    }
    
    /**
     * Apply bounding box deltas to anchors to get proposals.
     * 
     * @param anchors Base anchors [N, 4]
     * @param deltas Predicted deltas [N, 4] (dx, dy, dw, dh)
     * @return Refined boxes [N, 4]
     */
    public static Tensor applyDeltas(Tensor anchors, Tensor deltas) {
        int N = anchors.shape[0];
        float[] proposals = new float[N * 4];
        
        for (int i = 0; i < N; i++) {
            // Anchor box
            float x1 = anchors.data[i * 4 + 0];
            float y1 = anchors.data[i * 4 + 1];
            float x2 = anchors.data[i * 4 + 2];
            float y2 = anchors.data[i * 4 + 3];
            
            float anchorW = x2 - x1;
            float anchorH = y2 - y1;
            float anchorCx = x1 + anchorW / 2;
            float anchorCy = y1 + anchorH / 2;
            
            // Deltas
            float dx = deltas.data[i * 4 + 0];
            float dy = deltas.data[i * 4 + 1];
            float dw = deltas.data[i * 4 + 2];
            float dh = deltas.data[i * 4 + 3];
            
            // Apply transformations
            float predCx = anchorCx + dx * anchorW;
            float predCy = anchorCy + dy * anchorH;
            float predW = anchorW * (float) Math.exp(dw);
            float predH = anchorH * (float) Math.exp(dh);
            
            // Convert back to [x1, y1, x2, y2]
            proposals[i * 4 + 0] = predCx - predW / 2;
            proposals[i * 4 + 1] = predCy - predH / 2;
            proposals[i * 4 + 2] = predCx + predW / 2;
            proposals[i * 4 + 3] = predCy + predH / 2;
        }
        
        return Torch.tensor(proposals, N, 4);
    }
    
    /**
     * Filter proposals by score and apply NMS (Non-Maximum Suppression).
     * 
     * @param proposals Bounding boxes [N, 4]
     * @param scores Objectness scores [N]
     * @param scoreThresh Minimum score threshold
     * @param nmsThresh IOU threshold for NMS
     * @param maxProposals Maximum number of proposals to keep
     * @return Filtered proposals
     */
    public static Tensor filterProposals(Tensor proposals, Tensor scores, 
                                        float scoreThresh, float nmsThresh, int maxProposals) {
        int N = proposals.shape[0];
        List<Integer> keepIndices = new ArrayList<>();
        
        // Filter by score threshold
        for (int i = 0; i < N; i++) {
            if (scores.data[i] >= scoreThresh) {
                keepIndices.add(i);
            }
        }
        
        // Sort by score (descending)
        keepIndices.sort((a, b) -> Float.compare(scores.data[b], scores.data[a]));
        
        // Apply NMS
        List<Integer> finalIndices = new ArrayList<>();
        boolean[] suppressed = new boolean[keepIndices.size()];
        
        for (int i = 0; i < keepIndices.size(); i++) {
            if (suppressed[i]) continue;
            
            int idx = keepIndices.get(i);
            finalIndices.add(idx);
            
            if (finalIndices.size() >= maxProposals) break;
            
            // Suppress overlapping boxes
            for (int j = i + 1; j < keepIndices.size(); j++) {
                if (suppressed[j]) continue;
                
                int idx2 = keepIndices.get(j);
                float iou = computeIOU(proposals, idx, idx2);
                
                if (iou > nmsThresh) {
                    suppressed[j] = true;
                }
            }
        }
        
        // Extract kept proposals
        float[] kept = new float[finalIndices.size() * 4];
        for (int i = 0; i < finalIndices.size(); i++) {
            int idx = finalIndices.get(i);
            System.arraycopy(proposals.data, idx * 4, kept, i * 4, 4);
        }
        
        return Torch.tensor(kept, finalIndices.size(), 4);
    }
    
    /**
     * Compute IOU (Intersection over Union) between two boxes.
     */
    private static float computeIOU(Tensor boxes, int idx1, int idx2) {
        float x1_1 = boxes.data[idx1 * 4 + 0];
        float y1_1 = boxes.data[idx1 * 4 + 1];
        float x2_1 = boxes.data[idx1 * 4 + 2];
        float y2_1 = boxes.data[idx1 * 4 + 3];
        
        float x1_2 = boxes.data[idx2 * 4 + 0];
        float y1_2 = boxes.data[idx2 * 4 + 1];
        float x2_2 = boxes.data[idx2 * 4 + 2];
        float y2_2 = boxes.data[idx2 * 4 + 3];
        
        // Intersection
        float interX1 = Math.max(x1_1, x1_2);
        float interY1 = Math.max(y1_1, y1_2);
        float interX2 = Math.min(x2_1, x2_2);
        float interY2 = Math.min(y2_1, y2_2);
        
        float interArea = Math.max(0, interX2 - interX1) * Math.max(0, interY2 - interY1);
        
        // Union
        float area1 = (x2_1 - x1_1) * (y2_1 - y1_1);
        float area2 = (x2_2 - x1_2) * (y2_2 - y1_2);
        float unionArea = area1 + area2 - interArea;
        
        return interArea / (unionArea + 1e-6f);
    }
}
