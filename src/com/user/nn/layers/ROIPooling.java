package com.user.nn.layers;

import com.user.nn.core.*;
import com.user.nn.core.Module;

/**
 * ROI (Region of Interest) Pooling layer for object detection.
 * Converts features from different sized regions into fixed-size feature vectors.
 * 
 * Used in: Fast R-CNN, Faster R-CNN, Mask R-CNN
 * 
 * Input:
 *   - features: [B, C, H, W] - feature maps from backbone
 *   - rois: [N, 5] - regions of interest [batch_idx, x1, y1, x2, y2]
 * 
 * Output:
 *   - [N, C, pool_h, pool_w] - pooled features for each ROI
 */
public class ROIPooling extends Module {
    private int pooledH;
    private int pooledW;
    private float spatialScale; // scale factor from input image to feature map
    
    /**
     * @param pooledH Output height after pooling
     * @param pooledW Output width after pooling
     * @param spatialScale Ratio of feature map size to input image size (e.g., 1/16 for typical CNNs)
     */
    public ROIPooling(int pooledH, int pooledW, float spatialScale) {
        this.pooledH = pooledH;
        this.pooledW = pooledW;
        this.spatialScale = spatialScale;
    }
    
    /**
     * Forward pass with separate feature maps and ROI tensor.
     * 
     * @param features Feature maps [B, C, H, W]
     * @param rois ROI coordinates [N, 5] where each row is [batch_idx, x1, y1, x2, y2]
     * @return Pooled features [N, C, pooledH, pooledW]
     */
    public Tensor forward(Tensor features, Tensor rois) {
        int B = features.shape[0]; // batch size
        int C = features.shape[1]; // channels
        int H = features.shape[2]; // feature map height
        int W = features.shape[3]; // feature map width
        
        int numRois = rois.shape[0];
        
        // Output tensor: [numRois, C, pooledH, pooledW]
        float[] outputData = new float[numRois * C * pooledH * pooledW];
        
        // For each ROI
        for (int n = 0; n < numRois; n++) {
            int roiOffset = n * 5;
            int batchIdx = (int) rois.data[roiOffset];
            
            // Scale ROI coordinates to feature map size
            int x1 = Math.round(rois.data[roiOffset + 1] * spatialScale);
            int y1 = Math.round(rois.data[roiOffset + 2] * spatialScale);
            int x2 = Math.round(rois.data[roiOffset + 3] * spatialScale);
            int y2 = Math.round(rois.data[roiOffset + 4] * spatialScale);
            
            // Clip to feature map boundaries
            x1 = Math.max(0, Math.min(x1, W - 1));
            y1 = Math.max(0, Math.min(y1, H - 1));
            x2 = Math.max(x1 + 1, Math.min(x2, W));
            y2 = Math.max(y1 + 1, Math.min(y2, H));
            
            int roiWidth = x2 - x1;
            int roiHeight = y2 - y1;
            
            // Divide ROI into pooling bins
            float binSizeH = (float) roiHeight / pooledH;
            float binSizeW = (float) roiWidth / pooledW;
            
            // For each channel
            for (int c = 0; c < C; c++) {
                // For each pooling bin
                for (int ph = 0; ph < pooledH; ph++) {
                    for (int pw = 0; pw < pooledW; pw++) {
                        // Calculate bin boundaries
                        int hStart = y1 + (int) Math.floor(ph * binSizeH);
                        int wStart = x1 + (int) Math.floor(pw * binSizeW);
                        int hEnd = y1 + (int) Math.ceil((ph + 1) * binSizeH);
                        int wEnd = x1 + (int) Math.ceil((pw + 1) * binSizeW);
                        
                        // Clip to ROI boundaries
                        hStart = Math.max(hStart, y1);
                        wStart = Math.max(wStart, x1);
                        hEnd = Math.min(hEnd, y2);
                        wEnd = Math.min(wEnd, x2);
                        
                        // Max pooling within the bin
                        float maxVal = Float.NEGATIVE_INFINITY;
                        for (int h = hStart; h < hEnd; h++) {
                            for (int w = wStart; w < wEnd; w++) {
                                int featureIdx = ((batchIdx * C + c) * H + h) * W + w;
                                maxVal = Math.max(maxVal, features.data[featureIdx]);
                            }
                        }
                        
                        // Handle empty bins
                        if (maxVal == Float.NEGATIVE_INFINITY) {
                            maxVal = 0f;
                        }
                        
                        int outputIdx = ((n * C + c) * pooledH + ph) * pooledW + pw;
                        outputData[outputIdx] = maxVal;
                    }
                }
            }
        }
        
        return Torch.tensor(outputData, numRois, C, pooledH, pooledW);
    }
    
    @Override
    public Tensor forward(Tensor x) {
        throw new UnsupportedOperationException(
            "ROIPooling requires both feature maps and ROIs. Use forward(features, rois) instead.");
    }
}
