package com.user.nn.losses;

import com.user.nn.core.*;
import com.user.nn.core.Module;

/**
 * Focal Loss for addressing class imbalance in object detection.
 * 
 * Focal Loss was introduced in RetinaNet to solve the extreme foreground-background
 * class imbalance problem in one-stage object detectors.
 * 
 * Formula:
 *   FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
 * 
 * where:
 *   - p_t: predicted probability for the true class
 *   - α_t: balancing factor (typically 0.25)
 *   - γ: focusing parameter (typically 2.0)
 *   - (1 - p_t)^γ: modulating factor that reduces loss for well-classified examples
 * 
 * Key Ideas:
 *   - Easy examples (high confidence, correct) contribute less to the loss
 *   - Hard examples (low confidence or misclassified) get more weight
 *   - Automatically down-weights the vast number of easy negatives
 * 
 * Comparison to Cross-Entropy:
 *   - Cross-Entropy: CE = -log(p_t)
 *   - Focal Loss: FL = -(1 - p_t)^γ * log(p_t)
 *   - When p_t→1 (easy example): FL→0 much faster than CE
 *   - When p_t→0 (hard example): FL≈CE
 * 
 * Reference: "Focal Loss for Dense Object Detection" (RetinaNet, 2017)
 * https://arxiv.org/abs/1708.02002
 */
public class FocalLoss extends Module {
    
    private float alpha;   // balancing factor for positive/negative samples
    private float gamma;   // focusing parameter (higher = more focus on hard examples)
    private String reduction; // 'none', 'mean', 'sum'
    
    /**
     * Create Focal Loss with default parameters.
     */
    public FocalLoss() {
        this(0.25f, 2.0f, "mean");
    }
    
    /**
     * Create Focal Loss with custom parameters.
     * 
     * @param alpha Balancing factor (0.25 recommended for object detection)
     * @param gamma Focusing parameter (2.0 recommended, higher = more focus on hard examples)
     * @param reduction Reduction mode: 'none', 'mean', or 'sum'
     */
    public FocalLoss(float alpha, float gamma, String reduction) {
        this.alpha = alpha;
        this.gamma = gamma;
        this.reduction = reduction;
    }
    
    /**
     * Compute focal loss for binary classification.
     * 
     * @param logits Predicted logits [N] or [N, 1]
     * @param targets Target labels [N] (0 or 1)
     * @return Loss scalar
     */
    public Tensor forwardBinary(Tensor logits, Tensor targets) {
        int N = logits.data.length;
        float[] lossData = new float[N];
        
        for (int i = 0; i < N; i++) {
            // Apply sigmoid to get probability
            float p = sigmoid(logits.data[i]);
            
            // Get true class probability
            float target = targets.data[i];
            float p_t = target * p + (1 - target) * (1 - p);
            
            // Compute focal loss
            float alpha_t = target * alpha + (1 - target) * (1 - alpha);
            float focal_weight = (float) Math.pow(1 - p_t, gamma);
            float ce_loss = -(float) Math.log(Math.max(p_t, 1e-8f));
            
            lossData[i] = alpha_t * focal_weight * ce_loss;
        }
        
        return applyReduction(lossData);
    }
    
    /**
     * Compute focal loss for multi-class classification.
     * 
     * @param logits Predicted logits [N, C]
     * @param targets Target class indices [N]
     * @return Loss scalar
     */
    public Tensor forwardMultiClass(Tensor logits, int[] targets) {
        int N = targets.length;
        int C = logits.shape[1]; // number of classes
        float[] lossData = new float[N];
        
        for (int i = 0; i < N; i++) {
            int targetClass = targets[i];
            
            // Apply softmax to get probabilities
            float[] probs = new float[C];
            float maxLogit = Float.NEGATIVE_INFINITY;
            for (int c = 0; c < C; c++) {
                maxLogit = Math.max(maxLogit, logits.data[i * C + c]);
            }
            
            float sumExp = 0;
            for (int c = 0; c < C; c++) {
                probs[c] = (float) Math.exp(logits.data[i * C + c] - maxLogit);
                sumExp += probs[c];
            }
            
            for (int c = 0; c < C; c++) {
                probs[c] /= sumExp;
            }
            
            // True class probability
            float p_t = probs[targetClass];
            
            // Compute focal loss
            float focal_weight = (float) Math.pow(1 - p_t, gamma);
            float ce_loss = -(float) Math.log(Math.max(p_t, 1e-8f));
            
            lossData[i] = alpha * focal_weight * ce_loss;
        }
        
        return applyReduction(lossData);
    }
    
    /**
     * Forward pass with Tensor targets.
     * 
     * @param logits Predicted logits [N, C] for multi-class or [N] for binary
     * @param targets Target tensor
     * @return Loss scalar
     */
    public Tensor forward(Tensor logits, Tensor targets) {
        // Detect binary vs multi-class based on shape
        if (logits.dim() == 1 || (logits.dim() == 2 && logits.shape[1] == 1)) {
            // Binary classification
            return forwardBinary(logits, targets);
        } else {
            // Multi-class classification
            int[] targetIndices = new int[targets.data.length];
            for (int i = 0; i < targets.data.length; i++) {
                targetIndices[i] = (int) targets.data[i];
            }
            return forwardMultiClass(logits, targetIndices);
        }
    }
    
    @Override
    public Tensor forward(Tensor x) {
        throw new UnsupportedOperationException(
            "FocalLoss requires both logits and targets. Use forward(logits, targets).");
    }
    
    /**
     * Apply reduction to loss values.
     */
    private Tensor applyReduction(float[] lossData) {
        if (reduction.equals("none")) {
            return Torch.tensor(lossData, lossData.length);
        } else if (reduction.equals("sum")) {
            float sum = 0;
            for (float loss : lossData) {
                sum += loss;
            }
            return Torch.tensor(new float[]{sum}, 1);
        } else { // mean
            float sum = 0;
            for (float loss : lossData) {
                sum += loss;
            }
            float mean = sum / lossData.length;
            return Torch.tensor(new float[]{mean}, 1);
        }
    }
    
    /**
     * Sigmoid activation.
     */
    private float sigmoid(float x) {
        return 1.0f / (1.0f + (float) Math.exp(-x));
    }
    
    /**
     * Compute focal loss with automatic differentiation support.
     * 
     * @param logits Input logits with requires_grad=true
     * @param targets Target labels
     * @return Loss tensor with gradient support
     */
    public static Tensor focalLoss(Tensor logits, int[] targets, float alpha, float gamma) {
        FocalLoss fl = new FocalLoss(alpha, gamma, "mean");
        return fl.forwardMultiClass(logits, targets);
    }
}
