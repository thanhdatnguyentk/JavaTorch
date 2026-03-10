package com.user.nn.utils.visualization;

import com.user.nn.core.Module;
import com.user.nn.core.Parameter;
import com.user.nn.core.Tensor;

import java.util.List;

/**
 * ML-specific visualization helpers.
 * Provides convenience methods for common machine learning visualizations.
 * 
 * Example:
 * <pre>
 * // Confusion matrix
 * int[][] cm = {{50, 2}, {3, 45}};
 * Plot cmPlot =MLViz.plotConfusionMatrix(cm, new String[]{"Class 0", "Class 1"});
 * 
 * // Training curves
 * TrainingHistory history = ...;
 * Plot curvesPlot = MLViz.plotTrainingCurves(history);
 * 
 * // Weight distribution
 * Tensor weights = model.parameters().get("layer.weight").tensor;
 * Plot histPlot = MLViz.plotWeightDistribution(weights);
 * </pre>
 */
public class MLViz {
    
    /**
     * Plot confusion matrix as a heatmap.
     * 
     * @param confusionMatrix Confusion matrix (predicted x actual)
     * @param classLabels Labels for each class
     * @return HeatmapPlot
     */
    public static HeatmapPlot plotConfusionMatrix(int[][] confusionMatrix, String[] classLabels) {
        int n = confusionMatrix.length;
        double[][] matrix = new double[n][n];
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                matrix[i][j] = confusionMatrix[i][j];
            }
        }
        
        HeatmapPlot plot = new HeatmapPlot(matrix);
        plot.setTickLabels(classLabels, classLabels);
        plot.setColormap("coolwarm");
        plot.setAnnotations(true);
        
        return plot;
    }
    
    /**
     * Plot confusion matrix from float matrix.
     */
    public static HeatmapPlot plotConfusionMatrix(double[][] confusionMatrix, String[] classLabels) {
        HeatmapPlot plot = new HeatmapPlot(confusionMatrix);
        plot.setTickLabels(classLabels, classLabels);
        plot.setColormap("coolwarm");
        plot.setAnnotations(true);
        return plot;
    }
    
    /**
     * Plot training curves with train/validation metrics.
     * Automatically separates loss and other metrics into different plots.
     * 
     * @param history Training history
     * @return LinePlot with all metrics
     */
    public static LinePlot plotTrainingCurves(TrainingHistory history) {
        return (LinePlot) history.plot();
    }
    
    /**
     * Plot loss curves (train and validation).
     */
    public static LinePlot plotLossCurves(TrainingHistory history) {
        String[] lossMetrics = history.getMetricNames().stream()
            .filter(name -> name.toLowerCase().contains("loss"))
            .toArray(String[]::new);
        
        return (LinePlot) history.plot(lossMetrics);
    }
    
    /**
     * Plot accuracy curves (train and validation).
     */
    public static LinePlot plotAccuracyCurves(TrainingHistory history) {
        String[] accMetrics = history.getMetricNames().stream()
            .filter(name -> name.toLowerCase().contains("acc"))
            .toArray(String[]::new);
        
        return (LinePlot) history.plot(accMetrics);
    }
    
    /**
     * Plot weight distribution as a histogram.
     * 
     * @param weights Weight tensor
     * @param label Label for the histogram
     * @return Histogram
     */
    public static Histogram plotWeightDistribution(Tensor weights, String label) {
        double[] data = toDoubleArray(weights);
        Histogram hist = new Histogram(data, label);
        hist.useScottBins(); // Use Scott's rule for binning
        return hist;
    }
    
    /**
     * Plot weight distribution with default label.
     */
    public static Histogram plotWeightDistribution(Tensor weights) {
        return plotWeightDistribution(weights, "Weight Distribution");
    }
    
    /**
     * Plot gradient flow (gradient norms per layer).
     * Useful for diagnosing vanishing/exploding gradients.
     * 
     * @param model The model
     * @return BarChart showing gradient norms
     */
    public static BarChart plotGradientFlow(Module model) {
        List<Parameter> params = model.parameters();
        
        int count = 0;
        for (Parameter p : params) {
            Tensor grad = p.getGrad();
            if (grad != null) {
                count++;
            }
        }
        
        String[] layerNames = new String[count];
        double[] gradNorms = new double[count];
        
        int idx = 0;
        for (int i = 0; i < params.size(); i++) {
            Parameter p = params.get(i);
            Tensor grad = p.getGrad();
            if (grad != null) {
                layerNames[idx] = "Layer " + i;
                gradNorms[idx] = calculateNorm(grad);
                idx++;
            }
        }
        
        BarChart chart = new BarChart(layerNames, gradNorms, "Gradient Norm");
        return chart;
    }
    
    /**
     * Plot 2D embeddings with color-coded labels.
     * Useful for visualizing learned representations (e.g., from t-SNE, PCA).
     * 
     * @param embeddings 2D embeddings (N x 2)
     * @param labels Class labels for each point
     * @param classNames Names of classes
     * @return ScatterPlot
     */
    public static ScatterPlot plotEmbeddings2D(Tensor embeddings, int[] labels, String[] classNames) {
        if (embeddings.shape.length != 2 || embeddings.shape[1] != 2) {
            throw new IllegalArgumentException("Embeddings must be N x 2");
        }
        
        int n = embeddings.shape[0];
        double[] x = new double[n];
        double[] y = new double[n];
        
        for (int i = 0; i < n; i++) {
            x[i] = embeddings.data[i * 2];
            y[i] = embeddings.data[i * 2 + 1];
        }
        
        ScatterPlot plot = new ScatterPlot(x, y, "Embeddings");
        
        // Color by label (normalize to 0-1)
        if (labels != null) {
            double[] colorValues = new double[n];
            int maxLabel = 0;
            for (int label : labels) {
                if (label > maxLabel) maxLabel = label;
            }
            for (int i = 0; i < n; i++) {
                colorValues[i] = maxLabel > 0 ? labels[i] / (double) maxLabel : 0;
            }
            plot.setColors(colorValues);
        }
        
        return plot;
    }
    
    /**
     * Plot embeddings without class names.
     */
    public static ScatterPlot plotEmbeddings2D(Tensor embeddings, int[] labels) {
        return plotEmbeddings2D(embeddings, labels, null);
    }
    
    /**
     * Plot parameter statistics comparison across models.
     * 
     * @param modelNames Names of models
     * @param paramCounts Parameter counts for each model
     * @return BarChart
     */
    public static BarChart plotModelComparison(String[] modelNames, double[] paramCounts) {
        return new BarChart(modelNames, paramCounts, "Parameters");
    }
    
    /**
     * Plot learning rate schedule.
     * 
     * @param epochs Epoch numbers
     * @param learningRates Learning rates at each epoch
     * @return LinePlot
     */
    public static LinePlot plotLearningRateSchedule(double[] epochs, double[] learningRates) {
        LinePlot plot = new LinePlot(epochs, learningRates, "Learning Rate");
        return plot;
    }
    
    /**
     * Plot attention weights as a heatmap.
     * Useful for visualizing transformer attention patterns.
     * 
     * @param attentionWeights Attention weight matrix (query x key)
     * @param queryLabels Labels for queries
     * @param keyLabels Labels for keys
     * @return HeatmapPlot
     */
    public static HeatmapPlot plotAttentionWeights(double[][] attentionWeights, 
                                                   String[] queryLabels, 
                                                   String[] keyLabels) {
        HeatmapPlot plot = new HeatmapPlot(attentionWeights);
        plot.setTickLabels(keyLabels, queryLabels);
        plot.setColormap("viridis");
        return plot;
    }
    
    /**
     * Plot activation distribution.
     */
    public static Histogram plotActivationDistribution(Tensor activations, String layerName) {
        double[] data = toDoubleArray(activations);
        Histogram hist = new Histogram(data, layerName + " Activations");
        hist.useFreedmanDiaconisBins();
        return hist;
    }
    
    // Helper methods
    
    /**
     * Convert tensor to double array.
     */
    private static double[] toDoubleArray(Tensor tensor) {
        double[] result = new double[tensor.data.length];
        for (int i = 0; i < tensor.data.length; i++) {
            result[i] = tensor.data[i];
        }
        return result;
    }
    
    /**
     * Calculate L2 norm of a tensor.
     */
    private static double calculateNorm(Tensor tensor) {
        double sum = 0;
        for (float v : tensor.data) {
            sum += v * v;
        }
        return Math.sqrt(sum);
    }
}
