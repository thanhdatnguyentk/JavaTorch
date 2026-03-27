package com.user.nn.utils.visualization;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

/**
 * Tracks training history (loss, metrics) over epochs.
 * Automatically plots training curves and exports to CSV.
 * 
 * Example:
 * <pre>
 * TrainingHistory history = new TrainingHistory();
 * 
 * for (int epoch = 0; epoch < 100; epoch++) {
 *     // Training...
 *     float trainLoss = ...;
 *     float trainAcc = ...;
 *     float valAcc = ...;
 *     
 *     Map&lt;String, Float&gt; metrics = new HashMap&lt;&gt;();
 *     metrics.put("train_loss", trainLoss);
 *     metrics.put("train_acc", trainAcc);
 *     metrics.put("val_acc", valAcc);
 *     
 *     history.record(epoch, metrics);
 * }
 * 
 * Plot plot = history.plot();
 * </pre>
 */
public class TrainingHistory {
    
    private List<Integer> epochs;
    private Map<String, List<Float>> metrics;
    
    public TrainingHistory() {
        this.epochs = new ArrayList<>();
        this.metrics = new LinkedHashMap<>();
    }
    
    /**
     * Record metrics for an epoch.
     * 
     * @param epoch Epoch number
     * @param metricValues Map of metric names to values
     */
    public void record(int epoch, Map<String, Float> metricValues) {
        epochs.add(epoch);
        
        for (Map.Entry<String, Float> entry : metricValues.entrySet()) {
            String metricName = entry.getKey();
            Float value = entry.getValue();
            
            metrics.putIfAbsent(metricName, new ArrayList<>());
            metrics.get(metricName).add(value);
        }
    }
    
    /**
     * Record a single metric value.
     */
    public void record(int epoch, String metricName, float value) {
        Map<String, Float> map = new HashMap<>();
        map.put(metricName, value);
        record(epoch, map);
    }
    
    /**
     * Get values for a specific metric.
     */
    public List<Float> getMetric(String metricName) {
        return metrics.getOrDefault(metricName, new ArrayList<>());
    }
    
    /**
     * Get the raw map of all metrics.
     */
    public Map<String, List<Float>> getMetrics() {
        return metrics;
    }

    /**
     * Get all metric names.
     */
    public Set<String> getMetricNames() {
        return metrics.keySet();
    }

    /**
     * Get epochs.
     */
    public List<Integer> getEpochs() {
        return epochs;
    }
    
    /**
     * Plot all metrics as line plots.
     * Splits loss metrics and other metrics into separate plots if both are present.
     */
    public Plot plot() {
        return plot(getMetricNames().toArray(new String[0]));
    }
    
    /**
     * Plot specific metrics.
     */
    public Plot plot(String... metricNames) {
        if (epochs.isEmpty()) {
            throw new IllegalStateException("No data recorded");
        }
        
        // Convert epoch list to double array
        double[] x = new double[epochs.size()];
        for (int i = 0; i < epochs.size(); i++) {
            x[i] = epochs.get(i);
        }
        
        // Create line plot
        LinePlot plot = new LinePlot();
        
        for (String metricName : metricNames) {
            List<Float> values = metrics.get(metricName);
            if (values != null && values.size() == x.length) {
                double[] y = new double[values.size()];
                for (int i = 0; i < values.size(); i++) {
                    y[i] = values.get(i);
                }
                plot.addSeries(x, y, metricName);
            }
        }
        
        return plot;
    }
    
    /**
     * Export history to CSV file.
     * 
     * @param filePath Output file path
     * @throws IOException If file cannot be written
     */
    public void saveCSV(String filePath) throws IOException {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath))) {
            // Write header
            writer.write("epoch");
            for (String metricName : metrics.keySet()) {
                writer.write(",");
                writer.write(metricName);
            }
            writer.newLine();
            
            // Write data
            for (int i = 0; i < epochs.size(); i++) {
                writer.write(String.valueOf(epochs.get(i)));
                
                for (String metricName : metrics.keySet()) {
                    writer.write(",");
                    List<Float> values = metrics.get(metricName);
                    if (i < values.size()) {
                        writer.write(String.valueOf(values.get(i)));
                    } else {
                        writer.write("");
                    }
                }
                writer.newLine();
            }
        }
    }
    
    /**
     * Get the latest value for a metric.
     */
    public Float getLatest(String metricName) {
        List<Float> values = metrics.get(metricName);
        if (values == null || values.isEmpty()) {
            return null;
        }
        return values.get(values.size() - 1);
    }
    
    /**
     * Get the best (minimum) value for a metric.
     */
    public Float getMin(String metricName) {
        List<Float> values = metrics.get(metricName);
        if (values == null || values.isEmpty()) {
            return null;
        }
        return Collections.min(values);
    }
    
    /**
     * Get the best (maximum) value for a metric.
     */
    public Float getMax(String metricName) {
        List<Float> values = metrics.get(metricName);
        if (values == null || values.isEmpty()) {
            return null;
        }
        return Collections.max(values);
    }
    
    /**
     * Get epoch where metric achieved minimum.
     */
    public Integer getMinEpoch(String metricName) {
        List<Float> values = metrics.get(metricName);
        if (values == null || values.isEmpty()) {
            return null;
        }
        int minIndex = 0;
        float minValue = values.get(0);
        for (int i = 1; i < values.size(); i++) {
            if (values.get(i) < minValue) {
                minValue = values.get(i);
                minIndex = i;
            }
        }
        return epochs.get(minIndex);
    }
    
    /**
     * Get epoch where metric achieved maximum.
     */
    public Integer getMaxEpoch(String metricName) {
        List<Float> values = metrics.get(metricName);
        if (values == null || values.isEmpty()) {
            return null;
        }
        int maxIndex = 0;
        float maxValue = values.get(0);
        for (int i = 1; i < values.size(); i++) {
            if (values.get(i) > maxValue) {
                maxValue = values.get(i);
                maxIndex = i;
            }
        }
        return epochs.get(maxIndex);
    }
    
    /**
     * Clear all recorded data.
     */
    public void clear() {
        epochs.clear();
        metrics.clear();
    }
}
