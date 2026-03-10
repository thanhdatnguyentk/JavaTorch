package com.user.nn.utils.progress;

import com.user.nn.core.Tensor;
import com.user.nn.dataloaders.Data.DataLoader;
import com.user.nn.metrics.MetricTracker;

import java.util.Iterator;
import java.util.Map;

/**
 * A wrapper around DataLoader that displays a progress bar.
 * 
 * Example usage:
 * <pre>
 * DataLoader loader = new DataLoader(dataset, 32, true, 2);
 * ProgressDataLoader progLoader = new ProgressDataLoader(loader, "Training");
 * 
 * for (Tensor[] batch : progLoader) {
 *     // Training code
 *     progLoader.setPostfix("loss", loss);
 * }
 * progLoader.close();
 * </pre>
 */
public class ProgressDataLoader implements Iterable<Tensor[]>, Iterator<Tensor[]>, AutoCloseable {
    
    private final DataLoader dataLoader;
    private final ProgressBar progressBar;
    private final boolean enabled;
    private Iterator<Tensor[]> iterator;
    
    // Optional metric tracking
    private MetricTracker metricTracker;
    private String[] metricNames;
    
    /**
     * Create a progress bar wrapper for a DataLoader.
     * 
     * @param dataLoader The underlying data loader
     * @param description Description text for progress bar
     */
    public ProgressDataLoader(DataLoader dataLoader, String description) {
        this(dataLoader, description, true);
    }
    
    /**
     * Create a progress bar wrapper with enable/disable option.
     * 
     * @param dataLoader The underlying data loader
     * @param description Description text for progress bar
     * @param enabled Whether to show the progress bar
     */
    public ProgressDataLoader(DataLoader dataLoader, String description, boolean enabled) {
        this.dataLoader = dataLoader;
        this.enabled = enabled;
        
        // Calculate total number of batches
        int totalBatches = calculateTotalBatches();
        
        this.progressBar = new ProgressBar(totalBatches, description, enabled, 30, System.err);
    }
    
    /**
     * Calculate total number of batches in the DataLoader.
     */
    private int calculateTotalBatches() {
        try {
            // Try to get dataset size via reflection
            java.lang.reflect.Field datasetField = dataLoader.getClass().getDeclaredField("dataset");
            datasetField.setAccessible(true);
            Object dataset = datasetField.get(dataLoader);
            
            java.lang.reflect.Method lenMethod = dataset.getClass().getMethod("len");
            int datasetSize = (int) lenMethod.invoke(dataset);
            
            java.lang.reflect.Field batchSizeField = dataLoader.getClass().getDeclaredField("batchSize");
            batchSizeField.setAccessible(true);
            int batchSize = batchSizeField.getInt(dataLoader);
            
            return (datasetSize + batchSize - 1) / batchSize;
        } catch (Exception e) {
            // Fallback: estimate from first iteration (not ideal)
            return 100; // Default guess
        }
    }
    
    /**
     * Attach a MetricTracker to display metrics in the progress bar.
     * 
     * @param tracker The metric tracker
     * @param names Names of metrics to display
     * @return this
     */
    public ProgressDataLoader withMetrics(MetricTracker tracker, String... names) {
        this.metricTracker = tracker;
        this.metricNames = names;
        return this;
    }
    
    /**
     * Set a postfix metric manually.
     */
    public void setPostfix(String key, Object value) {
        if (enabled) {
            progressBar.setPostfix(key, value);
        }
    }
    
    /**
     * Set multiple postfix metrics.
     */
    public void setPostfix(Map<String, Object> metrics) {
        if (enabled) {
            progressBar.setPostfix(metrics);
        }
    }
    
    /**
     * Update postfix metrics from MetricTracker if attached.
     */
    private void updateMetricsFromTracker() {
        if (metricTracker != null && metricNames != null && enabled) {
            for (String name : metricNames) {
                float value = metricTracker.getAverage(name);
                if (!Float.isNaN(value)) {
                    progressBar.setPostfix(name, value);
                }
            }
        }
    }
    
    @Override
    public Iterator<Tensor[]> iterator() {
        // Reset progress bar and get new iterator
        progressBar.reset();
        iterator = dataLoader.iterator();
        return this;
    }
    
    @Override
    public boolean hasNext() {
        return iterator.hasNext();
    }
    
    @Override
    public Tensor[] next() {
        Tensor[] batch = iterator.next();
        
        if (enabled) {
            progressBar.update(1);
            updateMetricsFromTracker();
        }
        
        return batch;
    }
    
    /**
     * Get the underlying DataLoader.
     */
    public DataLoader getDataLoader() {
        return dataLoader;
    }
    
    /**
     * Get the ProgressBar (for advanced usage).
     */
    public ProgressBar getProgressBar() {
        return progressBar;
    }
    
    /**
     * Close the progress bar and DataLoader.
     */
    @Override
    public void close() {
        progressBar.close();
        // DataLoader cleanup is typically done in try-with-resources at higher level
    }
    
    /**
     * Manually refresh the progress bar (e.g., after updating metrics).
     */
    public void refresh() {
        if (enabled) {
            progressBar.refresh();
        }
    }
}
