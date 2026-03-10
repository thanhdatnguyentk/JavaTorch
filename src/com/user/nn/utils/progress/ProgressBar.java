package com.user.nn.utils.progress;

import java.io.PrintStream;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * A tqdm-like progress bar for Java.
 * 
 * Example usage:
 * <pre>
 * ProgressBar bar = new ProgressBar(100, "Processing");
 * for (int i = 0; i < 100; i++) {
 *     // Do work
 *     bar.update(1);
 *     bar.setPostfix("loss", 0.5f);
 * }
 * bar.close();
 * </pre>
 */
public class ProgressBar implements AutoCloseable {
    
    private final int total;
    private final String description;
    private final AtomicInteger current;
    private final AtomicLong startTime;
    private final Map<String, Object> postfixMetrics;
    private final boolean enabled;
    private final int width;
    private final PrintStream output;
    
    private volatile boolean closed = false;
    private Thread refreshThread;
    private final long refreshIntervalMs = 100; // Refresh every 100ms
    private long lastRefreshTime = 0;
    
    /**
     * Create a progress bar with default settings.
     * 
     * @param total Total number of items
     * @param description Description text
     */
    public ProgressBar(int total, String description) {
        this(total, description, true, 30, System.err);
    }
    
    /**
     * Create a progress bar with custom settings.
     * 
     * @param total Total number of items
     * @param description Description text
     * @param enabled Whether to show the progress bar
     * @param width Width of the progress bar in characters
     * @param output Output stream (typically System.err)
     */
    public ProgressBar(int total, String description, boolean enabled, int width, PrintStream output) {
        this.total = total;
        this.description = description != null ? description : "";
        this.current = new AtomicInteger(0);
        this.startTime = new AtomicLong(System.currentTimeMillis());
        this.postfixMetrics = new LinkedHashMap<>();
        this.enabled = enabled && AnsiCodes.isEnabled();
        this.width = width;
        this.output = output;
        
        if (this.enabled) {
            // Print initial bar
            refresh();
        }
    }
    
    /**
     * Update progress by n items.
     */
    public synchronized void update(int n) {
        if (closed) return;
        
        int newCurrent = current.addAndGet(n);
        
        // Refresh if enough time has passed or completed
        long now = System.currentTimeMillis();
        if (newCurrent >= total || (now - lastRefreshTime) >= refreshIntervalMs) {
            refresh();
            lastRefreshTime = now;
        }
    }
    
    /**
     * Update progress by 1.
     */
    public void update() {
        update(1);
    }
    
    /**
     * Set the current position directly.
     */
    public synchronized void setProgress(int n) {
        if (closed) return;
        current.set(Math.min(n, total));
        refresh();
    }
    
    /**
     * Set a postfix metric (e.g., loss, accuracy).
     */
    public synchronized void setPostfix(String key, Object value) {
        if (closed) return;
        postfixMetrics.put(key, value);
    }
    
    /**
     * Set multiple postfix metrics.
     */
    public synchronized void setPostfix(Map<String, Object> metrics) {
        if (closed) return;
        postfixMetrics.putAll(metrics);
    }
    
    /**
     * Clear all postfix metrics.
     */
    public synchronized void clearPostfix() {
        postfixMetrics.clear();
    }
    
    /**
     * Refresh the display.
     */
    public synchronized void refresh() {
        if (!enabled || closed) return;
        
        int curr = current.get();
        double percentage = total > 0 ? (100.0 * curr) / total : 0.0;
        
        // Calculate elapsed time
        long elapsed = System.currentTimeMillis() - startTime.get();
        String elapsedStr = MetricFormatter.formatTime(elapsed);
        
        // Calculate rate and ETA
        double rate = elapsed > 0 ? (1000.0 * curr) / elapsed : 0.0;
        String rateStr = MetricFormatter.formatRate(rate);
        
        long eta = 0;
        if (rate > 0 && curr < total) {
            eta = (long) ((total - curr) * 1000.0 / rate);
        }
        String etaStr = MetricFormatter.formatTime(eta);
        
        // Build progress bar
        StringBuilder bar = new StringBuilder();
        
        // Clear line and move to start
        bar.append(AnsiCodes.clearLine());
        
        // Description
        if (!description.isEmpty()) {
            bar.append(description).append(": ");
        }
        
        // Percentage
        bar.append(String.format("%3d%%", (int) percentage));
        bar.append(" ");
        
        // Visual bar
        bar.append(buildBar(curr, total, width));
        bar.append(" ");
        
        // Counter
        bar.append(String.format("%d/%d", curr, total));
        bar.append(" ");
        
        // Time info
        bar.append("[");
        bar.append(elapsedStr);
        if (curr < total && eta > 0) {
            bar.append("<");
            bar.append(etaStr);
        }
        bar.append(", ");
        bar.append(rateStr);
        bar.append("]");
        
        // Postfix metrics
        if (!postfixMetrics.isEmpty()) {
            bar.append(" ");
            bar.append(formatPostfix());
        }
        
        // Write to output
        output.print(bar.toString());
        output.flush();
    }
    
    /**
     * Build the visual progress bar.
     */
    private String buildBar(int current, int total, int width) {
        double fraction = total > 0 ? (double) current / total : 0.0;
        int filled = (int) (width * fraction);
        
        StringBuilder bar = new StringBuilder();
        bar.append(AnsiCodes.BRIGHT_BLUE);
        bar.append("[");
        
        // Filled portion
        for (int i = 0; i < filled; i++) {
            bar.append("=");
        }
        
        // Arrow head
        if (filled < width && current < total) {
            bar.append(">");
            filled++;
        }
        
        // Empty portion
        for (int i = filled; i < width; i++) {
            bar.append(" ");
        }
        
        bar.append("]");
        bar.append(AnsiCodes.RESET);
        
        return bar.toString();
    }
    
    /**
     * Format postfix metrics.
     */
    private String formatPostfix() {
        StringBuilder sb = new StringBuilder();
        boolean first = true;
        for (Map.Entry<String, Object> entry : postfixMetrics.entrySet()) {
            if (!first) {
                sb.append(", ");
            }
            sb.append(entry.getKey()).append("=");
            
            Object value = entry.getValue();
            if (value instanceof Float || value instanceof Double) {
                sb.append(String.format("%.4f", ((Number) value).doubleValue()));
            } else {
                sb.append(value);
            }
            first = false;
        }
        return sb.toString();
    }
    
    /**
     * Reset the progress bar.
     */
    public synchronized void reset() {
        current.set(0);
        startTime.set(System.currentTimeMillis());
        postfixMetrics.clear();
        lastRefreshTime = 0;
        if (enabled) {
            refresh();
        }
    }
    
    /**
     * Close the progress bar and move to the next line.
     */
    @Override
    public synchronized void close() {
        if (closed) return;
        closed = true;
        
        if (enabled) {
            // Final refresh
            refresh();
            // Move to next line
            output.println();
            output.flush();
        }
        
        if (refreshThread != null && refreshThread.isAlive()) {
            refreshThread.interrupt();
        }
    }
    
    /**
     * Get current progress.
     */
    public int getCurrent() {
        return current.get();
    }
    
    /**
     * Get total items.
     */
    public int getTotal() {
        return total;
    }
    
    /**
     * Check if completed.
     */
    public boolean isCompleted() {
        return current.get() >= total;
    }
    
    /**
     * Get elapsed time in milliseconds.
     */
    public long getElapsedMs() {
        return System.currentTimeMillis() - startTime.get();
    }
    
    /**
     * Get current rate (items per second).
     */
    public double getRate() {
        long elapsed = getElapsedMs();
        return elapsed > 0 ? (1000.0 * current.get()) / elapsed : 0.0;
    }
}
