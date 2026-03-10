package com.user.nn.utils.progress;

import java.util.ArrayDeque;
import java.util.Deque;

/**
 * Utility class for formatting metrics in progress bars.
 */
public class MetricFormatter {
    
    /**
     * Format time in milliseconds to human-readable string.
     * Examples: "00:05", "01:23", "12:34:56"
     */
    public static String formatTime(long milliseconds) {
        long seconds = milliseconds / 1000;
        long minutes = seconds / 60;
        long hours = minutes / 60;
        
        seconds %= 60;
        minutes %= 60;
        
        if (hours > 0) {
            return String.format("%02d:%02d:%02d", hours, minutes, seconds);
        } else {
            return String.format("%02d:%02d", minutes, seconds);
        }
    }
    
    /**
     * Format rate (items per second) with appropriate units.
     * Examples: "10.5 it/s", "0.5 it/s"
     */
    public static String formatRate(double itemsPerSecond) {
        if (itemsPerSecond >= 1.0) {
            return String.format("%.2f it/s", itemsPerSecond);
        } else if (itemsPerSecond > 0) {
            // Show as seconds per item if less than 1 it/s
            double secondsPerItem = 1.0 / itemsPerSecond;
            return String.format("%.2f s/it", secondsPerItem);
        } else {
            return "0.00 it/s";
        }
    }
    
    /**
     * Format throughput (samples per second).
     * Examples: "1.2K samples/s", "500 samples/s"
     */
    public static String formatThroughput(double samplesPerSecond) {
        if (samplesPerSecond >= 1000000) {
            return String.format("%.1fM samples/s", samplesPerSecond / 1000000);
        } else if (samplesPerSecond >= 1000) {
            return String.format("%.1fK samples/s", samplesPerSecond / 1000);
        } else {
            return String.format("%.0f samples/s", samplesPerSecond);
        }
    }
    
    /**
     * Format memory size with appropriate units.
     * Examples: "1.5 MB", "512 KB", "2.0 GB"
     */
    public static String formatMemory(long bytes) {
        if (bytes >= 1024L * 1024 * 1024) {
            return String.format("%.1f GB", bytes / (1024.0 * 1024 * 1024));
        } else if (bytes >= 1024L * 1024) {
            return String.format("%.1f MB", bytes / (1024.0 * 1024));
        } else if (bytes >= 1024) {
            return String.format("%.1f KB", bytes / 1024.0);
        } else {
            return bytes + " B";
        }
    }
    
    /**
     * Format floating point number with appropriate precision.
     */
    public static String formatFloat(double value, int precision) {
        String format = String.format("%%.%df", precision);
        return String.format(format, value);
    }
    
    /**
     * Running average calculator using exponential moving average.
     */
    public static class ExponentialMovingAverage {
        private double value;
        private boolean initialized;
        private final double alpha; // Smoothing factor (0-1)
        
        /**
         * Create EMA with default smoothing factor 0.1.
         */
        public ExponentialMovingAverage() {
            this(0.1);
        }
        
        /**
         * Create EMA with custom smoothing factor.
         * @param alpha Smoothing factor (0-1). Lower = smoother. Typical: 0.1
         */
        public ExponentialMovingAverage(double alpha) {
            this.alpha = Math.max(0.0, Math.min(1.0, alpha));
            this.value = 0.0;
            this.initialized = false;
        }
        
        /**
         * Update with new value.
         */
        public void update(double newValue) {
            if (!initialized) {
                value = newValue;
                initialized = true;
            } else {
                value = alpha * newValue + (1 - alpha) * value;
            }
        }
        
        /**
         * Get current average.
         */
        public double getValue() {
            return value;
        }
        
        /**
         * Reset the average.
         */
        public void reset() {
            value = 0.0;
            initialized = false;
        }
    }
    
    /**
     * Running average calculator using simple moving average with fixed window.
     */
    public static class SimpleMovingAverage {
        private final Deque<Double> window;
        private final int maxSize;
        private double sum;
        
        /**
         * Create SMA with window size.
         * @param windowSize Number of values to average
         */
        public SimpleMovingAverage(int windowSize) {
            this.maxSize = Math.max(1, windowSize);
            this.window = new ArrayDeque<>(maxSize);
            this.sum = 0.0;
        }
        
        /**
         * Update with new value.
         */
        public void update(double newValue) {
            window.addLast(newValue);
            sum += newValue;
            
            if (window.size() > maxSize) {
                double removed = window.removeFirst();
                sum -= removed;
            }
        }
        
        /**
         * Get current average.
         */
        public double getValue() {
            return window.isEmpty() ? 0.0 : sum / window.size();
        }
        
        /**
         * Get number of values in window.
         */
        public int getCount() {
            return window.size();
        }
        
        /**
         * Reset the average.
         */
        public void reset() {
            window.clear();
            sum = 0.0;
        }
    }
    
    /**
     * Helper to calculate running statistics (mean, variance, std).
     */
    public static class RunningStats {
        private int count;
        private double mean;
        private double m2; // Sum of squared differences from mean
        
        public RunningStats() {
            reset();
        }
        
        /**
         * Update with new value using Welford's online algorithm.
         */
        public void update(double value) {
            count++;
            double delta = value - mean;
            mean += delta / count;
            double delta2 = value - mean;
            m2 += delta * delta2;
        }
        
        /**
         * Get mean.
         */
        public double getMean() {
            return mean;
        }
        
        /**
         * Get variance.
         */
        public double getVariance() {
            return count > 1 ? m2 / (count - 1) : 0.0;
        }
        
        /**
         * Get standard deviation.
         */
        public double getStdDev() {
            return Math.sqrt(getVariance());
        }
        
        /**
         * Get count.
         */
        public int getCount() {
            return count;
        }
        
        /**
         * Reset.
         */
        public void reset() {
            count = 0;
            mean = 0.0;
            m2 = 0.0;
        }
    }
}
