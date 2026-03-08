package com.user.nn.metrics;

import java.util.HashMap;
import java.util.Map;

/**
 * Utility to track multiple metrics and losses over an epoch.
 */
public class MetricTracker {
    private final Map<String, Metric> metrics = new HashMap<>();
    private final Map<String, Float> runningSums = new HashMap<>();
    private final Map<String, Integer> counts = new HashMap<>();

    public void addMetric(String name, Metric metric) {
        metrics.put(name, metric);
    }

    public void update(String name, float value) {
        runningSums.put(name, runningSums.getOrDefault(name, 0f) + value);
        counts.put(name, counts.getOrDefault(name, 0) + 1);
    }

    public float getAverage(String name) {
        int count = counts.getOrDefault(name, 0);
        return count == 0 ? 0f : runningSums.getOrDefault(name, 0f) / count;
    }

    public void reset() {
        for (Metric m : metrics.values()) {
            m.reset();
        }
        runningSums.clear();
        counts.clear();
    }

    public Metric getMetric(String name) {
        return metrics.get(name);
    }
}
