package com.user.nn.benchmark;

import java.util.Arrays;

public final class BenchmarkStats {
    private BenchmarkStats() {
    }

    public static double percentile(double[] values, double p) {
        if (values == null || values.length == 0) {
            return Double.NaN;
        }
        if (p <= 0.0) {
            return min(values);
        }
        if (p >= 100.0) {
            return max(values);
        }

        double[] sorted = Arrays.copyOf(values, values.length);
        Arrays.sort(sorted);
        double rank = (p / 100.0) * (sorted.length - 1);
        int lower = (int) Math.floor(rank);
        int upper = (int) Math.ceil(rank);
        if (lower == upper) {
            return sorted[lower];
        }

        double w = rank - lower;
        return sorted[lower] * (1.0 - w) + sorted[upper] * w;
    }

    public static double min(double[] values) {
        double m = values[0];
        for (int i = 1; i < values.length; i++) {
            m = Math.min(m, values[i]);
        }
        return m;
    }

    public static double max(double[] values) {
        double m = values[0];
        for (int i = 1; i < values.length; i++) {
            m = Math.max(m, values[i]);
        }
        return m;
    }
}
