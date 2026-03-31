package com.user.nn.core;

import static jcuda.runtime.JCuda.cudaMemGetInfo;

/**
 * Lightweight VRAM monitor for long-running training loops.
 *
 * Tracks:
 * - GPU process usage from cudaMemGetInfo(total - free)
 * - pool current/peak usage when GpuMemoryPool is enabled
 * - minimum free memory seen so far
 */
public class GpuMemoryMonitor {
    private final boolean enabled;
    private final int logEverySteps;
    private final double warnUtilizationThreshold;
    private final String tag;

    private long peakProcessUsedBytes = 0;
    private long peakPoolUsedBytes = 0;
    private long minFreeBytes = Long.MAX_VALUE;
    private long totalBytes = 0;

    public GpuMemoryMonitor(boolean enabled, int logEverySteps, double warnUtilizationThreshold, String tag) {
        this.enabled = enabled;
        this.logEverySteps = Math.max(1, logEverySteps);
        this.warnUtilizationThreshold = warnUtilizationThreshold;
        this.tag = tag == null ? "GpuMemoryMonitor" : tag;
    }

    public void recordStep(int step, String phase) {
        if (!enabled || !CUDAOps.isAvailable()) {
            return;
        }

        Snapshot s = sample();
        if (s == null) {
            return;
        }

        if (step % logEverySteps == 0) {
            System.out.printf(
                    "[%s][VRAM] phase=%s step=%d used=%s free=%s pool=%s poolPeak=%s%n",
                    tag,
                    phase,
                    step,
                    formatMb(s.processUsedBytes),
                    formatMb(s.freeBytes),
                    formatMb(s.poolUsedBytes),
                    formatMb(s.poolPeakBytes));
                    
            if (Tensor.fallbackAllocations > 0) {
                System.out.printf("[%s][VRAM][WARN] Fallback fallbackAllocations: %d%n", tag, Tensor.fallbackAllocations);
            }
        }

        double util = s.totalBytes > 0 ? (double) s.processUsedBytes / s.totalBytes : 0.0;
        if (warnUtilizationThreshold > 0.0 && util >= warnUtilizationThreshold) {
            System.out.printf(
                    "[%s][VRAM][WARN] High utilization: %.1f%% (used=%s, free=%s)%n",
                    tag,
                    util * 100.0,
                    formatMb(s.processUsedBytes),
                    formatMb(s.freeBytes));
        }
    }

    public void recordEpoch(int epoch, int totalEpochs) {
        if (!enabled || !CUDAOps.isAvailable()) {
            return;
        }

        Snapshot s = sample();
        if (s == null) {
            return;
        }

        System.out.printf(
                "[%s][VRAM] epoch=%d/%d used=%s peakUsed=%s minFree=%s poolPeak=%s%n",
                tag,
                epoch,
                totalEpochs,
                formatMb(s.processUsedBytes),
                formatMb(peakProcessUsedBytes),
                formatMb(minFreeBytes),
                formatMb(peakPoolUsedBytes));
                
        if (Tensor.fallbackAllocations > 0) {
            System.out.printf("[%s][VRAM][WARN] Total fallback allocations: %d%n", tag, Tensor.fallbackAllocations);
        }
    }

    public void printSummary() {
        if (!enabled || !CUDAOps.isAvailable()) {
            return;
        }

        if (totalBytes <= 0) {
            return;
        }

        System.out.printf(
                "[%s][VRAM][SUMMARY] total=%s peakUsed=%s minFree=%s poolPeak=%s%n",
                tag,
                formatMb(totalBytes),
                formatMb(peakProcessUsedBytes),
                formatMb(minFreeBytes),
                formatMb(peakPoolUsedBytes));
    }

    private Snapshot sample() {
        try {
            long[] free = new long[1];
            long[] total = new long[1];
            cudaMemGetInfo(free, total);

            long processUsed = Math.max(0L, total[0] - free[0]);
            long poolUsed = GpuMemoryPool.isInitialized() ? GpuMemoryPool.getUsedBytes() : 0L;
            long poolPeak = GpuMemoryPool.isInitialized() ? GpuMemoryPool.getPeakUsedBytes() : 0L;

            totalBytes = total[0];
            if (processUsed > peakProcessUsedBytes) {
                peakProcessUsedBytes = processUsed;
            }
            if (poolPeak > peakPoolUsedBytes) {
                peakPoolUsedBytes = poolPeak;
            }
            if (free[0] < minFreeBytes) {
                minFreeBytes = free[0];
            }

            return new Snapshot(processUsed, free[0], total[0], poolUsed, poolPeak);
        } catch (Throwable t) {
            System.err.println("[" + tag + "] Failed to sample VRAM: " + t.getMessage());
            return null;
        }
    }

    private static String formatMb(long bytes) {
        return String.format("%.1fMB", bytes / (1024.0 * 1024.0));
    }

    private static final class Snapshot {
        final long processUsedBytes;
        final long freeBytes;
        final long totalBytes;
        final long poolUsedBytes;
        final long poolPeakBytes;

        Snapshot(long processUsedBytes, long freeBytes, long totalBytes, long poolUsedBytes, long poolPeakBytes) {
            this.processUsedBytes = processUsedBytes;
            this.freeBytes = freeBytes;
            this.totalBytes = totalBytes;
            this.poolUsedBytes = poolUsedBytes;
            this.poolPeakBytes = poolPeakBytes;
        }
    }
}
