package com.user.nn.utils.dashboard;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.*;
import java.util.concurrent.*;

/**
 * Collects GPU/CPU system metrics for the System Monitor dashboard.
 * Provides:
 *   - GPU temperature, utilization, VRAM usage
 *   - CPU utilization
 *   - Active alerts (CUDA warnings, gradient clipping, checkpoints)
 *   - Pipeline timing data
 */
public class DashboardMetricsCollector {

    private final List<Map<String, Object>> alerts = new CopyOnWriteArrayList<>();
    private final Map<String, List<Map<String, Object>>> sampleHistory = new ConcurrentHashMap<>();
    private final Map<String, Object> systemMetrics = new ConcurrentHashMap<>();
    private final List<Map<String, Object>> confusionMatrixHistory = new CopyOnWriteArrayList<>();
    
    // Live prediction grid data
    private volatile List<Map<String, Object>> livePredictions = new ArrayList<>();
    
    // NLP specific
    private final List<Map<String, Object>> nlpTextStream = new CopyOnWriteArrayList<>();
    
    // Detection specific
    private volatile Map<String, Float> perClassMAP = new LinkedHashMap<>();
    private volatile Map<String, Float> lossBreakdown = new LinkedHashMap<>();
    
    // Pipeline timing
    private volatile Map<String, Double> pipelineTiming = new LinkedHashMap<>();

    private ScheduledExecutorService scheduler;

    public DashboardMetricsCollector() {
        // Initialize default pipeline timing
        pipelineTiming.put("disk_io", 0.0);
        pipelineTiming.put("dataloader_cpu", 0.0);
        pipelineTiming.put("pcie_transfer", 0.0);
        pipelineTiming.put("gpu_compute", 0.0);
    }

    /** Start periodic GPU metrics collection (every 2 seconds) */
    public void startPeriodicCollection() {
        if (scheduler != null) return;
        scheduler = Executors.newSingleThreadScheduledExecutor(r -> {
            Thread t = new Thread(r, "DashboardMetricsCollector");
            t.setDaemon(true);
            return t;
        });
        scheduler.scheduleAtFixedRate(this::collectGPUMetrics, 0, 2, TimeUnit.SECONDS);
    }

    public void stop() {
        if (scheduler != null) {
            scheduler.shutdownNow();
            scheduler = null;
        }
    }

    /** Collect GPU metrics via nvidia-smi (temperature, utilization) */
    private void collectGPUMetrics() {
        try {
            ProcessBuilder pb = new ProcessBuilder(
                "nvidia-smi",
                "--query-gpu=temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,name",
                "--format=csv,noheader,nounits"
            );
            pb.redirectErrorStream(true);
            Process proc = pb.start();
            BufferedReader reader = new BufferedReader(new InputStreamReader(proc.getInputStream()));
            String line = reader.readLine();
            proc.waitFor(3, TimeUnit.SECONDS);
            proc.destroyForcibly();

            if (line != null && !line.isEmpty()) {
                String[] parts = line.split(",");
                if (parts.length >= 6) {
                    systemMetrics.put("gpu_temp", Float.parseFloat(parts[0].trim()));
                    systemMetrics.put("gpu_util", Float.parseFloat(parts[1].trim()));
                    systemMetrics.put("gpu_mem_util", Float.parseFloat(parts[2].trim()));
                    systemMetrics.put("gpu_mem_used_mb", Float.parseFloat(parts[3].trim()));
                    systemMetrics.put("gpu_mem_total_mb", Float.parseFloat(parts[4].trim()));
                    systemMetrics.put("gpu_name", parts[5].trim());
                }
            }
        } catch (Exception e) {
            // nvidia-smi not available or failed
        }

        // CPU utilization via OS
        try {
            com.sun.management.OperatingSystemMXBean osBean =
                (com.sun.management.OperatingSystemMXBean) java.lang.management.ManagementFactory.getOperatingSystemMXBean();
            double cpuLoad = osBean.getCpuLoad();
            if (cpuLoad >= 0) {
                systemMetrics.put("cpu_util", (float) (cpuLoad * 100.0));
            }
        } catch (Exception e) {
            // Fallback
        }
    }

    // ==================== ALERTS ====================

    /** Add a system alert (e.g., CUDA warnings, gradient clipping) */
    public void addAlert(String severity, String title, String message) {
        Map<String, Object> alert = new LinkedHashMap<>();
        alert.put("severity", severity); // "warning", "info", "error"
        alert.put("title", title);
        alert.put("message", message);
        alert.put("timestamp", System.currentTimeMillis());
        alerts.add(alert);
        // Keep last 50 alerts
        while (alerts.size() > 50) alerts.remove(0);
    }

    public List<Map<String, Object>> getAlerts() {
        return new ArrayList<>(alerts);
    }

    // ==================== CONFUSION MATRIX ====================

    /** Record confusion matrix for an epoch */
    public void recordConfusionMatrix(int epoch, int[][] matrix, String[] labels) {
        Map<String, Object> entry = new LinkedHashMap<>();
        entry.put("epoch", epoch);
        entry.put("matrix", matrix);
        entry.put("labels", labels);
        confusionMatrixHistory.add(entry);
    }

    public Map<String, Object> getLatestConfusionMatrix() {
        if (confusionMatrixHistory.isEmpty()) return null;
        return confusionMatrixHistory.get(confusionMatrixHistory.size() - 1);
    }

    // ==================== LIVE PREDICTIONS ====================

    /** 
     * Record live predictions for the validation stream.
     * Each prediction: {imageBase64, predictedLabel, actualLabel, correct, topK: [{label, confidence}]}
     */
    public void setLivePredictions(List<Map<String, Object>> predictions) {
        this.livePredictions = predictions;
    }

    public List<Map<String, Object>> getLivePredictions() {
        return livePredictions;
    }

    // ==================== GAN SAMPLE HISTORY ====================

    /** Store samples for a specific epoch (for time-lapse slider) */
    public void recordGANSamples(int epoch, List<String> base64Samples) {
        Map<String, Object> entry = new LinkedHashMap<>();
        entry.put("epoch", epoch);
        entry.put("samples", base64Samples);
        sampleHistory.computeIfAbsent("gan", k -> new CopyOnWriteArrayList<>()).add(entry);
        // Keep last 100 epochs
        List<Map<String, Object>> history = sampleHistory.get("gan");
        while (history.size() > 100) history.remove(0);
    }

    /** Get all GAN sample history for time-lapse */
    public List<Map<String, Object>> getGANSampleHistory() {
        return sampleHistory.getOrDefault("gan", new ArrayList<>());
    }

    private Map<String, Float> tokenImportanceSum = new HashMap<>();
    private Map<String, Integer> tokenImportanceCount = new HashMap<>();

    /** Add a text sample to the NLP stream (chat-log style) */
    public void addNLPTextSample(String text, String predictedLabel, float confidence, Map<String, Float> tokenWeights) {
        Map<String, Object> entry = new LinkedHashMap<>();
        entry.put("text", text);
        entry.put("label", predictedLabel);
        entry.put("confidence", confidence);
        if (tokenWeights != null) {
            entry.put("tokenWeights", tokenWeights);
            // Aggregate token importance for distribution chart
            for (Map.Entry<String, Float> w : tokenWeights.entrySet()) {
                tokenImportanceSum.put(w.getKey(), tokenImportanceSum.getOrDefault(w.getKey(), 0f) + w.getValue());
                tokenImportanceCount.put(w.getKey(), tokenImportanceCount.getOrDefault(w.getKey(), 0) + 1);
            }
        }
        entry.put("timestamp", System.currentTimeMillis());
        nlpTextStream.add(entry);
        // Keep last 30
        while (nlpTextStream.size() > 30) nlpTextStream.remove(0);
    }

    public List<Map<String, Object>> getNLPTextStream() {
        return new ArrayList<>(nlpTextStream);
    }

    public Map<String, Float> getTokenDistribution() {
        // Calculate average importance and sort by value descending, return top 15
        List<Map.Entry<String, Float>> list = new ArrayList<>();
        for (String token : tokenImportanceSum.keySet()) {
            list.add(new AbstractMap.SimpleEntry<>(token, tokenImportanceSum.get(token) / tokenImportanceCount.get(token)));
        }
        list.sort((a, b) -> Float.compare(b.getValue(), a.getValue()));
        
        Map<String, Float> top = new LinkedHashMap<>();
        for (int i = 0; i < Math.min(15, list.size()); i++) {
            top.put(list.get(i).getKey(), list.get(i).getValue());
        }
        return top;
    }

    // ==================== DETECTION ====================

    public void setPerClassMAP(Map<String, Float> mapScores) {
        this.perClassMAP = new LinkedHashMap<>(mapScores);
    }

    public Map<String, Float> getPerClassMAP() {
        return perClassMAP;
    }

    public void setLossBreakdown(Map<String, Float> breakdown) {
        this.lossBreakdown = new LinkedHashMap<>(breakdown);
    }

    public Map<String, Float> getLossBreakdown() {
        return lossBreakdown;
    }

    // ==================== PIPELINE TIMING ====================

    public void setPipelineTiming(String stage, double ms) {
        pipelineTiming.put(stage, ms);
    }

    public Map<String, Double> getPipelineTiming() {
        return new LinkedHashMap<>(pipelineTiming);
    }

    // ==================== SYSTEM METRICS ====================

    public Map<String, Object> getSystemMetrics() {
        return new LinkedHashMap<>(systemMetrics);
    }

    /** Build full system payload for broadcasting */
    public Map<String, Object> buildSystemPayload() {
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("system", getSystemMetrics());
        payload.put("alerts", getAlerts());
        payload.put("pipeline", getPipelineTiming());
        
        // Add VRAM from JCuda if available
        try {
            if (com.user.nn.core.CUDAOps.isAvailable()) {
                long[] free = new long[1];
                long[] total = new long[1];
                jcuda.runtime.JCuda.cudaMemGetInfo(free, total);
                long used = Math.max(0L, total[0] - free[0]);
                long poolUsed = com.user.nn.core.GpuMemoryPool.isInitialized() 
                    ? com.user.nn.core.GpuMemoryPool.getUsedBytes() : 0L;
                
                Map<String, Object> vram = new LinkedHashMap<>();
                vram.put("processUsedMB", used / (1024.0 * 1024.0));
                vram.put("poolUsedMB", poolUsed / (1024.0 * 1024.0));
                vram.put("totalMB", total[0] / (1024.0 * 1024.0));
                vram.put("freeMB", free[0] / (1024.0 * 1024.0));
                vram.put("utilizationPercent", (double) used / total[0] * 100.0);
                payload.put("vram", vram);
            }
        } catch (Throwable t) {
            // ignore
        }
        
        return payload;
    }
}
