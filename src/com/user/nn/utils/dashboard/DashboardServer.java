package com.user.nn.utils.dashboard;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.user.nn.utils.visualization.TrainingHistory;
import io.javalin.Javalin;
import io.javalin.http.staticfiles.Location;
import io.javalin.websocket.WsContext;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;

public class DashboardServer {

    private final int port;
    private Javalin app;
    private final TrainingHistory history;
    private String currentTaskType = "generic"; // e.g., "classification", "detection", "gan", "nlp"
    
    // Store registered prediction handlers per task path
    private final Map<String, PredictHandler> predictHandlers = new ConcurrentHashMap<>();
    
    // Websocket connections to broadcast live metrics
    private final Queue<WsContext> wsSessions = new ConcurrentLinkedQueue<>();
    private final ObjectMapper mapper = new ObjectMapper();

    // Rich metrics collector for dashboard features
    private final DashboardMetricsCollector metricsCollector = new DashboardMetricsCollector();

    // Training control
    private volatile boolean trainingPaused = false;
    private volatile String modelName = "Model";
    private volatile int currentEpoch = 0;
    private volatile int totalEpochs = 0;

    public DashboardServer(int port, TrainingHistory history) {
        this.port = port;
        this.history = history;
    }

    public void setTaskType(String taskType) {
        this.currentTaskType = taskType;
    }

    public void setModelInfo(String name, int totalEpochs) {
        this.modelName = name;
        this.totalEpochs = totalEpochs;
    }

    public void setCurrentEpoch(int epoch) {
        this.currentEpoch = epoch;
    }

    public boolean isTrainingPaused() {
        return trainingPaused;
    }

    public DashboardMetricsCollector getMetricsCollector() {
        return metricsCollector;
    }

    public DashboardServer start() {
        if (app != null) return this;

        // Start GPU metrics collection in background
        metricsCollector.startPeriodicCollection();

        ClassLoader classLoader = Thread.currentThread().getContextClassLoader();
        
        app = Javalin.create(config -> {
            config.plugins.enableCors(cors -> cors.add(it -> it.anyHost()));
            config.staticFiles.add(staticFileConfig -> {
                // Use external path for development to allow instant UI updates without rebuild
                String devPath = "core/src/main/resources/public";
                if (new File(devPath).exists()) {
                    staticFileConfig.directory = devPath;
                    staticFileConfig.location = Location.EXTERNAL;
                } else {
                    staticFileConfig.directory = "public";
                    staticFileConfig.location = Location.CLASSPATH;
                }
            });
            config.jsonMapper(new io.javalin.json.JavalinJackson(mapper));
        }).start(port);

        setupRoutes();
        System.out.println("[DashboardServer] Started at http://localhost:" + port + " [Task: " + currentTaskType + "]");
        return this;
    }
    
    public void stop() {
        metricsCollector.stop();
        if (app != null) {
            app.stop();
            app = null;
        }
    }

    private void setupRoutes() {
        // API để load lịch sử train (Offline viewing / initial load)
        app.get("/api/metrics", ctx -> {
            Map<String, Object> data = new HashMap<>();
            data.put("taskType", currentTaskType);
            data.put("epochs", history.getEpochs());
            data.put("modelName", modelName);
            data.put("currentEpoch", currentEpoch);
            data.put("totalEpochs", totalEpochs);
            Map<String, List<Float>> series = new HashMap<>();
            try {
                var map = history.getMetrics();
                series.putAll(map);
            } catch(Exception e) {}
            data.put("series", series);
            ctx.json(data);
        });

        // System metrics endpoint
        app.get("/api/system", ctx -> {
            ctx.json(metricsCollector.buildSystemPayload());
        });

        // Alerts endpoint
        app.get("/api/alerts", ctx -> {
            ctx.json(metricsCollector.getAlerts());
        });

        // Confusion matrix endpoint
        app.get("/api/confusion", ctx -> {
            Map<String, Object> cm = metricsCollector.getLatestConfusionMatrix();
            if (cm != null) {
                ctx.json(cm);
            } else {
                ctx.json(Map.of("matrix", new int[0][0], "labels", new String[0]));
            }
        });

        // Live predictions endpoint
        app.get("/api/predictions", ctx -> {
            ctx.json(metricsCollector.getLivePredictions());
        });

        // GAN sample history for time-lapse
        app.get("/api/gan/history", ctx -> {
            ctx.json(metricsCollector.getGANSampleHistory());
        });

        // NLP text stream
        app.get("/api/nlp/stream", ctx -> {
            ctx.json(metricsCollector.getNLPTextStream());
        });

        // NLP token distribution
        app.get("/api/nlp/tokens", ctx -> {
            ctx.json(metricsCollector.getTokenDistribution());
        });

        // Detection leaderboard
        app.get("/api/detection/leaderboard", ctx -> {
            ctx.json(metricsCollector.getPerClassMAP());
        });

        // Training control (pause / resume)
        app.post("/api/training/pause", ctx -> {
            trainingPaused = true;
            ctx.json(Map.of("status", "paused"));
        });
        app.post("/api/training/resume", ctx -> {
            trainingPaused = false;
            ctx.json(Map.of("status", "running"));
        });
        app.get("/api/training/status", ctx -> {
            ctx.json(Map.of("paused", trainingPaused, "epoch", currentEpoch, "totalEpochs", totalEpochs, "modelName", modelName));
        });

        // Đăng kí nhận predict 
        app.post("/api/predict/{task}", ctx -> {
            String task = ctx.pathParam("task");
            PredictHandler handler = predictHandlers.get(task);
            if (handler == null) {
                ctx.status(404).json(Map.of("error", "No handler registered for task: " + task));
                return;
            }

            try {
                String text = ctx.formParam("text");
                var uploadedFile = ctx.uploadedFile("file");
                
                String fileName = null;
                java.io.InputStream stream = null;
                
                if (uploadedFile != null) {
                    fileName = uploadedFile.filename();
                    stream = uploadedFile.content();
                }

                Object result = handler.predict(fileName, stream, text);
                ctx.json(Map.of("success", true, "result", result));
            } catch (Exception e) {
                e.printStackTrace();
                ctx.status(500).json(Map.of("error", e.getMessage()));
            }
        });

        // Websocket truyền log real-time
        app.ws("/ws/metrics", ws -> {
            ws.onConnect(ctx -> wsSessions.add(ctx));
            ws.onClose(ctx -> wsSessions.remove(ctx));
        });
    }

    /**
     * Broadcast 1 log mới nhất từ Training loop lên tất cả Web users (Generic / Rich data).
     */
    public void broadcastTaskMetrics(int epoch, Map<String, Object> metrics) {
        if (wsSessions.isEmpty()) return;

        this.currentEpoch = epoch;

        Map<String, Object> payload = new HashMap<>();
        payload.put("taskType", currentTaskType);
        payload.put("epoch", epoch);
        payload.put("metrics", metrics);
        payload.put("timestamp", System.currentTimeMillis());
        payload.put("modelName", modelName);
        payload.put("totalEpochs", totalEpochs);
        payload.put("paused", trainingPaused);

        // Attach system metrics
        try {
            Map<String, Object> sys = metricsCollector.getSystemMetrics();
            if (!sys.isEmpty()) payload.put("system", sys);
        } catch (Throwable t) {}

        try {
            if (com.user.nn.core.CUDAOps.isAvailable()) {
                long[] free = new long[1];
                long[] total = new long[1];
                jcuda.runtime.JCuda.cudaMemGetInfo(free, total);
                long processUsed = Math.max(0L, total[0] - free[0]);
                long poolUsed = com.user.nn.core.GpuMemoryPool.isInitialized() ? com.user.nn.core.GpuMemoryPool.getUsedBytes() : 0L;
                
                Map<String, Object> vram = new HashMap<>();
                vram.put("processUsedMB", processUsed / (1024.0 * 1024.0));
                vram.put("poolUsedMB", poolUsed / (1024.0 * 1024.0));
                vram.put("totalMB", total[0] / (1024.0 * 1024.0));
                vram.put("freeMB", free[0] / (1024.0 * 1024.0));
                vram.put("utilizationPercent", (double) processUsed / total[0] * 100.0);
                payload.put("vram", vram);
            }
        } catch (Throwable t) {}

        String json;
        try {
            json = mapper.writeValueAsString(payload);
            for (WsContext ctx : wsSessions) {
                try {
                    ctx.send(json);
                } catch (Exception wsEx) {
                    System.err.println("Failed to send WS message, removing session.");
                    wsSessions.remove(ctx);
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Backward compatibility for legacy training loops using Float metrics.
     */
    public void broadcastMetrics(int epoch, Map<String, Float> metrics) {
        Map<String, Object> objMetrics = new HashMap<>(metrics);
        broadcastTaskMetrics(epoch, objMetrics);
    }

    /**
     * Đăng ký 1 Adapter xử lý inference
     */
    public void registerHandler(String taskName, PredictHandler handler) {
        predictHandlers.put(taskName, handler);
    }
    
    /**
     * Xuất state ra file JSON để xem Offline sau.
     */
    public void exportDashboardData(String filePath) throws IOException {
        Map<String, Object> data = new HashMap<>();
        data.put("epochs", history.getEpochs());
        data.put("series", history.getMetrics());
        
        mapper.writerWithDefaultPrettyPrinter().writeValue(new File(filePath), data);
        System.out.println("[DashboardServer] Exported dashboard state to: " + filePath);
    }
    
    /**
     * Khởi chạy độc lập (Offline View)
     */
    public static void serveOffline(int port, String jsonPath) {
        ObjectMapper mapper = new ObjectMapper();
        File dataFile = new File(jsonPath);
        
        if (!dataFile.exists()) {
            System.err.println("[Error] Cannot find offline data file: " + jsonPath);
            return;
        }

        try {
            Map<?, ?> data = mapper.readValue(dataFile, Map.class);
            
            Javalin offlineApp = Javalin.create(config -> {
                config.staticFiles.add(staticFileConfig -> {
                    staticFileConfig.directory = "public"; 
                    staticFileConfig.location = Location.CLASSPATH;
                });
                config.jsonMapper(new io.javalin.json.JavalinJackson(mapper));
            }).start(port);

            offlineApp.get("/api/metrics", ctx -> ctx.json(data));
            
            System.out.println("[DashboardServer - Offline Mode] Started at http://localhost:" + port);
            System.out.println("Serving data from: " + jsonPath);
            
        } catch (IOException e) {
            System.err.println("[Error] Failed to load offline data: " + e.getMessage());
        }
    }
    
    public static void main(String[] args) {
        if (args.length < 2) {
            System.out.println("Usage: DashboardServer <port> <path_to_json>");
            return;
        }
        int p = Integer.parseInt(args[0]);
        String path = args[1];
        serveOffline(p, path);
    }
}
