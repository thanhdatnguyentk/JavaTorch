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
    
    // Store registered prediction handlers per task path
    private final Map<String, PredictHandler> predictHandlers = new ConcurrentHashMap<>();
    
    // Websocket connections to broadcast live metrics
    private final Queue<WsContext> wsSessions = new ConcurrentLinkedQueue<>();
    private final ObjectMapper mapper = new ObjectMapper();

    public DashboardServer(int port, TrainingHistory history) {
        this.port = port;
        this.history = history;
    }

    public DashboardServer start() {
        if (app != null) return this;

        ClassLoader classLoader = Thread.currentThread().getContextClassLoader();
        
        app = Javalin.create(config -> {
            config.plugins.enableCors(cors -> cors.add(it -> it.anyHost()));
            config.staticFiles.add(staticFileConfig -> {
                staticFileConfig.directory = "public"; // Points to src/main/resources/public
                staticFileConfig.location = Location.CLASSPATH;
            });
            config.jsonMapper(new io.javalin.json.JavalinJackson(mapper));
        }).start(port);

        setupRoutes();
        System.out.println("[DashboardServer] Started at http://localhost:" + port);
        return this;
    }
    
    public void stop() {
        if (app != null) {
            app.stop();
            app = null;
        }
    }

    private void setupRoutes() {
        // API để load lịch sử train (Offline viewing / initial load)
        app.get("/api/metrics", ctx -> {
            // Lấy dữ liệu từ TrainingHistory trả về list các history points
            Map<String, Object> data = new HashMap<>();
            data.put("epochs", history.getEpochs());
            Map<String, List<Float>> series = new HashMap<>();
            // Assuming TrainingHistory has getMetrics() Map
            try {
                // Tạm thời tự extract thông qua list custom nếu có
                var map = history.getMetrics();
                series.putAll(map);
            } catch(Exception e) {
                // ignore
            }
            data.put("series", series);
            ctx.json(data);
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
     * Broadcast 1 log mới nhất từ Training loop lên tát cả Web users.
     */
    public void broadcastMetrics(int epoch, Map<String, Float> metrics) {
        if (wsSessions.isEmpty()) return;

        Map<String, Object> payload = new HashMap<>();
        payload.put("epoch", epoch);
        payload.put("metrics", metrics);
        payload.put("timestamp", System.currentTimeMillis());

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
                payload.put("vram", vram);
            }
        } catch (Throwable t) {
            // ignore
        }

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
            // Read saved state
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
    
    /**
     * CLI entry point cho chế độ Offline.
     * Sử dụng: java com.user.nn.utils.dashboard.DashboardServer <port> <path_to_json>
     */
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
